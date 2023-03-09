import numpy as np
import h5py
import argparse
import caffe

def feature_extraction(input_path, output_path, prototxt_path, weight_path, mean_path, Ns, N_images, model):
    '''
    input_path: path to the saved frames
    output_path: where to save output
    prototxt_path: path to .prototxt file
    weight_path: path to .caffemodel
    mean_path: path to the mean ImageNet image 
    Ns: Number of segments
    N_images: Number of frames
    model: UC or DC
    '''
######################################## Load Caffe model ###################################################
    net = caffe.Net(prototxt_path, weight_path, caffe.TEST)

    # load the mean ImageNet image (as distributed with Caffe) for subtraction  
    mu = np.load(mean_path)
    mu = mu[:, 15:242, 15:242]  # the mean (BGR) pixel values

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel


    layer_name_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

    numlist = np.arange(0, N_images, 1) 
    for seg in range(1, Ns + 1):
        foldpath = output_path+'/'+model+"/seg"+{str(seg)}+".h5"
        store = h5py.File(foldpath, 'w')
        act = {}
        for lay_idx in range(0, len(layer_name_list)):
            layer_name = layer_name_list[lay_idx]
            grp1 = store.create_group(layer_name)
            temp = net.blobs[layer_name].data.shape
            temp = list(temp)
            temp[0] = len(numlist)
            temp = tuple(temp)
            act[lay_idx] = grp1.create_dataset('data', temp, dtype='float16')
        k = 0
        for im in numlist:
            image = caffe.io.load_image(input_path + '/frameseg' + str(seg) + '/im-' + str(im + 1) + '.jpg')
            transformed_image = transformer.preprocess('data', image)
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            for lay_idx in range(0, len(layer_name_list)):
                layer_name = layer_name_list[lay_idx]
                act[lay_idx][k, :] = net.blobs[layer_name].data
            k = k + 1
        store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='path to the saved frames')
    parser.add_argument('--output_path', type=str, required=True, help='path to the saved output')
    parser.add_argument('--prototxt_path', type=str, required=True, help='path to .prototxt file')
    parser.add_argument('--weight_path', type=str, required=True, help='path to .prototxt file')
    parser.add_argument('--mean_path', type=str, required=True, help='path to the mean ImageNet image ')
    parser.add_argument('--Ns', type=int, required=True, help='Number of segments')
    parser.add_argument('--N_images', type=int, required=True,help='Number of frames')
    parser.add_argument('--model', type=str, required=True, help='UC or DC')
    args = parser.parse_args()   

    feature_extraction(args.input_path,args.output_path, args.prototxt_path, args.weight_path, args.mean_path, args.Ns, args.N_images, args.model)