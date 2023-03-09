import numpy as np
import h5py
import rcsrc.fwrf as fwrf
import argparse

def RC_features(train_path, test_path, output_path, lx, ly, nx, ny, smin, smax, ns, trn_size):

    hf1 = h5py.File(output_path, 'w')

    sharedModel_specs = [[(0., lx), (0., ly), (smin, smax)], [fwrf.linspace(nx), fwrf.linspace(ny), fwrf.logspace(ns)]]
    log_act_func = lambda x: np.log(1+np.sqrt(np.abs(x)))


    for lay in range (0,5):
        fmaps = []
        fmaps_sizes = []
        fmaps_count = 0

        layerlist = [u'conv1', u'conv2', u'conv3', u'conv4', u'conv5', u'fc6', u'fc7', u'fc8']

        trn_feature_dict = h5py.File(train_path, 'r').get(layerlist[lay] + '/data')[:]
        val_feature_dict = h5py.File(test_path, 'r').get(layerlist[lay] + '/data')[:]

        fmap = np.concatenate((np.array(trn_feature_dict, dtype=np.float32), np.array(val_feature_dict, dtype=np.float32)), axis=0)
        fmaps += [fmap,]
        fmaps_sizes  += [fmap.shape,]
        fmaps_count += fmap.shape[1]
        mst_data, _, _ = fwrf.model_space_tensor(fmaps, sharedModel_specs, nonlinearity=log_act_func, \
                    zscore=True, trn_size=trn_size, batches=(200, nx*ny), view_angle=lx, verbose=True, dry_run=False)
        trn_mst_data = mst_data[:trn_size]
        val_mst_data = mst_data[trn_size:]

        gp1=hf1.create_group(layerlist[lay])
        gp1.create_dataset('train', data=trn_mst_data , dtype=np.float32)
        gp1.create_dataset('test', data=val_mst_data, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True, help='path to the features')
    parser.add_argument('--test_path', type=str, required=True, help='path to the features')
    parser.add_argument('--output_path', type=str, required=True, help='where to save the output')
    # parser.add_argument('--model', type=str, required=True, help='RC or DRC or SRC')
    parser.add_argument('--lx', type=float, required=True, help='view angle')
    parser.add_argument('--ly', type=float, required=True, help='view angle')
    parser.add_argument('--nx', type=int, required=True, help='number of centers along x axis')
    parser.add_argument('--ny', type=int, required=True, help='number of centers along y axis')
    parser.add_argument('--smin', type=float, required=True, help='min of radius')
    parser.add_argument('--smax', type=float, required=True, help='max of radius')
    parser.add_argument('--ns', type=float, required=True, help='num of radius')
    parser.add_argument('--trn_size', type=int, required=True, help='train size')
    args = parser.parse_args()   

    RC_features(args.train_path, args.test_path, args.output_path, args.model, args.lx, args.ly, args.nx, args.ny, args.smin, args.smax, args.ns, args.trn_size)