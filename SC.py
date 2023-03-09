import argparse
import os
import h5py
import numpy as np
from scrsc.sorted_index import sorted_index

layerlist = [u'conv1', u'conv2', u'conv3', u'conv4', u'conv5']

def compressed_map(lay, N_removed, input_path,  output_path, N_seg):
 
    N_remained = len(sorted_index[lay-1]) - N_removed
        
    for seg in range (1,N_seg+1):
    
        f1 = h5py.File(input_path+'/CaffeNet_feature_maps_seg' + str(seg) + '.h5', 'r')

        if not os.path.exists(output_path +'/'+ str(N_remained)):
            os.mkdir(output_path +'/'+ str(N_remained))      
        store = h5py.File(output_path +'/'+ str(N_remained) + '/CaffeNet_feature_maps_seg' + str(seg) + '.h5', "a")
 
        fmap = f1.get(layerlist[lay-1] + '/data')

        pruned_map=np.zeros([fmap.shape[0], N_remained, fmap.shape[2], fmap.shape[3]])
        for i in sorted_index[lay-1][N_removed:]:
            pruned_map[:, j, :, :] = fmap[:, i, :, :]
            j = j + 1
            
        grp1 = store.create_group(layerlist[lay])
        grp1.create_dataset('data', data=pruned_map, dtype='float16')

    store.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lay', type=int, required=True, help='layer index')
    parser.add_argument('--N_removed', type=int, required=True, help='how many filters to remove')
    parser.add_argument('--input_path', type=str, required=True, help='where UC maps are saved')
    parser.add_argument('--output_path', type=str, required=True, help='where to save the output')
    parser.add_argument('--N_seg', type=int, required=True, help='number of segments')
    args = parser.parse_args()   

    compressed_map(args.lay, args.N_removed, args.input_path,  args.output_path, args.N_seg)