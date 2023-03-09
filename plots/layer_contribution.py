import numpy as np
import scipy

layername = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',  'fc6', 'fc7', 'fc8']

def layer_cont(output_path, nv, val_voxel_data, val_pred_all, val_pred_lw):
    # calculate covariances
    partition_r = np.ndarray(shape=(len(layername), nv))
    for v in range(nv):
        full_c = np.cov(val_pred_all[:,v], val_voxel_data[:,v])
        for l in range(len(layername)):
            part_c = np.cov(val_pred_lw[l,v], val_voxel_data[:,v])
            partition_r[l,v] = part_c[0,1]/np.sqrt(full_c[0,0]*full_c[1,1])

    partition_R_avg=np.zeros(len(layername))
    for l in range(len(layername)):
        values = partition_r[l,v]/val_pred_all[:,v]
        partition_R_avg[l] = np.mean(values)

    dict_= {"layer_cont":partition_R_avg}
    scipy.io.savemat(output_path,dict_)
    