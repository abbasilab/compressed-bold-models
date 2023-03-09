
import numpy as np
import argparse
import h5py 
import scipy

fpX = np.float32

def svModelSpace(sharedModel_specs):
    vm = np.asarray(sharedModel_specs[0])
    nt = np.prod([sms.length for sms in sharedModel_specs[1]])           
    rx, ry, rs = [sms(vm[i,0], vm[i,1]) for i,sms in enumerate(sharedModel_specs[1])]
    xs, ys, ss = np.meshgrid(rx, ry, rs, indexing='ij')
    return xs.reshape((1,nt)).astype(dtype=fpX), ys.reshape((1,nt)).astype(dtype=fpX), ss.reshape((1,nt)).astype(dtype=fpX) 
class subdivision_1d(object):
    def __init__(self, n_div=1, dtype=np.float32):
        self.length = n_div
        self.dtype = dtype
        
    def __call__(self, center, width):
        '''	returns a list of point positions '''
        return [center] * self.length
class linspace(subdivision_1d):    
    def __init__(self, n_div, right_bound=False, dtype=np.float32, **kwargs):
        super(linspace, self).__init__(n_div, dtype=np.float32, **kwargs)
        self.__rb = right_bound
        
    def __call__(self, center, width):
        if self.length<=1:
            return [center]     
        if self.__rb:
            d = width/(self.length-1)
            vmin, vmax = center, center+width  
        else:
            d = width/self.length
            vmin, vmax = center+(d-width)/2, center+width/2 
        return np.arange(vmin, vmax+1e-12, d).astype(dtype=self.dtype)
    
class logspace(subdivision_1d):    
    def __init__(self, n_div, dtype=np.float32, **kwargs):
        super(logspace, self).__init__(n_div, dtype=np.float32, **kwargs)
               
    def __call__(self, start, stop):    
        if self.length <= 1:
            return [start]
        lstart = np.log(start+1e-12)
        lstop = np.log(stop+1e-12)
        dlog = (lstop-lstart)/(self.length-1)
        return np.exp(np.arange(lstart, lstop+1e-12, dlog)).astype(self.dtype)


def svModelSpace(sharedModel_specs):
    fpX = np.float32
    vm = np.asarray(sharedModel_specs[0])
    nt = np.prod([sms.length for sms in sharedModel_specs[1]])           
    rx, ry, rs = [sms(vm[i,0], vm[i,1]) for i,sms in enumerate(sharedModel_specs[1])]
    xs, ys, ss = np.meshgrid(rx, ry, rs, indexing='ij')
    return xs.reshape((1,nt)).astype(dtype=fpX), ys.reshape((1,nt)).astype(dtype=fpX), ss.reshape((1,nt)).astype(dtype=fpX)

def receptive_field_plot(input_path,lx,ly,smin,smax,nx,ny,ns, output_path):
    input_ = h5py.File(input_path) #column 0 top voxel index, column 1 index of top model
    n_voxels = input_["model"].shape[1]
    sharedModel_specs = [[(0., lx), (0., ly), (smin, smax)], [linspace(nx), linspace(ny), logspace(ns)]]
    mx, my, ms=svModelSpace(sharedModel_specs)

    rf_=np.zeros((n_voxels,3))
    for i in range (n_voxels):
        rf_[i,0]= mx[input_[0][i]]
        rf_[i,1] = my[input_[0][i]]
        rf_[i,2] = ms[input_[0][i]]
 
    mdic = {'rf_':rf_}
    scipy.io.savemat("C:/Users/F/Desktop/encoding_git/encoding_model/b.mat",mdic)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='where top voxels and best corresponding model is saved')
    parser.add_argument('--lx', type=float, required=True, help='view angle')
    parser.add_argument('--ly', type=float, required=True, help='view angle')
    parser.add_argument('--nx', type=int, required=True, help='number of centers along x axis')
    parser.add_argument('--ny', type=int, required=True, help='number of centers along y axis')
    parser.add_argument('--smin', type=float, required=True, help='min of radius')
    parser.add_argument('--smax', type=float, required=True, help='max of radius')
    parser.add_argument('--ns', type=float, required=True, help='num of radius')
    parser.add_argument('--output_path', type=str, required=True, help='where to save the output')
    args = parser.parse_args()   

    receptive_field_plot(args.input_path, args.lx, args.ly, args.nx, args.ny, args.smin, args.smax, args.ns)