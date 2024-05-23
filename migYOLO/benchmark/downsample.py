import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import GaussianBlur
import torch.nn as nn
import os
from os import sys
import time
from skimage import io
import matplotlib.image
from migYOLO.utils.readYAML import read_config_file

num_ims = int(sys.argv[1])
num_MTIFFs = num_ims // 200

if num_ims % 200 != 0:
    raise ValueError("Number of images must be a multiple of 200")

conf = read_config_file('benchmarkConf.yaml')['downsample']

batch_size = conf['batch_size']

'''Dataset reads MTIFFs, downsamples and Gaussian smoothes them, and then writes PNGs for each MTIFF frame to be passed into YOLO'''
class CustomDataset(Dataset):
    def __init__(self, file_paths,darkim):
        self.file_paths = file_paths
        self.darkim = darkim
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):

        imname = os.listdir(conf['data_dir'])[0]
        '''Use only one MTIFF to run benchmark but save space. We make this owrk with the following line'''
        os.rename(conf['data_dir']+'/%s'%(imname),self.file_paths[idx])

        
        data = io.imread(self.file_paths[idx],plugin='tifffile')-self.darkim

        if conf['GPU']:
            data = torch.from_numpy(data).cuda()
        else:
            data = torch.from_numpy(data)
        '''4x4 binning'''
        ap = nn.AvgPool2d(kernel_size=(4,4),stride=(4,4),divisor_override=1)
        '''Gaussian smoothing'''
        gb = GaussianBlur(kernel_size=9, sigma=(4/3,4/3))
        blurred_tensor = gb(ap(data))
        blurred_tensor[blurred_tensor<0] = 0
        fname = os.path.splitext(os.path.split(self.file_paths[idx])[1])[0]
        path = conf['png_dir']+'/'+fname+'.npy'
        if not os.path.exists(path):
            os.makedirs(path)
        for i,im in enumerate(blurred_tensor):
            arr = np.copy(im.cpu())
            arr2 = np.log10(arr+1)
            matplotlib.image.imsave('%s/%s.png'%(path,i), arr2, vmin=1.4, vmax= 4,cmap = 'jet')
        
        return blurred_tensor

file_paths = []
for i in range(num_MTIFFs):
    file_paths.append(conf['data_dir']+'/%s.%s.MTIFF'%(os.path.split(conf['data_dir'])[1],i))

dark = np.load(conf['dark_file'])
dataset = CustomDataset(file_paths,dark)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers = conf['num_workers'], shuffle=False)

# Now you can iterate over the dataloader to get batches of data

st = time.time()
for i,batch in enumerate(dataloader):
    b = batch
    for j in range(i*batch_size,(i+1)*batch_size):
        try:
            bt = b[j%batch_size].cpu().numpy()
            np.save(conf['outpath']+'/%s'%(os.path.splitext(os.path.split(dataloader.dataset.file_paths[j])[1])[0]),bt)
        except:
            print("No file")
ed = time.time()
print('Time to downsample %s images = %s seconds'%(num_ims,ed-st))
