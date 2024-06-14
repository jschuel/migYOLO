'''Module downsamples images and then performs YOLO inference. The YOLO outputs are saved
to the 'outpath' specified in globalConf.yaml'''

import migYOLO.pipeline.pipeline as mp
from migYOLO.utils.readYAML import read_config_file
from ultralytics import YOLO
from skimage import io
import os
from tqdm import tqdm

'''Load configurations'''
config = read_config_file('globalConf.yaml')
downsampleConfig = config['downsample']
yoloConfig = config['yoloConf']

'''Read in raw image files and output downsampled and Gaussian Filtered image'''
def downsample_images(raw_image_file):
    ims = io.imread(raw_image_file,plugin='tifffile')
    
    '''The defaults for all other arguments of mp.downsample() are sufficient here'''
    ds = mp.downsample(ims)
    return ds.processedImages

'''Read in raw images, process them with mp.downsample(), perform YOLO inference on them
with mp.yolo'''
def main():
    imdir = downsampleConfig['data_dir'] #directory of raw iamges
    '''Get the list of raw image files which we will loop through'''
    infiles = [imdir + '/' + fi for fi in sorted(os.listdir(imdir)) if '.MTIFF' in fi]
    print("Processing these files: %s\n"%(infiles))
    for infile in infiles:
        '''YOLO output name is <name_of_input_image>.feather'''
        outfilename = os.path.splitext(os.path.split(infile)[1])[0]+'.feather'
        '''Get the name of weights file to make separare YOLO output directories based
        on the weights file we use'''
        modelname = os.path.splitext(os.path.split(yoloConfig['model'])[1])[0]
        outdir = yoloConfig['outpath'] + '/' + modelname #outpath of YOLO output
        print('Processing image %s. YOLO output will be written to %s\n'%(os.path.split(infile)[1],outdir+'/'+outfilename))
        '''Downsample and Gaussian filter'''
        processed_ims = downsample_images(infile)

        '''Pass output into YOLO'''
        mp.yolo(infile = processed_ims,
             outpath = outdir,
             outfilename = outfilename,
             model = YOLO(yoloConfig['model']), #Ultralytics' YOLO() wraps around our model filestring
             vignetting_map_file = yoloConfig['vignetting_map'],
             calibration_file = yoloConfig['calibration_file'],
             png_threshold = yoloConfig['png_threshold'],
             png_saturation = yoloConfig['png_saturation'],
             save_pixels = yoloConfig['save_pixels'],
             migdal_cut = yoloConfig['migdal_cut'])

if __name__ == '__main__':
    main()
