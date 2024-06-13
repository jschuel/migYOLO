from migYOLO.pipeline.pipeline import yolo
from migYOLO.utils.readYAML import read_config_file
import os
from ultralytics import YOLO

conf = read_config_file('testConf.yaml')['yoloConf']

'''Grab a sample downsampled image'''
downsample_dir = conf['downsample_path']

'''check if the downsampled image directory is empty, if so throw an error'''
if len(os.listdir(downsample_dir)) == 0:
    raise ValueError("There are no images in %s. To rectify this, please run downsample.py!"%(downsample_dir))

'''Choose an image from the downsample dircetory'''
downsample_filename = os.listdir(downsample_dir)[0]
downsample_file = downsample_dir + '/' + downsample_filename

'''downsample_filename has a '.npy' or '.npz' extension, let's use the name without the extension as the name for our YOLO output file and add '.feather' to it'''
outfilename = os.path.splitext(downsample_filename)[0]+'.feather'

'''The .yaml configuration file controls essentially every aspect of the inputs to YOLO. Please adjust configurations there'''
yolo(infile = downsample_file,
     outpath = conf['outpath'],
     outfilename = outfilename,
     model = YOLO(conf['model']), #Ultralytics' YOLO() wraps around our model filestring
     vignetting_map_file = conf['vignetting_map'],
     calibration_file = conf['calibration_file'],
     png_threshold = conf['png_threshold'],
     png_saturation = conf['png_saturation'],
     remove_downsample = True, #When running a larger script we often set this to True
     save_pixels = False,
     migdal_cut = conf['migdal_cut'])
