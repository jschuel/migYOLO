from migYOLO.pipeline.pipeline import yolo
from migYOLO.utils.readYAML import read_config_file
import os
from ultralytics import YOLO

conf = read_config_file('globalConf.yaml')['yoloConf']

'''Grab a sample downsampled image'''
downsample_dir = conf['downsample_path']

'''check if the downsampled image directory is empty, if so throw an error'''
if len(os.listdir(downsample_dir)) == 0:
    raise ValueError("There are no images in %s. To rectify this, please run downsample.py!"%(downsample_dir))

'''Get downsampled image files'''
infiles = [downsample_dir + '/' + fi for fi in sorted(os.listdir(downsample_dir))]

'''Loop through all input files and run YOLO on them'''
for infile in infiles:
    '''Get input image file name without extension and append .feather to save YOLO output'''
    outfilename = os.path.splitext(os.path.split(infile)[1])[0]+'.feather'
    modelname = os.path.splitext(os.path.split(conf['model'])[1])[0] #name of mode without .pt we're going to make a separate output directory for the model name we use
    outdir = conf['outpath'] + '/' + modelname

    '''The .yaml configuration file controls essentially every aspect of the inputs to YOLO. Please adjust configurations there'''
    yolo(infile = infile,
         outpath = outdir,
         outfilename = outfilename,
         model = YOLO(conf['model']), #Ultralytics' YOLO() wraps around our model filestring
         vignetting_map_file = conf['vignetting_map'],
         calibration_file = conf['calibration_file'],
         png_threshold = conf['png_threshold'],
         png_saturation = conf['png_saturation'],
         remove_downsample = False, #When running a larger script we often set this to True
         save_pixels = conf['save_pixels'],
         migdal_cut = conf['migdal_cut'])
