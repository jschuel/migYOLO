from migYOLO.utils.readYAML import read_config_file
from migYOLO.pipeline.pipeline import downsample
import os

'''Load downsampling configuration'''
conf = read_config_file('testConf.yaml')['downsample']

'''Get a test input file'''
indir = conf['data_dir']
test_infile = indir+'/'+os.listdir(indir)[0] #Grab the first file in the test_infile directory

'''Specify output directory and output file'''
outdir = conf['outpath']
outfilename = 'test'

'''Specify Master dark file'''
darkfile = conf['dark_file']

'''Run downsampling script'''
ds = downsample(infile = test_infile,
           outpath = outdir,
           outfile = outfilename,
           masterdarkfile = darkfile)
