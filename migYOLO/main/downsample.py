from migYOLO.utils.readYAML import read_config_file
from migYOLO.pipeline.pipeline import downsample
import os
from tqdm import tqdm

'''Load downsampling configuration'''
conf = read_config_file('globalConf.yaml')['downsample']

'''Get a test input file'''
indir = conf['data_dir']
infiles = [indir+'/'+fi for fi in sorted(os.listdir(indir))]
outdir = conf['outpath']

for fi in infiles:
    outfilename = os.path.splitext(os.path.split(fi)[1])[0] #get name of input image without the file extension
    '''Specify Master dark file'''
    darkfile = conf['dark_file']

    '''Run downsampling script'''
    ds = downsample(infile = fi,
           outpath = outdir,
           outfile = outfilename,
           masterdarkfile = darkfile)
