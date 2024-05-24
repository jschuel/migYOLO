'''
JTS Updated 05/2024: Modules for real time bounding box identification of OTPC images using a pretrained YOLOv8 algorithm. Downsampled, dark subtracted images are fed into the yolo module. The yolo module reads in a weights file and draws bounding boxes around all objects of interest (tracks or sparks)
in the image. The bounding boxes predict the class of the pixels within the box (ER,NR,NR afterglow, spark, spark afterglow, proton) which can be used for quick analysis.
The yolo module also performs analysis on the pixels within the bounding boxes, including measures of intensity (both individual pixel and entire track), track length (using weighted moments),
angle (after head/tail correction), and head charge fractions. The yolo analysis pipeline can also be fed in a vignetting correction map and will perform a vignetting correction, allowing for
accurate Fe55 energy calibrations.
'''

'''Class assignment keys
{0: 'ER', 1: 'Hot_pixel', 2: 'NR', 3: 'Proton', 4: 'Proton_AG', 5: 'Rolling_shutter', 6: 'Spark', 7: 'Spark_AG', 8: 'Storm']
'''

import numpy as np
import pandas as pd
from skimage import io
from ultralytics import YOLO
from io import BytesIO
import os
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur
from migYOLO.pipeline.migdal_skim import group_and_search
from migYOLO.utils.track_fit import compute_SVD_length_angle_and_HCF
from migYOLO.utils.convert_numpy_to_png import numpy_to_png
from migYOLO.utils.vignetting_correction import simple_vignetting_correction

class downsample:
    '''infile: Path of input image file. By default we read in batches of 200 images
       masterdarkfile: Path of master dark file. Must be in .npy or .npz format. Pretrained YOLO files are trained with dark subtracted images 
       so this should not be None when using pretrained models for MIGDAL
       outfile: Path of output for processed image data. Specify this if you want to save downsampled images
       bin_size: downsample binning size
       filter_kernel_size: Size of Gaussian filter
       filter_sigma: Sigma of Gaussian filter
       
       The default values of the parameters are what we use for the pretrained YOLO models 
    '''
    def __init__(self, infile, outpath = None, outfile = None, masterdarkfile = '../data/dark/sample_master_dark.npy', bin_size = 4, filter_kernel_size = 9, filter_sigma = 4/3):
        
        self.bin_size = bin_size
        self.masterdarkfile = masterdarkfile
        dark = self.load_dark()

        '''Add flexibility to read in numpy arrays or image files'''
        if isinstance(infile, np.ndarray):
            self.images = infile-dark

        elif isinstance(infile, str):
            self.infile = infile
            #load images
            self.images = self.load_data()
            self.images = self.images - dark
            
        #downsample
        self.downSampledImages = self.downsample_images()

        #Gaussian filter
        self.processedImages = self.gaussianFilter(self.downSampledImages)
        self.processedImages[self.processedImages < 0] = 0
        #If an outfile path is specified, then we save the output
        if outpath is not None:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            self.save_output(outpath+'/'+outfile)
            print('SUCCESS: Image has been %s x %s binned and save to %s/%s.npz'%(bin_size,bin_size,outpath,outfile))

    def load_dark(self):
        if self.masterdarkfile is not None:
            if os.path.splitext(self.masterdarkfile)[1] == '.npy':
                dark = np.load(self.masterdarkfile)
            elif os.path.splitext(self.masterdarkfile)[1] == '.npz':
                dark = np.load(self.masterdarkfile)
                dark = dark[dark.files[0]]
            else:
                raise ValueError("Master dark needs to be .npy or .npz. Make sure this is the case or set masterdarkfile to None to not dark subtract")
        return dark
            
    def load_data(self):
        images = io.imread(self.infile,plugin='pil')
        return images

    def downsample_images(self):
        #Performs downsampling by a factor of bin_size
        ap = nn.AvgPool2d(kernel_size = self.bin_size, stride = self.bin_size, divisor_override = 1)
        ds = ap(torch.tensor(self.images).float().unsqueeze(0)).squeeze(0).numpy()
        return ds

    def gaussianFilter(self,image):
        gb = GaussianBlur(kernel_size = 9, sigma = (4/3,4/3))
        filtered = gb(torch.tensor(image).unsqueeze(0)).squeeze(0).numpy()
        return filtered

    def save_output(self,output):
        np.savez_compressed(output,self.processedImages)
        
class yolo:
    
    def __init__(self, infile, outpath, outfilename, model=None, vignetting_map_file=None,calibration_file=None,png_threshold=1.4, png_saturation=4, remove_downsample = True, save_pixels = False, **skim_kwargs):
        '''infile: Can either be a numpy array or a numpy filstring that will be opened. This is checked at the beginning of the script
        outpath: Output file path
        outfilename = Output file name (should have a .feather extension)
        model: YOLO model
        vignetting_map_file: .pkl file with vignetting map obtained from an Fe55 run. If None, vignetting corrections will be skipped
        calibration_file: Intensity (after vignetting corrections) to energy calibration map. Only specify if performing vignetting corrections
        png_threshold: Currently migYOLO.utils.vignetting_correction assumes logarithmic colorscales. For 4x4 binned, Gaussian filtered images, we trained YOLO using a threhsold of 1.4
        png_saturation: Same as png_threhsold but for the maximum allowed intensity. For 4x4 binned, Gaussian filtered images, we trained YOLO using a limit of 4
        remove_downsample: Set to True if we want to remove the downsampled image numpy array after processing with YOLO. It's a good idea to do this to save disk space
        save_pixels: Set to True if you want to save the pixel contents within each bounding box. Recommended to keep this false
        
        **skim_kwargs inputs relevant to migYOLO.pipeline.migdal_skim's group_and_search class that's called in this class
        '''

        '''Check if input is a numpy array or filename. Throw error if not'''
        if isinstance(infile, np.ndarray):
            self.ims = infile
        elif isinstance(infile, str):
            if os.path.splitext(infile)[1] == '.npy':
                self.ims = np.load(infile)
            elif os.path.splitext(infile)[1] == '.npz':
                ims = np.load(infile)
                self.ims = ims[ims.files[0]]
            else:
                raise ValueError("infile must be a numpy array or a .npy / .npz file")
        else:
            raise ValueError("infile must be a numpy array or a .npy / .npz file")
                
        self.model          = model
        self.png_threshold  = png_threshold
        self.png_saturation = png_saturation

        '''Create pngs for YOLO to perform inference on'''
        pngs = numpy_to_png(np.log10(self.ims+1),vmin = png_threshold,vmax = png_saturation)
        self.pngs = [Image.open(BytesIO(pngs[i])) for i in range(0,len(self.ims))]

        '''Perform YOLO inferece'''
        self.data = self.yolo_inference()

        '''Apply vignetting correction if a file is specified''' 
        if vignetting_map_file is not None:
            if calibration_file is None:
                raise ValueError("calibration_file must be specified to apply vignetting corrections")
            simple_vignetting_correction(self.data,vignetting_map_file,calibration_file)

            '''Perform Migdal effect skim only if we did vignetting corrections because we need the energy column'''
            gs = group_and_search(in_data = self.data, **skim_kwargs)
            self.data = gs.data
            self.comb = gs.comb
        else:
            '''If we don't perform a Migdal search, self.comb needs to be None'''
            self.comb = None
            
        if save_pixels:
            self.outpath = outpath+'/with_pixel_hits/'
        else:
            '''Remove pixel-level info if we don't want to save pixels'''
            self.data = self.data[[col for col in self.data if self.data[col].dtype != 'O']]
            self.outpath = outpath+'/without_pixel_hits/'

        if len(self.data) > 0:
            if not os.path.exists(self.outpath):
                os.makedirs(self.outpath)
            self.data.to_feather(self.outpath+outfilename)
            print("SUCCESS: YOLO outputs saved to %s\n"%(self.outpath+outfilename))
            if not os.path.exists(self.outpath+'/migdal_candidates/'):
                os.makedirs(self.outpath+'/migdal_candidates/')
            if self.comb is not None:
                print("Migdal candidate found! Saved to %s"%(self.outpath+'/migdal_candidates/'+outfilename))
                '''Need to reset index if we save as feather'''
                self.comb.index = [i for i in range(0,len(self.comb))]
                self.comb.to_feather(self.outpath+'/migdal_candidates/'+outfilename)
            else:
                print("No Migdal candidates found with the input migdal_cut.")
        else:
            print("File empty, didn't write")

        if remove_downsample and isinstance(infile, str):
            os.remove(infile)

    def yolo_inference(self):
        colsgauss       = []
        rowsgauss       = []
        cgausses        = [] #light in gaussian filtered image
        true_lengths    = [] #lengths rescaled to original coordinates
        probs           = [] #yolo classification probablility
        clfs            = [] #yolo predicted class
        idxs            = [] #original image indices since images are being split up by bounding box
        true_angles     = [] #Agnles w.r.t 2048 x 1152 pixels
        HCFs            = [] #Head charge fraction, algorithm will vary for ERs and NRs
        colmax          = [] #max x bounding box downsample
        rowmax          = [] #max y bounding box downsamples
        colmin          = [] 
        rowmin          = []
        tracks_in_frame = []

        print('Performing YOLO inference\n')

        batch_size = len(self.pngs) #200 for an MTIFF

        '''Perform YOLO inference on all images'''
        results = self.model.predict(self.pngs,batch = batch_size,verbose = False, workers = 1,imgsz=512,rect=True)        

        '''Bounding box analysis'''
        for i,gf in enumerate(self.ims): # Loop through images
            
            boxes = results[i].boxes.data.cpu().numpy() #Array contains bounding box coordinates with prediction of species and associated probabilities. We move it to the CPU for the rest of the analysis                          
            for k, box in enumerate(boxes): # Loops through each object in image[i]
                tracks_in_frame.append(len(boxes))
                ys = (box[1:4:2]).astype('int')
                xs = (box[0:3:2]).astype('int')
                cropgf = torch.tensor(gf[ys[0]:ys[1],xs[0]:xs[1]]).to_sparse()
                
                rowgf,colgf = cropgf.indices()
                colgf = colgf.numpy()+xs[0]
                rowgf = rowgf.numpy()+ys[0]
                cgauss = cropgf.values().numpy()
                
                colsgauss.append(colgf)
                rowsgauss.append(rowgf)
                colmax.append(xs.max())
                colmin.append(xs.min())
                rowmax.append(ys.max())
                rowmin.append(ys.min())
                cgausses.append(cgauss)
                
                length, theta, HCF = compute_SVD_length_angle_and_HCF(colgf,rowgf,cgauss,box[5])

                true_lengths.append(length)
                true_angles.append(theta) #angle rescaled to orignal dimensions after head/tail
                HCFs.append(HCF) #HCF after head/tail correction

                idxs.append(i) #original image index
                clfs.append(box[5]) #bounding box classification species, e.g. ER, NR, alpha, spark
                probs.append(box[4]) #confidence score associated with bounding box classification
                                
        '''Store information in pandas dataframe'''
        
        df = pd.DataFrame()

        df['col']                  = colsgauss
        df['row']                  = rowsgauss
        df['cgauss']               = cgausses
        df['true_length']          = true_lengths
        df['theta']                = true_angles
        df['HCF']                  = HCFs
        df['prediction']           = clfs
        df['prob']                 = probs
        df['gauss_intensity']      = df['cgauss'].apply(lambda x: x.sum()) #intensity of each pixel above sigma masterDark sigmas within the bounding box
        df['qmax']                 = df['cgauss'].apply(lambda x: x.max())
        df['colmax']               = colmax #maximum column in the bounding box for fiducialization (220 is edge when nx = 221 in the downsampled image)
        df['rowmax']               = rowmax #maximum row in the bounding box for fiducialization (220 is edge when ny = 221 in the downsampled image)
        df['colmin']               = colmin #minimum column in the bounding box for fiducialization (0 is edge)
        df['rowmin']               = rowmin #min row in bounding box (0 is edge)
        df['original_index']       = idxs
        df['tracks_in_orig_frame'] = tracks_in_frame
        
        return df

