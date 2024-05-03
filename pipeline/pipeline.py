'''
JTS 01/2024 (Updated for 4x4 Gaussian filtered YOLO inference): Modules for real time bounding box identification of OTPC images using a pretrained YOLOv8 algorithm. Downsampled, dark subtracted images are fed into the yolo module. The yolo module reads in a weights file and draws bounding boxes around all objects of interest (tracks or sparks)
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
import matplotlib.image
from ultralytics import YOLO
import os
import shutil
from tqdm import tqdm
from PIL import Image
import scipy.sparse.linalg #For SVD length fits
import torch
import time
from migdal_skim import group_and_search
import subprocess

class yolo():
    '''
    Does the following
    1. Reads in dark subtracted, downsampled image array
    2. Saves png files in a created "png_dump" directory
    3. Performs YOLO inference on pngs
    4. Writes a pandas dataframe with sparse coordinates of light within YOLO bounding box, as well as track level quantities
    5. Deletes pngs
    '''
    
    def __init__(self,infile,outpathbase = 'yolo_outfiles',create_pngs = False, run_yolo = True, model=None,vignetting_correction_file=None,calibration_file=None,png_threshold=1.4, png_saturation=4, delete_output = False,**kwargs):
        self.infile  = infile
        self.gb = np.load(self.infile)
        self.fname = os.path.split(self.infile)[1]
        self.fnum = os.path.splitext(self.fname)[0][-4:]
        self.run = os.path.splitext(os.path.splitext(self.fname)[0])[0]
        self.date = self.run[self.run.find('T')-6:self.run.find('T')]
        self.subdate = self.date[:-2]
        self.outpath = outpathbase+'/'+self.run
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
        
        self.vignetting_correction_file = vignetting_correction_file
        
        self.xlogical_pixel_boundaries = np.array([[0,45],[46,91],[92,138],[139,185],[186,232],
[233,279],[280,326],[327,373],[374,419],[420,465],[466,511]])
        self.ylogical_pixel_boundaries = np.array([[0,40],[41,81],[82,122],[123,164],[165,205],
[206,246],[247,287]])
        self.calibration_file = calibration_file

        self.model          = model #Yolov8 nano for now

        self.tmp_png_dir    = '../data/pngs/'+self.fname
        self.downsample_file = '../data/downsampled_images/'+self.fname
        print(self.tmp_png_dir)
        self.png_threshold  = png_threshold #threshold for fixed dynamic range pngs
        self.png_saturation = png_saturation #saturation limit for fixed dynamic range pngs

        '''Create pngs for YOLO to perform inference on'''
        if create_pngs:
            self.create_pngs()
        if run_yolo:
            self.data           = self.yolo_inference() #summary data after yolo_inference

            '''Apply vignetting correction if a file is specified''' 
            try:
                self.vignetting_map = self.load_vignetting_map()
                self.apply_vignetting_correction(self.data)
                self.data.drop(columns=['logicalPixel']) #remove logical pixel column so we can save as feather
                self.data['corrected_intensity'] = self.data['vignetting_correction_factor']*self.data['gauss_intensity']
                calibration = np.load(self.calibration_file) #1x3 array of [amp of spectrum, mean, sigma] 
                self.data['energy'] = 5.9/calibration[1]*self.data['corrected_intensity']
            except:
                print("Vignetting correction not applied")
            self.data['qmax'] = self.data['cgauss'].apply(lambda x: x.max())
            self.data['fnum'] = self.fnum
            self.data['tracks_in_orig_frame'] = self.data.sort_values(by=['fnum','original_index']).groupby(['fnum','original_index'])[['fnum','original_index']].transform('count')['original_index']

            self.data.index = [i for i in range(0,len(self.data))]
            
            self.data = self.data[[col for col in self.data if self.data[col].dtype != 'O']]
            
            gs = group_and_search(in_data = self.data, **kwargs)
            self.data = gs.data
            self.comb = gs.comb
            
            try:
                if not os.path.exists(self.outpath+'/without_pixel_hits/'):
                    os.makedirs(self.outpath+'/without_pixel_hits/')
                self.data.to_feather(self.outpath+'/without_pixel_hits/%s.%s.feather'%(self.run,self.fnum))
                if delete_output:
                    os.remove(self.outpath+'/without_pixel_hits/%s.%s.feather'%(self.run,self.fnum))
                if not os.path.exists(self.outpath+'/migdal_candidates/'):
                    os.makedirs(self.outpath+'/migdal_candidates/')
                if self.comb is not None:
                    self.comb.to_feather(self.outpath+'/migdal_candidates/%s.%s.feather'%(self.run,self.fnum))
                    if delete_output:
                        os.remove(self.outpath+'/migdal_candidates/%s.%s.feather'%(self.run,self.fnum))
                    
            except: 
                print("File empty, didn't write")

            shutil.rmtree(self.tmp_png_dir)
            os.remove(self.downsample_file)
        
    '''Pngs that YOLO is evaluated on. Bounding boxes are mapped back to Gaussian blurred tensors'''
    def create_pngs(self):
        path = self.tmp_png_dir
        if not os.path.exists(path):
            print('Making directory %s to store temporary png files for YOLO.\n'%(path))
            os.makedirs(path)
        else:
            raise OSError("Directory %s already exists, please set temporary_png_dir to a directory name that doesn't currently exist"%(path))

        print('Writing %s temporary pngs to the %s directory\n'%(len(self.gb),path))
        for i,im in enumerate(self.gb):
            arr = np.copy(im)
            arr2 = np.log10(arr+1)
            matplotlib.image.imsave('%s/%s.png'%(path,i), arr2, vmin=self.png_threshold, vmax=self.png_saturation,cmap = 'jet')

    def compute_SVD_length_angle_and_HCF(self,col,row,intensity,pred):
        if pred == 0 or pred == 2 or pred == 3 or pred == 4:
            idx = np.where(intensity>10)[0]
        else:
            return -1, -1, -1
        data = np.concatenate([[col[idx].T,row[idx].T]]).T
        A = data - data.mean(axis=0)
        vv = scipy.sparse.linalg.svds(A,k=1)[2]
        proj = (data @ vv.T).T[0]
        length = (proj.max()-proj.min())*80/512
        PA = vv[0]
        theta = np.arctan2(PA[0],PA[1])*180/np.pi
        if pred == 3 or pred == 4:
            return length, -1, -1
        '''Make all events point upward'''
        if theta < 0:
            theta += 180
            PA = -1 * PA
            proj = -1*proj
        midp = 0.5*(proj.max()+proj.min())
        uc = 0
        lc = 0
        inten = intensity[idx]
        for i,val in enumerate(proj):
            if val > midp:
                uc += inten[i]
            elif val < midp:
                lc += inten[i]
            elif val == midp and i%2 == 0:
                uc += inten[i]
            elif val == midp and i%2 != 0:
                lc += inten[i]
        HCF = uc/(uc+lc)
        if pred == 0 and HCF < 0.5:
            theta -= 180
            HCF = 1-HCF
        elif pred == 2 and HCF > 0.5:
            theta -= 180
            HCF = 1-HCF
        return length, theta, HCF

    def yolo_inference(self):
        colsgauss       = []
        rowsgauss       = []
        cgausses        = [] #light in gaussian filtered image
        true_lengths  = [] #lengths rescaled to original coordinates
        probs         = [] #yolo classification probablility
        clfs          = [] #yolo predicted class
        idxs          = [] #original image indices since images are being split up by bounding box
        true_angles   = [] #Agnles w.r.t 2048 x 1152 pixels
        HCFs          = [] #Head charge fraction, algorithm will vary for ERs and NRs
        logicalPixels = [] #11 x 11 logical pixel grid for vignetting
        colmax        = [] #max x bounding box downsample
        rowmax        = [] #max y bounding box downsamples
        colmin        = [] 
        rowmin        = []
        '''Make array of all images in batch'''
        images = [Image.open("%s/%s.png"%(self.tmp_png_dir,i)) for i in range(0,len(self.gb))]
        print('Performing YOLO inference\n')
        batch_size = len(images) #200 for an MTIFF
        '''Perform YOLO inference on all images'''
        
        results = self.model.predict(images,batch = batch_size,verbose = False, workers = 1,imgsz=512,rect=True)
        

        '''Bounding box analysis'''
        for i,gf in enumerate(self.gb):
            boxes = results[i].boxes.data.cpu().numpy() #Array contains bounding box coordinates with prediction of species and associated probabilities. We move it to the CPU for the rest of the analysis
            num_boxes = len(boxes)
                            
            for k, box in enumerate(boxes):
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
                
                length, theta, HCF = self.compute_SVD_length_angle_and_HCF(colgf,rowgf,cgauss,box[5])

                true_lengths.append(length)
                true_angles.append(theta) #angle rescaled to orignal dimensions after head/tail
                HCFs.append(HCF) #HCF after head/tail correction

                idxs.append(i) #original image index
                clfs.append(box[5]) #bounding box classification species, e.g. ER, NR, alpha, spark
                probs.append(box[4]) #confidence score associated with bounding box classification
                
                '''Assign logical pixels for vignetting correction if a correction file is specified'''
                if self.vignetting_correction_file is not None:
                    xLogicalPix = np.abs((colgf.mean()-self.xlogical_pixel_boundaries.mean(axis = 1))).argmin()
                    yLogicalPix = np.abs((rowgf.mean()-self.ylogical_pixel_boundaries.mean(axis = 1))).argmin()
                    logicalPixels.append((xLogicalPix,yLogicalPix))
                    
                '''TODO: Add in IoU computation and create a counter for ERs and NRs that overlap'''
                
        '''Store information in pandas dataframe'''
        
        df = pd.DataFrame()

        df['col']       = colsgauss
        df['row']       = rowsgauss
        df['cgauss']         = cgausses
        df['true_length']    = true_lengths
        df['theta']          = true_angles
        df['HCF']            = HCFs
        df['prediction']     = clfs
        df['prob']           = probs
        df['gauss_intensity']  = df['cgauss'].apply(lambda x: x.sum()) #intensity of each pixel above sigma masterDark sigmas within the bounding box
        df['colmax']         = colmax #maximum column in the bounding box for fiducialization (220 is edge when nx = 221 in the downsampled image)
        df['rowmax']         = rowmax #maximum row in the bounding box for fiducialization (220 is edge when ny = 221 in the downsampled image)
        df['colmin']         = colmin #minimum column in the bounding box for fiducialization (0 is edge)
        df['rowmin']         = rowmin #min row in bounding box (0 is edge)
        df['original_index'] = idxs
        
        if self.vignetting_correction_file is not None:
            df['logicalPixel']   = logicalPixels #temporary tuple corresponding to an 11 x 11 logical pixel used for vignetting correction scale factors. This dataframe column is deleted before the file is saved

        return df

    def load_vignetting_map(self):
        vmap = pd.read_pickle(self.vignetting_correction_file)
        vignetting_map = {}
        for index, value in zip(vmap.index,vmap.values.flatten()):
            vignetting_map[index] = value
        return vignetting_map
    
    def apply_vignetting_correction(self,df):
        arr = np.array([self.vignetting_map[df['logicalPixel'].iloc[i]] for i in range(0,len(df))])
        df['vignetting_correction_factor'] = arr.max()/arr #increase intensity of events in logical pixels with lower means
        


if __name__ == '__main__':

    model = YOLO('models/yolo4by4.pt')
    calibration = 'calibration_files/MIG_Fe55_568V_UQ_240201T105931.CAL.npy'
    vignetting_map = 'calibration_files/MIG_Fe55_568V_UQ_240201T105931.CAL.pkl'
    migdal_cut = 'NR_energy > 60'# & centroid_dist < 6'
    
    def is_screen_running(screen_name):
        try:
            # Run the 'screen -ls' command and capture the output
            output = subprocess.check_output(["screen", "-ls"]).decode("utf-8")
            # Check if the screen session is in the output
            return screen_name in output
        except subprocess.CalledProcessError:
            return False
        
    def save_timestamp():
        fname = 'timestamps/%s.txt'%(time.time())
        with open(fname, "w") as f:
            f.write(str(time.time()))

    #save initial timestamp
    save_timestamp()
    time.sleep(3)
    fis = ['downsampled_images/'+fi for fi in sorted(os.listdir('downsampled_images')) if '.npy' in fi]
    while is_screen_running('Downsample') or len(fis) > 0:
        fis = ['downsampled_images/'+fi for fi in sorted(os.listdir('downsampled_images')) if '.npy' in fi]    
        for fi in fis:
            t0 = time.time()
            try:
                yolo(infile=fi,outpathbase = 'yolo_outfiles',model=model,create_pngs = False,vignetting_correction_file=vignetting_map,calibration_file=calibration,png_threshold=1.4, png_saturation=4,migdal_cut = migdal_cut)
            except:
                print("Wait 1.5 second")
                time.sleep(1.5)
                #try:
                yolo(infile=fi,outpathbase = 'yolo_outfiles',model=model,create_pngs = False,vignetting_correction_file=vignetting_map,calibration_file=calibration,png_threshold=1.4, png_saturation=4,migdal_cut = migdal_cut)
                #except:
                #    print("Wait another 0.5 second")
                #    time.sleep(0.5)
                #    yolo(infile=fi,outpathbase = 'yolo_outfiles',model=model,create_pngs = False,vignetting_correction_file=vignetting_map,calibration_file=calibration,png_threshold=1.4, png_saturation=4,migdal_cut = 'NR_energy > 60 & centroid_dist < 6')
            tf = time.time()
            print(tf-t0)
            save_timestamp()
    #yolo(infile=fi,outpathbase = 'yolo_outfiles',model=model,vignetting_correction_file=None,calibration_file=None,png_threshold=1.4, png_saturation=4)
