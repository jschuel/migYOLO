'''Defines a simple vignetting correction over an 11 x 7 grid with
a predefined vignetting correction map. the logical pixel boundaries
assume a 512 x 288 image which are the dimensions of 4x4 binned MIGDAL
images. The calibration files are energy calibration files written
during Fe55 runs'''

import pandas as pd
import numpy as np

def simple_vignetting_correction(df,vignetting_map_file,calibration_file):
    '''Load vignetting map'''
    def load_vignetting_map(vmap_file):
        vmap = pd.read_pickle(vmap_file)
        vignetting_map = {}
        for index, value in zip(vmap.index,vmap.values.flatten()):
            vignetting_map[index] = value
        return vignetting_map

    vignetting_map = load_vignetting_map(vmap_file = vignetting_map_file)
    '''Define logical pixel values'''
    xlogical_pixel_boundaries = np.array([[0,45],[46,91],[92,138],[139,185],[186,232],
[233,279],[280,326],[327,373],[374,419],[420,465],[466,511]])
    ylogical_pixel_boundaries = np.array([[0,40],[41,81],[82,122],[123,164],[165,205],
[206,246],[247,287]])

    '''Loop through DataFrame to obtain logical pixels'''
    logical_pixels = []
    for i in range(0,len(df)):
        tmp = df.iloc[i]
        xLogicalPix = np.abs((tmp['col'].mean()-xlogical_pixel_boundaries.mean(axis = 1))).argmin()
        yLogicalPix = np.abs((tmp['row'].mean()-ylogical_pixel_boundaries.mean(axis = 1))).argmin()
        logical_pixels.append((xLogicalPix,yLogicalPix))
    df['logicalPixel'] = logical_pixels

    '''Perform corrections'''
    arr = np.array([vignetting_map[df['logicalPixel'].iloc[i]] for i in range(0,len(df))])
    df['vignetting_correction_factor'] = arr.max()/arr #increase intensity of events in logical pixels with lower means

    df.drop(columns=['logicalPixel']) #no longer needede

    df['corrected_intensity'] = df['vignetting_correction_factor']*df['gauss_intensity']

    '''Load calibration file and then compute the energy'''
    calibration = np.load(calibration_file)

    df['energy'] = 5.9/calibration[1]*df['corrected_intensity']

