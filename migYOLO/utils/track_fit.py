'''Track fitting function inputs sparse column, row, and intensity coordinates
as well as YOLO's track prediction. If the track is an ER, NR, proton, or proton afterglow, the algorithm will fit a principal axis to the track and then compute the track length using a singular value decomposition. If the track is an ER or NR, the algorithm will further compute the head charge fraction to determine the tracks principal vector and will then compute the angle phi with respect to the positive x axis of the ca,era readout. If the track is a proton, the algorithm will return -1 for the angle and head charge fraction. If the event is none of these four classes of tracks, -1 will be returned for the length, angle, and head charge fraction''' 

import numpy as np
import scipy.sparse.linalg

def compute_SVD_length_angle_and_HCF(col,row,intensity,pred):
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
