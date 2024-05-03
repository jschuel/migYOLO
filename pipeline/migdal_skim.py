import pandas as pd
import numpy as np
#import time
import warnings
warnings.simplefilter('ignore')

class group_and_search:
    def __init__(self,in_data=None,in_file=None,migdal_cut=None):
        #st = time.time()
        if in_data is not None:
            self.data = in_data
        elif in_file is not None:
            self.data = pd.read_feather(in_file)
        else:
            self.data = pd.read_feather('test.feather')
        self.initial_grp = self.group_data()

        #Make dataframes for current and previous events to put them in the same row
        curr = self.initial_grp[1:].reset_index()
        prev = self.initial_grp[:-1].reset_index()

        #put previous BB info into curr for flagging afterglow in O(n) instead of O(n^2)
        curr['prevcmin'] = prev['colmin']
        curr['prevcmax'] = prev['colmax']
        curr['prevrmin'] = prev['rowmin']
        curr['prevrmax'] = prev['rowmax']
        curr['prevenergy'] = prev['energy']
        curr['prevqmax'] = prev['qmax']

        self.ags = self.determine_afterglow(curr)
        self.rs = self.flag_rolling_shutter(curr)

        self.data = self.add_afterglow_to_data()

        '''Process data before grouping again to analyze'''
        self.fid = self.data.query('RS_flag == 0 & AG_flag == 0 & isFiducial == 1 & tracks_in_orig_frame >= 1')
        self.fid['centroidx'] = (self.fid['colmin']+self.fid['colmax'])/2
        self.fid['centroidy'] = (self.fid['rowmin']+self.fid['rowmax'])/2
        self.grp = self.fid[['original_index','prediction','colmin','colmax','rowmin','rowmax','energy','true_length','centroidx','centroidy']].groupby(['original_index'])[['prediction','colmin','colmax','rowmin','rowmax','energy','true_length','centroidx','centroidy']].agg(list).reset_index()

        self.grp['IoU'], self.grp['centroid_dist'] = self.compute_overlapping_tracks_and_centroid_distances()
        self.grp['coinc_flag'] = self.grp['prediction'].apply(lambda x: 0 in x and 2 in x)

        '''Flag events that contain a single ER and NR without afterglow'''
        self.grp = self.grp.query('coinc_flag == 1')
        self.grp.index = [i for i in range(0,len(self.grp))]

        twotrack = self.grp[self.grp['prediction'].apply(lambda x: len(x)==2)]
        moretrack = self.grp[self.grp['prediction'].apply(lambda x: len(x)>2)]

        if len(twotrack) == 0 and len(moretrack) == 0:
            #print("No Migdal candidates")
            #ed = time.time()
            #print('Total time: %s seconds'%(ed-st))
            self.comb = None
            return None
            
        elif len(twotrack) > 0 and len(moretrack) == 0:
            for col in ['IoU','centroid_dist']:
                twotrack[col] = twotrack[col].apply(lambda x: x[0])
            twotrack = twotrack[twotrack['IoU']>=0]
            if len(twotrack) > 0:
                comb = twotrack
            else:
                #print("No Migdal candidates")
                #ed = time.time()
                #print('Total time: %s seconds'%(ed-st))
                #return None
                self.comb = None
                return None
            
        elif len(twotrack) == 0 and len(moretrack) > 0:
            moretrack = pd.concat([self.expand_entry(moretrack,i) for i in range(0,len(moretrack))])
            moretrack.index = [i for i in range(0,len(moretrack))]
            moretrack = moretrack[moretrack['IoU']>=0]
            if len(moretrack) > 0:
                comb = moretrack
            else:
                #print("No Migdal candidates")
                #ed = time.time()
                #print('Total time: %s seconds'%(ed-st))
                self.comb = None
                return None
        else:
            '''Only pick tracks with IoU < 0.95 (see why IoU >= 0 means IoU < 0.95 in compute_overlapping_tracks_and_centroid_distance()'''
            for col in ['IoU','centroid_dist']:
                twotrack[col] = twotrack[col].apply(lambda x: x[0])
            twotrack = twotrack[twotrack['IoU']>=0]
            moretrack = pd.concat([self.expand_entry(moretrack,i) for i in range(0,len(moretrack))])
            moretrack.index = [i for i in range(0,len(moretrack))]
            moretrack = moretrack[moretrack['IoU']>=0]
            comb = pd.concat([twotrack,moretrack])
            comb.index = [i for i in range(0,len(comb))]

        if len(comb) > 0:
            '''Add columns to make relevant selections on two-track pairs'''
            comb['NR_idx']=comb['prediction'].apply(lambda x: np.where(np.array(x) == 2)[0][0])
            comb['ER_idx'] = np.abs(comb['NR_idx']-1)
            comb['ER_length'] = [comb['true_length'].iloc[i][comb['ER_idx'].iloc[i]] for i in range(0,len(comb))]
            comb['NR_length'] = [comb['true_length'].iloc[i][comb['NR_idx'].iloc[i]] for i in range(0,len(comb))]
            comb['ER_energy'] = [comb['energy'].iloc[i][comb['ER_idx'].iloc[i]] for i in range(0,len(comb))]
            comb['NR_energy'] = [comb['energy'].iloc[i][comb['NR_idx'].iloc[i]] for i in range(0,len(comb))]
            comb['Ediff'] = np.abs(comb['NR_energy']-comb['ER_energy'])

            '''Print number of Migdal candidates satisfying the criteria of interest'''
            #print(len(comb.query('NR_energy >= 60 & centroid_dist <= 6')))
            if migdal_cut is not None:
                try:
                    self.comb = comb.query(migdal_cut)
                except:
                    self.comb = None
            else:
                try:
                    self.comb = comb.query('NR_energy >= 60 & centroid_dist <= 6')
                except:
                    self.comb = None
            if len(self.comb) == 0:
                self.comb = None
        else:
            self.comb = None
            return None
            #print("No Migdal candidates")
            
        #ed = time.time()
        #print('Total time: %s seconds'%(ed-st))
        #self.comb = comb
        #print(in_file,self.num_cands)
        
    '''Initial grouping for afterglow and rolling shutter ID'''    
    def group_data(self):
        #self.data = self.data.query('tracks_in_orig_frame > 1')
        grp = self.data[['original_index','prediction','colmin','colmax','rowmin','rowmax','energy','qmax']].groupby(['original_index'])[['prediction','colmin','colmax','rowmin','rowmax','energy','qmax']].agg(list).reset_index()
        grp['diff_index'] = grp['original_index'].diff()
        return grp

    def determine_afterglow(self,df):
        ags_all = []
        for row in df.iterrows():
            diff = row[1]['diff_index']
            preds = row[1]['prediction']
            if diff == 1 or diff == -199:
                ags = []
                for cmin1,cmax1,rmin1,rmax1,pred in zip(row[1]['colmin'],row[1]['colmax'],row[1]['rowmin'],row[1]['rowmax'],preds):
                    if pred == 0:
                        ag_flag = 0
                        for j in range(0,len(row[1]['prevcmin'])):
                            cmin2 = row[1]['prevcmin'][j]
                            cmax2 = row[1]['prevcmax'][j]
                            rmin2 = row[1]['prevrmin'][j]
                            rmax2 = row[1]['prevrmax'][j]
                            eprev = row[1]['prevenergy'][j]
                            cmaxprev = row[1]['prevqmax'][j]
                            IoU = self.bb_intersection_over_union(cmin1, cmax1, rmin1, rmax1, cmin2, cmax2, rmin2, rmax2)
                            if IoU > 0 and cmaxprev > 150:
                                ag_flag = 1
                                break
                        ags.append(ag_flag)
                    else:
                        ags.append(0)
                ags_all.append(ags)
            else:
                ags_all.append([0 for pred in preds])
        return np.concatenate(ags_all)

    def flag_rolling_shutter(self,df):
        rs_all = []
        for row in df.iterrows():
            preds = row[1]['prediction']
            rs = []
            rs_flag = 0
            for pred in preds:
                if pred == 5 or pred == 8:
                    rs_flag = 1
            for pred in preds:
                rs.append(rs_flag)
            rs_all.append(rs)
        return np.concatenate(rs_all)

    def add_afterglow_to_data(self):
        diff = len(self.data)-len(self.ags)
        self.data = self.data[diff:]
        self.data.index = [i for i in range(0,len(self.data))]
        self.data['AG_flag'] = self.ags
        self.data['RS_flag'] = self.rs
        self.data['isFiducial'] = 0
        index = self.data.query('colmin > 1 & rowmin > 1 & colmax < 510 & rowmax < 286').index.to_numpy()
        self.data['isFiducial'][index] = 1
        return self.data

    '''Overlapping tracks and centroid distances for grouped frames'''
    def compute_overlapping_tracks_and_centroid_distances(self):
        IoUs_pass = []
        centroid_dist = []
        for row in self.grp.iterrows():
            preds = row[1]['prediction']
            IoUs = []
            cdist = []
            if len(preds) == 1 or 0 not in preds:
                IoUs_pass.append([0])
                centroid_dist.append([0])
            else:
                for i,pred in enumerate(preds):
                    if pred == 0:
                        for j in range(0,len(preds)):
                            if i == j:
                                continue
                            else:
                                if row[1]['prediction'][j] == 2:
                                    cmin1 = row[1]['colmin'][i]
                                    cmax1 = row[1]['colmax'][i]
                                    rmin1 = row[1]['rowmin'][i]
                                    rmax1 = row[1]['rowmax'][i]
                                    cmin2 = row[1]['colmin'][j]
                                    cmax2 = row[1]['colmax'][j]
                                    rmin2 = row[1]['rowmin'][j]
                                    rmax2 = row[1]['rowmax'][j]
                                    IoU = self.bb_intersection_over_union(cmin1, cmax1, rmin1, rmax1, cmin2, cmax2, rmin2, rmax2)
                                    if IoU > 0 and IoU <= 0.95:
                                        IoUs.append(IoU)
                                    elif IoU > 0.95:
                                        IoUs.append(-1)
                                    else:
                                        IoUs.append(0)
                                    centx1 = row[1]['centroidx'][i]
                                    centy1 = row[1]['centroidy'][i]
                                    centx2 = row[1]['centroidx'][j]
                                    centy2 = row[1]['centroidy'][j]
                                    cdist.append(np.sqrt((centx2-centx1)**2+(centy2-centy1)**2)*80/512)
                    else:
                        continue
                IoUs_pass.append(IoUs)
                centroid_dist.append(cdist)
        return IoUs_pass, centroid_dist

    '''Utility method to expand frames with >2 tracks into all combinations of two-track pairs'''
    def expand_entry(self,df,i):
        def get_indices(arr):
            pairs = []
            for i in range(len(arr)):
                if arr[i] == 2:
                    for j in range(i+1, len(arr)):
                        if arr[j] == 0:
                            pairs.append([i, j])
                elif arr[i] == 0:
                    for j in range(i+1, len(arr)):
                        if arr[j] == 2:
                            pairs.append([i, j])
            return pairs
        tmp = df.iloc[i]
        indices = get_indices(tmp['prediction'])
        result = {}
        for col in tmp.index.to_numpy():
            if col == 'original_index' or col == 'coinc_flag' or col == 'ntracks':
                result[col] = []
                for index in indices:
                    result[col].append(tmp[col])
            elif len(tmp[col]) > 1 and col != 'IoU' and col != 'centroid_dist':
                result[col] = []
                for index in indices:
                    result[col].append([tmp[col][index[0]],tmp[col][index[1]]])
            else:
                result[col] = []
                for i,index in enumerate(indices):
                    result[col].append(tmp[col][i])
        return pd.DataFrame.from_dict(result)

    '''Bounding box IoU utility function'''

    def bb_intersection_over_union(self, xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(xmin1, xmin2)
        yA = max(ymin1, ymin2)
        xB = min(xmax1, xmax2)
        yB = min(ymax1, ymax2)
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
        boxBArea = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / (boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

if __name__ == '__main__':
    import os
    path = '/home/jeff/workspace/Migdal_backup/gui/sample_data/gui_data/'
    for fi in sorted(os.listdir(path)):
        a = group_and_search(in_file = path+fi,migdal_cut = '4 <= ER_energy <= 6 & 60 <= NR_energy <= 300 & centroid_dist <= 6')
        #a = group_and_search(in_file = path+fi,migdal_cut = 'NR_energy >= 60 & centroid_dist <= 6')
    #group_and_search()
    #a = group_and_search(in_file = '/home/jeff/workspace/Migdal_backup/gui/sample_data/gui_data/MIG_DD_570V_240205T134921.DATA.0970.feather')
