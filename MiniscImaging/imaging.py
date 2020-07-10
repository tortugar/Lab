#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compatible with python 3, but not yet fully tested
"""
import os
import os.path
import numpy as np
import matplotlib.pylab as plt
import h5py
import scipy.signal
import re
import scipy.io as so
from roipoly import roipoly
import shutil
import pdb
import sleepy
import pandas as pd
import seaborn as sns
from scipy import linalg as LA

### DEBUGGER
#import pdb


class TIFFStack:
    """
    Simple Wrapper class for H5DF stacks.
    """

    def __init__(self, path, name, nblock=1000) :
        self.nblock = 1000
        self.path = path
        self.name = name

        self.fid = h5py.File(os.path.join(path, name), 'r')
        self.dset = self.fid['images']
        self.nx = self.dset.shape[1]
        self.ny = self.dset.shape[2]
        self.nframes = self.dset.shape[0]

        a = re.split('\.', self.name)
        self.mean_name = a[0] + 'mean.mat'
        if os.path.isfile(os.path.join(self.path, self.mean_name)) :
            self.avg = so.loadmat(os.path.join(self.path, self.mean_name))['mean']
        else :
            self.avg = np.zeros((self.nx, self.ny))
            
        
    def mean(self):
        """
        def mean():
        calculate average frame, by averaging over z direction
        """
        A = np.zeros((self.nx, self.ny))
        niter = int(np.ceil(self.nframes/(1.0*self.nblock)))
        for i in range(niter) :
            nup = np.min(((i+1)*self.nblock, self.nframes))
            A = A + np.sum(self.dset[i*self.nblock:nup,:,:], axis=0)
        self.avg = A.astype('float64') / (self.nframes*1.0)
        return self.avg


    def save_mean(self):
        """
        save_mean() :
        save self.avg to mat file
        """
        print(self.mean_name)
        so.savemat(os.path.join(self.path, self.mean_name), {'mean' : self.avg})


    def set_mean(self, A):
        self.avg = A


    def slice(self, x, y) :
        A = np.zeros((self.nframes,))
        niter = int(np.ceil(self.nframes/(1.0*self.nblock)))
        for i in range(niter) :
            nup = np.min(((i+1)*self.nblock, self.nframes))
            A[i*self.nblock:nup] = self.dset[i*self.nblock:nup,x,y][0]

        return A


    def get_rois(self, roi_list):
        """
        ROI = get_rois(roi_list) 
        return matrix @ROI where the columns hold the actitivity of all rois. 
        @roilist   -     list of rois (each roi is a tuple with two arrays for x and y coordinates) 
        @RETURN: 
        @ROI       -     matrix where columns hold roi activities 
        """   
        nroi = len(roi_list)
        ROI = np.zeros((self.nframes, nroi))
        niter = int(np.ceil(self.nframes / (self.nblock*1.0)))

        # read the image stack blockwise
        for i in range(niter) :
            nup = np.min(((i+1)*self.nblock, self.nframes))
            D = self.dset[i*self.nblock:nup,0:self.nx,0:self.ny]
            iroi = 0
            for roi in roi_list :
                a = roi[0]
                b = roi[1]
                ROI[i*self.nblock:nup:,iroi] = D[:,a,b].mean(axis=1)
                iroi = iroi+1
        
        return ROI


    def close(self) :
        self.fid.close()



def save_roilist(ipath, name, roi_list, bnd_list, roi_id=0) :
    """
    save a list of ROIs following my naming convention
    the roi list is save in a mat file called
    recording_$name_roin(\d).mat
    where (\d) serves as an individual identifier to save multiple roi lists
    ipath     -     Imaging base folder
    name      -     Name of \"recording\"
    @RETURN:
    (fname,n) -     name of roilist following naming convention:
                    recording_ $name _ roilistn$n.mat
    """
    ddir = os.path.join(ipath, name)
    fname_base = 'recording_' + name + '_roilistn' 
    files = [f for f in os.listdir(ddir) if re.match(fname_base, f)]
    l = []
    for f in files :
        a = re.search('^' + fname_base + "(\d+)", f)
        if a :
            l.append(int(a.group(1)))
            
    if roi_id == 0:
        n = 1
        if l: n = max(l) + 1
    else:
        n = roi_id

    fname = fname_base + str(n) + '.mat'
    print("Saving roi list to %s" % fname)
    so.savemat(os.path.join(ddir, fname), {'roi_list': roi_list, 'bnd_list': bnd_list})

    return n, fname



def load_roilist(ipath, name, n) :
    """
    load_roilist(ipath, name, n) :
    load roi list $n of recording $name located in imaging folder $ipath
    @RETURN:
    roi_list and bnd_list
    """
    ddir = os.path.join(ipath, name)
    fname_base = 'recording_' + name + '_roilistn'
    fname = fname_base + str(n) + '.mat'
    a = so.loadmat(os.path.join(ddir, fname))['roi_list']
    b = so.loadmat(os.path.join(ddir, fname))['bnd_list']
    # 10/06/17 needed to introduce special case for just one ROI; still some bug here
    # need to figure out:
    if len(a) == 1:
        roi_list = [(a[0][0], a[0][1])]
        bnd_list = [(b[0][0], b[0][1])]
    else:
        roi_list = [(k[0][0], k[1][0]) for k in a]
        bnd_list = [(list(k[0][0]), list(k[1][0])) for k in b]
        #bnd_list = [(k[0][0], k[1][0]) for k in b]
    
    return (roi_list, bnd_list)



def show_rois(ipath, name, idx_list, bnd_list,blk=False) :
    """
    show_rois(ipath, name, idx_list, bnd_list)
    display all ROIs along with IDs
    ipath     -     imaging folder
    name      -     name of recording
    idx_list  -     list of all x, y coordinates within the ROI
    bnd_list  -     list of all boundary pixels
    
    use only for imaging_gui.py
    """
    ddir = os.path.join(ipath, name)
    #img = so.loadmat(os.path.join(ddir, 'recording_' + name + '_alignedmean.mat'))['mean']
    img = so.loadmat(os.path.join(ddir, 'recording_' + name + '_diskmean.mat'))['mean']

    # show the image with both ROIs and their mean values
    plt.figure()
    ax = plt.subplot(111)
    ax.imshow(img, cmap='gray')

    # get colormap
    nroi = len(idx_list)
    cmap = plt.get_cmap('jet')
    cmap = cmap(list(range(0, 256)))[:, 0:3]
    cmap = downsample_matrix(cmap, int(np.floor(1.0*256/nroi)))

    for (r,i) in zip(bnd_list,list(range(0, len(bnd_list)))):
        allxpoints = r[0]
        allypoints = r[1]
        l = plt.Line2D(allxpoints+[allxpoints[0]], allypoints+[allypoints[0]], color=cmap[i,:], lw=2.)
        ax.add_line(l)
        plt.draw()
        plt.text(allxpoints[0], allypoints[0], str(i), fontsize=14, color=cmap[i,:],bbox=dict(facecolor='w', alpha=0.))
    plt.xticks([])
    plt.yticks([])
    plt.show(block=blk)
    


def draw_rois(ROIs, axes, c, show_num=True):
    """
    helper function for plot_rois
    """
    i=0
    for (x,y) in ROIs :
        l = plt.Line2D(y+[y[0]], x+[x[0]], color=c)
        if show_num:
            axes.text(np.max(y)-5, np.min(x)+7, str(i), fontsize=10, color=c,bbox=dict(facecolor='w', alpha=0.))
        axes.add_line(l)
        i = i+1



def plot_rois(ipath, name, roi_id, roi_set=[], amap=True, show_num=True):
    """
    plot ROIs along with activity map or average image
    :param ipath, name: imaging false folder and recording name
    :param roi_id: int, id of roi list 
    :param roi_set: list, selectino of ROIs within roi list; if empty, all ROIs
           are plotted
    :param amap: if True, plot activity map, otherwise show average frame
    :param show_num: if True, show ROI numbers
    """
    if amap:
        image = so.loadmat(os.path.join(ipath, name, 'recording_' + name + '_actmap.mat'))['mean']
    else:
        image = so.loadmat(os.path.join(ipath, name, 'recording_' + name + '_alignedmean.mat'))['mean']
    plt.figure()
    axes = plt.subplot(111)
    axes.imshow(image, cmap='gray', vmin=0, vmax=np.percentile(image, 99.5))
    
    (ROI_coords, ROIs) = load_roilist(ipath, name, roi_id)
    if len(roi_set) > 0:
        ROIs_sel = [ROIs[s] for s in roi_set]
    else:
        ROIs_sel = ROIs
    
    draw_rois(ROIs_sel, axes, 'red', show_num=show_num)



def show_roilist(ipath, name, n, blk=True) :
    """
    show roilist n of imaging name
    see also: load_roilist, show_rois
    """
    (roi_list, bnd_list) = load_roilist(ipath, name, n)
    show_rois(ipath, name, roi_list, bnd_list, blk)
    
    plt.show(block=blk)



def save_catraces(ipath, name, n, ROI, bROI, F=0, cf=0) :
    """
    save the calcium traces belonging to roilist n in file
    called recording_$name_traces$n.mat


    ipath      -      imaging base folder
    name       -      name of considered recording
    n          -      id of considered roilist
    ROI        -      calcium traces of ROIs
    bROI       -      background (surround) of ROIs
    cf         -      correction factor
    """
    ddir = os.path.join(ipath, name)
    fname_base = 'recording_' + name + '_tracesn' 
    
    fname = fname_base + str(n) + '.mat'
    print("Saving Ca traces of roilist %d to %s" % (n, fname))

    so.savemat(os.path.join(ddir, fname), {'ROI': ROI, 'bROI': bROI, 'F' : F, 'cf' : cf})
    return fname




def calc_brstates(ipath, name, roi_id, cf=0, bcorr=1) :
    """
    calculate average DF/F activity for each brain state
    see also plot_catraces
    
    @PARAM:
    ipath     -     base imaging folder
    name      -     name of imaging sessino
    roi_id    -     ID of ROI list to be used
    cf        -     correction factor
    bcorr     -     [0|1] baseline correction
    
    @RETURN:
    nroi x 3 matrix listing i row i the brain state dependent active of ROI i
    
    """
    # sometimes the first frames are black; discard these frames
    FIRSTFRAMES = 100
    
    D = so.loadmat(os.path.join(ipath, name, 'recording_' + name + '_tracesn' + str(roi_id) + '.mat'))
    ROI  = D['ROI']
    bROI = D['bROI']
    img_time = imaging_timing(ipath, name)
    nframes = np.min((ROI.shape[0], img_time.shape[0]))
    
    F = np.zeros((nframes, ROI.shape[1]))
    for i in range(ROI.shape[1]) :
        a = ROI[0:nframes,i]-cf*bROI[0:nframes,i]
        pc = np.percentile(a, 20)

        #baseline correction
        if bcorr == 1:
            idx = np.where(a<pc)[0]
            idx = idx[np.where(idx>FIRSTFRAMES)[0]]
            v = a[idx]
            t = img_time[idx]
            p = least_squares(t, v, 1)[0]
            basel = img_time*p[0]+p[1]
        else :
            basel = pc

        F[:,i] = np.divide(a-basel, basel)

    # collect brainstate information
    sdt = 2.5
    M = sleepy.load_stateidx(ipath, name)[0]
    sp_time = np.arange(0, sdt*M.shape[0], sdt)
    # assigns to each frame a time point:
    img_time = imaging_timing(ipath, name)

    # Second, brain state dependent averages
    nroi = F.shape[1]
    S = np.zeros((nroi,3))
    tborder = 10

    for i in range(nroi) :
        for istate in range(1,4) :
            idx = np.nonzero(M==istate)[0]
            seq = get_sequences(idx)

            fidx = []
            for s in seq :
                # eeg2img_time: EEG time |---> Frame Indices
                a = eeg2img_time([s[0]*sdt, s[-1]*sdt], img_time)
                fidx = fidx + list(range(a[0],a[1]+1))

            S[i,istate-1] = np.mean(F[np.array(fidx),i])

    return S


### DEPRICATED ################################################################
def calc_catransition(ipath, name, roi_id, statei, statej, pre, post, thr_pre, thr_post, cf=0) :
    """
    collect all transitions (of all ROIs) from $statei to $statej
    @PARAM:
    ipath     -     base imaging folder
    name      -     name of imaging session
    roi_id    -     ID of roi list
    statei    -     consider transitions from statei to ...
    statej    -     statej
    pre       -     cut out $pre seconds before and ..
    post      -     $post seconds after the transition point
    thr_pre   -     enfore that $thr_pre seconds of the preceding period are indeed $statei
    thr_post  -     enfore that $thr_post seconds of the following periods are indeed $statej

    @RETURN:
    List of $nroi X time matrices for each transition following the required criteria
    """

    SR = get_snr(ipath, name)
    D = so.loadmat(os.path.join(ipath, name, 'recording_'+name+'_tracesn'+str(roi_id)+'.mat'))
    ROI  = D['ROI']
    bROI = D['bROI']
    img_time = imaging_timing(ipath, name)
    nframes = np.min((ROI.shape[0], img_time.shape[0]))

    # make sure img_time and ROI, bROI have same length
    # (although that shouldn't be a problem
    if len(img_time) > nframes :
        img_time = img_time[0:nframes]

    # ROI: time x "no. of ROIs"
    if ROI.shape[0] > nframes :
        ROI  = ROI[0:nframes,:]
        bROI = bROI[0:nframes,:]
    
    nroi = ROI.shape[1]
    F = np.zeros((nframes, nroi))

    for i in range(nroi) :
        A = ROI[:,i] - cf*bROI[:,i]
        Basel = baseline_correction(A, img_time)
        F[:,i] = np.divide(A-Basel, Basel)


    # load brainstate annotation
    sdt = 2.5
    idt = 1. / SR
    M = sleepy.load_stateidx(ipath, name)[0]
    sp_time = np.arange(0, sdt*M.shape[0], sdt)


    idx = np.nonzero(M==statei)[0]
    seq = get_sequences(idx)
    # HERE 

    # preceding and following time interval in frame indices:
    Blocks = []
    ipre  = int(np.round(pre/idt))
    ipost = int(np.round(post/idt))
    
    for s in seq :
        #s[-1]+1 is the start index of state j in BS time
        #istart_jindex is the first imaging frame of state j
        istart_jindex   = eeg2img_time([(s[-1]+1)*sdt], img_time)[0]
        dur_seq    = len(s)*sdt
        
        if (istart_jindex-ipre > 0) and (istart_jindex+ipost < len(img_time)) :
            if dur_seq >= thr_pre :
                if M[s[-1]+1] == statej :
                    
                    j = s[-1]+1
                    while( (M[j] == statej) and (j < len(M)-1) ) :
                        j = j+1
                    j=j-1
                    dur_statej = (j-(s[-1]+1)+1)*sdt

                    if dur_statej >= thr_post :
                        A = F[(istart_jindex-ipre):(istart_jindex+ipost),:]
                        Blocks.append(A.T)
                        
    return Blocks

    

def baseline_correction(F, time, perc=20, firstframes=100):
    pc = np.percentile(F, perc)
    idx = np.where(F<pc)[0]
    idx = idx[np.where(idx>firstframes)[0]]
    p = least_squares(time[idx], F[idx], 1)[0]
    basel = time*p[0]+p[1]

    return basel



def calculate_dff(ipath, name, roi_id):
    
    # load correction factor
    if os.path.isfile(os.path.join(ipath, name, '%s_cf.mat'%name)):
        cf = so.loadmat(os.path.join(ipath, name, '%s_cf.mat'%name), squeeze_me=True)['cf']  
    else:
        cf = 0.9

    D = so.loadmat(os.path.join(ipath, name, 'recording_' + name + '_tracesn' + str(roi_id) + '.mat'))
    ROI  = D['ROI']
    bROI = D['bROI']
    img_time = imaging_timing(ipath, name)
    
    nframes = ROI.shape[0]
    DFF = np.zeros((nframes, ROI.shape[1]))
    for i in range(ROI.shape[1]) :
        a = ROI[0:nframes,i]-cf*bROI[0:nframes,i]
        
        basel = baseline_correction(a, img_time)
        DFF[:,i] = np.divide(a-basel, basel)

    so.savemat(os.path.join(ipath, name, 'recording_' + name + '_dffn' + str(roi_id) + '.mat'), {'dff':DFF})



def load_roimapping(map_file):
    """
    :param map_file: csv_file with columns ROI 'ID', 'mouse' and all recordings.
           Each recordings is one column (using standard naming convention mouseid_datani).
           Each entry of a recording column has the format n-m, where n is the roilist id 
           and m is the actual ROI (of roilist n). 
           A given ROI 'ID' can be present in multiple recordings; if it is not
           part of a given recording the corresponding column entry is 'X'
    :return df, pd.DataFrame with columns 'ID', 'mouse', 'recording1', ...
    """
    
    df = pd.read_csv(map_file)
    rois = list(df['ID'])
                
    for r in rois:
        if re.match('^#\d+', str(r)):
            idx = df[df['ID']==r].index
            df = df.drop(idx)
                
    roi_mapping = df
    return roi_mapping



def merge_roimapping(mappings):
    map1 = mappings[0]
    
    for map2 in mappings[1:]:
        d={}
        d['mouse'] = list(map1['mouse']) + list(map2['mouse'])
        
        id1 = list(map1['ID'])
        id2 = list(map2['ID'])
        id2 = [i+max(id1)+1 for i in id2]
        d['ID'] = id1 + id2
                
        recordings1 = list(map1.columns)
        recordings1 = [r for r in recordings1 if re.match('^\S+_\d{6}n\d+$', r)]    

        recordings2 = list(map2.columns)
        recordings2 = [r for r in recordings2 if re.match('^\S+_\d{6}n\d+$', r)]    
        
        for r in recordings1:
            d[r] = list(map1[r]) + ['X']*len(id2)
            
        for r in recordings2:
            d[r] = ['X']*len(id1) + list(map2[r])
        
        map1 = pd.DataFrame(d)
    
    return map1



def brstate_dff(ipath, mapping, pzscore=False, class_mode='basic', single_mice=True):
    """
    calculate average ROI DF/F activity during each brain state and then 
    perform statistics for ROIs to classify them into REM-max, Wake-max, or NREM-max.
    For each ROI anova is performed, followed by Tukey-test
    
    :param ipath: base imaging folder
    :param mapping: pandas DataFrame as returned by &load_roimapping.
           The frame contains a column for the 
    :param pzscore: if True, z-score DF/F traces
    :param class_mode: class_mode == 'basic': classify ROIs into 
                       REM-max, Wake-max and NREM-max ROIs
                       class_mode == 'rem': further separate REM-max ROIs 
                       into REM > Wake > NREM (R>W>N) and REM > NREM > Wake (R>N>W) ROIs
    :param single_mice: boolean, if True use separate colors for single mice in 
                        summary plots
    """
    
    import pingouin as pg
    
    rois = list(mapping['ID'])
    
    roi_stateval = {}
    for r in rois:
        roi_stateval[r] = {1:[], 2:[], 3:[]}
    
    recordings = list(mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]    
    for rec in recordings:
        sr = sleepy.get_snr(ipath, rec)
        nbin = int(np.round(sr)*2.5)
        sdt = nbin * (1.0/sr)
        rec_map = mapping[mapping[rec] != 'X']   

        roi_list = int(re.split('-', rec_map.iloc[0][rec])[0])
        # load DF/F for recording rec 
        dff_file = os.path.join(ipath, rec, 'recording_' + rec + '_dffn' + str(roi_list) + '.mat')
        if not(os.path.isfile(dff_file)):
            calculate_dff(ipath, rec, roi_list)
        DFF = so.loadmat(dff_file, squeeze_me=True)['dff']
        # brainstate
        M = sleepy.load_stateidx(ipath, rec)[0]
        state_idx = {1:[], 2:[], 3:[]}
        
        # load imaging timing
        img_time = imaging_timing(ipath, rec)

        for state in [1,2,3]:
            seq = sleepy.get_sequences(np.where(M==state)[0])
            for s in seq:                
                # eeg2img_time: EEG time |---> Frame Indices
                a = eeg2img_time([s[0]*sdt, s[-1]*sdt], img_time)
                idx = range(a[0],a[1]+1)                
                state_idx[state] += list(idx)

        for index, row in rec_map.iterrows():
            s=row[rec]
            a = re.split('-', s)
            roi_list = int(a[0])
            roi_num  = int(a[1])
            print(row)
            print(DFF.shape)
            dff = DFF[:,roi_num] * 100
            if pzscore:
                dff = (dff-dff.mean()) / dff.std()
            
            for state in [1,2,3]:
                roi_stateval[row['ID']][state].append(dff[state_idx[state]])

    for r in rois:
        for state in [1,2,3]:
            roi_stateval[r][state] = np.concatenate(roi_stateval[r][state])
    
    
    columns = ['ID', 'R', 'W', 'N', 'F-anova', 'P-anova', 'P-tukey', 'Type']
    data = []
    for r in rois:
        stateval = roi_stateval[r]
        val = np.concatenate([stateval[1], stateval[2], stateval[3]])
        state = ['R']*len(stateval[1]) + ['W']*len(stateval[2]) + ['N']*len(stateval[3])
        d = {'state':state, 'val':val}
        df = pd.DataFrame(d)

        res = pg.anova(data=df, dv='val', between='state')
        
        res2 = pg.pairwise_tukey(data=df, dv='val', between='state')
 
        def _get_mean(s):
            return df[df['state']==s]['val'].mean()
 
        rmean = _get_mean('R')
        wmean = _get_mean('W')
        nmean = _get_mean('N')
        
        if class_mode == 'basic':
            
            roi_type = 'X'
            # REM-max
            if (rmean > wmean) and (rmean > nmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'R')]
                cond2 = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]
                
                if cond1['p-tukey'].iloc[0] <= 0.05 and cond2['p-tukey'].iloc[0] <= 0.05:
                    roi_type = 'R-max'
            # W-max
            elif (wmean > nmean) and (wmean > rmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]
                cond2 = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]
    
                if cond1['p-tukey'].iloc[0] <= 0.05 and cond2['p-tukey'].iloc[0] <= 0.05:
                    roi_type = 'W-max'
            # N-max 
            elif (nmean > wmean) and (nmean > rmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]
                cond2 = res2[(res2['A'] == 'N') & (res2['B'] == 'R')]
    
                if cond1['p-tukey'].iloc[0] <= 0.05 and cond2['p-tukey'].iloc[0] <= 0.05:
                    roi_type = 'N-max'
                    
            else:
                roi_type = 'X'
                            
            tmp = [r, rmean, wmean, nmean, res.F.iloc[0], res['p-unc'].iloc[0], res2['p-tukey'].iloc[0], roi_type]
            data.append(tmp)
            
        
        else:
            # R>W>N
            if (rmean > wmean) and (rmean > nmean) and (wmean  > nmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'R')]
                cond2 = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]
                
                if cond1['p-tukey'].iloc[0] <= 0.05 and cond2['p-tukey'].iloc[0] <= 0.05:
                    roi_type = 'R>W>N'
            # R>N>W
            elif (rmean > wmean) and (rmean > nmean) and (nmean  > wmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'R')]
                cond2 = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]
                
                if cond1['p-tukey'].iloc[0] <= 0.05 and cond2['p-tukey'].iloc[0] <= 0.05:
                    roi_type = 'R>N>W'
            # W-max
            elif (wmean > nmean) and (wmean > rmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]
                cond2 = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]
    
                if cond1['p-tukey'].iloc[0] <= 0.05 and cond2['p-tukey'].iloc[0] <= 0.05:
                    roi_type = 'W-max'
            # N-max 
            elif (nmean > wmean) and (nmean > rmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]
                cond2 = res2[(res2['A'] == 'N') & (res2['B'] == 'R')]
    
                if cond1['p-tukey'].iloc[0] <= 0.05 and cond2['p-tukey'].iloc[0] <= 0.05:
                    roi_type = 'N-max'
                    
            else:
                roi_type = 'X'
                            
            tmp = [r, rmean, wmean, nmean, res.F.iloc[0], res['p-unc'].iloc[0], res2['p-tukey'].iloc[0], roi_type]
            data.append(tmp)


    df_class = pd.DataFrame(data, columns=columns)
    df_class = pd.merge(mapping, df_class, on='ID')

    mice = [m for m in df_class['mouse'].unique()]
    j = 0
    mdict = {}
    for m in mice:
        mdict[m] = j
        j+=1
    clrs = sns.color_palette("husl", len(mice))

    plt.ion()
    plt.figure()
    types = df_class['Type'].unique()
    types = [i for i in types if not (i=='X')]
    types.sort()
    
    j = 1
    for typ in types:
        mouse_shown = {m:0 for m in mice}
        plt.subplot('1%d%d' % (len(types), j))
        df = df_class[df_class['Type']==typ][['mouse', 'R', 'N', 'W']]
        #df = pd.melt(df, var_name='state', value_name='dff')  
        
        sns.barplot(data=df[['R', 'N', 'W']], color='gray')
        for index, row in df.iterrows():
            if single_mice:
                m=row['mouse']                
                if mouse_shown[m] > 0:
                    plt.plot(['R', 'N', 'W'], row[['R', 'N', 'W']], color=clrs[mdict[m]])
                else:
                    plt.plot(['R', 'N', 'W'], row[['R', 'N', 'W']], color=clrs[mdict[m]], label=m)
                    mouse_shown[m] += 1   
            else:
                plt.plot(['R', 'N', 'W'], row[['R', 'N', 'W']], color='black')

        sns.despine()
        plt.title(typ)
        plt.legend()
        if j == 1:
            if not pzscore:
                plt.ylabel('DF/F (%)')
            else:
                plt.ylabel('DF/F (z-scored)')
        j += 1
    
    return df_class
                        

            
def time_morph(X, nstates):
    """
    upsample vector or matrix X to nstates states
    :param X, vector or matrix; if matrix, the rows are upsampled.
    :param nstates, number of elements or rows of returned vector or matrix
    
    I want to upsample m by a factor of x such that
    x*m % nstates == 0, 
    a simple soluation is to set x = nstates
    then nstates * m / nstates = m.
    so upsampling X by a factor of nstates and then downsampling by a factor
    of m is a simple solution...
    """
    m = X.shape[0]
    A = upsample_mx(X, nstates)
    # now we have m * nstates rows
    if X.ndim == 1:
        Y = downsample_vec(A, int((m*nstates)/nstates))
    else:
        Y = downsample_mx(A, int((m*nstates)/nstates))
    # now we have m rows as requested 
    return Y


def downsample_mx(X, nbin):
    """
    y = downsample_vec(x, nbin)
    downsample the vector x by replacing nbin consecutive 
    bin by their mean 
    @RETURN: the downsampled vector 
    """
    n_down = int(np.floor(X.shape[0] / nbin))
    X = X[0:n_down*nbin,:]
    X_down = np.zeros((n_down,X.shape[1]))

    # 0 1 2 | 3 4 5 | 6 7 8 
    for i in range(nbin) :
        idx = range(i, int(n_down*nbin), int(nbin))
        X_down += X[idx,:]

    return X_down / nbin    



def downsample_vec(x, nbin):
    """
    y = downsample_vec(x, nbin)
    downsample the vector x by replacing nbin consecutive 
    bin by their mean 
    @RETURN: the downsampled vector 
    """
    n_down = int(np.floor(len(x) / nbin))
    x = x[0:n_down*nbin]
    x_down = np.zeros((n_down,))

    # 0 1 2 | 3 4 5 | 6 7 8 
    for i in range(nbin) :
        idx = range(i, int(n_down*nbin), int(nbin))
        x_down += x[idx]

    return x_down / nbin   



def upsample_mx(x, nbin):
    """
    if x is a vector:
        upsample the given vector $x by duplicating each element $nbin times
    if x is a 2d array:
        upsample each matrix by duplication each row $nbin times        
    """
    if nbin == 1:
        return x
    
    nelem = x.shape[0]
    if x.ndim == 1:        
        y = np.zeros((nelem*nbin,))
        for k in range(nbin):
            y[k::nbin] = x
    else:
        y = np.zeros((nelem*nbin,x.shape[1]))
        for k in range(nbin):
            y[k::nbin,:] = x

    return y


       
def brstate_transitions(ipath, roi_mapping, transitions, pre, post, si_threshold, sj_threshold, xdt=1.0, pzscore=False):
    """
    calculate average DF/F activity for ROIs along brain state transitions
    :param ipath: base folder
    :param roi_mapping: pandas DataFrame with columns specificing ROI 'ID', recordings ('MouseID_dateni')
    :param transitions: list of tuples to denote transitions to be considered;
           1 - REM, 2 - Wake, 3 - NREM; For example to calculate NREM to REM and REM to wake transitions,
           type [(3,1), (1,2)]
    :param pre: time before transition in s
    :param post: time after transition
    :param si_threshold: list of floats, thresholds how long REM, Wake, NREM should be at least before the transition.
           So, if there's a REM to Wake transition, but the duration of REM is shorter then si_threshold[0], then this
           transition if discarded.
    :param sj_threshold: list of floats, thresholds how long REM, Wake, NREM should be at least after the transition
    :param xdt: time resolution for DF/F activity
    :param pzscore: bool, if True zscore DF/F signals

    Example call:
    mx, df = imaging.brstate_transitions(ppath, df_class[df_class['Type']=='N-max'], [[3,1], [1,2]], 60, 30, [60, 60, 60], [30, 30, 30])
    """
    
    rois = list(roi_mapping['ID'])
    
    roi_transact_si = dict()
    roi_transact_sj = dict()
    
    roi_length = dict()
    #trans_spe = dict()
    #trans_spm = dict()
    states = {1:'R', 2:'W', 3:'N'}
    for (si,sj) in transitions:
        sid = states[si] + states[sj]
        roi_transact_si[sid] = {r:[] for r in rois}
        roi_transact_sj[sid] = {r:[] for r in rois}

        roi_length[sid] = {r:[] for r in rois}
        #trans_spe[sid] = []
        #trans_spm[sid] = []

    roi_stateval = {}
    for r in rois:
        roi_stateval[r] = {1:[], 2:[], 3:[]}
    
    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]    
    
    for rec in recordings:
        sr = sleepy.get_snr(ipath, rec)
        nbin = int(np.round(sr)*2.5)
        sdt = nbin * (1.0/sr)
        rec_map = roi_mapping[roi_mapping[rec] != 'X']   
        if sum(roi_mapping[rec] == 'X') == len(roi_mapping):
            print('Recording %s: no ROI present' % rec)
            continue

        roi_list = int(re.split('-', rec_map.iloc[0][rec])[0])
        # load DF/F for recording rec 
        dff_file = os.path.join(ipath, rec, 'recording_' + rec + '_dffn' + str(roi_list) + '.mat')
        if not(os.path.isfile(dff_file)):
            calculate_dff(ipath, rec, roi_list)
        DFF = so.loadmat(dff_file, squeeze_me=True)['dff']
        # brainstate
        M = sleepy.load_stateidx(ipath, rec)[0]
        
        # load imaging timing
        img_time = imaging_timing(ipath, rec)
        idt = np.diff(img_time).mean()
                
        ipre  = int(np.round(pre/idt))
        ipost = int(np.round(post/idt))

        for index, row in rec_map.iterrows():
            s=row[rec]
            a = re.split('-', s)
            roi_num  = int(a[1])
            dff = DFF[:,roi_num]
            if pzscore:
                dff = (dff-dff.mean())/dff.std()

            for (si,sj) in transitions:
                sid = states[si] + states[sj]
                seq = sleepy.get_sequences(np.where(M==si)[0])
                for s in seq:
                    ti = s[-1]
    
                    # check if next state is sj; only then continue
                    if ti < len(M)-1 and M[ti+1] == sj:
                        # go into future
                        p = ti+1
                        while p<len(M)-1 and M[p] == sj:
                            p += 1
                        p -= 1
                        sj_idx = list(range(ti+1, p+1))
                        # so the indices of state si are seq
                        # the indices of state sj are sj_idx
        
                        jstart_dff   = eeg2img_time([(s[-1]+1)*sdt], img_time)[0]

                        if ipre <= jstart_dff < len(dff)-ipost and len(s)*sdt >= si_threshold[si-1] and len(sj_idx)*sdt >= sj_threshold[sj-1]:
    
                            istart_dff = eeg2img_time([(s[-1]+1)*sdt - pre],  img_time)[0]
                            jend_dff   = eeg2img_time([(s[-1]+1)*sdt + post], img_time)[0]
                            
                            act_si = dff[istart_dff:jstart_dff]
                            act_sj = dff[jstart_dff:jend_dff]
                            
                            #act_si = dff[istart_dff:jstart_dff]
                            #act_sj = dff[jstart_dff:jend_dff]
                            
                            #act = np.concatenate((act_si, act_sj))
                            roi_length[sid][row['ID']].append((len(act_si), len(act_sj)))
                        
                            roi_transact_si[sid][row['ID']].append(act_si)
                            roi_transact_sj[sid][row['ID']].append(act_sj)
    
    si_len = []
    sj_len = []
    
    for index, row in roi_mapping.iterrows():
        for (si,sj) in transitions:
            sid = states[si] + states[sj]
            tmp = roi_length[sid][row['ID']]
            tmp_si = [l[0] for l in tmp]
            tmp_sj = [l[1] for l in tmp]
            
            si_len += tmp_si
            sj_len += tmp_sj
            
    #si_min = max(si_len)
    #sj_min = max(sj_len)
    ntime = 0
    for index, row in roi_mapping.iterrows():
        for (si,sj) in transitions:
            sid = states[si] + states[sj]
            
            tmp_si = roi_transact_si[sid][row['ID']]
            tmp_sj = roi_transact_sj[sid][row['ID']]

            if len(tmp_si) > 0:                
                #tmp_si = np.vstack([t[-si_min:] for t in tmp_si])
                #roi_transact_si[sid][row['ID']] = tmp_si
                
                #tmp_sj = np.vstack([t[0:sj_min] for t in tmp_sj])
                #roi_transact_sj[sid][row['ID']] = tmp_sj
                #pdb.set_trace()
                tmp_si = np.vstack([time_morph(t,int(pre/xdt)) for t in tmp_si])
                tmp_sj = np.vstack([time_morph(t,int(post/xdt)) for t in tmp_sj])
                roi_transact_si[sid][row['ID']] = tmp_si
                roi_transact_sj[sid][row['ID']] = tmp_sj
                ntime = tmp_si.shape[1] + tmp_sj.shape[1]
            else:
                print(row['ID'])

            print('Done with ROI %d' % row['ID'])

    ti = np.linspace(-pre, -xdt, int(pre/xdt))
    tj = np.linspace(0, post-xdt, int(post/xdt))
    xtime = np.concatenate((ti,tj))
    #ntime = tmp_si.shape[1] + tmp_sj.shape[1]
    
    roi_transact_mean = dict()
    mx_transact = dict()
    for (si,sj) in transitions:
        sid = states[si] + states[sj]
        roi_transact_mean[sid] = {r:[] for r in rois}
        mx_transact[sid] = np.zeros((len(rois), ntime))
    
    j = 0
    d = {'ID':[], 'mouse':[], 'time':[], 'dff':[], 'trans':[]}
    for index, row in roi_mapping.iterrows():
        for (si,sj) in transitions:
            sid = states[si] + states[sj]
            tmp_si = roi_transact_si[sid][row['ID']]
            tmp_sj = roi_transact_sj[sid][row['ID']]
            roi_transact_mean[sid][row['ID']] = np.hstack([tmp_si, tmp_sj])
            mx_transact[sid][j,:] = roi_transact_mean[sid][row['ID']].mean(axis=0)

            d['ID'] += [row['ID']]*ntime
            d['mouse'] += [row['mouse']]*ntime
            d['time'] += list(xtime)
            d['dff'] += list(mx_transact[sid][j,:])
            d['trans'] += [sid]*ntime
       
        j+=1

    df = pd.DataFrame(d)            

    return mx_transact, df
                

    
def plot_catraces(ipath, name, roi_id, cf=0, bcorr=1, pspec = False, vm=[], freq_max=30,
                  pemg_ampl=True, r_mu = [10, 100], dff_legend=100):
    """
    plot Ca traces in a nice and organized way along with brain state annotation
    @ARGS
    ipath     -     Imaging main folder
    name      -     Name of Imaging session
    roi_id    -     id of roi list to be plotted
    cf        -     correction factor
    bcorr     -     baseline correction? [0|1]
    pspec     -     plot EEG spectrogram and EMG?
    vm        -     tuple, upper and lower range of color range of EEG spectrogram
    freq_max  -     maximum frequency for EEG spectrogram
    pemg_ampl -     if True, plot EMG amplitude, otherwise raw EMG
    r_mu      -     frequency range for EMG amplitude calculation
    dff_legend -    value between 0 and 100, length of scale bar for DF/F signal
    """
    import matplotlib.patches as patches
    plt.ion()

    # sometimes the first frames are black; discard these frames
    FIRSTFRAMES = 100
    
    D = so.loadmat(os.path.join(ipath, name, 'recording_' + name + '_tracesn' + str(roi_id) + '.mat'))
    ROI  = D['ROI']
    bROI = D['bROI']
    if os.path.isfile(os.path.join(ipath, name, 'AUX.mat')):
        img_time = imaging_timing(ipath, name)
        nframes = np.min((ROI.shape[0], img_time.shape[0]))
    else:
        #img_time = so.loadmat(os.path.join(ipath, name, 'img_time.mat'), squeeze_me=True)['time']
        img_time = imaging_timing(ipath, name)

    #This now takes the timestamp directly from the file
    #d = np.loadtxt(os.path.join(ipath, name,'timestamp.dat'),
    #                    delimiter='\t',
    #                    skiprows=1,
    #                    dtype={'names': ('camNum', 'frameNum', 'sysClock', 'buffer'),
    #                    'formats': ('float', 'float', 'float', 'float')})
    #img_time = d['sysClock']/1000.0
    #img_time = img_time[2:]


    #ROI = ROI[2:,:]
    #bROI= bROI[2:,:]

    #sampleROI1 = ROI[:,0]
    #sampleROI2 = ROI[:,3]
    #nonrepeats = []
    #for i in range(len(sampleROI1)-1):
    #    if (sampleROI1[i] != sampleROI1[i+1]) and (sampleROI2[i] != sampleROI2[i+1]):
    #        nonrepeats.append(i)
    #nonrepeats = np.array(nonrepeats)

    #img_time = img_time[nonrepeats]
    #ROI = ROI[nonrepeats,:]
    #bROI = bROI[nonrepeats, :]
    #nframes = len(nonrepeats)

    nframes = ROI.shape[0]
    F = np.zeros((nframes, ROI.shape[1]))
    for i in range(ROI.shape[1]) :
        a = ROI[0:nframes,i]-cf*bROI[0:nframes,i]
        pc = np.percentile(a, 20)

        #baseline correction
        if bcorr == 1:
            idx = np.where(a<pc)[0]
            idx = idx[np.where(idx>FIRSTFRAMES)[0]]
            v = a[idx]
            t = img_time[idx]
            p = least_squares(t, v, 1)[0]
            # 10/06/17 added 0:nframes to process Johnny's data
            #basel = img_time[0:nframes]*p[0]+p[1]
            basel = img_time*p[0]+p[1]
        else :
            basel = pc

        F[:,i] = np.divide(a-basel, basel)
    

    # create colormap for CA traces
    nroi = F.shape[1]
    cmap = plt.get_cmap('jet')
    cmap = cmap(list(range(0, 256)))[:,0:3]
    cmap = downsample_matrix(cmap, int(np.floor(256/nroi)))
    fmax = F.max()
    
    # collect brainstate information
    sdt = 2.5
    M = sleepy.load_stateidx(ipath, name)[0]
    sp_time = np.arange(0, sdt*M.shape[0], sdt)

    # Plotting all together: First, time dependent calcium traces
    plt.figure()
    axes_dff = plt.axes([0.1, 0.1, 0.8, 0.4])
    for istate in range(1,4):
        idx = np.nonzero(M==istate)[0]
        seq = sleepy.get_sequences(idx)
    
        for s in seq :
            if istate == 1 :
                axes_dff.add_patch(patches.Rectangle((s[0]*sdt, -fmax), len(s)*sdt, (nroi+1)*fmax, color=[0.8, 1.0, 1.0]))
            if istate == 2 :
                axes_dff.add_patch(patches.Rectangle((s[0]*sdt, -fmax), len(s)*sdt, (nroi+1)*fmax, color=[1, 0.8, 1]))
        
    for i in range(nroi):
        plt.plot(img_time[0:nframes], F[0:nframes,i]+i*fmax, color=cmap[i,:])
        plt.text(100, i*fmax+fmax/4, str(i), fontsize=12, color=cmap[i,:],bbox=dict(facecolor='w', alpha=0.))

    # vertical legend for DF/F
    plt.plot(np.ones((2,))*(img_time[nframes-1]-30), [0, dff_legend/100.0], color='black', lw=1)

    plt.xlim((0, img_time[nframes-1]))
    plt.ylim([-fmax, fmax*nroi])
    plt.yticks([])
    plt.xlabel('Time (s)')
    plt.show(block=False)
    axes_dff.spines["left"].set_visible(False)
    sleepy.box_off(axes_dff)

    # Second figure: brain state dependent averages
    nroi = F.shape[1]
    S = np.zeros((nroi,3))
    tborder = 10

    for i in range(nroi) :
        for istate in range(1,4) :
            idx = np.nonzero(M==istate)[0]
            seq = sleepy.get_sequences(idx)

            fidx = []
            for s in seq :
                if len(s)*2.5 >= 1.0*tborder :
                    a = eeg2img_time([s[0]*sdt + tborder, s[-1]*sdt], img_time)
                    fidx = fidx+list(range(a[0],a[1]+1))
            # pdb.set_trace()
            S[i,istate-1] = np.mean(F[np.array(fidx),i])

    if pspec:
        P = so.loadmat(os.path.join(ipath, name, 'sp_' + name + '.mat'), squeeze_me=True)
        freq = P['freq']
        ifreq = np.where(freq <= freq_max)[0]
        ES = P['SP'][ifreq, :]
        med = np.median(ES.max(axis=0))
        if len(vm) == 0:
            vm = [0, med*2.5]

        axes_spec = plt.axes([0.1, 0.7, 0.8, 0.2], sharex=axes_dff)
        axes_spec.pcolorfast(sp_time, freq[ifreq], ES[ifreq, :], cmap='jet', vmin=vm[0], vmax=vm[1])
        axes_spec.axis('tight')
        #axes_spec.set_xticklabels([])
        #axes_spec.set_xticks([])
        axes_spec.spines["bottom"].set_visible(False)
        plt.ylabel('Freq (Hz)')
        sleepy.box_off(axes_spec)
        plt.xlim([sp_time[0], sp_time[-1]])

        if pemg_ampl:
            P = so.loadmat(os.path.join(ipath, name, 'msp_%s.mat' % name), squeeze_me=True)
            SPEMG = P['mSP']
        else:
            emg = so.loadmat(os.path.join(ipath, name, 'EMG.mat'), squeeze_me=True)['EMG']
        axes_emg = plt.axes([0.1, 0.57, 0.8, 0.1], sharex=axes_dff)
        
        if pemg_ampl:
            i_mu = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
            p_mu = np.sqrt(SPEMG[i_mu, :].sum(axis=0) * (freq[1] - freq[0]))
            axes_emg.plot(sp_time, p_mu, color='black')

            # * 1000: to go from mV to uV
            #if len(emg_ticks) > 0:
            #    axes_emg.set_yticks(emg_ticks)
            plt.ylabel('Ampl. ' + '$\mathrm{(\mu V)}$')
            plt.xlim((sp_time[0], sp_time[-1]))
        else:
            SR = sleepy.get_snr(ipath, name)
            t_emg = np.arange(0, emg.shape[0])*(1.0/SR)
            axes_emg.plot(t_emg, emg, color='black', lw=0.2)
            plt.xlim((t_emg[0], t_emg[-1] + 1))
            plt.ylabel('EMG ' + '$\mathrm{(\mu V)}$')
            
        sleepy.box_off(axes_emg)

        plt.setp(axes_spec.get_xticklabels(), visible=False)
        plt.setp(axes_emg.get_xticklabels(), visible=False)

    # plot brain state dependent DF/F averages
    plt.figure()
    ax = plt.subplot(121)
    y = S.mean(axis=0)
    std = S.std(axis=0)
    state_colors = np.array([[0.8, 1.0, 1.0], [1.0, 0.8, 1.0], [0.8, 0.8, 0.8]])
    #plt.bar(np.array([1,2,3])-0.25, y, width=0.5, yerr=std/np.sqrt(nroi))
    for i in range(3) :
        plt.bar(i+1, y[i], width=0.5, yerr=std[i]/np.sqrt(nroi), color=state_colors[i,:], align='center')
    
    plt.ylabel('$\Delta$F/F')
    plt.xticks([1,2,3], ('REM', 'Wake', 'NREM'))

    # single cells
    for i in range(nroi) :
        plt.plot(np.arange(1,4), S[i,:], color=cmap[i,:], marker='o')
    sleepy.box_off(ax)    
    plt.show()

    t = img_time[0:nframes] 
    #F = ROI[0:nframes,:] - bROI[0:nframes,:]

    return F, t


    
def plot_catraces_simple(ipath, name, roi_id, cf=0, bcorr=1, SR=0):
    """
    plot Ca traces of all ROIs without any brainstate information
    @ARGS
    ipath     -     Imaging main folder
    name      -     Name of Imaging session
    roi_id    -     id of roi list to be plotted
    cf        -     correction factor
    bcorr     -     baseline correction? [0|1]
    SR        -     sampling rate or frames per second of ca camera
    """
    # sometimes the first frames are black; discard these frames
    FIRSTFRAMES = 100
    
    D = so.loadmat(os.path.join(ipath, name, 'recording_' + name + '_tracesn' + str(roi_id) + '.mat'))
    ROI  = D['ROI']
    bROI = D['bROI']
    nframes = ROI.shape[0]
    if SR==0:
        SR = get_snr(ipath, name)
    dt = 1.0/SR
    img_time = np.arange(0,nframes*dt,dt)
    
    F = np.zeros((nframes, ROI.shape[1]))
    for i in range(ROI.shape[1]) :
        a = ROI[0:nframes,i]-cf*bROI[0:nframes,i]
        pc = np.percentile(a, 20)

        #baseline correction
        if bcorr == 1:
            idx = np.where(a<pc)[0]
            idx = idx[np.where(idx>FIRSTFRAMES)[0]]
            v = a[idx]
            t = img_time[idx]
            p = least_squares(t, v, 1)[0]
            basel = img_time*p[0]+p[1]
        else :
            basel = pc
        # Delta F/F
        F[:,i] = np.divide(a-basel, basel)

    # create colormap for CA traces
    nroi = F.shape[1]
    cmap = plt.get_cmap('jet')
    cmap = cmap(list(range(0, 256)))[:,0:3]
    cmap = downsample_matrix(cmap, int(np.floor(256/nroi)))
    fmax = F.max()


    # Plotting all together: First, time dependent calcium traces
    plt.figure()
    plt.subplot(111)
    
    for i in range(nroi):
        plt.plot(img_time[0:nframes], F[0:nframes,i]+i*fmax, color=cmap[i,:])
        plt.text(10, i*fmax+fmax/4, str(i), fontsize=14, color=cmap[i,:],bbox=dict(facecolor='w', alpha=0.))


    # vertical indication 20% DF/F
    plt.plot(np.ones((2,))*(img_time[nframes-1]-30), [0, 0.2], color='black', lw=3)

    plt.xlim((0, img_time[nframes-1]))
    plt.ylim([-fmax, fmax*nroi])
    plt.yticks([])
    plt.xlabel('Time (s)')
    plt.show(block=False)



def plot_catraces_avg(ipath, name, roi_id, roi_set=[], cf=0, bcorr=1, pspec = False, vm=[], freq_max=30, 
                      pemg_ampl=True, r_mu = [10, 100], dff_legend=20):
    """
    Average calcium traces over set of rois and plot the avg Ca trace 
    in a nice and organized way along with brain state annotation
    
    @ARGS
    ipath     -     Imaging main folder
    name      -     Name of Imaging session
    roi_id    -     id of roi list to be plotted
    roi_set   -     set (list of integers) that are selected to calculated avg dF/F signal;
                    if roi_set == [], calculate average across all rois
    cf        -     correction factor
    bcorr     -     baseline correction? [0|1]
    pspec     -     plot EEG spectrogram and EMG?
    vm        -     tuple, upper and lower range of color range of EEG spectrogram
    freq_max  -     maximum frequency for EEG spectrogram
    pemg_ampl -     if True, plot EMG amplitude, otherwise raw EMG
    r_mu      -     frequency range for EMG amplitude calculation
    dff_legend-     value between 0 and 100, length of scale bar for DF/F signal
    """
    import matplotlib.patches as patches
    plt.ion()

    # sometimes the first frames are black; discard these frames
    FIRSTFRAMES = 100
    
    D = so.loadmat(os.path.join(ipath, name, 'recording_' + name + '_tracesn' + str(roi_id) + '.mat'))
    ROI  = D['ROI']
    bROI = D['bROI']
    img_time = so.loadmat(os.path.join(ipath, name, 'img_time.mat'), squeeze_me=True)['time']        
    nframes = ROI.shape[0]
    F = np.zeros((nframes, ROI.shape[1]))

    for i in range(ROI.shape[1]) :
        a = ROI[0:nframes,i]-cf*bROI[0:nframes,i]
        pc = np.percentile(a, 20)

        #baseline correction
        if bcorr == 1:
            idx = np.where(a<pc)[0]
            idx = idx[np.where(idx>FIRSTFRAMES)[0]]
            v = a[idx]
            t = img_time[idx]
            p = least_squares(t, v, 1)[0]
            # 10/06/17 added 0:nframes to process Johnny's data
            #basel = img_time[0:nframes]*p[0]+p[1]
            basel = img_time*p[0]+p[1]
        else :
            basel = pc

        F[:,i] = np.divide(a-basel, basel)

    # create colormap for CA traces
    if len(roi_set) > 0:
        nroi = len(roi_set)
    else:
        nroi = F.shape[1]
    if len(roi_set) == 0:
        roi_set = list(range(0, nroi))
    
    Fmean = F[:,roi_set].mean(axis=1)

    cmap = plt.get_cmap('jet')
    cmap = cmap(list(range(0, 256)))[:,0:3]
    cmap = downsample_matrix(cmap, int(np.floor(256/nroi)))
    fmax = Fmean.max()
    fmin = Fmean.min()
    amp = fmax-fmin
    
    # collect brainstate information
    sdt = 2.5
    M = sleepy.load_stateidx(ipath, name)[0]
    sp_time = np.arange(0, sdt*M.shape[0], sdt)

    # Plotting all together: First, time dependent calcium traces
    plt.figure()
    axes_dff = plt.axes([0.1, 0.1, 0.8, 0.4])
    for istate in range(1,4):
        idx = np.nonzero(M==istate)[0]
        seq = sleepy.get_sequences(idx)
    
        for s in seq :
            if istate == 1 :
                axes_dff.add_patch(patches.Rectangle((s[0]*sdt, fmin-0.1*amp), len(s)*sdt, amp*1.2, color=[0.8, 1.0, 1.0]))
            if istate == 2 :
                axes_dff.add_patch(patches.Rectangle((s[0]*sdt, fmin-0.1*amp), len(s)*sdt, amp*1.2, color=[1, 0.8, 1]))
        
    plt.plot(img_time[0:nframes], Fmean[0:nframes], color='blue')

    # vertical legend for DF/F
    plt.plot(np.ones((2,))*(img_time[nframes-1]-30), [0, dff_legend/100.0], color='black', lw=3)

    plt.xlim((0, img_time[nframes-1]))
    plt.ylim([fmin-0.1*amp, fmax+0.1*amp])
    plt.yticks([])
    plt.xlabel('Time (s)')
    plt.show(block=False)
    axes_dff.spines["left"].set_visible(False)
    sleepy.box_off(axes_dff)

    if pspec:
        P = so.loadmat(os.path.join(ipath, name, 'sp_' + name + '.mat'), squeeze_me=True)
        freq = P['freq']
        ifreq = np.where(freq <= freq_max)[0]
        ES = P['SP'][ifreq, :]
        med = np.median(ES.max(axis=0))
        if len(vm) == 0:
            vm = [0, med*2.5]

        axes_spec = plt.axes([0.1, 0.7, 0.8, 0.2], sharex=axes_dff)
        axes_spec.pcolorfast(sp_time, freq[ifreq], ES[ifreq, :], cmap='jet', vmin=vm[0], vmax=vm[1])
        axes_spec.axis('tight')
        axes_spec.spines["bottom"].set_visible(False)
        plt.ylabel('Freq (Hz)')
        sleepy.box_off(axes_spec)
        plt.xlim([sp_time[0], sp_time[-1]])

        if pemg_ampl:
            P = so.loadmat(os.path.join(ipath, name, 'msp_%s.mat' % name), squeeze_me=True)
            SPEMG = P['mSP']
        else:
            emg = so.loadmat(os.path.join(ipath, name, 'EMG.mat'), squeeze_me=True)['EMG']
        axes_emg = plt.axes([0.1, 0.57, 0.8, 0.1], sharex=axes_dff)
        
        if pemg_ampl:
            i_mu = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
            p_mu = np.sqrt(SPEMG[i_mu, :].sum(axis=0) * (freq[1] - freq[0]))
            axes_emg.plot(sp_time, p_mu, color='black')
            plt.ylabel('Ampl. ' + '$\mathrm{(\mu V)}$')
            plt.xlim((sp_time[0], sp_time[-1]))
        else:
            SR = sleepy.get_snr(ipath, name)
            t_emg = np.arange(0, emg.shape[0])*(1.0/SR)
            axes_emg.plot(t_emg, emg, color='black', lw=0.2)
            plt.xlim((t_emg[0], t_emg[-1] + 1))
            plt.ylabel('EMG ' + '$\mathrm{(\mu V)}$')
            
        sleepy.box_off(axes_emg)

        plt.setp(axes_spec.get_xticklabels(), visible=False)
        plt.setp(axes_emg.get_xticklabels(), visible=False)

    t = img_time[0:nframes] 

    return Fmean, t


 
def get_xmlfields(ppath, name, fields) :
    """
    get_xmlfields(ppath, name, fields) :\
    get the values of the attributes listed in @fields
    I assume tchat these value are numeric.
    search for lines like
    <attr name=\"frames\">18352</attr>
    """
    fid = open(os.path.join(ppath, name), 'r')    
    lines = fid.readlines()
    fid.close()
    values = []
    for f in fields :
        for l in lines :
            a = re.search(r'<attr name=\"' + f + '\">([\d\.]+)', l)
            if a :
                values.append(float(a.group(1)))

    return values


def get_snr(ipath, name) :
    """
    simple function to get imaging sampling rate
    ipath    -    imaging base folder
    name     -    name of imaging session (folder)
    """
    return get_xmlfields(os.path.join(ipath, name), 'recording_' + name + '.xml', ["fps"])[0]



def get_infofields(ppath, name, fields) :
    """
    folder $ppath contains a .txt file named $name.
    read out the lines starting with the strings
    in list @fields.
    I assume the following syntax: fields[i]:\s+(.*)
    @RETURN:
    The $1 values
    """
    fid = open(os.path.join(ppath, name), 'r')    
    lines = fid.readlines()
    fid.close()
    values = []
    for f in fields :
        for l in lines :
            a = re.search("^" + f + ":" + "\s+(.*)", l)
            if a :
                if re.match('[\d\.]+', a.group(1)) :
                    values.append(float(a.group(1)))
                    
                else :
                    # $1 is a string
                    values.append(a.group(1))
            
    return values
    


def get_infoparam(ppath, name, field) :
    """
    similar to get_infofields;
    NOTE: field is a single string
    and the function does not check for the type
    of the values for field.
    In fact, it just returns the string following field
    """
    fid = open(os.path.join(ppath, name), 'r')    
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines :
        a = re.search("^" + field + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))
            
    return values



def add_infoparam(ifile, field, vals):
    """
    :param ifile: text file, typically info.txt
    :param field: string, parameter name
    :param vals: list of parameters
    """
    fid = open(ifile, 'a')
    vals = [str(s) for s in vals]
    param = " ".join(vals)
    fid.write('%s:\t%s' % (field, param))
    fid.write(os.linesep)
    fid.close()


        
def tiff2h5(ppath, name, nframes, nblock=1000) :
    """
    tiff2h5(ppath, name) :\
    transform tif stack $ppath/$name to h5fd file named as $name,\
    except that it ends with .mat instead of .tif
    """
    fbase = re.split('\.', name)[0]
    stack = TIFF.open(os.path.join(ppath, name))
    (nx, ny) = stack.read_image().shape
    fmat_name = os.path.join(ppath, fbase + '.mat')
    if os.path.isfile(fmat_name) == 1 :
        os.remove(fmat_name)

    print("Creating h5 file %s" % fmat_name)
    fmat = h5py.File(fmat_name, 'w')
    dset = fmat.create_dataset('images', shape=(nframes, nx, ny), dtype='uint16')

    j=0
    k=0
    offset = 0
    D = np.zeros((nblock, nx, ny))
    for i in stack.iter_images() :
        D[j,:,:] = i
        j=j+1
        k=k+1

        if j == nblock :
            dset[offset:offset+j,:,:] = D[0:j,:,:]
            offset = offset + j
            j=0
    dset[offset:offset+j] = D[0:j,:,:]

    stack.close()
    fmat.close()

        

def h52tiff(ppath, name) :
    """
    h52tiff(ppath, name) : \
    convert h5 stack to tiff stack
    """
    fbase = re.split('\.', name)[0]
    
    stack_fid = h5py.File(os.path.join(ppath, name), 'r')
    dset = stack_fid['images']
    (nframes, nx, ny) = dset.shape

    tiff_name = os.path.join(ppath, fbase + '.tif')
    if os.path.isfile(tiff_name) == 1 :
        os.remove(tiff_name)
    tiff = TIFF.open(tiff_name, mode='w')

    for i in range(nframes) :
        tiff.write_image(dset[i,:,:])

    tiff.close()
    stack_fid.close()



def h52tiff_split(ppath, name):
    """
    convert h5 stack to tiff stack.
    tiff stack only allow for a maximum size of 4GB.
    If the stack exceeds this limit, create a new stack.
    File one is name stem.tif, file two is named stem-001.tif etc.
    """
    FLIMIT = 4000000000
    fbase = re.split('\.', name)[0]
    
    stack_fid = h5py.File(os.path.join(ppath, name), 'r')
    dset = stack_fid['images']
    (nframes, nx, ny) = dset.shape

    tiff_name = os.path.join(ppath, fbase + '-001.tif')
    if os.path.isfile(tiff_name) == 1 :
        os.remove(tiff_name)
    tiff = TIFF.open(tiff_name, mode='w')

    fcount = 2
    for i in range(nframes) :
        tiff.write_image(dset[i,:,:])
        if os.path.getsize(tiff_name) > FLIMIT :
            # generate next tiff file
            s = "%03d" % fcount
            tiff.close()
            tiff_name = os.path.join(ppath, fbase + '-' + s + '.tif')
            if os.path.isfile(tiff_name) == 1 :
                os.remove(tiff_name)

            tiff = TIFF.open(tiff_name, mode='w')
            fcount = fcount + 1

    tiff.close()
    stack_fid.close()



def split_hdf5(ppath, name, nblock=10000, dtype='uint8'):
    """
    split hdf5 stack to into several hdf5 stacks.
    The new files are called "re.split('\.', name)[0]" + i + .hdf5,
    with i = -001, -002, ... .

    :param ppath: folder containing hdf5 imaging stack
    :param name: hdf5 imaging stack
    :param nblock: number of frames per hdf5 file
    :param dtype: data type of hdf5 stacks

    """
    fbase = re.split('\.', name)[0]
    
    fid_stack = h5py.File(os.path.join(ppath, name), 'r')
    dset = fid_stack['images']
    (nframes, nx, ny) = dset.shape

    # open first hdf5 file
    split_name = os.path.join(ppath, fbase + '-001.hdf5')
    if os.path.isfile(split_name) == 1 :
        os.remove(split_name)
    fid_split = h5py.File(split_name, 'w')
    
    if nblock>nframes:
        nsize=nframes
    else:
        nsize=nblock
    dset_split = fid_split.create_dataset('images', shape=(nsize, nx, ny), dtype=dtype)

    j=0
    offset=0
    fcount = 1
    D = np.zeros((nblock, nx, ny), dtype=dtype)
    for i in range(nframes):
        D[j,:,:] = (dset[i,:,:])
        j+=1

        if j==nblock:
            print('writing stack %d'%fcount)
            dset_split[0:j,:,:] = D[0:j,:,:]
            offset += j
            fid_split.close()

            # open next data set block
            j=0
            fcount += 1
            s = "%03d" % fcount
            split_name = os.path.join(ppath, fbase + '-' + s + '.hdf5')
            if os.path.isfile(split_name):
                os.remove(split_name)
            fid_split = h5py.File(split_name, 'w')

            if nblock > nframes-offset:
                nsize = nframes-offset
            else:
                nsize = nblock
            dset_split = fid_split.create_dataset('images', shape=(nsize, nx, ny), dtype=dtype)

    dset_split[offset:offset + j] = D[0:j, :, :]

    fid_stack.close()
    fid_split.close()

    return fcount



def combine_hdf5(path, name, nframes=0, dtype='uint8', new_name=''):
    """
    Combine several image stack saved as hdf5 into a single hdf5 file
    The function assumes that the hdf5 stack are names fstem-001.hdf5, fstem-002.hdf5 etc.
    The new stack is called fstem.hdf5

    :param path: folder with imaging session
    :param name: name of the first hdf5 file, e.g. M1_010120n1_downsampcorr-001.tif
    :param nframes: if 0, readout the frame number from info.txt file (parameter FRAMES:)

    :return:
    """
    # get file stem
    stem = re.split('-\d\d\d', name)[0]
    # get all files in $path
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files = [f for f in files if re.match(stem + '\-\d\d\d\.hdf5', f)]
    files.sort()

    if len(new_name) == 0:
        new_name = os.path.join(path, stem + '-test.hdf5')

    if os.path.isfile(new_name):
        os.remove(new_name)

    print("new stack will be called %s" % new_name)
    fid = h5py.File(os.path.join(path, name), 'r')
    dset = fid['images']
    (n, nx, ny) = dset.shape
    fid.close()

    if nframes == 0:
        nframes = int(get_infofields(path, 'info.txt', ['FRAMES'])[0])

    fid = h5py.File(new_name, 'w')
    dset = fid.create_dataset('images', shape=(nframes, nx, ny), dtype=dtype)

    offset=0
    for f in files:
        fid_split = h5py.File(os.path.join(path, f), 'r')
        print("processing stack %s"%f)
        dset_split = fid_split['images']
        nframes_split = dset_split.shape[0]

        dset[offset:offset+nframes_split,:,:] = dset_split[0:nframes_split,:,:]
        offset += nframes_split
        fid_split.close()

    fid.close()



# def h52tiff_annotation_time(ppath, name, sdt=2.5, location='SE') :
#     """
#     ppath     -      path containing tiff stack
#     name      -      name of tiff stack
#     convert h5 stack to tiff stack.
#     tiff stack only allow fr a maximum size of 4GB.
#     If the stack exceeds this limit, create a new stack.
#     File one is name stem.tif, file two is named stem-001.tif etc.
#
#     Additionally, add to each frame the current brain state annotation.
#     """
#     FLIMIT = 4000000000
#     MAX_INT = np.power(2, 15)
#     fbase = re.split('\.', name)[0]
#
#     stack_fid = h5py.File(os.path.join(ppath, name), 'r')
#     dset = stack_fid['images']
#     (nframes, nx, ny) = dset.shape
#
#     tiff_name = os.path.join(ppath, fbase + '.tif')
#     if os.path.isfile(tiff_name) == 1 :
#         os.remove(tiff_name)
#     tiff = TIFF.open(tiff_name, mode='w')
#
#     # Sleep State Data
#     # get imaging time
#     base_path = '/'.join(ppath.split('/')[0:-1])
#     recording = re.search('recording_(\S+_\S+)_', name).group(1)
#     itime = imaging_timing(base_path, recording)
#     M = map(int, load_stateidx(base_path, recording))
#     stime = np.arange(0, len(M))*sdt
#     Letters = [L.astype('uint16') for L in read_letters()]
#     Numbers = [N.astype('uint16') for N in read_numbers()]
#     (lx, ly) = Letters[0].shape
#     (dx, dy) = Numbers[0].shape
#
#     # timing movie frame -> index brain state
#     def closest_time(it, st):
#         return np.argmin(np.abs(it-st))
#
#     fcount = 1
#     for i in range(np.min((len(itime),nframes))) :
#         if i % 1000 == 0 :
#             print("Done with %d of %d frames" % (i, nframes))
#
#         frame = dset[i,:,:]
#
#         # Add R,N,W annotation
#         # get the index of the brainstate index that is closest to the timing
#         # of the current frame
#         # state = M[int(eeg2img_time(itime[i], stime))]
#         state = M[closest_time((itime[i], stime))]
#         MAX_INT = frame.max()
#         # write state onto frame
#         if location == 'SE' :
#             piece = frame[nx-lx:, ny-ly:]
#             piece[np.where(Letters[state-1]==1)] = MAX_INT
#             frame[nx-lx:, ny-ly:] = piece
#         elif location == 'NE' :
#             piece = frame[:lx:, ny-ly:]
#             piece[np.where(Letters[state-1]==1)] = MAX_INT
#             frame[:lx:, ny-ly:] = piece
#         elif location == 'NW' :
#             piece = frame[:lx:, :ly]
#             piece[np.where(Letters[state-1]==1)] = MAX_INT
#             frame[:lx:, :ly] = piece
#         else : # location == 'SW'
#             piece = frame[nx-lx:, :ly]
#             piece[np.where(Letters[state-1]==1)] = MAX_INT
#             frame[nx-lx:, :ly] = piece
#
#         # Add Time,
#         # Could be faster if a preallocated an array for timestamp
#         # instead of concatenating
#         timestamp = np.zeros((dx,1))
#         timestring = str(int(round(itime[i])))
#         for i in range(len(timestring)) :
#             timestamp=np.concatenate((timestamp,Numbers[int(timestring[i])]), axis=1)
#             timestamp=np.concatenate((timestamp,np.zeros((dx,1))), axis=1)
#         # add time to frame
#         timepiece = frame[0:timestamp.shape[0], 0:timestamp.shape[1]]
#         timepiece[np.where(timestamp==1)] = MAX_INT
#         frame[0:timestamp.shape[0], 0:timestamp.shape[1]] = timepiece
#
#         # write frame
#         tiff.write_image(frame)
#
#         # if too large, split TIFF stack
#         if os.path.getsize(tiff_name) > FLIMIT :
#             # generate next tiff file
#             s = "%03d" % fcount
#             tiff.close()
#             tiff_name = os.path.join(ppath, fbase + '-' + s + '.tif')
#             if os.path.isfile(tiff_name) == 1 :
#                 os.remove(tiff_name)
#
#             tiff = TIFF.open(tiff_name, mode='w')
#             fcount = fcount + 1
#
#     tiff.close()
#     stack_fid.close()


    
#
# def read_letters() :
#     """
#     read masks for R, W, N saved in letters.txt
#     """
#     fid = open('letters.txt')
#     lines = fid.readlines()
#     fid.close()
#     L = re.findall("<([\s\S]*?)>",''.join(lines))
#
#     Letters = []
#     for letter in L:
#         A = []
#         lines = letter.split('\n')
#         for l in lines :
#             if not(re.match('^\s*$', l)) :
#                 numbers = l.split(' ')
#                 b = [int(i) for i in numbers if not(i=='')]
#                 A.append(b)
#         Letters.append(np.array(A))
#
#     return Letters
#
#
# def read_numbers() :
#     """
#     read masks for R, W, N saved in letters.txt
#     """
#     fid = open('numbers.txt')
#     lines = fid.readlines()
#     fid.close()
#     L = re.findall("<([\s\S]*?)>",''.join(lines))
#
#     Numbers = []
#     for letter in L:
#         A = []
#         lines = letter.split('\n')
#         for l in lines :
#             if not(re.match('^\s*$', l)) :
#                 numbers = l.split(' ')
#                 b = [int(i) for i in numbers if not(i=='')]
#                 A.append(b)
#         Numbers.append(np.array(A))
#
#     return Numbers


    
def combine_tiffs(path, name, ndown) :
    """
    nframes = combine_tiffs(path, name, ndown) : 
    $path     -     folder containing the tiff file $name 
    $name     -     name of the \"first\" tiff file following 
                    the naming convention:   recording_[\d_]+.tif 
                    the second file is named recording_[\d_]+-001.tif
    $ndown    -     spatial downsampling
    @RETURN:
    $nframes  -     number of frames
    """
    # get file stem
    stem = re.split('\.', name)[0]
    # get all files in $path
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files = [f for f in files if re.match(stem + '\-\d\d\d\.tif', f)]
    files.sort()
    files.insert(0, name)
    new_name = stem + '_downsamp.hdf5'
    print("new stack will be called %s" % new_name)
    #new_stack = TIFF.open(os.path.join(path, new_name), mode='w')

    # start combining tiffs
    nframes = 0
    p = 0
    for f in files :
        if p == 0 :
            new_stack = TIFF.open(os.path.join(path, new_name), mode='w')
            p = p+1
        else :
            new_stack = TIFF.open(os.path.join(path, new_name), mode='a')
        stack = TIFF.open(os.path.join(path, f))
        print("reading stack %s" % f)
    
        for l in stack.iter_images() :
            # spatial downsampling
            # pdb.set_trace()
            if ndown > 1:
                l = downsample_image(l, ndown)
            new_stack.write_image(l)
            nframes = nframes + 1
            if (nframes % 100) == 0 :
                print("Done with frame %d" % nframes)
        # important:
        stack.close()
        new_stack.close()    

    return nframes



def combine_tiffs2h5(path, name, nframes, ndown, nblock=100) :
    """
    combine all tiff files in folder $path to an hdf5 file;
    spatially downsample imaging stack by factor $ndown.
    
    path      -     (absolute) recording folter
    name      -     name of the first tiff file, e.g. recording_V2_100215n1.tif
                    
    nframes   -     total number of frames
    ndown     -     spatial downsampling
    nblock    -     size of blocks that are processed at once.

    NOTE: the output file will be saved in $path/recording_V2_100215n1_downsamp.mat
    """
    # get file stem
    stem = re.split('\.', name)[0]
    # get all files in $path
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files = [f for f in files if re.match(stem + '\-\d\d\d\.tif', f)]
    files.sort()
    files.insert(0, name)
    new_name = stem + '_downsamp.hdf5'
    if os.path.isfile(new_name) == 1 :
        os.remove(new_name)

    print("new stack will be called %s" % new_name)
    #new_stack = TIFF.open(os.path.join(path, new_name), mode='w')
    # get dimensions of images
    tmp_stack = TIFF.open(os.path.join(path, files[0]))
    (nx, ny) = tmp_stack.read_image().shape
    tmp_stack.close()
    # dimensions of downsampled images:
    nx_down = int(np.floor(nx / (ndown*1.0)))
    ny_down = int(np.floor(ny / (ndown*1.0)))
    
    fmat = h5py.File(os.path.join(path,new_name), 'w')
    dset = fmat.create_dataset('images', shape=(nframes, nx_down, ny_down), dtype='uint8')

    # start combining tiffs
    j = 0
    offset = 0
    iframes = 0
    D = np.zeros((nblock, nx_down, ny_down))
    for f in files :
        #open the current tiff stack
        stack = TIFF.open(os.path.join(path, f))
        print("reading stack %s" % f)

        # read frame by frame of the current stack
        for l in stack.iter_images() :
            # spatial downsampling
            if ndown > 1:
                l = downsample_image(l, ndown)

            D[j,:,:] = l
            iframes = iframes+1
            j = j+1
            
            if j==nblock :
                dset[offset:offset+j,:,:] = D[0:j,:,:]
                offset = offset + j
                j=0
                
            if (iframes % 100) == 0 : print("Done with frame %d" % iframes)

        # 6/24/16
        # make sure to not forget the last frame of the current tiff stack
        dset[offset:offset+j,:,:] = D[0:j,:,:]

        # important:
        stack.close()

    fmat.close()    

    return iframes



def avi2h5(rawfolder, ppath, name, ndown=1):
    """
    transform avi movies as recorded by miniscope in folder $rawfolder
    to an hdf5 stack saved in $ppath/$name/recording_$name_downsamp.hdf5

    :param rawfolder: folder containing *.avi files
    :param ndown: spatially downsample each frame by factor $ndown
    
    """
    import cv2
    
    new_name = 'recording_' + name + '_downsamp.hdf5'
    files = os.listdir(os.path.join(rawfolder))
    files = [f for f in files if re.match('^msCam\d+', f)]
    print(files)

    # sort files
    num = {}
    for f in files:
        a = re.findall('Cam(\d+)\.avi$', f)
        num[int(a[0])] = f
        
    keys = list(num.keys())
    keys.sort()
    files_new = []
    for k in keys:
        files_new.append(num[k])
    files = files_new
    
    # count number of frames in each video and total frame number
    nframes = 0
    frames_per_video = {}
    get_dim = True
    for f in files:
        cap = cv2.VideoCapture(os.path.join(rawfolder, f))        
        ret = True
        vframes = 0
        while (ret):
            ret, frame = cap.read()
            
            if get_dim:
                (nx, ny) = (frame.shape[0], frame.shape[1])
                get_dim = False

            if ret:
                nframes += 1
                vframes += 1
        frames_per_video[f] = vframes
        cap.release()
    ####################################################
        
    # create directory for recording
    if not os.path.isdir(os.path.join(ppath, name)):
        os.mkdir(os.path.join(ppath, name))
            
    fmat = h5py.File(os.path.join(ppath, name, new_name), 'w')
    nx_down = int(np.floor(nx / (ndown*1.0)))
    ny_down = int(np.floor(ny / (ndown*1.0)))
    A = np.zeros((nx_down, ny_down))
    # images are unit8; that's how they're returned from cv2
    dset = fmat.create_dataset('images', shape=(nframes, nx_down, ny_down), dtype='uint8')
    
    offset = 0
    for f in files:
        #cap = cv2.VideoCapture(os.path.join(rawfolder, f))
        #ret = True
        print("converting video %s" % f)

        D = np.zeros((frames_per_video[f], nx_down, ny_down))
        cap = cv2.VideoCapture(os.path.join(rawfolder, f))                    
        for j in range(frames_per_video[f]):
            ret,frame = cap.read()                
            #frame = frame.sum(axis=2) / 3
            frame = frame[:,:,0]
            if ndown > 1:
                frame = downsample_image(frame, ndown)
            D[j,:,:] = frame
            A += frame

        dset[offset:offset+j+1] = D
        offset = offset+j+1
        cap.release()
    fmat.close()
    
    A = A / (1.0 * nframes)
    so.savemat(os.path.join(ppath, name, 'recording_' + name + '_mean.mat'), {'mean': A})

    # copy file timestamp.dat and settings_and_notes.txt from rawdata to $ppath/$name
    shutil.copyfile(os.path.join(rawfolder, 'timestamp.dat'), os.path.join(ppath, name))
    shutil.copyfile(os.path.join(rawfolder, 'settings_and_notes.dat'), os.path.join(ppath, name))



def downsample_hdf5(path, name, ndown, nblock=1000, psave_mean=1) :
    """
    path      -     (absolute) recording folter
    name      -     name of hdf5 file
                    
    nframes   -     total number of frames
    ndown     -     spatial downsampling
    nblock    -     size of blocks that are processed at once.


    NOTE: the output file will be saved in $path/recording_V2_100215n1_downsamp.mat
    """
    
    # get file stem
    stem = re.split('\.', name)[0]
    # get all files in $path
    f = os.path.join(path, name)

    new_name = stem + '_downsamp.hdf5'
    if os.path.isfile(new_name) == 1 :
        os.remove(new_name)

    print("new stack will be called %s" % new_name)

    # get dimensions of images
    fid = h5py.File(f, 'r')
    stack = fid['images']
    (nframes, nx, ny) = stack.shape
    # dimensions of downsampled images:
    nx_down = int(np.floor(nx / (ndown*1.0)))
    ny_down = int(np.floor(ny / (ndown*1.0)))
    
    fmat = h5py.File(os.path.join(path,new_name), 'w')
    dset = fmat.create_dataset('images', shape=(nframes, nx_down, ny_down), dtype='uint16')

    # start combining tiffs
    j = 0 # where we are in the current block
    offset = 0
    iframes = 0
    D = np.zeros((nblock, nx_down, ny_down))
    A = np.zeros((nx_down, ny_down))
    
    # converting from stack ==> dset
    # blockwise processing of hdf5 stack:
    for i in range(nframes) :
        image = stack[i,:,:]
        # spatial downsampling
        if ndown > 1:
            image = downsample_image(image, ndown)
        D[j,:,:] = image
        j = j+1

        # write full block
        if j==nblock :
            dset[offset:offset+j,:,:] = D[0:nblock,:,:]
            offset = offset + nblock
            A[:,:] = A[:,:] + D.sum(axis=0)
            j=0
        if (i % 1000) == 0 : print("Done with frame %d" % i)

    # make sure to not forget the last frame of the current tiff stack
    if j>0:
        dset[offset:offset+j,:,:] = D[0:j,:,:]

    # save the average frame
    A = A / (nframes*1.0)
    if psave_mean == 1:
        so.savemat(os.path.join(path, stem + '_mean.mat'), {'mean' : A})
    
    fid.close()
    fmat.close()    
    
    return iframes



def downsample_image(img, ndown, dtype='uint16') :
    """
    downsample 2D array by calculating the mean of of $ndown consecutive elements.
    @img      -      2D array
    $ndown    -      downsampling factor in x and y direction
    [$dtype]  -      data typ of downsampled image, default: uint16
    @RETURN:
    dowsampled image
    @NOTE:
    if (length of x or y direction % ndown) != 0,
    the last bin (with length < $ndown) is neglected.
    """
    (nx, ny) = img.shape
    nx_down = int(np.floor(nx / (ndown*1.0)))
    ny_down = int(np.floor(ny / (ndown*1.0)))
    nx_adj = nx_down*ndown
    ny_adj = ny_down*ndown
    A = np.zeros((nx_down, ny_down))
    
    for i in range(ndown) :
        A = A + img[i:nx_adj:ndown, i:ny_adj:ndown]
    A = A / (ndown*1.0)
    
    return A.astype(dtype)



def roi_manual(ppath, name, pdisk=1) :
    """
    roi_list = roi_manual(ppath, name) : \
    manual roi selection
    """
    fstem = '_'.join(re.split('_', name)[0:-1])
    if pdisk == 1:
        img = so.loadmat(os.path.join(ppath, fstem + '_diskmean.mat'))['mean']
    else :
        img = so.loadmat(os.path.join(ppath, fstem + '_downsampmean.mat'))['mean']
        
    roi_list = []
    pcontinue = 1
    iid = 0
    while(pcontinue) :
        plt.imshow(img, cmap='gray')
        [x.displayROI() for x in roi_list]
        r = roipoly(roicolor='r')
        roi_list.append(r)

        plt.imshow(img, cmap='gray')
        [x.displayROI() for x in roi_list]
        [x.displayID (img, i) for (x,i) in zip(roi_list, range(0, len(roi_list)))]
        plt.show(block=False)
        iid = iid+1
        s = input('continue selecting rois?\n0 - no\n1 - yes\n')
        plt.close()
        if s == 0 :
            pcontinue = 0

    idx_list = [np.nonzero(r.getMask(img)==True) for r in roi_list]
    bnd_list = [(r.allxpoints, r.allypoints) for r in roi_list]
    return (idx_list, bnd_list)


###################################################################################
### MOTION CORRECTION #############################################################
###################################################################################

def disk_filter(ppath, name, nframes, pfilt=1, nblock=1000, psave=1) :
    """A = disk_filter(ppath, name, nframes pfilt=1, nblock=1000)
    $ppath/$name refers to a TIFF stack.
    """
    print("performing disk filtering ...")
    # load disk filter
    disk = so.loadmat('disk.mat')['h']
    stack = TIFF.open(os.path.join(ppath, name))
    (nx, ny) = stack.read_image().shape
    
    # h5f5 file holding the filtered stack:
    a = re.split('_', name)[0:-1]
    disk_fname = '_'.join(a) + '_disk.mat'
    disk_meanfname = '_'.join(a) + '_diskmean.mat'
    
    h5_file = os.path.join(ppath, disk_fname)
    
    if os.path.isfile(h5_file) == 1 :
        # delete file
        print("deleting file %s..." % h5_file)
        os.remove(h5_file)

    f = h5py.File(h5_file, 'w')
    # reference to data set holding all frames
    dset = f.create_dataset('images', (nframes, nx, ny))

    D = np.zeros((nblock, nx, ny))
    A = np.zeros((nx, ny)) # average across frames
    offset = 0
    k = 0  #absolute counter for all frames in the tiff stack
    if pfilt == 1 :
        j = 0   #relative counter for each block
        for image in stack.iter_images() :
            tmp = image - scipy.signal.fftconvolve(image.astype('float64'), disk, mode='same')
            D[j,:,:] = tmp
            A = A + tmp
            j = j+1
            k = k+1

            if j == nblock or k == nframes:
                print("writing block...")
                # write to mat file
                dset[offset:offset+j,:,:] = D[0:j,:,:]
                offset = offset + j
                j = 0
    else :
        j = 0
        for image in stack.iter_images() :
             dset[j,:,:] = image
             j = j+1
             
    # make sure to close all files
    stack.close()
    f.close()

    A = A / (nframes*1.0)

    if psave == 1:
        print("saving mean frame to %s" % disk_meanfname)
        so.savemat(os.path.join(ppath, disk_meanfname), {'mean' : A})
    
    return A



def disk_filter_h5(ppath, name, pfilt=1, nblock=1000, psave=1, nw=2):
    """A = disk_filter(ppath, name, pfilt=1, nblock=1000)
    $ppath/$name refers to a H5DF stack.
    performs disk filtering on a h5df stack

    NOTE: The output file will be named ppath/recording_V2_100215n1_disk.hdf5;
    The resulting file (float64) is much larger than the original stack (uint8),
    but it's safe to delete the _disk.hdf5 file at any time, as it can be regenerated.
    Naming: name refers to the mat stack that is to be filtered
    the name of the output file includes everything up to the last '_'
    after the last '_' 'disk.mat' is appended to the new file name
    
    """
    print("performing disk filtering ...")
    # load disk filter
    disk = so.loadmat('disk.mat')['h']
    fid = h5py.File(os.path.join(ppath, name), 'r')
    stack = fid['images']
    (nframes, nx, ny) = stack.shape
    
    # h5f5 file holding the filtered stack:
    a = re.split('_', name)[0:-1]
    disk_fname = '_'.join(a) + '_disk.hdf5'
    disk_meanfname = '_'.join(a) + '_diskmean.mat'
    h5_file = os.path.join(ppath, disk_fname)
    
    if os.path.isfile(h5_file) == 1:
        # delete file
        print("deleting file %s..." % h5_file)
        os.remove(h5_file)

    f = h5py.File(h5_file, 'w')
    # reference to data set holding all frames
    dset = f.create_dataset('images', (nframes, nx, ny))

    filt = 1.0*np.ones((nw,nw)) / (nw*nw)
    D = np.zeros((nblock, nx, ny))
    A = np.zeros((nx, ny)) # average across frames
    offset = 0
    k = 0  #absolute counter for all frames in the tiff stack
    if pfilt == 1 :
        j = 0   #relative counter for each block
        for index in range(0, nframes) :
            image = stack[index,:,:]
            if nw > 1:
                # would float32 also work?
                image = scipy.signal.fftconvolve(image.astype('float64'), filt, mode='same')
            tmp = image - scipy.signal.fftconvolve(image.astype('float64'), disk, mode='same')
            D[j,:,:] = tmp
            A = A + tmp
            j = j+1
            k = k+1

            if j == nblock or k == nframes :
                print("writing frame %d of %d frames" % (k, nframes))
                # write to mat file
                dset[offset:offset+j,:,:] = D[0:j,:,:]
                offset = offset + j
                j = 0
             
    # make sure to close all files
    fid.close()
    f.close()

    A = A / (nframes*1.0)

    if psave == 1:
        print("saving mean frame to %s" % disk_meanfname)
        so.savemat(os.path.join(ppath, disk_meanfname), {'mean' : A})
    
    return A



def activity_map(ppath, name, nw=2):
    """
    calculate activity map of hdf5 image stack. 
    
    :param ppath: base folder containing all hdf5 and mat files. 
    :param name: name of the hdf5 movie stack.
                 ppath/name refers to a H5DF stack.
    
    Naming: the name of the output file includes everything up to the last '_'
    after the last '_' 'actmap.mat' is appended to save the 2D matrix
    representing the activity map.
    
    Note: The function expects that the file *_alignedmean.mat exists. 
    The file is generated by the function align_frames()
    
    """

    a = re.split('_', name)[0:-1]
    fstem = '_'.join(a)
    actmap_fname = fstem + '_actmap.mat'
    mean_fname   = fstem + '_alignedmean.mat'

    # load stack
    fid = h5py.File(os.path.join(ppath, name), 'r')
    stack = fid['images']
    (nframes, nx, ny) = stack.shape

    # load meanframe
    mean_frame = so.loadmat(os.path.join(ppath, mean_fname))['mean']
    mean_flu = np.mean(mean_frame)
    filt = 1.0*np.ones((nw, nw)) / (nw*nw)

    A = np.zeros((nx, ny)) # average across frames
    for index in range(0, nframes):
        image = stack[index,:,:]
        proc_image = (image - mean_frame) / (mean_frame + mean_flu)
        proc_image = scipy.signal.fftconvolve(proc_image.astype('float64'), filt, mode='same')
        proc_image = np.power(proc_image, 3)

        A = A + proc_image

    A = A / (nframes * 1.0)
    so.savemat(os.path.join(ppath, actmap_fname), {'mean' : A})
    # make sure to close all files
    fid.close()



# 2D convolution using the convolution's FFT property
def conv2(a,b):
    ma,na = a.shape
    mb,nb = b.shape
    return np.fft.ifft2(np.fft.fft2(a,[2*ma-1,2*na-1])*np.fft.fft2(b,[2*mb-1,2*nb-1]))


# compute a normalized 2D cross correlation using convolutions
# this will give the same output as matlab, albeit in row-major order
def normxcorr2(b,a):
    """
    normxcorr2(b,a)
    @a     -     template
    @b     -     image to filter
    implemented formula:
    a * b / sqrt( var(a) var(t * 1) )
    """
    a = a.copy()
    b = b.copy()
    a = a-a.mean()
    b = b-b.mean()
    (nx, ny) = a.shape

    c = scipy.signal.fftconvolve(a,np.flipud(np.fliplr(b)))
    #c = scipy.signal.fftconvolve(a,b)
    a = scipy.signal.fftconvolve(np.power(a,2), np.ones(b.shape)) # this factor depends on correlation shift (u,v)
    
    b = np.power(b,2).sum() # this factor is independent of the correlation shift (u,v)
    c = c/np.sqrt(a*b)
    c = np.abs(c)
    (sx, sy) = np.unravel_index(c.argmax(), c.shape)

    sx = sx-nx+1
    sy = sy-ny+1
    return (-sx,-sy)




### DEPRICATED ##########################################  
def phase_correlation(a, b) :
    """
    x,y = phase_correlation(A,B)

    if x > 0: shift A sx rows down,            i.e. A[sx:,:] == B[:b.shape[0]-sx,:]
    if y > 0: shift A sy columns to the right, i.e. A[:,sy:] == B[:,:B.shape[1]-sy]
    if x < 0: shift B sx rows down,            i.e. B[sx:,:] == A[:B.shape[0]-sx,:]
    if y < 0: shift B sy columns to the right, i.e. B[:,sy:] == A[:,:A.shape[1]-sy]
    """
    a = a.copy()
    b = b.copy()
    #a = a-np.mean(a)
    #b = b-np.mean(b)
    G_a = np.fft.fft2(a)
    G_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a * conj_b
    R /= np.absolute(R)
    r = np.fft.ifft2(R).real

    (nx, ny) = a.shape
    (sx, sy) = np.unravel_index(r.argmax(), r.shape)
    
    if (nx-sx) < sx :
        sx = -(nx-sx)
    if (ny-sy) < sy :
        sy = -(ny-sy)
    
    return (sx, sy)
##########################################################


def get_corr_ranges(nx, ny, sx, sy) :
    """
    get_corr_ranges(nx, ny, sx, sy) :
    helper function for output of phase_correlation
    A[arangex, arangey] == B[brangex, brangey]
    """

    if sx >= 0 :
        arangex = np.arange(sx,nx)
        brangex = np.arange(0,nx-sx)
    if sx < 0 :
        sx = -sx
        arangex = np.arange(0,nx-sx)
        brangex = np.arange(sx,nx)
    if sy >= 0 :
        arangey = np.arange(sy,ny)
        brangey = np.arange(0,ny-sy)
    if sy < 0 :
        sy = -sy
        arangey = np.arange(0,ny-sy)
        brangey = np.arange(sy, ny)

    return arangex, arangey, brangex, brangey



def align_frames(ppath, name, nframes=-1, nblock=1000, pdisk=1, psave=1, pwrite=1, pcorr_time=True):
    """
    Shift = align_frames(ppath, name, nblock=1000): 
    motion correlction by phase correlation
    This function assumes that the following files exist:
    o name-stem_diskmean.mat  | name-stem_mean.mat
    o name-stem_downsamp.hdf5 | name-stem_disk.hdf5
    o name-stem_downsamp.hdf5
    
    The function creates a new hdf5 stack called
    o name-stem_aligned.hdf5

    @ARGS:
    ppath    -    absolute path of imaging session
    name     -    name of image stack to be motion corrected
    nblock   -    number of frames processed / iteration
    pdisk    -    perform motion correction using disk filtered image stack
    psave    -    if psave==1: save aligned mean frame and Shift vector
    pwrite   -    if pwrite==1: write frames to hdf5 data set called name-stem_aligned.hdf5\
                  otherwise just return Shift vector specifying displacement in x and y direction
    pcorr_time -  use stack the stack name-stem_downsampcorr.hdf5, where redundant frames were kicked out
    @RETURN:
    Shift    -    2D vector specifying correction in x and y direction
    """
    plt.ioff()
    fstem = '_'.join(re.split('_', name)[0:-1])
    if pdisk == 1 :
        (idx, ref) = align_stencil(ppath, name, pdisk=1)
    else :
        (idx, ref) = align_stencil(ppath, name, pdisk=0)

    #plt.imshow(ref, cmap='gray')
    
    # original downsampled stack
    if not pcorr_time:
        ostack = TIFFStack(ppath, fstem + '_downsamp.hdf5', nblock=1000)
    else:
        ostack = TIFFStack(ppath, fstem + '_downsampcorr.hdf5', nblock=1000)
    if pdisk == 1 :
        # disk (high-pass filtered) stack
        dstack = TIFFStack(ppath, fstem + '_disk.hdf5', nblock=1000)
    else :
        dstack = ostack

    (nx, ny) = (dstack.nx, dstack.ny)
    if nframes < 0 :
        nframes = dstack.nframes

    # Image shift
    Shift = np.zeros((nframes, 2))

    # Stack holding aligned frames
    aligned_fname = os.path.join(ppath, fstem + '_aligned.hdf5')
    if pwrite == 1 and os.path.isfile(aligned_fname) == 1:
        os.remove(aligned_fname)
    if pwrite == 1 :
        astack = h5py.File(aligned_fname, 'w')
        aset = astack.create_dataset('images', shape=(nframes, nx, ny), dtype='uint8')
        # aligned mean frame
        amean = np.zeros((nx, ny))
    
    # block-wise correlation
    k = 0 # absolute count of frames
    niter = int(np.ceil(nframes / (1.0*nblock)))
    for i in range(niter) :
        nup = np.min(((i+1)*nblock, nframes))
        # Original Stack
        C = ostack.dset[i*nblock:nup,:,:]
        # Filtered Stack
        D = dstack.dset[i*nblock:nup,:,:]
        # Motion Corrected Frames
        N = np.zeros((nup-(i*nblock), nx, ny))

        for j in range(D.shape[0]) :
            (sx, sy) = normxcorr2(ref, D[j, idx[0]:idx[1], idx[2]:idx[3]])
            if np.abs(sx)>20:
                sx = 0
            if np.abs(sy)>20:
                sy = 0
             
            (arangex, arangey, brangex, brangey) = get_corr_ranges(nx, ny, sx, sy)
            if pwrite == 1:
                N[j,arangex[0]:arangex[-1]+1,arangey[0]:arangey[-1]+1] = C[j,brangex[0]:brangex[-1]+1,brangey[0]:brangey[-1]+1]
            Shift[k,0] = sx
            Shift[k,1] = sy
            # frame count:
            k = k+1
    
        if pwrite == 1:
            aset[i*nblock:nup,:,:] = N
            amean = amean + np.sum(N, axis=0)
        if (k % 1000) == 0 :
            print("Done with frame %d of %d frames" % (k, nframes))
        # end of i iteration

    # correction done
    if pwrite == 1:
        amean = amean / (nframes * 1.0)

    # Closing all files and save aligned mean frame
    if pwrite == 1:
        astack.close()
    ostack.close()
    if pdisk == 1: dstack.close()

    if (pwrite == 1) and (psave == 1):
        mean_fname = os.path.join(ppath, fstem + '_alignedmean.mat')
        print("saving mean aligned frame to %s" % mean_fname)
        so.savemat(mean_fname, {'mean' : amean, 'shift' : Shift})

    return Shift



def align_stencil(ppath, name, pdisk=1, psquare=1, psave=1):
    """
    (idx_list, Template) = align_stencil(ppath, name, pdisk=1, psquare=1, psave=1) 
    manually cut out a high contrast piece from the imaging stack used
    for motion correction.

    NOT YET WORKING UNDER PY37

    @RETURN:
    idx_list    -      coordinates specifying the upper left and lower right corner (x1, x2, y1, y2)
                       of a square
    """
    plt.ioff()
    fstem = '_'.join(re.split('_', name)[0:-1])
    if pdisk == 1 :
        img = so.loadmat(os.path.join(ppath, fstem + '_diskmean.mat'))['mean']
    else :
        img = so.loadmat(os.path.join(ppath, fstem + '_mean.mat'))['mean']
    new_name = os.path.join(ppath, fstem + '_stencil.mat')

    plt.figure()
    plt.imshow(img, cmap='gray')
    r = roipoly(roicolor='r')
    idx_list = np.nonzero(r.getMask(img)==True)

    if psquare == 1 :
        (x1, x2) = (np.min(idx_list[0]), np.max(idx_list[0])+1)
        (y1, y2) = (np.min(idx_list[1]), np.max(idx_list[1])+1) 
        A = img[x1:x2, y1:y2]
        A = A-A.mean()

        idx_list = (x1, x2, y1, y2)
    else :
        A = np.zeros(img.shape)
        mean = img[idx_list[0], idx_list[1]].mean()
        A[idx_list[0], idx_list[1]] = img[idx_list[0], idx_list[1]]-mean

    if psave :
        so.savemat(os.path.join(ppath, new_name), {'mean' : A, 'idx' : idx_list})

    return (idx_list, A)


def crop_dataset(ppath, name, nblock=1000):
    """
    crop_dataset(ppath, name)
    draw a square on the average image used to crop the stack
    the cropped image stack will be saved to 'name-stem_cropped.hdf5'
    """
    fstem = '_'.join(re.split('_', name)[0:-1])
    name_cropped = fstem + '_cropped.hdf5'
    # original stack:
    ostack = TIFFStack(ppath, fstem + '_downsamp.hdf5', nblock=1000)
    if not(os.path.exists(os.path.join(ppath, fstem + '_mean.mat'))) :
        print("Calculating mean for stack %s" % name)
        avg = ostack.mean()
        ostack.save_mean()
        mean_name = os.path.join(ppath, fstem + '_mean.mat')
        so.savemat(os.path.join(ppath, mean_name), {'mean' : avg})
        
    idx = align_stencil(ppath, name, pdisk=0, psquare=1, psave=0)[0]
    
    nframes = ostack.nframes
    nx = idx[1]-idx[0]
    ny = idx[3]-idx[2]
    # cropped stack:
    cstack = h5py.File(os.path.join(ppath, name_cropped), 'w')
    cset = cstack.create_dataset('images', shape=(nframes, nx, ny), dtype='uint16')

    print("Cropping stack %s)" % name)
    print("New stack will be saved to %s" % name_cropped)
    niter = int(np.ceil(nframes / (1.0*nblock)))
    for i in range(niter) :
        nup = np.min(((i+1)*nblock, nframes))
        cset[i*nblock:nup,:,:] = ostack.dset[i*nblock:nup,idx[0]:idx[1], idx[2]:idx[3]]
    ostack.close()
    cstack.close()


# neuropilSubt -> extractBackgroundPxls
def halo_subt(roi_list, rng, nx, ny, zonez=2) :
    """
    (Bgk, Halo) = halo_subt(roi_list, rng, nx, ny) :
    rng   -    range of pixels that we go maximally to the left/right | up/down 
               to calculate the halo for neuropil subtraction 
    nx,ny -    dimensions of image frame
    zonez -    'zone zero', zone between ROI and Halo that is excluded (as ROI or Halo)
    """
    rad = []
    urange = []
    vrange = []
    for roi in roi_list :
        u = ( np.max([0, np.min(roi[0])-rng]), np.min([nx, np.max(roi[0])+rng]) )
        v = ( np.max([0, np.min(roi[1])-rng]), np.min([ny, np.max(roi[1])+rng]) )
        rad.append( np.max( (np.diff(u), np.diff(v)) ) )
        urange.append( u )
        vrange.append( v )

    (Bkg, Halo) = ([], [])
    # calclate "background" for each roi
    for roi in roi_list :
        # get for current roi all potential background pixels
        x_ctr = np.mean(roi[0])
        y_ctr = np.mean(roi[1])
        (x1, x2) = ( np.max([0, np.min(roi[0])-rng]), np.min([nx, np.max(roi[0])+rng+1]) )
        (y1, y2) = ( np.max([0, np.min(roi[1])-rng]), np.min([ny, np.max(roi[1])+rng+1]) )
        rad = np.max( (x2-x1, y2-y1) ) / 2.0
        #(X, Y) = ( np.arange(x1, x2), np.arange(y1, y2) )
        X = np.array(())
        Y = np.array(())
        len_x = x2-x1
        len_y = y2-y1
        for y in np.arange(y1,y2) :
            Y = np.append(Y, np.ones((len_x,))*y)
        for i in np.arange(len_y) :
            X = np.append(X, np.arange(x1,x2))
        

        # Check if background pixels fall into a circle around the center of the ROI
        z = np.sqrt( np.power(X - x_ctr, 2) + np.power( Y - y_ctr, 2) )
        idx = np.where(z <= rad)[0]
        X_bkg = X[idx]
        Y_bkg = Y[idx]
        Bkg.append( (X_bkg.astype('int'), Y_bkg.astype('int')) )

        # Cut out the center of each circle, that gives you then the "Halo"
        # (1) get rid of any pixel within the ROI
        nidx = []
        for soi in roi_list :
            for (i,j) in zip(soi[0], soi[1]) :
                k = np.where((X_bkg == i) & (Y_bkg == j))[0]
                if k: nidx.append(k[0])

        X_bkg = np.delete(X_bkg, nidx)
        Y_bkg = np.delete(Y_bkg, nidx)
        
        # (2) cut of pixels neighboring the ROI
        nidx = []
        for (ri, rj) in zip(roi[0], roi[1]) :
            for i in range(-zonez,zonez+1) :
                for j in range(-zonez,zonez+1) :
                    k = np.where((X_bkg == ri+i) & (Y_bkg == rj+j))[0]
                    if k: nidx.append(k[0])
        
        X_bkg = np.delete(X_bkg, nidx)
        Y_bkg = np.delete(Y_bkg, nidx)

        X_bkg = X_bkg.astype('int')
        Y_bkg = Y_bkg.astype('int')
        Halo.append( (X_bkg, Y_bkg) )

    return (Bkg, Halo)



def correction_factor(ipath, name, psave=1) :
    """
    cf = correction_factor(ipath, name[,pave=1])
    once called the program will load the average frame of the stack (_alignedmean.mat)
    left-click to draw an outline of a polygon; double click to finish the polygon. 
    """
    fname = os.path.join(ipath, name, 'recording_' + name + '_alignedmean.mat')
    img = so.loadmat(fname)['mean']

    print("Select OFF-Field\nLeft-click to select corners of a polygon;\ndouble click once you are happy.")
    plt.imshow(img, cmap='gray')
    r = roipoly(roicolor='r')
    off_idx = np.nonzero(r.getMask(img)==True) 
    offset = img[off_idx].mean()
    
    print("Select Blood Vessel")
    plt.imshow(img, cmap='gray')
    r = roipoly(roicolor='r')
    blood_idx = np.nonzero(r.getMask(img)==True) 
    Fbv = img[blood_idx].mean()
    
    print("Select Blood Vessel Vicinity")
    plt.imshow(img, cmap='gray')
    r = roipoly(roicolor='r')
    vic_idx = np.nonzero(r.getMask(img)==True) 
    Fbv_v = img[vic_idx].mean()
    
    # (Blood-vessel - offset) / (Blood-vessel-vicinity - offset) 
    cf = (Fbv-offset)/(Fbv_v-offset)

    outname = os.path.join(ipath, name, name + '_cf.mat')
    so.savemat(outname, {'cf': cf})
    
    return cf


##############################################################################
### INTERFACING SIMA #########################################################
##############################################################################
def sima_motion_correction(path, name, method='pt2d', max_displace=[30,30]) :
    """
    sima_motion_correction(path, name)
    @ARGS:
    path    -    path where imaging stack is located
    name    -    name of the imaging stack file
    method  -    string specifying the method to be used for motion correction
    """
    import sima
    import sima.misc
    import sima.motion
    from sima import Sequence

    fstem = '_'.join(re.split('_', name)[0:-1])
    # absolute file name of imaging stacK:
    ifile = os.path.join(os.path.join(path, name))
    sima_path = os.path.join(path, fstem + '.sima')
    if os.path.exists(sima_path) :
        shutil.rmtree(sima_path)
    print(sima_path)
    
    seq = [Sequence.create('HDF5', ifile, 'txy')]
    # SIMA sequence arranged as follows:
    # time, plane, y, x, channel
    # in my case, plane and channel are always 1 

    #initialize imaging data set
    dataset = sima.ImagingDataset(seq, sima_path)
    
    
    if method == 'pt2d' :
        mc_approach = sima.motion.PlaneTranslation2D(max_displacement=max_displace)
        print("starting motion correction ...")

        # dataset = mc_approach.correct(seq, sima_path)
        #print "exporting data ..."
        # dataset.export_frames([os.path.join(path, fstem + '_aligned-sima.hdf5')], fmt='HDF5')
        #transform ti my hdf5 format:
        # sima_to_myformat(path, fstem + '_aligned-sima.hdf5', fstem + '_mc.hdf5')
    if method == 'hm' :
        mc_approach = sima.motion.HiddenMarkov2D(granularity=1)

    if method == 'fd' :
        mc_approach = sima.motion.DiscreteFourier2D(verbose=True)
        
    shift = mc_approach.estimate(dataset)
    # I have typically just one data set:
    return shift[0][:,0,:]



def sima_to_myformat(path, file, new_file, nblock=1000) :
    fid = h5py.File(os.path.join(path, file), 'r')
    dset = fid['imaging']
    nframes = dset.shape[0]
    nx = dset.shape[2]
    ny = dset.shape[3]
    fid_new = h5py.File(os.path.join(path, new_file), 'w')
    dset_new = fid_new.create_dataset('images', shape=(nframes, nx, ny), dtype='uint16')

    niter = int(np.ceil(nframes/(1.0*nblock)))
    for i in range(niter) :
        nup = np.min(((i+1)*nblock, nframes))
        dset_new[i*nblock:nup,:,:] = dset[i*nblock:nup,0,:,:,0]

    fid.close()
    fid_new.close()
        


###############################################################################
### FUNCTIONS RELATED TO EEG/SLEEP STATE DATA #################################
###############################################################################
def imaging_timing(ipath, name, SR=1000):
    """
    imaging_timing(ipath, name, SR=1000)
    SR    -    sampling rate of EEG data
    get from the intan AUX.mat file the timing of each frame
    @RETURN:
    absolute time point of each frame
    """
    if os.path.isfile(os.path.join(ipath, name, 'AUX.mat')):
        A = so.loadmat(os.path.join(ipath, name, 'AUX.mat'))['AUX']
        B = np.array([a[0] for a in A]).astype('int64')
        # that I have to figure out: something with types wrong
        idx = np.where(np.diff(B)>0.2)[0]+1
        idx = np.append([0], idx)
        img_time = idx * (1.0/SR)
    else:
        # Justin's version:
        #d = np.loadtxt(os.path.join(ipath, name,'timestamp.dat'),
        #                    delimiter='\t',
        #                    skiprows=1,
        #                    dtype={'names': ('camNum', 'frameNum', 'sysClock', 'buffer'),
        #                    'formats': ('float', 'float', 'float', 'float')})
        #img_time = d['sysClock']/SR
        # time first time point is trash
        #img_time = img_time[1:]

        img_time = so.loadmat(os.path.join(ipath, name, 'img_time.mat'), squeeze_me=True)['time']

    return img_time



def minisc_timing(ipath, name):
    """
    throw out redundant imaging frames
    """
    
    # load timestamp file
    d = np.loadtxt(os.path.join(ipath, name,'timestamp.dat'),
                            delimiter='\t',
                            skiprows=1,
                            dtype={'names': ('camNum', 'frameNum', 'sysClock', 'buffer'),
                            'formats': ('float', 'float', 'float', 'float')})
    img_time = d['sysClock']/1000
    img_time[0] = 0
    nframes = int(d['frameNum'][-1])

    # load stack
    fid = h5py.File(os.path.join(ipath, name, 'recording_%s_downsamp.hdf5'%name), 'r')
    stack = fid['images']
    nx = stack.shape[1]
    ny = stack.shape[2]

    fname_stack_new = os.path.join(ipath, name, 'recording_%s_downsampcorr.hdf5'%name)
    fid_new = h5py.File(fname_stack_new, 'w')

    unique_imgs = [0]
    nframes_new = 1
    red_frame = 0
    i = 0
    for index in range(1, nframes):
        print(index)
        image1 = stack[index-1,:,:]
        image2 = stack[index,:,:]

        if not(np.array_equal(image1, image2)):
            unique_imgs.append(index)
            nframes_new += 1
        else:
            red_frame += 1

    print('%d frames were dropped' % red_frame)

    stack_new = fid_new.create_dataset('images', shape=(nframes_new, nx, ny), dtype='uint8')
    stack_new[0,:,:] = stack[0,:,:]
    i=0
    for index in unique_imgs:
        stack_new[i,:,:] = stack[index,:,:]
        i+=1
    
    corr_img_time = img_time[np.array(unique_imgs)]
    so.savemat(os.path.join(ipath, name, 'img_time.mat'), {'time' : corr_img_time})

    fid.close()
    fid_new.close()
    # add nframes to info file
    add_infoparam(os.path.join(ipath, name, 'info.txt'), 'FRAMES', [nframes_new])



# TIME(EEG) -> TIME(IMAGING)
def eeg2img_time(t_eeg, itime) :
    """
    eeg2img_time(t_eeg, itime)
    @t_eeg     -    array of time points
    @itime     -    array with time points of all imaging frames
    you have a certain time point in the eeg;
    now you want to have the index of the imaging frame
    that is closest to this time point.
    @RETURN
    indices of the imaging frames that are closest to given time points 
    in @t_eeg
    """
    idx = [np.argmin(np.abs(itime-t)) for t in t_eeg]
    return idx


def frame_number(ipath, name) :
    """
    read the number of frames from the info file
    in recording ipath/name
    """
    ddir = os.path.join(ipath, name)
    nframes = int(get_infofields(ddir, 'info.txt', ['FRAMES'])[0])
    return nframes


##########################################################
### EXPORT FUNCTIONS: HDF5 -> MATLAB #####################
##########################################################
def export_stack2mat(path, name, istart, iend, id=1):
    """
    write frames istart to iend-1 into a matlab file, called
    name-stem_export.mat
    """
    fstem = '_'.join(re.split('_', name)[0:-1])
    exp_name = os.path.join(path, fstem + '_export' + str(id) + '.mat')
    ostack = TIFFStack(path, name)
    nx = ostack.nx
    ny = ostack.ny
    D = ostack.dset[istart:iend,:,:]
    D = np.zeros((nx, ny, iend-istart))
    j=0
    for i in range(istart, iend) :
        D[:,:,j] = ostack.dset[i,:,:]
        j=j+1
    so.savemat(os.path.join(path, exp_name), {'stack' : D, 'istart' : istart, 'iend' : iend})    

    ostack.close()



def caim_snip(ppath, recording, frame_thr=1.0):
    """
    The camera strobe signal is saved in laser_$recording.mat. Each "jump" of the signal
    corresponds to a frame flip.


    :param frame_thr: float, a "frame" lasting longer than frame_thr [s], is considered as
                      camera turned off
    """
    # to use this function, first remove all the prexisting EEG and EMG spectrogram sp.mat files, remidx text, and EEG2 and EMG2
    if os.path.isfile(os.path.join(ppath, recording, 'laser_orig_%s.mat' % recording)):
        laser = so.loadmat(os.path.join(ppath, recording, 'laser_orig_' + recording + '.mat'))['laser'][0]
    else:
        laser = so.loadmat(os.path.join(ppath, recording, 'laser_' + recording + '.mat'))['laser'][0]
    signal_onoff = laser.copy()

    # OLD
    # sometimes the camera signal starts with 1s
    #    if signal_onoff[0] == 1:
    #        cam_on = np.where(np.diff(signal_onoff)==-1)[0] + 1
    #        cam_on = cam_on[0]
    #        signal_onoff[0:cam_on] = 0
    #        signal_onoff[cam_on] = 1

    # sometimes the camera signal doesn't end with 0. In this case,
    # I define the time point where the camera switches for the last
    # time from 0 to 1 as the end point of the recording
    #    cam_off = -1
    #    if signal_onoff[-1] == 1:
    #        cam_off = np.where(np.diff(signal_onoff) == 1)[0][-1] + 1
    #        signal_onoff[cam_off::] = 0
    #
    #    start_i, end_i = laser_start_end(signal_onoff, SR=1000)
    #    start_i = start_i[0]
    #    real_start_i = start_i
    #    end_i = end_i[0]
    #    if cam_off > -1:
    #        end_i = cam_off
    #    initial_cut = signal_onoff[start_i:end_i]
    #    for i in range(len(initial_cut)):
    #        if initial_cut[i] == 0:
    #            start_i = i
    #            break
    #    real_start_i = real_start_i + start_i
    #    real_end_i = end_i + 50

    # NEW
    SR = sleepy.get_snr(ppath, recording)
    dt = 1.0 / SR
    iframe_thr = frame_thr / dt
    cam_off = np.where(np.diff(signal_onoff) == -1)[0] + 1
    cam_on = np.where(np.diff(signal_onoff) == 1)[0] + 1
    cam_jumps = np.sort(np.union1d(cam_on, cam_off))

    d = np.diff(cam_jumps)
    ilong_frame = np.where(d > iframe_thr)[0]

    plt.ion()
    plt.figure()
    plt.plot(signal_onoff[cam_jumps[0]:cam_jumps[-1]])
    plt.plot(cam_jumps[ilong_frame]-cam_jumps[0], np.ones((len(ilong_frame),)), 'r.')
    plt.show()

    real_start_i = cam_jumps[0]
    if len(ilong_frame) == 1:
        # there's a long frame right at the beginning
        if cam_jumps[ilong_frame[0]] == cam_on[0]:
            real_start_i = cam_jumps[ilong_frame[0]]
        # otherwhise, we assume the long frame is right at the end
        else:
            real_end_i = cam_jumps[ilong_frame[0]]
    elif len(ilong_frame) == 0:
        real_end_i = cam_jumps[-1]
    # there are two long frames, we assume that happened at the beginning
    # and at the end:
    elif len(ilong_frame) == 2:
        real_start_i = cam_jumps[ilong_frame[0]]
        real_end_i   = cam_jumps[ilong_frame[1]]
    else:
        print('Something wrong here? No idea what to do...')

    if os.path.isfile(os.path.join(ppath, recording, 'EEG_orig.mat')):
        eeg = np.squeeze(so.loadmat(os.path.join(ppath, recording, 'EEG_orig.mat'))['EEG'])
    else:
        eeg = np.squeeze(so.loadmat(os.path.join(ppath, recording, 'EEG.mat'))['EEG'])

    if os.path.isfile(os.path.join(ppath, recording, 'EMG_orig.mat')):
        emg = np.squeeze(so.loadmat(os.path.join(ppath, recording, 'EMG_orig.mat'))['EMG'])
    else:
        emg = np.squeeze(so.loadmat(os.path.join(ppath, recording, 'EMG.mat'))['EMG'])

    if os.path.isfile(os.path.join(ppath, recording, 'EEG2_orig.mat')):
        eeg2 = np.squeeze(so.loadmat(os.path.join(ppath, recording, 'EEG2_orig.mat'))['EEG2'])
    else:
        eeg2 = np.squeeze(so.loadmat(os.path.join(ppath, recording, 'EEG2.mat'))['EEG2'])

    if os.path.isfile(os.path.join(ppath, recording, 'EMG2_orig.mat')):
        emg2 = np.squeeze(so.loadmat(os.path.join(ppath, recording, 'EMG2_orig.mat'))['EMG2'])
    else:
        emg2 = np.squeeze(so.loadmat(os.path.join(ppath, recording, 'EMG2.mat'))['EMG2'])

    eeg_cut = eeg[real_start_i:real_end_i]
    emg_cut = emg[real_start_i:real_end_i]
    eeg2_cut = eeg2[real_start_i:real_end_i]
    emg2_cut = emg2[real_start_i:real_end_i]
    laser_cut = signal_onoff[real_start_i:real_end_i]

    so.savemat(os.path.join(ppath, recording, 'EEG.mat'), {'EEG': eeg_cut})
    so.savemat(os.path.join(ppath, recording, 'EMG.mat'), {'EMG': emg_cut})
    so.savemat(os.path.join(ppath, recording, 'EEG2.mat'), {'EEG2': eeg2_cut})
    so.savemat(os.path.join(ppath, recording, 'EMG2.mat'), {'EMG2': emg2_cut})
    so.savemat(os.path.join(ppath, recording, 'laser_' + recording + '.mat'), {'laser': laser_cut})

    so.savemat(os.path.join(ppath, recording, 'EEG_orig.mat'), {'EEG': eeg})
    so.savemat(os.path.join(ppath, recording, 'EMG_orig.mat'), {'EMG': emg})
    so.savemat(os.path.join(ppath, recording, 'EEG2_orig.mat'), {'EEG2': eeg})
    so.savemat(os.path.join(ppath, recording, 'EMG2_orig.mat'), {'EMG2': emg})
    so.savemat(os.path.join(ppath, recording, 'laser_orig_' + recording + '.mat'), {'laser': laser})

    sleepy.calculate_spectrum(ppath, recording)

    if os.path.isfile(os.path.join(ppath, recording, 'remidx_%s.txt' % recording)):
        os.remove(os.path.join(ppath, recording, 'remidx_%s.txt' % recording))


##################################################################################################
### Basic Utility Functions ######################################################################
##################################################################################################

def least_squares(x, y, n):
    A = np.zeros((len(x), n + 1))
    for i in range(n + 1):
        A[:, n - i] = np.power(x, i)

    p = LA.lstsq(A, y)[0]

    r2 = -1
    # #calculate r2 coefficient
    # S_tot = np.var(y - np.mean(y))
    # f = np.zeros( (len(x),1) )
    # for i in range(n+1) :
    #     f = f + np.power(x, n-i) * p[i]

    # S_res = np.var(y - f)
    # print S_res, S_tot
    # r2 = 1 - S_res / S_tot

    return p, r2


def downsample_matrix(X, nbin):
    """
    y = downsample_matrix(X, nbin)
    downsample the matrix X by replacing nbin consecutive \
    rows by their mean \
    @RETURN: the downsampled matrix
    """
    n_down = int(np.floor(X.shape[0] / nbin))
    X = X[0:n_down * nbin, :]
    X_down = np.zeros((n_down, X.shape[1]))

    # 0 1 2 | 3 4 5 | 6 7 8
    for i in range(nbin):
        idx = range(i, int(n_down * nbin), int(nbin))
        X_down += X[idx, :]

    return X_down / nbin




