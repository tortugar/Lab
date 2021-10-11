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
import sleepy
import pandas as pd
import seaborn as sns
from scipy import linalg as LA
import scipy.stats as stats
### DEBUGGER
import pdb



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
    


def draw_rois(ROIs, axes, c, show_num=True, lw=1):
    """
    helper function for plot_rois
    
    :param c: color
    """
    i=0
    for (x,y) in ROIs :
        l = plt.Line2D(y+[y[0]], x+[x[0]], color=c, lw=lw)
        if show_num:
            axes.text(np.max(y)-5, np.min(x)+7, str(i), fontsize=10, color=c,bbox=dict(facecolor='w', alpha=0.))
        axes.add_line(l)
        i = i+1



def plot_rois(ipath, name, roi_id, roi_set=[], amap=True, show_num=True, pplot=True):
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

    if pplot:
        plt.figure()
        axes = plt.subplot(111)
        axes.imshow(image, cmap='gray', vmin=0, vmax=np.percentile(image, 99.5))
    
    (ROI_coords, ROIs) = load_roilist(ipath, name, roi_id)
    if len(roi_set) > 0:
        ROIs_sel = [ROIs[s] for s in roi_set]
    else:
        ROIs_sel = ROIs
    
    if pplot: 
        draw_rois(ROIs_sel, axes, 'red', show_num=show_num)
        
    return ROIs_sel



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
    #sp_time = np.arange(0, sdt*M.shape[0], sdt)
    # assigns to each frame a time point:
    img_time = imaging_timing(ipath, name)

    # Second, brain state dependent averages
    nroi = F.shape[1]
    S = np.zeros((nroi,3))
    tborder = 10

    for i in range(nroi) :
        for istate in range(1,4) :
            idx = np.nonzero(M==istate)[0]
            seq = sleepy.get_sequences(idx)

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



def brstate_dff(ipath, mapping, pzscore=False, class_mode='basic', single_mice=True, dff_filt=False, f_cutoff=2.0, dff_var='dff'):
    """
    calculate average ROI DF/F activity during each brain state and then 
    perform statistics for ROIs to classify them into REM-max, Wake-max, or NREM-max.
    For each ROI anova is performed, followed by Tukey-test
    
    :param ipath: base imaging folder
    :param mapping: pandas DataFrame as returned by &load_roimapping.
           The frame contains one column for each recording in the given data set.
    :param pzscore: if True, z-score DF/F traces
    :param class_mode: class_mode == 'basic': classify ROIs into 
                       REM-max, Wake-max and NREM-max ROIs
                       class_mode == 'rem': further separate REM-max ROIs 
                       into REM > Wake > NREM (R>W>N) and REM > NREM > Wake (R>N>W) ROIs
    :param single_mice: boolean, if True use separate colors for single mice in 
                        summary plots
    :param dff_var: Default is 'dff'. Use raw DF/F signal ('dff'), denoised DF/F signal 
                    using OASIS algorithm ('dff_dn'), or deconvoluted DF/F signal ('dff_sp').
                    In case of 'dff_dn' or 'dff_sp', make sure to run first script denoise_dff.py
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
        DFF = so.loadmat(dff_file, squeeze_me=True)[dff_var]
        # brainstate
        M = sleepy.load_stateidx(ipath, rec)[0]
        state_idx = {1:[], 2:[], 3:[]}
        
        # load imaging timing
        img_time = imaging_timing(ipath, rec)
        isr = 1/np.mean(np.diff(img_time))
        print(isr)

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
            roi_num  = int(a[1])
            dff = DFF[:,roi_num] 
            
            if dff_filt:
                w0 = f_cutoff / (isr*0.5)
                dff = sleepy.my_lpfilter(dff, w0)
            
            if pzscore:
                dff = (dff-dff.mean()) / dff.std()
            else:
                dff = dff*100
            
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
                
                if cond1['p-tukey'].iloc[0] < 0.05 and cond2['p-tukey'].iloc[0] < 0.05 and res['p-unc'].iloc[0] < 0.05:
                    roi_type = 'R-max'
            # W-max
            elif (wmean > nmean) and (wmean > rmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]
                cond2 = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]
    
                if cond1['p-tukey'].iloc[0] < 0.05 and cond2['p-tukey'].iloc[0] < 0.05 and res['p-unc'].iloc[0] < 0.05:
                    roi_type = 'W-max'
            # N-max 
            elif (nmean > wmean) and (nmean > rmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]
                cond2 = res2[(res2['A'] == 'N') & (res2['B'] == 'R')]
    
                if cond1['p-tukey'].iloc[0] < 0.05 and cond2['p-tukey'].iloc[0] < 0.05 and res['p-unc'].iloc[0] < 0.05:
                    roi_type = 'N-max'
                    
            else:
                roi_type = 'X'
                            
            tmp = [r, rmean, wmean, nmean, res.F.iloc[0], res['p-unc'].iloc[0], res2['p-tukey'].iloc[0], roi_type]
            data.append(tmp)
            
        
        else:
            roi_type = 'X'

            #print(res)
            # R>N>W
            if (rmean > wmean) and (rmean > nmean) and (nmean  > wmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'R')]
                cond2 = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]
                # NEW
                cond3 = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]

                if cond1['p-tukey'].iloc[0] < 0.05 and cond2['p-tukey'].iloc[0] < 0.05 and cond3['p-tukey'].iloc[0] < 0.05 and res['p-unc'].iloc[0] < 0.05:
                    roi_type = 'R>N>W'
                    
            # R>W>N
            elif (rmean > wmean) and (rmean > nmean) and (wmean  > nmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'R')]
                cond2 = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]
                # NEW
                cond3 = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]
                
                if cond1['p-tukey'].iloc[0] < 0.05 and cond2['p-tukey'].iloc[0] < 0.05 and cond3['p-tukey'].iloc[0] < 0.05 and res['p-unc'].iloc[0] < 0.05:
                    roi_type = 'R>W>N'
                    
            # W-max
            elif (wmean > nmean) and (wmean > rmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]
                cond2 = res2[(res2['A'] == 'R') & (res2['B'] == 'W')]
    
                if cond1['p-tukey'].iloc[0] < 0.05 and cond2['p-tukey'].iloc[0] < 0.05 and res['p-unc'].iloc[0] < 0.05:
                    roi_type = 'W-max'
            # N-max 
            elif (nmean > wmean) and (nmean > rmean):  
                cond1 = res2[(res2['A'] == 'N') & (res2['B'] == 'W')]
                cond2 = res2[(res2['A'] == 'N') & (res2['B'] == 'R')]
    
                if cond1['p-tukey'].iloc[0] < 0.05 and cond2['p-tukey'].iloc[0] < 0.05 and res['p-unc'].iloc[0] < 0.05:
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
    plt.ion()
    plt.figure()
    for typ in types:
        mouse_shown = {m:0 for m in mice}
        plt.subplot(int('1%d%d' % (len(types), j)))
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


       
def brstate_transitions(ipath, roi_mapping, transitions, pre, post, si_threshold, sj_threshold, xdt=1.0, pzscore=False,
                        pspec=True, fmax=20, ylim=[], xticks=[], vm=[], cb_ticks=[], ma_thr=0, ma_rem_exception=True):
    """
    calculate average DF/F activity for ROIs along brain state transitions
    :param ipath: base folder
    :param roi_mapping: pandas DataFrame with columns specificing ROI 'ID', recordings ('MouseID_dateni')
    :param transitions: list of tuples to denote transitions to be considered;
           1 - REM, 2 - Wake, 3 - NREM; For example to calculate NREM to REM and REM to wake transitions,
           type [(3,1), (1,2)]
    :param pre: time before transition in s
    :param post: time after transition;
           NOTE: Ideally pre and post are integer multiples of the brain state time bin,
                 which is typically 2.5 s
    :param si_threshold: list of floats, thresholds how long REM, Wake, NREM should be at least before the transition.
           So, if there's a REM to Wake transition, but the duration of REM is shorter then si_threshold[0], then this
           transition if discarded.
    :param sj_threshold: list of floats, thresholds how long REM, Wake, NREM should be at least after the transition
    :param xdt: time resolution for DF/F activity
    :param pzscore: bool, if True zscore DF/F signals
    :param pspec: bool, if True also calculate normalized EEG spectrum as the transition; if True, also a figure is
           plotting summarizing the results
    :param fmax: maximum frequency shown for EEG spectrogram
    :param ylim: list, specifying y-limits for y axis of DF/F plots, for example ylim=[0, 10] will limit the yrange
           from 0 to 10
    :param xticks: list, xticks for DF/F plots
    :param vm: saturation of EEG spectrogram
    :param cb_ticks: ticks for colorbar
    :param ma_thr: wake sequences s with round(len(s)*dt) <= ma_thr are set to NREM
    :param ma_rem_exception: if True, don't interprete wake periods following REM as MA (i.e. NREM)

    Example call:
    mx, df = imaging.brstate_transitions(ppath, df_class[df_class['Type']=='N-max'], [[3,1], [1,2]], 60, 30, [60, 60, 60], [30, 30, 30])

    :return: mx: dict; Transition type --> np.array of size 'number ROIs' x number of time bins
             df: pd.DataFrame with columns 'ID', 'mouse', 'time', 'dff', 'trans'
             Note: For each ROI all transitions of type 'trans' are averaged. So for a given timepoint 'time'
             only only average 'dff' for transition 'trans' exists.

    """
    ishift = 1
    
    rois = list(roi_mapping['ID'])
    states = {1:'R', 2:'W', 3:'N'}

    roi_transact_si = dict()
    roi_transact_sj = dict()
    roi_length = dict()
    if pspec:
        roi_transspe_si = dict()
        roi_transspe_sj = dict()

    for (si,sj) in transitions:
        sid = states[si] + states[sj]
        roi_transact_si[sid] = {r:[] for r in rois}
        roi_transact_sj[sid] = {r:[] for r in rois}
        roi_transspe_si[sid] = {r:[] for r in rois}
        roi_transspe_sj[sid] = {r:[] for r in rois}

        roi_length[sid] = {r:[] for r in rois}

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

        # flatten out MAs
        if ma_thr>0:
            seq = sleepy.get_sequences(np.where(M==2)[0])
            for s in seq:
                if np.round(len(s)*sdt) <= ma_thr:
                    if ma_rem_exception:
                        if (s[0]>1) and (M[s[0] - 1] != 1):
                            M[s] = 3
                    else:
                        M[s] = 3
        
        # load imaging timing
        img_time = imaging_timing(ipath, rec)
        idt = np.diff(img_time).mean()
                
        ipre  = int(np.round(pre/idt))
        ipost = int(np.round(post/idt))

        # load spectrogram and normalize
        if pspec:
            P = so.loadmat(os.path.join(ipath, rec, 'sp_%s.mat' % rec), squeeze_me=True)
            SP = P['SP']
            freq = P['freq']
            ifreq = np.where(freq <= fmax)[0]
            sp_mean = SP.mean(axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)

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

                        # search for the index in img_time that's closest to the timepoint of transition
                        # according to brain state time:
                        jstart_dff   = eeg2img_time([(s[-1]+ishift)*sdt], img_time)[0]

                        if ipre <= jstart_dff < len(dff)-ipost and len(s)*sdt >= si_threshold[si-1] and len(sj_idx)*sdt >= sj_threshold[sj-1]:
                            # get start and end time points for transition sequence:
                            istart_dff = eeg2img_time([(s[-1]+1+ishift)*sdt - pre],  img_time)[0]
                            jend_dff   = eeg2img_time([(s[-1]+1+ishift)*sdt + post], img_time)[0]
                            
                            act_si = dff[istart_dff:jstart_dff]
                            act_sj = dff[jstart_dff:jend_dff]
                            roi_transact_si[sid][row['ID']].append(act_si)
                            roi_transact_sj[sid][row['ID']].append(act_sj)
                            roi_length[sid][row['ID']].append((len(act_si), len(act_sj)))

                            # get EEG spectrogram data
                            if pspec:
                                jstart_spec = s[-1]+1
                                istart_spec = int(np.round(((s[-1]+1)*sdt - pre) / sdt))
                                jend_spec = int(np.round(((s[-1]+1)*sdt + post) / sdt))

                                spe_si = SP[ifreq, istart_spec:jstart_spec]
                                spe_sj = SP[ifreq, jstart_spec:jend_spec]
                                roi_transspe_si[sid][row['ID']].append(spe_si)
                                roi_transspe_sj[sid][row['ID']].append(spe_sj)

    ntime = 0
    ntime_spe = (0,0)
    for index, row in roi_mapping.iterrows():
        for (si,sj) in transitions:
            sid = states[si] + states[sj]
            
            tmp_si = roi_transact_si[sid][row['ID']]
            tmp_sj = roi_transact_sj[sid][row['ID']]

            if len(tmp_si) > 0:                
                tmp_si = np.vstack([time_morph(t,int(pre/xdt)) for t in tmp_si])
                tmp_sj = np.vstack([time_morph(t,1+int(post/xdt)) for t in tmp_sj])
                roi_transact_si[sid][row['ID']] = tmp_si
                roi_transact_sj[sid][row['ID']] = tmp_sj
                ntime = tmp_si.shape[1] + tmp_sj.shape[1]

                if pspec:
                    tmp_si = np.array(roi_transspe_si[sid][row['ID']])
                    tmp_sj = np.array(roi_transspe_sj[sid][row['ID']])

                    roi_transspe_si[sid][row['ID']] = tmp_si
                    roi_transspe_sj[sid][row['ID']] = tmp_sj
                    ntime_spe = (tmp_si.shape[2], tmp_sj.shape[2])
            else:
                print('No transitions for ROI %d' % row['ID'])

            print('Done with ROI %d' % row['ID'])

    ti = np.linspace(-pre, -xdt, int(pre/xdt))
    tj = np.linspace(0, post-xdt+xdt, 1+int(post/xdt))
    xtime = np.concatenate((ti,tj))
    stime = np.concatenate((np.arange(-2.5*ntime_spe[0], -0.1, 2.5), np.arange(0, sdt*(ntime_spe[1]-1)+0.1, sdt)))

    #roi_transact_mean = dict()
    mx_transact = dict()
    mx_transspe = dict()
    for (si,sj) in transitions:
        sid = states[si] + states[sj]
        #roi_transact_mean[sid] = {r:[] for r in rois}
        mx_transact[sid] = np.zeros((len(rois), ntime))
        mx_transspe[sid] = np.zeros((len(ifreq), ntime_spe[0]+ntime_spe[1], len(rois)))

    j = 0
    d = {'ID':[], 'mouse':[], 'time':[], 'dff':[], 'trans':[]}
    for index, row in roi_mapping.iterrows():
        for (si,sj) in transitions:
            sid = states[si] + states[sj]
            print(sid)
            tmp_si = roi_transact_si[sid][row['ID']]
            tmp_sj = roi_transact_sj[sid][row['ID']]
            #roi_transact_mean[sid][row['ID']] = np.hstack([tmp_si, tmp_sj])
            #mx_transact[sid][j,:] = roi_transact_mean[sid][row['ID']].mean(axis=0)
            tmp = np.hstack([tmp_si, tmp_sj])
            mx_transact[sid][j, :] = tmp.mean(axis=0)

            if pspec:
                tmp_si = roi_transspe_si[sid][row['ID']]
                tmp_sj = roi_transspe_sj[sid][row['ID']]
                tmp = np.hstack([tmp_si.mean(axis=0), tmp_sj.mean(axis=0)])
                mx_transspe[sid][:,:,j] = tmp

            d['ID'] += [row['ID']]*ntime
            d['mouse'] += [row['mouse']]*ntime
            d['time'] += list(xtime)
            d['dff'] += list(mx_transact[sid][j,:])
            d['trans'] += [sid]*ntime
       
        j+=1

    df = pd.DataFrame(d)            

    ### PLOTTING #######################################################################################################
    if pspec:
        plt.ion()
        ntrans = len(transitions)
        plt.figure(figsize=(10, 5))
        nx = 1.0/ntrans
        dx = 0.2 * nx
        nroi = len(rois)
        f = freq[ifreq]
        i = 0
        for (si,sj) in transitions:
            tr = states[si] + states[sj]
            # plot DF/F
            ax = plt.axes([nx*i+dx, 0.15, nx-dx-dx/3.0, 0.3])
            tmp = mx_transact[tr].mean(axis=0)
            plt.plot(xtime, tmp, color='blue')
            sem = np.std(mx_transact[tr],axis=0) / np.sqrt(nroi)
            ax.fill_between(xtime, tmp - sem, tmp + sem, color=(0, 0, 1), alpha=0.5, edgecolor=None)

            sleepy.box_off(ax)
            plt.xlabel('Time (s)')
            plt.xlim([xtime[0], xtime[-1]])
            if i==0:
                if pzscore:
                    plt.ylabel('$\Delta$F/F (z-scored)')
                else:
                    plt.ylabel('$\Delta$F/F (%)')
            if len(ylim) == 2:
                plt.ylim(ylim)
            if len(xticks) == 2:
                plt.xticks(xticks)
            # END - DF/F

            # plot spectrogram
            if i==0:
                axes_cbar = plt.axes([nx*i+dx +1.5*dx, 0.55+0.25+0.03, nx - dx-dx/3.0, 0.1])
            ax = plt.axes([nx * i + dx, 0.55, nx - dx-dx/3.0, 0.25])
            plt.title(states[si] + ' $\\rightarrow$ ' + states[sj])

            # dimensions of trans_spe: frequencies x time x number of ROIs
            # so, to average over mice, average over 3rd dimension (axis)
            im = ax.pcolorfast(stime, f, mx_transspe[tr].mean(axis=2), cmap='jet')

            if len(vm) > 0:
                im.set_clim(vm)
            ax.set_xticks([0])
            ax.set_xticklabels([])
            if i==0:
                plt.ylabel('Freq. (Hz)')
            if i>0:
                ax.set_yticklabels([])
            sleepy.box_off(ax)
            # END - spectrogram

            # colorbar for EEG spectrogram
            if i==0:
                cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0, orientation='horizontal')
                cb.set_label('Rel. Power')
                cb.ax.xaxis.set_ticks_position("top")
                cb.ax.xaxis.set_label_position('top')
                if len(cb_ticks) > 0:
                    cb.set_ticks(cb_ticks)
                axes_cbar.set_alpha(0.0)
                axes_cbar.spines["top"].set_visible(False)
                axes_cbar.spines["right"].set_visible(False)
                axes_cbar.spines["bottom"].set_visible(False)
                axes_cbar.spines["left"].set_visible(False)
                axes_cbar.axes.get_xaxis().set_visible(False)
                axes_cbar.axes.get_yaxis().set_visible(False)
            # END - colorbar

            i += 1

    return mx_transact, df



def brstate_transitions_simple(ipath, roi_mapping, transitions, pre, post, si_threshold, sj_threshold, pzscore=False,
                        pspec=True, fmax=20, ylim=[], xticks=[], vm=[], cb_ticks=[], ma_thr=0, ma_rem_exception=True, spe_filt=[], dff_var='dff'):
    """
    calculate average DF/F activity for ROIs along brain state transitions. The DF/F activity is binned like the hypnogram,
    using the function &downsample_dff2bs(...)
    See also brstate_transitions
    
    :param ipath: base folder
    :param roi_mapping: pandas DataFrame with columns specificing ROI 'ID', recordings ('MouseID_dateni')
    :param transitions: list of tuples to denote transitions to be considered;
           1 - REM, 2 - Wake, 3 - NREM; For example to calculate NREM to REM and REM to wake transitions,
           type [(3,1), (1,2)]
    :param pre: time before transition in s
    :param post: time after transition;
           NOTE: Ideally pre and post are integer multiples of the brain state time bin,
                 which is typically 2.5 s
    :param si_threshold: list of floats, thresholds how long REM, Wake, NREM should be at least before the transition.
           So, if there's a REM to Wake transition, but the duration of REM is shorter then si_threshold[0], then this
           transition if discarded.
    :param sj_threshold: list of floats, thresholds how long REM, Wake, NREM should be at least after the transition
    :param pzscore: bool, if True zscore DF/F signals
    :param pspec: bool, if True also calculate normalized EEG spectrum as the transition; if True, also a figure is
           plotting summarizing the results
    :param fmax: maximum frequency shown for EEG spectrogram
    :param ylim: list, specifying y-limits for y axis of DF/F plots, for example ylim=[0, 10] will limit the yrange
           from 0 to 10
    :param xticks: list, xticks for DF/F plots
    :param vm: saturation of EEG spectrogram
    :param cb_ticks: ticks for colorbar
    :param ma_thr: wake sequences s with round(len(s)*dt) <= ma_thr are set to NREM
    :param ma_rem_exception: if True, don't interprete wake periods following REM as MA (i.e. NREM)
    :param dff_var: string. Three options: 'dff' - raw DF/F signal; 
           'dff_dn' - denoised DF/F signal; 'dff_sp' - deconvolved DF/F signal

    Example call:
    mx, df = imaging.brstate_transitions(ppath, df_class[df_class['Type']=='N-max'], [[3,1], [1,2]], 60, 30, [60, 60, 60], [30, 30, 30])

    :return: mx: dict; Transition type --> np.array of size 'number ROIs' x number of time bins
             df: pd.DataFrame with columns 'ID', 'mouse', 'time', 'dff', 'trans'
             Note: For each ROI all transitions of type 'trans' are averaged. So for a given timepoint 'time'
             only only average 'dff' for transition 'trans' exists.

    """    
    rois = list(roi_mapping['ID'])
    states = {1:'R', 2:'W', 3:'N'}

    if pspec:
        #roi_transspe = dict()
        trials_transspe = dict()

    for (si,sj) in transitions:
        sid = states[si] + states[sj]
        #roi_transspe[sid] = {r:[] for r in rois}
        trials_transspe[sid] = []

    roi_stateval = {}
    for r in rois:
        roi_stateval[r] = {1:[], 2:[], 3:[]}
    
    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]    
    data = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
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
        DFF = so.loadmat(dff_file, squeeze_me=True)[dff_var]
        DFF = downsample_dff2bs(ipath, rec, roi_list, psave=False, peaks=False, dff_data = DFF)
        
        # brainstate
        M = sleepy.load_stateidx(ipath, rec)[0]

        # flatten out MAs
        if ma_thr>0:
            seq = sleepy.get_sequences(np.where(M==2)[0])
            for s in seq:
                if np.round(len(s)*sdt) <= ma_thr:
                    if ma_rem_exception:
                        if (s[0]>1) and (M[s[0] - 1] != 1):
                            M[s] = 3
                    else:
                        M[s] = 3
        
        # set time variables
        ipre  = int(np.round(pre/sdt))
        ipost = int(np.round(post/sdt))+1
        tdff = np.arange(-ipre*sdt, ipost*sdt-sdt + sdt/2, sdt)
        m = len(tdff)

        # load spectrogram and normalize
        if pspec:
            P = so.loadmat(os.path.join(ipath, rec, 'sp_%s.mat' % rec), squeeze_me=True)
            SP = P['SP']
            freq = P['freq']
            ifreq = np.where(freq <= fmax)[0]
            f = freq[ifreq]
            sp_mean = SP.mean(axis=1)
            
            if len(spe_filt) > 0:
                filt = np.ones(spe_filt)
                filt = np.divide(filt, filt.sum())
                SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')            
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)


        got_spe = {}
        for (si,sj) in transitions:
            sid = states[si] + states[sj]        
            got_spe[sid] = False
            
        for index, row in rec_map.iterrows():
            # for recording $rec, process all ROIs
            s=row[rec]
            a = re.split('-', s)
            roi_num  = int(a[1])
            dff = DFF[:,roi_num]
            ID = row['ID']
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

                        # search for the index in img_time that's closest to the timepoint of transition
                        # according to brain state time:
                        if ipre <= ti < len(M)-ipost and len(s)*sdt >= si_threshold[si-1] and len(sj_idx)*sdt >= sj_threshold[sj-1]:

                            #act_si = dff[ti-ipre+1:ti+1]
                            #act_sj = dff[ti+1:ti+ipost+1]
                            act = dff[ti-ipre+1:ti+ipost+1]
                            
                            #act = np.concatenate((act_si, act_sj))

                            data += zip([ID]*m, [idf]*m, [rec]*m, [sid]*m, act, tdff)

                            # get EEG spectrogram data
                            if pspec and not(got_spe[sid]):
                                spe_si = SP[ifreq,ti-ipre+1:ti+1]
                                spe_sj = SP[ifreq,ti+1:ti+ipost+1]
                                spe = np.concatenate((spe_si, spe_sj), axis=1)
                                
                                #roi_transspe[sid][row['ID']].append(spe)
                                trials_transspe[sid].append(spe)
            
                got_spe[sid] = True
                                
    df = pd.DataFrame(data=data, columns = ['ID', 'mouse', 'recording', 'trans', 'dff', 'time'])

    nroi = roi_mapping.ID.unique().shape[0]
    mx_transact = dict()
    mx_transspe = dict()
    for (si,sj) in transitions:
        sid = states[si] + states[sj]
        mx_transact[sid] = np.zeros((nroi, m))
        mx_transspe[sid] = np.zeros((nroi, len(f), m))

    i_roi = 0
    for index, row in roi_mapping.iterrows():
        for (si,sj) in transitions:
            sid = states[si] + states[sj]
            
            #pdb.set_trace()
            tmp = df.loc[(df.ID == row['ID']) & (df.trans == sid), :]
            tmp= np.array(tmp.groupby(['time']).mean()['dff'])
            
            if len(tmp) > 0:
                mx_transact[sid][i_roi,:] = tmp
                
                #if pspec:
                    #tmp = np.array(roi_transspe[sid][row['ID']])
                    #mx_transspe[sid][i_roi,:,:] = tmp.mean(axis=0)
                #    pdb.set_trace()
                #    trials_transspe[sid] = np.array(trials_transspe[sid]).mean(axis=0)
                
            else:
                print('No transitions for ROI %d' % row['ID'])
            
        i_roi += 1
        
    if pspec:
        for (si,sj) in transitions:
            sid = states[si] + states[sj]
            trials_transspe[sid] = np.array(trials_transspe[sid]).mean(axis=0)
            
    ### PLOTTING #######################################################################################################
    if pspec:
        plt.ion()
        ntrans = len(transitions)
        plt.figure(figsize=(10, 5))
        nx = 1.0/ntrans
        dx = 0.2 * nx
        nroi = len(rois)
        f = freq[ifreq]
        i = 0
        for (si,sj) in transitions:
            tr = states[si] + states[sj]
            # plot DF/F
            ax = plt.axes([nx*i+dx, 0.15, nx-dx-dx/3.0, 0.3])
            tmp = mx_transact[tr].mean(axis=0)
            plt.plot(tdff, tmp, color='blue')
            sem = np.std(mx_transact[tr],axis=0) / np.sqrt(nroi)
            ax.fill_between(tdff, tmp - sem, tmp + sem, color=(0, 0, 1), alpha=0.5, edgecolor=None)

            sleepy.box_off(ax)
            plt.xlabel('Time (s)')
            plt.xlim([tdff[0], tdff[-1]])
            if i==0:
                if pzscore:
                    plt.ylabel('$\Delta$F/F (z-scored)')
                else:
                    plt.ylabel('$\Delta$F/F (%)')
            if len(ylim) == 2:
                plt.ylim(ylim)
            if len(xticks) == 2:
                plt.xticks(xticks)
            # END - DF/F

            # plot spectrogram
            if i==0:
                axes_cbar = plt.axes([nx*i+dx +1.5*dx, 0.55+0.25+0.03, nx - dx-dx/3.0, 0.1])
            ax = plt.axes([nx * i + dx, 0.55, nx - dx-dx/3.0, 0.25])
            plt.title(states[si] + ' $\\rightarrow$ ' + states[sj])

            # dimensions of trans_spe: number of ROIs x frequencies x time
            # so, to average over mice, average over 0th dimension (axis)
            #im = ax.pcolorfast(tdff, f, mx_transspe[tr].mean(axis=0), cmap='jet')
            im = ax.pcolorfast(tdff, f, trials_transspe[tr][:,:-1], cmap='jet')
            # Why the :-1?
            # Say we want to plot 0 to 5 s: So for the DF/F signal, 
            # there a point for 0, 2.5 and 5s
            # x  x  x
            # These are the two bins in the spectrogr
            #  -- -- --
            # |  |  |  |
            # 0 2.5 5 7.5
            # So here paradoxially only 2 bins are needed to go from 0 to 5s

            if len(vm) > 0:
                im.set_clim(vm)
            ax.set_xticks([0])
            ax.set_xticklabels([])
            if i==0:
                plt.ylabel('Freq. (Hz)')
            if i>0:
                ax.set_yticklabels([])
            sleepy.box_off(ax)
            # END - spectrogram

            # colorbar for EEG spectrogram
            if i==0:
                cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0, orientation='horizontal')
                cb.set_label('Rel. Power')
                cb.ax.xaxis.set_ticks_position("top")
                cb.ax.xaxis.set_label_position('top')
                if len(cb_ticks) > 0:
                    cb.set_ticks(cb_ticks)
                axes_cbar.set_alpha(0.0)
                axes_cbar.spines["top"].set_visible(False)
                axes_cbar.spines["right"].set_visible(False)
                axes_cbar.spines["bottom"].set_visible(False)
                axes_cbar.spines["left"].set_visible(False)
                axes_cbar.axes.get_xaxis().set_visible(False)
                axes_cbar.axes.get_yaxis().set_visible(False)
            # END - colorbar

            i += 1

    df = df.groupby(['ID', 'mouse', 'recording', 'trans', 'time']).mean().reset_index()

    return mx_transact, df
                                


def brstate_transition_stats(df, timevec, trans_type, bonf=True):
    """
    Test at which time point a brain state transition becomes significant.
    The functions works together with brstate_transition_stats
    :param df: pandas DataFrame as returned (2nd argument) from brstate_transitions().
           The data frame contains columns 'ID' (ROI id number), 'mouse', 'time', 'dff' and 'trans'
           (the transition type, encoded through 2 character strings, e.g. 'NR').
    :param timevec: np.array or list, defining time intervals for which ttest is performed to test
           which time intervals are different from the baseline interval.
           timevec[0] to timevec[1] defines the baseline interval.
    :param trans_type:
    :return: pandas DataFrame, with coluns 'time', 'p' (p-value), and 'sig' (1 - significant, 0 - non-significant

    Example call:
    # get brainstate transitions as DataFrame:
    mx, df = imaging.brstate_transitions(path, df_class[df_class['Type']=='R-max'], [[3,1], [1,2]], 60, 20, [60, 60, 60], [20, 20, 20])
    # use DataFrame as input for brstate_transition_stats:
    imaging.brstate_transition_stats(df, np.arange(-60, 21, 10), 'NR')

    -60 to -50s will serve as baseline interval.
    The average DF/F activity during each sucessive interval (-50 to -40, -40 to -30, etc.).
    The interval -50 to -40 will be referenced as -45s in the time column of the returned DataFrame
    """

    tmp = df[(df.time >= timevec[0]) & (df.time < timevec[1])]
    bsl = tmp[tmp.trans == trans_type].groupby(['ID']).mean()['dff']

    tref = []
    pval = []
    sig = []
    if bonf:
        alpha = 0.05 / (len(timevec)-2)
    else:
        alpha = 0.05
    for (i,j) in zip(timevec[1:-1], timevec[2:]):
        tmp = df[(df.time >= i) & (df.time < j)]
        val = tmp[tmp.trans == trans_type].groupby(['ID']).mean()['dff']

        p = stats.ttest_rel(bsl, val)[1]
        if p < alpha:
            sig.append(1)
        else:
            sig.append(0)
        pval.append(p)
        tref.append(i + (j-i)/2)

    df_stats = pd.DataFrame({'time':tref, 'p':pval, 'sig':sig})
    print(df_stats)

    return df_stats



def dff_interrem(ipath, roi_mapping, nstates_rem=10, nstates_irem=20, pspec=True,
                 pzscore=True, fmax=30):
    """
    Calculate DF/F activity of ROIs in given roi_mapping across time-normalized
    inter-REM interval

    Parameters
    ----------
    ipath : string
        Imaging base folder
    roi_mapping : pd.DataFrame
        ROI mapping, as returned by load_roimapping. DataFrame with 
        columns "ROI ID", "mouse", "Recording1", "Recording2", etc.
        Example:
                    ID mouse J361_081320n1
                0    0  J361           1-0
                1    1  J361           1-1
                2    2  J361           1-2
                3    3  J361           1-3

    nstates_rem : int, optional
        Number of normalized time bins
    nstates_irem : int, optional
        DESCRIPTION. The default is 20.
    pspec : bool, optional
        DESCRIPTION. The default is True.
    pzscore : bool, optional
        DESCRIPTION. The default is True.
    fmax : int, optional
        Maximum frequency in EEG spectrogram (currently not implemented)

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns "mouse name", "ROI ID", "recording", "DF/F", "normalized time point"
 
    """
    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]   
    
    data = []
    for rec in recordings:
        print(rec)
        idf = re.split('_', rec)[0]
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
        #pdb.set_trace()

        # load spectrogram and normalize
        if pspec:
            P = so.loadmat(os.path.join(ipath, rec, 'sp_%s.mat' % rec), squeeze_me=True)
            SP = P['SP']
            freq = P['freq']
            ifreq = np.where(freq <= fmax)[0]
            sp_mean = SP.mean(axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)

        # go through all ROIs in recording rec
        for index, row in rec_map.iterrows():
            s=row[rec]
            roi_num = int(re.split('-', s)[1])
            dff = DFF[:,roi_num]
            if pzscore:
                dff = (dff-dff.mean())/dff.std()
            
            # All REM sleep sequences
            seq = sleepy.get_sequences(np.where(M==1)[0])
            if len(seq) >= 2:
                for (si, sj) in zip(seq[0:-1], seq[1:]):
                    irem_idx = np.arange(si[-1]+1, sj[0], dtype='int')
                    
                    # returns indices of the imaging frames that are closest to given time points 
                    a = eeg2img_time([si[0]*sdt, si[-1]*sdt],  img_time)
                    rempre_idx = np.arange(a[0], a[-1]+1, dtype='int')
                    
                    a = eeg2img_time([sj[0]*sdt, sj[-1]*sdt],  img_time)
                    rempost_idx = np.arange(a[0], a[-1]+1, dtype='int')
                    
                    irem_idx = np.arange(rempre_idx[-1]+1, rempost_idx[0], dtype='int')
                    
                    dff_pre  = time_morph(dff[rempre_idx], nstates_rem)
                    dff_irem = time_morph(dff[irem_idx], nstates_irem)
                    dff_post = time_morph(dff[rempost_idx], nstates_rem)
    
                    dff_rir = np.concatenate((dff_pre, dff_irem, dff_post))
                    
                    m = dff_rir.shape[0]
                    tlabel = list(range(m))
                    
                    
                    data += zip([idf]*m, [roi_num]*m, [rec]*m, dff_rir, tlabel)
    

    df = pd.DataFrame(data=data, columns=['mouse', 'ID', 'recording',  'dff', 'time'])
    return df



def dff_remrem_sections(ipath, roi_mapping, nsections=5, pzscore=False, ma_thr=10):
    """
    Calculate the NREM/wake activity of the given ROIs within consecutive sections 
    of the inter-REM cycle

    Parameters
    ----------
    ipath : string
        DESCRIPTION.
    roi_mapping : pd.DataFrame
        DESCRIPTION.
    nsections : int, optional
        DESCRIPTION. The default is 5.
    pzscore : bool, optional
        DESCRIPTION. The default is False.
    ma_thr : int, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    df_trials : pd.DataFrame
        DESCRIPTION.

    """
    
    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]   
    
    data = []
    ev = 0
    for rec in recordings:
        print(rec)
        idf = re.split('_', rec)[0]
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
        #DFF = so.loadmat(dff_file, squeeze_me=True)['dffd']
        DFF = downsample_dff2bs(ipath, rec, roi_list)
        # brainstate
        M = sleepy.load_stateidx(ipath, rec)[0]
        # flatten out microarousals
        if ma_thr > 0:
            seq = sleepy.get_sequences(np.where(M == 2)[0])
            for s in seq:
                if np.round(len(s)*sdt) < ma_thr:
                    if (s[0] > 1) and (M[s[0] - 1] != 1):
                        M[s] = 3
        
        # go through all ROIs in recording rec
        for index, row in rec_map.iterrows():
            s=row[rec]
            roi_num = int(re.split('-', s)[1])
            dff = DFF[:,roi_num]
            if pzscore:
                dff = (dff-dff.mean())/dff.std()
            
            # All REM sleep sequences
            seq = sleepy.get_sequences(np.where(M==1)[0])
            if len(seq) >= 2:
                for (si, sj) in zip(seq[0:-1], seq[1:]):
                    
                    irem_idx = np.arange(si[-1]+1, sj[0], dtype='int')
                    m = len(irem_idx)
                    
                    M_up = upsample_mx( M[irem_idx], nsections )
                    M_up = np.round(M_up)
                    dff_up = upsample_mx(dff[irem_idx], nsections)

                    single_event_nrem = []
                    single_event_wake = []
                    for p in range(nsections):
                        # for each m consecutive bins calculate average NREM, REM, Wake activity
                        mi = list(range(p*m, (p+1)*m))

                        idcut = np.intersect1d(mi, np.where(M_up == 2)[0])
                        if len(idcut) == 0:
                            wake_dff = np.nan
                        else:
                            wake_dff = np.nanmean(dff_up[idcut])
                        single_event_wake.append(wake_dff)
    
                        idcut = np.intersect1d(mi, np.where(M_up == 3)[0])
                        if len(idcut) == 0:
                            nrem_dff = np.nan
                        else:
                            nrem_dff = np.nanmean(dff_up[idcut])
                        single_event_nrem.append(nrem_dff)

                    idur = len(irem_idx)*sdt
                    data += zip([idf]*nsections, [roi_num]*nsections, [ev]*nsections, list(range(1, nsections+1)), single_event_nrem, ['NREM']*nsections, [idur]*nsections)
                    data += zip([idf]*nsections, [roi_num]*nsections, [ev]*nsections, list(range(1, nsections+1)), single_event_wake, ['Wake']*nsections, [idur]*nsections)
                    ev += 1
    
    df_trials = pd.DataFrame(columns = ['mouse', 'ID', 'event', 'section', 'dff', 'state', 'idur'], data=data)
    
    return df_trials                



def bout_timecourse(ipath, roi_mapping, states, min_dur={}, nsections=4, ma_thr=10, ma_rem_exception=True, pzscore=True):
    """
    Caluculate timecourse of DF/F activity throughout single episodes.
        

    Parameters
    ----------
    ipath : str
        imaging base folder.
    roi_mapping : str
        ROI mapping.
    states : list
        List with possible element 'REM', 'Wake', or 'NREM'.
    min_dur : dict, optional
        Minimal duration for each state, specified as dictionary of the form {'REM':x, 'Wake':y, 'NREM':z}. The default is {}.
    nsections : int, optional
        Number of bins for normalized time. The default is 4.
    ma_thr : float, optional
        Microarousal threshold. The default is 10.
    ma_rem_exception : bool, optional
        If True, leave wake periods directly following REM as wake, even if they're shorter than $ma_thr. The default is True.
    pzscore : bool, optional
        If True, z-score DF/F activity. The default is True.

    Returns
    -------
    df : pd.DataFrame
        with columns ['mouse', 'ID', 'Type', 'section', 'state', 'dff'].
    """
    if len(min_dur) == 0:
        min_dur['REM'] = 0
        min_dur['Wake'] = ma_thr
        min_dur['NREM'] = 0
    
    state_map = {'REM':1, 'Wake':2, 'NREM':3}

    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]   
    
    data = []
    for rec in recordings:
        print(rec)
        img_time = imaging_timing(ipath, rec)

        idf = re.split('_', rec)[0]
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
        if pzscore:
            for j in range(DFF.shape[1]):
                DFF[:,j] = (DFF[:,j]-DFF[:,j].mean()) / DFF[:,j].std()

        # brainstate
        M = sleepy.load_stateidx(ipath, rec)[0]
        # flatten out microarousals
        if ma_thr > 0:
            seq = sleepy.get_sequences(np.where(M == 2)[0])
            for s in seq:
                if np.round(len(s)*sdt) < ma_thr:
                    if (s[0] > 1) and (M[s[0] - 1] != 1):
                        M[s] = 3
        
        teeg = np.arange(0, len(M)*nbin) * (1/sr)
        for state in states:
            seq = sleepy.get_sequences(np.where(M==state_map[state])[0])
            seq = [s for s in seq if len(s)*sdt >= min_dur[state]]
        
            for s in seq:
                si = s[0]*nbin
                sj = s[-1]*nbin-1
                
                ti = teeg[si]
                tj = teeg[sj]
                
                dur = tj - ti
                dur_section = dur/nsections
                
                for i in range(nsections):
                    a = ti+i*dur_section
                    b = ti+(i+1)*dur_section
                    
                    idx = eeg2img_time([a,b], img_time)
                    
                    dff_cut = DFF[idx[0]:idx[-1]+1].mean(axis=0)
                    
                    for (index, row) in rec_map.iterrows():
                        ID = row['ID']
                        Type = row['Type']
                        
                        s=row[rec]
                        roi_num = int(re.split('-', s)[1])
                        roi_list = int(re.split('-', s)[0])
                        
                        data += [[idf, ID, s, Type, i, state, dff_cut[roi_num]]]
                                                
    df = pd.DataFrame(data=data, columns=['mouse', 'ID', 'ROI', 'Type', 'section', 'state', 'dff'])
    df = df.groupby(['mouse', 'ID', 'Type', 'section', 'state']).mean().reset_index()
    
    return df



def pearson_state_corr(ipath, roi_mapping, band, ma_thr=10, min_dur=20, pnorm_spec=True, pzscore=True):
    """
    calculate brain state dependent pearson correlation for given list of ROIs

    Parameters
    ----------
    ipath : TYPE
        DESCRIPTION.
    roi_mapping : pd.DataFrame
        ROI mapping: pd.DataFrame with columns:
        ['mouse', 'ID', 'recording1', 'recording2', ...]
    band : tuple
        Frequency band, specified as [f_low, f_high]
    ma_thr : float, optional
        MA threshold. The default is 10.
    pnorm_spec : bool, optional
        If True, normalize EEG spectrogram before calculating pearson's r. The default is True.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with one row for each ROI. The columns are
        ['mouse', 'recording', 'ID', 'r', 'p', 'sig', 'state']
        'r' is the pearson correlation coefficient
        'p' the p-value of the correlation
        'sig': yes or no, depending on whether the correlation is significant or not
        'state': REM, Wake, or NREM 
    """
    
    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]   
    
    data = []
    state_map = {1:'REM', 2:'Wake', 3:'NREM'}
    for rec in recordings:
        print(rec)
        idf = re.split('_', rec)[0]
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

        DFF = downsample_dff2bs(ipath, rec, roi_list)
        # brainstate
        M = sleepy.load_stateidx(ipath, rec)[0]
        # flatten out microarousals
        if ma_thr > 0:
            seq = sleepy.get_sequences(np.where(M == 2)[0])
            for s in seq:
                if np.round(len(s)*sdt) < ma_thr:
                    #if (s[0] > 1) and (M[s[0] - 1] != 1):
                    M[s] = 3
        

        # load spectrogram and normalize
        P = so.loadmat(os.path.join(ipath, rec, 'sp_%s.mat' % rec), squeeze_me=True)
        SP = P['SP']
        freq = P['freq']
        ifreq = np.where((freq >= band[0]) & (freq <= band[1]))[0]
        df = freq[1] - freq[0]

        if pnorm_spec:
            sp_mean = SP.mean(axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
            pow_band = SP[ifreq,:].mean(axis=0)
        else:
            pow_band = SP[ifreq,:].sum(axis=0)*df


        ######################################################################
        # go through all ROIs in recording rec
        for index, row in rec_map.iterrows():
            s=row[rec]
            roi_num = int(re.split('-', s)[1])
            dff = DFF[:,roi_num]
                
            if pzscore:
                dff = (dff - dff.mean()) / dff.std()
            
            for s in [1,2,3]:
                idx = np.where(M==s)[0]
                seq = sleepy.get_sequences(idx)
                seq = [s for s in seq if np.round(len(s)*sdt) >= min_dur]
                idx = np.concatenate(seq)
                
                r,p = scipy.stats.pearsonr(dff[idx], pow_band[idx])
                if p < 0.05:
                    sig = 'yes'
                else:
                    sig = 'no'
                    
                data.append([idf, rec, roi_num, r, p, sig, state_map[s]])
    
    df = pd.DataFrame(data=data, columns=['mouse', 'recording', 'ID', 'r', 'p', 'sig', 'state'])    

    return df



def bandpass_corr_state(ipath, roi_mapping, band, fft_win=2.5, perc_overlap=0.8, win=120, state=3, 
                        tbreak=0, pzscore=True, ma_thr=10, ma_rem_exception=True, pnorm_spec=True,
                        mode='cross'):
    """
    calculate cross-correlcation between each single ROI in the given ROI_mapping and
    the the power in the given frequency band

    We correlate(DF/F, power)
    So, a negative peak means that 'DFF' precedes 'power'.
    A positive peak in turn means that 'DFF' follows 'power'.


    Parameters
    ----------
    ipath : string
        Imaging base folder
    roi_mapping : pd.DataFrame
        ROI mapping
    band : tuple/list
        tuple or list with two floats, specificing lower and uppoer bound of the frequency band
    fft_win : float, optional
        Time window for FFT calculation (very different from parameter $win). The default is 2.5.
    perc_overlap : float, optional
        value between 0 and 1; percentage of overlap between two
        sucessive FFT windows. The default is 0.8.
    win : float, optional
        Time window for each cross correlation is calculated. The default is 120.
    state : int, optional
        1-REM, 2-Wake, 3-NREM. The default is 3.
    tbreak : float, optional
        DESCRIPTION. The default is 0.
    pzscore : bool, optional
        DESCRIPTION. The default is True.
    ma_thr : float, optional
        DESCRIPTION. The default is 10.
    ma_rem_exception : bool, optional
        DESCRIPTION. The default is True.
    pnorm_spec : bool, optional
        DESCRIPTION. The default is True.
    mode : string, optional
        DESCRIPTION. The default is 'cross'.

    Returns
    -------
    df : np.DataFrame
        with columns ['mouse', 'recording', 'ID', 'time', 'cc'].
        'ID' is the unique id of an ROI
        'cc' is the cross correlation 

    """
    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]    
    
    CC = []
    for rec in recordings:
        print(rec)
        idf = re.split('_', rec)[0]
        sr = sleepy.get_snr(ipath, rec)
        nbin = int(np.round(sr)*2.5)
        sdt = nbin * (1.0/sr)
        
        nwin = int(np.round(sr * fft_win))
        if nwin % 2 == 1:
            nwin += 1
        noverlap = int(nwin*perc_overlap)
    
        EEG = so.loadmat(os.path.join(ipath, rec, 'EEG.mat'), squeeze_me=True)['EEG']
        bs_time = np.arange(0, EEG.shape[0])*(1/sr)
        
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
        
        if ma_thr>0:
            seq = sleepy.get_sequences(np.where(M==2)[0])
            for s in seq:
                if np.round(len(s)*sdt) <= ma_thr:
                    if ma_rem_exception:
                        if (s[0]>1) and (M[s[0] - 1] != 1):
                            M[s] = 3
                    else:
                        M[s] = 3
        
        img_time = imaging_timing(ipath, rec)
        
        # get all bouts of state $state
        seq = sleepy.get_sequences(np.where(M == state)[0], ibreak=int(tbreak/2.5)+1)
        # as the cross-correlation should range from -$win to $win, each bout
        # needs to be at least 2 * $win seconds long
        seq = [s for s in seq if len(s)*2.5 > 2*win]
        
        seq_num = 0
        for s in seq:
            i = s[0] * nbin
            j = s[-1] * nbin + 1
            EEGcut = EEG[i:j]
            bs_time_cut = bs_time[i:j]            

            f, t, Pow = scipy.signal.spectrogram(EEGcut, nperseg=nwin, noverlap=noverlap, fs=sr)
            if pnorm_spec:
                sp_mean = Pow.mean(axis=1)
                Pow = np.divide(Pow, np.tile(sp_mean, (Pow.shape[1], 1)).T)
    
            ifreq = np.where((f >= band[0]) & (f <= band[1]))[0]
            dt = t[1] - t[0]
            iwin = int(win / dt)
            cc_time = np.arange(-iwin, iwin+1) * dt

    
            if mode == 'cross':
                if not pnorm_spec:
                    pow_band = Pow[ifreq, :].sum(axis=0) * (f[1]-f[0])
                else:
                    pow_band = Pow[ifreq, :].mean(axis=0)
    
                pow_band -= pow_band.mean()

            DFFd = downsample_overlap(DFF, bs_time_cut, img_time, nwin, noverlap)

                        
            # go through all ROIs in recording rec
            for index, row in rec_map.iterrows():
                s=row[rec]
                ROI_ID = row['ID']
                roi_num = int(re.split('-', s)[1])
                dffd = DFFd[:,roi_num]
                if 'Type' in row.index:
                    typ = row['Type']
                else:
                    typ = 'X'
                
                if pzscore:
                    dffd = (dffd - dffd.mean()) / dffd.std()
    
                # to avoid that any time lags in the cross-correlation are due to different
                # ways of resampling the power band and dff signal, we downsample
                # the DFF signal in the exactly the same way as the "windowing" for the 
                # EEG spectrogram calculation: 
                # We use the same window size, the same amount of overlap to calculate
                # the average DFF activity for each time point.
        
                dffd -= dffd.mean()

                if mode != 'cross':
                    pow_band = dffd
    
                m = np.min([pow_band.shape[0], dffd.shape[0]])
                # Say we correlate x and y;
                # x and y have length m
                # then the correlation vector cc will have length 2*m - 1
                # the center element with lag 0 will be cc[m-1]
                norm = np.nanstd(dffd[0::]) * np.nanstd(pow_band[0::])
                # for used normalization, see: https://en.wikipedia.org/wiki/Cross-correlation
                
                #xx = scipy.signal.correlate(dffd[1:m], pow_band[0:m - 1])/ norm
                xx = (1/m) * scipy.signal.correlate(dffd[0::], pow_band[0::])/ norm
                ii = np.arange(len(xx) / 2 - iwin, len(xx) / 2 + iwin + 1)
                ii = [int(i) for i in ii]
                
                ii = np.concatenate((np.arange(m-iwin-1, m), np.arange(m, m+iwin, dtype='int')))
                # note: point ii[iwin] is the "0", so xx[ii[iwin]] corresponds to the 0-lag correlation point
                #CC.append(xx[ii])

                #pdb.set_trace()
                CC += zip([idf]*len(ii), [rec]*len(ii), [ROI_ID]*len(ii), [typ]*len(ii),
                          cc_time, xx[ii], [seq_num]*len(ii))
        
            seq_num += 1
    df = pd.DataFrame(data=CC, columns=['mouse', 'recording', 'ID', 'Type', 'time', 'cc', 'seq'])
    df = df.groupby(['mouse', 'ID', 'Type', 'time']).mean().reset_index()
    df = df[['mouse', 'ID', 'Type', 'time', 'cc']]
    
    return df



def bandpass_avgcorr_state(ipath, roi_mapping, band, fft_win=2.5, perc_overlap=0.8, win=120, state=3, 
                        tbreak=0, pzscore=True, ma_thr=10, ma_rem_exception=True, pnorm_spec=True,
                        mode='cross'):
    """
    Calculate the cross-correlation for each type (class) of ROIs per mouse.
    That is, for each recording and class average the DF/F activity and then calculate
    the cross-correlation.

    See also banpass_corr_state    

    Parameters
    ----------
    ipath : TYPE
        DESCRIPTION.
    roi_mapping : TYPE
        DESCRIPTION.
    band : TYPE
        DESCRIPTION.
    fft_win : TYPE, optional
        DESCRIPTION. The default is 2.5.
    perc_overlap : TYPE, optional
        DESCRIPTION. The default is 0.8.
    win : TYPE, optional
        DESCRIPTION. The default is 120.
    state : TYPE, optional
        DESCRIPTION. The default is 3.
    tbreak : TYPE, optional
        DESCRIPTION. The default is 0.
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 10.
    ma_rem_exception : TYPE, optional
        DESCRIPTION. The default is True.
    pnorm_spec : TYPE, optional
        DESCRIPTION. The default is True.
    mode : TYPE, optional
        DESCRIPTION. The default is 'cross'.

    Returns
    -------
    None.

    """
    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]    
    
    CC = []
    for rec in recordings:
        print(rec)
        idf = re.split('_', rec)[0]
        sr = sleepy.get_snr(ipath, rec)
        nbin = int(np.round(sr)*2.5)
        sdt = nbin * (1.0/sr)
        
        
        rec_map = roi_mapping[roi_mapping[rec] != 'X']   
        if sum(roi_mapping[rec] == 'X') == len(roi_mapping):
            print('Recording %s: no ROI present' % rec)
            continue

        roi_list = int(re.split('-', rec_map.iloc[0][rec])[0])
        types = rec_map.Type.unique()
        types.sort()
        print(types)

        nwin = int(np.round(sr * fft_win))
        if nwin % 2 == 1:
            nwin += 1
        noverlap = int(nwin*perc_overlap)
    
        EEG = so.loadmat(os.path.join(ipath, rec, 'EEG.mat'), squeeze_me=True)['EEG']
        bs_time = np.arange(0, EEG.shape[0])*(1/sr)
        
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
        
        if ma_thr>0:
            seq = sleepy.get_sequences(np.where(M==2)[0])
            for s in seq:
                if np.round(len(s)*sdt) <= ma_thr:
                    if ma_rem_exception:
                        if (s[0]>1) and (M[s[0] - 1] != 1):
                            M[s] = 3
                    else:
                        M[s] = 3
        
        img_time = imaging_timing(ipath, rec)
        
        # get all bouts of state $state
        seq = sleepy.get_sequences(np.where(M == state)[0], ibreak=int(tbreak/2.5)+1)
        # as the cross-correlation should range from -$win to $win, each bout
        # needs to be at least 2 * $win seconds long
        seq = [s for s in seq if len(s)*2.5 > 2*win]
        

        roi_dict = {typ:[] for typ in types}
        ######################################################################
        # go through all ROIs in recording rec
        for typ in types:
            dfs = rec_map[rec_map.Type == typ]
            roi_ids = []
            for index, row in dfs.iterrows():
                s=row[rec]
                roi_num = int(re.split('-', s)[1])
                roi_ids.append(roi_num)

        roi_dict[typ] = roi_ids

        nrois = 0
        for typ in roi_dict:
            nrois += len(roi_dict[typ])

        seq_num = 0
        for s in seq:
            i = s[0] * nbin
            j = s[-1] * nbin + 1
            EEGcut = EEG[i:j]
            bs_time_cut = bs_time[i:j]            

            f, t, Pow = scipy.signal.spectrogram(EEGcut, nperseg=nwin, noverlap=noverlap, fs=sr)
            if pnorm_spec:
                sp_mean = Pow.mean(axis=1)
                Pow = np.divide(Pow, np.tile(sp_mean, (Pow.shape[1], 1)).T)
    
            ifreq = np.where((f >= band[0]) & (f <= band[1]))[0]
            dt = t[1] - t[0]
            iwin = int(win / dt)
            cc_time = np.arange(-iwin, iwin+1) * dt

    
            if mode == 'cross':
                if not pnorm_spec:
                    pow_band = Pow[ifreq, :].sum(axis=0) * (f[1]-f[0])
                else:
                    pow_band = Pow[ifreq, :].mean(axis=0)
    
                pow_band -= pow_band.mean()

            DFFd = downsample_overlap(DFF, bs_time_cut, img_time, nwin, noverlap)

            if pzscore:
                for i in range(DFFd.shape[1]):
                    DFFd[:,i] = (DFFd[:,i] - DFFd[:,i].mean()) / DFFd[:,i].std()
            
            for typ in types:
                roi_ids = roi_dict[typ]

                dffd = DFFd[:,roi_ids].mean(axis=1)
    
                # to avoid that any time lags in the cross-correlation are due to different
                # ways of resampling the power band and dff signal, we downsample
                # the DFF signal in the exactly the same way as the "windowing" for the 
                # EEG spectrogram calculation: 
                # We use the same window size, the same amount of overlap to calculate
                # the average DFF activity for each time point.
        
                dffd -= dffd.mean()

                if mode != 'cross':
                    pow_band = dffd
    
                m = np.min([pow_band.shape[0], dffd.shape[0]])
                # Say we correlate x and y;
                # x and y have length m
                # then the correlation vector cc will have length 2*m - 1
                # the center element with lag 0 will be cc[m-1]
                norm = np.nanstd(dffd[0::]) * np.nanstd(pow_band[0::])
                # for used normalization, see: https://en.wikipedia.org/wiki/Cross-correlation
                
                #xx = scipy.signal.correlate(dffd[1:m], pow_band[0:m - 1])/ norm
                xx = (1/m) * scipy.signal.correlate(dffd[0::], pow_band[0::])/ norm
                ii = np.arange(len(xx) / 2 - iwin, len(xx) / 2 + iwin + 1)
                ii = [int(i) for i in ii]
                
                ii = np.concatenate((np.arange(m-iwin-1, m), np.arange(m, m+iwin, dtype='int')))
                # note: point ii[iwin] is the "0", so xx[ii[iwin]] corresponds to the 0-lag correlation point
                #CC.append(xx[ii])

                #pdb.set_trace()
                CC += zip([idf]*len(ii), [rec]*len(ii), [typ]*len(ii),
                          cc_time, xx[ii], [seq_num]*len(ii))

        
            seq_num += 1

    df = pd.DataFrame(data=CC, columns=['mouse', 'recording', 'Type', 'time', 'cc', 'seq'])
    df = df.groupby(['mouse', 'Type', 'time']).mean().reset_index()
    df = df[['mouse', 'Type', 'time', 'cc']]
    return df
     


def downsample_overlap(x, tcut, img_time, nwin, noverlap):
    """
    downsample DF/F activity the same way as done for EEG spectrogram
    calculating using shifting, overlapping windows; the "difficulty" is
    that EEG and imaging time are not perfectly aligned with each other.

    Parameters
    ----------
    x : np.array
        complete imaging vector of size (n,)
    tcut : np.array
        time interval (absolute time of each time step) for downsampling
    img_time : np.array
        vector specifying time point of each imaging frame
    nwin : int
        size of window for consecutive averaging
    noverlap : int
        number of time points overlapping within two consecutive windows

    Returns
    -------
    np.array
        downsampled imaging vector

    """

    nsubwin = nwin-noverlap
    
    n = len(tcut)
    n_down = int(np.floor((n-noverlap)/nsubwin))
    x_down = np.zeros((n_down,x.shape[1]))

    j = 0
    for i in range(0, n-nwin+1, nsubwin):
        twin = tcut[i:i+nwin]
        idx = eeg2img_time(twin[[0, -1]], img_time)
        x_down[j,:] = x[idx[0]:idx[-1]+1,:].mean(axis=0)
        j += 1

    return np.array(x_down)
    


def irem_corr(ipath, roi_mapping, pzscore=True, ma_thr=10, ma_rem_exception=True):
    """
    collect data to correlate REM_pre with NREM and Wake duration during the
    following inter-REM interval as well as the DF/F activity for all of these
    epochs. 

    Parameters
    ----------
    ipath : TYPE
        Imaging base folder.
    roi_mapping : TYPE
        pd.DataFrame specificing for each ROI ('ID') the recording in which it was detected.
    ma_thr : TYPE, fload
        microarousal [MA] threshold. The default is 10.
    ma_rem_exception : TYPE, bol
        If true, don't set wake periods following REM as MA. The default is True.

    Returns
    -------
    pd.DataFrame with columns
    ['mouse', 'recording', 'rem_pre', 'rem_post', 'dur_irem', 'dur_inrem', 
     'dur_iwake', 'dff_pre', 'dff_post', 'dff_irem', 'dff_inrem', 'dff_iwake']

    """
    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]   
    
    data = []
    for rec in recordings:
        print(rec)
        idf = re.split('_', rec)[0]
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

        DFF = downsample_dff2bs(ipath, rec, roi_list)
        # brainstate
        M = sleepy.load_stateidx(ipath, rec)[0]
        # flatten out microarousals
        if ma_thr > 0:
            seq = sleepy.get_sequences(np.where(M == 2)[0])
            for s in seq:
                if np.round(len(s)*sdt) < ma_thr:
                    #if (s[0] > 1) and (M[s[0] - 1] != 1):
                    M[s] = 3
        
        ######################################################################
        # go through all ROIs in recording rec
        for index, row in rec_map.iterrows():
            s=row[rec]
            roi_num = int(re.split('-', s)[1])
            dff = DFF[:,roi_num]
                
            if pzscore:
                dff = (dff - dff.mean()) / dff.std()
                
            
            seq = sleepy.get_sequences(np.where(M==1)[0])
            if len(seq) >= 2:
                for (si, sj) in zip(seq[:-1], seq[1:]):
                    # indices of inter-REM period                                
                    idx = range(si[-1]+1,sj[0])
    
                m_cut = M[idx]                
                dff_cut = dff[idx]
                
                rem_pre = len(si)*sdt
                rem_post = len(sj)*sdt
                dur_irem = len(idx)*sdt
                dur_inrem = len(np.where(m_cut==3)[0])*sdt
                dur_iwake = len(np.where(m_cut==2)[0])*sdt
                
                dff_irem = np.mean(dff_cut).mean()
                dff_inrem = dff_cut[np.where(m_cut==3)[0]].mean()
                
                dff_iwake = dff_cut[np.where(m_cut==2)[0]].mean()
                dff_pre = dff[si].mean()
                dff_post = dff[sj].mean()

                data.append([idf, rec, rem_pre, rem_post, dur_irem, dur_inrem, dur_iwake, dff_pre, dff_post, dff_irem, dff_inrem, dff_iwake])
        
    df = pd.DataFrame(data=data, columns=['mouse', 'recording', 'rem_pre', 'rem_post', 'dur_irem', 'dur_inrem', 'dur_iwake', 'dff_pre', 'dff_post', 'dff_irem', 'dff_inrem', 'dff_iwake'])
        
    return df
        


def pearson_state_avgcorr(ipath, roi_mapping, bands, ma_thr=10, min_dur=20, pnorm_spec=True, pzscore=True):
    """
    For each recording in the ROI mapping, determine first the average activity
    for each neurons class and then correlated the average activty with different
    frequency bands in the EEG powerspectrum
    """
    
    
    df_avg = plot_catraces_avgclasses(ipath, roi_mapping, pplot=False)[0]
    # columns: mouse      recording   type    time       dff  state
    recordings = df_avg.recording.unique()
    data = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
        dfs = df_avg[df_avg.recording == rec]
        dfs = dfs.groupby(['mouse', 'type', 'time', 'state']).mean().reset_index()
    
        # load EEG spectrogram
        P = so.loadmat(os.path.join(ipath, rec, 'sp_%s.mat' % rec), squeeze_me=True)
        SP = P['SP']
        freq = P['freq']
        df = freq[1] - freq[0]

        M = sleepy.load_stateidx(ipath, rec)[0]

        if pnorm_spec:
            sp_mean = SP.mean(axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)

    
        for b in bands:
            
            band = bands[b]            
            ifreq = np.where((freq >= band[0]) & (freq <= band[1]))[0]
            if pnorm_spec:
                pow_band = SP[ifreq,:].mean(axis=0)
            else:
                pow_band = SP[ifreq,:].sum(axis=0)*df

            print(rec)
            types = dfs.type.unique()
            for typ in types:
                print(typ)
                dff = dfs[dfs.type==typ]['dff']
                dff = np.array(dff)
                for s in [1,2,3]:
                    idx = np.where(M==s)[0]
                    r,p = scipy.stats.pearsonr(dff[idx], pow_band[idx])
                    
                    data+=[[idf, rec, b, typ, s, r, p]]
                    
    df = pd.DataFrame(data=data, columns=['mouse', 'recording', 'band', 'type', 'state', 'r', 'p'])   
    return df
            


def find_capeaks(DFF, prom=0.8, f_cutoff=1.0, isr=20):
    """
    

    Parameters
    ----------
    DFF : TYPE
        DESCRIPTION.
    prom : TYPE, optional
        DESCRIPTION. The default is 0.8.
    f_cutoff : TYPE, optional
        DESCRIPTION. The default is 1.0.
    isr : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    Peaks : dict
        dict: ROI --> indices of imaging frames with ca peaks.
    DFF : TYPE
        2D np.array with lowpass filtered calcium signals.

    """
    nroi = DFF.shape[1]        
    Peaks = np.zeros(DFF.shape)
    w0 = f_cutoff / (isr*0.5)
    
    Peaks = {}
    for i in range(nroi):
        dff = DFF[:,i]
        
        dff = sleepy.my_lpfilter(dff, w0)
        DFF[:,i] = dff
        
        prom = np.percentile(dff, 80)
        idx = scipy.signal.find_peaks(dff, prominence=prom)[0]
                
        Peaks[i] = idx

    return Peaks, DFF



def peak_avgs(ipath, roi_mapping,ma_thr=10, ma_rem_exception=True):
    """
    Detect peaks in the calcium signal and calculate for each ROI the mean frequency 
    of speaks during each brain state (quantified as peaks per second)

    Parameters
    ----------
    ipath : string
        imaging folder.
    roi_mapping : pd.DataFrame
        ROI mapping.
    ma_thr : float, optional
        MA threshold. The default is 10.
    ma_rem_exception : float, optional
        If True, MAs following REM stay as "Wake". The default is True.

    Returns
    -------
    df : pd.DataFrame
        with columns ['mouse', 'recording', 'ID', 'ROI', 'Type', 'peaks', 'state'].
        'peaks' measure the average frequency of ca-peaks per second

    """
    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]    
    state_map = {1:'REM', 2:'Wake', 3:'NREM'}
    
    # dict: type -> ID -> peak spectrograms
    data = []
    for rec in recordings:
        print(rec)
        idf = re.split('_', rec)[0]
        sr = sleepy.get_snr(ipath, rec)
        nbin = int(np.round(sr)*2.5)
        sdt = nbin * (1.0/sr)
                
        rec_map = roi_mapping[roi_mapping[rec] != 'X']   
        if sum(roi_mapping[rec] == 'X') == len(roi_mapping):
            print('Recording %s: no ROI present' % rec)
            continue

        roi_list = int(re.split('-', rec_map.iloc[0][rec])[0])
        types = rec_map.Type.unique()
        types.sort()

        # load DF/F for recording rec 
        dff_file = os.path.join(ipath, rec, 'recording_' + rec + '_dffn' + str(roi_list) + '.mat')
        if not(os.path.isfile(dff_file)):
            calculate_dff(ipath, rec, roi_list)
        DFF = so.loadmat(dff_file, squeeze_me=True)['dff']
        # brainstate
        M = sleepy.load_stateidx(ipath, rec)[0]
        
        if ma_thr>0:
            seq = sleepy.get_sequences(np.where(M==2)[0])
            for s in seq:
                if np.round(len(s)*sdt) <= ma_thr:
                    if ma_rem_exception:
                        if (s[0]>1) and (M[s[0] - 1] != 1):
                            M[s] = 3
                    else:
                        M[s] = 3
        
        # get all peaks in DF/F signal
        # nroi = DFF.shape[1]        
        # Peaks = np.zeros(DFF.shape)
        # for i in range(nroi):
        #     dff = DFF[:,i]
        #     dff = sleepy.my_lpfilter(dff, 0.1)
        #     DFF[:,i] = dff
            
        #     prom = np.percentile(dff, 80)
        #     idx = scipy.signal.find_peaks(dff, prominence=prom)[0]
                    
        #     Peaks[idx,i] = 1
        Peaks = np.zeros(DFF.shape)
        peak_idx, DFF = find_capeaks(DFF)
        for i in peak_idx:
            idx = peak_idx[i]
            Peaks[idx,i] = 1
        
        
        dffd = downsample_dff2bs(ipath, rec, roi_list, peaks=True, dff_data=Peaks)
    
    
        for index, row in rec_map.iterrows():
            s=row[rec]
            ID = row['ID']
            roi_num = int(re.split('-', s)[1])
            typ = row['Type']
            
            for state in [1,2,3]:
                state_idx = np.where(M==state)[0]
                
                mstate = dffd[state_idx,roi_num].mean()
                
                data += [[idf, rec, ID, s, typ, mstate, state_map[state]]]
                
    df = pd.DataFrame(data=data, columns=['mouse', 'recording', 'ID', 'ROI', 'Type', 'peaks', 'state'])
    return df
                
            
    
def peak_correlation(ipath, roi_mapping, win=10, ma_thr=10, ma_rem_exception=True, 
                     pnorm_spec=True, fmax=(0,100), states=[1,2,3], spec_type='sp_mat'):
    """
    calculate ca-peak triggered EEG spectrogram for all given ROIs.

    Parameters
    ----------
    ipath : TYPE
        DESCRIPTION.
    roi_mapping : TYPE
        DESCRIPTION.
    win : int, optional
        The ca-peak triggered spectrogram ranges from -win to +win seconds centered on each peak. 
        The default is 10.
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 10.
    ma_rem_exception : TYPE, optional
        DESCRIPTION. The default is True.
    pnorm_spec : TYPE, optional
        DESCRIPTION. The default is True.
    fmax : tuple, optional
        Frequency range for ca-peak triggered spectrogram. The default is (0,100).
    states : list, optional
        Only calculat the ca-peak triggered spectrogram for ca peaks occuring during 
        the brain states specified in @stages; 1-REM, 2-Wake, 3-NREM. The default is [1,2,3].
    spec_type : string, optional
        Values: sp_mat or recalc. If 'sp_mat', use the existing EEG spectrogram
        saved in sp_*.mat; otherwise recalculate the EEG spectrogram using a refined
        frequency axis. The default is 'sp_mat'.

    Returns
    -------
    roi_spec : dict
        dict: ROI Type --> 2D array (ROI ID x frequency x time).
    time : TYPE
        np.array; time axis of the ca peak-triggered spectrogram.
    TYPE
        np.array; frequency axis of the ca peak-triggered spectrogram.

    """
    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]    
    
    # dict: type -> ID -> peak spectrograms
    roi_spec = {}
    for rec in recordings:
        print(rec)
        #idf = re.split('_', rec)[0]
        sr = sleepy.get_snr(ipath, rec)
        nbin = int(np.round(sr)*2.5)
        sdt = nbin * (1.0/sr)
                
        rec_map = roi_mapping[roi_mapping[rec] != 'X']   
        if sum(roi_mapping[rec] == 'X') == len(roi_mapping):
            print('Recording %s: no ROI present' % rec)
            continue

        roi_list = int(re.split('-', rec_map.iloc[0][rec])[0])
        types = rec_map.Type.unique()
        types.sort()

        # load DF/F for recording rec 
        dff_file = os.path.join(ipath, rec, 'recording_' + rec + '_dffn' + str(roi_list) + '.mat')
        if not(os.path.isfile(dff_file)):
            calculate_dff(ipath, rec, roi_list)
        DFF = so.loadmat(dff_file, squeeze_me=True)['dff']
        # brainstate
        M = sleepy.load_stateidx(ipath, rec)[0]
        
        if ma_thr>0:
            seq = sleepy.get_sequences(np.where(M==2)[0])
            for s in seq:
                if np.round(len(s)*sdt) <= ma_thr:
                    if ma_rem_exception:
                        if (s[0]>1) and (M[s[0] - 1] != 1):
                            M[s] = 3
                    else:
                        M[s] = 3
        
        img_time = imaging_timing(ipath, rec)
        
        # get all peaks in DF/F signal
        nroi = DFF.shape[1]        
        Peaks = {}
        for i in range(nroi):
            dff = DFF[:,i]
            dff = sleepy.my_lpfilter(dff, 0.05)
            DFF[:,i] = dff
            prom = np.percentile(dff, 80)
            idx = scipy.signal.find_peaks(dff, prominence=prom)[0]
            Peaks[i] = idx

        # load spectrogram
        # load EEG and EMG spectrogram
        if spec_type == 'sp_mat':
            P = so.loadmat(os.path.join(ipath, rec, 'sp_%s.mat' % rec), squeeze_me=True)
            SPEEG = P['SP'] 
            if pnorm_spec == True:
                sp_mean = SPEEG.mean(axis=1)
                SPEEG = np.divide(SPEEG, np.tile(sp_mean, (SPEEG.shape[1], 1)).T)
            freq = P['freq']
            ifreq = np.where((freq>=fmax[0]) & (freq <= fmax[1]))[0]
            iwin = int(win/sdt)
            time = np.arange(-iwin, iwin+1)*sdt

        else:
            nsr_seg = 4
            perc_overlap = 0.8
            nfilt = [3,2]
            EEG = so.loadmat(os.path.join(ipath, rec, 'EEG.mat'), squeeze_me=True)['EEG']
            freq, tspec, SPEEG = scipy.signal.spectrogram(EEG, fs=sr, window='hanning', nperseg=int(nsr_seg * sr), noverlap=int(nsr_seg * sr * perc_overlap))
            filt = np.ones(nfilt)
            filt = filt / filt.sum()
            SPEEG = scipy.signal.convolve2d(SPEEG, filt, boundary='symm', mode='same')

            if pnorm_spec == True:
                sp_mean = SPEEG.mean(axis=1)
                SPEEG = np.divide(SPEEG, np.tile(sp_mean, (SPEEG.shape[1], 1)).T)

            dt = tspec[2]-tspec[1]
            iwin = int(win/dt)
            time = np.arange(-iwin, iwin+1)*dt
            ifreq = np.where((freq>=fmax[0]) & (freq <= fmax[1]))[0]

        ######################################################################
        # collect all ROIs for each ROI Type
        roi_dict = {typ:[] for typ in types}
        # go through all ROIs in recording rec
        for typ in types:
            dfs = rec_map[rec_map.Type == typ]
            roi_ids = []
            for index, row in dfs.iterrows():
                s=row[rec]
                ID = row['ID']
                roi_num = int(re.split('-', s)[1])
                roi_ids.append((roi_num,ID))                
            roi_dict[typ] = roi_ids        
        
        # dict: typ -> 3D array (rois x frequency x time)
        for typ in types:
            for (roi_j,ID) in roi_dict[typ]:

                curr_roi_spec = []
                for i in Peaks[roi_j]:
                    ibs = int(np.round(img_time[i]/sdt))
                    
                    if spec_type == 'sp_mat':
                        ispec = ibs
                    else:
                        ispec = np.argmin(np.abs(tspec - img_time[i]))
                    
                    if not (M[ibs] in states):
                        continue
                    
                    if ispec>=iwin and ispec < len(M)-iwin:
                        tmp = SPEEG[ifreq, ispec-iwin:ispec+iwin+1]
                        curr_roi_spec.append(tmp)

                # now we went through all peaks for roi_j
                if not typ in roi_spec:
                    roi_spec[typ] = {}
                if not ID in roi_spec[typ]:
                    roi_spec[typ][ID] = []
                roi_spec[typ][ID] += curr_roi_spec
                    
    for typ in roi_spec:
        for ID in roi_spec[typ]:
            tmp = roi_spec[typ][ID]
            #pdb.set_trace()
            if len(tmp) > 0:
                roi_spec[typ][ID] = np.array(tmp).mean(axis=0)
    
    for typ in roi_spec:
        tmp = list(roi_spec[typ].values())
        val = [v for v in tmp if len(v)>0]
        roi_spec[typ] = np.array(val)
    
    return roi_spec, time, freq[ifreq]



def detect_spindles(ppath, name, M=[], pplot=False, std_thr=1.5, sigma=[7,15]):
    """
    Detect spindles during NREM (only) using the algorithm described in Niethard et al. 2018.
    

    Parameters
    ----------
    ppath : string
        Imaging base folder.
    name : string
        Name of recording.
    M : np.array as returned by sleepy.load_stateidx[0], optional
        Brain state sequence. If M == [], reload brain state sequence in function. The default is [].
    pplot : bool, optional
        If True, plot figures labeling detected spindles in raw EEG along with hynogram. The default is False.

    Returns
    -------
    spindle : list
        for each spindle the list contains a list with time point of 
        [spindle_onset, spindle_center, spindle_offset].
    t : np.array
        time vector of raw EEG.

    """
    from scipy.signal import hilbert

    EEG = so.loadmat(os.path.join(ppath, name, 'EEG.mat'), squeeze_me=True)['EEG']
    sr = sleepy.get_snr(ppath, name)
    nbin = 2500
    
    # before 10 to 16
    w0 = sigma[0] / (sr/2)
    w1 = sigma[1] / (sr/2)
    
    EEGbp = sleepy.my_bpfilter(EEG, w0, w1)
    res = hilbert(EEGbp)

    amplitude_envelope = np.abs(res)

    if len(M) == 0:
        M = sleepy.load_stateidx(ppath, name)[0]
    
    seq = sleepy.get_sequences(np.where(M==3)[0])
    
    nrem_idx = []
    for s in seq:
        si = s[0]
        sj = s[-1]
        
        nrem_idx += list(range(si*nbin, sj*nbin))
        
        
    std = np.std(amplitude_envelope[nrem_idx])
    thr = std * std_thr
    
    seq = sleepy.get_sequences(np.where(amplitude_envelope > thr)[0])
    
    spindle = []
    spindle_idx = []
    for s in seq:
        if 0.5 <= len(s)*(1/sr) <= 3:
            onset = s[0]
            ctr = (s[-1] - s[0]) * 0.5 + s[0]
            offset = s[-1]

            if M[int(onset/nbin)] == 3:
                spindle.append([onset, ctr, offset])   
                spindle_idx += list(range(s[0], s[-1]+1))
            
    ctr = [int(s[1]) for s in spindle]
    
    t = np.arange(0, EEG.shape[0])*(1/sr)

    if pplot:
        plt.figure()
        plt.title(name)
    
        st = np.arange(0, len(M)) * (nbin*(1/sr))   
        ax = plt.subplot(211)
        plt.plot(st, M)
        
        ax = plt.subplot(212, sharex=ax)
        plt.plot(t, EEG)
        plt.plot(t, EEGbp)
        plt.plot(t, amplitude_envelope)    
        plt.plot(t[spindle_idx], amplitude_envelope[spindle_idx], 'r.')
    
    return spindle, t



def spindle_correlation(ipath, roi_mapping, ma_thr=10, ma_rem_exception=True, 
                        pzscore=True, pre=3, post=5, xdt=0.1, std_thr=1.5, pplot_spindles=False):
    """
    Correlate the onset, center (midpoint) and offset of sleep spindles with activity of single ROIs. 
    Spindles are detected using the algorithm described in Niethard et al. 2021, J Neurosc.

    Parameters
    ----------
    ipath : string
        DESCRIPTION.
    roi_mapping : pd.DataFrame
        ROI mapping.
    ma_thr : float, optional
        Microarousal threshold. The default is 10.
    ma_rem_exception : bool, optional
        If True, leave wake perids < $ma_thr as wake if they directly follow REM. The default is True.
    pzscore : bool, optional
        If True, . The default is True.
    pre : float, optional
        Time [in seconds] before (onset, center, offset) of spindle. The default is 3.
    post : float, optional
        Time [in seconds] before (onset, center, offset) of spindle. The default is 5.
    xdt : float, optional
        Temporal resultion of plotted results. 
        If xdt == 0, plot a fixed number of imaging frames before (pre / (1/20)) and after spindle (post / (1/20));
        assuming that the imaging sampling rate is 20 Hz.
        The default is 0.1.
        
    std_thr : fload, optional
        Parameter for the threshold for spindle detection. The default is 1.5.
    pplot_spindles : bool, optional
        If True, plot for each recording in $roi_mapping a plot showing the raw EEG along with labeled spindles. 
        The default is False.

    Returns
    -------
    dfm : pd.DataFrame
        ... with columns 
        ['mouse' ,'recording', 'Type', 'ID', 'ROI', 'dff_ctr', 'dff_onset', 'dff_offset' 'time'].
        'dff_onset' is the spindle triggered DF/F activity relative to the onset of each spindle.
        'dff_offset' is the activity relative to the offset of each spindle.
        'dff_center' is the activity relative to the center midpoint of each spindle
    """
    IFR = 20
    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]    
    
    # dict: type -> ID -> peak spectrograms
    data = []
    for rec in recordings:
        print(rec)
        img_time = imaging_timing(ipath, rec)
        idt = np.diff(img_time).mean()
        if xdt == 0:
            idt = 1/IFR
        ipre = int(pre/idt)
        ipost = int(post/idt)


        idf = re.split('_', rec)[0]
        sr = sleepy.get_snr(ipath, rec)
        nbin = int(np.round(sr)*2.5)
        sdt = nbin * (1.0/sr)
                
        rec_map = roi_mapping[roi_mapping[rec] != 'X']   
        if sum(roi_mapping[rec] == 'X') == len(roi_mapping):
            print('Recording %s: no ROI present' % rec)
            continue

        roi_list = int(re.split('-', rec_map.iloc[0][rec])[0])
        types = rec_map.Type.unique()
        types.sort()

        # load DF/F for recording rec 
        dff_file = os.path.join(ipath, rec, 'recording_' + rec + '_dffn' + str(roi_list) + '.mat')
        if not(os.path.isfile(dff_file)):
            calculate_dff(ipath, rec, roi_list)
        DFF = so.loadmat(dff_file, squeeze_me=True)['dff']
        # brainstate
        M = sleepy.load_stateidx(ipath, rec)[0]
        
        if ma_thr>0:
            seq = sleepy.get_sequences(np.where(M==2)[0])
            for s in seq:
                if np.round(len(s)*sdt) <= ma_thr:
                    if ma_rem_exception:
                        if (s[0]>1) and (M[s[0] - 1] != 1):
                            M[s] = 3
                    else:
                        M[s] = 3
        
        spindles, t_eeg = detect_spindles(ipath, rec, M=M, pplot=pplot_spindles, std_thr=std_thr)
        spindles_ctr    = [int(s[1]) for s in spindles]
        spindles_onset  = [int(s[0]) for s in spindles]
        spindles_offset = [int(s[2]) for s in spindles]
        
        ######################################################################
        # collect all ROIs for each ROI Type
        roi_dict = {typ:[] for typ in types}
        # go through all ROIs in recording rec
        for typ in types:
            dfs = rec_map[rec_map.Type == typ]
            roi_ids = []
            for index, row in dfs.iterrows():
                s=row[rec]
                ID = row['ID']
                roi_num = int(re.split('-', s)[1])
                roi_list = int(re.split('-', s)[0])
                roi_ids.append((roi_num,ID))                
            roi_dict[typ] = roi_ids        
        
        # dict: typ -> 3D array (rois x frequency x time)
        for typ in types:
            for (roi_j,ID) in roi_dict[typ]:

                dff = DFF[:,roi_j]
                if pzscore:
                    dff = (dff-dff.mean()) / dff.std()
                else:
                    dff *= 100

                for ctr, onset, offset in zip(spindles_ctr, spindles_onset, spindles_offset):

                    if xdt == 0:                    
                        onset = eeg2img_time([t_eeg[onset]], img_time)[0]
                        dff_onset = dff[onset-ipre:onset+ipost]

                        ctr = eeg2img_time([t_eeg[ctr]], img_time)[0]                        
                        dff_ctr =   dff[ctr-ipre:ctr+ipost]
                        
                        t_spindle = np.arange(-ipre, ipost) * idt

                        m = len(dff_ctr)
                        data += zip([idf]*m, [rec]*m, [typ]*m, [ID]*m, [str(roi_list)+'-'+str(roi_num)]*m, dff_ctr, dff_onset, t_spindle)
                        
                    else:                        
                        # onset
                        ti = t_eeg[onset]
                        tj = t_eeg[ctr]
                        tk = t_eeg[offset]
                        if ti-ipre > 0 and tk+ipost < img_time[-1]:

                            idx_pre = eeg2img_time([ti-pre, ti], img_time)                        
                            dff_pre  = time_morph(dff[idx_pre[0] : idx_pre[1]], int(pre/xdt))
                            
                            idx_post = eeg2img_time([ti, ti+post], img_time)                        
                            dff_post = time_morph(dff[idx_post[0] : idx_post[1]], int(post/xdt))                             
                            dff_onset = np.concatenate((dff_pre, dff_post))
                        
                            # center of spindle
                            idx_pre = eeg2img_time([tj-pre, tj], img_time)                        
                            dff_pre  = time_morph(dff[idx_pre[0] : idx_pre[1]], int(pre/xdt))
                            
                            idx_post = eeg2img_time([tj, tj+post], img_time)                        
                            dff_post = time_morph(dff[idx_post[0] : idx_post[1]], int(post/xdt))                             
                            dff_ctr = np.concatenate((dff_pre, dff_post))

                            # offset of spindle
                            idx_pre = eeg2img_time([tk-pre, tk], img_time)                        
                            dff_pre  = time_morph(dff[idx_pre[0] : idx_pre[1]], int(pre/xdt))
                            
                            idx_post = eeg2img_time([tk, tk+post], img_time)                        
                            dff_post = time_morph(dff[idx_post[0] : idx_post[1]], int(post/xdt))                             
                            dff_offset = np.concatenate((dff_pre, dff_post))


                            t_spindle = np.arange(-int(pre/xdt), int(post/xdt))*xdt
                                                    
                            
                            m = len(dff_onset)
                            data += zip([idf]*m, [rec]*m, [typ]*m, [ID]*m, [str(roi_list)+'-'+str(roi_num)]*m, dff_ctr, dff_onset, dff_offset, t_spindle)
                                                                        
    df = pd.DataFrame(data=data, columns=['mouse' ,'recording', 'Type', 'ID', 'ROI', 'dff_ctr', 'dff_onset', 'dff_offset', 'time'])        
    # average across ROIs
    dfm = df.groupby(['mouse', 'Type', 'ID', 'time']).mean().reset_index()
    return dfm



def phrem_correlation(ipath, roi_mapping, pre, post, xdt=0.1, pzscore=True, 
                      ma_thr=10, ma_rem_exception=False, pfilt=False, f_cutoff=2.0, 
                      local_mean='pre', roi_avg=True, eeg_spec=False, dff_var='dff'):
    """
    Calculate phasic REM-triggered for all ROIs

    Parameters
    ----------
    ipath : string
        base folder.
    roi_mapping : pd.DataFrame
        ROI mapping
    pre : float
        Time [in seconds] before phREM onset.
    post : float
        Time [in seconds] after phREM onset.
    xdt : float, optional
        Binning of DF/F signals [in seconds]. The default is 0.1.
    ma_thr : float, optional
        Microarousal threshold. The default is 10.
    pzscore : bool, optional
        If true, z-score DF/F signals (across whole recording session). The default is True.
    ma_rem_exception : bool, optional
        If true, MAs after REM, stay Wake. The default is False.
    pfilt: bool, optional
        if True, lowpass filter DF/F signal. 
    f_cutoff: float, optional
        Cutoff frequency for (optional) lowpass filter for DF/F signal
    local_mean : string, optional
        Two options: 'pre' or 'prepost'. 
        If 'pre', substract from each phREM-triggered DF/F signals, the mean calculated
        over the time window from -pre to 0s (= onset of phREM).
        If 'prepost', subtract mean calculated for window from -pre to post.
        The default is 'pre'.
    roi_avg : bool, optional
        If true, calculate for each ROI, average across all phREM events. The default is True.

    Returns
    -------
    df : pd.DataFrame
        with columns ['mouse' ,'recording', 'Type', 'ID', 'ROI', 'dff_ctr', 'dff_onset', 'dff_offset', 'time', 'phrem_ID'].
        
    df_phrem: pd.DataFrame
        average DF/F activity during phasic REM and during baseline interval of equal duration preceding the phasic REM episode
        columns ['mouse' ,'recording', 'Type', 'ID', 'dff_pre', 'dff_post', 'phrem_ID']

    """
    pfilt = True
    f_cutoff = 2.0
    nsr_seg = 2
    fmax = 20
    IFR = 20
    w0 = f_cutoff / (IFR*0.5)

    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]    
    
    # dict: type -> ID -> peak spectrograms
    data = []
    phrem_ID = 0
    data_spec = []
    data_phrem = []
    for rec in recordings:
        print(rec)
        img_time = imaging_timing(ipath, rec)
        idt = np.diff(img_time).mean()
        if xdt == 0:
            idt = 1/IFR
        ipre = int(pre/idt)
        ipost = int(post/idt)


        idf = re.split('_', rec)[0]
        sr = sleepy.get_snr(ipath, rec)
        nbin = int(np.round(sr)*2.5)
        sdt = nbin * (1.0/sr)
                
        rec_map = roi_mapping[roi_mapping[rec] != 'X']   
        if sum(roi_mapping[rec] == 'X') == len(roi_mapping):
            print('Recording %s: no ROI present' % rec)
            continue

        roi_list = int(re.split('-', rec_map.iloc[0][rec])[0])
        types = rec_map.Type.unique()
        types.sort()

        # load DF/F for recording rec 
        dff_file = os.path.join(ipath, rec, 'recording_' + rec + '_dffn' + str(roi_list) + '.mat')
        if not(os.path.isfile(dff_file)):
            calculate_dff(ipath, rec, roi_list)
        DFF = so.loadmat(dff_file, squeeze_me=True)[dff_var]
        # brainstate
        M = sleepy.load_stateidx(ipath, rec)[0]
        
        if ma_thr>0:
            seq = sleepy.get_sequences(np.where(M==2)[0])
            for s in seq:
                if np.round(len(s)*sdt) <= ma_thr:
                    if ma_rem_exception:
                        if (s[0]>1) and (M[s[0] - 1] != 1):
                            M[s] = 3
                    else:
                        M[s] = 3
                        
        if eeg_spec:
            EEG = so.loadmat(os.path.join(ipath, rec, 'EEG.mat'), squeeze_me=True)['EEG']
        
        phrem = sleepy.phasic_rem(ipath, rec)
        phrem_list = []
        for p in phrem:
            ph_list = phrem[p]
            
            for ph in ph_list:
                onset = ph[0]*(1/sr)
                offset = ph[-1]*(1/sr)
                ctr = onset + (offset-onset)/2
                phrem_list.append([onset, ctr, offset])
        
        phrem_ctr = [p[1] for p in phrem_list]
        phrem_onset = [p[0] for p in phrem_list]
        phrem_offset = [p[-1] for p in phrem_list]
        
    
        ######################################################################
        # collect all ROIs for each ROI Type
        roi_dict = {typ:[] for typ in types}
        # go through all ROIs in recording rec
        for typ in types:
            dfs = rec_map[rec_map.Type == typ]
            roi_ids = []
            for index, row in dfs.iterrows():
                s=row[rec]
                ID = row['ID']
                roi_num = int(re.split('-', s)[1])
                roi_list = int(re.split('-', s)[0])
                roi_ids.append((roi_num,ID))                
            roi_dict[typ] = roi_ids        
        
        ######################################################################
        # calculate phasic REM events
        if eeg_spec:
            ipre_spec  = int(pre*sr  + (nsr_seg/2)*sr)
            ipost_spec = int(post*sr + (nsr_seg/2)*sr)
            
            for t_onset in phrem_onset:
        
                istart = int(t_onset*sr)
                freq_spec, t_spec, SP_spec = scipy.signal.spectrogram(EEG[istart-ipre_spec:istart+ipost_spec+1], fs=sr, 
                                                                      window='hann', nperseg=nsr_seg * int(sr), noverlap= int(0.95 * nsr_seg * int(sr)))                    
                ifreq = np.where(freq_spec <= fmax)[0]
                freq_spec = freq_spec[ifreq]
                SP_spec = SP_spec[ifreq,:]
            
                pxx = SP_spec.T.flatten()
                fvec_spec = np.tile(freq_spec, SP_spec.shape[1])
                tvec_spec = np.repeat(t_spec, SP_spec.shape[0])
                m = len(fvec_spec)
                data_spec += zip([idf]*m, [rec]*m,  fvec_spec, tvec_spec, pxx)
        
        ######################################################################
        
        
        # dict: typ -> 3D array (rois x frequency x time)
        for typ in types:
            for (roi_j,ID) in roi_dict[typ]:

                dff = DFF[:,roi_j]
                if pzscore:
                    dff = (dff-dff.mean()) / dff.std()
                else:
                    dff *= 100
                
                if pfilt:
                    dff = sleepy.my_lpfilter(dff, w0)

                for t_ctr, t_onset, t_offset in zip(phrem_ctr, phrem_onset, phrem_offset):
                    if xdt == 0:                    
                        onset = eeg2img_time([t_onset], img_time)[0]
                        dff_onset = dff[onset-ipre:onset+ipost]

                        ctr = eeg2img_time([t_ctr], img_time)[0]                        
                        dff_ctr =   dff[ctr-ipre:ctr+ipost]
                        
                        t_phrem = np.arange(-ipre, ipost) * idt

                        m = len(dff_ctr)
                        data += zip([idf]*m, [rec]*m, [typ]*m, [ID]*m, [str(roi_list)+'-'+str(roi_num)]*m, dff_ctr, dff_onset, t_phrem)
                        
                        
                    else:                        
                        # onset
                        ti = t_onset
                        tj = t_ctr
                        tk = t_offset
                        if ti-ipre > 0 and tk+ipost < img_time[-1]:
                            t_phrem = np.arange(-int(pre/xdt), int(post/xdt))*xdt   
                            
                            idx_pre = eeg2img_time([ti-pre, ti], img_time)                        
                            dff_pre  = time_morph(dff[idx_pre[0] : idx_pre[1]], int(pre/xdt))
                            
                            idx_post = eeg2img_time([ti, ti+post], img_time)                        
                            dff_post = time_morph(dff[idx_post[0] : idx_post[1]], int(post/xdt))                             
                            dff_onset = np.concatenate((dff_pre, dff_post))
                            if local_mean=='pre':
                                dff_onset = dff_onset - dff_onset[np.where(t_phrem<0)].mean() 
                            elif local_mean=='prepost':
                                dff_onset = dff_onset - dff_onset.mean() 
                        
                            # center of spindle
                            idx_pre = eeg2img_time([tj-pre, tj], img_time)                        
                            dff_pre  = time_morph(dff[idx_pre[0] : idx_pre[1]], int(pre/xdt))
                            
                            idx_post = eeg2img_time([tj, tj+post], img_time)                        
                            dff_post = time_morph(dff[idx_post[0] : idx_post[1]], int(post/xdt))                             
                            dff_ctr = np.concatenate((dff_pre, dff_post))
                            if local_mean=='pre':
                                dff_ctr -= dff_ctr[np.where(t_phrem<0)].mean()
                            elif local_mean=='prepost':
                                dff_ctr -= dff_ctr.mean()


                            # offset of spindle
                            idx_pre = eeg2img_time([tk-pre, tk], img_time)                        
                            dff_pre  = time_morph(dff[idx_pre[0] : idx_pre[1]], int(pre/xdt))
                            
                            idx_post = eeg2img_time([tk, tk+post], img_time)                        
                            dff_post = time_morph(dff[idx_post[0] : idx_post[1]], int(post/xdt))                             
                            dff_offset = np.concatenate((dff_pre, dff_post))
                            if local_mean=='pre':
                                dff_offset -= dff_offset[np.where(t_phrem<0)].mean()
                            elif local_mean == 'prepost':
                                dff_offset -= dff_offset.mean()
                            
                            m = len(dff_onset)
                            data += zip([idf]*m, [rec]*m, [typ]*m, [ID]*m, [str(roi_list)+'-'+str(roi_num)]*m, 
                                        dff_ctr, dff_onset, dff_offset, t_phrem, [phrem_ID]*m)
                            
                            
                            ph_dur = tk-ti
                            idx_pre = eeg2img_time([ti-ph_dur, ti], img_time)                        
                            dff_pre  = dff[idx_pre[0] : idx_pre[1]+1].mean()
                            
                            idx_post = eeg2img_time([ti, tk], img_time)                        
                            dff_post = dff[idx_post[0] : idx_post[1]+1].mean()

                            data_phrem += [[idf, rec, typ, ID, dff_pre,  'pre',  phrem_ID]]
                            data_phrem += [[idf, rec, typ, ID, dff_post, 'post', phrem_ID]]
                            data_phrem += [[idf, rec, typ, ID, dff_post - dff_pre, 'diff', phrem_ID]]
                            
                            
                            phrem_ID += 1
                                                                        
    df = pd.DataFrame(data=data, columns=['mouse' ,'recording', 'Type', 'ID', 'ROI', 'dff_ctr', 'dff_onset', 'dff_offset', 'time', 'phrem_ID'])        
    df_phrem = pd.DataFrame(data=data_phrem, columns=['mouse' ,'recording', 'Type', 'ID', 'dff', 'time', 'phrem_ID'])        
    
    # average across ROIs
    if roi_avg:
        df = df.groupby(['mouse', 'Type', 'ID', 'time']).mean().reset_index()
    
    df_spec = []
    if eeg_spec:
        df_spec = pd.DataFrame(data=data_spec, columns=['mouse', 'recording', 'freq', 'time', 'pow'])
    
    return df, df_spec, df_phrem



def downsample_dff2bs(ipath, rec, roi_list, psave=False, peaks=False, dff_data = []):

    sr = sleepy.get_snr(ipath, rec)
    nbin = int(np.round(sr)*2.5)
    sdt = nbin * (1.0/sr)

    P = so.loadmat(os.path.join(ipath, rec, 'sp_%s.mat' % rec), squeeze_me=True)
    bs_time = P['t'] + sdt
    bs_time = np.concatenate(([0], bs_time))

    dff_file = os.path.join(ipath, rec, 'recording_' + rec + '_dffn' + str(roi_list) + '.mat')
    
    # load DF/F for recording rec
    if len(dff_data) == 0:
        #dff_file = os.path.join(ipath, rec, 'recording_' + rec + '_dffn' + str(roi_list) + '.mat')
        if not(os.path.isfile(dff_file)):
            calculate_dff(ipath, rec, roi_list)
        DFF = so.loadmat(dff_file, squeeze_me=True)['dff']
    else:
        DFF = dff_data
    
    # load imaging timing
    img_time = imaging_timing(ipath, rec)

    # indices of the imaging frames that are closest to given time points
    idx = eeg2img_time(bs_time, img_time)    
    
    if not peaks:
        data = []
        for (a,b) in zip(idx[0:-1], idx[1:]):
            data.append(DFF[a:b,:].mean(axis=0))        
        dffd = np.array(data)
    
        if psave:
            so.savemat(dff_file, {'dff':DFF, 'dffd':dffd})
    else:
        data = []
        for (a,b) in zip(idx[0:-1], idx[1:]):
            dt = img_time[b]-img_time[a]
            data.append(dff_data[a:b,:].sum(axis=0) / dt)        
        dffd = np.array(data)
        
    return dffd



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



def plot_catraces_avgclasses(ipath, roi_mapping, pplot=True, vm=-1, tstart=0, tend=-1, 
                             emg_ticks=[], cb_ticks=[], show_avgs=True, pnorm_spec=False, 
                             fmax=20, show_peaks=False, dff_filt=False, f_cutoff=2.0, dff_var='dff'):
    """
    For each mouse in the given ROI mapping, plot all ROIs sorted by different classes. 
    If $show_avgs = True, plot for each class the average across all ROIs.

    Parameters
    ----------
    ipath : string
        DESCRIPTION.
    roi_mapping : pd.DataFrame
        Must contain column 'Type.
    pplot : TYPE, optional
        DESCRIPTION. The default is True.
    vm : TYPE, optional
        DESCRIPTION. The default is -1.
    tstart : float, optional
        Beginning [in seconds] of the shown interval in the recording. The default is 0.
        NOTE: tstart and tend only work for show_avgs=False
    tend : TYPE, optional
        DESCRIPTION. The default is -1.
    emg_ticks : TYPE, optional
        DESCRIPTION. The default is [].
    cb_ticks : TYPE, optional
        DESCRIPTION. The default is [].
    show_avgs : bool, optional
        If true, plot for each 'Type' of ROIs the average DF/F activity. The default is True.
        If show_peaks == True, then plot the avearge frequency of Peaks for each ROI 'Type'
    dff_filt: bool, optional
        If True, lowpass filter calcium traces with cutoff frequency...
    f_cutoff: float, optional
        Cutoff frequency for lowpass filter.
    dff_var: string, optional
        Type of DF/F signal:
            'dff': Use unprocessed DF/F signal
            'dff_dn': Use denoised DF/F signal
            'dff_sp': Use deconvoluted DF/F signal
        For denoising and deconvolution run the script denoise_dff.py

    Returns
    -------
    df : TYPE
        DESCRIPTION.
    dfr : TYPE
        DESCRIPTION.

    """

    tlegend = 300
    types = roi_mapping.Type.unique()
    types.sort()
    clrs = sns.color_palette("husl", 5)
    color_dict = {}        
    i = 0
    for typ in types:
        color_dict[typ] = clrs[i]
        i += 1

    recordings = list(roi_mapping.columns)
    recordings = [r for r in recordings if re.match('^\S+_\d{6}n\d+$', r)]   
    
    data_r = []
    for rec in recordings:
        print(rec)
        idf = re.split('_', rec)[0]
        sr = sleepy.get_snr(ipath, rec)
        nbin = int(np.round(sr)*2.5)
        sdt = nbin * (1.0/sr)
        rec_map = roi_mapping[roi_mapping[rec] != 'X']   
        if sum(roi_mapping[rec] == 'X') == len(roi_mapping):
            print('Recording %s: no ROI present' % rec)
            continue

        roi_list = int(re.split('-', rec_map.iloc[0][rec])[0])
        types = rec_map.Type.unique()
        types.sort()
        print(types)
        
        M = sleepy.load_stateidx(ipath, rec)[0]
        M = sleepy.load_stateidx(ipath, rec)[0]
        if tend == -1:
            iend = len(M)
        else:
            iend = int(np.round(tend/sdt))
        istart = int(np.round(tstart/sdt))
        t = np.arange(istart, iend)*sdt

        
        # load DF/F for recording rec 
        dff_file = os.path.join(ipath, rec, 'recording_' + rec + '_dffn' + str(roi_list) + '.mat')
        if not(os.path.isfile(dff_file)):
            calculate_dff(ipath, rec, roi_list)

        #pdb.set_trace()
        DFF = so.loadmat(dff_file, squeeze_me=True)[dff_var]
        
        img_time = imaging_timing(ipath, rec) 
        [istart_dff, iend_dff] = eeg2img_time([istart*sdt, iend*sdt], img_time)


        if show_avgs and show_peaks:
            Peaks = find_capeaks(DFF)[0]
            DFF2 = np.zeros(DFF.shape)
            for i in range(DFF.shape[1]):
                DFF2[Peaks[i],i] = 1
            Peaks = DFF2        
            Peaks = downsample_dff2bs(ipath, rec, roi_list, peaks=True, dff_data=Peaks)
            
            for i in range(DFF.shape[1]):
                Peaks[:,i] = sleepy.smooth_data(Peaks[:,i],1)
        elif not show_avgs and show_peaks:
            Peaks = find_capeaks(DFF)[0]
        
        #t = np.arange(0, len(M))*sdt
        
        roi_dict = {typ:[] for typ in types}
        ######################################################################
        # go through all ROIs in recording rec
        for typ in types:
            dfs = rec_map[rec_map.Type == typ]
            roi_ids = []
            for index, row in dfs.iterrows():
                s=row[rec]
                roi_num = int(re.split('-', s)[1])
                roi_ids.append(roi_num)
            if len(roi_ids) > 0:
                
                dff = DFF[:,roi_ids].mean(axis=1)
                m = len(dff)
            if len(roi_ids) >= 2:
                rval = []
                block = DFF[:,roi_ids]
                for s in [1,2,3]:
                    idx = np.where(M==s)[0]
                    for i in range(block.shape[1]):
                        for j in range(i+1, block.shape[1]):
                            r, p = scipy.stats.pearsonr(block[idx,i], block[idx,j])
                            rval.append(r)
                        
                    data_r.append([idf, rec, typ, np.array(rval).mean(), s])
                
            roi_dict[typ] = roi_ids

        nrois = 0
        for typ in roi_dict:
            nrois += len(roi_dict[typ])

        r_mu = [10, 100]
        if pplot:
            
            # M = sleepy.load_stateidx(ipath, rec)[0]
            # if tend == -1:
            #     iend = len(M)
            # else:
            #     iend = int(np.round(tend/sdt))
            # istart = int(np.round(tstart/sdt))
            # t = np.arange(istart, iend)*sdt
            
            # load EEG and EMG spectrogram
            P = so.loadmat(os.path.join(ipath, rec, 'sp_%s.mat' % rec), squeeze_me=True)
            SPEEG = P['SP'] 
            if pnorm_spec == False:
                med = np.median(SPEEG.max(axis=0))
                #if vm == -1:
                #    vm = med*2.5
            else:
                sp_mean = SPEEG.mean(axis=1)
                SPEEG = np.divide(SPEEG, np.tile(sp_mean, (SPEEG.shape[1], 1)).T)
            med = np.median(SPEEG.max(axis=0))
            if vm == -1:
                vm = med*2.5
            
            freq = P['freq']
            P = so.loadmat(os.path.join(ipath, rec, 'msp_%s.mat' % rec), squeeze_me=True)
            SPEMG = P['mSP'] 
        
        
            plt.ion()
            plt.figure(figsize=(10,8))
        
            # show hypnogram
            # HYPNOGRAM AXES
            axes_brs = plt.axes([0.1, 0.63, 0.8, 0.03])
            cmap = plt.cm.jet
            my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
            tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M[istart:iend]]), vmin=0, vmax=3)
        
            tmp.set_cmap(my_map)
            axes_brs.axis('tight')
            axes_brs.axes.get_xaxis().set_visible(False)
            axes_brs.axes.get_yaxis().set_visible(False)
            axes_brs.spines["top"].set_visible(False)
            axes_brs.spines["right"].set_visible(False)
            axes_brs.spines["bottom"].set_visible(False)
            axes_brs.spines["left"].set_visible(False)

            
            # show spectrogram
            ifreq = np.where(freq <= fmax)[0]
            # axes for colorbar
            axes_cbar = plt.axes([0.82, 0.8, 0.1, 0.15])
            # axes for EEG spectrogram
            # AXES SPECTROGRAM 
            axes_spec = plt.axes([0.1, 0.8, 0.8, 0.15], sharex=axes_brs)                                         
            im = axes_spec.pcolorfast(t, freq[ifreq], SPEEG[ifreq, istart:iend], cmap='jet', vmin=0, vmax=vm)
            axes_spec.axis('tight')
            #axes_spec.set_xticklabels([])
            #axes_spec.set_xticks([])
            axes_spec.spines["bottom"].set_visible(False)
            axes_spec.axes.get_xaxis().set_visible(False)

            plt.ylabel('Freq (Hz)')
            sleepy.box_off(axes_spec)
            plt.xlim([t[0], t[-1]])
            plt.title(rec)
            
            # colorbar for EEG spectrogram
            cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0)
            cb.set_label('Power ($\mathrm{\mu}$V$^2$s)')
            if len(cb_ticks) > 0:
                cb.set_ticks(cb_ticks)
            axes_cbar.set_alpha(0.0)
            axes_cbar.spines["top"].set_visible(False)
            axes_cbar.spines["right"].set_visible(False)
            axes_cbar.spines["bottom"].set_visible(False)
            axes_cbar.spines["left"].set_visible(False)
            axes_cbar.axes.get_xaxis().set_visible(False)
            axes_cbar.axes.get_yaxis().set_visible(False)
            
            # show EMG
            i_mu = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
            # * 1000: to go from mV to uV
            p_mu = np.sqrt(SPEMG[i_mu, :].sum(axis=0) * (freq[1] - freq[0])) 
            axes_emg = plt.axes([0.1, 0.68, 0.8, 0.1], sharex=axes_spec)
            axes_emg.plot(t, p_mu[istart:iend], color='black')
            axes_emg.patch.set_alpha(0.0)
            axes_emg.spines["bottom"].set_visible(False)
            axes_emg.axes.get_xaxis().set_visible(False)

            if len(emg_ticks) > 0:
                axes_emg.set_yticks(emg_ticks)
            plt.ylabel('Ampl. ' + '$\mathrm{(\mu V)}$')
            plt.xlim((t[0], t[-1]))
            sleepy.box_off(axes_emg)
            
            # DF/F signals
            axes_dff = plt.axes([0.1, .05, 0.8, 0.5], sharex=axes_spec)
            #dff_file = os.path.join(ipath, rec, 'recording_' + rec + '_dffn' + str(roi_list) + '.mat')
            #if not(os.path.isfile(dff_file)):
            #    calculate_dff(ipath, rec, roi_list)
            #DFF = so.loadmat(dff_file, squeeze_me=True)['dff']
            img_time = imaging_timing(ipath, rec) 
            [istart_dff, iend_dff] = eeg2img_time([istart*sdt, iend*sdt], img_time)
            
            
            
            isr = 1/np.mean(np.diff(img_time))
            nframes = DFF.shape[0]
            nroi    = DFF.shape[1]
            dffmax = DFF.max()
            
            if dff_filt:
                for i in range(nroi):
                    dff = DFF[:,i]
                    w0 = f_cutoff / (isr*0.5)
                    DFF[:,i] = sleepy.my_lpfilter(dff, w0)
                    
            if not show_avgs:                
                # get calcium peaks
                # Peaks = {}
                # for i in range(nroi):
                #     dff = DFF[:,i]
                #     dff = sleepy.my_lpfilter(dff, 0.1)                    
                #     prom = np.percentile(dff, 80)
                #     idx = scipy.signal.find_peaks(dff, prominence=prom)[0]
                #     Peaks[i] = idx
                
                i = 0
                for typ in types:
                    roi_ids = roi_dict[typ]
                    for roi_count, j in enumerate(roi_ids):
                        #plt.plot(img_time[0:nframes], DFF[0:nframes,j]+i*dffmax, color=color_dict[typ])
                        plt.plot(img_time[istart_dff:iend_dff], DFF[istart_dff:iend_dff,j]+i*dffmax, color=color_dict[typ])
                        
                        if show_peaks:
                            plt.plot(img_time[Peaks[j]],  DFF[Peaks[j],j]+i*dffmax, 'r.', markersize=4)
                        if roi_count == 0:
                            plt.text(img_time[istart_dff]+100, i*dffmax+dffmax/4, str(j) + ' ' + typ, fontsize=12, color=color_dict[typ],bbox=dict(facecolor='w', alpha=0.))
                        i += 1
    
                plt.ylim([-dffmax, dffmax*nrois])
            else: 
                # plot averages of Calcium traces for each class
                #if show_peaks:
                #    DFF2 = np.zeros(DFF.shape)
                #    for i in range(DFF.shape[1]):
                #        DFF2[Peaks[i],i] = 1
                #    DFF = DFF2
                #pdb.set_trace()
                
                if not show_peaks:
                    dmax = []
                    for typ in types:
                        roi_ids = roi_dict[typ]
                        dmax.append(DFF[:,roi_ids].mean(axis=1).max())
                    dffmax = np.max(np.array(dmax))
    
                    ntypes = len(types)
                    i = 0
                    
                    for typ in types:
                        roi_ids = roi_dict[typ]
                        plt.plot(img_time[0:nframes], DFF[0:nframes,roi_ids].mean(axis=1)+i*dffmax, color=color_dict[typ])
                        plt.text(100, i*dffmax+dffmax/4, 'avg' + ' ' + typ, fontsize=12, color=color_dict[typ],bbox=dict(facecolor='w', alpha=0.))
                        i += 1
                    plt.ylim([-dffmax, dffmax*(ntypes)])
                else:
                    dmax = []
                    for typ in types:
                        roi_ids = roi_dict[typ]
                        dmax.append(Peaks[:,roi_ids].mean(axis=1).max())
                    dffmax = np.max(np.array(dmax))
    
                    ntypes = len(types)
                    i = 0
                    tpeaks = np.arange(0, Peaks.shape[0]) * sdt
                    for typ in types:
                        roi_ids = roi_dict[typ]
                        
                        plt.plot(tpeaks, Peaks[:,roi_ids].mean(axis=1)+i*dffmax, color=color_dict[typ])
                        plt.text(100, i*dffmax+dffmax/4, 'avg' + ' ' + typ, fontsize=12, color=color_dict[typ],bbox=dict(facecolor='w', alpha=0.))
                        i += 1
                    plt.ylim([-dffmax, dffmax*(ntypes)])
                    
                    
                    
            plt.title('Time (s)')

            # AXES TIME LEGEND
            axes_legend = plt.axes([0.1, 0.58, 0.8, 0.03], sharex=axes_dff)
            plt.ylim((0, 1.1))
            #plt.xlim([t[0], t[-1]])
            plt.xlim([img_time[istart_dff], img_time[iend_dff]])
            axes_legend.plot([0, tlegend], [1, 1], color='black')
            axes_legend.text(tlegend/4.0, 0.0, str(tlegend) + ' s')
            axes_legend.patch.set_alpha(0.0)
            axes_legend.spines["top"].set_visible(False)
            axes_legend.spines["right"].set_visible(False)
            axes_legend.spines["bottom"].set_visible(False)
            axes_legend.spines["left"].set_visible(False)
            axes_legend.axes.get_xaxis().set_visible(False)
            axes_legend.axes.get_yaxis().set_visible(False)

    #dfr = pd.DataFrame(data=data_r, columns=['mouse', 'recording', 'type', 'r', 'state'])
    #df = pd.DataFrame(data=data, columns=['mouse', 'recording', 'type', 'time', 'dff', 'state'])

    #return df, dfr
    


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
        D[j,:,:] = dset[i,:,:]
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

# OLD VERSION
# def disk_filter(ppath, name, nframes, pfilt=1, nblock=1000, psave=1) :
#     """A = disk_filter(ppath, name, nframes pfilt=1, nblock=1000)
#     $ppath/$name refers to a TIFF stack.
#     """
#     print("performing disk filtering ...")
#     # load disk filter
#     disk = so.loadmat('disk.mat')['h']
#     stack = TIFF.open(os.path.join(ppath, name))
#     (nx, ny) = stack.read_image().shape
    
#     # h5f5 file holding the filtered stack:
#     a = re.split('_', name)[0:-1]
#     disk_fname = '_'.join(a) + '_disk.mat'
#     disk_meanfname = '_'.join(a) + '_diskmean.mat'
    
#     h5_file = os.path.join(ppath, disk_fname)
    
#     if os.path.isfile(h5_file) == 1 :
#         # delete file
#         print("deleting file %s..." % h5_file)
#         os.remove(h5_file)

#     f = h5py.File(h5_file, 'w')
#     # reference to data set holding all frames
#     dset = f.create_dataset('images', (nframes, nx, ny))

#     D = np.zeros((nblock, nx, ny))
#     A = np.zeros((nx, ny)) # average across frames
#     offset = 0
#     k = 0  #absolute counter for all frames in the tiff stack
#     if pfilt == 1 :
#         j = 0   #relative counter for each block
#         for image in stack.iter_images() :
#             tmp = image - scipy.signal.fftconvolve(image.astype('float64'), disk, mode='same')
#             D[j,:,:] = tmp
#             A = A + tmp
#             j = j+1
#             k = k+1

#             if j == nblock or k == nframes:
#                 print("writing block...")
#                 # write to mat file
#                 dset[offset:offset+j,:,:] = D[0:j,:,:]
#                 offset = offset + j
#                 j = 0
#     else :
#         j = 0
#         for image in stack.iter_images() :
#              dset[j,:,:] = image
#              j = j+1
             
#     # make sure to close all files
#     stack.close()
#     f.close()

#     A = A / (nframes*1.0)

#     if psave == 1:
#         print("saving mean frame to %s" % disk_meanfname)
#         so.savemat(os.path.join(ppath, disk_meanfname), {'mean' : A})
    
#     return A



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
    # necessary to make it work in python 3:
    plt.show(block=True)

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
    print('No. of frames in timestamp.dat %d; No. of frames in recording_downsamp.hdf5: %d' % (nframes, stack.shape[0]))
    nframes = stack.shape[0]


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
            real_start_i = cam_jumps[ilong_frame[0]+1]
        # otherwhise, we assume the long frame is right at the end
        else:
            real_end_i = cam_jumps[ilong_frame[0]]
    elif len(ilong_frame) == 0:
        real_end_i = cam_jumps[-1]
    # there are two long frames, we assume that happened at the beginning
    # and at the end:
    elif len(ilong_frame) == 2:
        real_start_i = cam_jumps[ilong_frame[0]+1]
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




