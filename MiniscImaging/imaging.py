import sys
# sys.path.append('/Users/tortugar/Google Drive/Berkeley/Data/Programming/CommonModules')
#sys.path.append('/home/franz/GDrive/Programming/CommonModules')
from Utility import *
# from libtiff import TIFF
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
        print self.mean_name
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
    print "Saving roi list to %s" % fname
    so.savemat(os.path.join(ddir, fname), {'roi_list': roi_list, 'bnd_list': bnd_list})

    return (n, fname)


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
    """
    ddir = os.path.join(ipath, name)
    img = so.loadmat(os.path.join(ddir, 'recording_' + name + '_alignedmean.mat'))['mean']
    img = so.loadmat(os.path.join(ddir, 'recording_' + name + '_diskmean.mat'))['mean']

    # show the image with both ROIs and their mean values
    plt.figure()
    ax = plt.subplot(111)
    ax.imshow(img, cmap='gray')

    # get colormap
    nroi = len(idx_list)
    cmap = plt.get_cmap('jet')
    cmap = cmap(range(0, 256))[:,0:3]
    cmap = downsample_matrix(cmap, int(np.floor(1.0*256/nroi)))

    for (r,i) in zip(bnd_list,range(0, len(bnd_list))) :
        allxpoints = r[0]
        allypoints = r[1]
        l = plt.Line2D(allxpoints+[allxpoints[0]], allypoints+[allypoints[0]], color=cmap[i,:], lw=2.)
        ax.add_line(l)
        plt.draw()
        plt.text(allxpoints[0], allypoints[0], str(i), fontsize=14, color=cmap[i,:],bbox=dict(facecolor='w', alpha=0.))
    plt.xticks([])
    plt.yticks([])
    plt.show(block=blk)
    


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
    print "Saving Ca traces of roilist %d to %s" % (n, fname)

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
    M = load_stateidx(ipath, name)
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
                fidx = fidx+range(a[0],a[1]+1)

            S[i,istate-1] = np.mean(F[np.array(fidx),i])

    return S


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
    M = load_stateidx(ipath, name)
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

    

def baseline_correction(F, time, perc=20, firstframes=100) :
    
    pc = np.percentile(F, 20)    
    idx = np.where(F<pc)[0]
    idx = idx[np.where(idx>firstframes)[0]]
    p = least_squares(time[idx], F[idx], 1)[0]
    basel = time*p[0]+p[1]

    return basel



 
def plot_catraces(ipath, name, roi_id, cf=0, bcorr=1, pltSpec = False) :
    """
    plot Ca traces in a nice and organized way along with brain state annotation
    @ARGS
    ipath     -     Imaging main folder
    name      -     Name of Imaging session
    roi_id    -     id of roi list to be plotted
    cf        -     correction factor
    bcorr     -     baseline correction? [0|1]
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
        #fps = 15 #float(get_infoparam(os.path.join(ipath, name), 'info.txt', 'FPS')[0])
        #nframes = ROI.shape[0]-2 #subtracting the first couple of bad frames
        #img_time = np.arange(0, nframes)*(1.0/15) #28.204/.14 for 4/5/19 #26.747 for 4/2/19)
        #print(nframes)
        img_time = so.loadmat(os.path.join(ipath, name, 'img_time.mat'), squeeze_me=True)['time']
        
    
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
    

    # pdb.set_trace()
    # create colormap for CA traces
    nroi = F.shape[1]
    cmap = plt.get_cmap('jet')
    cmap = cmap(range(0, 256))[:,0:3]
    cmap = downsample_matrix(cmap, int(np.floor(256/nroi)))
    fmax = F.max()

    
    # collect brainstate information
    sdt = 2.5
    M = load_stateidx(ipath, name)
    sp_time = np.arange(0, sdt*M.shape[0], sdt)
    #img_time = imaging_timing(ipath, name)

    # Plotting all together: First, time dependent calcium traces
    plt.figure()
    ax = plt.subplot(212)
    for istate in range(1,4):
        idx = np.nonzero(M==istate)[0]
        seq = get_sequences(idx)
    
        for s in seq :
            if istate == 1 :
                ax.add_patch(patches.Rectangle((s[0]*sdt, -fmax), len(s)*sdt, (nroi+1)*fmax, color=[0.8, 1.0, 1.0]))
            if istate == 2 :
                ax.add_patch(patches.Rectangle((s[0]*sdt, -fmax), len(s)*sdt, (nroi+1)*fmax, color=[1, 0.8, 1]))
    
    
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
                if len(s)*2.5 >= 1.0*tborder :
                    a = eeg2img_time([s[0]*sdt + tborder, s[-1]*sdt], img_time)
                    fidx = fidx+range(a[0],a[1]+1)
            # pdb.set_trace()
            S[i,istate-1] = np.mean(F[np.array(fidx),i])

    if pltSpec == True:
        ES = so.loadmat(os.path.join(ipath, name, 'sp_' + name + '.mat'))
        ES = ES['SP'][0:60, :]
        ax2 = plt.subplot(5,1,2, sharex=ax)
        med = np.median(ES.max(axis=0))
        ax2.pcolorfast(sp_time, np.linspace(0,30,60), ES, vmin = 0, vmax = med*1.4)
        ax2.set_xlim([0,sp_time[-1]])


    plt.figure()
    # average bars
    plt.subplot(121)
    y = S.mean(axis=0)
    std = S.std(axis=0)
    state_colors = np.array([[0.8, 1.0, 1.0], [1.0, 0.8, 1.0], [0.8, 0.8, 0.8]])
    #plt.bar(np.array([1,2,3])-0.25, y, width=0.5, yerr=std/np.sqrt(nroi))
    for i in range(3) :
        plt.bar(i+1-0.25, y[i], width=0.5, yerr=std[i]/np.sqrt(nroi), color=state_colors[i,:])
    
    plt.ylabel('DF/F')
    plt.xticks([1,2,3], ('REM', 'Wake', 'NREM'))

    # single cells
    for i in range(nroi) :
        plt.plot(range(1,4), S[i,:], color=cmap[i,:], marker='o')
    plt.show()


    
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
    import matplotlib.patches as patches

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
    cmap = cmap(range(0, 256))[:,0:3]
    print nroi
    cmap = downsample_matrix(cmap, int(np.floor(256/nroi)))
    fmax = F.max()


    # Plotting all together: First, time dependent calcium traces
    plt.figure()
    ax = plt.subplot(111)
    
    
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

    print "Creating h5 file %s" % fmat_name
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



def h52tiff_split(ppath, name) :
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

    tiff_name = os.path.join(ppath, fbase + '.tif')
    if os.path.isfile(tiff_name) == 1 :
        os.remove(tiff_name)
    tiff = TIFF.open(tiff_name, mode='w')

    fcount = 1
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


# def h52tiff_annotation(ppath, name, sdt=2.5, location='SE') :
#     """
#     ppath     -      path containing tiff stack
#     name      -      name of tiff stack
#     convert h5 stack to tiff stack.
#     tiff stack only allow for a maximum size of 4GB.
#     If the stack exceeds this limit, create a new stack.
#     File one is name stem.tif, file two is named stem-001.tif etc.
    
#     Additionally, add to each frame the current brain state annotation.
#     """
#     FLIMIT = 4000000000
#     MAX_INT = np.power(2, 15)
#     fbase = re.split('\.', name)[0]
    
#     stack_fid = h5py.File(os.path.join(ppath, name), 'r')
#     dset = stack_fid['images']
#     (nframes, nx, ny) = dset.shape

#     tiff_name = os.path.join(ppath, fbase + '.tif')
#     if os.path.isfile(tiff_name) == 1 :
#         os.remove(tiff_name)
#     tiff = TIFF.open(tiff_name, mode='w')

#     # Sleep State Data
#     # get imaging time
#     base_path = '/'.join(ppath.split('/')[0:-1])
#     recording = re.search('recording_(\S+_\S+)_', name).group(1)
#     itime = imaging_timing(base_path, recording)
#     M = map(int, load_stateidx(base_path, recording))
#     stime = np.arange(0, len(M))*sdt
#     Letters = [L.astype('uint16') for L in read_letters()]    
#     (lx, ly) = Letters[0].shape

#     # timing movie frame -> index brain state
#     closest_time = lambda(it, st) : np.argmin(np.abs(it-st))

#     pdb.set_trace()
#     fcount = 1
#     for i in range(nframes) :
#         frame = dset[i,:,:]
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
            
#         # write frame        
#         tiff.write_image(frame)

#         # if too large, split TIFF stack
#         if os.path.getsize(tiff_name) > FLIMIT :
#             # generate next tiff file
#             s = "%03d" % fcount
#             tiff.close()
#             tiff_name = os.path.join(ppath, fbase + '-' + s + '.tif')
#             if os.path.isfile(tiff_name) == 1 :
#                 os.remove(tiff_name)

#             tiff = TIFF.open(tiff_name, mode='w')
#             fcount = fcount + 1

#     tiff.close()
#     stack_fid.close()


def h52tiff_annotation_time(ppath, name, sdt=2.5, location='SE') :
    """
    ppath     -      path containing tiff stack
    name      -      name of tiff stack
    convert h5 stack to tiff stack.
    tiff stack only allow fr a maximum size of 4GB.
    If the stack exceeds this limit, create a new stack.
    File one is name stem.tif, file two is named stem-001.tif etc.
    
    Additionally, add to each frame the current brain state annotation.
    """
    FLIMIT = 4000000000
    MAX_INT = np.power(2, 15)
    fbase = re.split('\.', name)[0]
    
    stack_fid = h5py.File(os.path.join(ppath, name), 'r')
    dset = stack_fid['images']
    (nframes, nx, ny) = dset.shape

    tiff_name = os.path.join(ppath, fbase + '.tif')
    if os.path.isfile(tiff_name) == 1 :
        os.remove(tiff_name)
    tiff = TIFF.open(tiff_name, mode='w')

    # Sleep State Data
    # get imaging time
    base_path = '/'.join(ppath.split('/')[0:-1])
    recording = re.search('recording_(\S+_\S+)_', name).group(1)
    itime = imaging_timing(base_path, recording)
    M = map(int, load_stateidx(base_path, recording))
    stime = np.arange(0, len(M))*sdt
    Letters = [L.astype('uint16') for L in read_letters()]
    Numbers = [N.astype('uint16') for N in read_numbers()]
    (lx, ly) = Letters[0].shape
    (dx, dy) = Numbers[0].shape

    # timing movie frame -> index brain state
    closest_time = lambda(it, st) : np.argmin(np.abs(it-st))

    fcount = 1
    for i in range(np.min((len(itime),nframes))) :
        if i % 1000 == 0 :
            print "Done with %d of %d frames" % (i, nframes)
        
        frame = dset[i,:,:]
        
        # Add R,N,W annotation
        # get the index of the brainstate index that is closest to the timing
        # of the current frame
        # state = M[int(eeg2img_time(itime[i], stime))]
        state = M[closest_time((itime[i], stime))]
        MAX_INT = frame.max()
        # write state onto frame
        if location == 'SE' :
            piece = frame[nx-lx:, ny-ly:]
            piece[np.where(Letters[state-1]==1)] = MAX_INT
            frame[nx-lx:, ny-ly:] = piece
        elif location == 'NE' :
            piece = frame[:lx:, ny-ly:]
            piece[np.where(Letters[state-1]==1)] = MAX_INT
            frame[:lx:, ny-ly:] = piece
        elif location == 'NW' :
            piece = frame[:lx:, :ly]
            piece[np.where(Letters[state-1]==1)] = MAX_INT
            frame[:lx:, :ly] = piece
        else : # location == 'SW'
            piece = frame[nx-lx:, :ly]
            piece[np.where(Letters[state-1]==1)] = MAX_INT
            frame[nx-lx:, :ly] = piece

        # Add Time,
        # Could be faster if a preallocated an array for timestamp
        # instead of concatenating
        timestamp = np.zeros((dx,1))
        timestring = str(int(round(itime[i])))
        for i in range(len(timestring)) :
            timestamp=np.concatenate((timestamp,Numbers[int(timestring[i])]), axis=1)
            timestamp=np.concatenate((timestamp,np.zeros((dx,1))), axis=1)
        # add time to frame
        timepiece = frame[0:timestamp.shape[0], 0:timestamp.shape[1]]
        timepiece[np.where(timestamp==1)] = MAX_INT
        frame[0:timestamp.shape[0], 0:timestamp.shape[1]] = timepiece
        
        # write frame        
        tiff.write_image(frame)

        # if too large, split TIFF stack
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


    
    
def read_letters() :
    """
    read masks for R, W, N saved in letters.txt
    """
    fid = open('letters.txt')
    lines = fid.readlines()
    fid.close()
    L = re.findall("<([\s\S]*?)>",''.join(lines))

    Letters = []
    for letter in L:
        A = []
        lines = letter.split('\n')
        for l in lines :
            if not(re.match('^\s*$', l)) :
                numbers = l.split(' ')
                b = [int(i) for i in numbers if not(i=='')]
                A.append(b)
        Letters.append(np.array(A))

    return Letters


def read_numbers() :
    """
    read masks for R, W, N saved in letters.txt
    """
    fid = open('numbers.txt')
    lines = fid.readlines()
    fid.close()
    L = re.findall("<([\s\S]*?)>",''.join(lines))

    Numbers = []
    for letter in L:
        A = []
        lines = letter.split('\n')
        for l in lines :
            if not(re.match('^\s*$', l)) :
                numbers = l.split(' ')
                b = [int(i) for i in numbers if not(i=='')]
                A.append(b)
        Numbers.append(np.array(A))

    return Numbers




    
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
    print "new stack will be called %s" % new_name
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
        print "reading stack %s" % f
    
        for l in stack.iter_images() :
            # spatial downsampling
            # pdb.set_trace()
            if ndown > 1:
                l = downsample_image(l, ndown)
            new_stack.write_image(l)
            nframes = nframes + 1
            if (nframes % 100) == 0 : print "Done with frame %d" % nframes
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

    print "new stack will be called %s" % new_name
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
        print "reading stack %s" % f

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
                
            if (iframes % 100) == 0 : print "Done with frame %d" % iframes    

        # 6/24/16
        # make sure to not forget the last frame of the current tiff stack
        dset[offset:offset+j,:,:] = D[0:j,:,:]
            
            
        # important:
        stack.close()

    fmat.close()    

    return iframes



def avi2h5(rawfolder, ppath, name, ndown=1):
    """
    transform avi movies in folder $rawfolder
    to a hdf5 stack saved in $ppath/$name/recording_$name_downsamp.hdf5
    :param ndown: spatially downsample each frame by factor $ndown
    
    """
    import cv2
    
    new_name = 'recording_' + name + '_downsamp.hdf5'
    files = os.listdir(os.path.join(rawfolder))
    files = [f for f in files if re.match('^msCam\d+', f)]
    print files

    # sort files
    num = {}
    for f in files:
        a = re.findall('Cam(\d+)\.avi$', f)
        num[int(a[0])] = f
        
    keys = num.keys()
    keys.sort()
    files_new = []
    for k in keys:
        files_new.append(num[k])
    files = files_new
    
    # count number of frames in each video and total frame number
    nframes = 0
    frames_per_video = {}
    get_dim = True
    ret = True
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
        cap = cv2.VideoCapture(os.path.join(rawfolder, f))        
        ret = True
        print "converting video %s" % f

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

    # copy file timestamp.dat to $ppath/$name
    shutil.copyfile(os.path.join(ppath, name, 'timestamp.dat'), os.path.join(ppath, name))


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

    print "new stack will be called %s" % new_name

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
        if (i % 1000) == 0 : print "Done with frame %d" % i    

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


### DEPRICATED ###############################################
def calc_meanprojection(ppath, name) :
    """(A, nframes) =  calc_meanprojection(ppath, name) \
    calculate the average frame of the tiff stack $ppath/$name
    @RETURN:
    mean frame @A
    number of frames in stack $nframes
    """
    stack = TIFF.open(os.path.join(ppath, name))
    A = np.zeros(stack.read_image().shape)

    print "calculating mean projection..."
    nframes = 0
    for i in stack.iter_images() :
        A = A+i
        nframes = nframes+1

    A = A.astype('float64')/nframes
    stack.close()
    return (A, nframes)
################################################################


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
    print "performing disk filtering ..."
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
        print "deleting file %s..." % h5_file
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

            if j == nblock or k == nframes :
                print "writing block..."
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
        print "saving mean frame to %s" % disk_meanfname
        so.savemat(os.path.join(ppath, disk_meanfname), {'mean' : A})
    
    return A



def disk_filter_h5(ppath, name, pfilt=1, nblock=1000, psave=1, nw=2):
    """A = disk_filter(ppath, name, pfilt=1, nblock=1000)
    $ppath/$name refers to a H5DF stack.
    performs disk filtering on a h5df stack

    NOTE: output file will be named ppath/recording_V2_100215n1_disk.mat
    Naming: name refers to the mat stack that is to be filtered
    the name of the output file includes everything up to the last '_'
    after the last '_' 'disk.mat' is appended to the new file name
    
    """
    print "performing disk filtering ..."
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
        print "deleting file %s..." % h5_file
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
        #for image in stack.iter_images() :
        for index in range(0, nframes) :
            image = stack[index,:,:]
            if nw > 1:
                image = scipy.signal.fftconvolve(image.astype('float64'), filt, mode='same')
            tmp = image - scipy.signal.fftconvolve(image.astype('float64'), disk, mode='same')
            D[j,:,:] = tmp
            A = A + tmp
            j = j+1
            k = k+1

            if j == nblock or k == nframes :
                print "writing frame %d of %d frames" % (k, nframes)
                # write to mat file
                dset[offset:offset+j,:,:] = D[0:j,:,:]
                offset = offset + j
                j = 0
             
    # make sure to close all files
    fid.close()
    f.close()

    A = A / (nframes*1.0)

    if psave == 1:
        print "saving mean frame to %s" % disk_meanfname
        so.savemat(os.path.join(ppath, disk_meanfname), {'mean' : A})
    
    return A


def activity_map(ppath, name, nw=2):
    """A = disk_filter(ppath, name, pfilt=1, nblock=1000)
    $ppath/$name refers to a H5DF stack.
    performs disk filtering on a h5df stack

    NOTE: output file will be named ppath/recording_V2_100215n1_disk.mat
    Naming: name refers to the mat stack that is to be filtered
    the name of the output file includes everything up to the last '_'
    after the last '_' 'disk.mat' is appended to the new file name
    """

    a = re.split('_', name)[0:-1]
    fstem = '_'.join(a)
    actmap_fname = fstem + '_actmap.mat'
    mean_fname   = fstem + '_mean.mat'

    # load stack
    fid = h5py.File(os.path.join(ppath, name), 'r')
    stack = fid['images']
    (nframes, nx, ny) = stack.shape

    # load meanframe
    pdb.set_trace()
    mean_frame = so.loadmat(os.path.join(ppath, mean_fname))['mean']
    mean_flu = np.mean(mean_frame)
    filt = 1.0*np.ones((nw, nw)) / (nw*nw)

    A = np.zeros((nx, ny)) # average across frames
    for index in range(0, nframes):
        print index
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
        arangex = range(sx,nx)
        brangex = range(0,nx-sx)
    if sx < 0 :
        sx = -sx
        arangex = range(0,nx-sx)
        brangex = range(sx,nx)
    if sy >= 0 :
        arangey = range(sy,ny)
        brangey = range(0,ny-sy)
    if sy < 0 :
        sy = -sy
        arangey = range(0,ny-sy)
        brangey = range(sy, ny)

    return (arangex, arangey, brangex, brangey)



def align_frames(ppath, name, nframes=-1, nblock=1000, pdisk=1, psave=1, pwrite=1, pcorr_time=True) :
    """
    Shift = align_frames(ppath, name, nblock=1000) : \
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
    #plt.ion()
    
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

    pdb.set_trace()
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
            print "Done with frame %d of %d frames" % (k, nframes)
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
        print "saving mean aligned frame to %s" % mean_fname
        so.savemat(mean_fname, {'mean' : amean, 'shift' : Shift})

    return Shift



def align_stencil(ppath, name, pdisk=1, psquare=1, psave=1) :
    """
    (idx_list, Template) = align_stencil(ppath, name, pdisk=1, psquare=1, psave=1) \
    manually cut out a high contrast piece from the imaging stack used
    for motion correction.

    @RETURN:
    idx_list    -      coordinates specifying the upper left and lower right corner (x1, x2, y1, y2)
                       of a square
    """
    fstem = '_'.join(re.split('_', name)[0:-1])
    if pdisk == 1 :
        img = so.loadmat(os.path.join(ppath, fstem + '_diskmean.mat'))['mean']
    else :
        img = so.loadmat(os.path.join(ppath, fstem + '_mean.mat'))['mean']
    new_name = os.path.join(ppath, fstem + '_stencil.mat')

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


def crop_dataset(ppath, name, nblock=1000) :
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
        print "Calculating mean for stack %s" % name
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

    print "Cropping stack %s)" % name
    print "New stack will be saved to %s" % name_cropped
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
        print np.min(roi[0])
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

    print "Select OFF-Field\nLeft-click to select corners of a polygon;\ndouble click once you are happy."
    plt.imshow(img, cmap='gray')
    r = roipoly(roicolor='r')
    off_idx = np.nonzero(r.getMask(img)==True) 
    offset = img[off_idx].mean()
    
    print "Select Blood Vessel"
    plt.imshow(img, cmap='gray')
    r = roipoly(roicolor='r')
    blood_idx = np.nonzero(r.getMask(img)==True) 
    Fbv = img[blood_idx].mean()
    
    print "Select Blood Vessel Vicinity"
    plt.imshow(img, cmap='gray')
    r = roipoly(roicolor='r')
    vic_idx = np.nonzero(r.getMask(img)==True) 
    Fbv_v = img[vic_idx].mean()
    
    # (Blood-vessel - offset) / (Blood-vessel-vicinity - offset) 
    cf = (Fbv-offset)/(Fbv_v-offset)

    outname = os.path.join(ipath, name, name + '_cf.mat')
    so.savemat(outname, {'cf': cf});
    
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
    print sima_path
    
    seq = [Sequence.create('HDF5', ifile, 'txy')]
    # SIMA sequence arranged as follows:
    # time, plane, y, x, channel
    # in my case, plane and channel are always 1 

    #initialize imaging data set
    dataset = sima.ImagingDataset(seq, sima_path)
    
    
    if method == 'pt2d' :
        mc_approach = sima.motion.PlaneTranslation2D(max_displacement=max_displace)
        print "starting motion correction ..."

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
        print index
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
    



if __name__ == '__main__' :

    ipath = '/media/Transcend/RawImaging'
    #name = 'VL2_080715n2'
    name = 'GVLP01_082615n1'
    
    rec = 'recording_' + name + '_downsamp.tif'
    mrec = 'recording_' + name + '_downsamp.mat'
    arec = 'recording_' + name + '_aligned.mat'
    ddir = os.path.join(ipath, name)
    
    pmotion = 1
    proi = 0


    ### Steps to do motion correction ###
    if pmotion == 1:
        # make this statement more convenient by defining a function called get_nframes:
        nframes = int(get_infofields(ddir, 'info.txt', ['FRAMES'])[0])
        # calculate and save mean of 
        
        # disk filter
        #disk_filter(ddir, rec, nframes, psave=1)
        disk_filter_h5(ddir, mrec, psave=1)
        # convert to h5 stack
        #tiff2h5(ddir, rec, nframes)

        # motion correction
        S = align_frames(ddir, mrec)
        # convert aligned stack to tif to check in imagej
        h52tiff_split(ddir, arec)

    if proi == 1:
        (roi_list, bnd_list) = roi_manual(ddir, rec)
        save_roilist(ipath, name, roi_list, bnd_list)

        # load "aligned recording" tiff stack
        stack = TIFFStack(ddir, arec)
        Bkg, Halo = halo_subt(roi_list, 20, stack.nx, stack.ny)
        ROI = stack.get_rois(roi_list)
        bROI = stack.get_rois(Halo)
        
        # get correction factor
        cf = correction_factor(ipath, name)

        B = np.zeros(ROI.shape)
        for i in range(ROI.shape[1]) :
            a = ROI[:,i]-cf*bROI[:,i]
            B[:,i] = (a-a.mean()) / a.mean()
            #B[:,i] = a / a.mean()

        

    ### Test manual ROI selection #######
    #stack = TIFFStack(ppath, 'test_aligned.mat')
    #roi_list = roi_manual(ppath, name)
    ### END[test manual ROI slection ####


    ### test halo/background selection ######
    # img = so.loadmat(os.path.join(ppath, ))['mean']
    # roi_list = []
    # stack = TIFFStack(ppath, mat_name)
    # stack.mean()
    # stack.save_mean()
    # img = stack.avg
    # if not(roi_list) :
    #     roi_list = roi_manual(ppath, mat_name)
    # Bkg, Halo = halo_subt(roi_list, 10)
    # A = np.zeros((stack.nx, stack.ny))
    # for (h, r) in zip(Halo, roi_list) :
    #     A[h[0], h[1]] = 1
    #     A[r[0], r[1]] = 2
    # plt.imshow(A)
    # plt.show()



    
    ### END[test halo/background selection ##


