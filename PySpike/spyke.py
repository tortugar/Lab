#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:47:58 2017

DATE 03/20/18
New:
made spike_detect_corr2 work
Added new optional parameter (rmforce) to detect_spikes_corr

DATA 05/03/19
migrated to Python 3.7

@author: Franz
"""

# make sure to have access to module sleepy
import re
import os.path
import numpy as np
import h5py
import matplotlib.pylab as plt
import scipy.io as so
from scipy import linalg as LA
import subprocess
import pickle
import scipy
import plotly
import plotly.graph_objs as go
import sleepy
import scipy.stats as stats
import pandas as pd
import seaborn as sns
from functools import reduce
# debugger
import pdb


def cohen_d(x,y):
    """
    correct if the population S.D. is expected to be equal for the two groups.
    :param x: 1D np.array or list of values in group 1
    :param y: 1D np.array or list of values in group 2
    :return: effect size
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)



def bonferroni_signtest(df, alpha=0.05):
    """
    Bonferroni correction for Wilcoxon ranked-sign test

    :param df: pandas DataFrame, columns are dependent samples (groups) that are compared with each other
    :param alpha: significance level, alpha is corrected by dividing through the number of pairwise comparisons
    :return:
    """
    groups = df.columns
    n = len(groups)
    diffs = []
    s = []
    p = []
    labels = []
    ntest = 0
    for i in range(n):
        for j in range(i+1, n):
            g1 = groups[i]
            g2 = groups[j]
            label = str(g1) + '<>' + str(g2)
            val = stats.wilcoxon(df[g1], df[g2])
            s.append(val[0])
            p.append(val[1])

            diff = df[g1].mean() - df[g2].mean()
            diffs.append(diff)
            labels.append(label)
            ntest += 1

    reject = []
    for sig in p:
        if sig < alpha / ntest:
            reject.append(True)
        else:
            reject.append(False)

    results = pd.DataFrame(index = labels, columns=['diffs', 'statisics', 'p-values', 'reject'])
    results['diffs'] = diffs
    results['statistics'] = s
    results['p-values'] = p
    results['reject'] = reject
    return results


def get_snr(ppath, name) :
    """
    read and return SR from file $ppath/$name/info.txt 
    """
    fid = open(os.path.join(ppath, name, 'info.txt'), 'rU')
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines :
        a = re.search("^" + 'SR' + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))            
    return float(values[0])



def load_grpfile(ppath, mouse_grouping):
    """
    setup dictionary mapping group number onto channels belonging to this group
    Grp -> Channels
    """
    grpfile = os.path.join(ppath, mouse_grouping)
        
    fid = open(grpfile, 'rU')    
    lines = fid.readlines()
    fid.close()
    grp = {}
    count = 0
    for l in lines :
        if re.match("^#", l):
            continue
        if re.match("\s+", l):
            continue
        if re.match('[\d,\s]+;', l):
            count += 1
            if count <= 2:
                continue
            else:                
                a = re.findall('(\d+)', l)         
                a = [int(i) for i in a]
                grp[count-2] = a                                                
        
    return grp



def get_lfp_ch(ppath, mouse_grouping):
    """
    load LFP channels from grouping file ppath/mouse_grouping
    lines with LFP channels start with "L".
    For example,
    L2, 3;
    indicates that channels 2 and 3 serve as LFP.
    @Return: list of integers; if there's no LFP channel, empty list
    """
    grpfile = os.path.join(ppath, mouse_grouping)

    fid = open(grpfile, 'rU')    
    lines = fid.readlines()
    fid.close()

    lfp = []
    for l in lines:
        if re.match('^L[\d,\s]+;', l):
            a = re.findall('([\d,\s]+;)', l)[0]
            a = re.findall('(\d+)', l)         
            a = [int(i) for i in a]
            lfp += a
            
    return lfp

            

def load_units_simple(ppath, file_listing):

    fid = open(os.path.join(ppath, file_listing), 'rU')
    lines = fid.readlines()

    units = []
    for line in lines:
        if re.match('^\D+\d+_\d+n\d+\s+\d+\s+\d+', line):
            a = re.split('\s+', line)
            unit = (int(a[1]), int(a[2]))
            name = a[0]

            units.append([name, unit])
    return units



class Recording:
    def __init__(self, name, grp, un, suppl = ''):
        self.name = name
        self.grp  = grp
        self.un   = un
        self.suppl = suppl
        self.path = ''

    def __repr__(self):
        s = "%s:(%d,%d)" % (self.name, self.grp, self.un)
        return s

    def set_path(self, ppath, backup=''):
        if self.path != '':
            return
        if len(backup) == 0:
            self.path = ppath
        else:
            if os.path.isdir(os.path.join(ppath, self.name)):
                self.path = ppath
            else:
                self.path = backup
        ddir = os.path.join(self.path, self.name)
        self.path, self.name = os.path.split(ddir)


def load_units(ppath, file_listing):
    """
    return all units in the given file listing
    
    Syntax of the unit listing:
    ID  name   grp    un 
    
    :param ppath: string, folder containing file listing
    :param file_listing: string
    :return: dict with all units as values and keys as unit ids going 0 to number of units;
             each unit, is a list of recordings; each recording is an object
             of class Recording)
    """

    fid = open(os.path.join(ppath, file_listing), 'rU')
    lines = fid.readlines()

    units = dict()
    newline = True
    j = -1 # index over units
    for l in lines:
        if re.match('^\s*#', l):
            continue

        if len(l)==0  or re.match('^\s*$', l):
            newline = True
            continue

        a = re.split('\s+', l)
        name = a[1]
        grp  = int(a[2])
        un   = int(a[3])
        r = Recording(name, grp, un)

        if not newline:
            #n = units[j]['num']
            #units[j]['num'] += 1
            #units[j]['recs'].append(r)
            units[j].append(r)
        else:
            newline = False
            #units[j] = {'num':1, 'recs':[]}
            #units[j]['recs'].append(r)
            j += 1
            units[j] = []
            units[j].append(r)

    return units



def get_eegemg_ch(ppath, mouse_grouping):
    """
    return number of EEG and EMG channels
    """
    grpfile = os.path.join(ppath, mouse_grouping)
        
    fid = open(grpfile, 'r')    
    lines = fid.readlines()
    fid.close()
    grp = {}
    count = 0
    for l in lines :
        if re.match("^#", l):
            continue
        if re.match("\s+", l):
            continue
        if re.match('[\d,\s]+;', l):
            count += 1
            if count > 2:
                continue
            else:                
                a = re.findall('(\d+)', l)         
                a = [int(i) for i in a]
                grp[count] = a                                                
        
    return list(grp.values())
        

    
def create_unitfile(ppath, name):
    """
    create unit annotation file
    """
    ufile = os.path.join(ppath, name, 'units_' + name + '.txt')
    if not(os.path.isfile(ufile)):
        fid = open(ufile, 'w')
        fid.write('#' + name + os.linesep)
        fid.write('#Quality: 1-6' + os.linesep)
        fid.write('#Driven: -1|0|1' + os.linesep)
        fid.write('#Group\tUnit\tElec\tTyp\tDriven\tQual\tDepth' + os.linesep)
        fid.close()


def get_infoparam(ifile, field):
    """
    NOTE: field is a single string
    and the function does not check for the type
    of the values for field.
    In fact, it just returns the string following field
    """
    fid = open(ifile, 'r')
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines:
        a = re.search("^" + field + ":" + "\s+(.*)", l)
        if a:
            values.append(a.group(1))

    return values


def add_unit_annotation(ppath, name, grp, un, typ, driven, quality, comment = ''):
    """
    add unit to annotation
    """
    ufile = os.path.join(ppath, name, 'units_' + name + '.txt')
    #sfile = os.path.join(ppath, name, 'Spk%d.npz' % grp)
    files = os.listdir(os.path.join(ppath, name))
    files = [f for f in files if re.match('^Spk' + str(grp), f)]
    sfile = ''
    if len(files) > 0:
        sfile = files[0]

    # create file if it doesn't exist
    if not(os.path.isfile(ufile)):
        create_unitfile(ppath, name)
    
    # Get electrode channel
    el = 0
    #if not(os.path.isfile(sfile)):
    if len(files) == 0:
        print("Unit (%d, %d) does not exist." % (grp, un))
        return
    else:
        el = unpack_unit(ppath, name, grp, un)[3]
        # explanation of why this is necessary:
        # https://stackoverflow.com/questions/22661764/storing-a-dict-with-np-savez-gives-unexpected-result/41566840
        #Spk = {key:Spk[key].item() for key in Spk if Spk[key].dtype == 'O'}
        #if not(Spk['ch'].has_key(un)):
        #    print "Grp %d does not have unit %d" % (grp, un)
        #    return
        #else:
        #    el = Spk['ch'][un]

    # get depth from info.txt
    depth = float(get_infoparam(os.path.join(ppath, name, 'info.txt'), 'depth')[0])

    # add unit
    fid = open(ufile, 'a')
    #fid.write(os.linesep)
    fid.write(('%d\t%d\t%d\t%s\t%d\t%d\t%2.f'+os.linesep) % (grp, un, el, typ, driven, quality, depth))    
    if len(comment) > 0:
        fid.write('#%s' % comment)
        fid.write(os.linesep)                 
    fid.write(os.linesep)
    fid.close()


    
def del_unit_annotation(ppath, name, grp, un):
    """
    delete unit (grp, un) in annotation file units_name.txt
    """
    ufile = os.path.join(ppath, name, 'units_' + name + '.txt')

    if not(os.path.isfile(ufile)):
        return
    else:
        fid = open(ufile, 'r')
    
    lines = fid.readlines()
    fid.close()
    

    new_lines = []
    got_unit = False
    for line in lines:
        if got_unit and re.match('^#', line):
            continue
        
        if got_unit and re.match('^\s+$', line):
            got_unit=False            
        
        if re.match('^%d\t%d' % (grp, un), line):
            got_unit=True
            continue
        else:
            new_lines.append(line)
            
    fid = open(ufile, 'w')
    for line in new_lines:
        fid.write(line)
    fid.close()
    


def cleanup_recording(ppath, name):
    """
    delete are raw channels files that are NOT part of a group which a clustered unit.
    More precisely, if no file Spki.npz exists for group i, all channels of group i 
    are deleted.
    """
    idf = re.split('_', name)[0]
    grp = load_grpfile(ppath, idf + '_grouping.txt')
    
    ddir = os.path.join(ppath, name)

    #spk = [f for f in os.listdir(ddir) if re.search('^Spk\d+\.npz', f)]
    spk = [f for f in os.listdir(ddir) if re.search('^Spk\d+', f)]
    
    # groups with saved units
    unit_groups = []
    for s in spk:
        a = re.findall('^Spk(\d+)', s)[0]
        unit_groups.append(int(a))
        
    groups_to_delete = list(set(grp.keys()) - set(unit_groups))
    channels_to_delete = []
    for g in groups_to_delete:
        channels_to_delete += grp[g]
        
    existing_channels = []
    channels = [f for f in os.listdir(ddir) if re.search('^ch_%s_'%name, f)]
    for c in channels:
        a = re.findall('_(\d+)\.mat$', c)[0]
        existing_channels.append(int(a))
        
    files_to_delete = []
    for c in channels_to_delete:
        f = 'ch_' + name + '_' + str(c) + '.mat'
        if os.path.isfile(os.path.join(ddir, f)):
            files_to_delete.append(f)
            
    print("Going to delete the following files:")
    for f in files_to_delete:
        print(f)

    ans = input("Are you sure you want to delete these files? [yes | no]\n> ")
    if ans == 'yes':
        for f in files_to_delete:
            os.unlink(os.path.join(ddir, f))


            
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



def downsample_overlap(x, nwin, noverlap):
    """
    Say,
    len(x)=10
    nwin=5
    nolverap=3
    1 2 3 4 5 6 7 8 9 10

    1 2 3 4 5
        3 4 5 6 7
            5 6 7 8 9

    :param x:
    :param nwin:
    :param noverlap:
    :return:
    """
    nsubwin = nwin-noverlap
    n_down = int(np.floor((x.shape[0]-noverlap)/nsubwin))
    x_down = np.zeros((n_down,))
    j = 0
    for i in range(0, x.shape[0]-nwin+1, nsubwin):
        x_down[j] = x[i:i+nwin].mean()
        j += 1

    return x_down



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



def load_laser(ppath, name):
    """
    load laser from recording ppath/name ...
    @RETURN: 
    @laser, vector of 0's and 1's 
    """ 
    # laser might be .mat or h5py file
    # perhaps we could find a better way of testing that
    file = os.path.join(ppath, name, 'laser_'+name+'.mat')
    try:
        laser = np.array(h5py.File(file,'r').get('laser'))
    except:
        laser = so.loadmat(file)['laser']
    return np.squeeze(laser)



def laser_waveform(laser, idx, channel, pre, post, twin, SR=1525.88):
    """
    laser     -     laser signal in EEG sampling rate
    idx       -     indices of spikes in Spike sampling rate
    channel   -     raw Spike channel
    pre,post  -     cut out laser waveform pre indices before, and post indices after laser onset
    """

    nch = len(channel)
    twin = (twin/1000.) * SR * 16
    lsr_start = (np.where(np.diff(laser)>0)[0]+1)*16
    idx_lsr = []
    #get all spike indices whithin twin indices following a laser onset
    for k in lsr_start:
        idx_lsr.append(idx[np.where((idx >= k) & (idx <= k+twin))[0]])
    
    # idx_lsr is a list of ndarrays; let's flatten them
    f = lambda x,y: np.concatenate((x,y))
    idx_lsr = reduce(f, idx_lsr)

    Wave = []
    for i in idx_lsr:
        if i>= pre and i+post < nch:
            Wave.append(channel[i-pre:i+post+1])
    
    return np.array(Wave)



def pca(data,dims=2) :
    """
    @data is a 2D matrix.
    each row is an observation (a data point).
    each column is a variable (sub data point)
    
    :return projection of data onto eigenvectors, eigenvalues, eigenvectors
    
    """
    m, n = data.shape
    # mean center the data, i.e. calculate the average "row" and subtract it
    # from all other rows
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    # Note:  each column represents a variable, while the rows contain observations.
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    # return projection of data onto eigenvectors, eigenvalues and eigenvectors  
    return np.dot(evecs.T, data.T).T, evals, evecs



def prune_grp(ppath, name, grp_dict):
    """
    throw away channels in grp_dict that do not exist;
    and then throw away groups without channels
    """
    new_dict = {}
    for g in grp_dict:
        for ch in grp_dict[g]:
            print("Testing channel %d" % ch)
            if not (os.path.isfile(os.path.join(ppath, name, 'ch_' + name + '_' + str(ch) + '.mat'))):
                print("removing channels %d = file %s" % (
                ch, os.path.join(ppath, name, 'ch_' + name + '_' + str(ch) + '.mat')))
                # new_dict[g].remove(ch)
            else:
                if g in new_dict:
                    new_dict[g].append(ch)
                else:
                    new_dict[g] = [ch]
    return new_dict



def hpfilter_channels(ppath, name, wf=300):
    """
    high pass filter all available channels in recording $ppath/$name
    :param wf: cutoff frequency
    :return: n/s
    """
    # load all existing channels
    files = os.listdir(os.path.join(ppath, name))
    files = [f for f in files if re.match('ch_', f)]
    sr = sleepy.get_snr(ppath, name)*16
    w0 = wf / (sr/2.0)
    print(w0)

    for f in files:
        fid = h5py.File(os.path.join(ppath, name, f), 'r')
        raw = np.squeeze(fid['D'])
        ch_len = raw.shape[0]
        raw = sleepy.my_hpfilter(raw, w0)
        fid.close()
        os.unlink(os.path.join(ppath, name, f))

        fid = h5py.File(os.path.join(ppath, name, f), 'w')
        dset = fid.create_dataset('D', shape=(ch_len,1), dtype='float32')
        dset[:,0] = raw
        fid.close()



def save_threshold(ppath, name, med_factor=4.0, grp_name=''):
    if os.path.isfile(os.path.join(ppath, name, 'thres.p')):
        os.unlink(os.path.join(ppath, name, 'thres.p'))

    Thres = dict()
    # mouse id
    idf = re.split('_', name)[0]
    # set grouping file name
    if grp_name == '':
        grp_name = idf + '_grouping.txt'
    # prune channels to be analyzed
    grp = load_grpfile(ppath, grp_name)
    # remove channels that do not exist
    grp = prune_grp(ppath, name, grp)
    # get all channels in use
    channels = []
    for g in grp:
        channels += grp[g]
    channels.sort()

    for ch in channels:
        fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % ch), 'r')
        raw = np.squeeze(fid['D'])
        #N = len(raw)

        med_noise = np.nanmedian(np.abs(raw)) / 0.6745
        Thres[ch] = med_noise * med_factor
        fid.close()

    with open(os.path.join(ppath, name, 'thres.p'), 'wb') as fp:
        pickle.dump({'Thres': Thres}, fp)
        fp.close()



def detect_spikes_corr(ppath, name, grp_name='', med_factor=4.0, spike_window=15, plaser=True, pspk=False, rmthres=False, igroup=0):
    """
    detect potential spikes, i.e. waveforms crossing a threshold
    
    The function extracts the following features for group i (saved in the *.fet.i file):
    For each spike (detected on one of the channels), the function cuts out the corresponding waveform
    on each of the channels. For each channel these waveforms are stored in matrices (one row - one waveform).
    The function then performes PCA on each of these matrices and saves the first 
    three PCA components for each channel.
    As a fourth parameter the function calculates the average waveform triggered by laser pulses
    and correlates this laser waveform with each spike (on each channel).
    Consequently, each channels comes with 4 parameters. 
    The second last parameter is the maximum height of the spike on either channel. 
    (If the height is larger on channel 1 then the rest, only the height for channel 1 is saved)
    The final parameter is the time point of each spike.
    Assuming group i has n channels, and there was a potential spike at timepoint tj, then that's the ordering of the features saved in *.fet.i. 
    
    PCA1-ch1 PCA1-ch1 PCA1-ch1 Laser_waveform_correlation-ch1 ... PCA1-chn PCA2-chn PCA3-chn Laser_waveform_correlation-chn Height tj
    
    
    @Parameters:
        ppath, name     recording
    @Optional:
        grp_name        grouping file for the current mouse
        med_factor      the threold a spike has to cross is given by the median of the 
                        raw channel * $med_factor
        spike_window    number of indices surrounding the center of a potential spike waveform: 
                        -15 to +15 with spike valley at 0
        plaser          If true, calculate waveform correlation between laser evoked an spontaneous spikes
                        If false, the corrleation is replaced with the 4th principal component
        pspk            if True, use existing spike times in spk_ts.p; otherwise recalculate spike times
                        for all groups or group igroup (if igroup > 0)
        rmthres         if True, remove existing threshold file (thres.p), and recalculate using
                        median * med_fractor
        igroup          integer, if 0, go through all electrode groups; otherwise only detect spikes
                        for group igroup
    """
    pup_spike = True
    pre=10
    post = 15

    lockout_win = 5
    total_samp=pre+post+1
    features=3
    # if there's no laser, we just will up the empty feature with the 4th principal component
    if not plaser:
        features = 4
        nc = features
    else:
        nc = features+1

    # duration of how long after laser pulse onset a spike is considered as laser triggered
    twin = 15 #ms
        
    # mouse id
    idf = re.split('_', name)[0]
    
    # set grouping file name
    if grp_name == '':
        grp_name = idf + '_grouping.txt'

    # prune channels to be analyzed
    grp = load_grpfile(ppath, grp_name)

    # remove channels that do not exist
    grp = prune_grp(ppath, name, grp)
    if igroup == 0:
        groups = list(grp.keys())
    else:
        groups = [igroup]

    # get all channels in use
    channels = []
    for g in groups:
        channels += grp[g]
    channels.sort()

    # load laser if it exists
    if os.path.isfile(os.path.join(ppath, name, 'laser_' + name + '.mat')) and plaser == True:
        laser = load_laser(ppath, name)
    else:
        plaser=False

    # True, if threshold file exists
    pthres = False
    # remove existing thres.p file
    if os.path.isfile(os.path.join(ppath, name, 'thres.p')) and rmthres == True:
        os.unlink(os.path.join(ppath, name, 'thres.p'))
        pthres = False
    # load existing thresholds from thres.p
    if os.path.isfile(os.path.join(ppath, name, 'thres.p')) and rmthres == False:
        with open(os.path.join(ppath, name, 'thres.p'), 'rb') as fp:
            try:
                tmp = pickle.load(fp)
            except UnicodeDecodeError:
                # to load pickle objects created using python 2
                tmp = pickle.load(fp, encoding='latin1')

            Thres = tmp['Thres']
            pthres = True
            fp.close()

    #if os.path.isfile(os.path.join(ppath, name, 'spk_ts.p')) and rmspk == True:
    #    os.unlink(os.path.join(ppath, name, 'spk_ts.p'))

    # pspk == True, i.e. load existing spike times and continue with writing .fet files
    if os.path.isfile(os.path.join(ppath, name, 'spk_ts.p')) and pspk==True:
        with open(os.path.join(ppath, name, 'spk_ts.p'), 'rb') as fp:
            try:
                tmp = pickle.load(fp)
            except UnicodeDecodeError:
                # to load pickle objects created using python 2
                tmp = pickle.load(fp, encoding='latin1')
            TS = tmp['TS']
            LWave = tmp['LWave']
            fp.close()
    # re-calculated spike spikes:
    else:
        # if there's already a spk_ts.p file load it and then overwrite,
        # the existing values
        if os.path.isfile(os.path.join(ppath, name, 'spk_ts.p')):
            with open(os.path.join(ppath, name, 'spk_ts.p'), 'rb') as fp:
                try:
                    tmp = pickle.load(fp)
                except UnicodeDecodeError:
                    # to load pickle objects created using python 2
                    tmp = pickle.load(fp, encoding='latin1')
                TS = tmp['TS']
                LWave = tmp['LWave']
                fp.close()
        # no spk_ts.p file, initialize TS and LWave
        else:
            # time points (indices) of spikes on each channel
            TS = {} # Dict: channel-id --> spike indices
            # average waveform on a specific channel
            LWave = {} # Dict: channel_id --> mean spike waveform
        # if there's no threshold file:
        if not pthres:
            Thres = {} # Dict: channel_ID --> threshold value

        for ch in channels:
            TS[ch] = []
            LWave[ch] = np.zeros((pre+post+1,))
            print("Getting spike times on channel %d" % (ch))
            fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % ch), 'r')
            raw = np.squeeze(fid['D'])
            N = len(raw)

            if not pthres:
                med_noise = np.nanmedian(np.abs(raw))/0.6745
                Thres[ch] = med_noise*med_factor

            peak_idx = np.where(raw > (1.0 * Thres[ch]))[0]
            valley_idx = np.where(raw < (-1.0 * Thres[ch]))[0]

            if len(valley_idx) > 0:
                # first index of spike, interval between first and second spike etc.
                peakISI = np.concatenate(([peak_idx[0]], np.diff(peak_idx)))            
                #peakISI=[peak_idx(1) peak_idx(2:end)-peak_idx(1:end-1)];
    
                valleyISI = np.concatenate(([valley_idx[0]], np.diff(valley_idx)))
                #valleyISI=[valley_idx(1) valley_idx(2:end)-valley_idx(1:end-1)];
                
                # interspike interval preceding first spike; interspike interval preceding 2nd spike etc.
                peak_idx   = peak_idx[np.where(peakISI>spike_window)]
                valley_idx = valley_idx[np.where(valleyISI>spike_window)] 
        
                # interspike interval following first spike; following 2nd spike etc.
                # the following conditions should be fullfilled by definition
                #peakISI   = [np.diff(peak_idx)   sig_length-peak_idx(end)];
                #valleyISI = [np.diff(valley_idx) sig_length-valley_idx(end)];
                #peak_idx=peak_idx(peakISI>spike_window);
                #valley_idx=valley_idx(valleyISI>spike_window);
        
                spk_ts = []
                if pup_spike:
                    for i in peak_idx:
                        wave_tmp = raw[i-spike_window:i+spike_window]
                        # we search for the valley value closest to the peak                    
                        spk_ts.append( np.argmin(wave_tmp) + i - spike_window )
    
                for i in valley_idx:
                    wave_tmp = raw[i-spike_window:i+spike_window]
                    if wave_tmp[np.argmin(wave_tmp)] < (-1*Thres[ch]):
                        spk_ts.append( np.argmin(wave_tmp) + i - spike_window )
                        
                spk_ts.sort()            
                spk_ts = np.array(spk_ts)
                spk_ts = spk_ts[np.where((spk_ts>=pre) & (spk_ts+post<len(raw)))]
                spk_ts = spk_ts[np.where( np.concatenate(([spk_ts[0]], np.diff(spk_ts))) > spike_window)]
                TS[ch] = spk_ts

                if len(spk_ts) > 0:
                    if plaser:
                        A = laser_waveform(laser, spk_ts, raw, pre, post, twin)
                        LWave[ch] = A.mean(axis=0)
    
            fid.close()

        #np.savez(os.path.join(ppath, name, 'spk_ts'), TS=TS, LWave=LWave, N=N)
        with open(os.path.join(ppath, name, 'spk_ts.p'), 'wb') as fp:
            pickle.dump({'TS':TS, 'LWave':LWave, 'N':N}, fp)
            fp.close()
        with open(os.path.join(ppath, name, 'thres.p'), 'wb') as fp:
            pickle.dump({'Thres':Thres}, fp)
            fp.close()

    # calculate features for each group
    f = lambda x,y: np.concatenate((x,y))
    for g in groups:
        # collect spike times on all channels
        spk_ts_curr = []
        for ch in grp[g]:
            spk_ts_curr.append(TS[ch])
        spk_ts_curr = np.unique(reduce(f, spk_ts_curr))
        #spk_ts = spk_ts[np.where((spk_ts>=pre) & (spk_ts+post<len(raw)))]
        spk_ts_curr.sort()
        
        #apply lockout window
        spk_ts_curr = spk_ts_curr[np.where( np.concatenate(([spk_ts_curr[0]], np.diff(spk_ts_curr))) > lockout_win )]
        elec_num = len(grp[g])
        spk_num = len(spk_ts_curr)
        MinMax = np.zeros((spk_num, elec_num))
        FetMat = np.zeros((len(spk_ts_curr), nc*elec_num+2))
        j = 0        
        for ch in grp[g]:
            fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % ch), 'r')
            raw = np.squeeze(fid['D'])

            # collect all spikes
            SpkMat = np.zeros((spk_num, total_samp))
            k=0
            for ii in spk_ts_curr:
                ii = int(ii)
                SpkMat[k,:] = raw[ii-pre:ii+post+1]
                k += 1
            
            fid.close()
            MinMax[:,j] = np.max(SpkMat, axis=1) - np.min(SpkMat, axis=1)
            if plaser:
                if len(LWave[ch])==0:
                    LWave[ch] = np.zeros((pre+post+1,))
                FetMat[:,j*nc+features] = ( np.sum(SpkMat * np.repeat( [LWave[ch]], spk_num, axis=0 ), axis=1) ) / 1000.0
            # Note: SpkMat is just a pointer! So any operation by pca will change the matrix
            score = pca(SpkMat.copy(), features)[0]
            FetMat[:,j*nc:j*nc+features] = score[:,0:features]  #The coordinates of the spikes
            
            j += 1
        
        FetMat[:,-2] = np.max(MinMax,axis=1)
        FetMat[:,-1] = spk_ts_curr
        FetMat = np.round(FetMat).astype('int')

        # write into file
        # description of file formats: http://klusters.sourceforge.net/UserManual/data-files.html
        
        # write feature .fet. file
        ### FORMAT
        # Spike 1: Coeff1-El1, Coeff2-El1, ... , Coeff1-El2, Coeff2-El2,...
        # Spike 2: Coeff1-El1, Coeff2-El1, ... , Coeff1-El2, Coeff2-El2,...
        # .
        # .
        # .    
        FetMat[np.where(np.isnan(FetMat))] = 0
        fetFileName = os.path.join(ppath, name, name + '.fet.' + str(g))
        if os.path.isfile(fetFileName):
            #remove existing fet file
            os.remove(fetFileName)
            
        f_handle = open(fetFileName, 'a')
        np.savetxt(f_handle, [FetMat.shape[1]], fmt="%d", delimiter=' ')
        np.savetxt(f_handle, FetMat, fmt="%d", delimiter= ' ')
        f_handle.close()
                
        ### Thw .res files contains all spike indices
        # Format:
        # Time of Spike 1
        # Time of Spike 2
        # .
        # .
        # .
        # Time of Spike m
        resFileName = os.path.join(ppath, name, name + '.res.' + str(g))
        np.savetxt(resFileName,spk_ts_curr, fmt='%d')
                
        # make .spk file for clustering
        print("Writing spike file for group %d ..." % g)
        spk_vec=np.zeros((total_samp*spk_num*elec_num,))
        #for j=1:elecNum
        j=0 # counter for electrode j
        for ch in grp[g]:
            fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % ch), 'r')
            raw = np.squeeze(fid['D'])
            
            for k in range(spk_num):
                if (spk_ts_curr[k] >= pre) and spk_ts_curr[k]+post<len(raw):
                    # elec_num*total_samp ... total number of data points sampled for each time point
                    # k ... spike index
                    # j ... electrode number
                    idx_vec = np.arange( elec_num*total_samp*k+j, elec_num*total_samp*(k+1), elec_num ) 
                    spk_vec[idx_vec] = raw[int(spk_ts_curr[k])-pre:int(spk_ts_curr[k])+post+1]
            j += 1
                        
        # Format:
        # sample1_electrode1_spike1
        # sample1_electrode2_spike1
        # ...
        # sample1_electrodeN2_spike1
        # sample2_electrode1_spike1
        # sample2_electrode2_spike1
        # ...
        # sample2_electrodeN2_spike1
        # ...
        # sampleN3_electrodeN2_spike1
        # ...
        # sample1_electrode1_spikeN1
        # ...
        # sampleN3_electrodeN2_spikeN1
        spkFileName = os.path.join(ppath, name, name + '.spk.' + str(g))
        spk_vec.astype('int16').tofile(spkFileName)
        pdb.set_trace()
            
    # write .xml configuration file
    print("Writing config file...")
    #var = "-g %s -d %s -r %s -p %d -q %d -n %d" % (grp_name, ppath, name, pre, post, nc)
    subprocess.Popen(["perl", "kluster_groups.pl", '-g', grp_name, '-d', ppath, '-r', name, '-p', str(pre), '-q', str(post), '-n', str(nc)])



def detect_spikes_correng(ppath, name, grp_name='', med_factor=4.0, spike_window=15, plaser=True, pspk=False,
                       rmthres=False, peng=True, igroup=0):
    """
    detect potential spikes, i.e. waveforms crossing a threshold
    The threshold is either determined by med_factor (med_factor * median of raw signal) or can be manually set
    using spike_view.py; for the second case, the threshold values for each electrode group are saved in the file thres.p

    The function generates the following features:
    PCA1-El1 PCA2-El1 PCA3-El1 Waveform-Corr Wakeform-Energy  PCA1-El2 PCA2-El2 PCA3-El2 Waveform-Corr Wakeform-Energy Timestamp
    i.e. each electrode comes with 5 features.


    @Parameters:
        ppath, name     recording
    @Optional:
        grp_name        grouping file for the current mouse
        med_factor      the threold a spike has to cross is given by the median of the
                        raw channel * $med_factor
        spike_window    number of indices surrounding the center of a potential spike waveform:
                        -15 to +15 with spike valley at 0
        plaser          If true, calculate waveform correlation between laser evoked an spontaneous spikes
                        If false, the corrleation is replaced with the 4th principal component
        pspk            if True, use existing spike times in spk_ts.p; otherwise recalculate spike times
                        for all groups or group igroup (if igroup > 0)
        rmthres         if True, remove existing threshold file (thres.p), and recalculate using
                        median * med_fractor
        peng            if True, use energy, otherwise use height of spikes
        igroup          integer, if 0, go through all electrode groups; otherwise only detect spikes
                        for group igroup
    """
    pup_spike = True
    pre = 10
    post = 15

    lockout_win = 5
    total_samp = pre + post + 1
    features = 3
    # if there's no laser, we just will up the empty feature with the 4th principal component
    #if not plaser:
    #    features = 4
    #    nc = features + 1
    #else:
    #    nc = features + 2
    nc = features + 2

    # duration of how long after laser pulse onset a spike is considered as laser triggered
    twin = 15  # ms

    # mouse id
    idf = re.split('_', name)[0]

    # set grouping file name
    if grp_name == '':
        grp_name = idf + '_grouping.txt'

    # prune channels to be analyzed
    grp = load_grpfile(ppath, grp_name)

    # remove channels that do not exist
    grp = prune_grp(ppath, name, grp)
    if igroup == 0:
        groups = list(grp.keys())
    else:
        groups = [igroup]

    # get all channels in use
    channels = []
    for g in groups:
        channels += grp[g]
    channels.sort()

    # load laser if it exists
    if os.path.isfile(os.path.join(ppath, name, 'laser_' + name + '.mat')) and plaser == True:
        laser = load_laser(ppath, name)
    else:
        plaser = False

    # True, if threshold file exists
    pthres = False
    # remove existing thres.p file
    if os.path.isfile(os.path.join(ppath, name, 'thres.p')) and rmthres == True:
        os.unlink(os.path.join(ppath, name, 'thres.p'))
        pthres = False
    # load existing thresholds from thres.p
    if os.path.isfile(os.path.join(ppath, name, 'thres.p')) and rmthres == False:
        with open(os.path.join(ppath, name, 'thres.p'), 'rb') as fp:
            try:
                tmp = pickle.load(fp)
            except UnicodeDecodeError:
                # to load pickle objects created using python 2
                tmp = pickle.load(fp, encoding='latin1')
            Thres = tmp['Thres']
            pthres = True
            fp.close()

    # pspk == True, i.e. load existing spike times and continue with writing .fet files
    if os.path.isfile(os.path.join(ppath, name, 'spk_ts.p')) and pspk == True:
        with open(os.path.join(ppath, name, 'spk_ts.p'), 'rb') as fp:
            try:
                tmp = pickle.load(fp)
            except UnicodeDecodeError:
                # to load pickle objects created using python 2
                tmp = pickle.load(fp, encoding='latin1')
            TS = tmp['TS']
            LWave = tmp['LWave']
            for ch in channels:
                if type(LWave[ch]) == np.float:
                    LWave[ch] = []
            fp.close()
    # re-calculated spike spikes:
    else:
        # if there's already a spk_ts.p file load it and then overwrite,
        # the existing values
        if os.path.isfile(os.path.join(ppath, name, 'spk_ts.p')):
            with open(os.path.join(ppath, name, 'spk_ts.p'), 'rb') as fp:
                try:
                    tmp = pickle.load(fp)
                except UnicodeDecodeError:
                    # to load pickle objects created using python 2
                    tmp = pickle.load(fp, encoding='latin1')
                TS = tmp['TS']
                LWave = tmp['LWave']
                for ch in channels:
                    if type(LWave[ch]) == np.float:
                        LWave[ch] = []
                fp.close()
        # no spk_ts.p file, initialize TS and LWave
        else:
            # time points (indices) of spikes on each channel
            TS = {}  # Dict: channel-id --> spike indices
            # average waveform on a specific channel
            LWave = {}  # Dict: channel_id --> mean spike waveform
        # if there's no threshold file:
        if not pthres:
            Thres = {}  # Dict: channel_ID --> threshold value

        for ch in channels:
            TS[ch] = []
            LWave[ch] = np.zeros((pre + post + 1,))
            print("Getting spike times on channel %d" % (ch))
            fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % ch), 'r')
            raw = np.squeeze(fid['D'])
            N = len(raw)

            if not pthres:
                med_noise = np.nanmedian(np.abs(raw)) / 0.6745
                Thres[ch] = med_noise * med_factor

            peak_idx = np.where(raw > (1.0 * Thres[ch]))[0]
            valley_idx = np.where(raw < (-1.0 * Thres[ch]))[0]

            if len(valley_idx) > 0:
                # first index of spike, interval between first and second spike etc.
                peakISI = np.concatenate(([peak_idx[0]], np.diff(peak_idx)))
                # peakISI=[peak_idx(1) peak_idx(2:end)-peak_idx(1:end-1)];

                valleyISI = np.concatenate(([valley_idx[0]], np.diff(valley_idx)))
                # valleyISI=[valley_idx(1) valley_idx(2:end)-valley_idx(1:end-1)];

                # interspike interval preceding first spike; interspike interval preceding 2nd spike etc.
                peak_idx = peak_idx[np.where(peakISI > spike_window)]
                valley_idx = valley_idx[np.where(valleyISI > spike_window)]

                spk_ts = []
                if pup_spike:
                    for i in peak_idx:
                        wave_tmp = raw[i - spike_window:i + spike_window]
                        # we search for the valley value closest to the peak
                        spk_ts.append(np.argmin(wave_tmp) + i - spike_window)

                for i in valley_idx:
                    wave_tmp = raw[i - spike_window:i + spike_window]
                    if wave_tmp[np.argmin(wave_tmp)] < (-1 * Thres[ch]):
                        spk_ts.append(np.argmin(wave_tmp) + i - spike_window)

                spk_ts.sort()
                spk_ts = np.array(spk_ts)
                spk_ts = spk_ts[np.where((spk_ts >= pre) & (spk_ts + post < len(raw)))]
                spk_ts = spk_ts[np.where(np.concatenate(([spk_ts[0]], np.diff(spk_ts))) > spike_window)]
                TS[ch] = spk_ts

                if len(spk_ts) > 0:
                    if plaser:
                        A = laser_waveform(laser, spk_ts, raw, pre, post, twin)
                        LWave[ch] = A.mean(axis=0)

            fid.close()

        # np.savez(os.path.join(ppath, name, 'spk_ts'), TS=TS, LWave=LWave, N=N)
        with open(os.path.join(ppath, name, 'spk_ts.p'), 'wb') as fp:
            pickle.dump({'TS': TS, 'LWave': LWave, 'N': N}, fp)
            fp.close()
        with open(os.path.join(ppath, name, 'thres.p'), 'wb') as fp:
            pickle.dump({'Thres': Thres}, fp)
            fp.close()

    # calculate features for each group
    f = lambda x, y: np.concatenate((x, y))
    for g in groups:
        # collect spike times on all channels
        spk_ts_curr = []
        for ch in grp[g]:
            spk_ts_curr.append(TS[ch])
        spk_ts_curr = np.unique(reduce(f, spk_ts_curr))
        # spk_ts = spk_ts[np.where((spk_ts>=pre) & (spk_ts+post<len(raw)))]
        spk_ts_curr.sort()

        # apply lockout window
        spk_ts_curr = spk_ts_curr[np.where(np.concatenate(([spk_ts_curr[0]], np.diff(spk_ts_curr))) > lockout_win)]
        elec_num = len(grp[g])
        spk_num = len(spk_ts_curr)
        #MinMax = np.zeros((spk_num, elec_num))
        FetMat = np.zeros((len(spk_ts_curr), nc * elec_num + 1))
        j = 0
        for ch in grp[g]:
            fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % ch), 'r')
            raw = np.squeeze(fid['D'])
            # collect all spikes
            SpkMat = np.zeros((spk_num, total_samp))
            k = 0
            for ii in spk_ts_curr:
                ii = int(ii)
                SpkMat[k, :] = raw[ii - pre:ii + post + 1]
                k += 1
            fid.close()

            #MinMax[:, j] = np.max(SpkMat, axis=1) - np.min(SpkMat, axis=1)
            # Note: SpkMat is just a pointer! So any operation by pca will change the matrix
            # perform PCA
            score = pca(SpkMat.copy(), features+1)[0]

            if plaser:
                # somehow necessary:
                if type(LWave[ch]) == np.float:
                    LWave[ch] = []
                if len(LWave[ch]) == 0:
                    LWave[ch] = np.zeros((pre + post + 1,))
                FetMat[:, j * nc + features] = (np.sum(SpkMat * np.repeat([LWave[ch]], spk_num, axis=0), axis=1)) / 1000.0
            else:
                FetMat[:, j * nc + features] = score[:, features]

            FetMat[:, j * nc:j * nc + features] = score[:, 0:features]  # The coordinates of the spikes
            if not peng:
                FetMat[:, j * nc + features + 1] = np.max(SpkMat, axis=1) - np.min(SpkMat, axis=1)
            else:
                FetMat[:, j * nc + features + 1] = np.sqrt(np.sum(SpkMat**2, axis=1) / SpkMat.shape[1])

            j += 1

        FetMat[:, -1] = spk_ts_curr
        FetMat = np.round(FetMat).astype('int')
        # write into file
        # description of file formats: http://klusters.sourceforge.net/UserManual/data-files.html

        # write feature .fet. file
        ### FORMAT
        # Spike 1: Coeff1-El1, Coeff2-El1, ... , Coeff1-El2, Coeff2-El2,...
        # Spike 2: Coeff1-El1, Coeff2-El1, ... , Coeff1-El2, Coeff2-El2,...
        # .
        # .
        # .
        FetMat[np.where(np.isnan(FetMat))] = 0
        fetFileName = os.path.join(ppath, name, name + '.fet.' + str(g))
        if os.path.isfile(fetFileName):
            # remove existing fet file
            os.remove(fetFileName)

        f_handle = open(fetFileName, 'a')
        np.savetxt(f_handle, [FetMat.shape[1]], fmt="%d", delimiter=' ')
        np.savetxt(f_handle, FetMat, fmt="%d", delimiter=' ')
        f_handle.close()

        ### Thw .res files contains all spike indices
        # Format:
        # Time of Spike 1
        # Time of Spike 2
        # .
        # .
        # .
        # Time of Spike m
        resFileName = os.path.join(ppath, name, name + '.res.' + str(g))
        np.savetxt(resFileName, spk_ts_curr, fmt='%d')

        # make .spk file for clustering
        print("Writing spike file for group %d ..." % g)
        spk_vec = np.zeros((total_samp * spk_num * elec_num,))
        # for j=1:elecNum
        j = 0  # counter for electrode j
        for ch in grp[g]:
            fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % ch), 'r')
            raw = np.squeeze(fid['D'])

            for k in range(spk_num):
                if (spk_ts_curr[k] >= pre) and spk_ts_curr[k] + post < len(raw):
                    # elec_num*total_samp ... total number of data points sampled for each time point
                    # k ... spike index
                    # j ... electrode number
                    idx_vec = np.arange(elec_num * total_samp * k + j, elec_num * total_samp * (k + 1), elec_num)
                    spk_vec[idx_vec] = raw[int(spk_ts_curr[k]) - pre:int(spk_ts_curr[k]) + post + 1]
            j += 1

        # Format:
        # sample1_electrode1_spike1
        # sample1_electrode2_spike1
        # ...
        # sample1_electrodeN2_spike1
        # sample2_electrode1_spike1
        # sample2_electrode2_spike1
        # ...
        # sample2_electrodeN2_spike1
        # ...
        # sampleN3_electrodeN2_spike1
        # ...
        # sample1_electrode1_spikeN1
        # ...
        # sampleN3_electrodeN2_spikeN1
        spkFileName = os.path.join(ppath, name, name + '.spk.' + str(g))
        spk_vec.astype('int16').tofile(spkFileName)

    # write .xml configuration file
    print("Writing config file...")
    subprocess.Popen(
        ["perl", "kluster_groups.pl", '-g', grp_name, '-d', ppath, '-r', name, '-p', str(pre), '-q', str(post), '-n',
         str(nc)])



def spike_threshold(data, th, sign=1):
    """
    function used by detect_spikes_corr2; potential spikes are detected as waveforms crossing the given threshold th either
    upwards (if sign=-1) or downwards (if sign = 1)
    :param data: spiking raw data
    :param th: threshold to pass to qualify as a spike
    :param sign: if sign==1, "valleys" are considered as spikes; if sign==-1, spikes are "mountains"
    :return: indices of spike waveforms
    """
    if sign == 1:
        lidx  = np.where(data[0:-2] > data[1:-1])[0]
        ridx  = np.where(data[1:-1] <= data[2:])[0]
        thidx = np.where(data[1:-1]<(-1*th))[0]

        sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1        
    else:
        lidx = np.where(data[0:-2] < data[1:-1])[0]
        ridx = np.where(data[1:-1] >= data[2:])[0]
        thidx = np.where(data[1:-1]>th)[0]

        sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1
                
    return sidx



def detect_spikes_corr2(ppath, name, grp_name='', med_factor=4.0, spike_window=15, plaser=True):
    """
    detect potential spikes, i.e. waveforms crossing a threshold

    Similar to detect_spikes_corr, with with two major differences: For detect_spikes_corr, the pca and laser correlation coefficients are calculated
    for each electrode (in a group). For detect_spikes_corr2 I calculate pca and laser corr coefficients for all waveforms
    at once. Consequently, in klusters there are consequently
    3 (pca components) + 1 (laser correlation) + 1 (spike size) + 1 (timestamp) = 6 compoments,
    irrespective of the number of electrodes

    Second, to detect potential spikes, the function simply just uses a simple threshold: All waveforms that pass
    (upwards or downwards) the threshold are considered as spikes.

    @Parameters:
        ppath, name     recording
    @Optional:
        grp_name        grouping file for the current mouse
        med_factor      the threold a spike has to cross is given by the median of the 
                        raw channel * $med_factor
        spike_window    number of indices surrounding the center of a potential spike waveform: 
                        -15 to +15 with spike valley at 0
    
    """
    pup_spike = True
    pre=10
    post = 15

    lockout_win = 5
    total_samp=pre+post+1
    if plaser:
        features=3
        nc = features+1
    else:
        features = 4
        nc = features

    # duration of how long after laser pulse onset a spike is considered as laser triggered
    twin = 15 #ms
        
    # mouse id
    idf = re.split('_', name)[0]
    
    # set grouping file name
    if grp_name == '':
        grp_name = idf + '_grouping.txt'

    # prune channels to be analyzed
    grp = load_grpfile(ppath, grp_name)

    # remove channels that do not exist
    grp = prune_grp(ppath, name, grp)

    # get existing channels
    channels = []
    for g in grp:
        channels += grp[g]
    channels.sort()

    # load laser if it exists
    if os.path.isfile(os.path.join(ppath, name, 'laser_' + name + '.mat')) and plaser == True:
        laser = load_laser(ppath, name)
    else:
        plaser=False

    if os.path.isfile(os.path.join(ppath, name, 'spk_ts.p')):
        with open(os.path.join(ppath, name, 'spk_ts.p'), 'rb') as fp:
            try:
                tmp = pickle.load(fp)
            except UnicodeDecodeError:
                # to load pickle objects created using python 2
                tmp = pickle.load(fp, encoding='latin1')
            TS = tmp['TS']
            LWave = tmp['LWave']
            for ch in channels:
                if type(LWave[ch]) == np.float:
                    LWave[ch] = []
            fp.close()
    else:
        # time points (indices) of spikes on each channel
        TS = {} # Dict: channel-id --> spike indices 
        # average waveform on a specific channel
        LWave = {} # Dict: channel_id --> mean spike waveform
        for ch in channels:
            TS[ch] = []
            LWave[ch] = np.zeros((pre+post+1,))
            print("Getting spike times on channel %d" % (ch))
            fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % ch), 'r')
            raw = np.squeeze(fid['D'])
            N = len(raw)
            
            med_noise = np.nanmedian(np.abs(raw))/0.6745
            valley_idx = spike_threshold(raw, med_noise*med_factor, sign=1)
            peak_idx   = spike_threshold(raw, med_noise*med_factor, sign=-1)
            if len(valley_idx) > 0:
                # first index of spike, interval between first and second spike etc.
                #peakISI = np.concatenate(([peak_idx[0]], np.diff(peak_idx)))            
                
                #valleyISI = np.concatenate(([valley_idx[0]], np.diff(valley_idx)))
                
                # interspike interval preceding first spike; interspike interval preceding 2nd spike etc.
                #peak_idx   = peak_idx[np.where(peakISI>spike_window)]
                #valley_idx = valley_idx[np.where(valleyISI>spike_window)] 

                spk_ts = valley_idx.tolist()
                if pup_spike:
                    for i in peak_idx:
                        if i > spike_window:
                            wave_tmp = raw[i-spike_window:i+spike_window]
                            spk_ts.append( np.argmin(wave_tmp) + i - spike_window )

                spk_ts.sort()            
                spk_ts = np.array(spk_ts)
                spk_ts = spk_ts[np.where((spk_ts>=pre) & (spk_ts+post<len(raw)))]
                spk_ts = spk_ts[np.where( np.concatenate(([spk_ts[0]], np.diff(spk_ts))) > spike_window)]
                TS[ch] = spk_ts        

                if len(spk_ts) > 0:
                    if plaser:
                        A = laser_waveform(laser, spk_ts, raw, pre, post, twin)
                        LWave[ch] = A.mean(axis=0)
            fid.close()

        #np.savez(os.path.join(ppath, name, 'spk_ts'), TS=TS, LWave=LWave, N=N)
        with open(os.path.join(ppath, name, 'spk_ts.p'), 'wb') as fp:
            pickle.dump({'TS':TS, 'LWave':LWave, 'N':N}, fp)
            fp.close()


    # calculate features for each group
    f = lambda x,y: np.concatenate((x,y))
    for g in grp:
        # collect spike times on all channels
        spk_ts_curr = []
        for ch in grp[g]:
            spk_ts_curr.append(TS[ch])
        spk_ts_curr = np.unique(reduce(f, spk_ts_curr))
        #spk_ts = spk_ts[np.where((spk_ts>=pre) & (spk_ts+post<len(raw)))]
        
        spk_ts_curr.sort()
        
        #apply lockout window
        spk_ts_curr = spk_ts_curr[np.where( np.concatenate(([spk_ts_curr[0]], np.diff(spk_ts_curr))) > lockout_win )]
        elec_num = len(grp[g])
        spk_num = len(spk_ts_curr)
        MinMax = np.zeros((spk_num, elec_num))
        SpkCorr = np.zeros((spk_num, elec_num))
        FetMat = np.zeros((spk_num, nc+1))
        SpkMatAll = np.zeros((spk_num, elec_num*total_samp))
        j = 0  # iteration index across electrodes
        for ch in grp[g]:
            fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % ch), 'r')
            raw = np.squeeze(fid['D'])

            # collect all spikes
            SpkMat = np.zeros((spk_num, total_samp))
            k=0
            for ii in spk_ts_curr:
                ii = int(ii)
                SpkMat[k,:] = raw[ii-pre:ii+post+1]
                SpkMatAll[k,j*total_samp:(j+1)*total_samp] = raw[ii - pre:ii + post + 1]
                k += 1
            
            fid.close()
            MinMax[:,j] = np.max(SpkMat, axis=1) - np.min(SpkMat, axis=1)
            if plaser:
                if len(LWave[ch])==0:
                    LWave[ch] = np.zeros((pre+post+1,))
                SpkCorr[:,j] = ( np.sum(SpkMat * np.repeat( [LWave[ch]], spk_num, axis=0 ), axis=1) ) / 1000.0
            # Note: SpkMat is just a pointer! So any operation by pca will change the matrix
            j += 1

        score = pca(SpkMatAll.copy(), features)[0]
        FetMat[:,0:features] = score[:,0:features]  #The coordinates of the spikes
        FetMat[:,-3] = np.max(SpkCorr,axis=1)
        FetMat[:,-2] = np.max(MinMax,axis=1)
        FetMat[:,-1] = spk_ts_curr
        FetMat = np.round(FetMat).astype('int')

        # write into file
        # description of file formats: http://klusters.sourceforge.net/UserManual/data-files.html
        
        # write feature .fet. file
        ### FORMAT
        # Spike 1: Coeff1-El1, Coeff2-El1, ... , Coeff1-El2, Coeff2-El2,...
        # Spike 2: Coeff1-El1, Coeff2-El1, ... , Coeff1-El2, Coeff2-El2,...
        # .
        # .
        # .    
        FetMat[np.where(np.isnan(FetMat))] = 0
        fetFileName = os.path.join(ppath, name, name + '.fet.' + str(g))
        if os.path.isfile(fetFileName):
            #remove existing fet file
            os.remove(fetFileName)
            
        f_handle = open(fetFileName, 'a')
        np.savetxt(f_handle, [FetMat.shape[1]], fmt="%d", delimiter=' ')
        np.savetxt(f_handle, FetMat, fmt="%d", delimiter= ' ')
        f_handle.close()
                
        ### Thw .res files contains all spike indices
        # Format:
        # Time of Spike 1
        # Time of Spike 2
        # .
        # .
        # .
        # Time of Spike m
        resFileName = os.path.join(ppath, name, name + '.res.' + str(g))
        np.savetxt(resFileName,spk_ts_curr, fmt='%d')
                
        # make .spk file for clustering
        print("Writing spike file for group %d ..." % g)
        spk_vec=np.zeros((total_samp*spk_num*elec_num,))
        #for j=1:elecNum
        j=0
        for ch in grp[g]:
            fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % ch), 'r')
            raw = np.squeeze(fid['D'])
            
            for k in range(spk_num):
                if (spk_ts_curr[k] >= pre) and spk_ts_curr[k]+post<len(raw):
                    # elec_num*total_samp ... total number of data points sampled for each time point
                    # k ... spike index
                    # j ... electrode number
                    idx_vec = np.arange( elec_num*total_samp*k+j, elec_num*total_samp*(k+1), elec_num )                    
                    spk_vec[idx_vec] = raw[int(spk_ts_curr[k])-pre:int(spk_ts_curr[k])+post+1]
            j += 1
                        
        # Format:
        # sample1_electrode1_spike1
        # sample1_electrode2_spike1
        # ...
        # sample1_electrodeN2_spike1
        # sample2_electrode1_spike1
        # sample2_electrode2_spike1
        # ...
        # sample2_electrodeN2_spike1
        # ...
        # sampleN3_electrodeN2_spike1
        # ...
        # sample1_electrode1_spikeN1
        # ...
        # sampleN3_electrodeN2_spikeN1
        spkFileName = os.path.join(ppath, name, name + '.spk.' + str(g))
        spk_vec.astype('int16').tofile(spkFileName)
            
    # write .xml configuration file
    print("Writing config file...")
    #var = "-g %s -d %s -r %s -p %d -q %d -n %d" % (grp_name, ppath, name, pre, post, nc)
    subprocess.Popen(["perl", "kluster_groups.pl", '-g', grp_name, '-d', ppath, '-r', name, '-p', str(pre), '-q', str(post), '-n', str(nc)])         
    
    

def klusters_spike_extract(ppath, name, groups, group_name='', pnoise=True, sep_mode=False):
    """
    convert spike times of a cluster saved in klusters to npz format
    """
    
    (pre, post) = (30, 30)
    
    FF = 16
    SR = get_snr(ppath, name)
    if SR==1000:
        FF = 25

    idf = re.split('_', name)[0]
    if group_name == '':
        group_name = idf + '_grouping.txt'

    # dict: group id --> channel id
    grp_dict = load_grpfile(ppath, group_name)
    
    # get number of time bins for spike channels
    if os.path.isfile(os.path.join(ppath, name, 'spk_ts.p')):
        fp = open(os.path.join(ppath, name, 'spk_ts.p'), 'rb')
        try:
            N = pickle.load(fp)['N']
        except UnicodeDecodeError:
            # to load pickle objects created using python 2
            fp.close()
            fp = open(os.path.join(ppath, name, 'spk_ts.p'), 'rb')
            N = pickle.load(fp, encoding='latin1')['N']
        fp.close()
    else:
        N = len(load_laser(ppath, name))*FF
    
    start_idx = 1
    if not pnoise:
        start_idx = 2
        
    for i in groups:
        # needed to calculate spike waveforms
        chs = grp_dict[i]
        
        # the clu file is generated by klusters and contains a sequence of integers
        # the file is as long as the .res. file (+ the inital entry) and 
        # specifies for each spike its cluster ID
        clu_file = os.path.join(ppath, name, name + '.clu.' + str(i))
        clu_vec = np.loadtxt(clu_file, dtype='int')
        num_clu = clu_vec[0]
        clu_vec = clu_vec[1:]
        
        # specifies for each spike the time index sampled at raw channel SR
        res_file = os.path.join(ppath, name, name + '.res.' + str(i))
        res_vec = np.loadtxt(res_file, dtype='int')

        SpikeIdx = {}   # indices of spikes on raw channels
        SpikeIdxd = {}  # indices of spikes in EEG time
        Train = {}      # Spike train binned in EEG time 
        MSpike = {}     # Average spike on "largest" channel for unit j
        MaxCh = {}      # channel on with unit j is largest    
        units = []      # units in group g
        if start_idx > 1:
            SpikeIdx[1] = []
            units = ['empty']
        for j in range(start_idx, num_clu+1):
            idx = res_vec[np.where(clu_vec==j)]
            # indices of spikes belonging to cluster j at raw channel SR
            SpikeIdx[j] = idx
                
            # downsamples spike train to LFP resolution
            #spikes = scipy.sparse.lil_matrix((N,1))
            spikes = np.zeros((N,))            
            spikes[idx] = SR*FF
            Train[j] = downsample_vec(spikes, FF)
            SpikeIdxd[j] = np.where(Train[j] > 0)[0]

            MS = np.zeros((len(idx), pre+post))
            U = np.zeros((len(chs), pre+post))
            ch_count = 0
            for c in chs:
                fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % c), 'r')
                D = fid['D']
                count = 0
                for k in idx:
                    if (k-pre >= 0 and k+post < N):
                        MS[count,:] = D[k-pre+1:k+post+1,0]
                        count += 1
                fid.close()
                U[ch_count,:] = MS.mean(axis=0)
                ch_count += 1

            P = np.zeros((len(chs),))
            for l in range(len(chs)):
                P[l] = np.max(U[l,:]) - np.min(U[l,:])
            ii = np.argmax(P)
            MaxCh[j] = chs[ii]
            MSpike[j] = U[ii,:]
            units.append(j)

            if sep_mode:
                np.savez(os.path.join(ppath, name, 'Spk%d-%d.npz' % (i, j)),
                        idx=SpikeIdx[j], idxd=SpikeIdxd[j], mspike=MSpike[j], train=Train[j], ch=MaxCh[j])

        #with open(os.path.join(ppath, name, 'Spk%d.p' % i), 'w') as fp:
        #    pickle.dump({'idx':SpikeIdx, 'idxd':SpikeIdxd, 'mspike':MSpike, 'train':Train, 'ch':MaxCh}, fp)
        if not sep_mode:
            np.savez(os.path.join(ppath, name, 'Spk%d.npz' % i),
                     idx=SpikeIdx, idxd=SpikeIdxd, mspike=MSpike, train=Train, ch=MaxCh, units=units)
        #fid = h5py.File(os.path.join(ppath, name, 'Spk%d.hdf5' % i), 'w')
        #fid['idx']    = SpikeIdx
        #fid['idxd']   = SpikeIdxd
        #fid['mspike'] = MSpike
        #fid['train']  = Train
        #fid['ch']     = MaxCh
        #fid['units']  = units
        #fid.close()



def unit_quality(ppath, name, igroup, nfet=3):
    """
    implementation of L_ratio and Isoluation Distance as described 
    in Schmitzer-Torbert2005
    
    @Paramters:
        ppath, name          recording base folder and recording name
        igroup               electrode group           
    
    @Return:
        Lratio, IDist        L_ratio and Isolation Distance; dicts: cluster_id --> L_ratio, I.D.
    
    """
    
    # my standard values:
    npre = 10 
    npost = 15
    nsample = npre+npost+1
        
    idf = re.split('_', name)[0]
    grp_name = idf + '_grouping.txt'
    spk_grps = load_grpfile(ppath, grp_name)
    nelec = len(spk_grps[igroup])

    clu_file = os.path.join(ppath, name, name + '.clu.' + str(igroup))
    
    clu_vec = np.loadtxt(clu_file)
    nclu  = int(clu_vec[0]) # number of clusters
    idclu = clu_vec[1:]     # cluster ID vector
    #nspikes = np.zeros((nclu,))
    nspikes = {}
    for i in range(1, nclu+1):
        nspikes[i] = len(np.where(idclu == i)[0])

    nspikes_total = sum(list(nspikes.values()))

    # indices of spikes for cluster 1..nclu
    ispikes = {} 
    for i in range(1, nclu+1):
        ispikes[i] = np.where(idclu==i)[0]
        
    ### load spk file ##########
    ddir = os.path.join(ppath, name)
    sfile = name + '.spk.' + str(igroup)
    fid = open(os.path.join(ddir, sfile), 'rU')
    # the vector contains all waveforms
    data = np.fromfile(fid, count=nspikes_total*nelec*nsample, dtype='int16')
    fid.close()
    
    ### read waveforms out of @data into matrix @Spikes, where each row
    ### corresponds to a spike waveform on the $nelec electrodes (just
    ### concatenated)
    Elec = np.zeros((nelec, nsample))
    Spikes = np.zeros((nspikes_total, nsample*nelec))
    # number of spikes per cluster:
    nset = {i:0 for i in range(1,nclu+1)}
    n = nsample*nelec
    for i in range(nspikes_total):        
        s = data[(i*n):(i+1)*n]
        for j in range(nelec):
            Elec[j,:] = s[j::nelec]
        el = np.reshape(Elec, [1, nelec*nsample])        
        idc = int(idclu[i])
        nset[idc] = nset[idc] + 1
        Spikes[i,:] = el

    ### test if one of the electrodes is the reference
    for i in range(nelec):
        B = Spikes[:,i*nsample:(i+1)*nsample]
        if B.sum() == 0:
            # electrode $i is reference
            c = Spikes.shape[1]
            d = np.setdiff1d(range(c), range(i*nsample,(i+1)*nsample))
            Spikes = Spikes[:,d]
            nelec = nelec-1
            print('Kicked out the reference electrode')
            break
        
    ### calculate features
    ### preallocate feature matrices
    Features = {}
    for i in range(1, nclu+1):
        Features[i] = np.zeros((nspikes_total, nfet*nelec))

    
    ### (1) energy in each spike on each electrode
    ### calculate the energy of each spike on each electrode
    Eng = np.zeros((nspikes_total, nelec))
    for i in range(nelec):
        # the the L2 norm for energy calculation:
        Eng[:,i] = np.sqrt( np.sum(Spikes[:,i*nsample:(i+1)*nsample]**2, axis=1) )        
        ### NOTE: the formula in the paper shows no square root, but the text
        ### describing the calculation of energy talks about the square root
    
    ### (2) first principal component
    for i in range(1, nclu+1):
        for j in range(nelec):
            ### calculate the pca for the spikes belonging to cluster $i
            C = Spikes[ispikes[i], j*nsample:(j+1)*nsample]                        
            C = np.divide(C, np.tile(Eng[ispikes[i],j], (nsample,1)).T)
            evecs = pca(C)[2]
            #if size(coords,2) == 0
            #    coords = zeros(size(coords,1), nfet);
            #end
            
            ### project ALL spikes into this pca space
            nSpikes = np.divide(Spikes[:,j*nsample:(j+1)*nsample], np.tile(Eng[:,j], (nsample,1)).T)        
            Features[i][:,j*nfet:(j+1)*nfet-1] = np.dot(nSpikes, evecs[:,:nfet-1])
        
    ### add the energy to the features
    for i in range(1, nclu+1):
        for j in range(nelec):
            Features[i][:,j*nfet-1] = Eng[:,j]; 

    ### Next, calculate M. Distance 
    ### the distance is calculated for each spike and each cluster
    ### pre-allocate cell array for distances
    Dist = {}
    for i in range(1, nclu+1):
        ### mean spike waveform for cluster $i
        mu = np.mean(Features[i][idclu==i,:], axis=0)
        
        ### x_i - mu for ALL spikes
        x = Features[i]-np.tile(mu, (Features[i].shape[0], 1))
        
        ### inverse of covariance matrix for the features of cluster $i
        R = np.cov(Features[i][ispikes[i],:], rowvar=False)
        icov = LA.solve(R, np.eye(nfet*nelec))

        ### calculate (x_i - mu)' icov * (x_i - mu)
        tmp = np.dot(x, icov) ### tmp is nspikes x nfet
        #Dist[i] = sum(tmp .* x, 2)
        Dist[i] = (tmp*x).sum(axis=1)
    
    ### get ids of spikes not belonging to cluster $i
    no_spikes = {}
    for i in range(1, nclu+1):
        no_spikes[i] = np.setdiff1d(range(nspikes_total), np.where(idclu==i)[0])

    LDist = {i:np.zeros((len(no_spikes[i]),)) for i in range(1, nclu+1)}
    for i in no_spikes:
        j=0
        for k in no_spikes[i]:
            LDist[i][j] = 1-scipy.stats.chi2.cdf(Dist[i][k], nfet*nelec);
            j += 1
    
    Lratio = {}
    for i in range(1, nclu+1):
        Lratio[i] = np.sum(LDist[i], axis=0) / nspikes[i]
    
    Dr = {}
    for i in range(1, nclu+1):
        Dr[i] = np.mean(Dist[i])
    
    ### Calculate Isolation Distance
    IDist = {i:-1 for i in range(2,nclu+1)}
    plt.figure()
    for i in range(2, nclu+1):
        plt.subplot(nclu-1, 1, i-1)
        if len(ispikes[i]) <= len(no_spikes[i]):
            noise_dist = Dist[i][no_spikes[i]]
            noise_dist.sort()
            IDist[i] = noise_dist[len(ispikes[i])]
            
            spikes_dist = Dist[i][ispikes[i]]
            spikes_dist.sort()
            
            plt.plot(noise_dist, color='k', linewidth=2)
            plt.plot(spikes_dist, color=[1, 0.75, 0], linewidth=2)
            plt.ylim([0, IDist[i]+IDist[i]/4])
            plt.xlim((0, 2*len(ispikes[i])))
            
        else:
            print('Cluster %d: Isolation Distance not defined' % i);

        plt.xlabel('D^2')
        plt.ylabel('Cumulative # Points')
    plt.draw()


    ### make figure showing the distances distr for each cluster
    plt.figure()
    edges = np.linspace(0, 100, 501)
    for i in range(2, nclu+1):
        
        H = {}
        for j in range(1, nclu+1):
            H[j] = np.histogram(Dist[i][ispikes[j]], edges)[0]
        
        plt.subplot(nclu-1, 1, i-1);
        plt.title('Cluster %d' % i)
        for j in range(1, nclu+1):
            H[j] = H[j] / (1.0*np.sum(H[j]))            
            ### that's the noise cluster
            if j==1:
                plt.plot(edges[0:-1], H[j], color=[1, 0.75, 0], linewidth=2)
            ### that's the currently considered cluster
            elif j==i:
                plt.plot(edges[0:-1], H[j], color='g', linewidth=2);
            else:
                plt.plot(edges[0:-1], H[j], color=[0.8, 0.8, 0.8], linewidth=2)                    
        plt.xlabel('D^2') 
        plt.ylabel('Count')
    plt.show(block=False)

    for i in IDist:
        print("Cluster %d LRatio: %f, IDist: %f" % (i, Lratio[i], IDist[i]))
    
    return Lratio, IDist



def write_unit_quality(ppath, name, igroup):
    """
    write output of unit_quality into a text file called qual$igroup\.txt
    This file is saved to the folder $ppath/$name
    """
    lr, idist = unit_quality(ppath, name, igroup)
    fid = open(os.path.join(ppath, name, 'qual' + str(igroup) + '.txt'), 'w')

    for g in lr:
        # if the isolation distance is not defined,
        # set it to -1
        b = -1
        if g in idist:
            b = idist[g]
        fid.write('%d:\t%f\t%f' % (g, lr[g], b))
        fid.write(os.linesep)

    fid.close()


    
def unpack_unit(ppath, name, grp, un):
    """
    Load a unit from the Spk%d.mat (old version) or Spk%d.npz (new version) file
    !! Units are counted as in klusters: 1 - Noise; 2 ... n, are clusters/units 1 ... n-1 !!
    @Parameters:
        ppath,name      recording
        (grp, un)       Unit $un of electrode/group $grp
    @Return:
        idx             indices of spikes at raw SR (25kHz)
        train           spike train in EEG SR resolution (1.5 kHz)
        mspike          average spike waveform
        ch              channel where waveform is largest
    """
    sfile = os.path.join(ppath, name, 'Spk' + str(grp))
    #files = os.listdir(os.path.join(ppath, name))
    #files = [f for f in files if re.match('^Spk' + str(grp) + '-\d+\.npz$', f)]
    spk_file = sfile + '-' + str(un) + '.npz'
    print("Loading unit %s, (%d, %d)" % (name, grp, un))
    if os.path.isfile(spk_file):
        #spk_file = os.path.join(ppath, name, 'Spk' + str(grp) + '-' + str(un) + '.npz')
        Spk = np.load(spk_file, encoding='bytes')
        train = Spk['train']
        mspike = Spk['mspike']
        idx = Spk['idx']
        ch = int(Spk['ch'])
        #return idx, train, mspike, ch

    elif os.path.isfile(sfile + '.mat'):
        # subtract -1 to get the right list index; klusters starts counting with 1
        un -= 1
        Spk = so.loadmat(sfile, struct_as_record=False, squeeze_me=True)['S']
        train = np.squeeze(Spk[un].train.toarray())
        mspike = Spk[un].mspike
        idx = Spk[un].idx
        ch  = Spk[un].ch
    elif os.path.isfile(sfile + '.npz'):
        Spk = np.load(sfile + '.npz', encoding='bytes', allow_pickle=True)
        # explanation of why this is necessary:
        # https://stackoverflow.com/questions/22661764/storing-a-dict-with-np-savez-gives-unexpected-result/41566840
        Spk = {key:Spk[key].item() for key in Spk if Spk[key].dtype == 'O'}
        train = Spk['train'][un]
        mspike = Spk['mspike'][un]
        idx = Spk['idx'][un]
        ch = Spk['ch'][un]
        
        #NEW 07/27/21
        print('saving %s' % (spk_file))
        np.savez(spk_file, idx=idx, mspike=mspike, train=train, ch=ch)
        
    else:
        print("Unit %s (%d, %d) does not seem to exist" % (name, grp, un))
        return

    return idx, train, mspike, ch
    
       

def firing_rate(ppath, name, grp, un):
    """
    calculate the firing rate for unit ($grp, $un) of recording $ppath/$name
    the firing rate is calculated by downsampling the spike train by summing
    up $NBIN consecutive bins. There's only one special thing happening here:

    If there's a laser stimulation period interpolate this period between the
    firing rate values of the rate directly preceding and following the
    timulation interval.
    """
    # Fixed Parameters
    # Time window to estimate firing rate preceding and following laser
    nmean = 3 

    # load spike train
    SR = get_snr(ppath, name)
    NBIN = int(np.round(SR)*2.5)
    train = unpack_unit(ppath, name, grp, un)[1]
    spikesd = downsample_vec(train, NBIN)

    # load laser
    plaser = False
    if os.path.isfile(os.path.join(ppath, name, 'laser_' + name + '.mat')):
        plaser = True
        Laser = load_laser(ppath, name)
        (idxs, idxe) = sleepy.laser_start_end(Laser, SR=SR)

        # don't understand why idxs is sometimes None??? Need to fix
        if np.sum(Laser) == 0:
            plaser = False

    if plaser:    
        # downsample EEG time to spectrogram time    
        idxs = [int(i/NBIN) for i in idxs]
        idxe = [int(i/NBIN) for i in idxe]
    
        for (a,b) in zip(idxs, idxe):
            # firing rate preceding laser:
            if a-nmean > 0:
                f1 = np.mean(spikesd[a-nmean:a])
            else:
                f1 = 0.0
                
            # firing rate following laser:
            if b+nmean < len(spikesd):
                f2 = np.mean(spikesd[b+1:b+nmean+1])
            else:
                f2 = f1
                
            spikesd[a:b+1] = np.linspace(f1, f2, b-a+1)
        
    return spikesd


        
def state_firing_rate(ppath, name, grp, un, pzscore=False, pplot=1):
    """
    average firing rate of given unit during NREM, REM, and Wake
    return: firing rate; dict mapping state onto average firing rate
    :param pplot: if pplot==1, plot figure using plotly, if pplot==2 use matplotlib
    """    
    spikesd = firing_rate(ppath, name, grp, un)
    
    if pzscore:
        spikesd = (spikesd - np.mean(spikesd)) / np.std(spikesd)

    M = sleepy.load_stateidx(ppath, name)[0]
    min_len = np.min([M.shape[0], spikesd.shape[0]])
    M = M[0:min_len]
    spikesd = spikesd[0:min_len]

    state_idx = {}
    FR = {i:[] for i in [2, 3, 1]}
    for i in range(1,4):
        state_idx[i] = np.where(M==i)[0]
        FR[i] = spikesd[state_idx[i]]

    # generate plot using plotly   
    if pplot==1:
        data = go.Bar(x = ['Wake','NREM', 'REM'],
                      y=[FR[i].mean() for i in [2, 3, 1]],
                      error_y=dict(type='data',
                                   array=[FR[i].std()/np.sqrt(len(FR[i])) for i in [2, 3, 1]],
                                   visible=True))

        layout = go.Layout(yaxis=dict(title='Firing rate (spikes/s)'),
                           font=dict(size=18),
                           autosize=False,
                           width=500, height=700)

        fig = go.Figure(data=[data], layout = layout)
        plotly.offline.plot(fig, filename='state_firing_rate.html')

    # generate plot using matplotlib
    if pplot==2:
        fr_mean = [FR[i].mean() for i in [2, 3, 1]]
        fr_std  = [FR[i].std()/np.sqrt(len(FR[i])) for i in [2, 3, 1]]
        plt.ion()
        plt.figure()
        ax = plt.axes([0.3, 0.1, 0.35, 0.8])
        ax.bar([1, 2, 3], fr_mean, yerr=fr_std, align='center', color='gray')
        sleepy.box_off(ax)
        plt.xticks([1, 2, 3], ['Wake', 'NREM', 'REM'], rotation=30)
        plt.ylabel('Spikes s$^{-1}$')
        plt.title(name)
        plt.show()

    return FR



def brstate_fr(ppath, name, units, offline_plot=True):
    """
    plot firing rates along with brain state using plotly.
    For example, brstate_fr(ppath, 'F14_022818n1', [(5,2)])

    @Parameters:
        units           The units to be plotted; list of tuples; So if you want to plot
                        unit 2 of group 5 and unit 3 of group 4, the argument is [(5,2), (4,3)]    
    """

    # load brain state
    mu = [10, 100]
    
    M,S = sleepy.load_stateidx(ppath, name)
    M[np.where(M==0)] = 3
    # load spectrogram
    SP = np.squeeze(so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name))['SP'])[0:40,:]
    freq = np.squeeze(so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name))['freq'])
    im = np.where((freq > mu[0]) & (freq<=mu[1]))[0]
    ampl = np.sqrt(np.squeeze(so.loadmat(os.path.join(ppath, name, 'msp_%s.mat' % name))['mSP'])[im,:].sum(axis=0)*(freq[1]-freq[0]))    
    
    fr = []
    for neuron in units:
        (grp, un) = neuron
        fr.append(firing_rate(ppath, name, grp, un))
              
    data_fr = []
    for f in fr:
        data_fr.append( go.Scatter(y = f, x = np.arange(0, len(f))*2.5, mode = 'lines', xaxis='x', yaxis='y') )

    data_sp = go.Heatmap(z = SP, y=freq[0:40], x=np.arange(0, len(f))*2.5, xaxis='x', yaxis='y4', colorscale='Jet', showscale= False)
    data_emg = go.Scatter(y = ampl, x=np.arange(0, len(f))*2.5, mode='lines', xaxis='x', yaxis='y3', 
                          line = dict(
                                  color = ('rgb(0, 0, 0)'),
                                  width = 2,
                                  )                          
                          )

    data_m = go.Heatmap(z=[M], x = np.arange(0, len(f))*2.5, showscale= False,
                        xaxis='x',
                        yaxis='y2',
                        colorscale =[[0, 'rgb(0,255,255)'],
                                     [1/3., 'rgb(0,255,255)'],
                                     [1/3., 'rgb(150,0,255)'],
                                     [2/3., 'rgb(150,0,255)'],
                                     [2/3., 'rgb(192,192,192)'],
                                     [3/3., 'rgb(192,192,192)'],
                                     ]                        
                        )

    #fig = tls.make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing = 0.5)
    #fig.append_trace(data_sp, 1,1)
    #fig.append_trace(data_m, 2,1)
    #fig.append_trace(data_fr, 3,1)

    layout = go.Layout(
        yaxis=dict(
            domain=[0.0, 0.6],
            title='Firing rate (spikes/s)',
        ),
        yaxis2=dict(
            domain=[0.62, 0.7],
            ticks = '',
            showticklabels=False
        ),
        yaxis3=dict(
             domain=[0.75, 0.85],
             title = 'EMG Ampl. (uV)'
        ),
        yaxis4=dict(
             domain=[0.9, 1.0],
             title = 'Freq (Hz)'
        ),                                
        xaxis=dict(
                domain=[0,1], 
                anchor='y',
                title='Time (s)',
        ),
        xaxis2=dict(
                domain=[0,1], 
                anchor='y2',
                showticklabels=False,
                ticks='',
                ),
        xaxis3=dict(
                domain=[0,1], 
                anchor='y3',
                showticklabels=False,
                ticks='',
                ),
        xaxis4=dict(
                domain=[0,1], 
                anchor='y4',
                showticklabels=False,
                ticks='',
                ),
        showlegend=False
        
        )
    fig = go.Figure(data=data_fr+[data_m, data_emg, data_sp], layout=layout)
    if offline_plot:
        plotly.offline.plot(fig, filename='brstate_fr.html')
    else:
        plotly.plotly.iplot(fig, filename='brstate_fr.html')



def write_fr(ppath, name, grp, un):
    """
    write average firing rates to file $ppath/$name/mean_fr_$grp$un.txt
    :param ppath: base folder
    :param name: recording name
    :param grp: group
    :param un: uni
    :return:
    """
    FR = state_firing_rate(ppath, name, grp, un, pplot=False)

    labels = {1:'REM', 2:'Wake', 3:'NREM'}
    mfile = open(os.path.join(ppath, name, 'mean_fr_%d-%d.txt' % (grp, un)), 'w')
    for s in [1, 2, 3]:
        mfile.write('%s: %f' % (labels[s], FR[s].mean()))
        mfile.write(os.linesep)
    mfile.close()



def brstate_fr_plt(ppath, name, units, vm=2.5, fmax=30, tstart=0, tend=-1, r_mu=[10, 100], 
                   ma_thr=10, ma_mode=False, emg_ampl=True, emg_corr=0):
    """
    plot firing rate along spectrogram, brainstate, EMG amplitude using matplotlib
    For example, spyke.brstate_fr_plt(ppath, name, [(1,3)]) to plot unit 1,3 of recording name, located in ppath
    :param ppath: base folder
    :param name: recording name
    :param units: list of tuples, e.g., [(1,2), [2,2]] to plot unit 2 of group 1 and unit 2 of group 2
    :param vm: float, set saturation level of EEG spectrogram heatmap
    :param fmax: float, maximum frequency of EEG spectrogram
    :param tstart: float, tstart, first time point to be shown (in [s])
    :param tend: float, last time point to be shown (in [s]), if -1 show everything till the end
    :param r_mu: tuple, frequency used for calculation of EMG amplitude
    :param ma_thr: float, microarousal threshold
    :param ma_mode: bool, if True plot microarousals in different colors
    :param emg_ampl: bool, if False plot raw EMG
    :param emg_corr: set EMG values > emg_corr to emg_corr; if 0, don't do anything
    :return: n/a
    """
    # load brainstate
    M,S = sleepy.load_stateidx(ppath, name)

    # load firing rates
    fr = []
    for neuron in units:
        (grp, un) = neuron
        f = firing_rate(ppath, name, grp, un)
        if len(f) < len(M):
            f = np.concatenate((f, [0]))
        fr.append(f)

    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name), squeeze_me=True)
    SPEEG = P['SP']
    # calculate median for choosing right saturation for heatmap
    med = np.median(SPEEG.max(axis=0))
    t = np.squeeze(P['t'])
    dt = t[1]-t[0]
    freq = P['freq']
    P = so.loadmat(os.path.join(ppath, name, 'msp_%s.mat' % name), squeeze_me=True)
    SPEMG = P['mSP']

    istart = int(np.round(tstart/dt))
    if tend == -1:
        iend = len(M)
    else:
        iend = int(np.round(tend/dt))
    M = M[istart:iend]
    M[M==0] = 3
    t = t[istart:iend]
    t = t-t[0]

    # to plot raw EMG:
    sr = get_snr(ppath, name)
    nbin = np.round(2.5 * sr)
    istart_emg = int(istart*nbin)
    iend_emg   = int((iend+1)*nbin)
    ddt = 1.0/sr
    t_emg = np.arange(0, iend_emg-istart_emg)*ddt

    if ma_thr > 0:
        seq = sleepy.get_sequences(np.where(M==2)[0])
        for s in seq:
            if np.round((len(s)*dt)) <= ma_thr:
                if not ma_mode:
                    M[s] = 3
                else:
                    M[s] = 4


    # plot the stuff
    plt.figure(figsize=(10,4))
    # plot brain state
    axes1 = plt.axes([0.1, 0.5, 0.8, 0.05])
    A = np.zeros((1, len(M)))
    A[0, :] = M
    cmap = plt.cm.jet
    if not ma_mode:
        my_map = cmap.from_list('ha', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
        tmp = axes1.pcolorfast(t, [0, 1], A, vmin=0, vmax=3)
    else:
        my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8], [1, 0, 1]], 5)
        tmp = axes1.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=5)
    tmp.set_cmap(my_map)
    axes1.axis('tight')
    tmp.axes.get_xaxis().set_visible(False)
    tmp.axes.get_yaxis().set_visible(False)
    sleepy.box_off(axes1)
    axes1.set_yticks([])
    axes1.get_xaxis().set_visible(False)
    axes1.get_yaxis().set_visible(False)
    axes1.spines["top"].set_visible(False)
    axes1.spines["right"].set_visible(False)
    axes1.spines["bottom"].set_visible(False)
    axes1.spines["left"].set_visible(False)

    # show spectrogram
    ifreq = np.where(freq <= fmax)[0]
    axes2 = plt.axes([0.1, 0.75, 0.8, 0.2], sharex=axes1)
    axes2.pcolorfast(t, freq[ifreq], SPEEG[ifreq, istart:iend], vmin=0, vmax=vm * med, cmap='jet')
    axes2.axis('tight')
    plt.ylabel('Freq (Hz)')
    sleepy.box_off(axes2)
    plt.xlim([t[0], t[-1]])

    # EMG amplitude
    axes3 = plt.axes([0.1, 0.61, 0.8, 0.1], sharex=axes2)
    if emg_ampl:
        i_mu = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
        p_mu = np.sqrt(SPEMG[i_mu, :].sum(axis=0) * (freq[1]-freq[0]))
        axes3.plot(t, p_mu[istart:iend], color='black')
        plt.ylabel('EMG $\mathrm{(\mu V)}$')
        plt.xlim((t[0], t[-1]+1))
        sleepy.box_off(axes3)
    else:
        emg = so.loadmat(os.path.join(ppath, name, 'EMG.mat'), squeeze_me=True)['EMG']
        if emg_corr>0:
            emg[np.where(np.abs(emg)>emg_corr)] = 0
        axes3.plot(t_emg, emg[istart_emg:iend_emg], color='black', lw=0.2)
        plt.xlim((t_emg[0], t_emg[-1] + 1))


    # plot firing rate
    clrs = [[0,0,1], [0.5, 0, 1], [0.6, 0.6, 1], [0, 1, 1]]
    axes4 = plt.axes([0.1, 0.15, 0.8, 0.3], sharex=axes3)
    j=0
    for f in fr:
        axes4.plot(t, f[istart:iend], color=clrs[j])
        j+=1
    sleepy.box_off(axes4)
    plt.ylabel('FR (spikes/s)')
    plt.xlim([t[0], t[-1]-10])
    plt.xlabel('Time (s)')

    # remove xtick labels on axes1 to axes3
    plt.setp(axes1.get_xticklabels(), visible=False)
    plt.setp(axes2.get_xticklabels(), visible=False)
    plt.setp(axes3.get_xticklabels(), visible=False)

    plt.show(block=False)



def plot_raw_channels(ppath, name, ch_id, tstart, tend, fig_file=''):
    """
    plot raw traces of channel $ch_id in recording $name, starting at second $tstart to second $tend.
    If a filename $fig_file is provided, the figure will be saved.
    """
    fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % ch_id), 'r')
    channel = fid['D']
    sr_raw = get_infoparam(os.path.join(ppath, name, 'info.txt'), 'SR_raw')[0]
    sr_raw = float(sr_raw)
    istart = int(tstart*sr_raw)
    iend =   int(tend*sr_raw)
    dt = 1.0 / sr_raw
    t = np.arange(0, (iend-istart-1)*dt+dt/2, dt)

    ch_shown = channel[istart:iend]
    fid.close()

    plt.ion()
    plt.figure()
    ax = plt.subplot(111)
    plt.plot(t, ch_shown, color='black')
    sleepy.box_off(ax)
    plt.xlabel('Time (s)')
    plt.xlim((t[0], t[-1]))
    plt.show()

    if len(fig_file) > 0:
        plt.savefig(fig_file, bbox_inches="tight")



def electrode_depth(ppath, idf, init_depth=0):
    """
    read out electrode_paths_ file for mouse $idf
    return the depth in microns traveled by the electrode from initial depth $init_depth
    """
    
    depth_file = 'electrode_paths_%s.txt' % idf
    fid = open(os.path.join(ppath, depth_file), 'r')

    lines = fid.readlines()

    depth = {}
    for line in lines:
        if re.match('^\s+$', line):
            continue
        if re.match('^#', line):
            continue
        
        if re.match('^EL', line):
            a = re.split('\s+', line)[1:]
            
            s = 1.0
            if a[2] == 'D':
                s = -1.0
            
            num = a[1]
            b = reduce(lambda x,y:x/y, [float(i) for i in re.split('/', num)])
            
            depth[a[0]] = b*s * 4 * 100
            
    return init_depth+sum(list(depth.values()))



def state_firing_avg_simple(ppath, file_listing):

    units = load_units_simple(ppath, file_listing)

    FR = []
    for u in units:
        fr = state_firing_rate(ppath, u[0], u[1][0], u[1][1], pplot=False)
        fr = [fr[i].mean() for i in [1,2,3]]
        FR.append(fr)

    FR = np.array(FR)

    # make figure
    plt.figure()
    ax = plt.axes([0.2, 0.1, 0.35, 0.8])
    plt.bar([1,2,3], FR.mean(axis=0), color='gray')
    plt.plot([1,2,3], FR.T, color='black')
    sleepy.box_off(ax)
    plt.xticks([1,2,3], ['REM', 'Wake', 'NREM'])
    plt.ylabel('Firing rate (spikes/s)')
    plt.show(block=False)



def state_firing_avg(ppath, unit_listing, backup='', pzscore=False):
    """
    calculate average state-dependent firing rate for units listed in $file_listing
    function generates bar plot. State dependent firing rates are plotted
    in the following order: REM, NREM, Wake.
    :param ppath: base folder
    :param unit_listing: file listing (file name) or dict of units
    :param backup: potential backup folder with recordings
    :param pzscore: if True, z-score data
    :return: np.array(units x firing rates for REM, Wake, NREM)
    """
    if type(unit_listing) == str:
        units = load_units(ppath, unit_listing)
    else:
        units = unit_listing

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    avg_fr = {k:[] for k in units}
    for k in units:
        for rec in units[k]:
            fr = state_firing_rate(rec.path, rec.name, rec.grp, rec.un, pplot=False, pzscore=pzscore)
            state_fr = {}
            for s in [1,2,3]:
                state_fr[s] = fr[s].mean()
            avg_fr[k].append(list(state_fr.values()))

    # calculate for unit average
    # across all recordings
    FR = np.zeros((len(units), 3))
    i = 0
    for k in units:
        # Note: np.array([[1,2,3]]) generates a
        # 1 x 3 array
        FR[i,:] = np.array(avg_fr[k]).mean(axis=0)
        i += 1
        
    # make figure
    sleepy.set_fontarial()
    plt.ion()
    plt.figure()
    ax = plt.axes([0.2, 0.1, 0.35, 0.8])
    plt.bar([1,2,3], FR[:,[0,2,1]].mean(axis=0), color='gray')
    plt.plot([1,2,3], FR[:,[0,2,1]].T, color='black')
    sleepy.box_off(ax)
    plt.xticks([1,2,3], ['REM', 'NREM', 'Wake'], rotation=30)
    plt.ylabel('Firing rate (spikes/s)')
    plt.show()

    names = []
    for k in units:
        rec = units[k][0]
        names.append('%s:(%d,%d)' % (rec.name, rec.grp, rec.un))

    columns = ['REM', 'Wake', 'NREM']
    df = pd.DataFrame(FR, columns=columns, index = names)
    # We have here dependent samples: REM, Wake, NREM firing rates are form the same units; so,
    # a repeated measures test is a correct choice.
    # Second, we should expect the data to be non-normal, so a non-parametric test is the gneeral choice
    # E.g. Friedman test would work; or Wilcoxon sign-rank test with Bonferroni correction
    res_fried = stats.friedmanchisquare(df['REM'], df['Wake'], df['NREM'])
    print("Is the population of units modulated by brain state?: Friedman X-square test: statistics: %.3f, p-value: %.3f" % (res_fried[0], res_fried[1]))

    # (3) Wilcoxon signed-rank test with post-hoc Bonferroni correction
    bonf = bonferroni_signtest(df)
    print("Wilcoxon-signtest with Bonferroni correction")
    print(bonf)

    # calculate effect size
    print("Estimate effect size:")
    for s in [1,2,3]:
        for q in range(s+1,4):
            c = cohen_d(df[columns[s-1]],df[columns[q-1]])
            print("Effect size, %s<>%s: %.3f" % (columns[s-1], columns[q-1], c))

    #return FR as pandas data frame
    return df



def unit_classification(ppath, file_listing, alpha = 0.05, backup=''):
    """
    Test tye type of each unit in $file_listing.
    Types tested for are REM-max, REM-on, REM>Wake>NREM, Wake-max, NREM-max, REM-off, sleep-active (=wake-min) units.
    The statistical test used is Ranksum test, together with Bonferroni correction
    :param ppath: base folder
    :param file_listing: file listing
    :param alpha: significance level
    :param backup: optional backup folder
    :return: dict with class/category as key and list of Recordings (objects of class Recording)
    """
    units = load_units(ppath, file_listing)
    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    FR = {k:[] for k in units}
    for k in units:
        state_fr = {s:np.array([]) for s in [1,2,3]}
        for rec in units[k]:
            spikesd = firing_rate(rec.path, rec.name, rec.grp, rec.un)
            M,_ = sleepy.load_stateidx(rec.path, rec.name)
            if len(spikesd) != len(M):
                n = np.min((len(spikesd), len(M)))
                M = M[0:n]
                spikesd = spikesd[0:n]

            for s in [1,2,3]:
                state_fr[s] = np.concatenate((state_fr[s], spikesd[np.where(M==s)[0]]))

        FR[k] = state_fr

    # Who is REM-max?
    rem_max = []
    rem_on  = []
    rem_wake = []
    for k in units:
        # test if REM is largest
        if FR[k][1].mean() > FR[k][2].mean() and FR[k][1].mean() > FR[k][3].mean():
            p1 = stats.ranksums(FR[k][1], FR[k][2])[1]
            p2 = stats.ranksums(FR[k][1], FR[k][3])[1]
            print(p1, p2)
            # Bonferroni correction
            if p1 < alpha/2. and p2 < alpha/2.:
                rem_max.append(k)

            # testing if unit is REM-on, i.e.
            # either Wake and NREM are not statistically different, so REM is the only state
            # "sticking out"
            # or NREM > Wake
            p3 = stats.ranksums(FR[k][2], FR[k][3])[1]
            if p1 < alpha/3. and p2 < alpha/3.:
                if p3 > alpha/3. or FR[k][3].mean() > FR[k][2].mean():
                    rem_on.append(k)
                else:
                    rem_wake.append(k)

    # test for Wake-max
    wake_max = []
    for k in units:
        # test if REM is largest
        if FR[k][2].mean() > FR[k][1].mean() and FR[k][2].mean() > FR[k][3].mean():
            p1 = stats.ranksums(FR[k][2], FR[k][1])[1]
            p2 = stats.ranksums(FR[k][2], FR[k][3])[1]
            # Bonferroni correction
            if p1 < alpha / 2. and p2 < alpha / 2.:
                wake_max.append(k)

    # test for NREM-max
    nrem_max = []
    for k in units:
        # test if REM is largest
        if FR[k][3].mean() > FR[k][1].mean() and FR[k][3].mean() > FR[k][2].mean():
            p1 = stats.ranksums(FR[k][3], FR[k][1])[1]
            p2 = stats.ranksums(FR[k][3], FR[k][2])[1]
            # Bonferroni correction
            if p1 < alpha / 2. and p2 < alpha / 2.:
                nrem_max.append(k)

    # test for REM-off = REM-min
    rem_off = []
    for k in units:
        # test if REM is largest
        if FR[k][1].mean() < FR[k][2].mean() and FR[k][1].mean() < FR[k][3].mean():
            p1 = stats.ranksums(FR[k][1], FR[k][2])[1]
            p2 = stats.ranksums(FR[k][1], FR[k][3])[1]
            # Bonferroni correction
            if p1 < alpha / 2. and p2 < alpha / 2.:
                rem_off.append(k)

    # who is sleep active = wake-min?
    wake_min = []
    for k in units:
        # test if REM is largest
        if FR[k][2].mean() < FR[k][1].mean() and FR[k][2].mean() < FR[k][3].mean():
            p1 = stats.ranksums(FR[k][2], FR[k][1])[1]
            p2 = stats.ranksums(FR[k][2], FR[k][3])[1]
            # Bonferroni correction
            if p1 < alpha / 2. and p2 < alpha / 2.:
                wake_min.append(k)

    unit_classes = {'rem_max': {k:units[k] for k in rem_max},
                    'rem_on':  {k:units[k] for k in rem_on},
                    'rem_wake':{k:units[k] for k in rem_wake},
                    'wake_max':{k:units[k] for k in wake_max},
                    'nrem_max':{k:units[k] for k in nrem_max},
                    'rem_off': {k:units[k] for k in rem_off},
                    'wake_min':{k:units[k] for k in wake_min}}

    # print groups of units:
    print("REM-max units:")
    for k in rem_max:
        print("%d: %s" % (k, units[k]))

    print("REM-on units:")
    for k in rem_on:
        print("%d: %s" % (k, units[k]))

    print("REM>Wake>NREM units:")
    for k in rem_wake:
        print("%d: %s" % (k, units[k]))

    print("Wake-max units:")
    for k in wake_max:
        print("%d: %s" % (k, units[k]))

    print("NREM-max units:")
    for k in nrem_max:
        print("%d: %s" % (k, units[k]))

    print("REM-off units:")
    for k in rem_off:
        print("%d: %s" % (k, units[k]))

    print("sleep-active units:")
    for k in wake_min:
        print("%d: %s" % (k, units[k]))

    return unit_classes



def unit_classification2(ppath, file_listing, alpha = 0.05, backup=''):
    """
    Test for type of each unit in $file_listing.
    Follows the classification as in Koike et al., J. Neurosci., 2017
    Types tested for are REM-max, REM-on, REM>Wake>NREM, Wake-max, NREM-max, REM-off, sleep-active (=wake-min) units.
    The statistical test used is Ranksum test, together with Bonferroni correction
    :param ppath: base folder
    :param file_listing: file listing
    :param alpha: significance level
    :param backup: optional backup folder
    :return: dict with class/category as key and list of Recordings (objects of class Recording)
    """
    units = load_units(ppath, file_listing)
    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    FR = {k:[] for k in units}
    for k in units:
        state_fr = {s:np.array([]) for s in [1,2,3]}
        for rec in units[k]:
            spikesd = firing_rate(rec.path, rec.name, rec.grp, rec.un)
            M,_ = sleepy.load_stateidx(rec.path, rec.name)
            if len(spikesd) != len(M):
                n = np.min((len(spikesd), len(M)))
                M = M[0:n]
                spikesd = spikesd[0:n]

            for s in [1,2,3]:
                state_fr[s] = np.concatenate((state_fr[s], spikesd[np.where(M==s)[0]]))

        FR[k] = state_fr

    # Who is REM-max?
    rem_max = []
    rem_nrem_wake  = []
    rem_wake_nrem = []
    for k in units:
        # test if REM is largest
        if FR[k][1].mean() > FR[k][2].mean() and FR[k][1].mean() > FR[k][3].mean():
            p1 = stats.ranksums(FR[k][1], FR[k][2])[1]
            p2 = stats.ranksums(FR[k][1], FR[k][3])[1]
            print(p1, p2)
            # Bonferroni correction
            if p1 < alpha/2. and p2 < alpha/2.:
                rem_max.append(k)

            # testing if unit is R>W>N or R>N>W
            p3 = stats.ranksums(FR[k][2], FR[k][3])[1]
            if p3 < alpha/3.:
                if FR[k][2].mean() > FR[k][3].mean():
                    rem_wake_nrem.append(k)
                else:
                    rem_nrem_wake.append(k)

    # test for Wake-max?
    wake_max = []
    wake_nrem_rem = []
    wake_rem_nrem = []
    for k in units:
        # test if REM is largest
        if FR[k][2].mean() > FR[k][1].mean() and FR[k][2].mean() > FR[k][3].mean():
            p1 = stats.ranksums(FR[k][2], FR[k][1])[1]
            p2 = stats.ranksums(FR[k][2], FR[k][3])[1]
            # Bonferroni correction
            if p1 < alpha / 2. and p2 < alpha / 2.:
                wake_max.append(k)

            p3 = stats.ranksums(FR[k][1], FR[k][3])[1]
            if p3 < alpha/3.:
                if FR[k][1].mean() > FR[k][3].mean():
                    wake_rem_nrem.append(k)
                else:
                    wake_nrem_rem.append(k)

    # test for NREM-max?
    nrem_max = []
    nrem_wake_rem = []
    nrem_rem_wake = []
    for k in units:
        # test if REM is largest
        if FR[k][3].mean() > FR[k][1].mean() and FR[k][3].mean() > FR[k][2].mean():
            p1 = stats.ranksums(FR[k][3], FR[k][1])[1]
            p2 = stats.ranksums(FR[k][3], FR[k][2])[1]
            # Bonferroni correction
            if p1 < alpha / 2. and p2 < alpha / 2.:
                nrem_max.append(k)

            p3 = stats.ranksums(FR[k][1], FR[k][2])[1]
            if p3 < alpha/3.:

                if FR[k][1].mean() > FR[k][2].mean():
                    nrem_rem_wake.append(k)
                else:
                    nrem_wake_rem.append(k)

    unit_classes = {'rem_max': {k:units[k] for k in rem_max},
                    'rem_wake_nrem': {k: units[k] for k in rem_wake_nrem},
                    'rem_nrem_wake': {k: units[k] for k in rem_nrem_wake},

                    'wake_max': {k: units[k] for k in wake_max},
                    'wake_nrem_rem':{k:units[k] for k in wake_nrem_rem},
                    'wake_rem_nrem': {k: units[k] for k in wake_rem_nrem},

                    'nrem_max':{k:units[k] for k in nrem_max},
                    'nrem_wake_rem': {k: units[k] for k in nrem_wake_rem},
                    'nrem_rem_wake': {k: units[k] for k in nrem_rem_wake}}

    # print groups of units:
    print("REM-max units:")
    for k in rem_max:
        print("%d: %s" % (k, units[k]))

    print("R>W>N units:")
    for k in rem_wake_nrem:
        print("%d: %s" % (k, units[k]))

    print("R>N>W units:")
    for k in rem_nrem_wake:
        print("%d: %s" % (k, units[k]))

    print("Wake-max units:")
    for k in wake_max:
        print("%d: %s" % (k, units[k]))

    print("W>R>N units:")
    for k in wake_rem_nrem:
        print("%d: %s" % (k, units[k]))

    print("W>N>R units:")
    for k in wake_nrem_rem:
        print("%d: %s" % (k, units[k]))

    print("NREM-max units:")
    for k in nrem_max:
        print("%d: %s" % (k, units[k]))

    print("N>W>R units:")
    for k in nrem_wake_rem:
        print("%d: %s" % (k, units[k]))

    print("N>R>W units:")
    for k in nrem_rem_wake:
        print("%d: %s" % (k, units[k]))

    return unit_classes



def _waveform_params(wvf, dt_spike, pplot=False):
    """
    extract half-amplitude duration and trough to peak duration; called by &waveform_classification
    Example:
    spyke._waveform_params(spyke.unpack_unit(ppath, name, 4, 7)[2], 1.0/25000, pplot=True)

    :param wvf: vector, waveform
    :param dt_spike: time resolution of spike recording
    :param pplot: if True, make instructie plot
    :return: half-amplitude duration, trough to peak duration
    """

    # get trough to peak time
    # valley index
    ival = np.argmin(wvf)
    imax = np.argmax(wvf[ival:]) + ival
    dur_trpeak = (imax - ival + 1) * dt_spike

    # get half-amplitude duration
    half_val = (wvf[ival] - wvf[0]) / 2.0

    ihalf1 = np.argmin(np.abs(wvf[0:ival] - half_val))
    ihalf2 = np.argmin(np.abs(wvf[ival:imax] - half_val))+ival

    dur_half = (ihalf2 - ihalf1 + 1) * dt_spike

    t = (np.arange(0, len(wvf))*dt_spike - dt_spike*ival)*1000

    if pplot:
        plt.ion()
        plt.figure()

        ax = plt.subplot(111)
        ax.plot(t, wvf, color='black')
        ax.plot([t[ihalf1], t[ihalf2]], [half_val, half_val], color='red')
        ax.plot([t[imax], t[imax]], [wvf[ival], wvf[imax]], color='orange')
        ax.plot([t[ival], t[imax]], [wvf[ival], wvf[ival]], color='orange')
        sleepy.box_off(ax)
        plt.xlabel('Time (ms)')
        #ax.spines["top"].set_visible(False)
        #ax.spines["right"].set_visible(False)
        #ax.spines["bottom"].set_visible(False)
        #ax.spines["left"].set_visible(False)
        #ax.axes.get_xaxis().set_visible(False)
        #ax.axes.get_yaxis().set_visible(False)

        plt.show()

    return dur_half, dur_trpeak



def waveform_classification(ppath, unit_listing, backup='', pplot=True):
    """
    extract half-amplitude duration and trough to peak duration from each unit specifief in $unit_list
    :param ppath: base folder
    :param unit_listing: string or dict; name of file listing (string) or dict specifying recordings for each unit
           as returned by &load_units
    :param backup: optional backup folder
    :param pplot: if True, plot figure
    :return: list of half amplitude duration for each unit, trough to peak duration for each unit,
            waveform for each unit, mean firing rate (averaged across all recordings)
    """
    PRE = 30
    if type(unit_listing) == str:
        units = load_units(ppath, unit_listing)
    else:
        units = unit_listing

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    waveforms = {k:[] for k in units}
    mean_firing = {k:[] for k in units}
    for k in units:
        for rec in units[k]:
            _, train, wvf, _ = unpack_unit(rec.path, rec.name, rec.grp, rec.un)
            sr = get_infoparam(os.path.join(rec.path, rec.name, 'info.txt'), 'SR_raw')[0]
            dt_spike = 1.0 / float(sr)
            waveforms[k].append(wvf)
            mean_firing[k].append(train.mean())

    # if unit has more than one recording (=waveforms) chose the largest waveform
    for k in waveforms:
        wvf = waveforms[k]
        size = [np.max(w)-np.min(w) for w in wvf]
        waveforms[k] = wvf[np.argmax(size)]
        mean_firing[k] = np.mean(np.array(mean_firing[k]))

    dur_trpeak = []
    dur_half = []
    wvf = []
    mean_fr = []
    for k in waveforms:
        a,b = _waveform_params(waveforms[k], dt_spike, pplot=True)
        dur_half.append(a)
        dur_trpeak.append(b)
        wvf.append(waveforms[k])
        mean_fr.append(mean_firing[k])

    if pplot:
        plt.ion()
        plt.figure()

        ax = plt.subplot(111)
        plt.scatter(np.array(dur_half)*1000, np.array(dur_trpeak)*1000)

        plt.xlabel('Half amplitude duration (ms)')
        plt.ylabel('Trough to peak duration (ms)')
        sleepy.box_off(ax)

        plt.figure()
        ax = plt.subplot(111)
        t = np.arange(0, len(wvf[0]))*dt_spike*1000 - (PRE-1)*dt_spike*1000
        plt.plot(t, np.array(wvf).T)
        plt.xlabel('Time (ms)')
        sleepy.box_off(ax)
        plt.show()

    return np.array(dur_half), np.array(dur_trpeak), np.array(wvf), np.array(mean_fr)



def fr_transitions(ppath, unit_listing, transitions, pre, post, si_threshold, sj_threshold,
                   backup='', mu=[10, 100], fmax=30, ma_thr=20, ma_polish=True, pemg=False, vm = [],
                   base_int = 10, ylim=[], pzscore=False, psingle_mode=True):
    """
    plot average firing rate along brainstate transitions.
    An example:
    spyke.fr_transitions(ppath, 'dm_units.txt', [[3,1], [1,2], [2,3], [3,2]],
                         60, 40, [10,60,60], [10, 40, 40], pemg=False)
    plots NREM to REM, REM to Wake, Wake to NREM and NREM to Wake transition
    for all units specified in dm_units.txt.
    REM periods have to be at least 10s long, before and after transition.
    Wake and NREM periods have to be at least 60 and 40 s long before the transition, respectively.

    :param ppath: base folder with recordings
    :param unit_listing: file listing or dict with Recordings
    :param transitions: list of tuples, e.g. [(3,1), (2,3)] tells the function to plot
           NREM to REM and Wake to NREM transitions
    :param pre: time before transition
    :param post: time after transition
    :param si_threshold: list with three entries (floats), specifying how long REM, Wake and NREM periods have
           to be at least before the transition
    :param sj_threshold: list with three entries (floats) how long REM, Wake, NREM has to last
           after the transition
    :param backup:
           potential backup folder with recordings
    @Optional:
    :param mu: frequencies to calculate EMG amplitude
    :param fmax: maximum frequency shown in EEG spectrogram
    :param ma_thr: threshold for microarousals
    :param ma_polish: if True, transfer microarousals to NREM
    :param pemg: if True, plot EMG
    :param vm: tuple (list with two elements), setting lower and upper range of spectrogram colormap
    :param base_int, float, duration of baseline interval, used as reference for statistical tests, when a firing rate
           change starts becoming significant
    :param ylim: tuple, min. and max. value for y-range
    :param pzscore, if True, z-score firing rates
    :param psingle_mode, if True, plot each single unit separately with its own color
    :return:
    """
    if type(unit_listing) == str:
        units = load_units(ppath, unit_listing)
    else:
        units = unit_listing

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    states = {1: 'R', 2: 'W', 3: 'N'}

    # dict: trans_ID -> unit -> list of transitions
    trans_fr  = dict()
    trans_spe = dict()
    trans_spm = dict()
    trans_fr_all = dict()
    for (si, sj) in transitions:
        sid = states[si] + states[sj]
        trans_fr[sid]     = []
        trans_fr_all[sid] = []
        trans_spe[sid]    = []
        trans_spm[sid]    = []

    for (si,sj) in transitions:
        sid = states[si] + states[sj]
        fr_unit  = {k:[] for k in units}
        spe_unit = {k:[] for k in units}
        spm_unit = {k:[] for k in units}

        for k in units:
            for rec in units[k]:
                sr = sleepy.get_snr(rec.path, rec.name)
                nbin = int(np.round(sr)*2.5)
                dt = (1.0/sr)*nbin
                ipre  = int(np.round(pre/dt))
                ipost = int(np.round(post/dt))
                ddir = os.path.join(rec.path, rec.name)
    
                # load firing rate
                spikesd = firing_rate(rec.path, rec.name, rec.grp, rec.un)
                if pzscore:
                    spikesd = (spikesd - spikesd.mean()) / spikesd.std()
    
                # load spectrogram and normalize
                P = so.loadmat(os.path.join(ddir, 'sp_%s.mat' % rec.name), squeeze_me=True)
                SP = P['SP']
                freq = P['freq']
                ifreq = np.where(freq <= fmax)[0]
                df = freq[1] - freq[0]
                sp_mean = SP.mean(axis=1)
                SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
    
                # load EMG
                imu = np.where((freq >= mu[0]) & (freq <= mu[1]))[0]
                MP = so.loadmat(os.path.join(ddir, 'msp_%s.mat' % rec.name), squeeze_me=True)['mSP']
                emg_ampl = np.sqrt(MP[imu, :].sum(axis=0) * df)
    
                # load brain state
                M, _ = sleepy.load_stateidx(rec.path, rec.name)
    
                M[np.where(M == 4)] = 3    
                if ma_polish:
                    seq = sleepy.get_sequences(np.where(M == 2)[0])
                    for s in seq:
                        if len(s) * dt < ma_thr:
                            if (s[0] > 1) and (M[s[0] - 1] != 1):
                                M[s] = 3
    
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
                        sj_idx = range(ti+1, p+1)
                        # so the indices of state si are s
                        # the indices of state sj are sj_idx
    
                        if ti >= ipre and ti<len(M)-ipost-1 and len(s)*dt >= si_threshold[si-1] and len(sj_idx)*dt >= sj_threshold[sj-1]:
                            fr_si = spikesd[ti-ipre+1:ti+1]
                            fr_sj = spikesd[ti+1:ti+ipost+1]
                            fr = np.concatenate((fr_si, fr_sj))
    
                            spe_si = SP[ifreq,ti-ipre+1:ti+1]
                            spe_sj = SP[ifreq,ti+1:ti+ipost+1]
    
                            spe = np.concatenate((spe_si, spe_sj), axis=1)
    
                            spm_si = emg_ampl[ti-ipre+1:ti+1]
                            spm_sj = emg_ampl[ti+1:ti+ipost+1]
                            spm = np.concatenate((spm_si, spm_sj))

                            fr_unit[k].append(fr)
                            trans_fr_all[sid].append(fr.copy())

                            spe_unit[k].append(spe)
                            spm_unit[k].append(spm)


        trans_fr[sid]  = fr_unit
        trans_spe[sid] = spe_unit
        trans_spm[sid] = spm_unit

    nunits = len(units)
    ntrans = len(trans_fr)
    nx = 1.0/ntrans
    dx = 0.2 * nx
    f = freq[ifreq]
    t = np.arange(-ipre * dt + dt, ipost * dt + dt / 2, dt)

    for tr in trans_fr:
        for k in trans_fr[tr]:
            # average for each transition tr and unit over trials
            if trans_fr[tr][k] == []:
                tmp = np.zeros((len(t),))
                tmp[:] = np.nan
                trans_fr[tr][k]  = tmp
                trans_spm[tr][k] = tmp.copy()
                tmp2 = np.zeros((len(f), len(t)))
                tmp2[:] = np.nan
                trans_spe[tr][k] = tmp2
            else:
                trans_fr[tr][k]  = np.array(trans_fr[tr][k]).mean(axis=0)
                # average over all spectrograms for transition $tr of unit $k
                trans_spe[tr][k] = np.mean(np.array(trans_spe[tr][k]), axis=0)
                trans_spm[tr][k] = np.array(trans_spm[tr][k]).mean(axis=0)

    trans_fr_mx = dict()
    trans_spe_mx = dict()
    trans_spm_mx = dict()
    for tr in trans_fr:
        trans_fr_mx[tr] = np.array(list(trans_fr[tr].values()))
        trans_spe_mx[tr] = np.array(list(trans_spe[tr].values()))
        trans_spm_mx[tr] = np.array(list(trans_spm[tr].values()))

    # STATISTICS: when does the change in firing rate become significant?
    ibin = int(np.round(base_int / dt))
    nbin = int(np.floor((pre+post)/base_int))
    trans_stats = {}
    index = np.arange(0, nbin) * base_int - pre
    index = [str(s)+'s' for s in index]
    df_stats = pd.DataFrame(index=index)
    for tr in trans_fr_mx:
        trans = trans_fr_mx[tr]
        base = trans[:,0:ibin].mean(axis=1)
        stats_vec = np.ones((nbin,))
        for i in range(1,nbin):
            p = stats.wilcoxon(base, trans[:,i*ibin:(i+1)*ibin].mean(axis=1))
            stats_vec[i] = p.pvalue
        trans_stats[tr] = stats_vec
        df_stats[tr] = trans_stats[tr]

    #----------------------------------------------------------------------------
    # FIGURE
    i = 0
    plt.ion()
    plt.figure()
    for s in transitions:
        tr = states[s[0]] + states[s[1]]
        # plot firing rate
        ax = plt.axes([nx*i+dx, 0.15, nx-dx-dx/2.0, 0.2])
        if psingle_mode:
            plt.plot(t, np.nanmean(trans_fr_mx[tr], axis=0), color='blue')
            tmp = np.nanmean(trans_fr_mx[tr], axis=0)
            sem = np.nanstd(trans_fr_mx[tr], axis=0)/np.sqrt(nunits)
            ax.fill_between(t, tmp-sem, tmp+sem, color=(0,0,1), alpha=0.5, edgecolor=None)
        else:
            clrs = sns.color_palette("husl", nunits)
            for k in range(nunits):
                ax.plot(t, trans_fr_mx[tr][k,:], color=clrs[k])

        sleepy.box_off(ax)
        plt.xlim([t[0], t[-1]])
        plt.xlabel('Time (s)')
        if len(ylim) == 2:
            plt.ylim(ylim)

        if i==0:
            if pzscore:
                plt.ylabel('FR (z-scored)')
            else:
                plt.ylabel('FR (spikes/s)')

        # plot emg amplitude
        if pemg:
            ax = plt.axes([nx * i + dx, 0.4, nx - dx-dx/2.0, 0.15])
            ax.plot(t, trans_spm_mx[tr].mean(axis=0))

        # plot spectrogram
        ax = plt.axes([nx * i + dx, 0.6, nx - dx-dx/2.0, 0.3])
        im = ax.pcolorfast(t, f, np.nanmean(trans_spe_mx[tr], axis=0), cmap='jet')
        ax.set_xticklabels([])
        if i==0:
            plt.ylabel('Freq. (Hz)')
        if i>0:
            ax.set_yticklabels([])
        if len(vm) > 0:
            im.set_clim(vm)

        sleepy.box_off(ax)
        i += 1
    plt.show()
    #----------------------------------------------------------------------------

    print(df_stats)
    return trans_fr, trans_fr_all, trans_spe, trans_spm, df_stats



def fr_transitions_singletrials(ppath, unit_listing, transition, pre, post, si_threshold, sj_threshold,
                   backup='', pzscore=True):
    """
    Plot each single trial during the given brain state transition as color coded matrix

    Example, mx = spyke.fr_transitions_singletrials(ppath, 'dm_units.txt', [3,1], 60, 40, 60, 0)

    :param ppath: base folder
    :param unit_listing: file listing or list of units
    :param transition: tuple, calculate activity during brain state transition from transition[0] to transition[1]
    :param pre: time before transition
    :param post: time following transition
    :param si_threshold: float (not list!), minimum duration of preceding period (= transition[0])
    :param sj_threshold: float (not list!), minimum duration of following period (= transition[1])
    :param backup: optional backup folder
    :param pzscore: if True, z-score data
    :return: MX  -   np.array(trials x firing rate)
    """
    if type(unit_listing) == str:
        units = load_units(ppath, unit_listing)
    else:
        units = unit_listing

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    fr_trans = fr_transitions(ppath, unit_listing, [transition], pre, post, [si_threshold]*3, [sj_threshold]*3,
                   backup=backup, pzscore=pzscore)[1]
    sid = list(fr_trans.keys())[0]
    MX = np.array(fr_trans[sid])

    rec = units[0][0]
    sr = sleepy.get_snr(rec.path, rec.name)
    nbin = int(np.round(sr) * 2.5)
    dt = (1.0 / sr) * nbin
    ipre = int(np.round(pre / dt))
    ipost = int(np.round(post / dt))
    t = np.arange(-ipre * dt + dt, ipost * dt + dt / 2, dt)
    ntrials = MX.shape[0]

    plt.figure()
    plt.pcolormesh(t, range(0, ntrials), MX)
    plt.show()

    return MX



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



def fr_sleepcycle(ppath, file_listing, backup='', nstates_rem=10, nstates_itrem=20,
                  pzscore=False, ma_polish=True, fmax=30, mu=[10,100], ma_thr=20,
                  single_mode=False, sf=0, ylim=[]):
    """
    plot firing rate of units in file_listing throughout the sleep cycle;
    A sleep cycle starts with REM periods i and ends with REM period i+1;
    REM periods and inter-vening NREM and Wake periods (=inter-REM periods)
    are normalized to the same length.
    Example,
    pyke.fr_sleepcycle(ppath, 'dm_units.txt', sf=1.0, ylim=(0, 40))

    :param ppath: recording base folder
    :param file_listing: listing of units
    :param backup: optional backup folder with recordings
    :param nstates_rem: the REM periods are resampled to nstates_rem states
    :param nstates_itrem: the inter-REM period is resampled to nstates_itrem states
    :param pzscore: if True, zscore firing rates
    :param ma_polish: if True, transfer microarousals (= wake periods shorter than ma_thr) to NREM
    :param fmax: maximal frequency for EEG spectrogram
    :param mu: tuple, min. and max. frequency for EMG amplitude calculation
    :param ma_thr: thresholds for microarousals
    :param single_mode: if True, plot all units, otherwise just average with s.e.m
    :param sf: float, if > 0, smooth each firing rate with Gaussian filter
    :param ylim: tuple, min. and max. firing rate shown on lower y-axis
    :return: np.arrays with firing rates, EEG spectrograms and EMG amplitudes
    """
    units = load_units(ppath, file_listing)
    nunits = len(units)

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)
            # load firing rate
            spikesd = firing_rate(rec.path, rec.name, rec.grp, rec.un)
            if sf>0:
                spikesd = sleepy.smooth_data(spikesd, sf)
            if pzscore:
                spikesd = (spikesd - spikesd.mean()) / spikesd.std()
            rec.suppl = spikesd
    
    # dict, unit ku --> np.array of all time morphed spectrograms
    SPcycle  = dict()
    # dict, unit ku --> np.array of all time morphed firing rates
    FRcycle  = dict()  
    EMGcycle = dict()
    for ku in units:
        # list collecting all spectrograms for unit ku
        sp_cycle_unit = []
        # list collecting all sleep cycle firing rates for unit ku
        fr_cycle_unit = []
        emg_cycle_unit = []

        for rec in units[ku]:
            sr = sleepy.get_snr(rec.path, rec.name)
            nbin = int(np.round(sr)*2.5)
            dt = (1.0/sr)*nbin
            ddir = os.path.join(rec.path, rec.name)
    
            # load firing rate
            spikesd = rec.suppl
    
            # load spectrogram and normalize
            P = so.loadmat(os.path.join(ddir, 'sp_%s.mat' % rec.name), squeeze_me=True)
            SP = P['SP']
            freq = P['freq']
            ifreq = np.where(freq <= fmax)[0]
            df = freq[1] - freq[0]
            sp_mean = SP.mean(axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
            SP = SP[ifreq,:]
    
            # load EMG
            imu = np.where((freq >= mu[0]) & (freq <= mu[1]))[0]
            MP = so.loadmat(os.path.join(ddir, 'msp_%s.mat' % rec.name), squeeze_me=True)['mSP']
            EMG = np.sqrt(MP[imu, :].sum(axis=0) * df)
    
            # load brain state
            M, _ = sleepy.load_stateidx(rec.path, rec.name)
            # make sure spikesd has same length as spikesd
            if M.shape[0] < spikesd.shape[0]:
                spikesd = spikesd[0:M.shape[0]]
            if spikesd.shape[0] < M.shape[0]:
                M = M[0:spikesd.shape[0]]

            M[np.where(M == 4)] = 3    
            if ma_polish:
                seq = sleepy.get_sequences(np.where(M == 2)[0])
                for s in seq:
                    if len(s) * dt < ma_thr:
                        if (s[0] > 1) and (M[s[0] - 1] != 1):
                            M[s] = 3
    
            seq = sleepy.get_sequences(np.where(M==1)[0])
            # We have at least 2 REM periods (i.e. one sleep cycle)
            if len(seq) >=2:
                for (si, sj) in zip(seq[:-1], seq[1:]):
                    # indices of inter-REM period
                    #idx = range(si[-1]+1, sj[0]-1)
                    idx = range(si[-1] + 1, sj[0])

                    fr_pre  = time_morph(spikesd[si], nstates_rem)
                    fr_post = time_morph(spikesd[sj], nstates_rem)
                    fr_itrem = time_morph(spikesd[idx], nstates_itrem)
                    fr_cycle = np.concatenate((fr_pre, fr_itrem, fr_post))
                    fr_cycle_unit.append(fr_cycle)
                    
                    SP_pre   = time_morph(SP[:,si].T, nstates_rem).T
                    SP_post  = time_morph(SP[:,sj].T, nstates_rem).T
                    SP_itrem = time_morph(SP[:,idx].T, nstates_itrem).T
                    sp_cycle_unit.append(np.concatenate((SP_pre, SP_itrem, SP_post),axis=1))               
               
                    EMG_pre   = time_morph(EMG[si], nstates_rem).T
                    EMG_post  = time_morph(EMG[sj], nstates_rem).T
                    EMG_itrem = time_morph(EMG[idx].T, nstates_itrem).T
                    emg_cycle_unit.append(np.concatenate((EMG_pre, EMG_itrem, EMG_post)))               

        # key -> np.array (number of cycles x frequences x time)
        SPcycle[ku]  = np.array(sp_cycle_unit)
        FRcycle[ku]  = np.array(fr_cycle_unit)
        EMGcycle[ku] = np.array(emg_cycle_unit)

    ntime = 2*nstates_rem+nstates_itrem
    nfreq = ifreq.shape[0]
    # average for each unit the firing rates
    FRmx  = np.zeros((nunits, ntime))
    SPmx  = np.zeros((nunits, nfreq, ntime))
    EMGmx = np.zeros((nunits, ntime))
    for ku in units:
        FRmx[ku,:]   = np.nanmean(FRcycle[ku], axis=0)
        SPmx[ku,:,:] = np.nanmean(SPcycle[ku], axis=0)
        EMGmx[ku,:]  = np.nanmean(EMGcycle[ku], axis=0)

    # plotting
    plt.ion()
    plt.figure(figsize=(10, 6))

    # plot spectrogram
    ax = plt.axes([0.15, 0.5, 0.75, 0.3])
    ax.pcolorfast(range(ntime), freq[ifreq], np.nanmean(SPmx, axis=0), cmap='jet')
    ax.set_xticks([nstates_rem, nstates_rem+nstates_itrem])
    plt.ylabel('Freq (Hz)')
    ax.set_xticklabels([])
    sleepy.box_off(ax)

    # plot firing rate
    ax = plt.axes([0.15, 0.1, 0.75, 0.3])
    if single_mode:
        plt.plot(range(ntime), FRmx.T)
    else:
        tmp = np.nanmean(FRmx, axis=0)
        sem = np.nanstd(FRmx, axis=0) / np.sqrt(nunits)
        plt.plot(range(ntime), tmp, color='blue')
        ax.fill_between(range(ntime), tmp - sem, tmp + sem, color=(0, 0, 1), alpha=0.5, edgecolor=None)
        plt.plot(range(ntime), tmp, color='blue')
    sleepy.box_off(ax)
    ax.set_xticks([nstates_rem, nstates_rem+nstates_itrem])
    plt.xlim((0, ntime-1))
    if len(ylim)>1:
        plt.ylim(ylim)
    if pzscore:
        plt.ylabel('FR (z-scored)')
    else:
        plt.ylabel('FR (spikes/s)')
    ax.set_xticklabels([])
    plt.show()

    return FRmx, SPmx, EMGmx



def fr_prepost_rem(ppath, unit_listing, istate, backup='', sf=0, pzscore=True, ma_thr=10, min_rem=0):
    """
    test if firing rate during last occurance of state $istate before REM is different from firing rate
    during first occurance of state $istate after REM
    Example:

    pre, post = spyke.fr_prepost_rem(ppath, 'ilpy_neurons.txt', 2, ma_thr=10, pzscore=True, min_rem=0)

    ...calculates the firing rates during wake periods preceding (pre) and following REM sleep (post). Every
    REM period is considered (min_rem=0), wake periods have to be at least 10 s, as $ma_thr=10.

    :param ppath:
    :param unit_listing:
    :param istate: int, 2 or 3: consider either Wake for NREM periods directly preceding and following REM period
    :param backup: optional backup folder
    :param sf: float, smoothing factor for firing rates
    :param pzscore: if True, z-score firing rates
    :param ma_thr: microarousal threshold; setting this threshold also implicitly sets a threshold for wake durations
    :param min_rem: minimum duration of REM sleep period to be considered
    :return: np.array(firing rates of state $istate preceding REM), np.array(firing rates of state $istate following REM)
    """
    if type(unit_listing) == str:
        units = load_units(ppath, unit_listing)
    else:
        units = unit_listing
    nunits = len(units)

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)
            # load firing rate
            spikesd = firing_rate(rec.path, rec.name, rec.grp, rec.un)
            if sf > 0:
                spikesd = sleepy.smooth_data(spikesd, sf)
            if pzscore:
                spikesd = (spikesd - spikesd.mean()) / spikesd.std()
            rec.suppl = spikesd

    FRpre = dict()
    FRpost = dict()
    fr_pre_rem = []
    fr_post_rem = []
    rem_dur = []
    for ku in units:
        fr_pre  = []
        fr_post = []

        for rec in units[ku]:
            sr = sleepy.get_snr(rec.path, rec.name)
            nbin = int(np.round(sr) * 2.5)
            dt = (1.0 / sr) * nbin

            # load firing rate
            spikesd = rec.suppl

            # load brain state
            M, _ = sleepy.load_stateidx(rec.path, rec.name)

            # polish out microarousals
            if ma_thr > 0:
                seq = sleepy.get_sequences(np.where(M == 2)[0])
                for s in seq:
                    if len(s)*dt <= ma_thr:
                        M[s] = 3

            # for each REM period search for preceding period of state $istate
            seq = sleepy.get_sequences(np.where(M==1)[0])
            for s in seq:
                if len(s)*dt < min_rem:
                    continue
                a = s[0]-1
                while a>0:
                    if M[a] == istate:
                        break
                    a -= 1

                if a == 0:
                    continue

                b = a
                while b > 0 and M[b] == istate:
                    b -= 1

                ipre_wake = range(b+1,a+1)

                a = s[-1]+1
                while a<len(M):
                    if M[a] == istate:
                        break
                    a+=1

                b = a
                while b<len(M)-1 and M[b]==istate:
                    b += 1

                ipost_wake = range(a,b)

                fr_pre.append(spikesd[ipre_wake].mean())
                fr_post.append(spikesd[ipost_wake].mean())
                fr_pre_rem.append(spikesd[ipre_wake].mean())
                fr_post_rem.append(spikesd[ipost_wake].mean())
                rem_dur.append(len(s)*dt)

        FRpre[ku]  = np.array(fr_pre).mean()
        FRpost[ku] = np.array(fr_post).mean()

    fr_mx = np.zeros((nunits,2))
    i = 0
    for ku in units:
        fr_mx[i,0] = FRpre[ku]
        fr_mx[i,1] = FRpost[ku]
        i += 1

    # plot figure
    state = ['REM', 'Wake', 'NREM']

    plt.ion()
    clrs = [[0, 1, 1 ],[0.5, 0, 1],[0.6, 0.6, 0.6]]
    plt.figure()
    plt.subplot(121)
    pre = np.array(fr_pre_rem)
    post = np.array(fr_post_rem)
    rem = np.array(rem_dur)
    fr_max = np.nanmax(pre)
    fr_min = np.nanmin(pre)
    plt.scatter(fr_pre_rem, fr_post_rem, label=state[istate-1], color=clrs[istate-1])
    x = np.arange(fr_min, fr_max, 0.01)
    plt.plot(x,x, color='black')
    plt.xlabel('$FR_{pre}$')
    plt.ylabel('$FR_{post}$')

    # plot difference between FR_post and FR_pre as histogram
    mask = ~np.isnan(post) & ~np.isnan(pre)
    D = post[mask]-pre[mask]
    plt.subplot(122)
    plt.hist(D, 40, color='black')
    plt.xlabel('$FR_{post}$-$FR_{pre}$')
    plt.show()

    # basic stats
    p = stats.ttest_rel(pre, post, nan_policy='omit')
    print("p-value for REM-induced changes in %s firing rates: %.3f" % (state[istate-1], p[1]))

    # plot REM sleep duration vs subsequent $istate firing rate
    plt.figure()
    plt.plot(rem_dur, fr_post_rem, 'o', color='black')
    plt.xlabel('REM duration (s)')
    if pzscore:
        plt.ylabel('$\mathrm{FR_{post}}$ (z-scored)')
    else:
        plt.ylabel('$FR_{post}$ (spikes/s)')

    mask = ~np.isnan(rem) & ~np.isnan(post)
    wres  = stats.linregress(rem[mask], post[mask])
    wslope = wres[0]
    wr    = wres[2]
    wpval = wres[3]
    print("p-value for REM duration vs subsequent %s firing rates: %.3f" % (state[istate-1], wpval))

    x = np.arange(0, np.max(rem))
    y = x*wslope+wr
    plt.plot(x, y, color='black')

    return pre, post



def fr_remrem_sections(ppath, file_listing, backup='', nsections=5,
                       pzscore=False, ma_polish=True, ma_thr=10, sf=0, ylim=[]):
    """
    plot NREM and wake activity for $nsections consecutive sections of the sleep cycle (interval between two
    consecutive REM periods)
    :param ppath: base folder
    :param file_listing: text file
    :param backup: optional backup folder
    :param nsections: number of sections for the sleep cycle
    :param pzscore: if True, z-score firing rates
    :param ma_polish: if True, polish out micro-arousals
    :param ma_thr: threshold for microarousals
    :param sf: smoothing factor for firing rate
    :param ylim: typle, y-limits for y axis
    :return:
    """
    units = load_units(ppath, file_listing)
    nunits = len(units)

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)
            # load firing rate
            spikesd = firing_rate(rec.path, rec.name, rec.grp, rec.un)
            if sf > 0:
                spikesd = sleepy.smooth_data(spikesd, sf)
            if pzscore:
                spikesd = (spikesd - spikesd.mean()) / spikesd.std()
            rec.suppl = spikesd

    wake_fr_mx = np.zeros((nunits, nsections))
    nrem_fr_mx = np.zeros((nunits, nsections))


    for ku in units:
        sections_wake = [[] for p in range(nsections)]
        sections_nrem = [[] for p in range(nsections)]

        for rec in units[ku]:
            sr = sleepy.get_snr(rec.path, rec.name)
            nbin = int(np.round(sr) * 2.5)
            dt = (1.0 / sr) * nbin
            #ddir = os.path.join(rec.path, rec.name)

            # load firing rate
            spikesd = rec.suppl

            # load brain state
            M, _ = sleepy.load_stateidx(rec.path, rec.name)
            # make sure spikesd has same length as spikesd
            if M.shape[0] < spikesd.shape[0]:
                spikesd = spikesd[0:M.shape[0]]
            if spikesd.shape[0] < M.shape[0]:
                M = M[0:spikesd.shape[0]]

            M[np.where(M == 4)] = 3
            if ma_polish:
                seq = sleepy.get_sequences(np.where(M == 2)[0])
                for s in seq:
                    if len(s) * dt < ma_thr:
                        if (s[0] > 1) and (M[s[0] - 1] != 1):
                            M[s] = 3

            seq = sleepy.get_sequences(np.where(M == 1)[0])
            # We have at least 2 REM periods (i.e. one sleep cycle)

            if len(seq) > 2:
                for (si,sj) in zip(seq[0:-1], seq[1:]):
                    # indices of inter-REM periods
                    idx = range(si[-1]+1,sj[0])

                    m = len(idx)
                    M_up = upsample_mx(M[idx], nsections)
                    M_up = np.round(M_up)
                    spikesd_up = upsample_mx(spikesd[idx], nsections)

                    for p in range(nsections):
                        # for each m consecutive bins calculate average NREM, REM, Wake activity
                        mi = range(p*m, (p+1)*m)

                        idcut = np.intersect1d(mi, np.where(M_up == 2)[0])
                        wake_fr = np.nanmean(spikesd_up[idcut])
                        sections_wake[p].append(wake_fr)

                        idcut = np.intersect1d(mi, np.where(M_up == 3)[0])
                        nrem_fr = np.nanmean(spikesd_up[idcut])
                        sections_nrem[p].append(nrem_fr)

        wake_fr_mx[ku,:] = [np.nanmean(np.array(w)) for w in sections_wake]
        nrem_fr_mx[ku,:] = [np.nanmean(np.array(w)) for w in sections_nrem]

    wx = []
    wy = []
    nx = []
    ny = []
    for i in range(wake_fr_mx.shape[1]):
        wx.append(wake_fr_mx[:,i])
        wy.append(np.ones(nunits,)*i)
        nx.append(nrem_fr_mx[:,i])
        ny.append(np.ones(nunits,)*i)

    wx = reduce(lambda x, y: np.concatenate((x, y)), wx)
    wy = reduce(lambda x, y: np.concatenate((x, y)), wy)
    nx = reduce(lambda x, y: np.concatenate((x, y)), nx)
    ny = reduce(lambda x, y: np.concatenate((x, y)), ny)

    mask = ~np.isnan(wx) & ~np.isnan(wy)
    wres  = stats.linregress(wx[mask], wy[mask])
    wslope = wres[0]
    wr    = wres[2]
    wpval = wres[3]

    mask = ~np.isnan(nx) & ~np.isnan(ny)
    nres  = stats.linregress(nx[mask], ny[mask])
    nslope = nres[0]
    nr    = nres[2]
    npval = nres[3]

    if wslope > 0:
        print("Wake firing rates increase throughout sleep cycle by a factor of %.2f; r=%.2f, p=%.2f" % (wslope, wr, wpval))
    else:
        print("Wake firing rates decrease throughout sleep cycle by a factor of %.2f; r=%.2f, p=%.2f" % (wslope, wr, wpval))

    if nslope > 0:
        print("NREM firing rates increase throughout sleep cycle by a factor of %.2f; r=%.2f, p=%.2f" % (nslope, nr, npval))
    else:
        print("NREM firing rates decrease throughout sleep cycle by a factor of %.2f; r=%.2f, p=%.2f" % (nslope, nr, npval))


    # plot figure
    plt.figure()
    plt.ion()
    ax = plt.subplot(121)
    for i in range(nsections):
        plt.plot(np.ones((nunits,))*i, nrem_fr_mx[:,i], 'o', color='gray')
    plt.plot(range(nsections), np.nanmean(nrem_fr_mx, axis=0), color='black')
    ax.set_xticks([])
    plt.xlabel('inter-NREM')
    sleepy.box_off(ax)
    plt.title('NREM')
    if len(ylim) == 2:
        plt.ylim(ylim)
    if pzscore:
        plt.ylabel('Firing rate (z-scored)')
    else:
        plt.ylabel('Firing rate (spikes/s)')

    ax = plt.subplot(122)
    for i in range(nsections):
        plt.plot(np.ones((nunits,))*i, wake_fr_mx[:,i], 'o', color='violet')
    plt.plot(range(nsections), np.nanmean(wake_fr_mx, axis=0), color='black')
    ax.set_xticks([])
    plt.xlabel('inter-NREM')
    sleepy.box_off(ax)
    plt.title('Wake')
    if len(ylim) == 2:
        plt.ylim(ylim)

    plt.show()

    return wake_fr_mx, nrem_fr_mx, wres, nres



def fr_statedur(ppath, unit_listing, backup='', sf=0, pzscore=True, ma_thr=0, pplot=True):
    """
    correlate the duration of each REM, Wake, or NREM period with the firing rate of each unit during these periods

    :param ppath: base folder
    :param unit_listing: unit listing
    :param backup: optional backup folder
    :param sf: smoothing factor for firing rate
    :param pzscore: if True, z-score firing rates
    :param ma_thr: float, if > 0
    :param pplot: if True, generate plot
    :return: DataFrame - with  unit_id, firing rate (fr), state (1,2,or3), duration  as columns
    """
    if type(unit_listing) == str:
        units = load_units(ppath, unit_listing)
    else:
        units = unit_listing

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    for k in units:
        for rec in units[k]:
            # load firing rate
            spikesd = firing_rate(rec.path, rec.name, rec.grp, rec.un)
            if sf > 0:
                spikesd = sleepy.smooth_data(spikesd, sf)
            if pzscore:
                spikesd = (spikesd - spikesd.mean()) / spikesd.std()
            rec.suppl = spikesd

    # df = pd.DataFrame(columns=['unit', 'state', 'dur', 'fr'])
    df_list = []
    for ku in units:
        for rec in units[ku]:
            sr = sleepy.get_snr(rec.path, rec.name)
            nbin = int(np.round(sr) * 2.5)
            dt = (1.0 / sr) * nbin

            # load firing rate
            spikesd = rec.suppl

            # load brain state
            M, _ = sleepy.load_stateidx(rec.path, rec.name)

            mmin = np.min((len(M), len(spikesd)))
            M = M[0:mmin]
            spikesd = spikesd[0:mmin]

            # polish out microarousals
            if ma_thr > 0:
                seq = sleepy.get_sequences(np.where(M == 2)[0])
                for s in seq:
                    if len(s)*dt <= ma_thr:
                        M[s] = 3

            # for each REM period search for preceding period of state $istate
            for istate in [1,2,3]:
                seq = sleepy.get_sequences(np.where(M==istate)[0])
                for s in seq:
                    dur = len(s)*dt
                    fr = spikesd[s].mean()
                    # columns = ['unit', 'state', 'dur', 'fr']
                    df_list.append([ku, istate, dur, fr])

    df = pd.DataFrame(df_list, columns=['unit', 'state', 'dur', 'fr'])

    states = ['REM', 'Wake', 'NREM']
    if pplot:
        plt.ion()
        plt.figure()
        sns.set_style("darkgrid")
        sns.set(font_scale=1.5)
        for istate in [1,2,3]:
            plt.subplot('13' + str(istate))
            sns.regplot(x='dur', y='fr', data=df[df.state==istate])
            plt.title(states[istate-1])
        plt.show()

    return df



def cmp_waveform(ppath, name, grp, un, win, pplot=True, pre=5, post=5, factor=16, fig_file=''):
    """
    plot average of spontaneous spike waveforms together with laser triggered (=following) waveforms
    and return the correlation coefficient
    :param ppath: base folder
    :param name: recording
    :param grp: group
    :param un: unit id
    :param win: definition of laser triggered spike: if the laser happens within $win ms is is
                considered as laser triggered
    :param pplot: if True, plot figure
    :param pre: time before trough of spike in ms
    :param post: time after trough of spike
    :param factor: factor = Sampling rate Spikes / Sampling rate EEG
    :param fig_file: if file path provided, save figure to the given file
    :return: correlation coefficient between spontaneous and laser triggered waveform
    """
    sr = get_snr(ppath, name)
    laser = load_laser(ppath, name)

    lsr_start = np.where(np.diff(laser) > 0)[0] + 1
    lsr_start *= factor
    idx, train, mspike, ch = unpack_unit(ppath, name, grp, un)
    n = int(np.round((win/1000.)*sr*factor))
    ipre = int(np.round((pre/1000.)*sr*factor))
    ipost = int(np.round((post/1000.)*sr*factor))

    idx_lsr = []
    for k in lsr_start:
        i = np.where((idx >= k) & (idx <= k+n))[0]
        # append spikes following laser
        idx_lsr.append(idx[i])
    idx_lsr = reduce(lambda x, y: np.concatenate((x, y)), idx_lsr)
    idx_bkg = np.setdiff1d(idx, idx_lsr)

    MS_lsr = np.zeros((len(idx_lsr), ipre+ipost))
    MS_bkg = np.zeros((len(idx_bkg), ipre+ipost))
    fid = h5py.File(os.path.join(ppath, name, 'ch_' + name + '_%d.mat' % ch), 'r')
    A = fid['D']
    len_ch = A.shape[0]

    j=0
    for k in idx_lsr:
        if k-ipre >= 0 and k+ipost<len_ch:
            MS_lsr[j,:] = A[k-ipre+1:k+ipost+1, 0]
            j += 1

    j=0
    for k in idx_bkg:
        if k-ipre >= 0 and k+ipost<len_ch:
            MS_bkg[j,:] = A[k-ipre+1:k+ipost+1, 0]
            j += 1
    fid.close()

    # calculate correlation coefficient
    cc = np.corrcoef(MS_lsr.mean(axis=0), MS_bkg.mean(axis=0))

    if pplot:
        plt.ion()
        plt.figure()
        ax = plt.subplot(111)

        ddt = (1.0/sr) / factor
        t = np.arange(-ipre*ddt+ddt, ipost*ddt+ddt/2, ddt) * 1000

        mean_lsr = MS_lsr.mean(axis=0)
        mean_bkg = MS_bkg.mean(axis=0)
        std_lsr  = MS_lsr.std(axis=0)
        std_bkg  = MS_bkg.std(axis=0)
        plt.fill_between(t, mean_lsr-std_lsr, mean_lsr+std_lsr, color=(0, 0, 1), alpha=0.5, edgecolor=None)
        plt.fill_between(t, mean_bkg-std_bkg, mean_bkg+std_bkg, color='black', alpha=0.5, edgecolor=None)
        plt.xlabel('Time (ms)')
        plt.xlim((t[0], t[-1]))
        sleepy.box_off(ax)
        plt.show()

    return cc[0,1]



def laser_delay(ppath, name, grp, un, win=10, offs=1, iter=1):
    """
    calculate first spike latency
    :param ppath: base folder
    :param name: recording name
    :param grp: group
    :param un: unit
    :param win: window length [ms], where the spike has to appear
    :param offs: offset for spike trains
    :param iter: take every $iter spike train starting from $offset
    :return: average delay and standard deviation (=jitter)
    """
    sr = get_snr(ppath, name)
    iwin = int(np.round((win/1000.0) * sr))
    # load laser
    laser = load_laser(ppath, name)
    len_laser = laser.shape[0]
    # get start and end of each laser stimulation train
    idxs, idxe = sleepy.laser_start_end(laser, SR=sr)
    spikesd = unpack_unit(ppath, name, grp, un)[1]
    # collect first spike latencies
    delay = []
    for (i,j) in zip(idxs[offs::iter], idxe[offs::iter]):
        if j+iwin <= len_laser:
            seq = laser[i:j+iwin+1]
            # seq is sequence not idx!!
            # get all the laser pulse onsets within @seq
            ons = np.diff(seq)
            ons = np.where(ons > 0)[0]+1
            ons = np.concatenate(([0], ons))
            ons = ons+i

            # get the timing of the first spike,
            # following each laser onset
            for k in ons:
                s = spikesd[k:k+iwin+1]
                d = np.where(s>0)[0]

                if len(d) > 0:
                    delay.append(d[0]*(1.0/sr)*1000)

    delay = np.array(delay)
    return delay.mean(), delay.std()



def laser_reliability(ppath, name, grp, un, win=10, offs=1, iter=1):
    """
    calculate how reliably the given unit follows laser stimulation
    :param ppath: base folder
    :param name: recording
    :param grp: group
    :param un: un
    :param win: window length [ms], where the spike has to appear to be considered as laser driven
    :param offs: offset for spike trains
    :param iter: take every $iter spike train starting from $offset
    :return:
    """

    sr = get_snr(ppath, name)
    dt = 1.0 / sr
    nwin = int(np.round((win / 1000.0) / dt))
    spikesd = unpack_unit(ppath, name, grp, un)[1]
    laser = load_laser(ppath, name)

    idxs, idxe = sleepy.laser_start_end(laser, SR=sr, intval=5)
    up_idx = np.where(np.diff(laser) > 0)[0] + 1
    if laser[0] > 0:
        up_idx = np.concatenate(([0], up_idx))

    M = []
    V = []
    for (i,j) in zip(idxs[offs::iter], idxe[offs::iter]):
        # j is start of a laser pulse train
        k = j+nwin
        roi = up_idx[np.where((up_idx >= i) & (up_idx <= k))[0]]
        V.append(np.diff(roi).mean() * dt)
        m = []
        for r in roi:
            m.append(np.sum(spikesd[r:r + nwin + 1]))
        M.append(m)

    l = min([len(i) for i in M])
    M = [i[0:l] for i in M]
    M = np.array(M)
    M[np.where(M > 0)] = 1.0

    return M.mean()
    


def laser_triggered_train(ppath, name, grp, un, pre, post, nbin=1, offs=0, iters=1, istate=-1):
    """
    plot each laser stimulation trial in a raster plot, plot laser triggered firing rate and
    bar plot showing firing rate before, during, and after laser stimulation
    :param ppath:
    :param name:
    :param grp: group
    :param un: unit
    :param pre: time before laser onset [s]
    :param post: time after laser onset [s]
    :param nbin: downsample firing rate by a factor of $nbin
    :param offs: int, the first shown laser trial
    :param iters: int, show every $iters laser trial
    :param istate: int, only plot laser trials, where the onset falls on brain state $istate
                   istate=-1 - consider all states, 
                   istate=1 - consider only REM trials
                   istate=2 - consider only Wake trials 
                   istate=3 - consider only NREM trials
    :return: n/a
    """
    import matplotlib.patches as patches

    # load brain state
    M = sleepy.load_stateidx(ppath, name)[0]

    sr = get_snr(ppath, name)
    dt = 1.0 / sr
    pre = int(np.round(pre/dt))
    post = int(np.round(post/dt))
    laser = sleepy.load_laser(ppath, name)
    idxs, idxe = sleepy.laser_start_end(laser)
    
    # only collect laser trials starting during brain state istate
    tmps = []
    tmpe = []
    if istate > -1:
        nbin_state = int(np.round(sr) * 2.5)
        for (i,j) in zip(idxs, idxe):
            if M[int(i/nbin_state)] == istate:
                tmps.append(i)
                tmpe.append(j)                
        idxs = np.array(tmps)
        idxe = np.array(tmpe)

    laser_dur = np.mean(idxe[offs::iters] - idxs[offs::iters] + 1)*dt
    print ('laser duration: ', laser_dur)

    train = unpack_unit(ppath, name, grp, un)[1]
    
    len_train = train.shape[0]
    raster = []
    fr_pre = []
    fr_lsr = []
    fr_post = []
    for (i,j) in zip(idxs[offs::iters], idxe[offs::iters]):
        if (i - pre >= 0) and (i + post < len_train):
            raster.append(train[i - pre:i + post + 1])
            fr_pre.append(train[i-pre:i].mean())
            fr_lsr.append(train[i:j+1].mean())
            fr_post.append(train[j+1:j+post+1].mean())

    fr_pre = np.array(fr_pre)
    fr_lsr = np.array(fr_lsr)
    fr_post = np.array(fr_post)

    # time x trials
    raster = np.array(raster).T
    raster = downsample_mx(raster, nbin).T
    dt = nbin*1.0/sr
    t = np.arange(0, raster.shape[1]) * dt - pre*(1.0/sr)


    plt.ion()
    sleepy.set_fontarial()
    plt.figure()
    ax = plt.axes([0.15, 0.4, 0.8, 0.25])
    ax.plot(t, raster.mean(axis=0), color='black')
    max_fr = np.max(raster.mean(axis=0))
    ylim_fr = max_fr + 0.1*max_fr
    plt.ylim([0, ylim_fr])
    ax.add_patch(
    patches.Rectangle((0, 0), laser_dur, ylim_fr, facecolor=[0.6, 0.6, 1], edgecolor=[0.6, 0.6, 1]))
    plt.xlim((t[0], t[-1]))
    plt.xlabel('Time (s)')
    plt.ylabel('FR (spikes/s)')
    sleepy.box_off(ax)

    ax = plt.axes([0.15, 0.7, 0.8, 0.2])
    R = raster.copy()
    R[np.where(raster>0)] = 1
    cmap = plt.cm.jet
    my_map = cmap.from_list('ha', [[1, 1, 1], [0, 0, 0]], 2)
    ax.pcolorfast(t, range(R.shape[0]), R, cmap=my_map)
    plt.xticks([])
    ax.spines["bottom"].set_visible(False)
    plt.ylabel('Trial No.')
    sleepy.box_off(ax)

    ax = plt.axes([0.15, 0.05, 0.8, 0.2])
    x = [-(1.0/sr)*pre/2.0, laser_dur/2.0, laser_dur + ((1.0/sr)*post-laser_dur)/2.0]
    ax.bar(x, [fr_pre.mean(), fr_lsr.mean(), fr_post.mean()], color='black', yerr=[fr_pre.std(), fr_lsr.std(), fr_post.std()], fill=False)
    plt.xlim((t[0], t[-1]))
    plt.xticks(x)
    ax.set_xticklabels(['pre', 'laser', 'post'])
    plt.ylabel('FR (spikes/s)')
    sleepy.box_off(ax)



def laser_triggered_train_emg(ppath, name, grp, un, pre, post, nbin=1, offs=0, iters=1, istate=-1):
    """
    EXPERIMENTAL
    plot each laser stimulation trial in a raster plot, plot laser triggered firing rate and
    emg power (= variance)
    :param ppath:
    :param name:
    :param grp: group
    :param un: unit
    :param pre: time before laser onset [s]
    :param post: time after laser onset [s]
    :param nbin: downsample firing rate by a factor of $nbin
    :param offs: int, the first shown laser trial
    :param iters: int, show every $iters laser trial
    :param istate: int, only plot laser trials, where the onset falls on brain state $istate
                   istate=-1 - consider all states, 
                   istate=1 - consider only REM trials
                   istate=2 - consider only Wake trials 
                   istate=3 - consider only NREM trials
    :return: n/a
    """
    import matplotlib.patches as patches

    # load brain state
    M = sleepy.load_stateidx(ppath, name)[0]
    
    # load EMG
    EMG = so.loadmat(os.path.join(ppath, name, 'EMG.mat'), squeeze_me=True)['EMG']

    sr = get_snr(ppath, name)
    dt = 1.0 / sr
    pre = int(np.round(pre/dt))
    post = int(np.round(post/dt))
    laser = sleepy.load_laser(ppath, name)
    idxs, idxe = sleepy.laser_start_end(laser)
    
    # only collect laser trials starting during brain state istate
    tmps = []
    tmpe = []
    if istate > -1:
        nbin_state = int(np.round(sr) * 2.5)
        for (i,j) in zip(idxs, idxe):
            if M[int(i/nbin_state)] == istate:
                tmps.append(i)
                tmpe.append(j)                
        idxs = np.array(tmps)
        idxe = np.array(tmpe)

    laser_dur = np.mean(idxe[offs::iters] - idxs[offs::iters] + 1)*dt
    print ('laser duration: ', laser_dur)
    train = unpack_unit(ppath, name, grp, un)[1]
    len_train = train.shape[0]
    raster = []
    fr_pre = []
    fr_lsr = []
    fr_post = []
    emg = []
    for (i,j) in zip(idxs[offs::iters], idxe[offs::iters]):
        if (i - pre >= 0) and (i + post < len_train):
            raster.append(train[i - pre:i + post + 1])
            fr_pre.append(train[i-pre:i].mean())
            fr_lsr.append(train[i:j+1].mean())
            fr_post.append(train[j+1:j+post+1].mean())
            emg.append(EMG[i - pre:i + post + 1])

    fr_pre = np.array(fr_pre)
    fr_lsr = np.array(fr_lsr)
    fr_post = np.array(fr_post)

    # time x trials
    raster = np.array(raster).T
    raster = downsample_mx(raster, nbin).T
    
    emg = np.array(emg).T
    emg = downsample_mx(emg, nbin).T
    emg_pow = np.mean(np.power(emg, 2), axis=0)
    
    dt = nbin*1.0/sr
    t = np.arange(0, raster.shape[1]) * dt - pre*(1.0/sr)

    plt.ion()
    plt.figure()
    ax = plt.axes([0.15, 0.4, 0.8, 0.25])
    ax.plot(t, raster.mean(axis=0), color='black')
    max_fr = np.max(raster.mean(axis=0))
    ylim_fr = max_fr + 0.1*max_fr
    plt.ylim([0, ylim_fr])
    ax.add_patch(
    patches.Rectangle((0, 0), laser_dur, ylim_fr, facecolor=[0.6, 0.6, 1], edgecolor=[0.6, 0.6, 1]))
    plt.xlim((t[0], t[-1]))
    plt.xlabel('Time (s)')
    plt.ylabel('FR (spikes/s)')
    sleepy.box_off(ax)

    ax = plt.axes([0.15, 0.7, 0.8, 0.2])
    R = raster.copy()
    R[np.where(raster>0)] = 1
    cmap = plt.cm.jet
    my_map = cmap.from_list('ha', [[1, 1, 1], [0, 0, 0]], 2)
    ax.pcolorfast(t, range(R.shape[0]), R, cmap=my_map)
    plt.xticks([])
    ax.spines["bottom"].set_visible(False)
    plt.ylabel('Trial No.')
    sleepy.box_off(ax)

    ax = plt.axes([0.15, 0.05, 0.8, 0.2])
    plt.xlim((t[0], t[-1]))

    plt.plot(t, emg_pow, color='black')
    plt.ylabel('EMG power')
    sleepy.box_off(ax)



def bandpass_corr_state(ppath, name, unit, band, fft_win=2.5, perc_overlap=0.8, win=120, state=3, 
                        tbreak=10, mode='cross', pzscore=True, pnorm_spec=True, pemg=False, pplot=True, sr=0):
    """
    correlate band in EEG spectrogram with calcium activity;
    plot cross-correlation for all intervals of brain state $state.
    Negative time points in the cross-correlation mean that the calcium activity
    precedes the EEG.

    see also &bandpass_corr which correlates calcium activity and EEG for the whole recording, irrespective of brain state

    Say we correlate:
        x = [0,1,0]
        y = [0,0,1]
    so y follows x, then the result of cross correlation is
        cc = [0, 1, 0, 0, 0]
    The center point is cc[len(x)-1]. So a negative peak means that y follows x

    We correlate DF/F with the power band, so a negative peak means that the power band follows DF/F 
    (or DF/F precedes the power band)

    :param ppath: base folder
    :param name: name of recording
    :param band: 2 element list, lower and upper range for EEG band
    :parma fft_win: The function recalculates the EEG spectrogram from scratch;
           it does not use the existing sp_$name.mat file.
           The window size to calculate the FFT is specific in seconds by fft_win.
    :param perc_overlap: float ranging from 0 to 1; specifies how much to consecutive
           FFT windows (specified by fft_win) overlap. For perc_overlap = 0.5 two
           consecutive FFT windows are half overlapping, the temporal resolution (x-axis)
           of the EEG specotrogram is then $fft_win/2. By using a larger value (say 0.9)
           the overlap increases, therefore also the temporal resolutios gets finer (0.1 * $fft_win)
    :param win: float, time range for cross-correlation, ranging from -$win/2 to $win/2 seconds
    :param state: correlate calcium activity and EEG power during state $state
    :param tbreak: maximum interruption of $state
    :param mode: string, 'cross' or 'auto'; if 'auto' perform autocorrelation of DF/F signal
    :param pemg: if True, perform analysis with EMG instead of EEG
    :param pplot: if True, plot figure
    :return: np.array, cross-correlation for each $state interval
    """
    (grp, un) = unit
    if sr==0:
        sr = get_snr(ppath, name)
    nbin = int(np.round(2.5 * sr))
    nwin = int(np.round(sr * fft_win))
    if nwin % 2 == 1:
        nwin += 1
    noverlap = int(nwin*perc_overlap)
    #dff = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)['dff']
    dff = unpack_unit(ppath, name, grp, un)[1]

    if pemg:
        EEG = so.loadmat(os.path.join(ppath, name, 'EMG.mat'), squeeze_me=True)['EMG']
    else:
        EEG = so.loadmat(os.path.join(ppath, name, 'EEG.mat'), squeeze_me=True)['EEG']

    M, S = sleepy.load_stateidx(ppath, name)

    # Say perc_overlap = 0.9, then we have 10 sub-windows (= 1 / (1-perc_overlap)
    # Each subwindow has nwin_sub = nwin * (1 - 0.9) data points
    # Each subwindow has 10 = timebins_per_fftwin = fft_win - perc_overlap*fft_win = fft_win * (1 - perc_overlap)
    # Now the fft_win ranges over 10 subwindows.
    # I downsample dff to time steps of the size of the subwindows. So for one FFT step we have 10 corresponding
    # dff values, which one to take? I would argue the center point

    #timebins_per_fftwin = int(1 / (1-perc_overlap))
    #dff_shift = int(timebins_per_fftwin/2)

    # get all bouts of state $state
    seq = sleepy.get_sequences(np.where(M == state)[0], ibreak=int(tbreak/2.5)+1)
    # as the cross-correlation should range from -$win to $win, each bout
    # needs to be at least 2 * $win seconds long
    seq = [s for s in seq if len(s)*2.5 > 2*win]
    CC = []
    for s in seq:
        i = s[0] * nbin
        j = s[-1] * nbin + 1
        EEGcut = EEG[i:j]
        dffcut = dff[i:j]
        # to avoid that any time lags in the cross-correlation are due to different
        # ways of resampling the power band and dff signal, we downsample
        # the DFF signal in the exactly the same way as the "windowing" for the 
        # EEG spectrogram calculation: 
        # We use the same window size, the same amount of overlap to calculate
        # the average DFF activity for each time point.

        if noverlap==0:
            dffd = sleepy.downsample_vec(dffcut, nwin)
        else:
            dffd = downsample_overlap(dffcut, nwin, noverlap)

        if pzscore:
            dffd = (dffd-dffd.mean()) / dffd.std()

        #Pow, f, t = sleepy.spectral_density(EEGcut, 2 * nwin, nwin, 1.0 / sr)
        f, t, Pow = scipy.signal.spectrogram(EEGcut, nperseg=nwin, noverlap=noverlap, fs=sr)
        if pnorm_spec:
            sp_mean = Pow.mean(axis=1)
            Pow = np.divide(Pow, np.tile(sp_mean, (Pow.shape[1], 1)).T)

        ifreq = np.where((f >= band[0]) & (f <= band[1]))[0]
        dt = t[1] - t[0]
        iwin = int(win / dt)

        if mode != 'cross':
            pow_band = dffd
        else:
            if not pnorm_spec:
                pow_band = Pow[ifreq, :].sum(axis=0) * (f[1]-f[0])
            else:
                pow_band = Pow[ifreq, :].mean(axis=0)

        pow_band -= pow_band.mean()
        
        dffd -= dffd.mean()
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
        CC.append(xx[ii])


    CC = np.array(CC)
    t = np.arange(-iwin, iwin+1) * dt

    if pplot:
        plt.ion()
        plt.figure()
        ax = plt.subplot(111)
        #t = np.arange(-iwin, iwin + 1) * dt
        # note: point t[iwin] is "0"
        c = np.sqrt(CC.shape[0])
        a = CC.mean(axis=0) - CC.std(axis=0)/c
        b = CC.mean(axis=0) + CC.std(axis=0)/c
        plt.fill_between(t,a,b, color='gray', alpha=0.5)
        plt.plot(t, np.nanmean(CC, axis=0), color='black')
        plt.xlabel('Time (s)')
        plt.ylabel('Corr. FR (spikes/s) vs. EEG')
        plt.xlim([t[0], t[-1]])
        sleepy.box_off(ax)
        plt.show()

    return CC, t



def bandpass_corr_state_avg(ppath, unit_listing, band, win=120, fft_win=2.5, perc_overlap=0.8, state=3, tbreak=10, mode='cross', pemg=False, backup=''):
    """
    Average state-dependent cross-correlation accross different units

    Parameters
    ----------
    ppath : TYPE
        base folder 
    unit_listing : string or list
        unit_listing: unit listing, either text file, or dict as returned 
        by load_units (each value of a list of Recording objects, see class Recording)
    band : tuple
        lower and upper band for frequency band in EEG spectrogram
    win : int, optional
        The time window of the cross-correlation ranges from -$win to $win
    fft_win : float, optional
        Size of the fft window used to calculate the spectrogram. The default is 2.5.
    perc_overlap : float, optional
        ranges from 0 to 1, specifies how much two consecutive FFT windows overlap. The default is 0.8.
    state : int, optional
        Brain state, typically 1, 2, or 3. The default is 3.
    tbreak : float, optional
        maximum allowed interruption of a sequence of state $state. The default is 10.
    mode : string, optional
        'auto' for auto-correlation or 'cross' for cross-correlation. The default is 'cross'.
    pemg : bool, optional
         if True, perform analysis with EMG instead of EEG. The default is False.
    backup : string, optional
        Path of potential backup folder. The default is '', which means no backup folder is used.

    Returns
    -------
    dfm : pd.DataFrame
        DataFrame with columns ['mouse', 'recording', 'unit', 'cc', 'time'].

    """
    data = []
    
    if type(unit_listing) == str:
        units = load_units(ppath, unit_listing)
    else:
        units = unit_listing

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    for k in units:
        for rec in units[k]:
            idf = re.split('_', rec.name)[0]
            CC, t = bandpass_corr_state(rec.path, rec.name, (rec.grp,rec.un), band, win=win, state=state, tbreak=tbreak, mode=mode, pemg=pemg, fft_win=fft_win, perc_overlap=perc_overlap, pplot=False, sr=0)
            ccmean = np.array(CC).mean(axis=0)
            m = ccmean.shape[0]
            unit_str = '%s, (%d, %d)' % (rec.name, rec.grp, rec.un)
            data += zip([idf]*m, [rec.name]*m, [unit_str]*m, list(ccmean), list(t))
        
    df = pd.DataFrame(data=data, columns=['mouse', 'recording', 'unit', 'cc', 'time'])
    
    plt.figure()
    dfm = df.groupby(['unit', 'time']).mean().reset_index()
    sns.lineplot(data=dfm, x='time', y='cc', ci=None)

    return dfm



def autocorrelogram(ppath, name, grp, un, win, fig_file=''):
    """
    calculate and plot auto-correlogram of the given unit.
    The auto-correlogram is normalized such that "1" indicates a 100% overlap of spikes.
    The peak at 0 ms is dropped.
    :param ppath: base folder
    :param name: recording name
    :param grp: group
    :param un: unit
    :param win: time window over which auto-correlation is computed, in milliseconds
    :param fig_file: if full file path is provided, save figure to $fig_file
    :return: auto-correlation
    """
    sr = get_snr(ppath, name)
    dt = 1/sr
    idx, train, mspike, ch = unpack_unit(ppath, name, grp, un)
    train[np.where(train>0)] = 1
    iwin = int(np.round((win/1000.0)/dt))
    num_spikes = np.sum(train)

    corr = np.zeros((iwin,))
    for s in range(1,iwin+1):
        sm = np.dot(train[s:], train[0:-s]).sum() / num_spikes
        corr[s-1] = sm

    corr = np.concatenate((corr[-1::-1], [0], corr))
    t = np.arange(-dt*iwin, dt*iwin+dt/2, dt)*1000

    plt.ion()
    plt.figure()
    ax = plt.subplot(111)
    plt.bar(t, corr, width=dt*1000, color='gray')
    sleepy.box_off(ax)
    plt.draw()
    plt.xlabel('Time (ms)')
    plt.ylabel('Correlation')
    plt.xlim((t[0], t[-1]))
    plt.draw()

    if len(fig_file) > 0:
        plt.savefig(fig_file)

    return corr



def statedep_autocorrelogram(ppath, name, grp, un, win=100, seq_dur = 10, pcomplete=True, pplot=True):
    """
    Calculate and plot the state dependent autocorrelogram for the given unit.
    The y-axis specifies the rate at which two spikes are separated by an
    interspike interval as shown on the x-axis.
    :param ppath: base folder
    :param name: recording
    :param grp: group
    :param un: unit
    :param win: total duration of autocorrelogram [ms]
    :param seq_dur: minimum duration [s] of state sequence
    :param pcomplete: if True, also plot (in black) the autocorrelorgram for the spike train irrespective of brain state
    :param pplot: if True, plot figure
    :return: dict: [R|W|N] --> np.array [autocorrelogram], np.array[time vector]
    """
    sr = get_snr(ppath, name)
    dt = 1.0 / sr
    nbin = int(np.round(sr) * 2.5)

    # minimum length of state sequence
    min_len = int(np.round(seq_dur / (nbin * dt)))
    # number of shifts to calculate autocorrelogram
    nstep   = int(np.round(0.001*win/dt))

    train = unpack_unit(ppath, name, grp, un)[1]
    train[np.where(train>0)] = 1
    M,_ = sleepy.load_stateidx(ppath, name)

    CC = dict()
    for istate in [1,2,3]:
        state_idx = np.where(M==istate)[0]
        seq = sleepy.get_sequences(state_idx)

        ccs = []
        for s in seq:
            if len(s) >= min_len:
                m = train[(s[0]*nbin):(s[-1]*nbin+1)]
                cc = np.zeros((nstep,))
                for i in range(1,nstep):
                    cc[i] = np.mean(np.multiply(m[i:], m[0:-i])) * (1.0/dt)
                ccs.append(cc)

        CC[istate] = np.array(ccs).mean(axis=0)

    if pcomplete:
        m = train
        cc = np.zeros((nstep,))
        for i in range(1, nstep):
            cc[i] = np.mean(np.multiply(m[i:], m[0:-i])) * (1.0 / dt)
        CC[0] = cc

    t = np.arange(0, nstep)*dt*1000
    if pplot:
        clrs = [[0, 1, 1], [0.5, 0, 1], [0.8, 0.8, 0.8]]
        plt.ion()
        plt.figure()
        ax = plt.subplot(111)
        for s in [1,2,3]:
            plt.semilogx(t, CC[s], color=clrs[s-1])
        if pcomplete:
            plt.semilogx(t, CC, color='black')
        plt.xlabel('Time (ms)')
        plt.ylabel('Rate (overlap/s)')
        sleepy.box_off(ax)
        plt.show()

    return CC, t



def statedep_autocorrelogram_avg(ppath, unit_listing, backup='', win=100, seq_dur=10, pcomplete=True):
    """
    calculate for each unit the state stependent autocorrelogram using spyke.statedep_autocorrelogram
    :param ppath: base folder
    :param unit_listing: unit listing, either text file, or dict as returned by load_units
    :param backup: optional backup folder
    :param win: temporal window in ms across which the autocorrelogram is computed
    :param seq_dur: minidum duration in s for a state sequence
    :param pcomplete: if True, plot autocorrelogram for whole spike train, irrespsective of brain state
    :return: dict: [0|R|W|N] --> np.array [units x autocorrelogram], np.array[time vector]
    """
    if type(unit_listing) == str:
        units = load_units(ppath, unit_listing)
    else:
        units = unit_listing

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    cc = {s:[] for s in [0,1,2,3]}
    for k in units:
        cc_unit = {s:[] for s in [0,1,2,3]}
        for rec in units[k]:
            xx, t = statedep_autocorrelogram(rec.path, rec.name, rec.grp, rec.un, win=win, pplot=False, pcomplete=pcomplete, seq_dur=seq_dur)
            for s in [0, 1,2,3]:
                cc_unit[s].append(xx[s])
        for s in [0,1,2,3]:
            cc[s].append(np.array(cc_unit[s]).mean(axis=0))

    # transform lists of arrays to 2D arrays
    for s in [0,1,2,3]:
        cc[s] = np.array(cc[s])

    clrs = [[0, 1, 1], [0.5, 0, 1], [0.8, 0.8, 0.8]]
    plt.ion()
    plt.figure()
    ax = plt.subplot(111)
    for s in [1,2,3]:
        plt.semilogx(t, cc[s].mean(axis=0), color=clrs[s-1])
    if pcomplete:
        plt.semilogx(t, cc[0].mean(axis=0), color='black')

    plt.xlabel('Time (ms)')
    plt.ylabel('Rate (overlap/s)')
    sleepy.box_off(ax)
    plt.show()

    return cc, t


def burst_detection(ppath, unit_listing, backup=''):
    if type(unit_listing) == str:
        units = load_units(ppath, unit_listing)
    else:
        units = unit_listing

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    nunits = len(units)
    total_spikes = np.zeros((nunits,))
    burst_index  = np.zeros((nunits,))
    j = 0
    for k in units:
        unit_bursts = []
        ntotal = 0
        for rec in units[k]:
            ppath = rec.path
            name  = rec.name
            sr = get_snr(ppath, name)
            dt = 1.0 / sr

            train = unpack_unit(rec.path, rec.name, rec.grp, rec.un)[1]
            train[np.where(train>0)] = 1

            idx = np.where(train == 1)[0]
            ntotal += len(idx)
            isi = np.diff(idx)

            idx_sel = np.where(isi < isi.mean())[0]
            ml = isi[idx_sel].mean()

            print(ml*dt)
            i = 0
            while i < len(idx_sel)-1:
                b = isi[idx_sel[i]]
                cluster = []
                while b < ml and i < len(idx_sel)-1:
                    cluster.append(idx[idx_sel[i]])
                    i += 1
                    b = isi[idx_sel[i]]

                if len(cluster) >= 2:
                    t0 = idx[idx_sel[i-1]]
                    d = isi[idx_sel[i-1]]
                    t1 = t0+d
                    cluster.append(t1)
                    unit_bursts.append(cluster)
                i += 1
        total_spikes[j] = ntotal
        burst_index[j] = 1.0*sum([len(s) for s in unit_bursts]) / ntotal
        j += 1

    pdb.set_trace()
    return burst_index


def burst_index(ppath, unit_listing, backup='', twin=15):
    """
    Calculate burst index for given units. The burst index of a unit is defined as the fraction of spikes
    with an interspike interval <$twin ms.
    :param ppath:
    :param unit_listing: unit listing, either text file, or dict as returned by load_units
    :param backup:
    :param twin: a spike followed by another spike with $twin ms is considered as bursty spike
    :return: np.array(units x burst index)
    """

    twin = twin/1000.
    if type(unit_listing) == str:
        units = load_units(ppath, unit_listing)
    else:
        units = unit_listing

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    nunits = len(units)
    Burst = np.zeros((nunits, 2))
    i = 0
    for k in units:
        count_burst = 0
        total_spikes = 0
        for rec in units[k]:
            ppath = rec.path
            name  = rec.name
            sr = get_snr(ppath, name)
            dt = 1.0 / sr
            iwin = int(np.round(twin/dt))
            train = unpack_unit(rec.path, rec.name, rec.grp, rec.un)[1]
            train[np.where(train>0)] = 1

            spike_idx = np.where(train==1)[0]
            total_spikes += len(spike_idx)
            for s in spike_idx:
                a = np.sum(train[s+1:s+iwin])
                if a>0:
                    count_burst += 1

        Burst[i,0] = count_burst
        Burst[i,1] = total_spikes
        i += 1

    burst_idx = np.divide(1.0*Burst[:,0], 1.0*Burst[:,1])

    return burst_idx



def state_burst_index(ppath, unit_listing, backup='', twin=15, min_len=0):
    """
    calculate state dependent burst index. The burst index of a unit is defined as the fraction of spikes
    with an interspike interval <$twin ms.
    Function plots figure showing burst index for REM, Wake, and NREM
    :param ppath: base folder with recordings
    :param unit_listing: unit listing, either text file, or dict as returned by load_units
    :param backup: optional backup folder
    :param twin: a spike followed by another spike with $twin ms is considered as bursty spike
    :param min_len: minimum duration of state sequence
    :return: dict: [1,2,3] --> np.array(burst index for each unit)
    """
    twin = twin/1000.
    if type(unit_listing) == str:
        units = load_units(ppath, unit_listing)
    else:
        units = unit_listing

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    nunits = len(units)
    Burst = {}
    for s in [1,2,3]:
        Burst[s] = np.zeros((nunits, 2))

    i = 0
    for k in units:
        for rec in units[k]:
            ppath = rec.path
            name  = rec.name
            sr = get_snr(ppath, name)
            nbin = int(np.round(sr) * 2.5)
            dt = 1.0 / sr
            iwin = int(np.round(twin/dt))

            train = unpack_unit(rec.path, rec.name, rec.grp, rec.un)[1]
            train[np.where(train>0)] = 1

            M, _ = sleepy.load_stateidx(ppath, name)

            for istate in [1, 2, 3]:
                count_burst = 0
                total_spikes = 0
                state_idx = np.where(M == istate)[0]
                seq = sleepy.get_sequences(state_idx)

                for s in seq:
                    if len(s) >= min_len * (dt*nbin):
                        m = train[(s[0] * nbin):(s[-1] * nbin + 1)]
                        spike_idx = np.where(m==1)[0]
                        total_spikes += len(spike_idx)
                        for si in spike_idx:
                            a = np.sum(m[si+1:si+iwin])
                            if a>0:
                                count_burst += 1

                Burst[istate][i,0] += count_burst
                Burst[istate][i,1] += total_spikes
        i += 1

    burst_idx = {}
    for s in [1,2,3]:
        burst_idx[s] = np.divide(1.0*Burst[s][:,0],Burst[s][:,1])

    # plot figure
    plt.ion()
    plt.figure()
    ax = plt.axes([0.15, 0.15, 0.5, 0.7])
    plt.bar([1,2,3], [burst_idx[s].mean() for s in [1,2,3]], color='gray')
    for i in range(nunits):
        plt.plot([1,2,3], [burst_idx[s][i] for s in range(1,4)], color='black')

    sleepy.box_off(ax)
    plt.xticks([1,2,3])
    plt.ylabel('Burst index')
    ax.set_xticklabels(['REM', 'Wake', 'NREM'], rotation=30)
    plt.show()

    return burst_idx



def statedep_isi(ppath, name, istate, grp, un, delta=2, win=100, norm=True, pplot=False):
    """
    calculate brain state dependent interspike interval distribution for a single unit.
    :param ppath:
    :param name:
    :param istate: brain state; 1 - REM, 2 - Wake, 3 - NREM
    :param grp: group
    :param un: unit within group
    :param delta: time bin [ms] for interspike intervals
    :param win: maximum value [ms] for interspike intervals
    :param norm: if True, normalize histogram, such that the sum of all elements yields 1
    :return: np.array(interspike interval distribution), np.array(interspike interval values
    """
    sr = get_snr(ppath, name)
    nbin = int(np.round(sr) * 2.5)
    sr_raw = float(get_infoparam(os.path.join(ppath, name, 'info.txt'), 'SR_raw')[0])
    # FF is the factor by which the EEG sampling rate (sr)
    # is decreased relative to single unit sampling rate (sr_raw)
    FF = int(sr_raw/sr)

    spike_idx = unpack_unit(ppath, name, grp, un)[0]

    M,_ = sleepy.load_stateidx(ppath, name)
    state_idx = np.where(M==istate)[0]
    seq = sleepy.get_sequences(state_idx)

    # for each istate period, determine the ISI
    diff = []

    for s in seq:
        # s[0]          is index according to spectrogram
        # s[0] *nbin    is index in EEG time
        # S[0] *nbin*FF is index is spike channel
        a = int(s[0]*nbin*FF)
        b = int((s[-1]+1)*nbin*FF)-1
        idx = np.where((spike_idx >= a) & (spike_idx <= b))[0]
        d = np.diff(spike_idx[idx]) * (1.0/sr_raw) * 1000
        pdb.set_trace()
        diff.append(d)

    # flatten list of arrays to single array
    diff = reduce(lambda x, y: np.concatenate((x, y), axis=0), diff)

    edges = np.arange(0, win, delta)
    isi, edges = np.histogram(diff, edges, density=norm)
    edges = edges[0:-1]+delta/2.0
    # the normalization garantuees that sum(isi) == 1
    if norm:
        isi   = isi*delta

    if pplot:
        plt.ion()
        plt.figure()
        ax = plt.subplot(111)
        for s in [1,2,3]:
            plt.bar(edges, isi, color='black', alpha=0.5, width=delta)
            sleepy.box_off(ax)
            plt.xlabel('ISI (ms)')
            plt.ylabel('Distribution')
        plt.show()

    return isi, edges



def statedep_isi_states(ppath, name, grp, un, delta=2, win=100, norm=True):
    """
    plot brainstate depending interspike interval (ISI) distribution for a single unit; uses function statedep_isi
    :param ppath: base folder
    :param name: recording
    :param grp: group
    :param un: unit
    :param delta: time bin for ISI distribution
    :param win: maximum time interval for interspike interval
    :param norm: if True, normalize ISI distrubtion
    :return: n/a
    """
    plt.ion()
    plt.figure()
    clrs = [[0, 1, 1], [0.5, 0, 1], [0.8, 0.8, 0.8]]
    for s in [1, 2, 3]:
        ax = plt.subplot(int('31' + str(s)))
        isi,edges = statedep_isi(ppath, name, s, grp, un, delta=delta, win=win, norm=norm, pplot=False)
        plt.bar(edges, isi, color=clrs[s-1], width=delta)
        sleepy.box_off(ax)
        plt.xlabel('ISI (ms)')
        plt.ylabel('Distribution')
    plt.show()



def statedep_isi_avg(ppath, unit_listing, backup='', delta=2, win=100, alpha=5, pplot=True):
    """
    average interspike interval (ISI) distribution across several units
    :param ppath: base folder
    :param unit_listing: ile listing (file name) or dict of units as retured by &load_units
    :param backup: optional backup folder
    :param delta: time bin [ms] for interspike intervals
    :param win: maximum value [ms] for interspike intervals
    :param alpha: specifies confidence interval.
    :param pplot: if True, normalize each histogram, such that the sum of all elements yields 1
    :return: dict: brainstate --> np.array(units x ISI distribution)
    """
    if type(unit_listing) == str:
        units = load_units(ppath, unit_listing)
    else:
        units = unit_listing

    for k in units:
        for rec in units[k]:
            rec.set_path(ppath, backup)

    isi = {s:[] for s in [1,2,3]}
    for k in units:
        isi_unit = {s:[] for s in [1,2,3]}
        for rec in units[k]:
            for s in [1,2,3]:
                a, edges = statedep_isi(rec.path, rec.name, s, rec.grp, rec.un, win=win, delta=delta)
                isi_unit[s].append(a)
        for s in [1,2,3]:
            isi[s].append(np.array(isi_unit[s]).mean(axis=0))

    # transform lists of arrays to 2D arrays
    for s in [1,2,3]:
        isi[s] = np.array(isi[s])

    if pplot:
        plt.ion()
        plt.figure()
        clrs = [[0, 1, 1], [0.5, 0, 1], [0.8, 0.8, 0.8]]
        for s in [1,2,3]:
            ax = plt.subplot(int('31' + str(s)))

            tmp = np.nanmean(isi[s], axis=0)
            a = np.nanpercentile(isi[s], alpha, axis=0)
            b = np.nanpercentile(isi[s], 100-alpha, axis=0)

            plt.fill_between(edges, a, b, alpha=0.5, color=clrs[s-1])
            plt.plot(edges, tmp, color='black')
            sleepy.box_off(ax)
            plt.xlabel('ISI (ms)')
            plt.ylabel('Distribution')
        plt.show()

    return isi


#######################################################################################
# Spectralfield  ######################################################################
####################################################################################### 
def spectralfield(ppath, name, grp, un, pre, post, fmax=20, theta=0, states=[1,2,3], sp_norm=True,
                  pzscore=False, pplot=True, pfilt=True, nsmooth=1, peeg=True):
    """
    Calculate the "receptive field = spectral field" best matching the EEG spectrogram onto the neural activity
    :param ppath: base folder
    :param name: name
    :param grp, un: group and unit id
    :param pre: no. of time bins into past
    :param post: no. of time bins into future
    :param fmax: maximum frequency for spectrogram
    :param theta: float or list of floats; regularization parameter for ridge regression. If a list is provided the
           the function picks the parameters resulting in best performance on the test set.
    :param states: states for which the spectralfield is calcualted
    :param sp_norm: if True, normalize spectrogram by dividing each frequency band by its mean
    :param pzscore: if True, z-score neural activity
    :param pplot: if True, plot spectralfield
    :param pfilt: if True, smooth spectrogram
    :param nsmooth: width parameter for Gaussian filter used to smooth firing rate
    :param peeg: if False, use EMG instead of EEG
    :return: k, f, k, perf; spectral field (np.array(frequency x time)), time, frequency vectors, maximum performance on test set
    """
    from pyphi import build_featmx, cross_validation, ridge_regression

    if not type(theta) == list:
        theta = [theta]

    if peeg:
        P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name), squeeze_me=True)
        SP = P['SP']
    else:
        P = so.loadmat(os.path.join(ppath, name, 'msp_%s.mat' % name), squeeze_me=True)
        SP = P['mSP']
    N = SP.shape[1]
    freq = P['freq']
    fdt = P['dt']
    ifreq = np.where(freq <= fmax)[0]
    nfreq = len(ifreq)

    if pfilt:
        filt = np.ones((3,1))
        filt = filt / np.sum(filt)
        SP = scipy.signal.convolve2d(SP, filt, mode='same')
    if sp_norm:
        for i in range(SP.shape[0]):
            SP[i,:] = SP[i,:] / SP[i,:].mean()

    MX = build_featmx(SP[ifreq,:], pre, post)

    # load unit
    SR = get_snr(ppath, name)
    NBIN = int(np.round(SR)*2.5)
    train = unpack_unit(ppath, name, grp, un)[1]
    spikesd = downsample_vec(train, NBIN)
    spikesd = sleepy.smooth_data(spikesd, nsmooth)
    if pzscore:
        spikesd = (spikesd-spikesd.mean()) / spikesd.std()

    ibin = np.array([], dtype='int64')
    M,K = sleepy.load_stateidx(ppath, name)
    M = M[pre:N-post]
    for i in states:
        tmp = np.where(M == i)[0]
        ibin = np.concatenate((ibin, tmp))

    MX = MX[ibin,:]
    spikesd = spikesd[ibin]

    # mean zero response vector and columns of stimulus matrix
    rmean = spikesd.mean()
    spikesd = spikesd - rmean
    mmean = MX.mean(axis=0)
    for i in range(MX.shape[1]):
        MX[:,i] -= mmean[i]

    # perform cross validation
    Etest, Etrain = cross_validation(MX, spikesd, theta)
    print("CV results on training set:")
    print(Etrain)
    print("CV results on test set")
    print(Etest)

    # calculate kernel for optimal theta
    imax = np.argmax(Etest)
    perf_max = Etest[imax]
    print("Recording %s; optimal theta: %2.f" % (name, theta[imax]))
    k = ridge_regression(MX, spikesd, theta[imax])
    k = np.reshape(k, ((pre + post), nfreq)).T

    t = np.arange(-pre, post) * fdt

    # calibration plot
    if pplot:
        plt.ion()
        #plt.figure()
        #ax = plt.subplot(111)
        #plt.scatter(np.dot(MX,k)+rmean, spikesd+rmean)
        #plt.xlabel('Prediction $(\Delta F/F)$')
        #plt.ylabel('Data $(\Delta F/F)$')
        #sleepy.box_off(ax)
        #plt.show()

        #plt.figure()
        #plt.plot(spikesd+rmean)
        #plt.plot(np.dot(MX,k)+rmean)

        plt.figure()
        plt.pcolormesh(t, freq[ifreq], k, cmap='bwr')
        plt.xlabel('Time (s)')
        plt.ylabel('Freq. (Hz)')
        plt.colorbar()

    return k, t, freq[ifreq], perf_max



def spectralfield_highres(ppath, name, grp, un, pre, post, fmax = 60, theta=[0], states=[1,2,3], 
                          nsr_seg=2, perc_overlap=0.75, nsmooth=0, pzscore=False, pplot=True):
    """
    Calculate the "spectral field" best mapping the EEG spectrogram onto the neural activity
    The spectrogram calculation is flexible, i.e. can be adjusted by the paramters nsr_seg and perc_overlap.

    :param ppath: base folder
    :param name: name
    :params grp, un: unit
    :param pre: duration (in seconds) that the spectral fields fields into the past
    :param post: duration (in seconds) that the spectral fields extends into the future
    :param fmax: maximum frequency for spectrogram
    :param theta: float or list of floats; regularization parameter for ridge regression. If a list is provided the
           the function picks the parameters resulting in best performance on the test set.
    :param states: brain states used for calculation; 1=REM, 2=Wake, 3=NREM
    :param nsr_seg: float, defines the window length used for FFT calculation in seconds.
    :param perc_overlap: percentage that two consecutive FFT windows overlap
    :param nsmooth: smoothing factor for firing rates calculation; if 0, no smoothing
    :param pzscore: if True, z-score firing rates
    :param pplot: if True, plot spectral field
    :return: k, f, k; spectral field (np.array(frequency x time)), time, frequency vectors,
    """        
    from pyphi import build_featmx, cross_validation, ridge_regression
    
    if not type(theta) == list:
        theta = [theta]

    sr = get_snr(ppath, name)
    nbin = np.round(2.5*sr)
    EEG = so.loadmat(os.path.join(ppath, name, 'EEG.mat'), squeeze_me=True)['EEG']
    # calculate "high resolution" EEG spectrogram
    freq, t, SP = scipy.signal.spectrogram(EEG, fs=sr, window='hanning', nperseg=int(nsr_seg * sr), noverlap=int(nsr_seg * sr * perc_overlap))
    # time is centered
    N = SP.shape[1]
    #fdt = freq[1]-freq[0]
    ifreq = np.where(freq <= fmax)[0]
    nfreq = len(ifreq)
    dt = t[1]-t[0]

    ipre  =  int(np.round(pre/dt))
    ipost = int(np.round(post/dt))

    pfilt=True
    if pfilt:
        filt = np.ones((6,1))
        filt = filt / np.sum(filt)
        SP = scipy.signal.convolve2d(SP, filt, mode='same')

    sp_norm=True
    if sp_norm:
        for i in range(SP.shape[0]):
            SP[i,:] = SP[i,:] / SP[i,:].mean()

    MX = build_featmx(SP[ifreq,:], ipre, ipost)

    # load spike train
    # time points per time bin in spectrogram:
    ndown = int(nsr_seg * sr) - int(nsr_seg * sr * perc_overlap)
    ninit = int(np.round(t[0]/dt))
    train = unpack_unit(ppath, name, grp, un)[1]
    spikesd = downsample_vec(train, ndown)
    spikesd = sleepy.smooth_data(spikesd, nsmooth)
    spikesd = spikesd[ninit:]
    if pzscore:
        spikesd = (spikesd-spikesd.mean()) / spikesd.std()

    spikesd = spikesd[ipre:N-ipost]

    ibin = np.array([], dtype='int64')
    M,K = sleepy.load_stateidx(ppath, name)
    for s in states:
        seq = sleepy.get_sequences(np.where(M==s)[0])
        for p in seq:
            seqm = np.arange(int(p[0]*nbin / ndown), int(((p[-1]+1)*nbin-1)/ ndown))
            ibin = np.concatenate((ibin, seqm))

    ibin = ibin[ibin>=ipre]
    ibin = ibin[ibin<N-ipost]


    MX = MX[ibin-ipre,:]
    spikesd = spikesd[ibin-ipre]
    # mean zero response vector and columns of stimulus matrix
    rmean = spikesd.mean()
    spikesd = spikesd - rmean
    mmean = MX.mean(axis=0)
    for i in range(MX.shape[1]):
        MX[:,i] -= mmean[i]

    # perform cross validation
    Etest, Etrain = cross_validation(MX, spikesd, theta)
    print("CV results on training set:")
    print(Etrain)
    print("CV results on test set")
    print(Etest)

    # calculate kernel for optimal theta
    imax = np.argmax(Etest)
    print("Recording %s; optimal theta: %2.f" % (name, theta[imax]))
    k = ridge_regression(MX, spikesd, theta[imax])
    k = np.reshape(k, ((ipre + ipost), nfreq)).T
    t = np.arange(-ipre, ipost) * dt

    perf_max = Etest[imax]

    if pplot:
        plt.ion()
        plt.figure()
        f = freq[ifreq]
        plt.pcolormesh(t, f, k, cmap='bwr')
        plt.xlabel('Time (s)')
        plt.ylabel('Freq. (Hz)')
        plt.colorbar()
        plt.show()

    return k, t, freq[ifreq], perf_max



#######################################################################################
# VIDEO Processing/Annotation Stuff ###################################################
####################################################################################### 
def tdt_videotiming(ppath, name, res=1.0):
    """
    Process video timing file from TDT; the (mat) file contains a dict with 'onset' and
    'offset' entries
    """    
    
    vfile = os.path.join(ppath, name, 'video_timing.mat')
    vfid = so.loadmat(vfile)
    onset = np.squeeze(vfid['onset'])
    ticks = np.squeeze(vfid['tick_onset'])
    
    
    # get video frame timepoints closest to ticks
    frame_tick_idx = []
    frame_tick_time = []
    for t in ticks:        
        d = np.where((onset-t)>0)[0]
        if len(d) > 0:
            iframe = d[0]            
            frame_tick_idx.append(iframe)
            frame_tick_time.append(onset[iframe])
    
    frame_tick_idx  = np.array(frame_tick_idx)
    frame_tick_time = np.array(frame_tick_time)

    # get number of video frames    
    files = [f for f in os.listdir(os.path.join(ppath, name, 'Stack')) if re.match('^fig\d+\.jpg', f)]
    nframes = max([int(re.findall('^fig(\d+)\.jpg', f)[0]) for f in files])
    
    # rewrite video_timing.mat
    so.savemat(os.path.join(ppath, name, 'video_timing.mat'), 
               {'onset':onset, 
                'offset':vfid['offset'],
               'nframes': nframes,
               'tick_onset':vfid['tick_onset'],
               'tick_offset':vfid['tick_offset'],
               'frame_tick_idx':frame_tick_idx,
               'frame_tick_time':frame_tick_time})
            

            
def avi_to_zippedstack(ppath, name):
    import scipy.ndimage
    
    #vfile = os.path.join(ppath, name, 'video_%s.mat' % name)
    
    if not(os.path.isdir(os.path.join(ppath, name))):
        os.mkdir(os.path.join(ppath, name, 'Stack'))

    files = [f for f in os.listdir(os.path.join(ppath, name, 'Stack')) if re.match('^fig\d+\.jpg', f)]
    nframes = max([int(re.findall('^fig(\d+)\.jpg', f)[0]) for f in files])
    
    
    fig = scipy.ndimage.imread(os.path.join(ppath, name, 'Stack', 'fig1.jpg'))
    (nx, ny) = fig.shape[0:2]    
    fid = h5py.File(os.path.join(ppath, name, 'stack_%s.h5' % name), 'w')
    fid.create_dataset('Stack', shape = (nframes, nx, ny), dtype='uint8')
    
    for i in range(nframes):
        fig = scipy.ndimage.imread(os.path.join(ppath, name, 'Stack', 'fig%d.jpg' % (i+1)))
        fid['Stack'][i,:,:] = fig.mean(axis=2).astype('uint8')
        
    fid.close()
    
        
    
    

    

    


