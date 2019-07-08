#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('/Users/tortugar/Google Drive/Penn/Programming/PySleep')
import sleepy
import numpy as np
import matplotlib.pylab as plt
import scipy.io as so
import os
import pandas as pd
import seaborn as sns
import re




#ppath = '/Volumes/My Passport/mPFC closed-loop data_for Franz'
ppath = '/Users/tortugar/Documents/Penn/Data/RawData/mPFCData/'
recordings = ['J55_111018n1', 'J53_110718n1', 'J31_101618n1', 'J54_110518n1']
#recordings = ['J55_111018n1', 'J53_110718n1', 'J54_110518n1']


#band = [60, 150]
band = [50, 100]
nwin = 5
# if intersect_mode==1, then only
# take the part of REM period that
# overlaps with laser
intersect_mode = True
# normalize EEG spectogram: yes or no:
pnorm = False
g_thr = 1
vm = 4
dt = 2.5
# if True, plot spectrogram and gamma band for each
# single recording:
debug_plot = False

# get list of all mice 
mice = []
for name in recordings:
    idf = re.split('_', name) [0]
    if idf not in mice:
        mice.append(idf)

trig_sp_mice = {m:[] for m in mice}
ps_mice = {m:[] for m in mice}
data = []
plt.ion()
for name in recordings:
    idf = re.split('_', name) [0]
    
    # load (normalized) EEG spectrogram
    SP, t, freq = sleepy.normalize_spectrogram(ppath, name, 150, band, pplot=False)
    M = sleepy.load_stateidx(ppath, name)[0]
    rem_idx = np.where(M==1)[0]
    if not pnorm:
        tmp = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name), squeeze_me=True)
        SP_orig = tmp['SP']
        freq_orig = tmp['freq']
        

    # load laser ######################################
    SR = sleepy.get_snr(ppath, name)
    NBIN = np.round(2.5*SR)
    lsr = sleepy.load_laser(ppath, name)
    idxs, idxe = sleepy.laser_start_end(lsr)

    # downsample EEG time to spectrogram time    
    idxs = [int(i/NBIN) for i in idxs]
    idxe = [int(i/NBIN) for i in idxe]

    laser_idx = []
    for (i,j) in zip(idxs, idxe):
        laser_idx += range(i,j+1)
    laser_idx = np.array(laser_idx)
    ###################################################
    
    iband = np.where((freq >= band[0]) & (freq <= band[-1]))[0]
    pow_band = SP[iband,:].mean(axis=0)
    
    #thr = pow_band[rem_idx].mean() + g_thr*pow_band[rem_idx].std()
    #thr = np.percentile(pow_band[np.setdiff1d(rem_idx, laser_idx)], 75)
    thr = pow_band.mean() + g_thr*pow_band.std()
    
    plt.subplot('1'+str(len(recordings)) + str(recordings.index(name)))
    plt.hist(pow_band, 50)
    plt.plot(thr, 0, 'ro')
    plt.draw()
    

    idx_g = np.where(pow_band > thr)[0]
    # phasic REM indices with high gamma:
    idx_g = np.intersect1d(idx_g, rem_idx)
    # tonic REM periods with low gamma:
    idx_t = np.setdiff1d(rem_idx, idx_g)
    # high gamma REM sequences:
    seq = sleepy.get_sequences(idx_g)

    # figure showing spectrogram along with gamma band including
    # colored gamma peaks
    if debug_plot:
        plt.ion()
        plt.figure()
        ax1 = plt.subplot(211)
        sp_med = np.median(SP.mean())
        plt.pcolormesh(t, freq, SP, vmin=0, vmax=sp_med*vm)

        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(t, pow_band)
        for s in seq:
            plt.plot(t[s], pow_band[s], color='red')

        plt.draw()
    ######################################################

    frames = []
    # indices of gamma peaks:
    idx_p = []
    for s in seq:
        i = np.argmax(pow_band[s])
        si = s[i]
        if si >= nwin and si+nwin < SP.shape[1]:
            tmp = SP[:,si-nwin:si+nwin+1]
            frames.append(tmp)
            idx_p.append(si)
    trig_sp = np.array(frames).mean(axis=0)
    idx_p = np.array(idx_p)
    trig_sp_mice[idf].append(trig_sp)

    if pnorm:
        SPEC = SP
        f = freq
    else:
        SPEC = SP_orig
        f = freq_orig

    ps_phasic = SPEC[:,idx_g].mean(axis=1)
    ps_tonic  = SPEC[:,idx_t].mean(axis=1)
    ps_rem  = SPEC[:,rem_idx].mean(axis=1)
    idx_rem_lsr = np.intersect1d(rem_idx, laser_idx)
    idx_rem_nolsr = np.setdiff1d(rem_idx, laser_idx)
    ps_rem_lsr = SPEC[:,idx_rem_lsr].mean(axis=1)
    ps_rem_nolsr = SPEC[:,idx_rem_nolsr].mean(axis=1)

    ps_mice[idf].append(np.array([ps_rem, ps_tonic, ps_phasic, ps_rem_lsr, ps_rem_nolsr]))

        
    # collect frequency of phasic events during REM periods
    # with ('yes') and w/o laser ('no')
    # get REM sequences:
    srem = sleepy.get_sequences(rem_idx)
    for s in srem:
        tmp = np.intersect1d(s, laser_idx)
        if len(tmp) == 0:
            r = np.intersect1d(s, idx_g)
            data.append([idf, 'no', 100*float(len(r)) / len(s)])
        else:
            if intersect_mode:
                s = np.intersect1d(s, laser_idx)
            r = np.intersect1d(s, idx_g)
            data.append([idf, 'yes', 100*float(len(r)) / len(s)])

# done with loop over recordings 

i = 0
trig_sp_mx = np.zeros((len(freq), 2*nwin+1, len(mice)))
for idf in mice:
    trig_sp_mx[:,:,i] = np.array(trig_sp_mice[idf]).mean(axis=0)
    i+=1

plt.figure()
med = np.median(trig_sp_mx.mean(axis=2).mean())
vm = 2
plt.pcolormesh(np.arange(-nwin, nwin+1)*dt, freq, trig_sp_mx.mean(axis=2), vmin=0, vmax=med*vm,
               cmap='jet')
plt.ylabel('Freq (Hz)')
plt.xlabel('Time (s)')

# generate data frame of frequencies of phasic events per REM
df = pd.DataFrame(columns=['subj', 'lsr', 'freq'], data=data)
df_mice = df.groupby(['subj', 'lsr'], as_index=False).mean() 
# and plot
plt.figure()
sns.barplot(y = 'freq', x='lsr', data=df_mice, ci=90, palette={'no':'gray', 'yes':'blue'})  
plt.ylabel('% phasic events during REM')
plt.show()


#%%
# plot average power spectra
# (mouse_id, type of spectrum, frequencies)
ps_mx = np.zeros((len(mice), 5, len(f)))
for (idf, i) in zip(mice, range(len(mice))):
    ps_mx[i,:,:] = np.array(ps_mice[idf]).mean(axis=0)

fmax = 120
ifreq = np.where(f <= fmax)[0]
labels = ['rem', 'tonic', 'phasic', 'lsr', 'nolsr']
plt.figure()
for i in range(5):
    plt.plot(f[ifreq], ps_mx.mean(axis=0)[i,ifreq], label=labels[i])
plt.legend()
plt.xlabel('Freq (Hz)')
plt.ylabel('Pow ($\mathrm{\mu V^2})$')

# approximate the powerspectrum during REM with and w/o laser by weighted
# combination of powerspectrum for tonic and phasic REM
plt.figure()
for i in [3, 4]:
    plt.plot(f[ifreq], ps_mx.mean(axis=0)[i,ifreq], label=labels[i])

approx_lsr = np.zeros((len(mice), len(f)))
for (idf, i) in zip(mice, range(len(mice))):
    freq_p = float(df_mice[(df_mice.lsr == 'yes') & (df_mice.subj == idf)]['freq']) / 100.0
    approx_lsr[i,:] = ps_mx[i,1,:]*(1-freq_p) + ps_mx[i,2,:]*freq_p

freq_p = float(df_mice[df_mice.lsr == 'yes'].mean()) / 100
plt.plot(f[ifreq], approx_lsr[:,ifreq].mean(axis=0), ls='--')    


approx_nolsr = np.zeros((len(mice), len(f)))
for (idf, i) in zip(mice, range(len(mice))):
    freq_p = float(df_mice[(df_mice.lsr == 'no') & (df_mice.subj == idf)]['freq']) / 100.0
    approx_nolsr[i,:] = ps_mx[i,1,:]*(1-freq_p) + ps_mx[i,2,:]*freq_p
plt.plot(f[ifreq], approx_nolsr[:,ifreq].mean(axis=0), ls='--')    
plt.legend()
plt.xlabel('Freq (Hz)')
plt.ylabel('Pow ($\mathrm{\mu V^2})$')


