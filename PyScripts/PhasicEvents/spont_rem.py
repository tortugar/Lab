#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:55:34 2019
plot normalized EEG spectrogram during NREM to REM transition 
along with EMG amplitude

@author: tortugar
"""

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
from functools import reduce


ppath = '/Users/tortugar/Documents/Penn/Data/RawData/'
recordings = ['J55_111018n1', 'J53_110718n1', 'J31_101618n1', 'J54_110518n1']
recordings = ['JS80_050319n1', 'JS79_050319n1']

# minimum duration of REM period
# to be included in calculation:
rem_thr = 60
pre = 60
post = 60
fmax = 150
mu = [10, 100]

# get list of all mice 
mice = []
for name in recordings:
    idf = re.split('_', name) [0]
    if idf not in mice:
        mice.append(idf)


spec_mice = {m:[] for m in mice}
ampl_mice = {m:[] for m in mice}
for name in recordings:
    idf = re.split('_', name) [0]

    # load EMG
    Q = so.loadmat(os.path.join(ppath, name, 'msp_%s.mat'%name), squeeze_me=True)
    SM = Q['mSP']
    freq = Q['freq']
    df = freq[1]-freq[0]
    imu = np.where((freq>=mu[0]) & (freq<=mu[-1]))[0]
    ampl = np.sqrt(SM[imu,:].sum(axis=0)*df)
      
    # load normalized spectrogram
    #SP, t, freq = sleepy.normalize_spectrogram(ppath, name, fmax, pplot=False)
    P = so.loadmat(os.path.join(ppath, name,  'sp_' + name + '.mat'), squeeze_me=True)
    SP = P['SP']
    freq = P['freq']
    t = P['t']

    ifreq = np.where(freq<=fmax)[0]
    SP = SP[ifreq,:]

    sp_mean = SP.mean(axis=1)
    SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)

    M = sleepy.load_stateidx(ppath, name)[0]
    rem_idx = np.where(M==1)[0]
    seq = sleepy.get_sequences(np.where(M==1)[0])
    
    SR = sleepy.get_snr(ppath, name)
    NBIN = np.round(2.5*SR)
    dt = NBIN * 1.0/SR
    
    ipre = int(np.round(pre/dt))
    ipost = int(np.round(post/dt))
    
    for s in seq:
        if len(s)*dt >= rem_thr and s[0]-pre >=0 and s[0]+ipost < len(M):
            tmp = SP[:, s[0]-ipre:s[0]+ipost]
            spec_mice[idf].append(tmp)

            tmp = ampl[s[0]-ipre:s[0]+ipost]
            ampl_mice[idf].append(tmp)
            
n = ipre+ipost
ampl_mx = np.zeros((len(mice), n))
spec_mx = np.zeros((len(ifreq), n, len(mice)))

for (i, idf) in zip(range(n), mice):
    ampl_mx[i,:] = np.array(ampl_mice[idf]).mean(axis=0)
    spec_mx[:,:,i] = np.array(spec_mice[idf]).mean(axis=0)
            

# plot figure ##################
##
sns.set()
plt.figure()
t = np.arange(-ipre, ipost)*dt
ax = plt.axes([0.1, 0.4, 0.8, 0.5])
ax.pcolor(t, freq[ifreq], spec_mx.mean(axis=2), cmap='jet')
sleepy.box_off(ax)
ax.set_xticklabels('') 
plt.ylabel('Freq (Hz)')


amp_data = list(np.reshape(ampl_mx, (len(mice)*len(t),)))
amp_time = list(t)*len(mice)
amp_idf = reduce(lambda x,y: x+y, [[b]*len(t) for b in mice])
data = [[a,b,c] for (a,b,c) in zip(amp_idf, amp_time, amp_data)]
df = pd.DataFrame(columns=['Idf', 'Time', 'Ampl'], data=data)

ax2 = plt.axes([0.1, 0.1, 0.8, 0.2])
sns.lineplot(data=df, x='Time', y='Ampl', ci='sd')
plt.plot(t, ampl_mx.mean(axis=0), color='r')
plt.ylim([0, 30])
plt.xlabel('Time (s)')
plt.xlim([t[0], t[-1]])
sleepy.box_off(ax2)
plt.show()







