
import sys
sys.path.append('/Users/tortugar/Google Drive/Penn/Programming/PySleep')
import sleepy
import numpy as np
import matplotlib.pylab as plt
import scipy.io as so
import os
import pandas as pd
import seaborn as sns

ppath = '/Users/tortugar/Documents/Penn/Data/RawData/'
ppath = '/Volumes/My Passport/mPFC closed-loop data_for Franz'
name = 'J31_082118n1'
#name = 'JS100_062119n1'
#name = 'JS76_050219n1'
name = 'J55_111018n1'
name = 'J53_110718n1'
#name = 'J31_101618n1'

band = [60, 150]
nwin = 10
# if intersect_mode==1, then only
# take the part of REM period that
# overlaps with laser
intersect_mode = 1
g_thr = 0.5
vm = 4

# load spectrogram
SP, t, freq = sleepy.normalize_spectrogram(ppath, name, 150, band)
M = sleepy.load_stateidx(ppath, name)[0]
rem_idx = np.where(M==1)[0]
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
thr = pow_band[rem_idx].mean() + g_thr*pow_band[rem_idx].std()
#thr = np.percentile(pow_band[rem_idx], 90)

idx_g = np.where(pow_band > thr)[0]
idx_g = np.intersect1d(idx_g, rem_idx)
idx_t = np.setdiff1d(rem_idx, idx_g)
seq = sleepy.get_sequences(idx_g)

# figure showing spectrogram along with gamma band including
# colored gamma peaks
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
    i = np.argmax(s)
    si = s[i]
    if si > nwin:
        tmp = SP[:,si-nwin:si+nwin]
        frames.append(tmp)
        idx_p.append(si)
trig_sp = np.array(frames).mean(axis=0)
idx_p = np.array(idx_p)

dt = 2.5
plt.figure()
plt.pcolor(np.arange(-nwin, nwin)*dt, freq, trig_sp, cmap='jet')
plt.show()

sp_phasic = SP_orig[:,idx_g].mean(axis=1)
sp_tonic  = SP_orig[:,idx_t].mean(axis=1)
sp_rem  = SP_orig[:,rem_idx].mean(axis=1)
#sp_phasic = SP[:,idx_g].mean(axis=1)
#sp_tonic  = SP[:,idx_t].mean(axis=1)

plt.figure()
plt.plot(freq_orig, sp_phasic)
plt.plot(freq_orig, sp_tonic)
plt.plot(freq_orig, sp_rem)
plt.draw()


# collect frequency of phasic events during REM periods
# with ('yes') and w/o laser ('no')
data = []                  
# get REM sequences:
srem = sleepy.get_sequences(rem_idx)
for s in srem:
    tmp = np.intersect1d(s, laser_idx)
    if len(tmp) == 0:
        r = np.intersect1d(s, idx_p)
        data.append(['no', 100*float(len(r)) / len(s)])
    else:
        if intersect_mode == 1:
            s = np.intersect1d(s, rem_idx)
        r = np.intersect1d(s, idx_p)
        data.append(['yes', 100*float(len(r)) / len(s)])

df = pd.DataFrame(columns=['lsr', 'freq'], data=data)

plt.figure()
sns.barplot(y = 'freq', x='lsr', data=df, ci=90, palette={'no':'gray', 'yes':'blue'})  
plt.show()

        
