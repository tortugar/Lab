#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 22:53:16 2018

@author: Franz
"""
import scipy.io as so
import os
import numpy as np
import re
import sleepy
import vypro
import matplotlib.pylab as plt
import pdb
from scipy import stats
import seaborn as sns
import pandas as pd
# seaborn is used for coloring single mice,
# https://seaborn.pydata.org/tutorial/color_palettes.html


def empty_annotation(ppath, name):
    SP = so.loadmat(os.path.join(ppath, name, 'sp_' + name + '.mat'), squeeze_me=True)['SP']
    nstates = SP.shape[1]
    
    M = np.ones((nstates,)) * 2
    K = np.zeros((nstates,))
    
    fid = open(os.path.join(ppath, name, 'remidx_' + name + '.txt'), 'w')
    for i in range(nstates):
        fid.write('%d\t%d\n' % (M[i], K[i]))
    fid.close()
    

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


def calculate_1s_spectrum(ppath, name, fres=1.0):
    """
    calculate powerspectrum used for sleep stage detection.
    Function assumes that data vectors EEG.mat and EMG.mat exist in recording
    folder ppath/name; these are used to calculate the powerspectrum
    fres   -   resolution of frequency axis

    all data saved in "true" mat files
    :return  EEG Spectrogram, EMG Spectrogram, frequency axis, time axis
    """

    SR = get_snr(ppath, name)
    n = round(SR)
    if n % 2 == 1:
        n += 1
    swin = n * 2

    fft_win = round(swin / 2)
    if (fres == 1.0) or (fres == 1):
        fft_win = int(fft_win)
    elif fres == 0.5:
        fft_win = 2 * int(fft_win)
    else:
        print("Resolution %f not allowed; please use either 1 or 0.5" % fres)

    (peeg2, pemg2) = (False, False)

    # Calculate EEG spectrogram
    EEG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EEG.mat'))['EEG'])
    Pxx, f, t = sleepy.spectral_density(EEG, int(swin), int(fft_win), 1 / SR)
    if os.path.isfile(os.path.join(ppath, name, 'EEG2.mat')):
        peeg2 = True
        EEG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EEG2.mat'))['EEG2'])
        Pxx2, f, t = sleepy.spectral_density(EEG, int(swin), int(fft_win), 1 / SR)
        # save the stuff to a .mat file
    spfile = os.path.join(ppath, name, 'sp_' + name + '.mat')
    if peeg2 == True:
        so.savemat(spfile, {'SP': Pxx, 'SP2': Pxx2, 'freq': f, 'dt': t[1] - t[0], 't': t})
    else:
        so.savemat(spfile, {'SP': Pxx, 'freq': f, 'dt': t[1] - t[0], 't': t})

    # Calculate EMG spectrogram
    EMG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EMG.mat'))['EMG'])
    Qxx, f, t = sleepy.spectral_density(EMG, int(swin), int(fft_win), 1 / SR)
    if os.path.isfile(os.path.join(ppath, name, 'EMG2.mat')):
        pemg2 = True
        EMG = np.squeeze(so.loadmat(os.path.join(ppath, name, 'EMG2.mat'))['EMG2'])
        Qxx2, f, t = sleepy.spectral_density(EMG, int(swin), int(fft_win), 1 / SR)
    # save the stuff to .mat file
    spfile = os.path.join(ppath, name, 'msp_' + name + '.mat')
    if pemg2 == True:
        so.savemat(spfile, {'mSP': Qxx, 'mSP2': Qxx2, 'freq': f, 'dt': t[1] - t[0], 't': t})
    else:
        so.savemat(spfile, {'mSP': Qxx, 'freq': f, 'dt': t[1] - t[0], 't': t})

    return Pxx, Qxx, f, t



def downsample_tone(ppath, name):
    SR = get_snr(ppath, name)
    n = round(SR)
    if n % 2 == 1:
        n += 1
    print(n)
    tone = so.loadmat(os.path.join(ppath, name, 'tone.mat'), squeeze_me=True)['tone']
    toned = downsample_vec(tone, int(n))
    toned[np.where(toned>0)] = 5
    so.savemat(os.path.join(ppath, name, 'tone.mat'), {'tone':tone, 'toned':toned})



def tone_freezing(ppath, name, freeze_file='', pplot=False):
    """
    calculate duration/percentage of freezing during each tone
    :param ppath: base folder
    :param name: recording
    :param freeze_file: annotation file for freezing
    :param pplot: if True, plot percentage freezing for each tone/trial
    :return: two vectors with total duration of freezing during each tone and percentages of freezing;
    """
    sr = get_snr(ppath, name)
    dt = 1/sr
    if len(freeze_file) == 0:
        freeze_file = 'vip_f.txt'

    ann, K = vypro.load_behann_file(os.path.join(ppath, name, freeze_file))
    time = list(ann.keys())
    time.sort()
    dt_beh = time[1] - time[0]
    time_arr = np.array(time)

    tone = so.loadmat(os.path.join(ppath, name, 'tone.mat'), squeeze_me=True)['tone']
    idxs, idxe = sleepy.laser_start_end(tone)
    idxs = [s*dt for s in idxs]
    idxe = [s*dt for s in idxe]

    freezing = []
    for t in time:
        freezing.append(ann[t])

    resp_dur = []
    resp_per = []
    for (i,j) in zip(idxs, idxe):
        idx = np.where((time_arr>=i) & (time_arr<=j))[0]
        beh = []
        for f in idx:
            state = ann[time[f]]
            if state == 'f':
                beh.append(1)
            else:
                beh.append(0)
        beh = np.array(beh)
        dur_freezing = np.sum(beh) * dt_beh
        per_freezing = 100 * np.sum(beh) / (1.0*len(idx))
        resp_dur.append(dur_freezing)
        resp_per.append(per_freezing)

    if pplot:
        plt.ion()
        plt.figure()
        #plt.figure(figsize=(4,5))
        ax = plt.axes([0.2, 0.1, 0.5, 0.8])
        plt.bar(range(1, len(idxs)+1), resp_per, color='gray')
        plt.xticks(range(1, len(idxs)+1))
        sleepy.box_off(ax)
        plt.xlabel('Trial No.')
        plt.ylabel('% Freezing')
        plt.ylim((0, 100))
        plt.show()

    return np.array(resp_dur), np.array(resp_per)



def write_tone_freezing(ppath, mice):
    dura, perc = tone_freezing(ppath, mice)
    fid = open(os.path.join(ppath, mice, 'freezing.txt'), 'w')
    
    for dur in dura:
        fid.write('%f\t' % dur)
    
    fid.write(os.linesep)
    
    for per in perc:
         fid.write('%f\t' % per)
        
    fid.close()



def tone_freezing_mice(ppath, mice, pplot=True, psingle_mouse=False, pstd=True, alpha=5.0, csv_file=''):
    """
    calculate and plot duration/percenage of freezing during each tone across mice
    :param ppath: base folder
    :param mice: list of recordings
    :param pplot: if True, plot figure
    :param psingle_mouse: if True, plot each single mouse
    :param pstd, if True plot STD, otherwise plot percentiles
    :param alpha: if pstd == False, plot the 1-alpha confidence interval (i.e. the lower and upper ends of the errorbars
           correspond to the $alpha/2 and 1-$alpha/2 percentiles
    :param csv_file, string, if non-empty, write data into the given csv file (file ending ".csv"). $csv_file can be single file name
           or full path including file name
    :return: two np.arrays, number of mice x number of trials, for duration and percentage of freezing during each tone
             same data in pd.DataFrame with mice as index and two levels of columns.
             To get the percentage freezing of all mice during trial2, type df['percentage']['trial2']
    """
    duration = []
    percentage = []
    mouse_order = []
    for name in mice:
        dur, perc = tone_freezing(ppath, name)
        duration.append(dur)
        percentage.append(perc)
        idf = re.split('_', name)[0]
        mouse_order.append(idf)

    duration = np.array(duration)
    percentage = np.array(percentage)
    ntrials = duration.shape[1]

    if pplot:
        clrs = sns.color_palette("husl", len(mouse_order))
        plt.ion()
        plt.figure()
        ax = plt.axes([0.2, 0.1, 0.5, 0.8])
        if psingle_mouse:
            for i in range(len(mouse_order)):
                plt.plot(range(1, ntrials+1), percentage[i,:], label=mouse_order[i], color=clrs[i])
            ax.legend(mouse_order, bbox_to_anchor = (0., 1.0, 1., .102), loc=3, mode='expand', ncol=len(mouse_order), frameon=False)
        else:
            if pstd:
                ax.errorbar(range(1, ntrials + 1), percentage.mean(axis=0), yerr=percentage.std(axis=0), fmt = '', marker = 'o', color='gray',
                            linewidth = 2, elinewidth=1.5)
            else:
                # plot using percentiles on errorbars
                err = np.vstack((percentage.mean(axis=0)-np.percentile(percentage, alpha/2.0, axis=0),
                                 np.percentile(percentage, (100-alpha/2.0), axis=0)-percentage.mean(axis=0)))
                ax.errorbar(range(1, ntrials + 1), percentage.mean(axis=0), yerr=err, fmt = '', marker = 'o', color='gray',
                            linewidth = 2, elinewidth=1.5)

        plt.xticks(range(1, ntrials+1))
        sleepy.box_off(ax)
        plt.xlabel('Trial No.')
        plt.ylabel('% Freezing')
        plt.ylim((0, 100))
        plt.show()

    # put duration and percentage into data frame
    trials = ['trial'+str(i) for i in range(1,ntrials+1)]
    cols = pd.MultiIndex.from_product([['duration', 'percentage'], trials], names=['stats', 'trials'])
    df = pd.DataFrame(index = mouse_order, columns=cols, data=np.hstack((duration,percentage)))
    if len(csv_file) > 0:
        df.to_csv(os.path.join(ppath, csv_file))

    return duration, percentage, df



def behavioral_spectrogram(ppath, name, beh_file, twin=3, fmax=30, mu=[10,100], pplot=True):
    """
    calculate EEG spectrogram and EMG amplitude for each symbol specified in the given annotation file ($beh_file),
    generated by video_processor_stack.py
    :param ppath: base folder
    :param name: recording
    :param beh_file: annotation file generated by video_processor_stack.py
    :param twin: time window for power spectrum calculation
    :param fmax: maximum frequency for EEG spectrogram
    :param mu: tuple, lower and upper limit for EEG frequency axis
    :param pplot: if True, generate plot
    :return: dict: symbols -> EEG spectrogram, frequency axis, dict: symbols -> EMG amplitude
    """
    sr = sleepy.get_snr(ppath, name)
    nwin = np.round(twin * sr)

    ann, _ = vypro.load_behann_file(os.path.join(ppath, name, beh_file))
    time = list(ann.keys())
    time.sort()
    symbols = list(set(ann.values()))

    # load EEG
    P = so.loadmat(os.path.join(ppath, name, 'EEG.mat'), squeeze_me=True)
    EEG = P['EEG']*1000
    EMG = so.loadmat(os.path.join(ppath, name, 'EMG.mat'), squeeze_me=True)['EMG']*1000


    # save for each symbol a list of indices, at which the symbol occurs
    sym2idx = {s:[] for s in symbols}
    for sym in symbols:
        i = 0
        idx = []
        for t in time:
            if ann[t] == sym:
                idx.append(i)
            i += 1
        sym2idx[sym] = np.array(idx)

    sym2pow = dict()
    sym2emg = dict()
    for sym in symbols:
        idx = sym2idx[sym]
        seq = sleepy.get_sequences(idx)
        pow_list = []
        emg_list = []
        for s in seq:
            d = time[s[-1]] - time[s[0]]
            if d >= twin:
                i = time[s[0]]
                j = time[s[-1]]
                i = int(np.round(i * sr))
                j = int(np.round(j * sr))
                pow_eeg, f = sleepy.power_spectrum(EEG[i:j], nwin, 1.0/sr)
                pow_list.append(pow_eeg)

                pow_emg, f = sleepy.power_spectrum(EMG[i:j], nwin, 1.0/sr)
                emg_list.append(pow_emg)

        sym2pow[sym] = np.array(pow_list).mean(axis=0)
        sym2emg[sym] = np.array(emg_list).mean(axis=0)

    ifreq = np.where(f<=fmax)[0]
    df = f[1]-f[0]
    imu = np.where((f>=mu[0]) & (f<=mu[1]))[0]
    for sym in symbols:
        sym2emg[sym] = np.sqrt(np.sum(sym2emg[sym][imu])*df)
        sym2pow[sym] = sym2pow[sym][ifreq]

    if pplot:
        clrs = sns.color_palette("husl", len(symbols))
        plt.ion()
        plt.figure(figsize=(8,4))
        ax = plt.subplot(121)
        i = 0
        for sym in symbols:
            plt.plot(f[ifreq], sym2pow[sym], label=sym, color=clrs[i])
            i += 1
        sleepy.box_off(ax)

        ax = plt.subplot(122)
        i=0
        idx = []
        for sym in symbols:
            plt.bar(i, sym2emg[sym], color=clrs[i])
            idx.append(i)
            i+=1
        sleepy.box_off(ax)
        plt.xticks(idx)
        ax.set_xticklabels(symbols)

        plt.show()

    return sym2pow, f[ifreq], sym2emg



def behavioral_spectrogram_recordings(ppath, recordings, beh_file, twin=3, fmax=30, mu=[10, 100]):
    """
    plot the EEG spectrogram and EMG amplitude across the given list of recordings for each symbol in annotation file $beh_file.
    I assume that each recording's annotation file has the same file name ($be_file).
    Average are calculated over recordings (not mice)
    :param ppath: base folder
    :param recordings: list of recordings
    :param beh_file: annotation file generated by video_processor_stack.py
    :param twin: time window used for power spectrum calculation; the longer the window the finer the frequency scale,
           but the noisier the power estimate.
    :param fmax: float, maximum EEG frequency shown in EEG spectrogram
    :param mu: tuple, lower and upper limit of frequency range used to calculate EMG amplitude
    :return: dict: symbols -> np.array(recordings x frequencies), vector(frequency axis), dict -> vector(EMG Amplitude for each recording)
    """
    rec2pow = {}
    rec2emg = {}
    for rec in recordings:
        rec2pow[rec], f, rec2emg[rec] = behavioral_spectrogram(ppath, rec, beh_file,
                                                               twin=twin, fmax=fmax, mu=mu, pplot=False)

    symbols = list(rec2pow[list(rec2pow.keys())[0]].keys())
    symbols.sort()

    sym2pow = {s:[] for s in symbols}
    sym2emg = {s:[] for s in symbols}
    for sym in symbols:
        for rec in recordings:
            sym2pow[sym].append(rec2pow[rec][sym])
            sym2emg[sym].append(rec2emg[rec][sym])

    for sym in symbols:
        sym2pow[sym] = np.array(sym2pow[sym])
        sym2emg[sym] = np.array(sym2emg[sym])

    clrs = sns.color_palette("husl", len(symbols))

    labels = []
    for s in symbols:
        if len(s) > 0:
            labels.append(s)
        else:
            labels.append('other')

    plt.ion()
    plt.figure(figsize=(10,5))
    ax = plt.subplot(121)
    i = 0
    for sym in symbols:
        tmp = sym2pow[sym].mean(axis=0)
        std = sym2pow[sym].std(axis=0)
        plt.fill_between(f, tmp-std, tmp+std, label=labels[i], color=clrs[i], alpha=0.5)
        i += 1
    sleepy.box_off(ax)
    plt.legend()
    plt.xlabel('Freq. (Hz)')
    plt.ylabel('Power ($\mu V^2$)')

    ax = plt.subplot(122)
    i = 0
    for sym in symbols:
        plt.bar(i, sym2emg[sym].mean(axis=0), edgecolor=clrs[i], fill=False)
        for j in range(len(recordings)):
            plt.plot(i, sym2emg[sym][j], 'o', color=clrs[i])
        i += 1
    sleepy.box_off(ax)
    plt.ylabel('EMG Ampl. ($\mu V$)')
    plt.xticks(range(len(symbols)))
    ax.set_xticklabels(labels)
    plt.show()

    return sym2pow, f, sym2emg



### Statistics ###################################################
def corr_freezing_sleep(fpath, spath, frecs, srecs, trials, sleep_stats=0, istate=1, tstart=0, tend=-1, ma_thr=20, min_dur=0, pplot=True):
    """
    Correlated freeezing behavior with sleep.
    The percentage of freezing can be correlated with any quantify calculated by sleepy.sleep_stats

    :param fpath: base folder of "fear" sessions
    :param spath: base folder of sleep recordings
    :param frecs: list of fear sessions
            NOTE: the order of mice in @frecs must be the same as in @srecs!
    :param srecs: list of sleep recordings
    :param trials: list of trials numbers (counting starts with 1)
    :param sleep_stats: Measured sleep variable (statistics):
            0 - percentage, 1 - episode duration, 2 - episode frequency, 3 - latency to first state occurance of state $istate
    :param istate: 1 - REM, 2 - Wake, 3 - NREM
    :param tstart: float, quantificiation of sleep starts at $start s
    :param tend: float, quantification of sleep ends at $tend s
    :param pplot: if True, plot figure showing scatter plot of freezing vs sleep
    :param ma_thr: float, wake periods shorter than $ma_thr are considered as microarousals and further converted to NREM
    :param min_dur: only used for sleep_stats == 3, minimal duration of state $istate to be counted
    :return: r ravlue, p value of linear fit
    """
    states = {1:'REM', 2:'Wake', 3:'NREM'}
    stats_label = {0:'(%)', 1:'Duration (s)', 2: 'Freq. (1/h)', 3: 'Onset latency (min)'}
    trials = [t-1 for t in trials]
    fmouse_order = []
    for rec in frecs:
        idf = re.split('_', rec)[0]
        if not idf in fmouse_order:
            fmouse_order.append(idf)

    smouse_order = []
    for rec in srecs:
        idf = re.split('_', rec)[0]
        if not idf in smouse_order:
            smouse_order.append(idf)

    if not smouse_order == fmouse_order:
        print("ERROR: sleep and fear recordings do not match!")
        print("...stopping")
        return

    if sleep_stats <= 2:
        perc = sleepy.sleep_stats(spath, srecs, pplot=False, tstart=tstart, tend=tend, ma_thr=ma_thr)[sleep_stats][:,istate-1]
    else:
        perc = sleepy.state_onset(spath, srecs, istate, min_dur, tstart=tstart, tend=tend, pplot=False)
    freezing = tone_freezing_mice(fpath, frecs, pplot=False)[1]
    freezing = np.mean(freezing[:,trials], axis=1)

    xmin = min(perc)
    xmax = max(perc)
    x = np.arange(xmin-(xmax-xmin)/10., xmax+(xmax-xmin)/10., .1)
    slope, intercept, r_value, p_value, _ = stats.linregress(perc, freezing)
    print("linear regression results: r value: %.3f, p value: %.3f" % (r_value, p_value))

    if pplot:
        # get as many different colors as mice
        clrs = sns.color_palette("husl", len(fmouse_order))
        plt.ion()
        plt.figure()
        ax = plt.subplot(111)
        for i in range(len(fmouse_order)):
            plt.scatter(perc[i], freezing[i], color=clrs[i])
            plt.text(perc[i], freezing[i]+1, fmouse_order[i], fontsize=11)

        plt.plot(x, x*slope+intercept, '--', color='gray')
        sleepy.box_off(ax)

        plt.xlabel(states[istate] + ' ' + stats_label[sleep_stats])
        plt.ylabel('Freezing (%)')
        plt.show()

    return r_value, p_value



def tone_paired_ttest(ppath, group1, trial1, group2, trial2):
    """
    Perform paired t-test between recordings during different days/trials. ex) extinction vs. recall, trial 1 vs. trial 10
    ppath= path for recordings
    group1, group2 = list of recordings to compare
    trial1, trial2 = list of trials to compare; trial1, trial2 starts with 1
    """
    
    trial1 = [t-1 for t in trial1]
    trial2 = [t-1 for t in trial2]
    
    dur_trial1 = []
    dur_trial2 = []
    perc_trial1 = []
    perc_trial2 = []
    
    for mice in group1:
        dur, perc = tone_freezing(ppath, mice)
        dur_trial1.append(np.mean(dur[trial1]))
        perc_trial1.append(np.mean(perc[trial1]))
    
    for mice in group2:
        dur, perc = tone_freezing(ppath, mice)
        dur_trial2.append(np.mean(dur[trial2]))
        perc_trial2.append(np.mean(perc[trial2]))
    
    print(perc_trial1)
    print(perc_trial2)
    
    dur_paired = stats.ttest_rel(dur_trial1, dur_trial2)
    perc_paired = stats.ttest_rel(perc_trial1, perc_trial2)

    return dur_paired, perc_paired



def tone_paired_ttest2(ppath, mice, trial1, trial2):
    """
    Perform paired t-test between trials within one session  ex)during extinction, compare trial 1 vs. trial 2
    ppath = path for recording
    mice = list of recordings to compare. should be recordings of the same session like conditioning, extinction or recall
    trial1, trial2 = trials to compare
    """
    
    duration, percentage = tone_freezing_mice(ppath, mice, pplot = False)
    
    ntrials = duration.shape[1]
    dur_trial = []
    perc_trial = []
    
    
    for i in range(ntrials):
        dur_each = []
        perc_each = []
        
        for dur in duration:
            dur_each.append(dur[i])
        
        for perc in percentage:
            perc_each.append(perc[i])
        
        dur_trial.append(dur_each)
        perc_trial.append(perc_each)
        
    dur_paired = stats.ttest_rel(dur_trial[trial1-1], dur_trial[trial2-1])
    perc_paired = stats.ttest_rel(perc_trial[trial1-1], perc_trial[trial2-1])
    
    return dur_paired, perc_paired
            


