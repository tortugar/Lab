#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Franz & Justin
"""
import os
import scipy.io as so
import scipy.signal
import re
import numpy as np
import sleepy
import plotly
import plotly.graph_objs as go
import matplotlib.pylab as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from functools import reduce

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
    downsample the vector x by replacing nbin consecutive \
    bin by their mean \
    @RETURN: the downsampled vector
    """
    n_down = int(np.floor(len(x) / nbin))
    x = x[0:n_down*nbin]
    x_down = np.zeros((n_down,))

    # 0 1 2 | 3 4 5 | 6 7 8
    for i in range(nbin) :
        idx = list(range(i, int(n_down*nbin), int(nbin)))
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



def least_squares(x, y, n):
    A = np.zeros((len(x), n + 1))
    for i in range(n + 1):
        A[:, n - i] = np.power(x, i)

    p = np.linalg.lstsq(A, y)[0]
    return p



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



def fit_dff(a465, a405, sr, nskip=5, wcut=2, wcut405=0, shift_only=False):
    """
    fit 405 to 465 signal
    :param a465: Calcium signal
    :param a405: isosbestic wavelength signal
    :param sr: sampling rate of 405 and 465 signals
    :param nskip: number of seconds at the beginning to throw away for the fit
    :param wcut: float, cutoff frequency (in Hz) for lowpass filtering of 465 signal
    :param wcut405: float, cutoff frequency (in Hz) for lowpass filtering of 405 signal
                    if wcut405 == 0, then wcut405 defaults to wcut
    :param shift_only: if True, only shift (and do not scale) 405 signal to optimally fit 465 signal
    :return: np.vector, DF/F signal
    """
    if wcut405 == 0:
        wcut405 = wcut

    w0 = wcut    / (0.5*sr)
    w1 = wcut405 / (0.5*sr)
    if w0>0:
        a465 = sleepy.my_lpfilter(a465, w0, N=4)
        a405 = sleepy.my_lpfilter(a405, w1, N=4)

    nstart = int(np.round(nskip*sr))
    if shift_only:
        X = np.vstack([np.ones(len(a405))]).T
    else:
        X = np.vstack([a405, np.ones(len(a405))]).T
    p = np.linalg.lstsq(X[nstart:,:], a465[nstart:])[0]
    a465_fit = np.dot(X, p)
    dff = np.divide((a465 - a465_fit), a465_fit)

    return dff



def fit_dff_perc(a465, a405, sr, nskip=5, wcut=2, perc=20):

    w0 = wcut / (0.5*sr)
    if w0>0:
        a405 = sleepy.my_lpfilter(a405, w0, N=4)
        a465 = sleepy.my_lpfilter(a465, w0, N=4)

    nstart = int(np.round(nskip*sr))
    X = np.vstack([a405, np.ones(len(a405))]).T
    p = np.linalg.lstsq(X[nstart:,:], a465[nstart:])[0]

    a465_fit = np.dot(X, p)
    dff = np.divide((a465 - a465_fit), a465_fit)

    pc = np.percentile(dff[nstart:], perc)
    idx = np.where(dff<pc)[0]
    idx = idx[np.where(idx>nstart)[0]]
    X = np.vstack([a405[idx], np.ones(len(idx))]).T
    p = np.linalg.lstsq(X, a465[idx])[0]
    a465_fit = a405*p[0] + p[1]
    dff = np.divide((a465 - a465_fit), a465_fit)

    return dff



def calculate_dff(ppath, name, nskip=5, wcut=2, wcut405=0, perc=0, shift_only=False):
    """
    Calculate DF/F signal for recording $ppath/$name.
    For fitting the DF/F signal disrecard the first $nskip seconds.
    The results are saved to the file $ppath/$name/DFF.mat
    :param ppath: base folder
    :param name: recording
    :param nskip: disregard the first $nskip seconds for DF/F fit
    :param wcut: lowpass filter 465 and 405 signal before fit
    :param perc: only use the lower $perc-th percentile of the DF/F signal
           for the fist; with this approach, it's possible to avoid largely
           negative DF/F signals; if this option is used, a value of perc=20
           is a good choice.
    """
    D = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)
    a465 = D['465']
    a405 = D['405']
    n = len(a465)
    SR = get_snr(ppath, name)
    dt = 1.0/SR

    if perc == 0:
        dff = fit_dff(a465, a405, SR, nskip, wcut=wcut, wcut405=wcut405, shift_only=shift_only)
    else:
        dff = fit_dff_perc(a465, a405, SR, nskip, wcut, perc)

    # number of time bins per
    nbins = int(np.round(SR)*2.5)
    k = int(np.ceil((1.0 * n) / nbins))
    dff2 = np.zeros((k*nbins,))
    dff2[:len(dff)]=dff
    dff = dff2
    dffd = downsample_vec(dff, nbins)

    t = np.linspace(0.0, n*dt, n+1)
    so.savemat(os.path.join(ppath, name, 'DFF.mat'), {'t': t, '405': a405, '465': a465, 'dff':dff, 'dffd':dffd})



def plot_rawtraces(ppath, name, tskip=10, wcut=2.0, ndown=100, vm=3, tstart=0, tend=-1,
                   pzscore=False, ylim=[], fig_file='', color_scheme=1, shift_only=False):
    """
    plot brain state, EEG spectrogram, EMG amplitude, photometry raw traces and DF/F.
    DF/F signal is re-calculated using the given parameters (does not use DFF.mat)
    :param ppath: base folder
    :param name: recording folder
    :param tskip: skip the first $tspike seconds for linear fit of 405 signal to 465 signal; does not affect what is plotted
    :param wcut: cutoff frequency for lowpass filter
    :param ndown: downsampling factor; important to make plots fast and interactive
    :param vm: float, control of colormap for EEG spectrogram (the max. color is vm * median)
    :param tstart: discard first $tstart seconds for plotting, no effect on linear fit
    :param tend: discard last $tend seconds for plotting, no effect on linear fit
    :param color_scheme, 1 - use conventional color scheme, 2 - use color scheme from Chung et al. 2017
    :param shift_only, boolean; if True, only shift the 405 signal to optimally match the 465 signal w/o scaling it.
            So, the formula is 465 ~ 405 + b
    :return: vector, DF/F
    """
    sr = get_snr(ppath, name)
    # number of time bins for each time bin in spectrogram
    nbin = int(np.round(sr)*2.5)
    sdt = nbin * (1/sr)
    nskip = int(tskip/sdt)
    # time step after downsampling
    dt = (1.0/sr)*ndown
    dt_eeg = 1.0 / sr

    # load photometry signals
    D = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)
    a465 = D['465']
    a405 = D['405']

    # lowpass filtering
    w0 = wcut / (0.5 * sr)
    if w0 > 0:
        a405 = sleepy.my_lpfilter(a405, w0, N=4)
        a465 = sleepy.my_lpfilter(a465, w0, N=4)

    # fit 405 to 465 signal
    nstart = int(np.round(nskip*nbin))
    X = np.vstack([a405, np.ones(len(a405))]).T

    if shift_only:
        X1 = np.ones((len(a405),1))
        p = np.linalg.lstsq(X1[nstart:], a465[nstart:]-a405[nstart:])[0]
        p = np.array([1, p[0]])
    else:
        p = np.linalg.lstsq(X[nstart:,:], a465[nstart:])[0]

    afit = np.dot(X, p)
    # DF/F
    dff = np.divide((a465-afit), afit)

    istart = int(np.round(tstart/dt_eeg))
    istart = int(np.round(istart / nbin) * nbin)
    if tend == -1:
        iend = dff.shape[0]
    else:
        iend = int(np.round(tend / dt_eeg))
        iend = int(np.round(iend / nbin) * nbin)

    istart_dn = int(istart / nbin)
    iend_dn   = int(iend / nbin)+1

    a465 = a465[istart:iend]
    a405 = a405[istart:iend]
    afit = afit[istart:iend]
    dff  = dff[istart:iend]

    # downsample all signals
    a465 = downsample_vec(a465,ndown)
    a405 = downsample_vec(a405, ndown)
    afit = downsample_vec(afit, ndown)
    dff = downsample_vec(dff, ndown)
    traw = np.linspace(0, (len(a405) - 1) * dt, len(a405))
    #it = np.argmin(np.abs(traw - nskip))

    # load brainstate
    M,S = sleepy.load_stateidx(ppath, name)
    M = M[istart_dn:iend_dn]

    fmax = 30
    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name), squeeze_me=True)
    SPEEG = P['SP']
    med = np.median(SPEEG.max(axis=0))
    t = np.squeeze(P['t'])[istart_dn:iend_dn]
    t -= t[0]
    freq = P['freq']
    P = so.loadmat(os.path.join(ppath, name, 'msp_%s.mat' % name), squeeze_me=True)
    SPEMG = P['mSP']

    plt.figure()
    plt.ion()
    axes1 = plt.axes([0.1, 0.9, 0.8, 0.05])
    A = np.zeros((1, len(M)))
    A[0, :] = M
    cmap = plt.cm.jet
    if color_scheme==1:
        my_map = cmap.from_list('ha', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    else:
        my_map = cmap.from_list('ha', [[0,0,0],[153./255, 76./255, 9./255], [120./255, 120./255, 120./255], [1, 0.75, 0]], 4)

    tmp = axes1.pcolorfast(t, [0,1], A, vmin=0, vmax=3)
    tmp.set_cmap(my_map)
    axes1.axis('tight')
    tmp.axes.get_xaxis().set_visible(False)
    tmp.axes.get_yaxis().set_visible(False)
    sleepy.box_off(axes1)
    axes1.set_yticks([])

    # show spectrogram
    ifreq = np.where(freq <= fmax)[0]
    axes2 = plt.axes([0.1, 0.75, 0.8, 0.1], sharex=axes1)
    axes2.pcolorfast(t,freq[ifreq],SPEEG[ifreq,istart_dn:iend_dn], vmin=0, vmax=vm*med, cmap='jet')
    axes2.axis('tight')
    plt.ylabel('Freq (Hz)')
    sleepy.box_off(axes2)
    plt.xlim([t[0], t[-1]])

    # EMG band
    r_mu = [50, 500]
    i_mu    = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    p_mu    = SPEMG[i_mu,istart_dn:iend_dn].mean(axis=0)
    axes3 = plt.axes([0.1, 0.6, 0.8, 0.1], sharex=axes2)
    axes3.plot(t, p_mu, color='gray')
    plt.ylabel('EMG (a.u.)')
    plt.xlim((t[0], t[-1]))
    sleepy.box_off(axes3)

    axes4 = plt.axes([0.1, 0.35, 0.8, 0.2], sharex=axes3)
    axes4.plot(traw, a405, color=[0.5,0,1])
    axes4.plot(traw, a465, color=[0,0,1])
    axes4.plot(traw, afit, color=[0.6, 0.6, 1])
    sleepy.box_off(axes4)
    plt.ylabel('Sig.')
    plt.xlim([traw[0], traw[-1]])

    # plot DF/F
    print('here')
    axes5 = plt.axes([0.1, 0.10, 0.8, 0.2], sharex=axes4)
    if pzscore:
        dff = (dff-dff.mean()) / dff.std()
    else:
        dff *= 100

    axes5.plot(traw, dff, color='k')
    sleepy.box_off(axes5)
    plt.xlim([traw[0], traw[-1]])
    plt.xlabel('Time (s)')
    plt.ylabel('DF/F (%)')
    if len(ylim) == 2:
        plt.ylim(ylim)

    plt.setp(axes1.get_xticklabels(), visible=False)
    plt.setp(axes2.get_xticklabels(), visible=False)
    plt.setp(axes3.get_xticklabels(), visible=False)
    plt.setp(axes4.get_xticklabels(), visible=False)
    plt.draw()
    plt.show()

    if len(fig_file) > 0:
        plt.savefig(fig_file)

    return dff



def plot_dff_example(ppath, name, tlegend, dff_scale=10, tstart=0, tend=-1, fmax=30, ma_thr=10,
                     vm=-1, cb_ticks=[], emg_ticks=[], r_mu=[10,100], fw_color=True, fig_file=''):

    sr = get_snr(ppath, name)
    nbin = np.round(2.5 * sr)
    dt = nbin * 1 / sr
    istart = int(np.round(tstart/dt))
    iend   = int(np.round(tend/dt))

    M,K = sleepy.load_stateidx(ppath, name)
    kcut = np.where(K>=0)[0]
    M = M[kcut]
    if tend == -1:
        iend = len(M)
    M = M[istart:iend]
    seq = sleepy.get_sequences(np.where(M==2)[0])
    for s in seq:
        if len(s)*dt <= ma_thr:
            M[s] = 3

    t = np.arange(0, len(M))*dt

    # load EEG and EMG spectrogram
    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name), squeeze_me=True)
    SPEEG = P['SP'] #/ 10000.
    med = np.median(SPEEG.max(axis=0))
    if vm == -1:
        vm = med*2.5
    freq = P['freq']
    P = so.loadmat(os.path.join(ppath, name, 'msp_%s.mat' % name), squeeze_me=True)
    SPEMG = P['mSP'] #/ 10000.

    # load DF/F
    dffd = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)['dffd']*100

    # create figure
    plt.ion()
    plt.figure(figsize=(10,4))

    # show brainstate
    axes_brs = plt.axes([0.1, 0.4, 0.8, 0.05])
    cmap = plt.cm.jet
    if fw_color:
        my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    else:
        my_map = cmap.from_list('brs', [[0,0,0], [153/255.0,76/255.0,9/255.0],
                                        [120/255.0,120/255.0,120/255.0], [1,0.75,0]], 4)
    tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)

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
    axes_cbar = plt.axes([0.82, 0.68, 0.1, 0.2])
    # axes for EEG spectrogram
    axes_spec = plt.axes([0.1, 0.68, 0.8, 0.2], sharex=axes_brs)
    im = axes_spec.pcolorfast(t, freq[ifreq], SPEEG[ifreq, istart:iend], cmap='jet', vmin=0, vmax=vm)
    axes_spec.axis('tight')
    axes_spec.set_xticklabels([])
    axes_spec.set_xticks([])
    axes_spec.spines["bottom"].set_visible(False)
    plt.ylabel('Freq (Hz)')
    sleepy.box_off(axes_spec)
    plt.xlim([t[0], t[-1]])

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
    p_mu = np.sqrt(SPEMG[i_mu, :].sum(axis=0) * (freq[1] - freq[0])) #* 1000.0
    axes_emg = plt.axes([0.1, 0.5, 0.8, 0.1], sharex=axes_spec)
    axes_emg.plot(t, p_mu[istart:iend], color='black')
    axes_emg.patch.set_alpha(0.0)
    axes_emg.spines["bottom"].set_visible(False)
    if len(emg_ticks) > 0:
        axes_emg.set_yticks(emg_ticks)
    plt.ylabel('Ampl. ' + '$\mathrm{(\mu V)}$')
    plt.xlim((t[0], t[-1] + 1))
    sleepy.box_off(axes_emg)

    # show DF/F
    axes_dff = plt.axes([0.1, 0.1, 0.8, 0.25])
    dff_max = np.max(dffd[istart:iend])
    dff_min = np.min(dffd[istart:iend])
    dff_edge = (dff_max - dff_min)*0.1
    axes_dff.plot(t, dffd[istart:iend], color='blue')
    plt.xlim([t[0], t[-1]])
    axes_dff.axes.get_xaxis().set_visible(False)
    axes_dff.axes.get_yaxis().set_visible(False)
    axes_dff.spines["top"].set_visible(False)
    axes_dff.spines["right"].set_visible(False)
    axes_dff.spines["bottom"].set_visible(False)
    axes_dff.spines["left"].set_visible(False)
    plt.ylim([dff_min-dff_edge, dff_max+dff_edge])

    # plot time legend
    axes_legend = plt.axes([0.1, 0.05, 0.8, 0.05])
    plt.ylim((0,1.1))
    plt.xlim([t[0], t[-1]])
    axes_legend.plot([0, tlegend], [1, 1], color='black')
    axes_legend.text(tlegend/4.0, 0.0, str(tlegend) + ' s')
    axes_legend.patch.set_alpha(0.0)
    axes_legend.spines["top"].set_visible(False)
    axes_legend.spines["right"].set_visible(False)
    axes_legend.spines["bottom"].set_visible(False)
    axes_legend.spines["left"].set_visible(False)
    axes_legend.axes.get_xaxis().set_visible(False)
    axes_legend.axes.get_yaxis().set_visible(False)

    axes_scale = plt.axes([0.05, 0.1, 0.05, 0.25])
    plt.ylim([dff_min-dff_edge, dff_max+dff_edge])
    plt.xlim((0, 1))
    plt.plot([0.6, 0.6], [dff_min-dff_edge, dff_min-dff_edge+dff_scale], color='black')
    plt.text(0.25, dff_min-dff_edge+dff_scale*0.6, str(dff_scale) + ' %' , rotation=90)
    axes_scale.patch.set_alpha(0.0)
    axes_scale.spines["top"].set_visible(False)
    axes_scale.spines["right"].set_visible(False)
    axes_scale.spines["bottom"].set_visible(False)
    axes_scale.spines["left"].set_visible(False)
    axes_scale.axes.get_xaxis().set_visible(False)
    axes_scale.axes.get_yaxis().set_visible(False)

    plt.show()
    if len(fig_file) > 0:
        sleepy.save_figure(fig_file)



def brstate_dff(ppath, name, nskip=30, offline_plot=True):
    """
    plot DF/F along with brain state using plotly. The DF/F signal is binned in 2.5 s bins, same as the
    EEG spectrogram.
    :param ppath: base recording folder
    :param name: recording
    :param nskip: number of seconds to be skipped at beginning (to hide initial artefact)
    :param offline_plot: boolean, if True use offline version of plotly
    :return:
    """
    sdt = 2.5
    nskip = int(nskip/sdt)

    # EMG range
    mu = [10, 100]
    # load brain state
    M, S = sleepy.load_stateidx(ppath, name)
    M[np.where(M == 0)] = 3
    M = M[nskip:]
    # load spectrogram
    SP = np.squeeze(so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name))['SP'])[0:40, nskip:]
    freq = np.squeeze(so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name))['freq'])
    im = np.where((freq > mu[0]) & (freq <= mu[1]))[0]
    ampl = np.sqrt(np.squeeze(so.loadmat(
        os.path.join(ppath, name, 'msp_%s.mat' % name))['mSP'])[im, :].sum(axis=0) *
                   (freq[1] - freq[0]))[nskip:]

    dff = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)['dffd'][nskip:]*100

    data_dff = go.Scatter(y=dff, x=np.arange(0, len(dff)) * 2.5, mode='lines', xaxis='x', yaxis='y')

    med = np.median(SP.max(axis=0))

    data_sp = go.Heatmap(z=SP, y=freq[0:40], x=np.arange(0, len(dff)) * 2.5,
                         zmin=0, zmax=med*3,
                         xaxis='x', yaxis='y4',
                         colorscale='Jet',
                         showscale=False)

    data_emg = go.Scatter(y=ampl, x=np.arange(0, len(dff)) * 2.5, mode='lines', xaxis='x', yaxis='y3',
                          line=dict(
                              color=('rgb(0, 0, 0)'),
                              width=2))

    data_m = go.Heatmap(z=[M], x=np.arange(0, len(dff)) * 2.5, showscale=False,
                        xaxis='x',
                        yaxis='y2',
                        colorscale=[[0, 'rgb(0,255,255)'],
                                    [1 / 3., 'rgb(0,255,255)'],
                                    [1 / 3., 'rgb(150,0,255)'],
                                    [2 / 3., 'rgb(150,0,255)'],
                                    [2 / 3., 'rgb(192,192,192)'],
                                    [3 / 3., 'rgb(192,192,192)']])

    layout = go.Layout(
        yaxis=dict(
            domain=[0.0, 0.6],
            title='DF/F (%)',
        ),
        yaxis2=dict(
            domain=[0.62, 0.7],
            ticks='',
            showticklabels=False
        ),
        yaxis3=dict(
            domain=[0.75, 0.85],
            title='EMG (uV)'
        ),
        yaxis4=dict(
            domain=[0.9, 1.0],
            title='Freq (Hz)'
        ),
        xaxis=dict(
            domain=[0, 1],
            anchor='y',
            title='Time (s)',
        ),
        xaxis2=dict(
            domain=[0, 1],
            anchor='y2',
            showticklabels=False,
            ticks='',
        ),
        xaxis3=dict(
            domain=[0, 1],
            anchor='y3',
            showticklabels=False,
            ticks='',
        ),
        xaxis4=dict(
            domain=[0, 1],
            anchor='y4',
            showticklabels=False,
            ticks='',
        ),
        showlegend=False

    )

    fig = go.Figure(data=[data_dff] + [data_m, data_emg, data_sp], layout=layout)
    if offline_plot:
        plotly.offline.plot(fig, filename='brstate_dff.html')
    else:
        plotly.plotly.iplot(fig, filename='brstate_dff.html')



def avg_activity(ppath, name, tstart = 10, tend = -1, awake=False, mu=[10,100]):
    """
    calculate average DF/F activity for given recording.
    &avg_activity uses
    :param ppath: base folder
    :param name: recording
    :param tstart: discard first $tstart seconds for calculation
    :param tend: discard last $tend seconds for calculation
    :param awake: it True, also plot active and quite wake. Active wake is determine
                  based on an EMG threshold; which is the mean EMG amplitude 
                  during wake + 1 std.
    :param mu: range of frequencies used to calculated EMG amplitude

    :return: np.array( average REM, Wake, NREM activity )
    """
    sr = get_snr(ppath, name)
    # number of time bins for each time bin in spectrogram
    nbin = int(np.round(sr)*2.5)
    sdt = nbin * (1/sr)
    istart = int(np.round(tstart/sdt))
    # load brain state
    M = sleepy.load_stateidx(ppath, name)[0]
    if tend == -1:
        iend = M.shape[0]
    else:
        iend = int(np.round(tend/sdt))
    M = M[istart:iend]

    ddir = os.path.join(ppath, name)
    if os.path.isfile(os.path.join(ddir, 'dffd.mat')):
        dff = so.loadmat(os.path.join(ddir, 'dffd.mat'), squeeze_me=True)['dffd']
    else:
        dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
        print('%s - saving dffd.mat' % name)
        so.savemat(os.path.join(ddir, 'dffd.mat'), {'dffd': dff})
    #dff = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)['dffd'][istart:iend]

    dff_mean = np.zeros((5,))
    dff_std  = np.zeros((5,))

    for i in range(1,4):
        idx = np.where(M==i)[0]
        dff_mean[i-1] = np.mean(dff[idx])*100
        dff_std[i-1]  = np.std(dff[idx]*100)/np.sqrt(len(idx))

    if not awake:
        plt.ion()
        plt.figure()
        ax = plt.axes([0.3, 0.1, 0.35, 0.8])
        ax.bar([1,2,3], dff_mean[0:3], yerr=dff_std[0:3], align='center', color='gray')
        sleepy.box_off(ax)
        plt.xticks([1,2,3], ['REM', 'Wake', 'NREM'])
        plt.ylabel('DF/F (%)')
        plt.title(name)
        plt.show()
    else:
        tmp = so.loadmat(os.path.join(ppath, name, 'msp_%s.mat'%name), squeeze_me=True)
        MSP = tmp['mSP']
        freq = tmp['freq']
        df = freq[1] - freq[0]
        imu = np.where((freq>=mu[0]) & (freq<=mu[1]))[0]
        
        MSP = MSP[:,istart:iend]
        widx = np.where(M==2)[0]
        ampl = np.sqrt(MSP[imu, :].sum(axis=0)*df)
        wampl = ampl[widx]
        thr = wampl.mean() + wampl.std()
        awk_idx = widx[np.where(wampl>thr)[0]]
        qwk_idx = np.setdiff1d(widx, awk_idx)
        M[awk_idx] = 4
        dff_mean[3] = np.mean(dff[awk_idx])*100       
        dff_std[3]  = np.std(dff[awk_idx]*100)/np.sqrt(len(awk_idx))
        dff_mean[4] = np.mean(dff[qwk_idx])*100       
        dff_std[4]  = np.std(dff[qwk_idx]*100)/np.sqrt(len(qwk_idx))
        
        pdb.set_trace()
        plt.ion()
        plt.figure()
        ax = plt.axes([0.3, 0.1, 0.35, 0.8])
        ax.bar([1,2,3,4,5], dff_mean, yerr=dff_std, align='center', color='gray')
        sleepy.box_off(ax)
        plt.xticks([1,2,3,4,5], ['REM', 'Wake', 'NREM', 'QWK', 'AWK'])
        plt.ylabel('DF/F (%)')
        plt.title(name)
        plt.show()
        
            
    return dff_mean



def avg_activity_recs(ppath, recordings, tstart=10, tend=-1, backup='', pzscore=False, awake=False,
                      ma_thr=0, mu=[10,100], fig_file='', csv_file=''):
    """
    calculate average DF/F activity during each brainstate for a list of recordings.
    If the recordings come from several mice, the function averages across mice.
    If all the recordings come from only one mouse, the function averages across the recordings.

    Statistics:
    (1) The function performs for each mouse one-way ANOVA to test if the DF/F activity is significantly modulated by
    the brainstate. Second, the function then performs a Tukey's post-hoc test to figure out which pair-wise group
    comparisons are significant and during which brain state a given mouse shows strongest calcium activity
    (2) The function then also calculates whether the population activity (across mice) is significantly modulated
    by the brain state using one-way ANOVA followed by Tukey's test

    :param ppath: base folder
    :param recordings: recording
    :param tstart: discard first $tstart seconds for calculation
    :param tend: discard last $tend seconds for calculation
    :param backup: string, optional backup folder for recordings; e.g. an external hard drive
    :param pzscore: if True, z-score data
    :param awake: it True, also plot active and quite wake. Active wake is determine
                  based on an EMG threshold; which is the mean EMG amplitude 
                  during wake + 1 std.
    :param ma_thr: set wake sequences <= ma_thr seconds to NREM
    :param mu: range of frequencies used to calculated EMG amplitude
    :param fig_file: string, if non-empty, save file to specific file name; can be just a file name
           or complete path including file name. If single filename, the file is saved, in the
           current working directory from where python was started
    :param csv_file: string, if non-empty, write average DF/F values into csv file
    :return: pandas Dataframe: rows - mouse IDs, columns - REM,Wake,NREM
    """
    if type(recordings) != list:
        recordings = [recordings]

    nstates = 3
    if awake:
        nstates = 5

    paths = dict()
    for rec in recordings:
        if os.path.isdir(os.path.join(ppath, rec)):
            paths[rec] = ppath
        else:
            paths[rec] = backup

    mean_act = {}
    mean_var = {}
    num_states = {}
    state_vals = {}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        mean_act[idf] = []
        mean_var[idf] = []
        state_vals[idf] = {1:[], 2:[], 3:[], 4:[], 5:[]}
        num_states[idf] = []

    for rec in recordings:
        idf = re.split('_', rec)[0]
        sr = get_snr(ppath, rec)
        # number of time bins for each time bin in spectrogram
        nbin = int(np.round(sr)*2.5)
        sdt = nbin * (1/sr)
        istart = int(np.round(tstart/sdt))
        # load brain state
        M = sleepy.load_stateidx(ppath, rec)[0]
        # flatten out microarousals
        if ma_thr > 0:
            seq = sleepy.get_sequences(np.where(M == 2)[0])
            for s in seq:
                if np.round(len(s)*sdt) <= ma_thr:
                    M[s] = 3

        if tend == -1:
            iend = M.shape[0]
        else:
            iend = int(np.round(tend/sdt))
        M = M[istart:iend]

        dff = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dffd'][istart:iend]
        if pzscore:
            dff = (dff-dff.mean()) / dff.std()
        else:
            dff *= 100.0

        dff_mean = np.zeros((5,))
        dff_var  = np.zeros((5,))
        num = np.zeros((5,))
        
        for i in range(1,4):
            idx = np.where(M==i)[0]
            dff_mean[i-1] = np.mean(dff[idx])
            dff_var[i-1]  = np.var(dff[idx])
            state_vals[idf][i] = np.concatenate((state_vals[idf][i], dff[idx]))
            num[i-1] += len(idx)
        
        if awake:
            tmp = so.loadmat(os.path.join(ppath, rec, 'msp_%s.mat'%rec), squeeze_me=True)
            MSP = tmp['mSP']
            freq = tmp['freq']
            df = freq[1] - freq[0]
            imu = np.where((freq>=mu[0]) & (freq<=mu[1]))[0]
            
            MSP = MSP[:,istart:iend]
            widx = np.where(M==2)[0]
            ampl = np.sqrt(MSP[imu, :].sum(axis=0)*df)
            wampl = ampl[widx]
            thr = wampl.mean() + wampl.std()
            awk_idx = widx[np.where(wampl>thr)[0]]
            qwk_idx = np.setdiff1d(widx, awk_idx)
            M[awk_idx] = 4
            dff_mean[3] = np.mean(dff[awk_idx])       
            dff_var[3]  = np.std(dff[awk_idx])/np.sqrt(len(awk_idx))
            dff_mean[4] = np.mean(dff[qwk_idx])       
            dff_var[4]  = np.std(dff[qwk_idx])/np.sqrt(len(qwk_idx))
        

        mean_act[idf].append(dff_mean)
        mean_var[idf].append(dff_var)
        num_states[idf].append(num)

    nmice = len(mean_act)
    mean_mx = np.zeros((nmice, 5))
    var_mx  = np.zeros((nmice, 5))
    num_mx  = np.zeros((nmice, 5))
    i = 0
    for idf in mean_act:
        mean_mx[i,:] = np.array(mean_act[idf]).mean(axis=0)
        var_mx[i, :] = np.array(mean_var[idf]).mean(axis=0)
        num_mx[i, :] = np.array(num_states[idf]).sum(axis=0)
        i += 1

    # FIGURE
    plt.figure()
    plt.ion()
    ax = plt.axes([0.3, 0.1, 0.35, 0.8])

    if nmice == 1:
        ax.bar(range(1, nstates+1), mean_mx[:,0:nstates], yerr=np.sqrt(var_mx[:,0:nstates]) / np.sqrt(num_mx[:,0:nstates]), align='center', color='gray')
    else:
        ax.bar(range(1, nstates+1), mean_mx[:,0:nstates].mean(axis=0), align='center', color='gray')
        for i in range(nmice):
            plt.plot(range(1, nstates+1), mean_mx[i,0:nstates], color='black')
    sleepy.box_off(ax)
    if awake:
        plt.xticks(range(1, nstates+1), ['REM', 'Wake', 'NREM', 'AWK', 'QWK'])
    else:
        plt.xticks(range(1, nstates+1), ['REM', 'Wake', 'NREM'])
    if not pzscore:
        plt.ylabel('$\Delta$F/F (%)')
    else:
        plt.ylabel('$\Delta$F/F (z scored)')
    plt.show()

    if len(fig_file) > 0:
        sleepy.save_figure(fig_file)

    # STATISTICSs
    # (1) Single mouse statistics
    from statsmodels.stats.multicomp import MultiComparison
    fvalues = {}
    pvalues = {}
    state_class = {idf:[] for idf in state_vals}
    for idf in state_vals:
        res_anova = stats.f_oneway(state_vals[idf][1], state_vals[idf][2], state_vals[idf][3])
        fvalues[idf] = res_anova[0]
        pvalues[idf] = res_anova[1]

        # get data in shape for tukey post-hoc analysis
        data = np.concatenate((state_vals[idf][1], state_vals[idf][2], state_vals[idf][3]))
        labels = np.concatenate( (np.ones((len(state_vals[idf][1]),), dtype='int'), 2*np.ones((len(state_vals[idf][2]),), dtype='int'),
                                  3*np.ones((len(state_vals[idf][3]),),dtype='int') ))
        mc = MultiComparison(data, labels)
        results = mc.tukeyhsd()
        print(results)
        meandiffs = results.meandiffs * -1
        reject = results.reject

        if reject[0]:
            if meandiffs[0] > 0:
                # R>W
                state_class[idf].append('R>W')
            else:
                state_class[idf].append('W>R')

        if reject[1]:
            if meandiffs[1] > 0:
                # R>N
                state_class[idf].append('R>N')
            else:
                state_class[idf].append('R<N')

        if reject[2]:
            if meandiffs[2] > 0:
                # W>N
                state_class[idf].append('W>N')
            else:
                state_class[idf].append('W<N')
        #print (results)

    for idf in state_vals:
        print("Mouse %s: f-value: %.3f, p-value: %.5f, significant grp comparisons: %s" % (idf, fvalues[idf], pvalues[idf], " ".join(state_class[idf])))

    # (2) perform analysis across populations of mice
    mice = list(state_vals.keys())
    res_anova = stats.friedmanchisquare(mean_mx[:,0], mean_mx[:,1], mean_mx[:,2])
    print("\nPopulation statistics - Friedman X-square test: statistics: %.3f, p-value: %.3f" % (res_anova[0], res_anova[1]))
    data = np.reshape(mean_mx[:,0:3].T, (nmice*3,))
    labels = ['REM']*nmice + ['Wake']*nmice + ['NREM']*nmice
    mc = MultiComparison(data, labels)
    results = mc.tukeyhsd()
    print(results)

    # Generate pandas DataFrame and save to csv file, if requested
    if awake:
        columns = ['REM', 'Wake', 'NREM', 'AWK', 'QWK']
    else:
        columns = ['REM', 'Wake', 'NREM']
    df = pd.DataFrame(mean_mx[:,:nstates], index=mice, columns=columns)
    if len(csv_file) > 0:
        if not re.match('\.csv$', csv_file):
            csv_file += '.csv'
        df.to_csv(csv_file)

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

    return df



def corr_activity(ppath, recordings, states, nskip=10, pzscore=True, bands=[]):
    """
    correlate DF/F during states with delta power, theta power, sigma power and EMG amplitude
    :param ppath: base filder
    :param recordings: list of recordings
    :param states: list of len 1 to 3, states to correlate EEG power with; if you want to correlate power during
           NREM and REM, then set states = [3,1]
    :param nskip: number of seconds in the beginning to be skipped
    :param pzscore, if Tue z-score activity, i.e. DF/F - mean(DF/F) / std(DF/F)
    :return: n/a
    """
    # Fixed Parameters
    sf_spectrum = 5
    if len(bands) == 0:
        eeg_bands = [[0.5, 4], [6, 10], [10, 15], [100, 150]]
    else:
        eeg_bands = bands
    # EMG band
    emg_bands = [[10, 100]]

    bands = eeg_bands + emg_bands
    bands = {k:bands[k] for k in range(len(bands))}
    nbands = len(bands)

    mice = dict()
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())

    # dict, Band -> Mouse ID -> values
    Pow = {m:{} for m in mice}
    DFF = {m:[] for m in mice}
    for m in mice:
        d = {b:[] for b in range(nbands)}
        Pow[m] = d

    for rec in recordings:
        idf = re.split('_', rec)[0]
        sr = get_snr(ppath, rec)
        # number of time bins for each time bin in spectrogram
        nbin = int(np.round(sr) * 2.5)
        sdt = nbin * (1 / sr)
        nskip = int(nskip / sdt)

        M = sleepy.load_stateidx(ppath, rec)[0][nskip:]

        ddir = os.path.join(ppath, rec)
        if os.path.isfile(os.path.join(ddir, 'dffd.mat')):
            dff_rec = so.loadmat(os.path.join(ddir, 'dffd.mat'), squeeze_me=True)['dffd']
        else:
            dff_rec = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
            print('%s - saving dffd.mat' % rec)
            so.savemat(os.path.join(ddir, 'dffd.mat'), {'dffd': dff_rec})
        #dff_rec = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dffd'][nskip:]*100.0
        if pzscore:
            dff_rec = (dff_rec - dff_rec.mean()) / dff_rec.std()

        # collect all brain state indices
        idx = []
        for s in states:
            idx.append(np.where(M==s)[0])
        bs_idx = reduce(lambda x,y:np.concatenate((x,y)), idx)

        # load spectrogram and normalize
        P = so.loadmat(os.path.join(ppath, rec, 'sp_%s.mat' % rec), squeeze_me=True)
        SP = P['SP']
        freq = P['freq']
        df = freq[1] - freq[0]
        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)

        # load EMG
        MP = so.loadmat(os.path.join(ppath, rec, 'msp_%s.mat' % rec), squeeze_me=True)['mSP']

        # calculate EEG bands
        for b in range(nbands-1):
            ifreq = np.where((freq >= bands[b][0]) & (freq <= bands[b][1]))[0]
            tmp = SP[ifreq,:].sum(axis=0)*df
            tmp = sleepy.smooth_data(tmp, sf_spectrum)
            Pow[idf][b] = np.concatenate((Pow[idf][b], tmp[bs_idx]))
        # add EMG band
        b = nbands-1
        ifreq = np.where((freq >= bands[b][0]) & (freq <= bands[b][1]))[0]
        tmp = MP[ifreq, :].sum(axis=0) * df
        tmp = sleepy.smooth_data(tmp, sf_spectrum)
        Pow[idf][b] = np.concatenate((Pow[idf][b], tmp[bs_idx]))

        DFF[idf] = np.concatenate((DFF[idf], dff_rec[bs_idx]))

    # collapse all Power values and dff values
    PowAll = {b:[] for b in bands}
    DFFAll = []
    for b in bands:
        for m in mice:
            PowAll[b] = np.concatenate((PowAll[b], Pow[m][b]))

    for m in mice:
        DFFAll = np.concatenate((DFFAll, DFF[m]))

    r_values = {}
    for b in bands:
        p = linregress(PowAll[b], DFFAll)
        r_values[b] = p

    plt.ion()
    plt.figure(figsize=(12,6))
    nx = 1.0/nbands
    dx = 0.2 * nx

    i=0
    for b in bands:
        ax = plt.axes([nx * i + dx, 0.15, nx - dx - dx / 2.0, 0.3])
        j=0
        for m in mice:
            ax.plot(Pow[m][b], DFF[m], '.', color=[j*nx,j*nx,j*nx])
            j+=1
        i+=1
        if b>0:
            #ax.set_yticklabels([])
            pass
        if b<nbands-1:
            plt.xlabel('EEG Power')
        else:
            plt.xlabel('EMG Power')
        plt.title('%.2f<f<%.2f, r2=%.2f' % (bands[b][0], bands[b][1], r_values[b][2]), fontsize=10)
        if b==0:
            if pzscore:
                plt.ylabel('DF/F (z-scored)')
            else:
                plt.ylabel('DF/F')

        sleepy.box_off(ax)

        x = np.linspace(PowAll[b].min(), PowAll[b].max(), 100)
        ax.plot(x, x*r_values[b][0]+r_values[b][1], color='blue')

    plt.draw()

    return r_values



def freqband_vs_activity(ppath, name, band, win=120, ndown=763, tstart=0, tend=-1, ma_thr=0, vm=-1, fmax=30, r_mu = [10, 100], states=[1,2,3]):
    """
    plot frequency band of EEG spectrogram vs DF/F
    and plot cross-correlation of DF/F and EEG band.
    Note: Negative time points in the cross-correlation mean that 
    the neural activity precedes activation in the power band

    :param ppath: base folder
    :param name: name of recording
    :param band: tuple, frequency band (e.g. (0.5, 4.5) for delta power)
    :param win: float, window size (in seconds) for cross-correlation; e.g. if win=2, then the cross-correlation
                will be shown from -2 to 2 s.
    :param ndown: downsmapling factor for DF/F signal
    :param tstart: not yet implemented
    :param tend: not yet implemented
    :param ma_thr: threshold for microarousals
    :param vm: saturation of heatmap, specifcies multiple of median power, values between 2 to 3 are reasonable
    :param fmax: maximum frequency shown for EEG spectrogram
    :param r_mu: frequency range to calculate EMG amplitude
    :param states: list of int, states for which cross-correlation is calculated
                   1 - REM, 2 - Wake, 3 - NREM
    :return: n/a
    """
    sr = get_snr(ppath, name)
    nbin = np.round(2.5 * sr)
    dt = nbin * 1 / sr
    istart = int(np.round(tstart/dt))
    iend   = int(np.round(tend/dt))

    M,K = sleepy.load_stateidx(ppath, name)
    kcut = np.where(K>=0)[0]
    M = M[kcut]
    if tend == -1:
        iend = len(M)
    M = M[istart:iend]
    seq = sleepy.get_sequences(np.where(M==2)[0])
    if ma_thr > 0:
        for s in seq:
            if len(s)*dt <= ma_thr:
                M[s] = 3
    # time axis for spectrogram and brain state
    t = np.arange(0, len(M))*dt+dt

    # load EEG and EMG spectrogram
    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name), squeeze_me=True)
    SPEEG = P['SP'] / 10000.
    med = np.median(SPEEG.max(axis=0))
    if vm == -1:
        vm = med*2.5
    freq = P['freq']
    dt = P['dt']
    P = so.loadmat(os.path.join(ppath, name, 'msp_%s.mat' % name), squeeze_me=True)
    SPEMG = P['mSP'] / 10000.

    # load DF/F
    dff = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)['dff']*100
    dffd = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)['dffd']*100
    dffd2 = downsample_vec(dff, int(ndown))
    tdff = np.arange(0, len(dffd2))*((1.0/sr)*ndown)

    # create figure
    plt.ion()
    plt.figure(figsize=(10,4))

    # show brainstate
    axes_brs = plt.axes([0.1, 0.92, 0.8, 0.05])
    cmap = plt.cm.jet
    my_map = cmap.from_list('brs', [[0, 0, 0], [0, 1, 1], [0.6, 0, 1], [0.8, 0.8, 0.8]], 4)
    tmp = axes_brs.pcolorfast(t, [0, 1], np.array([M]), vmin=0, vmax=3)
    tmp.set_cmap(my_map)
    #axes_brs.set_xticklabels([])
    #axes_brs.axis('tight')
    axes_brs.axes.get_xaxis().set_visible(False)
    axes_brs.axes.get_yaxis().set_visible(False)
    axes_brs.spines["top"].set_visible(False)
    axes_brs.spines["right"].set_visible(False)
    axes_brs.spines["bottom"].set_visible(False)
    axes_brs.spines["left"].set_visible(False)

    # show spectrogram
    ifreq = np.where(freq <= fmax)[0]
    # axes for EEG spectrogram
    axes_spec = plt.axes([0.1, 0.70, 0.8, 0.2], sharex=axes_brs)
    axes_spec.pcolorfast(t, freq[ifreq], SPEEG[ifreq, istart:iend], cmap='jet', vmin=0, vmax=vm)
    #axes_spec.axis('tight')
    #axes_spec.set_xticklabels([])
    axes_spec.spines["bottom"].set_visible(False)
    plt.ylabel('Freq (Hz)')
    sleepy.box_off(axes_spec)

    # show EMG
    axes_emg = plt.axes([0.1, 0.57, 0.8, 0.1], sharex=axes_brs)
    i_mu = np.where((freq >= r_mu[0]) & (freq <= r_mu[1]))[0]
    # * 1000: to go from mV to uV
    p_mu = np.sqrt(SPEMG[i_mu, :].sum(axis=0) * (freq[1] - freq[0]))
    axes_emg.plot(t, p_mu, color='black')
    plt.ylabel('EMG')
    sleepy.box_off(axes_emg)

    # show band
    iband = np.where((freq>=band[0]) & (freq<band[1]))[0]
    axes_band = plt.axes([0.1, 0.35, 0.8, 0.18], sharex=axes_brs)
    pow_band = SPEEG[iband,:].sum(axis=0)*dt
    plt.plot(t, pow_band, color='gray')
    plt.xlim((t[0], t[-1]))
    plt.ylabel('EEG band')
    sleepy.box_off(axes_band)

    # plot dff
    axes_dff = plt.axes([0.1, 0.1, 0.8, 0.18], sharex=axes_band)
    plt.plot(tdff, dffd2, color='blue')
    plt.xlim((t[0], t[-1]))
    plt.xlabel('Time (s)')
    plt.ylabel('$\Delta F / F$')
    sleepy.box_off(axes_dff)

    plt.setp(axes_brs.get_xticklabels(), visible=False)
    plt.setp(axes_spec.get_xticklabels(), visible=False)
    plt.setp(axes_emg.get_xticklabels(), visible=False)
    plt.setp(axes_band.get_xticklabels(), visible=False)
    plt.show()

    # next plot figure showing cross correlation
    plt.figure()
    ax = plt.subplot(111)
    sleepy.box_off(ax)
    plt.title('Cross-Correlation between $\Delta$F/F and power band')
    m = np.min([pow_band.shape[0], dffd.shape[0]])

    tmp = np.array([], dtype='int')
    for s in states:
        idx = np.where(M==s)[0]
        tmp = np.concatenate((tmp, idx))
    idx = tmp
    idx = idx[idx < (m-1)]
    iwin = win / dt
    #pdb.set_trace()
    xx = scipy.signal.correlate(dffd[0:m][idx]-dffd.mean(), pow_band[0:m][idx]-pow_band.mean(),  'same')
    # negative time points mean that the neural activity precedes activation in the power band
    ii = np.arange(len(xx)/2-iwin, len(xx)/2+iwin+1)
    ii = [int(i) for i in ii]
    t = np.arange(-iwin, iwin+1)*dt
    plt.plot(t, xx[ii], color='k')
    plt.xlim([t[0], t[-1]])
    plt.xlabel('Time (s)')
    plt.ylabel('Correlation (a.u.)')
    


def bandpass_corr(ppath, name, band, win=120, state=3, tbreak=60, pemg=False):

    sr = get_snr(ppath, name)
    nwin = int(np.round(sr/4.0))+1
    dff = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)['dff']*100
    dffd = sleepy.downsample_vec(dff, int(nwin))
    if pemg:
        EEG = so.loadmat(os.path.join(ppath, name, 'EMG.mat'), squeeze_me=True)['EMG']
    else:
        EEG = so.loadmat(os.path.join(ppath, name, 'EEG.mat'), squeeze_me=True)['EEG']

    Pow, f, t = sleepy.spectral_density(EEG, 2*nwin, nwin, 1.0/sr)
    ifreq = np.where((f>=band[0]) & (f<=band[1]))[0]
    band = Pow[ifreq,:].sum(axis=0)

    plt.ion()
    plt.figure()
    dt = t[1]-t[0]
    iwin = win / dt
    m = np.min([band.shape[0], dffd.shape[0]])
    xx = scipy.signal.correlate(dffd[1:m], band[0:m-1])
    ii = np.arange(len(xx)/2-iwin, len(xx)/2+iwin+1)
    ii = [int(i) for i in ii]
    t = np.arange(-iwin, iwin+1)*dt
    plt.plot(t, xx[ii])
    plt.show()



def bandpass_corr_state(ppath, name, band, fft_win=2.5, perc_overlap=0.8, win=120, state=3, 
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
    if sr==0:
        sr = get_snr(ppath, name)
    nbin = int(np.round(2.5 * sr))
    nwin = int(np.round(sr * fft_win))
    if nwin % 2 == 1:
        nwin += 1
    noverlap = int(nwin*perc_overlap)
    dff = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)['dff']

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
        plt.ylabel('Corr. $\Delta F/F$ vs. EEG')
        plt.xlim([t[0], t[-1]])
        sleepy.box_off(ax)
        plt.show()

    return CC, t



def bandpass_corr_state_avg(ppath, recordings, band, win=120, fft_win=2.5, perc_overlap=0.8, state=3, tbreak=10, mode='cross', pemg=False, sr=0):
    
    data = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
        CC, t = bandpass_corr_state(ppath, rec, band, win=win, state=state, tbreak=tbreak, mode=mode, pemg=pemg, fft_win=fft_win, perc_overlap=perc_overlap, pplot=False, sr=0)
        ccmean = np.array(CC).mean(axis=0)
        data += zip([idf]*ccmean.shape[0], [rec]*ccmean.shape[0], list(ccmean), list(t))
        
    df = pd.DataFrame(data=data, columns=['mouse', 'recording', 'cc', 'time'])
    
    plt.figure()
    dfm = df.groupby(['mouse', 'recording', 'time']).mean().reset_index()
    sns.lineplot(data=df.groupby(['mouse', 'time']).mean().reset_index(), x='time', y='cc', ci=None)

    return dfm



def pearson_state_corr(ppath, recordings, band, pnorm_spec=True, pzscore=True, ma_thr=0, pplot=True):
    """
    calculate the Pearson correlation between the power in the given frequency band and the DF/F activity
    :param ppath: base folder
    :param recordings: list of recordings
    :param band: tuple, frequency band
    :param pnorm_spec: bool, if True normalize the EEG spectrogram before calculation power in given band
    :param pzscore: if True, z-score DF/F signal
    :param pplot: if True, plot correlation
    :return: pandas.DataFrame, with mouse, recording, r-value, p-value, signficiance and state as columns
    """
    data = []
    raw_data = []
    state_map = {1:'REM', 2:'Wake', 3:'NREM'}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        print('Processing mouse %s' % idf)
        # load DF/F
        ddir = os.path.join(ppath, rec)
        if os.path.isfile(os.path.join(ddir, 'dffd.mat')):
            dff = so.loadmat(os.path.join(ddir, 'dffd.mat'), squeeze_me=True)['dffd']
        else:
            dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
            print('%s - saving dffd.mat' % rec)
            so.savemat(os.path.join(ddir, 'dffd.mat'), {'dffd': dff})


        if pzscore:
            dff = (dff - dff.mean()) / dff.std()
        else:
            dff *= 100

        # load spectrogram and normalize
        P = so.loadmat(os.path.join(ddir, 'sp_%s.mat' % rec), squeeze_me=True)
        SP = P['SP']
        freq = P['freq']
        ifreq = np.where((freq >= band[0]) & (freq <= band[1]))[0]
        df = freq[1] - freq[0]


        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr)*2.5)
        dt = (1.0/sr)*nbin


        if pnorm_spec:
            sp_mean = SP.mean(axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
            pow_band = SP[ifreq, :].mean(axis=0)
        else:
            pow_band = SP[ifreq,:].sum(axis=0)*df

        M = sleepy.load_stateidx(ppath, rec)[0]
        seq = sleepy.get_sequences(np.where(M==2)[0])
        if ma_thr > 0:
            for s in seq:
                if np.round(len(s)*dt) <= ma_thr:
                    M[s] = 3
        
        for s in [1,2,3]:
            idx = np.where(M==s)[0]
            r,p = scipy.stats.pearsonr(dff[idx], pow_band[idx])
            if p < 0.05:
                sig = 'yes'
            else:
                sig = 'no'
            data.append([idf, rec, r, p, sig, state_map[s]])
            raw_data += zip([idf]*len(idx), dff[idx], pow_band[idx], [state_map[s]]*len(idx))

    df = pd.DataFrame(data=data, columns=['mouse', 'recording', 'r', 'p', 'sig', 'state'])
    df_raw = pd.DataFrame(data=raw_data, columns=['mouse', 'dff', 'band', 'state'])

    if pplot:
        dfm_sig = df[df.p > 0.05].groupby(['mouse', 'state']).mean().reset_index()
        dfm_nsg = df[df.p <= 0.05].groupby(['mouse', 'state']).mean().reset_index()

        plt.figure()
        sns.swarmplot(data=dfm_sig, x='state', y='r', hue='sig')

    return df, df_raw



def activity_transitions(ppath, recordings, transitions, pre, post, si_threshold, sj_threshold,
                         backup='', mu=[10, 100], fmax=30, ma_thr=0, ylim=[], xticks=[], cb_ticks=[], vm=[],
                         tstart=0, tend=-1, pzscore=False, sf=0, base_int=10, mouse_stats=True, mouse_avg=True, ci='sem', fig_file=''):
    """
    calculate average DFF activity along brain state transitions
    :param ppath: base folder
    :param recordings: list of recordings
    :param transitions: list of tuples to denote transitions to be considered;
           1 - REM, 2 - Wake, 3 - NREM; For example to calculate NREM to REM and REM to wake transitions,
           type [(3,1), (1,2)]
    :param pre: time before transition in s
    :param post: time after transition
    :param si_threshold: list of floats, thresholds how long REM, Wake, NREM should be at least before the transition.
           So, if there's a REM to Wake transition, but the duration of REM is shorter then si_threshold[0], then this
           transition if discarded.
    :param sj_threshold: list of floats, thresholds how long REM, Wake, NREM should be at least after the transition
    :param backup: string, potential backup path
    :param mu: tuple, specifying the lower and upper limit of frequencies for EMG amplitude calculation
    :param fmax: maximum frequency shown for EEG spectrogram
    :param ma_thr: threshold in seconds for microarousals; i.e. wake periods < ma_thr are considered as microarousals
           and instead treated as NREM (if ma_polish == True)
    :param tstart: only consider transitions happenening after $tstart seconds
    :param tend: only consider transitions up to $tend seconds
    :param ylim: list, specifying y-limits for y axis of DF/F plots, for example ylim=[0, 10] will limit the yrange
           from 0 to 10
    :param xticks: list, xticks for DF/F plots
    :param cb_ticks: ticks for colorbar
    :param vm: tuple, min and max value for colorbar of EEG spectrogram
    :param pzscore: if True, z-score DF/F values
    :param sf: float, smoothing factor
    :param base_int: float, duration of baseline interval (first $base_int seconds), for statistics to test,
           when the activity becomes significantly different from baseline. 
           Uses relative (paired) t-test. 
           All subsequent bins after basline, have the same width. 
           NOTE: To know which time points are significantly different from baseline,
           apply Bonferroni correction: If n time steps are compared with baseline,
           then divide the significance criterion (alphs = 0.05) by n.
    :param mouse_stats: if True, calculate statistics across single mice, otherwise
           perform statistics across single trials
    :param mouse_avg: if True, average across mice (and not individual trials)
    :param ci: string or int; if 'sem', plot shading showing the standard error of the mean
           If set to number x (e.g. x=95), plot the x% (e.g. 95%) confidence interval

    :return: trans_act:  dict: transitions --> np.array(mouse id x timepoint),
             trans_act_trials: dict: transitions --> np.array(all single transitions x timepoint)
             t: np.array, time axis for transitions.
             df:         pd.DataFrame: index - time intervals, columns - transitions,
                         reports all the p-values for comparison of baseline interval (first $base_int seconds) vs.
                         each consecutive interval of equal duration. 
             df_trans:   pd.DataFrame. All individual trials. the DataFrame 
                         has the following columns: ['mouse', 'time', 'dff', 'trans']
                         `trans` refers to the type of transition
    
    Todo: return further dataframe holding all single trials along with mouse identity
    """
    # corrected time step. The actual timestep is not exactly 2.5 (but slightly off)
    # This makes trouble when using pandas (groupby.mean) to average across mice for each time point.
    # With the "real" dt each time point from each trial is different.
    dtc = 2.5

    if type(recordings) != list:
        recordings = [recordings]

    if len(vm) > 0:
        cb_ticks=vm

    paths = dict()
    for rec in recordings:
        if os.path.isdir(os.path.join(ppath, rec)):
            paths[rec] = ppath
        else:
            paths[rec] = backup

    states = {1:'R', 2:'W', 3:'N'}
    mice = dict()
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())

    trans_act = dict()
    trans_spe = dict()
    trans_spm = dict()
    trans_act_trials = dict()
    trans_spe_trials = dict()
    data = []
    for (si,sj) in transitions:
        sid = states[si] + states[sj]
        # dict: transition type -> mouse -> DFF transitions
        trans_act[sid] = []
        trans_spe[sid] = []
        trans_spm[sid] = []
        trans_act_trials[sid] = []
        trans_spe_trials[sid] = []

    for (si,sj) in transitions:
        sid = states[si] + states[sj]
        act_mouse = {m:[] for m in mice}
        spe_mouse = {m:[] for m in mice}
        spm_mouse = {m:[] for m in mice}
        for rec in recordings:
            print(rec)
            idf = re.split('_', rec)[0]
            sr = sleepy.get_snr(ppath, rec)
            nbin = int(np.round(sr)*2.5)
            dt = (1.0/sr)*nbin
            ipre  = int(np.round(pre/dt))
            ipost = int(np.round(post/dt)) + 1
            # Why the +1?
            # Say dt=10 and pre and post = 30
            # then ipre and ipost = 3... at first thought
            # the preceding period starts with
            # -30 -20 -10 
            # by definitino I say that the post period starts with 0, so:
            # 0 10 20 30... but this are four states!
            # so because of the 0, ipost = 3+1
            
            # load DF/F
            ddir = os.path.join(paths[rec], rec)
            # NEW 10/15/2020
            if os.path.isfile(os.path.join(ddir, 'dffd.mat')):
                dff = so.loadmat(os.path.join(ddir, 'dffd.mat'), squeeze_me=True)['dffd']
            else:
                dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
                print('%s - saving dffd.mat' % rec)
                so.savemat(os.path.join(ddir, 'dffd.mat'), {'dffd':dff})
            
            # OLD
            #dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
            if sf > 0:
                dff = sleepy.smooth_data(dff, sf)

            if pzscore:
                dff = (dff-dff.mean())/dff.std()
            else:
                dff *= 100

            # load spectrogram and normalize
            P = so.loadmat(os.path.join(ddir, 'sp_%s.mat' % rec), squeeze_me=True)
            SP = P['SP']
            freq = P['freq']
            ifreq = np.where(freq <= fmax)[0]
            df = freq[1]-freq[0]

            sp_mean = SP.mean(axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)

            # load EMG
            imu = np.where((freq>=mu[0]) & (freq<=mu[1]))[0]
            MP = so.loadmat(os.path.join(ddir, 'msp_%s.mat' % rec), squeeze_me=True)['mSP']
            emg_ampl = np.sqrt(MP[imu,:].sum(axis=0)*df)

            # load brain state
            M, _ = sleepy.load_stateidx(paths[rec], rec)

            # set istart, iend
            istart = int(np.round(tstart/dt))
            if tend == -1:
                iend = len(M)
            else:
                iend = int(np.round(tend/dt))

            # some sleep state corrections
            M[np.where(M==4)]=3

            if ma_thr>0:
                seq = sleepy.get_sequences(np.where(M==2)[0])
                for s in seq:
                    if np.round(len(s)*dt) <= ma_thr:
                        if (s[0]>1) and (M[s[0] - 1] != 1):
                            M[s] = 3

            seq = sleepy.get_sequences(np.where(M==si)[0])
            for s in seq:
                # the last time point in state si; so ti+1 is the first time point in state sj
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

                    if ipre <= ti < len(M)-ipost and len(s)*dt >= si_threshold[si-1] and len(sj_idx)*dt >= sj_threshold[sj-1] and istart <= ti < iend:
                        act_si = dff[ti-ipre+1:ti+1]
                        act_sj = dff[ti+1:ti+ipost+1]
                        act = np.concatenate((act_si, act_sj))

                        spe_si = SP[ifreq,ti-ipre+1:ti+1]
                        spe_sj = SP[ifreq,ti+1:ti+ipost+1]

                        spe = np.concatenate((spe_si, spe_sj), axis=1)

                        spm_si = emg_ampl[ti-ipre+1:ti+1]
                        spm_sj = emg_ampl[ti+1:ti+ipost+1]
                        spm = np.concatenate((spm_si, spm_sj))

                        act_mouse[idf].append(act)
                        spe_mouse[idf].append(spe)
                        spm_mouse[idf].append(spm)


                        t = np.arange(-ipre*dtc, ipost*dtc - dtc + dtc / 2, dtc)
                        m = len(t)
                        data += zip([idf]*m, t, act, [sid]*m)

        trans_act[sid] = act_mouse
        trans_spe[sid] = spe_mouse
        trans_spm[sid] = spm_mouse

    df_trans = pd.DataFrame(data=data, columns=['mouse', 'time', 'dff', 'trans'])
    # generate matrices for each transition type holding all single trials in each row
    for tr in trans_act_trials:
        for mouse in trans_act[tr]:
            trans_act_trials[tr] += trans_act[tr][mouse] 
            trans_spe_trials[tr] += trans_spe[tr][mouse]
    
    for tr in trans_act_trials:
        trans_act_trials[tr] = np.vstack(trans_act_trials[tr])
        trans_spe_trials[tr] = np.array(trans_spe_trials[tr])

    for tr in trans_act: 
        for mouse in trans_act[tr]:
            # average for each transition tr and mouse over trials
            trans_act[tr][mouse] = np.array(trans_act[tr][mouse]).mean(axis=0)
            trans_spm[tr][mouse] = np.array(trans_spm[tr][mouse]).mean(axis=0)
            trans_spe[tr][mouse] = np.array(trans_spe[tr][mouse]).mean(axis=0)

    # let's get rid of mouse identity by replacing dict with np.arrays
    for tr in trans_act:
        trans_act[tr] = np.array(list(trans_act[tr].values()))
        trans_spm[tr] = np.array(list(trans_spm[tr].values()))
        trans_spe[tr] = np.array(list(trans_spe[tr].values()))

    # set variables helpful for plotting
    ntrans = len(trans_act)
    nmice = len(mice)
    nx = 1.0/ntrans
    dx = 0.2 * nx
    f = freq[ifreq]
    #t = np.arange(-ipre*dt+dt, ipost*dt + dt/2, dt)
    t = np.arange(-ipre*dt, ipost*dt-dt + dt/2, dt)
    
    tinit = -ipre*dt
    i = 0
    plt.ion()
    if len(transitions) > 2:
        plt.figure(figsize=(10, 5))
    else:
        plt.figure()
    for (si,sj) in transitions:
        tr = states[si] + states[sj]
        # plot DF/F
        ax = plt.axes([nx*i+dx, 0.15, nx-dx-dx/3.0, 0.3])
        if nmice == 1:
            # dimensions: number of mice x time
            if mouse_avg:
                plt.plot(t, trans_act[tr].mean(axis=0), color='blue')
            else:
                plt.plot(t, trans_act_trials[tr].mean(axis=0), color='blue')
        else:
            if mouse_avg:
                if ci == 'sem':
                    # mean is linear
                    tmp = trans_act[tr].mean(axis=0)
                    # std is not linear

                    sem = np.std(trans_act[tr],axis=0) / np.sqrt(nmice)
                    plt.plot(t, tmp, color='blue')
                    ax.fill_between(t, tmp - sem, tmp + sem, color=(0, 0, 1), alpha=0.5, edgecolor=None)
                else:
                    dfm = df_trans.groupby(['mouse', 'trans', 'time']).mean().reset_index()
                    # df_trans = pd.DataFrame(data=data, columns=['mouse', 'time', 'dff', 'trans'])
                    sns.lineplot(data=dfm, x='time', y='dff', color='blue')
                    
            else:
                # plot average across individual trials:
                if ci == 'sem':
                    # mean is linear
                    tmp = trans_act_trials[tr].mean(axis=0)
                    # std is not linear
                    sem = np.std(trans_act_trials[tr],axis=0) / np.sqrt(trans_act_trials[tr].shape[0])
                    plt.plot(t, tmp, color='blue')
                    ax.fill_between(t, tmp - sem, tmp + sem, color=(0, 0, 1), alpha=0.5, edgecolor=None)
                else:
                    sns.lineplot(data=df_trans, x='time', y='dff', color='blue')

        sleepy.box_off(ax)
        plt.xlabel('Time (s)')
        plt.xlim([t[0], t[-1]])
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
            axes_cbar = plt.axes([nx * i + dx+dx*2, 0.55+0.25+0.03, nx - dx-dx/3.0, 0.1])
        ax = plt.axes([nx * i + dx, 0.55, nx - dx-dx/3.0, 0.25])
        plt.title(states[si] + ' $\\rightarrow$ ' + states[sj])

        if mouse_avg:
            im = ax.pcolorfast(t, f, trans_spe[tr].mean(axis=0), cmap='jet')
        else:
            im = ax.pcolorfast(t, f, trans_spe_trials[tr].mean(axis=0), cmap='jet')

        if len(vm) > 0:
            im.set_clim(vm)

        ax.set_xticks([0])
        ax.set_xticklabels([])
        if i==0:
            plt.ylabel('Freq. (Hz)')
        if i>0:
            ax.set_yticklabels([])
        sleepy.box_off(ax)

        if i==0:
            # colorbar for EEG spectrogram
            cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0, orientation='horizontal')
            #cb = plt.colorbar(im, cax=ax, orientation="horizontal")
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

        i += 1
    plt.show()

    if len(fig_file) > 0:
        sleepy.save_figure(fig_file)

    # Statistics: When does activity becomes significantly different from baseline?
    ibin = int(np.round(base_int / dt))
    nbin = int(np.floor((pre+post)/base_int))
    data = []
    for tr in trans_act:
        if mouse_stats:
            trans = trans_act[tr]
        else:
            trans = trans_act_trials[tr]
        base = trans[:,0:ibin].mean(axis=1)
        for i in range(1,nbin):
            p = stats.ttest_rel(base, trans[:,i*ibin:(i+1)*ibin].mean(axis=1))
            sig = 'no'
            if p.pvalue < (0.05 / (nbin-1)):
                sig = 'yes'
            tpoint = i*(ibin*dt)+tinit + ibin*dt/2
            tpoint = float('%.2f'%tpoint)
            
            data.append([tpoint, p.pvalue, sig, tr])
    df_stats = pd.DataFrame(data = data, columns = ['time', 'p-value', 'sig', 'trans'])
    print(df_stats)

    return trans_act, trans_act_trials, t, df_stats, df_trans



def activity_transitions_ma(ppath, recordings, transitions, pre, post, si_threshold, sj_threshold,
                         backup='', mu=[10, 100], fmax=30, ma_thr=0, ylim=[], xticks=[], cb_ticks=[], vm=[],
                         tstart=0, tend=-1, pzscore=False, sf=0, base_int=10, mouse_stats=True, mouse_avg=True, fig_file=''):
    """
    calculate average DFF activity along brain state transitions
    :param ppath: base folder
    :param recordings: list of recordings
    :param transitions: list of tuples to denote transitions to be considered;
           1 - REM, 2 - Wake, 3 - NREM; For example to calculate NREM to REM and REM to wake transitions,
           type [(3,1), (1,2)]
    :param pre: time before transition in s
    :param post: time after transition
    :param si_threshold: list of floats, thresholds how long REM, Wake, NREM should be at least before the transition.
           So, if there's a REM to Wake transition, but the duration of REM is shorter then si_threshold[0], then this
           transition if discarded.
    :param sj_threshold: list of floats, thresholds how long REM, Wake, NREM should be at least after the transition
    :param backup: string, potential backup path
    :param mu: tuple, specifying the lower and upper limit of frequencies for EMG amplitude calculation
    :param fmax: maximum frequency shown for EEG spectrogram
    :param ma_thr: threshold in seconds for microarousals; i.e. wake periods < ma_thr are considered as microarousals
           and instead treated as NREM (if ma_polish == True)
    :param tstart: only consider transitions happenening after $tstart seconds
    :param tend: only consider transitions up to $tend seconds
    :param ylim: list, specifying y-limits for y axis of DF/F plots, for example ylim=[0, 10] will limit the yrange
           from 0 to 10
    :param xticks: list, xticks for DF/F plots
    :param cb_ticks: ticks for colorbar
    :param vm: tuple, min and max value for colorbar of EEG spectrogram
    :param pzscore: if True, z-score DF/F values
    :param sf: float, smoothing factor
    :param base_int: float, duration of baseline interval (first $base_int seconds), for statistics to test,
           when the activity becomes significantly different from baseline. 
           Uses relative (paired) t-test. 
           All subsequent bins after basline, have the same width. 
           NOTE: To know which time points are significantly different from baseline,
           apply Bonferroni correction: If n time steps are compared with baseline,
           then divide the significance criterion (alphs = 0.05) by n.
    :param mouse_stats: if True, calculate statistics across single mice, otherwise
           perform statistics across single trials

    :return: trans_act:  dict: transitions --> np.array(mouse id x timepoint),
             trans_act_trials: dict: transitions --> np.array(all single transitions x timepoint)
             t: np.array, time axis for transitions.
             df:         pd.DataFrame: index - time intervals, columns - transitions,
                         reports all the p-values for comparison of baseline interval (first $base_int seconds) vs.
                         each consecutive interval of equal duration. 
                         
    Todo: return further dataframe holding all single trials along with mouse identity
    """
    if type(recordings) != list:
        recordings = [recordings]

    if len(vm) > 0:
        cb_ticks=vm

    paths = dict()
    for rec in recordings:
        if os.path.isdir(os.path.join(ppath, rec)):
            paths[rec] = ppath
        else:
            paths[rec] = backup

    states = {1:'R', 2:'W', 3:'N', 4:'M'}
    mice = dict()
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())

    trans_act = dict()
    trans_spe = dict()
    trans_spm = dict()
    trans_act_trials = dict()
    trans_spe_trials = dict()
    for (si,sj) in transitions:
        sid = states[si] + states[sj]
        # dict: transition type -> mouse -> DFF transitions
        trans_act[sid] = []
        trans_spe[sid] = []
        trans_spm[sid] = []
        trans_act_trials[sid] = []
        trans_spe_trials[sid] = []

    for (si,sj) in transitions:
        sid = states[si] + states[sj]
        act_mouse = {m:[] for m in mice}
        spe_mouse = {m:[] for m in mice}
        spm_mouse = {m:[] for m in mice}
        for rec in recordings:
            print(rec)
            idf = re.split('_', rec)[0]
            sr = sleepy.get_snr(ppath, rec)
            nbin = int(np.round(sr)*2.5)
            dt = (1.0/sr)*nbin
            ipre  = int(np.round(pre/dt))
            ipost = int(np.round(post/dt)) + 1
            # Why the +1?
            # Say dt=10 and pre and post = 30
            # then ipre and ipost = 3... at first thought
            # the preceding period starts with
            # -30 -20 -10 
            # by definitino I say that the post period starts with 0, so:
            # 0 10 20 30... but this are four states!
            # so because of the 0, ipost = 3+1
            
            # load DF/F
            ddir = os.path.join(paths[rec], rec)
            # NEW 10/15/2020
            if os.path.isfile(os.path.join(ddir, 'dffd.mat')):
                dff = so.loadmat(os.path.join(ddir, 'dffd.mat'), squeeze_me=True)['dffd']
            else:
                dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
                print('%s - saving dffd.mat' % rec)
                so.savemat(os.path.join(ddir, 'dffd.mat'), {'dffd':dff})
            
            # OLD
            #dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
            if sf > 0:
                dff = sleepy.smooth_data(dff, sf)

            if pzscore:
                dff = (dff-dff.mean())/dff.std()
            else:
                dff *= 100

            # load spectrogram and normalize
            P = so.loadmat(os.path.join(ddir, 'sp_%s.mat' % rec), squeeze_me=True)
            SP = P['SP']
            freq = P['freq']
            ifreq = np.where(freq <= fmax)[0]
            df = freq[1]-freq[0]

            sp_mean = SP.mean(axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)

            # load EMG
            imu = np.where((freq>=mu[0]) & (freq<=mu[1]))[0]
            MP = so.loadmat(os.path.join(ddir, 'msp_%s.mat' % rec), squeeze_me=True)['mSP']
            emg_ampl = np.sqrt(MP[imu,:].sum(axis=0)*df)

            # load brain state
            M, _ = sleepy.load_stateidx(paths[rec], rec)

            # set istart, iend
            istart = int(np.round(tstart/dt))
            if tend == -1:
                iend = len(M)
            else:
                iend = int(np.round(tend/dt))

            # some sleep state corrections
            # M[np.where(M==4)]=3

            if ma_thr>0:
                seq = sleepy.get_sequences(np.where(M==2)[0])
                for s in seq:
                    if len(s)*dt < ma_thr:
                        if (s[0]>1) and (M[s[0] - 1] != 1):
                            M[s] = 4

            seq = sleepy.get_sequences(np.where(M==si)[0])
            for s in seq:
                # the last time point in state si; so ti+1 is the first time point in state sj
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

                    if ipre <= ti < len(M)-ipost and len(s)*dt >= si_threshold[si-1] and len(sj_idx)*dt >= sj_threshold[sj-1] and istart <= ti < iend:
                        act_si = dff[ti-ipre+1:ti+1]
                        act_sj = dff[ti+1:ti+ipost+1]
                        act = np.concatenate((act_si, act_sj))

                        spe_si = SP[ifreq,ti-ipre+1:ti+1]
                        spe_sj = SP[ifreq,ti+1:ti+ipost+1]

                        spe = np.concatenate((spe_si, spe_sj), axis=1)

                        spm_si = emg_ampl[ti-ipre+1:ti+1]
                        spm_sj = emg_ampl[ti+1:ti+ipost+1]
                        spm = np.concatenate((spm_si, spm_sj))

                        act_mouse[idf].append(act)
                        spe_mouse[idf].append(spe)
                        spm_mouse[idf].append(spm)

        trans_act[sid] = act_mouse
        trans_spe[sid] = spe_mouse
        trans_spm[sid] = spm_mouse

    # generate matrices for each transition type holding all single trials in each row
    for tr in trans_act_trials:
        for mouse in trans_act[tr]:
            trans_act_trials[tr] += trans_act[tr][mouse] 
            trans_spe_trials[tr] += trans_spe[tr][mouse]
    
    for tr in trans_act_trials:
        trans_act_trials[tr] = np.vstack(trans_act_trials[tr])
        trans_spe_trials[tr] = np.array(trans_spe_trials[tr])

    for tr in trans_act: 
        for mouse in trans_act[tr]:
            # average for each transition tr and mouse over trials
            trans_act[tr][mouse] = np.array(trans_act[tr][mouse]).mean(axis=0)
            trans_spm[tr][mouse] = np.array(trans_spm[tr][mouse]).mean(axis=0)
            trans_spe[tr][mouse] = np.array(trans_spe[tr][mouse]).mean(axis=0)

    # let's get rid of mouse identity by replacing dict with np.arrays
    for tr in trans_act:
        trans_act[tr] = np.array(list(trans_act[tr].values()))
        trans_spm[tr] = np.array(list(trans_spm[tr].values()))
        trans_spe[tr] = np.array(list(trans_spe[tr].values()))

    # set variables helpful for plotting
    ntrans = len(trans_act)
    nmice = len(mice)
    nx = 1.0/ntrans
    dx = 0.2 * nx
    f = freq[ifreq]
    #t = np.arange(-ipre*dt+dt, ipost*dt + dt/2, dt)
    t = np.arange(-ipre*dt, ipost*dt-dt + dt/2, dt)
    
    tinit = -ipre*dt
    i = 0
    plt.ion()
    if len(transitions) > 2:
        plt.figure(figsize=(10, 5))
    else:
        plt.figure()
    for (si,sj) in transitions:
        tr = states[si] + states[sj]
        # plot DF/F
        ax = plt.axes([nx*i+dx, 0.15, nx-dx-dx/3.0, 0.3])
        if nmice == 1:
            # dimensions: number of mice x time
            if mouse_avg:
                plt.plot(t, trans_act[tr].mean(axis=0), color='blue')
            else:
                plt.plot(t, trans_act_trials[tr].mean(axis=0), color='blue')
        else:
            # mean is linear
            tmp = trans_act[tr].mean(axis=0)
            # std is not linear
            sem = np.std(trans_act[tr],axis=0) / np.sqrt(nmice)
            plt.plot(t, tmp, color='blue')
            ax.fill_between(t, tmp - sem, tmp + sem, color=(0, 0, 1), alpha=0.5, edgecolor=None)
        sleepy.box_off(ax)
        plt.xlabel('Time (s)')
        plt.xlim([t[0], t[-1]])
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
            axes_cbar = plt.axes([nx * i + dx+dx*2, 0.55+0.25+0.03, nx - dx-dx/3.0, 0.1])
        ax = plt.axes([nx * i + dx, 0.55, nx - dx-dx/3.0, 0.25])
        plt.title(states[si] + ' $\\rightarrow$ ' + states[sj])

        # if statement does not make much sense here...
        #if nmice==1:
            # dimensions of trans_spe: number of mice x frequencies x time
            # so, to average over mice, average over first dimension (axis)
            #im = ax.pcolorfast(t, f, trans_spe[tr].mean(axis=0), cmap='jet')
        #else:
        
        if mouse_avg:
            im = ax.pcolorfast(t, f, trans_spe[tr].mean(axis=0), cmap='jet')
        else:
            im = ax.pcolorfast(t, f, trans_spe_trials[tr].mean(axis=0), cmap='jet')

        if len(vm) > 0:
            im.set_clim(vm)

        ax.set_xticks([0])
        ax.set_xticklabels([])
        if i==0:
            plt.ylabel('Freq. (Hz)')
        if i>0:
            ax.set_yticklabels([])
        sleepy.box_off(ax)

        if i==0:
            # colorbar for EEG spectrogram
            cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0, orientation='horizontal')
            #cb = plt.colorbar(im, cax=ax, orientation="horizontal")
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

        i += 1
    plt.show()

    if len(fig_file) > 0:
        sleepy.save_figure(fig_file)

    # Statistics: When does activity becomes significantly different from baseline?
    ibin = int(np.round(base_int / dt))
    nbin = int(np.floor((pre+post)/base_int))
    data = []
    for tr in trans_act:
        if mouse_stats:
            trans = trans_act[tr]
        else:
            trans = trans_act_trials[tr]
        base = trans[:,0:ibin].mean(axis=1)
        for i in range(1,nbin):
            p = stats.ttest_rel(base, trans[:,i*ibin:(i+1)*ibin].mean(axis=1))
            sig = 'no'
            if p.pvalue < (0.05 / (nbin-1)):
                sig = 'yes'
            tpoint = i*(ibin*dt)+tinit + ibin*dt/2
            tpoint = float('%.2f'%tpoint)
            
            data.append([tpoint, p.pvalue, sig, tr])
    df = pd.DataFrame(data = data, columns = ['time', 'p-value', 'sig', 'trans'])
    print(df)

    return trans_act, trans_act_trials, t, df




def downsample_mx(X, nbin):
    """
    y = downsample_vec(x, nbin)
    downsample the vector x by replacing nbin consecutive
    bin by their mean
    @RETURN: the downsampled vector
    """
    n_down = int(np.floor(X.shape[0] / nbin))
    X = X[0:n_down * nbin, :]
    X_down = np.zeros((n_down, X.shape[1]))

    # 0 1 2 | 3 4 5 | 6 7 8
    for i in range(nbin):
        idx = list(range(i, int(n_down * nbin), int(nbin)))
        X_down += X[idx, :]

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
        y = np.zeros((nelem * nbin,))
        for k in range(nbin):
            y[k::nbin] = x
    else:
        y = np.zeros((nelem * nbin, x.shape[1]))
        for k in range(nbin):
            y[k::nbin, :] = x

    return y



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
        Y = downsample_vec(A, int((m * nstates) / nstates))
    else:
        Y = downsample_mx(A, int((m * nstates) / nstates))
    # now we have m rows as requested
    return Y



def dff_stateseq(ppath, recordings, sequence, nstates, thres, sign=['>','>','>'], ma_thr=10, fmax=30, ylim=[], vm = [],
                         pzscore=False, sf=0, backup='', fig_file=''):
    """
    calculate average DF/F activity during three consecutive states sequences. For averaging, each state sequence
    is normalized in time.
    Example call,
    pyphi.dff_stateseq(ppath, name, [2,3,2], [5,10,5], [20, 60, 20], sign=['<', '>', '<'], fmax=60, ylim=[-0.5, 1])
    :param ppath: base folder
    :param recordings: list of recordings
    :param sequence: list of 3 integers, describing the 3 state sequence
    :param nstates: number of bins for each state
    :param thres: list of 3 floats, minimal or maximal duration for each of the 3 consecutive states
    :param sign: 3 element list, composed of either '<' or '>', if sign[i] == '<', then the duration of the i-th state
           should be shorter than thres[i]
    :param ma_thr: microarousal threshold
    :param fmax: maximal frequency shown in spectrogram
    :param ylim: tuple, y range for DF/F
    :param vm: tuple, color saturation for EEG spectrogram
    :param pzscore: boolean, if True, z-score activity
    :param sf: smoothing factor
    :param backup: optional backup folder
    :param fig_file: if specified save figure to $fig_file
    :return:
    """
    if type(recordings) != list:
        recordings = [recordings]

    paths = dict()
    for rec in recordings:
        if os.path.isdir(os.path.join(ppath, rec)):
            paths[rec] = ppath
        else:
            paths[rec] = backup

    states = {1:'REM', 2:'Wake', 3:'NREM'}
    mice = dict()
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice[idf] = 1
    mice = list(mice.keys())

    data = []
    ev = 0
    dff_mouse = {m: [] for m in mice}
    spe_mouse = {m: [] for m in mice}
    for rec in recordings:
        idf = re.split('_', rec)[0]

        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin

        # load DF/F
        ddir = os.path.join(paths[rec], rec)
        # load DF/F
        ddir = os.path.join(paths[rec], rec)
        if os.path.isfile(os.path.join(ddir, 'dffd.mat')):
            dff = so.loadmat(os.path.join(ddir, 'dffd.mat'), squeeze_me=True)['dffd']
        else:
            dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
            print('%s - saving dffd.mat' % rec)
            so.savemat(os.path.join(ddir, 'dffd.mat'), {'dffd':dff})

        #dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
        if sf > 0:
            dff = sleepy.smooth_data(dff, sf)

        if pzscore:
            dff = (dff - dff.mean()) / dff.std()
        else:
            dff *= 100

        # load spectrogram and normalize
        P = so.loadmat(os.path.join(ddir, 'sp_%s.mat' % rec), squeeze_me=True)
        SP = P['SP']
        freq = P['freq']
        ifreq = np.where(freq <= fmax)[0]
        #df = freq[1] - freq[0]

        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        SP = SP[ifreq,:]

        # load brain state
        M, _ = sleepy.load_stateidx(paths[rec], rec)
        lenm = len(M)

        #flatten out microarousals
        if ma_thr > 0:
            seq = sleepy.get_sequences(np.where(M == 2)[0])
            for s in seq:
                if len(s) * dt < ma_thr:
                    if (s[0] > 1) and (M[s[0] - 1] != 1):
                        M[s] = 3

        seqi = sleepy.get_sequences(np.where(M==sequence[0])[0])
        for x in seqi:
            if x[-1]+1 < lenm:
                if M[x[-1]+1] == sequence[1]:
                    i = x[-1]+1
                    while(i<lenm-1) and M[i]==sequence[1]:
                        i+=1
                    if M[i] == sequence[2]:
                        j = i
                        while (j<lenm-1) and M[j] == sequence[2]:
                            j+=1
                        idxi = x
                        idxj = list(range(x[-1]+1,i))
                        idxk = list(range(i,j))

                        pass_thresholds = True
                        if sign[0] == '<':
                            if len(idxi)*dt >= thres[0]:
                                pass_thresholds = False
                        else:
                            if len(idxi)*dt <= thres[0]:
                                pass_thresholds = False

                        if sign[1] == '<':
                            if len(idxj)*dt >= thres[1]:
                                pass_thresholds = False
                        else:
                            if len(idxj)*dt <= thres[1]:
                                pass_thresholds = False

                        if sign[2] == '<':
                            if len(idxk)*dt >= thres[2]:
                                pass_thresholds = False
                        else:
                            if len(idxk)*dt <= thres[2]:
                                pass_thresholds = False

                        if pass_thresholds:
                            dffi = time_morph(dff[idxi], nstates[0])
                            dffj = time_morph(dff[idxj], nstates[1])
                            dffk = time_morph(dff[idxk], nstates[2])

                            SPi = time_morph(SP[:, idxi].T, nstates[0]).T
                            SPj = time_morph(SP[:, idxj].T, nstates[1]).T
                            SPk = time_morph(SP[:, idxk].T, nstates[2]).T

                            dff_mouse[idf].append(np.concatenate((dffi, dffj, dffk)))
                            spe_mouse[idf].append(np.concatenate((SPi, SPj, SPk), axis=1))
                            ev += 1
                            
                            data += zip([idf]*sum(nstates), [ev]*sum(nstates), [ii for ii in range(sum(nstates))], list(np.concatenate((dffi, dffj, dffk))))
                            
    df = pd.DataFrame(data=data, columns=['mouse', 'event', 'section', 'dff'])

    seq_dff = {}
    seq_spe = {}
    #seq_dff_trials = []
    for mouse in mice:
        # average for each transition tr and mouse over trials
        seq_dff[mouse] = np.array(dff_mouse[mouse]).mean(axis=0)
        seq_spe[mouse] = np.array(spe_mouse[mouse]).mean(axis=0)
        #seq_dff_trials += dff_mouse[mouse]

    i = 0
    mx_dff = np.zeros((len(mice), sum(nstates)))
    mx_spe = np.zeros((len(mice), len(ifreq), sum(nstates)))
    for mouse in mice:
        mx_dff[i,:] = seq_dff[mouse]
        mx_spe[i,:,:] = seq_spe[mouse]
        i += 1

    # figure
    x = np.arange(1, sum(nstates)+1)
    cumx = np.concatenate(([1],np.cumsum(nstates)))
    plt.ion()
    plt.figure()
    # Spectrogram
    ax_spe = plt.axes([0.15, 0.6, 0.8, 0.35])
    im = ax_spe.pcolorfast(x, freq[ifreq], mx_spe.mean(axis=0), cmap='jet')
    ax_spe.set_xticks(cumx)
    ax_spe.set_xticklabels([])
    sleepy.box_off(ax_spe)
    plt.ylabel('Freq. (Hz)')
    plt.title(states[sequence[0]] + ' $\\rightarrow$ ' + states[sequence[1]]  + ' $\\rightarrow$ ' + states[sequence[2]])
    if len(vm) > 0:
        im.set_clim(vm)

    # DF/F
    nmice = len(mice)
    ax_dff = plt.axes([0.15, 0.15, 0.8, 0.35])
    ax_dff.plot(x, mx_dff.mean(axis=0), color='blue')
    if nmice > 1:
        tmp = mx_dff.mean(axis=0)
        sem = np.std(mx_dff, axis=0) / np.sqrt(nmice)
        ax_dff.fill_between(x, tmp - sem, tmp + sem, color=(0, 0, 1), alpha=0.5, edgecolor=None)
    ax_dff.set_xticks(cumx)
    ax_dff.set_xticklabels([])
    plt.xlim([1, sum(nstates)])
    sleepy.box_off(ax_dff)
    if not pzscore:
        plt.ylabel('$\Delta$ F / F (%)')
    else:
        plt.ylabel('$\Delta$ F / F (z scored)')
    if len(ylim) > 0:
        plt.ylim(ylim)
    plt.show()

    if len(fig_file) > 0:
        sleepy.save_figure(fig_file)

    return mx_dff, mx_spe, df



def irem_corr(ppath, recordings, pzscore=True, ma_thr=0, rem_break=0, min_irem=-1, sf=0):
    """
    

    Parameters
    ----------
    ppath : string
        base folder.
    recordings : []
        list of recordings.
    pzscore : TYPE, optional
        DESCRIPTION. The default is True.
    ma_thr : TYPE, optional
        DESCRIPTION. The default is 0.
    rem_break : TYPE, optional
        allow for interruptions in REM periods. That is,
        two REM periods that are separated by less than $rem_break seconds
        are fused to a single REM period
    min_irem: float, optional
        inter-REM intervals with duration < $min_irem are excluded from the analysis.
        The default is -1, i.e. every inter-REM interval is included.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    
    data = []
    for rec in recordings:
        ddir = os.path.join(ppath, rec)
        idf = re.split('_', rec)[0]
        print(idf)

        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr)*2.5)
        dt = (1.0/sr)*nbin

        if os.path.isfile(os.path.join(ddir, 'dffd.mat')):
            dff = so.loadmat(os.path.join(ddir, 'dffd.mat'), squeeze_me=True)['dffd']
        else:
            dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
            print('%s - saving dffd.mat' % rec)
            so.savemat(os.path.join(ddir, 'dffd.mat'), {'dffd':dff})

        if pzscore:
            dff = (dff - dff.mean()) / dff.std()
        else:
            dff = dff*100

        if sf > 0:
            dff = sleepy.smooth_data(dff, sf)


        M = sleepy.load_stateidx(ppath, rec)[0]
        if ma_thr > 0:
            seq = sleepy.get_sequences(np.where(M == 2)[0])
            for s in seq:
                if np.round(len(s)*dt) < ma_thr:
                    if (s[0] > 1) and (M[s[0] - 1] != 1):
                        M[s] = 3

        mmin = np.min((len(M), dff.shape[0]))
        M = M[0:mmin]
        dff = dff[0:mmin]

        seq = sleepy.get_sequences(np.where(M==1)[0], np.round(rem_break/dt).astype('int')+1)
        if len(seq) >= 2:
            for (si, sj) in zip(seq[:-1], seq[1:]):
                # indices of inter-REM period                                
                idx = range(si[-1]+1,sj[0])
                
                if min_irem > 0 and np.round(len(idx)*dt) < min_irem:
                    print('skipping inter-REM period')
                    continue

                m_cut = M[idx]                
                dff_cut = dff[idx]
                
                rem_pre = len(si)*dt
                rem_post = len(sj)*dt
                dur_irem = len(idx)*dt
                dur_inrem = len(np.where(m_cut==3)[0])*dt
                dur_iwake = len(np.where(m_cut==2)[0])*dt
                
                dff_irem = np.mean(dff_cut).mean()
                dff_inrem = dff_cut[np.where(m_cut==3)[0]].mean()
                
                dff_iwake = dff_cut[np.where(m_cut==2)[0]].mean()
                dff_pre = dff[si].mean()
                dff_post = dff[sj].mean()

                data.append([idf, rec, rem_pre, rem_post, dur_irem, dur_inrem, dur_iwake, dff_pre, dff_post, dff_irem, dff_inrem, dff_iwake])
        
    df = pd.DataFrame(data=data, columns=['mouse', 'recording', 'rem_pre', 'rem_post', 'dur_irem', 'dur_inrem', 'dur_iwake', 'dff_pre', 'dff_post', 'dff_irem', 'dff_inrem', 'dff_iwake'])
        
    return df



def dff_sleepcycle(ppath, recordings, backup='', nstates_rem=10, nstates_itrem=20,
                  pzscore=False, fmax=30, ma_thr=0, single_mode=False, sf=0, 
                  vm=[], cb_ticks=[], ylim=[], rem_break=0, fig_file=''):
    """
    plot average activity across time-normalized sleep cycle;
    a single sleep cycle starts and ends with a REM period
    :param ppath: base folder
    :param recordings: list or recordings
    :param backup: backup folder
    :param nstates_rem: number of bins for REM periods
    :param nstates_itrem: number of bins for inter-REM period
    :param pzscore: if True, z-score DF/F
    :param fmax: maximal frequency for EEG spectrogram
    :param ma_thr: thresholds for microarousals
    :param single_mode: if True, plot all units, otherwise just average with s.e.m
    :param sf: float, if > 0, smooth each firing rate with Gaussian filter
    :param vm: tuple, saturation of EEG heat map
    :param cb_ticks: ticks for colorbar (for EEG spectrogram)
    :param ylim: y range for DF/F axis
    :param fig_file: if specified save figure to $fig_file

    :return: np.arrays for DF/F for each mouse and EEG spectrogram for each mouse
    (mice x 2*nstates_rem + ntstates_itrem, mice x frequencies x 2*nstates_rem + ntstates_itrem)
    """

    if type(recordings) != list:
        recordings = [recordings]

    paths = dict()
    for rec in recordings:
        if os.path.isdir(os.path.join(ppath, rec)):
            paths[rec] = ppath
        else:
            paths[rec] = backup

    #states = {1:'REM', 2:'Wake', 3:'NREM'}
    mice = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice.append(idf)

    dff_cycle_mouse = {m:[] for m in mice}
    sp_cycle_mouse  = {m:[] for m in mice}
    #df_spec = pd.DataFrame()
    data = []
    section_labels = range(0, nstates_rem*2 + nstates_itrem)
    for rec in recordings:
        idf = re.split('_', rec)[0]

        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin

        # load DF/F
        ddir = os.path.join(paths[rec], rec)
        if os.path.isfile(os.path.join(ddir, 'dffd.mat')):
            dff = so.loadmat(os.path.join(ddir, 'dffd.mat'), squeeze_me=True)['dffd']
        else:
            dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
            print('%s - saving dffd.mat' % rec)
            so.savemat(os.path.join(ddir, 'dffd.mat'), {'dffd':dff})

        # load DF/F
        #ddir = os.path.join(paths[rec], rec)
        #dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
        if sf > 0:
            dff = sleepy.smooth_data(dff, sf)

        if pzscore:
            dff = (dff - dff.mean()) / dff.std()
        else:
            dff *= 100

        # load spectrogram and normalize
        P = so.loadmat(os.path.join(ddir, 'sp_%s.mat' % rec), squeeze_me=True)
        SP = P['SP']
        freq = P['freq']
        ifreq = np.where(freq <= fmax)[0]
        #df = freq[1] - freq[0]

        sp_mean = SP.mean(axis=1)
        SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
        SP = SP[ifreq, :]

        # load brain state
        M, _ = sleepy.load_stateidx(paths[rec], rec)

        # flatten out microarousals
        if ma_thr > 0:
            seq = sleepy.get_sequences(np.where(M == 2)[0])
            for s in seq:
                if len(s) * dt < ma_thr:
                    if (s[0] > 1) and (M[s[0] - 1] != 1):
                        M[s] = 3

        seq = sleepy.get_sequences(np.where(M == 1)[0], np.round(rem_break/dt).astype('int')+1)
        # We have at least 2 REM periods (i.e. one sleep cycle)
        if len(seq) >= 2:
            for (si, sj) in zip(seq[:-1], seq[1:]):
                # indices of inter-REM period
                idx = list(range(si[-1]+1, sj[0]))

                dff_pre = time_morph(dff[si], nstates_rem)
                dff_post = time_morph(dff[sj], nstates_rem)
                dff_itrem = time_morph(dff[idx], nstates_itrem)
                dff_cycle = np.concatenate((dff_pre, dff_itrem, dff_post))
                dff_cycle_mouse[idf].append(dff_cycle)

                m = dff_cycle.shape[0]
                data += zip([idf]*m, dff_cycle, section_labels)

                SP_pre = time_morph(SP[:, si].T, nstates_rem).T
                SP_post = time_morph(SP[:, sj].T, nstates_rem).T
                SP_itrem = time_morph(SP[:, idx].T, nstates_itrem).T
                sp_cycle_mouse[idf].append(np.concatenate((SP_pre, SP_itrem, SP_post), axis=1))

                #sp_cycle = np.concatenate((SP_pre, SP_itrem, SP_post), axis=1)
                #df = nparray2df(sp_cycle, freq['ifreq'], section_labels, 'pow', 'freq', 'dff')
                #df['mouse'] = idf
                #df_spec = df_spec.append(df)


    df_dff = pd.DataFrame(data=data, columns=['mouse', 'dff' ,'time'])
    ntime = 2*nstates_rem+nstates_itrem
    nfreq = ifreq.shape[0]
    nmice = len(mice)
    # average for each unit the firing rates
    for mouse in mice:
        sp_cycle_mouse[mouse]  = np.array( sp_cycle_mouse[mouse])
        dff_cycle_mouse[mouse] = np.array(dff_cycle_mouse[mouse])
    # transform dict to np.array
    DFFmx  = np.zeros((nmice, ntime))
    SPmx = np.zeros((nmice, nfreq, ntime))
    for (mouse, i) in zip(mice, list(range(nmice))):
        SPmx[i,:] = np.mean(sp_cycle_mouse[mouse], axis=0)
        DFFmx[i,:] = np.mean(dff_cycle_mouse[mouse], axis=0)

    # plotting
    plt.ion()
    plt.figure(figsize=(10, 6))
    # plot spectrogram
    # axes for colorbar
    axes_cbar = plt.axes([0.8, 0.85, 0.1, 0.2])
    if len(vm) > 0:
        cb_ticks = vm
    ax = plt.axes([0.15, 0.5, 0.75, 0.3])
    if len(vm) == 0:
        im = ax.pcolorfast(list(range(ntime)), freq[ifreq], np.nanmean(SPmx,axis=0), cmap='jet')
    else:
        im = ax.pcolorfast(list(range(ntime)), freq[ifreq], np.nanmean(SPmx,axis=0), cmap='jet', vmin=vm[0], vmax=vm[1])
    ax.set_xticks([nstates_rem, nstates_rem + nstates_itrem])
    plt.ylabel('Freq (Hz)')
    ax.set_xticklabels([])
    sleepy.box_off(ax)
    if len(vm) > 0:
        im.set_clim(vm)

    # plot firing rate
    ax = plt.axes([0.15, 0.1, 0.75, 0.3])
    if single_mode:
        plt.plot(list(range(ntime)), DFFmx.T)
    else:
        tmp = np.nanmean(DFFmx, axis=0)
        plt.plot(list(range(ntime)), tmp, color='blue')
        if nmice > 1:
            sem = np.nanstd(DFFmx, axis=0) / np.sqrt(nmice)
            ax.fill_between(list(range(ntime)), tmp - sem, tmp + sem, color=(0, 0, 1), alpha=0.5, edgecolor=None)
    sleepy.box_off(ax)
    ax.set_xticks([nstates_rem, nstates_rem + nstates_itrem])
    plt.xlim((0, ntime - 1))
    if len(ylim) > 1:
        plt.ylim(ylim)
    if pzscore:
        plt.ylabel('$\Delta$F/F (z scored)')
    else:
        plt.ylabel('$\Delta$F/F (%)')
    ax.set_xticklabels([])

    # colorbar for EEG spectrogram
    cb = plt.colorbar(im, ax=axes_cbar, pad=0.0, aspect=10.0, orientation='horizontal')
    cb.set_label('Rel. Power')
    cb.ax.xaxis.set_ticks_position("bottom")
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
    if len(fig_file) > 0:
        sleepy.save_figure(fig_file)

    return DFFmx, SPmx, df_dff #, df_spec



def dff_remrem_sections(ppath, recordings, backup='', nsections=5,
                       pzscore=False, ma_thr=10, sf=0, ylim=[], rem_break=0, min_irem=-1):
    """
    plot NREM and wake activity for $nsections consecutive sections 
    of the sleep cycle (interval between two consecutive REM periods)
    Example:
        df = pyphi.dff_remrem_sections(ppath, recordings, pzscore=True)
    
    :param ppath: base folder
    :param recordings: list of recordings
    :param backup: optional backup folder
    :param nsections: number of sections for the sleep cycle
    :param pzscore: if True, z-score firing rates
    :param ma_polish: if True, polish out micro-arousals
    :param ma_thr: threshold for microarousals
    :param sf: smoothing factor for firing rate
    :param ylim: typle, y-limits for y axis
    :param rem_break: float, two consecutive REM sleep periods that are interrupted
                      by <= $rem_break seconds are fused to one REM period
    :param min_len: float, minimum duration for inter-REM period; if inter-REM duration
                    is < min_irem than discard this interval; default = -1 (no limitation
                    for inter-REM duration)
    :return:
    """
    if type(recordings) != list:
        recordings = [recordings]

    paths = dict()
    for rec in recordings:
        if os.path.isdir(os.path.join(ppath, rec)):
            paths[rec] = ppath
        else:
            paths[rec] = backup

    mice = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice.append(idf)

    list_mx = []
    data = []
    ev = 0
    for rec in recordings:
        sections_wake = [[] for p in range(nsections)]
        sections_nrem = [[] for p in range(nsections)]

        idf = re.split('_', rec)[0]
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin

        # load DF/F
        ddir = os.path.join(paths[rec], rec)
        if os.path.isfile(os.path.join(ddir, 'dffd.mat')):
            dff = so.loadmat(os.path.join(ddir, 'dffd.mat'), squeeze_me=True)['dffd']
        else:
            dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
            print('%s - saving dffd.mat' % rec)
            so.savemat(os.path.join(ddir, 'dffd.mat'), {'dffd':dff})

        #dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
        if sf > 0:
            dff = sleepy.smooth_data(dff, sf)

        if pzscore:
            dff = (dff - dff.mean()) / dff.std()
        else:
            dff *= 100

        # load brain state
        M,_ = sleepy.load_stateidx(paths[rec], rec)

        # flatten out microarousals
        if ma_thr > 0:
            seq = sleepy.get_sequences(np.where(M == 2)[0])
            for s in seq:
                if np.round(len(s)*dt) < ma_thr:
                    if (s[0] > 1) and (M[s[0] - 1] != 1):
                        M[s] = 3

        seq = sleepy.get_sequences(np.where(M == 1)[0], np.round(rem_break/dt).astype('int')+1)
        # We have at least 2 REM periods (i.e. one sleep cycle)
        if len(seq) >= 2:
            for (si, sj) in zip(seq[:-1], seq[1:]):
                # indices of inter-REM period
                idx = range(si[-1]+1,sj[0])

                if min_irem > 0 and len(idx)*dt < min_irem:
                    print('skipping inter-REM period')
                    continue

                m = len(idx)
                M_up = upsample_mx(M[idx], nsections)
                M_up = np.round(M_up)
                dff_up = upsample_mx(dff[idx], nsections)

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
                    sections_wake[p].append(wake_dff)
                    single_event_wake.append(wake_dff)

                    idcut = np.intersect1d(mi, np.where(M_up == 3)[0])
                    if len(idcut) == 0:
                        nrem_dff = np.nan
                    else:
                        nrem_dff = np.nanmean(dff_up[idcut])
                    sections_nrem[p].append(nrem_dff)
                    single_event_nrem.append(nrem_dff)
                
                idur = len(idx)*dt
                data += zip([idf]*nsections, [ev]*nsections, list(range(1, nsections+1)), single_event_nrem, ['NREM']*nsections, [idur]*nsections)
                data += zip([idf]*nsections, [ev]*nsections, list(range(1, nsections+1)), single_event_wake, ['Wake']*nsections, [idur]*nsections)
                ev += 1
            

        list_mx += list(zip([np.nanmean(np.array(w)) for w in sections_wake], list(range(1, nsections+1)), ['Wake']*nsections, [idf]*nsections))
        list_mx += list(zip([np.nanmean(np.array(w)) for w in sections_nrem], list(range(1, nsections+1)), ['NREM']*nsections, [idf]*nsections))
        
    df_trials = pd.DataFrame(columns = ['mouse', 'event', 'section', 'dff', 'state', 'idur'], data=data)
    df = pd.DataFrame(columns = ['dff', 'section', 'state', 'idf'], data=list_mx)
    df = df.groupby(['idf', 'state', 'section'], as_index=False).mean()
    df_mean = df.groupby(['section', 'state'], as_index=False).mean()   

    # plot for each section DF/F for all mice, the average, and a linear regression line
    plt.figure()
    colors = np.array([[0, 1, 1], [0.5, 0, 1], [0.6, 0.6, 0.6]])

    plt.subplot(211)
    sns.lineplot(x='section', y='dff', data=df_mean[df_mean.state=='NREM'], color=colors[2,:])
    sns.regplot(x='section', y='dff', data=df[df.state=='NREM'], color='black')
    plt.xlabel('')
    plt.xticks(list(range(1, nsections+1)))
    if pzscore:
        plt.ylabel('$\mathrm{\Delta F / F}$ (z-scored)')
    else:
        plt.ylabel('$\mathrm{\Delta F / F}$ (%)')
    if len(ylim) > 0:
        plt.ylabel(ylim)
    sns.despine()
    
    plt.subplot(212)
    sns.lineplot(x='section', y='dff', data=df_mean[df_mean.state=='Wake'], color=colors[1,:])
    sns.regplot(x='section', y='dff', data=df[df.state=='Wake'], color='black')
    plt.xticks(list(range(1, nsections+1)))
    if pzscore:
        plt.ylabel('$\mathrm{\Delta F / F}$ (z-scored)')
    else:
        plt.ylabel('$\mathrm{\Delta F / F}$ (%)')
    if len(ylim) > 0:
        plt.ylabel(ylim)
    sns.despine()

    dfw = df[df.state=='Wake'] 
    dfn = df[df.state=='NREM'] 

    maskw = ~np.isnan(dfw.iloc[:,2]) & ~np.isnan(dfw.iloc[:,3])
    maskn = ~np.isnan(dfn.iloc[:,2]) & ~np.isnan(dfn.iloc[:,3])
    
    x = np.array(dfw.iloc[:,2])
    y = np.array(dfw.iloc[:,3])
    wres  = stats.linregress(x[maskw], y[maskw])

    x = np.array(dfn.iloc[:,2])
    y = np.array(dfn.iloc[:,3])
    nres  = stats.linregress(x[maskn], y[maskn])
    
    if wres.slope > 0:
        print("Wake firing rates increase throughout sleep cycle by a factor of %.2f; r=%.2f, p=%.2f" % (wres.slope, wres.rvalue, wres.pvalue))
    else:
        print("Wake firing rates decrease throughout sleep cycle by a factor of %.2f; r=%.2f, p=%.2f" % (wres.slope, wres.rvalue, wres.pvalue))

    if nres.slope > 0:
        print("NREM firing rates increase throughout sleep cycle by a factor of %.2f; r=%.2f, p=%.2f" % (nres.slope, nres.rvalue, nres.pvalue))
    else:
        print("NREM firing rates decrease throughout sleep cycle by a factor of %.2f; r=%.2f, p=%.2f" % (nres.slope, nres.rvalue, nres.pvalue))

    return df, df_trials



def dff_vs_statedur(ppath, recordings, istate, pzscore=False, sf=0, fs=1.5, ma_thr=0, backup=''):
    """
    Plot duration of single REM, Wake, or NREM periods vs DF/F
    Example call:
    pyphi.dff_vs_statedur(ppath, name, 3)
    plots duration of NREM periods vs DF/F
    :param ppath: base folder
    :param recordings: single string, or list of strings specifying recordings
    :param istate: int, 1 - REM, 2 - Wake, 3 - NREM
    :param pzscore: if True, z-score DF/F signal
    :param sf: float, smoothing factor
    :param fs: float, scaling of font, the larger the value, the larger the font
    :return: n/a
    """
    if type(recordings) != list:
        recordings = [recordings]

    paths = dict()
    for rec in recordings:
        if os.path.isdir(os.path.join(ppath, rec)):
            paths[rec] = ppath
        else:
            paths[rec] = backup

    dur = []
    act = []
    for rec in recordings:
        sr = sleepy.get_snr(ppath, rec)
        nbin = int(np.round(sr) * 2.5)
        dt = (1.0 / sr) * nbin

        # load DF/F
        ddir = os.path.join(paths[rec], rec)
        dff = so.loadmat(os.path.join(ddir, 'DFF.mat'), squeeze_me=True)['dffd']
        if sf > 0:
            dff = sleepy.smooth_data(dff, sf)

        if pzscore:
            dff = (dff - dff.mean()) / dff.std()
        else:
            dff *= 100

        M,_ = sleepy.load_stateidx(ppath, rec)
        if ma_thr > 0:
            seq = sleepy.get_sequences(np.where(M == 2)[0])
            for s in seq:
                if len(s) * dt < ma_thr:
                    if (s[0] > 1) and (M[s[0] - 1] != 1):
                        M[s] = 3

        seq = sleepy.get_sequences(np.where(M==istate)[0])

        for s in seq:
            d = len(s)*dt
            a = dff[s].mean()
            dur.append(d)
            act.append(a)

    dur = np.array(dur)
    act = np.array(act)

    sns.set(color_codes=True)
    sns.set(font_scale=fs)
    plt.figure()
    _,ax = plt.subplots()
    plt.ion()
    sns.regplot(dur, act)
    sleepy.box_off(ax)
    plt.xlabel('Duration (s)')
    plt.ylabel('$\mathrm{\Delta F / F}$')
    plt.show()



def dff_infraslow(ppath, recordings, ma_thr=10, min_dur = 160,
                  band=[10,15], state=3, win=100, pplot=True,
                  pnorm=False, spec_norm=True, spec_filt=False, box=[1,4],
                  pzscore=True, tstart=0, tend=-1, peeg2=False, dff_control=False):
    """
    calculate powerspectrum of EEG spectrogram to identify oscillations in sleep activity within different frequency bands;
    only contineous NREM periods are considered for infraslow calculation.
    
    Example call: 
        df = pyphi.dff_infraslow(ppath, recordings, win=128, min_dur=3*60, pplot=False, pnorm=True) 
    
    @PARAMETERS:
    ppath        -       base folder of recordings
    recordings   -       single recording name or list of recordings
    
    @OPTIONAL:
    ma_thr       -       microarousal threshold; wake periods <= $min_dur are transferred to NREM
    min_dur      -       minimal duration [s] of a NREM period
    band         -       frequency band used for calculation
    win          -       window (number of indices) for FFT calculation
    pplot        -       if True, plot window showing result
    pnorm        -       if True, normalize "infraslow" PSD (for each mouse) by its total mean power (like in Lecci et al. 2017)
    pnorm_spec   -       if True, normalize EEG spectrogram before calculating FFT for PSD (that we call infraslow)
    
    @RETURN:
    df           -       pd.DataFrame
    """
    min_dur = np.max([win*2.5, min_dur])
    
    if type(recordings) != list:
        recordings = [recordings]

    Spec = {}    
    DFF  = {}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        Spec[idf] = []
        DFF[idf] = []
        
    mice = list(Spec.keys())
    
    for rec in recordings:
        idf = re.split('_', rec)[0]

        # sampling rate and time bin for spectrogram
        SR = get_snr(ppath, rec)
        NBIN = int(np.round(2.5*SR))
        dt = NBIN * 1/SR

        dt = 2.5

        istart = int(np.round(tstart/dt))
        if tend > -1:
            iend   = int(np.round(tend/dt))

        # load sleep state
        M = sleepy.load_stateidx(ppath, rec)[0]
        if tend == -1:
            iend = M.shape[0]
        M = M[istart:iend]
        seq = sleepy.get_sequences(np.where(M==state)[0], np.round(ma_thr/dt)+1)
        seq = [list(range(s[0], s[-1]+1)) for s in seq]
        
        # load frequency band
        P = so.loadmat(os.path.join(ppath, rec,  'sp_' + rec + '.mat'))
        if not peeg2:
            SP = np.squeeze(P['SP'])[:,istart:iend]
        else:
            SP = np.squeeze(P['SP2'])[:, istart:iend]
        freq = np.squeeze(P['freq'])
        ifreq = np.where((freq>=band[0]) & (freq<=band[1]))[0]
        if spec_filt:
            filt = np.ones(box)
            filt = np.divide(filt, filt.sum())
            SP = scipy.signal.convolve2d(SP, filt, boundary='symm', mode='same')

        if spec_norm:
            sp_mean = SP[:, :].mean(axis=1)
            SP = np.divide(SP, np.tile(sp_mean, (SP.shape[1], 1)).T)
            pow_band = SP[ifreq,:].mean(axis=0)
        else:
            pow_band = SP[ifreq, :].sum(axis=0) * (freq[1]-freq[0])
            nidx = np.where(M==3)[0]
            pow_band = pow_band / pow_band[nidx].mean()

        if not dff_control:
            if os.path.isfile(os.path.join(ppath, rec, 'dffd.mat')):
                dffd = so.loadmat(os.path.join(ppath, rec, 'dffd.mat'), squeeze_me=True)['dffd']
            else:
                dffd = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dffd']
                so.savemat(os.path.join(ppath, rec, 'dffd.mat'), {'dffd': dffd})
        else:
            if os.path.isfile(os.path.join(ppath, rec, 'dffd_405.mat')):
                dffd = so.loadmat(os.path.join(ppath, rec, 'dffd_405.mat'), squeeze_me=True)['dffd']
            else:
                dff_405 = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['405']
                #pdb.set_trace()
                dffd = downsample_vec(dff_405, NBIN)
                print('%s - saving dffd_405.mat' % rec)
                so.savemat(os.path.join(ppath, rec, 'dffd_405.mat'), {'dffd': dffd})
        dffd = dffd[istart:iend]

        if pzscore:
            dffd = (dffd-dffd.mean()) / dffd.std()
        
        seq = [s for s in seq if len(s)*dt >= min_dur]   
        for s in seq:
            y,f = sleepy.power_spectrum(pow_band[s], win, dt)
            Spec[idf].append(y)
            
            y,f = sleepy.power_spectrum(dffd[s], win, dt)
            DFF[idf].append(y)
            
    # Transform %Spec to ndarray
    SpecMx = np.zeros((len(Spec), len(f)))
    DFFMx  = np.zeros((len(Spec), len(f)))
    
    data = []
    i=0
    for idf in Spec:
        SpecMx[i,:] = np.array(Spec[idf]).mean(axis=0)
        DFFMx[i,:]  = np.array(DFF[idf]).mean(axis=0)
        if pnorm==True:
            SpecMx[i,:] = SpecMx[i,:] / SpecMx[i,:].mean()#/LA.norm(SpecMx[i,:])
            DFFMx[i,:]  = DFFMx[i,:] / DFFMx[i,:].mean()#/LA.norm(DFFMx[i,:])
            
        data += zip([idf]*len(f), f, SpecMx[i,:], ['spec']*len(f))
        data += zip([idf]*len(f), f, DFFMx[i,:], ['dff']*len(f))
        
        i += 1

    if pplot:
        plt.figure()
        ax = plt.axes([0.1, 0.1, 0.8, 0.8])
        
        y = SpecMx[:,:]
        if len(mice) <= 1:
            ax.plot(f, y.mean(axis=0), color='gray', lw=2)
            
        else:
            ax.errorbar(f, y.mean(axis=0), yerr=y.std(axis=0), color='gray', fmt='-o')

        sleepy.box_off(ax)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (uV^2)')
        plt.show()

    df = pd.DataFrame(data=data, columns=['mouse', 'freq', 'pow', 'type'])

    return df



### CA EVENT DETECTION #################################################################################################
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
        lidx = np.where(data[0:-2] > data[1:-1])[0]
        ridx = np.where(data[1:-1] <= data[2:])[0]
        thidx = np.where(data[1:-1] < (-1 * th))[0]

        sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx)) + 1
    else:
        lidx = np.where(data[0:-2] < data[1:-1])[0]
        ridx = np.where(data[1:-1] >= data[2:])[0]
        thidx = np.where(data[1:-1] > th)[0]

        sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx)) + 1

    return sidx



def event_detection(ppath, name, nskip=5, tstart=0, tend=-1, nstd=2, pplot=True):

    # window size for waveforms
    win = 0.1

    sr = get_snr(ppath, name)
    dt = 1.0 / sr
    #nbin = int(np.round(sr) * 2.5)


    D = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)
    a405 = D['405']
    a465 = D['465']
    #ndown = D['dffd'].shape[0]
    #dff = calculate_dff(ppath, name, nskip, wcut=40, perc=perc)

    istart = int(tstart*sr)
    if tend>-1:
        iend = int(tend*sr)
    else:
        iend = len(a465)
    iwin = int(win/dt)

    w1 = 2  / (0.5*sr)
    w2 = 40 / (0.5*sr)
    a405 = sleepy.my_lpfilter(a405, w2, N=4)
    a465 = sleepy.my_lpfilter(a465, w2, N=4)

    nstart = int(np.round(nskip*sr))
    X = np.vstack([a405, np.ones(len(a405))]).T
    p = np.linalg.lstsq(X[nstart:,:], a465[nstart:])[0]
    a465_fit = np.dot(X, p)
    dff = np.divide((a465 - a465_fit), a465_fit)

    w1 = 1  / (0.5*sr)
    w2 = 40 / (0.5*sr)
    #dff_hp = dff - sleepy.my_lpfilter(dff, w1)
    dff_hp = sleepy.my_bpfilter(dff, w1, w2)
    dff_hp = dff_hp ** 2

    div = (dff_hp[1:] - dff_hp[0:-1]) / dt
    dff_div = (dff[1:] - dff[0:-1]) / dt
    s = np.max([nstart, istart])
    th = np.mean(div[s:]) + nstd * np.std(div[s:])

    idx = spike_threshold(div[istart:iend], th, sign=-1)+istart
    # only keep waveformed where the signal increases (i.e. d dff/ dt = div > 0)
    idx_sel = []
    for i in idx:
        if dff_div[i] > 0:
            idx_sel.append(i)
    idx_sel = np.array(idx_sel)

    train = np.zeros((len(dff),))
    #pdb.set_trace()
    train[idx_sel] = 1

    # cut out all transient waveforms
    Waves = []
    idx_max = []
    for i in idx_sel:
        if i >= iwin and i+iwin < len(dff):
            win = dff[i - iwin:i + iwin]
            Waves.append(win)
            j = np.argmax(win[iwin:])
            idx_max.append(i + j)
    idx_max = np.array(idx_max)
    Waves = np.array(Waves)

    # plot figure
    if pplot:
        plt.figure()
        plt.ion()
        plt.ion()

        A = downsample_vec(dff[istart:iend], 20)
        B = downsample_vec(div[istart:iend], 20)
        #idx_sel_dn = int((idx_sel-istart)/20)
        #pdb.set_trace()
        idx_max_dn = [int(i) for i in (idx_max-istart)/20]

        n = len(A)
        ta = np.arange(0, len(A)) * ((1.0/sr)*20)
        tb = np.arange(0, len(B)) * ((1.0/sr)*20)
        pdb.set_trace()
        plt.plot(ta, A)
        plt.plot(tb, B)
        plt.plot(ta, np.ones((n,))*th)
        plt.plot(ta[idx_max_dn], A[idx_max_dn], 'ro')

        plt.figure()
        t = np.arange(-iwin, iwin)*dt
        plt.plot(t, np.array(Waves).mean(axis=0))

    return train, Waves



def avg_events(ppath, recs, nstd=2, tstart=0, tend=-1):
    """
    plot number of calcium events dependent on brain state.
    Calcium events are calculated as described in Eban Rothschild et al., 2016
    :param ppath:
    :param recs: list of recordings
    :param nstd: defines threshold to be classified as "event"
    :param tstart: beginning of interval used for analysis
    :param tend: end of interval used for analysis
    :return:
    """
    import seaborn as sns

    mice = []
    for name in recs:
        idf = re.split('_', name)[0]
        if not idf in mice:
            mice.append(idf)

    brstate_mouse = {m:[] for m in mice}
    for name in recs:
        idf = re.split('_', name)[0]
        train = event_detection(ppath, name, nskip=5, tstart=tstart, tend=tend, nstd=nstd)[0]
        sr = get_snr(ppath, name)
        nbin = int(np.round(sr) * 2.5)
        dt = nbin * 1 / sr

        traind = downsample_vec(train*sr, nbin)
        istart = int(np.round(tstart / dt))
        if tend > -1:
            iend = int(np.round(tend / dt))
        else:
            iend = len(traind)

        M,_ = sleepy.load_stateidx(ppath, name)

        br_state = {s:[] for s in [1,2,3]}
        for s in [1,2,3]:
            idx = np.where(M==s)[0]
            idx = idx[np.where(idx<len(traind))[0]]
            idx = np.intersect1d(idx, np.arange(istart, iend+1))
            br_state[s] = traind[idx].mean()

        brstate_mouse[idf] = br_state

    brstate_mx = np.zeros((len(mice), 3))
    i = 0
    for m in mice:
        for s in [1,2,3]:
            brstate_mx[i,s-1] = brstate_mouse[m][s]
        i += 1

    df = pd.DataFrame(brstate_mx, columns=['REM', 'Wake', 'NREM'], index=mice)
    colors = np.array([[0, 1, 1], [0.5, 0, 1], [0.6, 0.6, 0.6]])
    plt.figure()
    sns.set_style('darkgrid')
    sns.barplot(data=df, palette=colors)
    sns.despine()
    plt.ylabel('Events/s')

    return df



def dff_spectrum(ppath, recordings, twin=30, tstart=0, tend=-1, ma_thr=20, pnorm=False, fmax=5, pzscore=True):
    """
    calculate power spectrum of DF/F signal dependent on brain state
    Example:
    pyphi.dff_spectrum(ppath, [name], twin=300, pnorm=False, pzscore=False, fmax=0.5)

    :param ppath: base folder
    :param recordings: single or list of recordings
    :param twin: time window for Fast Fourier Transform (FFT).
                 The larger twin, the finer the frequency axis. But if too large, REM sleep might be lost.
    :param tstart: starting time
    :param tend: end time
    :param ma_thr: microarousal threshold
    :param pnorm: if True, normalize spectrum by average power in each frequency band
    :param fmax: maximum frequency shown on frequency axis
    :param pzscore: if True, z-score DF/F signal
    :return:
    """

    if type(recordings) != list:
        recordings = [recordings]

    Mice = {}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in Mice:
            Mice[idf] = [rec]
        else:
            Mice[idf].append(rec)

    mouse_order = []
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mouse_order:
            mouse_order.append(idf)

    Spectra = {m:{1:[], 2:[], 3:[]} for m in mouse_order}
    for idf in mouse_order:
        for rec in Mice[idf]:
            DFF = so.loadmat(os.path.join(ppath, rec, 'DFF.mat'), squeeze_me=True)['dff']
            if pzscore:
                DFF = (DFF-DFF.mean()) / DFF.std()

            # load brain state for recording rec
            M,S = sleepy.load_stateidx(ppath, rec)
            sr = get_snr(ppath, rec)
            # number of time bins for each time bin in spectrogram
            nbin = int(np.round(sr) * 2.5)
            # duration of time bin in spectrogram / brainstate
            dt = nbin * 1/sr
            nwin = np.round(twin*sr)

            istart = int(np.round(tstart/dt))
            if tend==-1:
                iend = M.shape[0]
            else:
                iend = int(np.round(tend/dt))
            istart_eeg = istart*nbin
            iend_eeg   = (iend-1)*nbin+1

            M[np.where(M==5)]=2
            # flatten out microarousals
            seq = sleepy.get_sequences(np.where(M==2)[0])
            for s in seq:
                if len(s)*dt <= ma_thr:
                    M[s] = 3

            # get all sequences of state $istate
            M = M[istart:iend]
            DFF = DFF[istart_eeg:iend_eeg]

            #if pnorm:
            pow_norm = sleepy.power_spectrum(DFF, nwin, 1.0 / sr)[0]

            for istate in [1, 2, 3]:
                seq = sleepy.get_sequences(np.where(M==istate)[0])

                for s in seq:
                    b = np.min((s[-1]*nbin, len(DFF)))
                    sup = list(range(s[0] * nbin, b))

                    if len(sup) >= nwin:
                        p, f = sleepy.power_spectrum(DFF[sup], nwin, 1.0/sr)
                        if pnorm:
                            p = np.divide(p, pow_norm)
                        Spectra[idf][istate].append(p)

    Pow = {i:np.zeros((len(mouse_order), len(f))) for i in [1,2,3]}
    for istate in [1,2,3]:
        i = 0
        for m in mouse_order:
            Pow[istate][i,:] = np.array(Spectra[m][istate]).mean(axis=0)
            i += 1

    # figure
    plt.ion()
    plt.figure()
    ax = plt.subplot(111)
    colors = [[0, 1, 1], [0.5, 0, 1], [0.6, 0.6, 0.6]]
    state = ['REM', 'Wake', 'NREM']
    ifreq = np.where(f <= fmax)[0]
    for istate in [1,2,3]:
        plt.plot(f[ifreq], Pow[istate][:,ifreq].mean(axis=0), color=colors[istate-1], label=state[istate-1])
    if not pnorm:
        plt.plot(f[ifreq], pow_norm[ifreq], color='black', label='all')
    plt.legend()
    sleepy.box_off(ax)
    plt.xlabel('Freq. (Hz)')
    plt.ylabel('Power (a.u.)')



### SPECTRALFIELDS #####################################################################################################
def spectralfield_highres(ppath, name, pre, post, fmax = 60, theta=0,
                          states=[1,2,3], nsr_seg=2, perc_overlap=0.75, pzscore=False, pplot=True):
    """
    Calculate the "receptive field = spectral field" best mapping the EEG spectrogram onto the neural activity
    The spectrogram calculation is flexible, i.e. can be adjusted by the paramters nsr_seg and perc_overlap.

    :param ppath: base folder
    :param name: name
    :param pre: duration (in seconds) that the spectral fields fields into the past
    :param post: duration (in seconds) that the spectral fields extends into the future
    :param fmax: maximum frequency for spectrogram
    :param theta: float or list of floats; regularization parameter for ridge regression. If a list is provided the
           the function picks the parameters resulting in best performance on the test set.
    :param states: brain states used for calculation; 1=REM, 2=Wake, 3=NREM
    :param nsr_seg: float, defines the window length used for FFT calculation in seconds.
    :param perc_overlap: percentage that two consecutive FFT windows overlap
    :param pzscore: if True, z-score DF/F signal
    :param pplot: if True, plot spectral field
    :return: k, f, k; spectral field (np.array(frequency x time)), time, frequency vectors,
    """
    if not type(theta) == list:
        theta = [theta]

    sr = get_snr(ppath, name)
    nbin = np.round(2.5*sr)
    EEG = so.loadmat(os.path.join(ppath, name, 'EEG.mat'), squeeze_me=True)['EEG']
    # calculate "high resolution" EEG spectrogram
    freq, t, SP = scipy.signal.spectrogram(EEG, fs=sr, window='hanning', nperseg=int(nsr_seg * sr), noverlap=int(nsr_seg * sr * perc_overlap))
    # time is centered
    N = SP.shape[1]
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

    # load DF/F
    # time points per time bin in spectrogram:
    ndown = int(nsr_seg * sr) - int(nsr_seg * sr * perc_overlap)
    ninit = int(np.round(t[0]/dt))
    dff = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)['dff']*100
    dffd = downsample_vec(dff, ndown)
    dffd = dffd[ninit:]

    if pzscore:
        dffd = (dffd-dffd.mean()) / dffd.std()
    dffd = dffd[ipre:N-ipost]

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
    dffd = dffd[ibin-ipre]
    # mean zero response vector and columns of stimulus matrix
    rmean = dffd.mean()
    dffd = dffd - rmean
    mmean = MX.mean(axis=0)
    for i in range(MX.shape[1]):
        MX[:,i] -= mmean[i]

    # perform cross validation
    Etest, Etrain = cross_validation(MX, dffd, theta)
    print("CV results on training set:")
    print(Etrain)
    print("CV results on test set")
    print(Etest)

    # calculate kernel for optimal theta
    imax = np.argmax(Etest)
    print("Recording %s; optimal theta: %2.f" % (name, theta[imax]))
    k = ridge_regression(MX, dffd, theta[imax])
    k = np.reshape(k, ((ipre + ipost), nfreq)).T
    t = np.arange(-ipre, ipost) * dt

    if pplot:
        plt.ion()
        plt.figure()
        f = freq[ifreq]

        #dfk = sleepy.nparray2df(k, f, t, 'coeff', 'freq', 'time')  
        #dfk = dfk.pivot("freq", "time", "coeff") 
        #ax=sns.heatmap(dfk, cbar=False, cmap="jet") 
        #ax.invert_yaxis()        
        #plt.ylabel('Freq (Hz)')
        #plt.xlabel('Time (s)')

        plt.pcolormesh(t, f, k, cmap='bwr')
        plt.xlabel('Time (s)')
        plt.ylabel('Freq. (Hz)')
        plt.colorbar()
        plt.show()

    return k, t, freq[ifreq]



def spectralfield(ppath, name, pre, post, fmax=20, theta=0, states=[1,2,3], sp_norm=True, pzscore=False, pplot=True, pfilt=True):
    """
    Calculate the "receptive field = spectral field" best matching the EEG spectrogram onto the neural activity
    :param ppath: base folder
    :param name: name
    :param pre: no. of time bins into past
    :param post: no. of time bins into future
    :param fmax: maximum frequency for spectrogram
    :param theta: float or list of floats; regularization parameter for ridge regression. If a list is provided the
           the function picks the parameters resulting in best performance on the test set.
    :param states: states for which the spectralfield is calcualted
    :param sp_norm: if True, normalize spectrogram by dividing each frequency band by its mean
    :param pzscore: if True, z-score neural activity
    :return: k, f, k; spectral field (np.array(frequency x time)), time, frequency vectors,
    """

    if not type(theta) == list:
        theta = [theta]

    P = so.loadmat(os.path.join(ppath, name, 'sp_%s.mat' % name), squeeze_me=True)
    SP = P['SP']
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

    # load DF/F
    dffd = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)['dffd']*100
    if pzscore:
        dffd = (dffd-dffd.mean()) / dffd.std()
    dffd = dffd[pre:N-post]


    ibin = np.array([], dtype='int64')
    M,K = sleepy.load_stateidx(ppath, name)
    M = M[pre:N-post]
    for i in states:
        tmp = np.where(M == i)[0]
        ibin = np.concatenate((ibin, tmp))

    MX = MX[ibin,:]
    dffd = dffd[ibin]

    # mean zero response vector and columns of stimulus matrix
    rmean = dffd.mean()
    dffd = dffd - rmean
    mmean = MX.mean(axis=0)
    for i in range(MX.shape[1]):
        MX[:,i] -= mmean[i]

    # perform cross validation
    Etest, Etrain = cross_validation(MX, dffd, theta)
    print("CV results on training set:")
    print(Etrain)
    print("CV results on test set")
    print(Etest)

    # calculate kernel for optimal theta
    imax = np.argmax(Etest)
    print("Recording %s; optimal theta: %2.f" % (name, theta[imax]))
    k = ridge_regression(MX, dffd, theta[imax])
    k = np.reshape(k, ((pre + post), nfreq)).T

    t = np.arange(-pre, post) * fdt

    # calibration plot
    if pplot:
        plt.ion()
        plt.figure()
        ax = plt.subplot(111)
        plt.scatter(np.dot(MX,k)+rmean, dffd+rmean)
        plt.xlabel('Prediction $(\Delta F/F)$')
        plt.ylabel('Data $(\Delta F/F)$')
        sleepy.box_off(ax)
        plt.show()

        plt.figure()
        plt.plot(dffd+rmean)
        plt.plot(np.dot(MX,k)+rmean)

        plt.figure()
        plt.pcolormesh(t, freq[ifreq], k, cmap='bwr')
        plt.xlabel('Time (s)')
        plt.ylabel('Freq. (Hz)')
        plt.colorbar()
        plt.show()

    return k, t, freq[ifreq]



def spectralfield_mice(ppath, recordings, pre, post, fmax=20, theta=0, states=[1,2,3], sp_norm=True, pzscore=False):
    """
    Calculate the "receptive field = spectral field" best matching the EEG spectrogram onto the neural activity
    :param ppath: base folder
    :param recordings: list of recording names
    :param pre: no. of time bins into past
    :param post: no. of time bins into future
    :param fmax: maximum frequency for spectrogram
    :param theta: float or list of floats; regularization parameter for ridge regression. If a list is provided the
           the function picks the parameters resulting in best performance on the test set.
    :param states: states for which the spectralfield is calcualted
    :param sp_norm: if True, normalize spectrogram by dividing each frequency band by its mean
    :param pzscore: if True, z-score neural activity
    :return:
           Specfields, t, f: np.array(frequency x time x mice), time vector, frequency vector
    """


    mice = []
    # get all the mice from where the reocrdings comes from
    for rec in recordings:
        idf = re.split('_', rec)[0]
        if not idf in mice:
            mice.append(idf)

    spec_fields = {k:[] for k in mice}
    for rec in recordings:
        idf = re.split('_', rec)[0]
        k,t,f = spectralfield(ppath, rec, pre, post, fmax=fmax, theta=theta, states=states, sp_norm=sp_norm, pzscore=pzscore, pplot=False)
        # collect all spectralfields k of mouse idf:
        spec_fields[idf].append(k)

    SpecFields = np.zeros((len(f), len(t), len(mice)))
    i = 0
    for m in mice:
        SpecFields[:,:,i] = np.array(spec_fields[m]).mean(axis=0)
        i += 1

    plt.ion()
    plt.figure()
    plt.pcolormesh(t, f, SpecFields.mean(axis=2), cmap='bwr')
    plt.xlabel('Time (s)')
    plt.ylabel('Freq. (Hz)')
    plt.colorbar()
    plt.show()

    return SpecFields, t, f



def cross_validation(S, r, theta, nfold=5):
    """
    perform $nfold crossvalidation for linear regression with power constraint (ridge regression)
    :param S: "stimulus" n x m matrix; rows are samples/trials/timepoints, columns are the variables (e.g. frequencies)
    :param r: "response" vector with length n;
    :param theta: regularization parameter for power constraint
    :param nfold: number of subsets that the dataset is divided into
    :return:
           Etest, Etrain: np.arrays(($nfold,)) holding the r2 value for performance on test and training set for each
           cross-validation iteration
    """

    ntheta = len(theta)

    ninp = r.shape[0] # number of rows of the feature matrix
    nsub = round(ninp / nfold) # size of CV subsets
    nparam = S.shape[1]

    # calculate the column mean of the feature matrix
    # mmean = np.mean(S)
    # mean zero S
    for i in range(nparam):
        S[:,i] -= S[:,i].mean()

    test_idx = []
    train_idx = []
    for i in range(nfold):
        idx = np.arange(i*nsub, min(((i+1)*nsub, ninp)), dtype='int')
        idy = np.setdiff1d(np.arange(0, ninp), idx)
        test_idx.append(idx)
        train_idx.append(idy)

    l = 0
    Etest = np.zeros((ntheta,))
    Etrain = np.zeros((ntheta,))
    for th in theta:
        ptest = np.zeros((nfold,))
        ptrain = np.zeros((nfold,))
        j = 0
        for (p, q) in zip(test_idx, train_idx):
            k = ridge_regression(S[q,:], r[q], th)
            pred_test  = np.dot(S[p,:], k)
            pred_train = np.dot(S[q,:], k)

            #pred_mean = pred_test.mean()
            #pred_test[pred_test >= pred_mean] = 1
            #pred_test[pred_test <  pred_mean] = 0
            #pdb.set_trace()

            rtest  = r[p]
            rtrain = r[q]

            ptest[j] = 1 - np.var(pred_test - rtest) / np.var(rtest)
            ptrain[j] = 1 - np.var(pred_train - rtrain) / np.var(rtrain)
            j += 1

        Etest[l] = np.mean(ptest)
        Etrain[l] = np.mean(ptrain)
        l += 1

    return Etest, Etrain



def build_featmx(MX, pre, post):
    """
    :param MX: np.array with dimenstions features (=rows) x time/samples (=columns)
    
    
    For row i (=time point i) in the feature matrix, we take a 2D slice S of
    matrix MX: S = MX[:,i-pre:i+post] and reshape S to a single vector 
    by concatenating the columns of S;
    The transformation from matrix to vector is done by
    S_vec = np.reshape(S.T, (nrows*(pre+post),))
    """
    
    # number of time points
    N = MX.shape[1]
    # number of frequencies
    nfreq = MX.shape[0]

    j = 0
    A = np.zeros((N - pre - post, nfreq * (pre + post)))
    for i in range(pre, N - post-1):
        C = MX[:, i - pre:i + post]

        # A = np.array([[1,2,3],[4,5,6]])
        # then np.reshape(A.T, (6,)) yields
        # array([1, 4, 2, 5, 3, 6])
        A[j,:] = np.reshape(C.T, (nfreq * (pre + post),))

        # the reshaping looks as follows:
        # A = [1 2 3; 1 2 3]
        # A_reshape = [1 1 2 2 3 3]

        j += 1

    return A



def ridge_regression(A, r, theta):
    """
    S = A
    
    r = S * k
    S' * r = (S'*S + I*theta) * k
    SR = (S'*S + I*theta) * k
    k = CC \ SR
    :return k: optimal linear soluation k that best approximates S*k = r
    """
    # copy matrix, because otherwise it's overwritten
    S = A.copy()
    AC = np.dot(S.T, S)
    n = S.shape[1]
    AC = AC + np.eye(n)*theta

    # k = AC \ SR
    SR = np.dot(S.T, r)
    k = scipy.linalg.solve(AC, SR)

    return k


