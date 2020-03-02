#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 22:19:57 2018

@author: tortugar
"""
import os
import matplotlib.pylab as plt
import scipy.ndimage
import numpy as np
from matplotlib.widgets import Button
import re
import scipy.io as so
import zipfile
import cv2
import sleepy
import subprocess
import spyke
import matplotlib.patches as patches
import seaborn as sns
import pdb


class Frame:
    """
    this class goes together with function select_frame
    """
    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig

        self.nx = []
        self.ny = []
    
    def get_rectangle(self, event):
        self.nx = self.ax.get_xlim()
        self.ny = self.ax.get_ylim()
        self.nx = list(map(int, [self.nx[0], self.nx[1]]))
        self.ny = list(map(int, [self.ny[0], self.ny[1]]))
        self.nx.sort()
        self.ny.sort()



def select_frame(ppath, name):

    img = cv2.imread(os.path.join(ppath, name, 'Stack', 'fig2.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    frame = Frame(ax, fig)
    plt.subplots_adjust(bottom=0.2) 
    ax.imshow(img, cmap='gray')

    axbutton = plt.axes([0.81, 0.05, 0.1, 0.075])
    bok = Button(axbutton, 'Ok')
    bok.on_clicked(frame.get_rectangle)
    plt.show()
    
    return frame.nx, frame.ny



def crop_frames(ppath, name):
    """
    presents the first image in ppath/name/Stack/ and then let's you crop the image, the same frame is applied
    to all other pictures
    :param ppath: base folder
    :param name: recording
    :return: number of frames
    """
    if not os.path.isdir(os.path.join(ppath, name, 'Stack')):
        unpack_zipstack(ppath, name)
    print("select mouse cage with zoom tool and then click ok and close window!")
    print("program execution continues once the figure is closed")
    nx, ny = select_frame(ppath, name)
    # number of video frames
    vfile = os.path.join(ppath, name, 'video_timing.mat')
    vfid = so.loadmat(vfile, squeeze_me=True)
    nframes = len(vfid['onset'])

    print('Starting frame cropping')
    for i in range(nframes):
        fig = os.path.join(ppath, name, 'Stack', 'fig%d.jpg' % (i+1))
        img = cv2.imread(fig)
        crop_img = img[ny[0]:ny[1]+1, nx[0]:nx[1],:]
        #cv2.imshow("cropped", crop_img)
        #cv2.waitKey(1)
        cv2.imwrite(fig, crop_img)
        if i%1000 == 0:
            print("Done with Frame %d out of %d Frames" % (i, nframes))
    return nframes



def create_behann_file(ppath, name, idf, symbols, t='tick', comment=""):
    """
    :param ppath: recording base folder
    :param name: recording name
    :param idf: idf of behavior file, which will be saved as vip_"idf".txt
    :param symbols: dict; keys are the keys associated with a behavior as specified by the corr. value
    :param t: time step of annotation; 
    if t is a float, create an equidistant time axis with increments of duration $t
    if t is a string, the following options exist:
        t == 'tick': time axis is given by the TDT ticks, which are triggered every s
        t == 'fdt': time axis is given by the EEG spectrogram, also used for brain state classification
    :param comment: comment printed in first line of annoation file
    :return:
    """
    # get duration of experiment
    vfile = os.path.join(ppath, name, 'video_timing.mat')
    vfid = so.loadmat(vfile, squeeze_me=True)
    ticks = np.squeeze(vfid['tick_onset'])
    duration = ticks[-1]
    if type(t) == float:
        tscale = np.arange(0, duration, t)
    elif type(t) == str:
        if t == 'tick':
            tscale = ticks
        elif t == 'fdt':
            tscale = so.loadmat(os.path.join(ppath, name, 'sp_' + name + '.mat'), squeeze_me=True)['t']
        
    bfile =  'vip_%s.txt' % idf
    fid = open(os.path.join(ppath, name, bfile), 'w')
    fid.write('#' + comment + os.linesep)
    fid.write("@symbols: ")
    for s in list(symbols.keys())[0:-1]:
        fid.write("%s-%s, " % (s, symbols[s]))
    # write last symbol w/o ','
    s = list(symbols.keys())[-1]
    fid.write("%s-%s" % (s, symbols[s]))
    fid.write(os.linesep)
    fid.write(('@t: %s'+os.linesep) % str(t))

    for tp in tscale:
        #fid.write(('%.3f\t'+os.linesep) % tp)
        fid.write(('%.3f\t\t0' + os.linesep) % tp)
    fid.close()



def load_behann_file(bfile):
    """
    :param bfile: behavior annotation file, called vip_.*\.txt
    :return: dict, with keys as annotated time points and a string of length one specifying the behavior
    """
    fid = open(bfile, newline=None)
    lines = fid.readlines()

    annotation = {}
    K = []
    for line in lines:
        if re.match('^#.*', line):
            continue
        if re.match('^@.*', line):
            continue
        if re.match('^[\d\.]+\t.*', line):
            line.rstrip()
            #a = re.findall('([\d\.]+)\t(.*)', line)[0]
            a = re.findall('([\d\.]+)\t(.*)\t(\d)', line)[0]
            annotation[float(a[0])] = a[1]
            K.append(int(a[2]))

    return annotation, np.array(K)



def save_behann_file(ann_file, symbols, tstep, annotation, K):
    """
    save behavioral annotation to file
    :param ann_file: string, annotation file (absolute path)
    :symbols: dict, defining symbol strings for behaviors
    :tstep: float, time stept of behavior annodation
    :param annotation: dict, time point (float) --> behavior, represented as single letter string
    """
    fid = open(ann_file, 'w')
    
    fid.write("@symbols: ")
    for s in list(symbols.keys())[0:-1]:
        fid.write("%s-%s, " % (s, symbols[s]))
    # write last symbol w/o ','
    s = list(symbols.keys())[-1]
    fid.write("%s-%s" % (s, symbols[s]))
    fid.write(os.linesep)
    fid.write(('@t: %s'+os.linesep) % str(tstep))
    
    # to make sure time points are numerically sorted
    time = np.array(list(annotation.keys()))
    time.sort()
    i = 0
    for t in time:
        fid.write(('%.3f\t%s\t%d'+os.linesep) % (t,annotation[t],K[i]))
        i += 1



def load_config(config):
    """
    load configuration file, which is named vid_NAME.txt

    @Return:
        dict of symbos, time step, behavior annotation file
    """
    fid = open(config, newline=None)

    lines = fid.readlines()
    fid.close()
    t = 1.0
    symbols = {}
    for line in lines:
        if re.match('^\s+$', line):
            continue
        if re.match('^#', line):
            continue
        if re.match('^@symbols:', line):
            a = re.split(':', line)[1]
            a = re.sub('\s', '', a)
            pairs = re.split(',', a)
            for p in pairs:                
                a = re.split('-', p)
                symbols[a[0]] = a[1]
        if re.match('^@t:', line):
            a = re.split(':', line)[1]
            a = re.sub('\s', '', a)
            t = a

    return symbols, t



def intan_video_timing(ppath, rec, tscale=0):
    """
    creates .mat file video_timing.mat which contains
    two variables:
    onset       -     the onset of each video frame as decoded from the cameras strobing signal
    tick_onset  -     defines a time scale along which behaviors/frames are annotated
    tscale      -     if tscale==0, the annotation time scale is the same as the timing
                      of the video frames; otherwise specific in s.
    """
    sr = sleepy.get_snr(ppath, rec)
    vfile = os.path.join(ppath, rec, 'videotime_' + rec + '.mat')
    vid = so.loadmat(vfile, squeeze_me=True)['video']

    # transform signal such that a movie frame onset
    # corresponds to a flip from 0 to 1
    vid = (vid - 1.0)*-1.0
    vid = vid.astype('int')
    len_rec = vid.shape[0]

    idxs,_ = sleepy.laser_start_end(vid, SR=sr, intval=0.01)
    onset = idxs * (1.0/sr)
    tend = int(len_rec / sr)
    if tscale == 0:
        tick_onset = onset.copy()
    else:
        tick_onset = np.arange(0, tend, tscale)
    so.savemat(os.path.join(ppath, rec, 'video_timing.mat'), {'onset':onset, 'tick_onset':tick_onset})



def intan_correct_videotiming(ppath, rec, fr=1, pplot=True):
    """
    set annotation time scale for videos from intan.
    The function assumes that the following two files exist (which are generated by data_processing.py):
    timestamp_"rec".mat and videotime_"rec".mat
    :param ppath: base folder
    :param rec: recording name
    :param fr: frame rate of time scale used for annotation; Say, you want to annotated each 0.5s timestep,
               then set fr = 2
    :param pplot: if True, plot distance between frames; for debugging purposes
    :return: onset, np.array; array with time points of each frame onset


    NOTE: 
    Only use if you have the following line in sleepRecording.py:
        self.str_timestamp = str(self.img_timestamp.seconds) + '.' + '%06d' % (self.img_timestamp.microSeconds) + '\r\n'
    """

    # get timestamp .mat file
    timestamp = 'timestamp_' + rec + '.mat'

    sr = sleepy.get_snr(ppath, rec)
    dt = 1.0/sr
    vfile = os.path.join(ppath, rec, 'videotime_' + rec + '.mat')
    vid = so.loadmat(vfile, squeeze_me=True)['video']

    # transform signal such that is movie frame onset
    # corresponds to a flip from 0 to 1
    vid = (vid - 1.0)*-1.0
    vid = vid.astype('int')

    
    idxs,_ = sleepy.laser_start_end(vid, SR=sr, intval=0.01)
    print('Number of frames according to strobe signal: %d' % len(idxs))
    A = so.loadmat(os.path.join(ppath, rec, timestamp), squeeze_me=True)
    tstamp = A['timelist']
    tstamp[np.where(tstamp<0)[0]] += 1000000
    cam_sr = np.mean(np.diff(idxs)*(1.0/sr))

    # list for each videoframe the corresponding index in idxs
    ndropped = 0
    idx_dropped = []
    onset_idx = [0]
    curr_onset_idx = 0
    d = np.diff(tstamp)
    for i in range(0, len(tstamp)-1):
        # d[0] is the distance between frame 1 and frame 0
        if d[i] > 1.5*cam_sr:
#            if d[i] + d[i+1] < 0.0: #before 0.5
#                # everything is fine
#                curr_onset_idx += 1
#                onset_idx.append(curr_onset_idx)
#            else:
#                n = np.round(d[i] / cam_sr)
#                curr_onset_idx += int(n)
#                ndropped += int(n)-1
#                onset_idx.append(curr_onset_idx)
#                idx_dropped.append(i)

            n = np.round(d[i] / cam_sr)
            curr_onset_idx += int(n)
            ndropped += int(n)-1
            onset_idx.append(curr_onset_idx)
            idx_dropped.append(i)
        else:
            curr_onset_idx += 1
            onset_idx.append(curr_onset_idx)

    print("Number of dropped frames: %d" % ndropped)
    print('Number of timestamps: %d' % len(tstamp) )

    onset_idx = np.array(onset_idx)
    idx = np.where(onset_idx < len(idxs))[0]
    onset = np.array(idxs[onset_idx[idx]]) * dt

    step = int(np.round(1.0/fr))
    tick_onset = idxs[0::step]*dt
    so.savemat(os.path.join(ppath, rec, 'video_timing.mat'), {'onset':onset, 'tick_onset':tick_onset})

    if pplot:
        plt.ion()
        plt.figure()
        plt.plot(d)
        plt.plot(idx_dropped, d[idx_dropped], '.')
        plt.ylabel('$\mathrm{|Frame_{i+1} - Frame_i|}$')
        plt.xlabel('Frame no.')
        sns.despine()

    if os.path.isdir(os.path.join(ppath, rec, 'Stack')):
        files = os.listdir(os.path.join(ppath, rec, 'Stack'))
        filesm = [f for f in files if re.match('fig\d+\.jpg', f)]
        print('Number of frames in videostack: %d' % len(filesm))
        
    return onset, idx_dropped



def intan_correct_videotiming_old(ppath, rec, fr=1, pplot=True):
    """
    set annotation time scale for videos from intan.
    The function assumes that the following two files exist (which are generated by data_processing.py):
    timestamp_"rec".mat and videotime_"rec".mat
    :param ppath: base folder
    :param rec: recording name
    :param fr: frame rate of time scale used for annotation; Say, you want to annotated each 0.5s timestep,
               then set fr = 2
    :param pplot: if True, plot distance between frames; for debugging purposes
    :return: onset, np.array; array with time points of each frame onset
    """

    # get timestamp .mat file
    timestamp = 'timestamp_' + rec + '.mat'

    sr = sleepy.get_snr(ppath, rec)
    dt = 1.0/sr
    vfile = os.path.join(ppath, rec, 'videotime_' + rec + '.mat')
    vid = so.loadmat(vfile, squeeze_me=True)['video']

    # transform signal such that is movie frame onset
    # corresponds to a flip from 0 to 1
    vid = (vid - 1.0)*-1.0
    vid = vid.astype('int')

    idxs,_ = sleepy.laser_start_end(vid, SR=sr, intval=0.01)
    print('Number of frames according to strobe signal: %d' % len(idxs))
    A = so.loadmat(os.path.join(ppath, rec, timestamp), squeeze_me=True)
    tstamp = A['timelist']
    tstamp[np.where(tstamp<0)[0]] += 1000000
    cam_sr = np.mean(np.diff(idxs)*(1.0/sr))

    # list for each videoframe the corresponding index in idxs
    ndropped = 0
    idx_dropped = []
    onset_idx = [0]
    curr_onset_idx = 0
    d = np.diff(tstamp)
    for i in range(0, len(tstamp)-1):
        # d[0] is the distance between frame 1 and frame 0
        if d[i] > 0.3:
            if d[i] + d[i+1] < 0.5: #before 0.5
                # everything is fine
                curr_onset_idx += 1
                onset_idx.append(curr_onset_idx)
            else:
                n = np.round(d[i] / cam_sr)
                curr_onset_idx += int(n)
                ndropped += int(n)-1
                onset_idx.append(curr_onset_idx)
                idx_dropped.append(i)
        else:
            curr_onset_idx += 1
            onset_idx.append(curr_onset_idx)

    print("n = %d frames were dropped" % ndropped)

    onset_idx = np.array(onset_idx)
    idx = np.where(onset_idx < len(idxs))[0]
    onset = np.array(idxs[onset_idx[idx]]) * dt

    step = int(np.round(1.0/fr))
    tick_onset = idxs[0::step]*dt
    so.savemat(os.path.join(ppath, rec, 'video_timing.mat'), {'onset':onset, 'tick_onset':tick_onset})

    if pplot:
        plt.ion()
        plt.figure()
        plt.plot(d)
        plt.plot(idx_dropped, d[idx_dropped], '.')

    return onset





def intan_correct_videotiming2(ppath, rec, fr=1, pplot=True):
    """
    set annotation time scale for videos from intan.
    alternative version to intan_correct_videotiming_old
    The function assumes that the following two files exist (which are generated by data_processing.py):
    timestamp_"rec".mat and videotime_"rec".mat

    :param ppath: base folder
    :param rec: recording name
    :param fr: frame rate of time scale used for annotation; Say, you want to annotated each 0.5s timestep,
               then set fr = 2
    :param pplot: if True, plot distance between frames; for debugging purposes
    :return: onset, np.array; array with time points of each frame onset
    """
    import pdb

    # get timestamp .mat file
    timestamp = 'timestamp_' + rec + '.mat'

    sr = sleepy.get_snr(ppath, rec)
    dt = 1.0/sr
    vfile = os.path.join(ppath, rec, 'videotime_' + rec + '.mat')
    vid = so.loadmat(vfile, squeeze_me=True)['video']

    # transform signal such that is movie frame onset
    # corresponds to a flip from 0 to 1
    vid = (vid - 1.0)*-1.0
    vid = vid.astype('int')

    pdb.set_trace()
    idxs,_ = sleepy.laser_start_end(vid, SR=sr, intval=0.01)
    print('number of frames according to strobe signal: %d' % len(idxs))
    A = so.loadmat(os.path.join(ppath, rec, timestamp), squeeze_me=True)
    tstamp = A['timelist']
    tstamp[np.where(tstamp<0)[0]] += 1000000

    # list for each videoframe the corresponding index in idxs
    ndropped = 0
    onset_idx = [0]
    curr_onset_idx = 0
    d = np.diff(tstamp)
    idx_dropped = []
    cam_sr = np.mean(np.diff(idxs)*(1.0/sr))
    #pdb.set_trace()
    lag = 0
    lags = []
    for i in range(0, len(tstamp)-1):
        # d[0] is the distance between frame 1 and frame 0
        if d[i] > 1000:
            if d[i] + d[i+1] < (2*cam_sr-lag):
                # everything is fine
                curr_onset_idx += 1
                onset_idx.append(curr_onset_idx)
                lag += d[i] - cam_sr

            else:
                print(d[i])
                n = np.floor((d[i]-lag) / cam_sr)
                curr_onset_idx += int(n)
                ndropped += int(n)-1
                onset_idx.append(curr_onset_idx)
                idx_dropped.append(i)
                lag += d[i] - n*cam_sr
        else:
            lag += d[i] - cam_sr
            lags.append(lag)
            if lag < cam_sr:
                curr_onset_idx += 1
                onset_idx.append(curr_onset_idx)
            else: 
                while  lag > cam_sr:
                    curr_onset_idx += 1
                    ndropped += 1
                    #onset_idx.append(curr_onset_idx)
                    idx_dropped.append(i)
                    lag -= cam_sr
            
        print(lag)

    pdb.set_trace()
    print("n = %d frames were dropped" % ndropped)

    onset_idx = np.array(onset_idx)
    idx = np.where(onset_idx < len(idxs))[0]
    onset = np.array(idxs[onset_idx[idx]]) * dt
    step = int(np.round(1.0/fr))
    tick_onset = idxs[0::step]*dt
    so.savemat(os.path.join(ppath, rec, 'video_timing.mat'), {'onset':onset, 'tick_onset':tick_onset})

    if pplot:
        plt.ion()
        plt.figure()
        plt.plot(d)
        plt.plot(idx_dropped, d[idx_dropped], '.')

    return onset, idxs



def tdt_video_timing(ppath, rec, dt=1):
    """
    set the timing for videos recorded by TDT Synapse
    """
    vfile = os.path.join(ppath, rec, 'video_timing' + '.mat')
    vid = so.loadmat(vfile, squeeze_me=True)
    onset  = vid['onset']
    offset = vid['offset']

    len_rec = onset[-1]
    tend = int(len_rec)
    tick_onset = np.arange(0, tend+dt, dt)
    so.savemat(os.path.join(ppath, rec, 'video_timing.mat'), {'onset':onset, 'offset':offset, 'tick_onset':tick_onset})



def extract_stack(video_file, stack_dir, ffmpeg_path='ffmpeg', togray=False):
    """
    ffmpeg -i 180501_101813_F17F18_vid.mkv -qscale:v 10  Stack2/fig%d.jpg
    :param video_file: movie file (absolute path)
    :param stack_dir: directory where single movie frames should be saved (the "stack")
    :param ffmpeg_path: under windows specify the complete path to ffmpeg executable;
    """
    if togray:
        s = ffmpeg_path + ' -i ' + video_file + ' -qscale:v 10 ' + ' -vf format=gray,format=yuv420p ' + os.path.join(stack_dir, 'fig%d.jpg')
    else:
        s = ffmpeg_path + ' -i ' + video_file + ' -qscale:v 10 ' +  os.path.join(stack_dir, 'fig%d.jpg')


    if not os.path.isdir(stack_dir):
        os.mkdir(stack_dir)
    subprocess.call(s, stdout=subprocess.PIPE, shell=True)

    files = os.listdir(os.path.join(stack_dir))
    files = [f for f in files if re.match('^fig\d+\.jpg$', f)]
    nframes = len(files)

    # to slow:
    #if togray:
    #    print "converting to gray scale"
    #    files = os.listdir(os.path.join(stack_dir))
    #    files = [f for f in files if re.match('^fig', f)]
    #    nframes = len(files)
    #    i=0
    #    for f in files:
    #        i+=1
    #        print i
    #        im = cv2.imread(os.path.join(stack_dir, f))
    #        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    #        cv2.imwrite(os.path.join(stack_dir, f), im)

    return nframes



def encode_video(ppath, name, fr=5, stack='Stack', ffmpeg_path='ffmpeg', ending='.jpg', outpath='.', vidname='video_'):
    """
    encode the the figure stack in $ppath/$name/$stack to a video using ffmpeg
    """
    s = ffmpeg_path + ' -r ' + str(fr) + ' -i ' + os.path.join(ppath, name, stack, 'fig%d'+ending) + ' -acodec libmp3lame -crf 28 ' + os.path.join(outpath, vidname + name + '.mov')
    subprocess.call(s, stdout=subprocess.PIPE, shell=True)



def extract_cropped_stack(video_file, stack_dir, crop_dim='up', ffmpeg_path='ffmpeg'):
    """
    extract each single movie frames from $videofile and save it to the folder $stack_dir.
    $crop_dim ('up' or 'down') specifies whether the upper or lower half of each movie frame is cropped.
    """
    s = ffmpeg_path + ' -i ' + video_file + ' -qscale:v 10 ' + ' -ss 00:00:00 -vframes 1' + ' ' + os.path.join(stack_dir, 'fig1.jpg')
    if not os.path.isdir(stack_dir):
        os.mkdir(stack_dir)
    subprocess.call(s, stdout=subprocess.PIPE, shell=True)
    im = cv2.imread(os.path.join(stack_dir, 'fig1.jpg'))
    nx = im.shape[0] #height
    ny = im.shape[1] #width

    nxh = int(nx/2)
    if crop_dim == 'up':
        # width, height, top-left-corner-width, top-left-corner-height
        crop = ' -filter:v crop=%d:%d:%d:%d ' % (ny,nxh,0,0)
    elif crop_dim == 'down':
        crop = ' -filter:v crop=%d:%d:%d:%d ' % (ny, nxh, 0, nxh)


    s = ffmpeg_path + ' -i ' + video_file + ' -qscale:v 10 ' +  crop + \
        os.path.join(stack_dir,'fig%d.jpg')
    subprocess.call(s, stdout=subprocess.PIPE, shell=True)



def unpack_zipstack(ppath, name):
    """
    unzip the video frame .zip folder
    :param ppath: recording base folder
    :param name: recording
    """
    zip_ref = zipfile.ZipFile(os.path.join(ppath, name, 'stack.zip'), 'r')
    zip_ref.extractall(os.path.join(ppath, name))



def zipdir(path, zip_file):
    """
    zip directory $path to zip file $zip_file
    Note: only the parent directory of path is preserved (i.e.
    if the folder is /A/B/C, all files and dirs within C are zipped and only
    directly C is preserved in the zipped file
    """
    print("zipping folder %s to %s" % (path, zip_file))
    # ziph is zipfile handle
    ziph = zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED, allowZip64 = True)

    rootdir = os.path.basename(path)

    for root, dirs, files in os.walk(path):
        for filename in files:
            # Write the file named filename to the archive,
            # giving it the archive name 'arcname'.
            filepath = os.path.join(path, filename)
            parentpath = os.path.relpath(filepath, path)
            arcname = os.path.join(rootdir, parentpath)

            ziph.write(filepath, arcname)
    ziph.close()



def fibpho_video(ppath, name, ts, te, fmax=20, emg_legend=1000, vm=2.0, time_legend=10, dff_legend=10, ffmpeg_path='ffmpeg'):
    """
    Generate video for fiber photometry recording.
    The function requires that ffmpeg is installed on your system (http://ffmpeg.org).
    Windows Users: Specify the full path to the ffmpeg program

    The resulting video has 1 Hz resolution and will be saved in folder $ppath/$name
    :param ppath: base folder
    :param name:
    :param ts: start time in seconds
    :param te: end time in second
    :param fmax: maximum frequency on EEG spectrogram
    :param emg_legend: EMG legend in microVolts!
    :param vm: controls saturation of EEG spectrogram; a value in the range from 1 to 2 should work best
    :param time_legend: time legend in seconds
    :param dff_legend: DF/F in %
    :param ffmpeg_path: full, absolute path to ffmpeg program; important for to set in Windows
    :return: n/a
    """

    # helper function ######################
    def closest_neighbor(vec, x):
        d = np.abs(vec-x)
        el = np.min(d)
        idx = np.argmin(d)
        return el, idx
    ########################################

    # setup figure arrangement
    sleepy.set_fontsize(12)
    plt.ion()
    plt.figure()
    plt.figure(figsize=(8, 6))

    ax_video = plt.axes([0.1, 0.55, 0.8, 0.43])
    ax_eeg   = plt.axes([0.1, 0.38, 0.8, 0.15])
    ax_emg   = plt.axes([0.1, 0.25, 0.8, 0.1])
    ax_bs    = plt.axes([0.1, 0.22, 0.8, 0.02])
    ax_bs_legend = plt.axes([0.1, 0.24, 0.2, 0.02])
    ax_dff   = plt.axes([0.1, 0.05, 0.8, 0.15])
    ax_dff_legend = plt.axes([0.05, 0.05, 0.05, 0.15])
    ax_time  = plt.axes([0.1, 0.001, 0.8, 0.03])

    movie_stack = os.path.join(ppath, name, 'MStack')
    if not(os.path.isdir(movie_stack)):
        os.mkdir(movie_stack)

    sr = sleepy.get_snr(ppath, name)
    M = sleepy.load_stateidx(ppath, name)[0]
    dt = 1.0 / sr

    nbins = int(np.round(sr) * 5.0 / 2)
    Mup = spyke.upsample_mx(M, nbins)

    EEG = so.loadmat(os.path.join(ppath, name, 'EEG.mat'), squeeze_me=True)['EEG']
    EMG = so.loadmat(os.path.join(ppath, name, 'EMG.mat'), squeeze_me=True)['EMG']
    vid_time = so.loadmat(os.path.join(ppath, name, 'video_timing.mat'), squeeze_me=True)['onset']
    len_eeg = EEG.shape[0]

    t = np.arange(0, len_eeg)*dt
    its = closest_neighbor(t, ts)[1]
    ite = closest_neighbor(t, te)[1]
    data_eeg = EEG[its:ite]
    states = sleepy.downsample_states(Mup[its:ite], int(np.round(sr)))
    state_map = [[0, 1, 1], [0.5, 0, 1], [0.6, 0.6, 0.6]]

    # load and resample DF/F
    dff = so.loadmat(os.path.join(ppath, name, 'DFF.mat'), squeeze_me=True)['dff']*100
    dff = spyke.downsample_vec(dff[its:ite], int(np.round(sr)))
    dff_max = np.max(dff)
    dff_max = dff_max + 0.2*dff_max
    dff_min = np.min(dff)
    dff_min = dff_min - 0.1*dff_min

    # setup axis for video
    ax_video.get_xaxis().set_visible(False)
    ax_video.get_yaxis().set_visible(False)
    ax_video.spines["top"].set_visible(False)
    ax_video.spines["right"].set_visible(False)
    ax_video.spines["bottom"].set_visible(False)
    ax_video.spines["left"].set_visible(False)

    # setup axes for EEG spectrogram
    sleepy.box_off(ax_eeg)
    ax_eeg.set_xticks([])
    plt.gcf().text(0.11, 0.49, 'EEG', color='white')
    plt.gcf().text(0.11, 0.18, '$\mathrm{\Delta F/F}$', color='blue')


    # setup axes for EMG
    ax_emg.get_xaxis().set_visible(False)
    ax_emg.get_yaxis().set_visible(False)
    ax_emg.spines["top"].set_visible(False)
    ax_emg.spines["right"].set_visible(False)
    ax_emg.spines["bottom"].set_visible(False)
    ax_emg.spines["left"].set_visible(False)
    emg_max = np.max(np.abs(EMG[its:ite]))
    emg_max = emg_max + emg_max*0.1
    plt.gcf().text(0.905, 0.31,  "%.1f mV" % (emg_legend/1000.0), rotation=90)
    plt.gcf().text(0.11, 0.35, 'EMG')

    ax_dff.get_xaxis().set_visible(False)
    ax_dff.get_yaxis().set_visible(False)
    ax_dff.spines["top"].set_visible(False)
    ax_dff.spines["right"].set_visible(False)
    ax_dff.spines["bottom"].set_visible(False)
    ax_dff.spines["left"].set_visible(False)

    ax_bs.get_xaxis().set_visible(False)
    ax_bs.get_yaxis().set_visible(False)
    ax_bs.spines["top"].set_visible(False)
    ax_bs.spines["right"].set_visible(False)
    ax_bs.spines["bottom"].set_visible(False)
    ax_bs.spines["left"].set_visible(False)

    # calculate spectrogram
    fspec, tspec, Sxx = scipy.signal.spectrogram(data_eeg, fs=sr, nperseg=int(2*np.round(sr)), noverlap=int(np.round(sr)))
    ifreq = np.where(fspec<=fmax)[0]
    med = np.median(Sxx.max(axis=0))

    # setup time legend (beolow DF/F panel)
    ax_time.plot((tspec[0], tspec[0]+time_legend), [1, 1], color='black', linewidth=2)
    ax_time.set_xlim((tspec[0], tspec[-1]))
    ax_time.set_ylim((-1,1.1))
    ax_time.get_xaxis().set_visible(False)
    ax_time.get_yaxis().set_visible(False)
    ax_time.spines["top"].set_visible(False)
    ax_time.spines["right"].set_visible(False)
    ax_time.spines["bottom"].set_visible(False)
    ax_time.spines["left"].set_visible(False)
    ax_time.text(tspec[0], -1, '%s s' % str(time_legend))

    # setup legend for DF/F
    ax_dff_legend.set_xlim((tspec[0], tspec[-1]))
    ax_dff_legend.set_ylim((-1,1))
    ax_dff_legend.get_xaxis().set_visible(False)
    ax_dff_legend.get_yaxis().set_visible(False)
    ax_dff_legend.spines["top"].set_visible(False)
    ax_dff_legend.spines["right"].set_visible(False)
    ax_dff_legend.spines["bottom"].set_visible(False)
    ax_dff_legend.spines["left"].set_visible(False)
    ax_dff_legend.set_ylim((dff_min, dff_max))
    ax_dff_legend.set_xlim((0, 1))
    ax_dff_legend.plot((0.5,0.5), (dff_min, dff_min+dff_legend), color='black', linewidth=2)
    ax_dff_legend.text(0, dff_min+dff_legend/2.0, str(dff_legend)+'%', rotation=90)

    # legend for brain states
    ax_bs_legend.set_ylim((0, 2))
    ax_bs_legend.set_xlim((0, 10))
    ax_bs_legend.text(0.5, 0.5, 'REM', color=state_map[0])
    ax_bs_legend.text(3.3, 0.5, 'NREM', color=state_map[2])
    ax_bs_legend.text(7, 0.5, 'Wake', color=state_map[1])
    ax_bs_legend.get_xaxis().set_visible(False)
    ax_bs_legend.get_yaxis().set_visible(False)
    ax_bs_legend.spines["top"].set_visible(False)
    ax_bs_legend.spines["right"].set_visible(False)
    ax_bs_legend.spines["bottom"].set_visible(False)
    ax_bs_legend.spines["left"].set_visible(False)

    tstart = t[its]
    for i in range(2,len(tspec)):
        curr_t = tstart + tspec[i]
        ax_eeg.cla()
        ax_eeg.pcolor(tspec[:i], fspec[ifreq], Sxx[ifreq,:i], vmin=0, vmax=med*vm)
        ax_eeg.set_xlim((tspec[0], tspec[-1]))
        ax_eeg.set_xticks([])
        ax_eeg.set_ylabel('Freq. (Hz)')

        # displary current movie frame
        ax_video.cla()
        iframe = closest_neighbor(vid_time, curr_t)[1]
        img = cv2.imread(os.path.join(ppath, name, 'Stack', 'fig%d.jpg' % (iframe+1)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_video.imshow(cv2.transpose(img))

        # show EMG
        emg = EMG[its:closest_neighbor(t, curr_t)[1]]
        ax_emg.cla()
        temg = np.arange(0, len(emg))*dt
        ax_emg.plot(temg, emg, color='black', linewidth=0.5)
        ax_emg.set_xlim((tspec[0], tspec[-1]))
        ax_emg.set_ylim((-emg_max, emg_max))
        # plot EMG legend
        ax_emg.plot(([tspec[-1]-1, tspec[-1]-1]), (-emg_legend/2.0, emg_legend/2.0), color='black', linewidth=2)

        # plot brain state patches
        j = i-1
        ax_bs.add_patch(patches.Rectangle((tspec[j], 0), tspec[j+1]-tspec[j], 1, facecolor=state_map[int(states[j])-1], edgecolor=state_map[int(states[j])-1]))
        ax_bs.set_xlim((tspec[0], tspec[-1]))
        ax_bs.set_ylim((0,1))

        ax_dff.cla()
        ax_dff.plot(tspec[:i], dff[:i], color='blue')
        ax_dff.set_xlim((tspec[0], tspec[-1]))
        ax_dff.set_ylim((dff_min, dff_max))

        plt.savefig(os.path.join(movie_stack, 'fig%d.png' % i))

    encode_video(ppath, name, stack='MStack', ending='.png', fr=10, outpath=os.path.join(ppath, name), ffmpeg_path=ffmpeg_path, vidname='movie_fibpho_')



def opto_video(ppath, name, ts, te, fmax=20, emg_legend=1000, vm=2.0, time_legend=10, ffmpeg_path='ffmpeg'):
    """
    Generate video for optogenetic sleep recording.
    The function requires that ffmpeg is installed on your system (http://ffmpeg.org).
    Windows Users: Specify the full path to the ffmpeg program

    The resulting video has 1 Hz resolution and will be saved in folder $ppath/$name
    :param ppath: base folder
    :param name: name of recording
    :param ts: start time in seconds
    :param te: end time in second
    :param fmax: maximum frequency on EEG spectrogram
    :param emg_legend: EMG legend in micro Volts
    :param vm: controls saturation of EEG spectrogram; a value in the range from 1 to 2 should work best
    :param time_legend: time legend in seconds
    :param ffmpeg_path: full, absolute path to ffmpeg program; important for to set in Windows
    :return: n/a
    """
    # helper function ######################
    def closest_neighbor(vec, x):
        d = np.abs(vec-x)
        el = np.min(d)
        idx = np.argmin(d)
        return el, idx
    ########################################

    # setup figure arrangement
    sleepy.set_fontsize(12)
    plt.ion()
    plt.figure()
    plt.figure(figsize=(8, 6))

    ax_video = plt.axes([0.1, 0.52, 0.8, 0.45])
    ax_laser = plt.axes([0.1, 0.45, 0.8, 0.03])
    ax_eeg   = plt.axes([0.1, 0.28, 0.8, 0.15])
    ax_emg   = plt.axes([0.1, 0.11, 0.8, 0.15])
    ax_bs    = plt.axes([0.1, 0.05, 0.8, 0.05])
    ax_time  = plt.axes([0.1, 0.01, 0.8, 0.031])

    movie_stack = os.path.join(ppath, name, 'MStack')
    if not(os.path.isdir(movie_stack)):
        os.mkdir(movie_stack)

    sr = sleepy.get_snr(ppath, name)
    M = sleepy.load_stateidx(ppath, name)[0]
    dt = 1.0 / sr

    nbins = int(np.round(sr) * 5.0 / 2)
    Mup = spyke.upsample_mx(M, nbins)

    EEG = so.loadmat(os.path.join(ppath, name, 'EEG.mat'), squeeze_me=True)['EEG']
    EMG = so.loadmat(os.path.join(ppath, name, 'EMG.mat'), squeeze_me=True)['EMG']
    vid_time = so.loadmat(os.path.join(ppath, name, 'video_timing.mat'), squeeze_me=True)['onset']
    len_eeg = EEG.shape[0]

    t = np.arange(0, len_eeg)*dt
    its = closest_neighbor(t, ts)[1]
    ite = closest_neighbor(t, te)[1]
    data_eeg = EEG[its:ite]
    states = sleepy.downsample_states(Mup[its:ite], int(np.round(sr)))
    state_map = [[0, 1, 1], [0.5, 0, 1], [0.6, 0.6, 0.6]]

    # load laser
    laser_map = [[1, 1, 1],[0, 0.3, 1]]
    laser = sleepy.load_laser(ppath, name)
    laser = laser[its:ite]
    idxs, idxe = sleepy.laser_start_end(laser, SR=sr)
    npers = int(np.round(sr))
    idxs = [int(i/npers) for i in idxs]
    idxe = [int(i/npers) for i in idxe]

    # setup axis for video
    ax_video.get_xaxis().set_visible(False)
    ax_video.get_yaxis().set_visible(False)
    ax_video.spines["top"].set_visible(False)
    ax_video.spines["right"].set_visible(False)
    ax_video.spines["bottom"].set_visible(False)
    ax_video.spines["left"].set_visible(False)

    # setup axes for EEG spectrogram
    sleepy.box_off(ax_eeg)
    ax_eeg.set_xticks([])
    plt.gcf().text(0.11, 0.38, 'EEG', color='white')

    # setup axes for EMG
    ax_emg.get_xaxis().set_visible(False)
    ax_emg.get_yaxis().set_visible(False)
    ax_emg.spines["top"].set_visible(False)
    ax_emg.spines["right"].set_visible(False)
    ax_emg.spines["bottom"].set_visible(False)
    ax_emg.spines["left"].set_visible(False)
    emg_max = np.max(np.abs(EMG[its:ite]))
    emg_max = emg_max + emg_max*0.1
    # write "EMG" and label EMG legend
    plt.gcf().text(0.905, 0.21,  "%.1f mV" % (emg_legend/1000.0), rotation=90)
    plt.gcf().text(0.11, 0.25, 'EMG')

    ax_bs.get_xaxis().set_visible(False)
    ax_bs.get_yaxis().set_visible(False)
    ax_bs.spines["top"].set_visible(False)
    ax_bs.spines["right"].set_visible(False)
    ax_bs.spines["bottom"].set_visible(False)
    ax_bs.spines["left"].set_visible(False)

    # calculate spectrogram
    fspec, tspec, Sxx = scipy.signal.spectrogram(data_eeg, fs=sr, nperseg=2*npers, noverlap=npers)
    ifreq = np.where(fspec<=fmax)[0]
    med = np.median(Sxx.max(axis=0))
    nspec = len(tspec)
    laser = np.zeros((nspec,))
    for (i,j) in zip(idxs, idxe):
        laser[i:j+1]=1

    # setup axis for laser
    ax_laser.get_xaxis().set_visible(False)
    ax_laser.get_yaxis().set_visible(False)
    ax_laser.spines["top"].set_visible(False)
    ax_laser.spines["right"].set_visible(False)
    ax_laser.spines["bottom"].set_visible(False)
    ax_laser.spines["left"].set_visible(False)
    # write "Laser"
    plt.gcf().text(0.11, 0.46, 'Laser', color=laser_map[1])

    # legend for brain states
    plt.gcf().text(0.7, 0.01,  'REM',  color=state_map[0])
    plt.gcf().text(0.77, 0.01,  'Wake', color=state_map[1])
    plt.gcf().text(0.84, 0.01,  'NREM', color=state_map[2])

    # setup time legend (beolow DF/F panel)
    ax_time.plot((tspec[0], tspec[0]+time_legend), [1, 1], color='black', linewidth=2)
    ax_time.set_xlim((tspec[0], tspec[-1]))
    ax_time.set_ylim((-1,1.1))
    ax_time.get_xaxis().set_visible(False)
    ax_time.get_yaxis().set_visible(False)
    ax_time.spines["top"].set_visible(False)
    ax_time.spines["right"].set_visible(False)
    ax_time.spines["bottom"].set_visible(False)
    ax_time.spines["left"].set_visible(False)
    ax_time.text(tspec[0], -1, '%s s' % str(time_legend))

    tstart = t[its]
    for i in range(2,len(tspec)):
        curr_t = tstart + tspec[i]
        ax_eeg.cla()
        ax_eeg.pcolor(tspec[:i], fspec[ifreq], Sxx[ifreq,:i], vmin=0, vmax=med*vm)
        ax_eeg.set_xlim((tspec[0], tspec[-1]))
        ax_eeg.set_xticks([])
        ax_eeg.set_ylabel('Freq. (Hz)')

        # displary current movie frame
        ax_video.cla()
        iframe = closest_neighbor(vid_time, curr_t)[1]
        img = cv2.imread(os.path.join(ppath, name, 'Stack', 'fig%d.jpg' % (iframe+1)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_video.imshow(img)

        # show EMG
        emg = EMG[its:closest_neighbor(t, curr_t)[1]]
        ax_emg.cla()
        temg = np.arange(0, len(emg))*dt
        ax_emg.plot(temg, emg, color='black', linewidth=0.5)
        ax_emg.set_xlim((tspec[0], tspec[-1]))
        ax_emg.set_ylim((-emg_max, emg_max))
        # plot EMG legend
        ax_emg.plot(([tspec[-1]-1, tspec[-1]-1]), (-emg_legend/2.0, emg_legend/2.0), color='black', linewidth=2)

        # plot brain state patches
        j = i-1
        ax_bs.add_patch(patches.Rectangle((tspec[j-1], 0), tspec[j]-tspec[j-1], 1, facecolor=state_map[int(states[j])-1], edgecolor=state_map[int(states[j])-1]))
        ax_bs.set_xlim((tspec[0], tspec[-1]))
        ax_bs.set_ylim((0,1))

        # plot laser
        #pdb.set_trace()
        ax_laser.add_patch(patches.Rectangle((tspec[j-1], 0), tspec[j]-tspec[j-1], 1, facecolor=laser_map[int(laser[j])], edgecolor=laser_map[int(laser[j])]))
        ax_laser.set_ylim((0,1))
        ax_laser.set_xlim((tspec[0], tspec[-1]))

        if i % 10 == 0:
            print("done with frame %d out of %d frames" % (i, len(tspec)))
        plt.savefig(os.path.join(movie_stack, 'fig%d.png' % i))

    encode_video(ppath, name, stack='MStack', ending='.png', fr=5, outpath=os.path.join(ppath, name), ffmpeg_path=ffmpeg_path, vidname='movie_opto_')



def opto_videoseq(ppath, name, ts_list, te_list, nidle=3, fmax=20, emg_legend=1000, 
                    vm=2.0, time_legend=10, ffmpeg_path='ffmpeg', titles=[], color_map='jet'):
    """
    Generate a sequence videos for optogenetic sleep recording (recorded using intan).
    Each video sequence is introduced by a title.

    Example call:
        vypro.opto_videoseq(ppath, name, [1000, 3000], [1100, 3100], nidle=5, titles=['part1', 'part2']) 

    Note: The function requires that ffmpeg is installed on your system (http://ffmpeg.org).
    Windows Users: Specify the full path to the ffmpeg program

    The resulting video has 1 Hz resolution and will be saved in folder $ppath/$name
    :param ppath: base folder
    :param name: name of recording
    :param ts: start time in seconds
    :param te: end time in second
    :param fmax: maximum frequency on EEG spectrogram
    :param emg_legend: EMG legend in micro Volts
    :param vm: controls saturation of EEG spectrogram; a value in the range from 1 to 2 should work best.
               If you want the colors to be more saturated, choose a smaller value
    :param time_legend: time legend in seconds
    :param ffmpeg_path: full, absolute path to ffmpeg program; important for to set in Windows
    :param titles: list of strings; provide a title for each movie sequence
    :param color_map: string; set a matplotlib colormap
    :return: n/a
    """
    # helper function ######################
    def closest_neighbor(vec, x):
        d = np.abs(vec-x)
        el = np.min(d)
        idx = np.argmin(d)
        return el, idx
    ########################################

    # setup figure arrangement
    sleepy.set_fontsize(12)
    sleepy.set_fontarial()
    plt.ion()
    plt.figure()
    plt.figure(figsize=(8, 6))

    islaser = False
    if os.path.isfile(os.path.join(ppath, name, 'laser_%s.mat'%name)):
        islaser = True

    ax_video = plt.axes([0.1, 0.52, 0.8, 0.45])
    if islaser:
        ax_laser = plt.axes([0.1, 0.45, 0.8, 0.03])
    ax_eeg   = plt.axes([0.1, 0.28, 0.8, 0.15])
    ax_emg   = plt.axes([0.1, 0.11, 0.8, 0.15])
    ax_emgleg = plt.axes([0.9, 0.11, 0.05, 0.15])
    ax_bs    = plt.axes([0.1, 0.05, 0.8, 0.05])
    ax_time  = plt.axes([0.1, 0.01, 0.8, 0.031])

    movie_stack = os.path.join(ppath, name, 'MStack')
    if not(os.path.isdir(movie_stack)):
        os.mkdir(movie_stack)

    sr = sleepy.get_snr(ppath, name)
    M = sleepy.load_stateidx(ppath, name)[0]
    dt = 1.0 / sr
    npers = int(np.round(sr))

    nbins = int(np.round(sr) * 5.0 / 2)
    Mup = spyke.upsample_mx(M, nbins)

    EEG = so.loadmat(os.path.join(ppath, name, 'EEG.mat'), squeeze_me=True)['EEG']
    EMG = so.loadmat(os.path.join(ppath, name, 'EMG.mat'), squeeze_me=True)['EMG']
    vid_time = so.loadmat(os.path.join(ppath, name, 'video_timing.mat'), squeeze_me=True)['onset']
    len_eeg = EEG.shape[0]
    if islaser:
        laser_cmpl = sleepy.load_laser(ppath, name)

    # index for saved figures
    ifig=0
    ipart=0
    for (ts, te) in zip(ts_list, te_list):

        t = np.arange(0, len_eeg)*dt
        its = closest_neighbor(t, ts)[1]
        ite = closest_neighbor(t, te)[1]
        data_eeg = EEG[its:ite]
        states = sleepy.downsample_states(Mup[its:ite], int(np.round(sr)))
        state_map = [[0, 1, 1], [0.5, 0, 1], [0.6, 0.6, 0.6]]
    
        # load laser
        if islaser:
            laser_map = [[1, 1, 1],[0, 0.3, 1]]
            laser = laser_cmpl[its:ite]
            idxs, idxe = sleepy.laser_start_end(laser, SR=sr)
            idxs = [int(i/npers) for i in idxs]
            idxe = [int(i/npers) for i in idxe]
    
        # setup axis for video
        ax_video.get_xaxis().set_visible(False)
        ax_video.get_yaxis().set_visible(False)
        ax_video.spines["top"].set_visible(False)
        ax_video.spines["right"].set_visible(False)
        ax_video.spines["bottom"].set_visible(False)
        ax_video.spines["left"].set_visible(False)
    
        # setup axes for EEG spectrogram
        sleepy.box_off(ax_eeg)
        ax_eeg.set_xticks([])
        plt.gcf().text(0.11, 0.38, 'EEG', color='white')
    
        # setup axes for EMG
        ax_emg.get_xaxis().set_visible(False)
        ax_emg.get_yaxis().set_visible(False)
        ax_emg.spines["top"].set_visible(False)
        ax_emg.spines["right"].set_visible(False)
        ax_emg.spines["bottom"].set_visible(False)
        ax_emg.spines["left"].set_visible(False)
        emg_max = np.max(np.abs(EMG[its:ite]))
        emg_max = emg_max + emg_max*0.1
        # write "EMG" and label EMG legend
        #plt.gcf().text(0.905, 0.21,  "%.1f mV" % (emg_legend/1000.0), rotation=90, verticalalignment='center', horizontalalignment='center')
        plt.gcf().text(0.11, 0.25, 'EMG')
        
        ax_emgleg.plot([0,0], (-emg_legend/2.0, emg_legend/2.0), color='red', linewidth=2)
        ax_emgleg.set_xlim([-1, 1])
        ax_emgleg.set_ylim((-emg_max, emg_max))
        ax_emgleg.get_xaxis().set_visible(False)
        ax_emgleg.get_yaxis().set_visible(False)
        ax_emgleg.spines["top"].set_visible(False)
        ax_emgleg.spines["right"].set_visible(False)
        ax_emgleg.spines["bottom"].set_visible(False)
        ax_emgleg.spines["left"].set_visible(False)
        ax_emgleg.text(0.5, 0, "%.1f mV" % (emg_legend/1000.0), rotation=90, verticalalignment='center', horizontalalignment='left', color='red', fontsize=12)
    
        ax_bs.get_xaxis().set_visible(False)
        ax_bs.get_yaxis().set_visible(False)
        ax_bs.spines["top"].set_visible(False)
        ax_bs.spines["right"].set_visible(False)
        ax_bs.spines["bottom"].set_visible(False)
        ax_bs.spines["left"].set_visible(False)
    
        # calculate spectrogram
        fspec, tspec, Sxx = scipy.signal.spectrogram(data_eeg, fs=sr, nperseg=2*npers, noverlap=npers)
        ifreq = np.where(fspec<=fmax)[0]
        med = np.median(Sxx.max(axis=0))
        nspec = len(tspec)
        if islaser:
            laser = np.zeros((nspec,))
            for (i,j) in zip(idxs, idxe):
                laser[i:j+1]=1
    
        # setup axis for laser
        if islaser:
            ax_laser.get_xaxis().set_visible(False)
            ax_laser.get_yaxis().set_visible(False)
            ax_laser.spines["top"].set_visible(False)
            ax_laser.spines["right"].set_visible(False)
            ax_laser.spines["bottom"].set_visible(False)
            ax_laser.spines["left"].set_visible(False)
            # write "Laser"
            plt.gcf().text(0.11, 0.46, 'Laser', color=laser_map[1])
    
        # legend for brain states
        plt.gcf().text(0.7, 0.01,  'REM',  color=state_map[0])
        plt.gcf().text(0.77, 0.01,  'Wake', color=state_map[1])
        plt.gcf().text(0.84, 0.01,  'NREM', color=state_map[2])
    
        # setup time legend (beolow DF/F panel)
        ax_time.plot((tspec[0], tspec[0]+time_legend), [1, 1], color='black', linewidth=2)
        ax_time.set_xlim((tspec[0], tspec[-1]))
        ax_time.set_ylim((-1,1.1))
        ax_time.get_xaxis().set_visible(False)
        ax_time.get_yaxis().set_visible(False)
        ax_time.spines["top"].set_visible(False)
        ax_time.spines["right"].set_visible(False)
        ax_time.spines["bottom"].set_visible(False)
        ax_time.spines["left"].set_visible(False)
        ax_time.text(tspec[0], -1, '%s s' % str(time_legend))
    
        tstart = t[its]
        #for i in range(2,len(tspec)):
        i=1
        idx_idle = 0
        while (i < len(tspec)):
            curr_t = tstart + tspec[i]
            ax_eeg.cla()
            ax_eeg.pcolor(tspec[:i+1], fspec[ifreq], Sxx[ifreq,:i+1], vmin=0, vmax=med*vm, cmap=color_map)
            ax_eeg.set_xlim((tspec[0], tspec[-1]))
            ax_eeg.set_xticks([])
            ax_eeg.set_ylabel('Freq. (Hz)')
    
            # display current movie frame
            ax_video.cla()
            iframe = closest_neighbor(vid_time, curr_t)[1]
            img = cv2.imread(os.path.join(ppath, name, 'Stack', 'fig%d.jpg' % (iframe+1)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax_video.imshow(img)
    
            # show EMG
            emg = EMG[its:closest_neighbor(t, curr_t)[1]]
            ax_emg.cla()
            temg = np.arange(0, len(emg))*dt
            ax_emg.plot(temg, emg, color='black', linewidth=0.5)
            ax_emg.set_xlim((tspec[0], tspec[-1]))
            ax_emg.set_ylim((-emg_max, emg_max))
            # plot EMG legend
            #ax_emg.plot(([tspec[-1]-1, tspec[-1]-1]), (-emg_legend/2.0, emg_legend/2.0), color='black', linewidth=2)
            #ax_emgleg.plot([0,0], (-emg_legend/2.0, emg_legend/2.0), color='red', linewidth=2)
            #ax_emgleg.set_xlim([-1, 1])
            #ax_emgleg.set_ylim((-emg_max, emg_max))
    
            # plot brain state patches
            ax_bs.add_patch(patches.Rectangle((tspec[i-1], 0), tspec[i]-tspec[i-1], 1, facecolor=state_map[int(states[i])-1], edgecolor=state_map[int(states[i])-1]))
            #pdb.set_trace()
            ax_bs.set_xlim((tspec[0], tspec[-1]))
            ax_bs.set_ylim((0,1))
    
            # plot laser
            if islaser:
                ax_laser.add_patch(patches.Rectangle((tspec[i-1], 0), tspec[i]-tspec[i-1], 1, facecolor=laser_map[int(laser[i])], edgecolor=laser_map[int(laser[i])]))
                ax_laser.set_ylim((0,1))
                ax_laser.set_xlim((tspec[0], tspec[-1]))
    
            if idx_idle < nidle:
                # write title on figure:
                ny = img.shape[0]
                nx = img.shape[1]
                if len(titles) >0:
                    ax_video.text(nx/2, ny/2, titles[ipart], color='yellow', 
                                  horizontalalignment='center', verticalalignment='center', fontsize=20)
                idx_idle += 1                
            else :
                i += 1

            if i % 10 == 0:
                print("done with frame %d out of %d frames in part %d" % (i, len(tspec), ipart+1))
            print(ifig)
            plt.savefig(os.path.join(movie_stack, 'fig%d.png' % ifig))

            ifig += 1
        ax_bs.cla()
        ipart += 1
    
    encode_video(ppath, name, stack='MStack', ending='.png', fr=5, outpath=os.path.join(ppath, name), ffmpeg_path=ffmpeg_path, vidname='movie_opto_')
