#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 16:48:55 2018

@author: Franz
"""

import sys
from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
import pyqtgraph as pg
import numpy as np
import scipy.io as so
import pyqtgraph.dockarea as dock
import os.path
import h5py
import pdb
import re
import cv2
import sleepy
import vypro



def param_check(x) :
    params = {}
    for i in range(0, len(x)) :
        a = x[i]
        if re.match('^-', a) > 0 :
            if i < len(x)-1 :
                params[a] = x[i+1]

    return params


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


def laser_start_end(laser, SR=1525.88, intval=5):
    """laser_start_end(laser, SR=TDT_sampling_rate, intval=5 [s]) ...
    print start and end index of laser stimulation periods\
    returns the tuple (istart, iend), both indices are inclusive,\
    i.e. part of the sequence
    
    laser    -    laser, vector of 0s and 1s
    intval   -    min time interval [s] between two laser sequences
    """
    idx = np.nonzero(laser > 0.1)[0]
    if len(idx) == 0 :
        return ([], [])
    
    idx2 = np.nonzero(np.diff(idx)*(1./SR) > intval)[0]
    istart = np.hstack([idx[0], idx[idx2+1]])
    iend   = np.hstack([idx[idx2], idx[-1]])    
    
    return (istart, iend)


def firing_rate(ppath, name, grp, un, nbin):
    """
    y = firing_rate(ppath, name, grp, un, nbin)
    """
    idx, train, mspike = unpack_unit(ppath, name, grp, un)
    spikesd = downsample_vec(train, nbin)
    return spikesd, mspike, idx, train


def unpack_unit(ppath, name, grp, un):
    """
    Load a unit from the Spk%d.mat (old version) or Spk%d.npz (new version) file
    !! Units are counted as in klusters: 1 - Noise; 2 ... n, are clusters/units 1 ... n-1 !!
    @Parameters:
        ppath,name      recording
        (grp, un)       Unit $un of electrode/group $grp
    @Return:
        idx             indices of spikes at raw SR
        train           spike train in EEG SR resolution
        mspike          average spike waveform
    """
    sfile = os.path.join(ppath, name, 'Spk' + str(grp))
    if os.path.isfile(sfile + '.mat'):
        # subtract -1 to get the right list index; klusters starts counting with 1
        un -= 1
        Spk = so.loadmat(sfile, struct_as_record=False, squeeze_me=True)['S']
        train = np.squeeze(Spk[un].train.toarray())
        mspike = Spk[un].mspike
        idx = Spk[un].idx
    elif os.path.isfile(sfile + '.npz'):
        # !!!
        # add 1 as units are saved in a dictionary with the key set by klusters
        # which starts counting from 1
        # un += 1
        Spk = np.load(sfile + '.npz')
        print "Loaded %s" % sfile + '.npz'
        # explanation of why this is necessary:
        # https://stackoverflow.com/questions/22661764/storing-a-dict-with-np-savez-gives-unexpected-result/41566840
        Spk = {key: Spk[key].item() for key in Spk if Spk[key].dtype == 'O'}
        train = Spk['train'][un]
        mspike = Spk['mspike'][un]
        idx = Spk['idx'][un]
    return idx, train, mspike



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
        idx = range(i, int(n_down*nbin), int(nbin))
        x_down += x[idx]

    return x_down / nbin



# Object holding the GUI window
class MainWindow(QtGui.QMainWindow):

    def __init__(self, ppath, name, config_file, parent=None):
        super(MainWindow, self).__init__(parent)        
        QtGui.QMainWindow.__init__(self)
        # sizing of Main Window
        self.setGeometry(QtCore.QRect(100, 0, 1500, 800))
    
        # Variables related to EEG representation
        # maximum length of shown EEG
    
        self.twin_eeg = 5
        self.twin_view = self.twin_eeg
        # number of time point in EEG; set in self.load_session()
        self.len_eeg = 0
    
        # variables for cell activity
        self.ephys = True
        # list of firing rates or DFF
        self.act_list = []
    
        # Variables related to behavior annotation
        self.tstep = 0.1
        # time series of annotated time points
        # list is set in load_video_timing and/or load_config
        self.tscale_ann = []
        # index of currently shown video frame
        self.image_idx = 0
        # index of currently annotated time point
        # Note that image_idx and tscale_index might not be identical
        # due to the delay at which the camera starts recording
        self.tscale_index = 10
        # the current time point shown in [s]
        self.curr_time = 0
        # the currently shown image frame
        self.image_frame = 0

        # in annotation mode each time series point in self.tscale_ann can be assigned a behavior
        self.pann_mode = False
        # only if configuration file is provided pconfig can be True; only if pconfig == True,
        # pann_mode can be True, i.e. annotation only works, 
        # if a configuration file was provided
        self.pconfig = False

        # annotate whole time range at once, activated by "space"
        self.pcollect_index = False
        # list of indices within range
        self.index_list = []

        # ndarray of currently displayed video frame
        self.curr_image = np.array([])

        # setup names for recording folder / mouse     
        self.name = name
        self.ppath = ppath        
        if self.name == '':
            self.openFileNameDialog()        
        self.mouse = re.split('_', self.name)[0]
        self.load_video_timing()
        self.curr_time = self.tscale_ann[self.tscale_index]

        # test if it's ephys or fiber photometry
        files = [f for f in os.listdir(os.path.join(self.ppath, self.name)) if re.match('^ch_', f)]
        if len(files) > 0:
            self.ephys = True
        else:
            self.ephys = False
        #if os.path.isfile(os.path.join(self.ppath, self.name, 'DFF.mat')) or os.path.isfile(os.path.join(self.ppath, self.name, 'tone.mat')):
        #    self.ephys = False

        # check for amplifier:
        amplifier = get_infoparam(os.path.join(self.ppath, self.name, 'info.txt'), 'amplifier')[0]
        self.img_offset = 1
        if amplifier == 'TDT':
            self.img_offset = 2


        # load configuration file
        self.config_file = config_file
        # graph for annotation, have to declare already here,
        # otherwise there's a problem with load_config()
        self.graph_annotation = pg.PlotWidget()
        # similar need to generate graph to show current time point and behavior annotation
        self.graph_treck      = pg.PlotWidget()
        # load behavior configuration and annotation
        self.load_config()
        

        # make sure Stack folder with unzipped images exists
        if not(os.path.isdir(os.path.join(self.ppath, self.name, 'Stack'))):
            if os.path.isfile(os.path.join(self.ppath, self.name, 'stack.zip')):
                vypro.unpack_zipstack(self.ppath, self.name)
            else:
                print "no video stack available for recording %s" % self.name

        # now that the configuration is loaded, set image index
        self.set_image_idx()

        # whole area is a docking area, where single docks can be added to
        self.area = dock.DockArea()      
        self.setCentralWidget(self.area)
        
        # Dock with video frame
        self.dock_video = dock.Dock("Video %s" % self.name, size=(50, 30))
        self.area.addDock(self.dock_video, 'left')
        
        # add layout for video player
        self.layout_video = pg.LayoutWidget()

        # graph to show current video frame
        self.graph_frame = pg.PlotWidget()
        self.layout_video.addWidget(self.graph_frame)

        # and finally add layout_video to dock_video
        self.dock_video.addWidget(self.layout_video)

        # Dock with graphs to display if unit is driven
        self.dock_annotation = dock.Dock("Annotation", size=(30, 30))
        self.area.addDock(self.dock_annotation, 'bottom')
        self.layout_annotation = pg.LayoutWidget()
        self.layout_annotation.addWidget(self.graph_annotation,         row=0, col=0, colspan=4, rowspan=1)

        # Row 1
        self.layout_annotation.addWidget(QtGui.QLabel("Symbols"), row=1, col=0)
        self.label_symbols = QtGui.QLineEdit(', '.join(k+'-'+self.symbols[k] for k in self.symbols.keys()))        
        self.layout_annotation.addWidget(self.label_symbols,            row=1, col=1, colspan=2)        
        self.button_create = QtGui.QPushButton('Create')
        self.layout_annotation.addWidget(self.button_create,            row=1, col=3)
        self.button_create.clicked.connect(self.create_annotation)         

        # Row 2
        # Activity - Grp, Unit, Plot
        self.layout_annotation.addWidget(QtGui.QLabel("Activity"),  row=2, col=0)
        self.edit_group  = QtGui.QLineEdit('')
        self.layout_annotation.addWidget(self.edit_group,           row=2, col=1, colspan=1)
        self.edit_unit   = QtGui.QLineEdit('')
        self.layout_annotation.addWidget(self.edit_unit,            row=2, col=2, colspan=1)
        self.button_plot = QtGui.QPushButton('Plot')
        self.layout_annotation.addWidget(self.button_plot,          row=2, col=3, colspan=1)
        self.button_plot.clicked.connect(self.plot_activity)

        # Row 3
        self.layout_annotation.addWidget(QtGui.QLabel("t step[s]"), row=3, col=0)
        self.edit_tstep = QtGui.QLineEdit('0.1')
        self.layout_annotation.addWidget(self.edit_tstep,           row=3, col=1)
        self.button_tstep = QtGui.QPushButton('Set')
        self.layout_annotation.addWidget(self.button_tstep,         row=3, col=2)
        self.button_tstep.clicked.connect(self.set_tstep)
        self.label_mode = QtGui.QLabel("Annotate")
        self.layout_annotation.addWidget(self.label_mode,           row=3, col=3)


        # finally add layout_annotation to dock_annotation
        self.dock_annotation.addWidget(self.layout_annotation)

        # dock for data
        self.dock_session = dock.Dock("Whole Session", size=(100, 600))
        self.dock_eeg = dock.Dock("EEG/EMG", size=(100, 800))
                
        self.area.addDock(self.dock_session, 'left')
        self.area.addDock(self.dock_eeg, 'bottom', self.dock_session)
        
        # show the whole GUIp
        self.show()
                
        self.display_mode()
        self.set_image()
        self.load_session()
        self.plot_treck()
        self.plot_eeg()
        

    def set_image(self):
        self.load_image()
        self.image_frame = pg.ImageItem() 
        self.graph_frame.clear()
        self.graph_frame.addItem(self.image_frame)   
        
        self.image_frame.clear()             
        self.image_frame.setImage(self.curr_image)     
        self.graph_frame.getViewBox().setAspectLocked(True)
        self.graph_frame.hideAxis('bottom')
        self.graph_frame.hideAxis('left')

        text = pg.TextItem(text='%.2f s' % self.frame_onset[self.image_idx], angle=0, border='w', fill=(0, 0, 255, 100))
        text.setPos(self.curr_image.shape[0]*0.01, self.curr_image.shape[1]-self.curr_image.shape[1]*0.01)
        self.graph_frame.addItem(text)


    def load_image(self):
        image = cv2.imread(os.path.join(self.ppath, self.name, 'Stack', 'fig%d.jpg' % (self.image_idx+self.img_offset)))
        self.curr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    def load_video_timing(self):
        vid = so.loadmat(os.path.join(ppath, name, 'video_timing.mat'), squeeze_me=True)
        self.frame_onset = vid['onset']
        self.tscale_ann = vid['tick_onset']
        #self.image_idx = np.argmin(np.abs(self.frame_onset-self.curr_time))
        #self.set_image_idx()
        
        
    def load_config(self):
        if os.path.isfile(self.config_file):
            self.symbols, self.tann = vypro.load_config(self.config_file)
            #self.annotation = vypro.load_behann_file(self.config_file)
            self.annotation, self.Kann = vypro.load_behann_file(self.config_file)
            self.tscale_ann = np.array(self.annotation.keys())
            self.tscale_ann.sort()
            self.symbol_series = [self.annotation[k] for k in self.tscale_ann]
            self.pconfig = True
            self.pann_mode = True

            # collect keys for symbols
            self.symbol_keys = []
            # dict: key number --> symbol string
            self.symbol_dict = {}
            # string symbol to number
            self.symbol_num = {'' : 0}
            self.num_symbol = {}
            i = 1
            for s in self.symbols:
                # list of key numbers
                self.symbol_keys.append(eval('QtCore.Qt.Key_' + s.upper()))
                self.symbol_dict[eval('QtCore.Qt.Key_' + s.upper())] = s
                self.symbol_num[s] = i
                i += 1
            # reverse self.symbol_num
            self.num_symbol = {self.symbol_num[k]:k for k in self.symbol_num.keys()}

            #translate string series to array
            self.symbol_array = np.zeros((len(self.symbol_series),), dtype='int')
            i = 0
            for s in self.symbol_series:
                self.symbol_array[i] = self.symbol_num[s]
                i += 1
                
            # generate ticks for behavior annotation plots
            self.ticks = [(0, '')]
            i = 1
            for s in self.symbols:
                self.ticks.append((i,s))
                i += 1            
        else:
            self.pconfig = False
            self.symbols = {'x':'xxx', 'y':'yyy'}
            self.ticks = [(0,''), (1,'x')]
            self.symbol_array = np.zeros((len(self.tscale_ann,)))
            self.Kann = np.zeros((len(self.tscale_ann),))


        # setup graph_annotation
        ax = self.graph_annotation.getAxis(name='left')
        label_style = {'color': '#FFF', 'font-size': '12pt'}
        ax.setLabel('Behavior', units='', **label_style)
        ax.setTicks([self.ticks])
        ax = self.graph_annotation.getAxis(name='bottom')
        ticks = [(j,'') for j in range(11)]
        ax.setTicks([ticks])
        nsymb = len(self.symbols)
        self.graph_annotation.setRange(yRange=(0, nsymb), xRange=(0,11), padding=None)
        # END [setup graph_annotation]
        
        ax = self.graph_treck.getAxis(name='left')
        ax.setTicks([self.ticks])
        

    def load_session(self):
        self.SR_eeg = sleepy.get_snr(self.ppath, self.name)
        self.dt = 1/self.SR_eeg
        # number of sampled point for one fourier bin
        #self.fbin = np.round((1/self.dt) * 2.5)

        # graphs going into dock_session
        self.graph_activity   = pg.PlotWidget()
        self.graph_brainstate = pg.PlotWidget()
        self.graph_eegspec    = pg.PlotWidget()
        self.graph_emgampl    = pg.PlotWidget()

        # graphs for dock EEG/EMG
        self.graph_eeg = pg.PlotWidget()
        self.graph_emg = pg.PlotWidget()

        # load data ###########################################################
        # load EEG/EMG
        self.eeg_pointer = 0
        self.emg_pointer = 0
        self.EEG_list = []
        self.EMG_list = []
        self.eeg_spec_list = []
        self.emg_amp_list = []
        EEG = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EEG.mat'))['EEG'])
        self.EEG_list.append(EEG)
        if os.path.isfile(os.path.join(self.ppath, self.name, 'EEG2.mat')):
            EEG2 = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EEG2.mat'))['EEG2'])
            self.EEG_list.append(EEG2)
        self.EEG = self.EEG_list[0]

        EMG = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EMG.mat'))['EMG'])
        self.EMG_list.append(EMG)
        if os.path.isfile(os.path.join(self.ppath, self.name, 'EMG2.mat')):
            EMG2 = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EMG2.mat'))['EMG2'])
            self.EMG_list.append(EMG2)
        self.EMG = self.EMG_list[0]

        # load spectrogram / EMG amplitude
        if not(os.path.isfile(os.path.join(self.ppath, self.name, 'sp_' + self.name + '.mat'))):
            # spectrogram does not exist, generate it
            sleepy.calculate_spectrum(self.ppath, self.name, fres=0.5)
            print("Calculating spectrogram for recording %s\n" % self.name)
        
        spec = so.loadmat(os.path.join(self.ppath, self.name, 'sp_' + self.name + '.mat'))
        self.ftime = spec['t'][0]
        self.fdt = spec['dt'][0][0]
        # New lines
        self.fbin = np.round((1 / self.dt) * self.fdt)
        if self.fbin % 2 == 1:
            self.fbin += 1
        # END new lines
        self.eeg_spec = spec['SP']
        self.eeg_spec_list.append(spec['SP'])
        if spec.has_key('SP2'):
            self.eeg_spec_list.append(spec['SP2'])
        # max color for spectrogram color-range
        self.color_max = np.max(self.eeg_spec)
        freq = np.squeeze(spec['freq'])
        self.ifreq = np.where(freq <= 25)[0][-1]
        self.fdx = freq[1]-freq[0]
        self.mfreq = np.where((freq>=10) & (freq <= 200))[0]
        
        emg_spec = so.loadmat(os.path.join(self.ppath, self.name, 'msp_' + self.name + '.mat'))
        EMGAmpl = np.sqrt(emg_spec['mSP'][self.mfreq,:].sum(axis=0))
        self.EMGAmpl = EMGAmpl
        self.emg_amp_list.append(EMGAmpl)
        if emg_spec.has_key('mSP2'):
            EMGAmpl2 = np.sqrt(emg_spec['mSP2'][self.mfreq,:].sum(axis=0))
            self.emg_amp_list.append(EMGAmpl2)

        self.nbin = len(self.ftime) #number of bins in fourier time
        self.len_eeg = len(self.EEG)
                        
        # load brain state
        if not(os.path.isfile(os.path.join(self.ppath, self.name, 'remidx_' + self.name + '.txt'))):
            # predict brain state
            M,S = sleepy.sleep_state(self.ppath, self.name, pwrite=1, pplot=0)
        (A,self.K) = sleepy.load_stateidx(self.ppath, self.name)
        # needs to be packed into 1 x nbin matrix for display
        self.M = np.zeros((1,self.nbin))
        self.M[0,:] = A
                
        # load laser
        self.laser = np.zeros((self.nbin,))
        laser_file = os.path.join(self.ppath, self.name, 'laser_' + self.name + '.mat')
        if os.path.isfile(laser_file):
            try:
                self.laser = np.squeeze(so.loadmat(laser_file)['laser'])            
            except:
                self.laser = np.squeeze(np.array(h5py.File(laser_file,'r').get('laser')))
        
        # downsample laser to brainstate time
        (idxs, idxe) = laser_start_end(self.laser, SR=self.SR_eeg)
        # downsample EEG time to spectrogram time    
        idxs = [int(i/self.fbin) for i in idxs]
        idxe = [int(i/self.fbin) for i in idxe]
        self.laser_dn = np.zeros((self.nbin,))
        for (i,j) in zip(idxs, idxe):
            self.laser_dn[i:j+1] = 1
                
        # max color for spectrogram
        self.color_max = np.max(self.eeg_spec)

        ### END load data #############################################################
        # Plot whole session
        self.dock_session.addWidget(self.graph_treck, row=0, col=0, rowspan=1)
        self.dock_session.addWidget(self.graph_activity, row=1, col=0, rowspan=1)
        self.dock_session.addWidget(self.graph_brainstate, row=2, col=0)
        self.dock_session.addWidget(self.graph_eegspec, row=3, col=0, rowspan=1)
        self.dock_session.addWidget(self.graph_emgampl, row=4, col=0)

        # display data
        # mix colormaps
        pos = np.array([0, 0.2, 0.4, 0.6, 0.8])
        #color = np.array([[0,255,255,255], [255,0,255, 255], [192,192,192,255], (0, 0, 0, 255), (255,255,0, 255)], dtype=np.ubyte)
        color = np.array([[0,255,255,255], [150,0,255, 255], [192,192,192,255], (0, 0, 0, 255), (255,255,0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.lut_brainstate = cmap.getLookupTable(0.0, 1.0, 5)
        
        pos = np.array([0., 0.05, .2, .4, .6, .9])
        color = np.array([[0, 0, 0, 255], [0,0,128,255], [0,255,0,255], [255,255,0, 255], (255,165,0,255), (255,0,0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.lut_spectrum = cmap.getLookupTable(0.0, 1.0, 256)

        # plot brainstate
        self.image_brainstate = pg.ImageItem() 
        self.graph_brainstate.addItem(self.image_brainstate)                
        self.image_brainstate.setImage(self.M.T)

        self.image_brainstate.scale(self.fdt,1)
        self.graph_brainstate.setMouseEnabled(x=True, y=False)
        #self.graph_brainstate.setXLink(self.graph_eegspec)
        limits = {'xMin': 0*self.fdt, 'xMax': self.ftime[-1], 'yMin': 0, 'yMax': 1}
        self.graph_brainstate.setLimits(**limits)

        ax = self.graph_brainstate.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('Brainstate', units='', **labelStyle)
        ax.setTicks([[(0, ''), (1, '')]])

        ax = self.graph_brainstate.getAxis(name='bottom')
        ax.setTicks([[]])

        self.image_brainstate.setLookupTable(self.lut_brainstate)
        self.image_brainstate.setLevels([1, 5])
                
        # plot EEG spectrum
        # clear plot and then reload ImageItem        
        self.graph_eegspec.clear()
        self.image_eegspec = pg.ImageItem() 
        self.graph_eegspec.addItem(self.image_eegspec)
        ax = self.graph_eegspec.getAxis(name='bottom')
        ax.setTicks([[]])
        
        # scale image to seconds, minutes or hours        
        self.image_eegspec.setImage(self.eeg_spec[0:self.ifreq,:].T)
        self.image_eegspec.scale(self.fdt, 1.0*self.fdx)
        
        # mousescroll only allowed along x axis
        self.graph_eegspec.setMouseEnabled(x=True, y=False)
        limits = {'xMin': 0*self.fdt, 'xMax': self.ftime[-1], 'yMin': 0, 'yMax': 20}
        self.graph_eegspec.setLimits(**limits)
        # label for y-axis
        ax = self.graph_eegspec.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('Freq', units='Hz', **labelStyle)
        # xTicks
        ax.setTicks([[(0, '0'), (10, '10')]])

        # colormap
        self.image_eegspec.setLookupTable(self.lut_spectrum)
        # link graph with self.graph_brainstate
        # and link all other graphs together
        self.graph_eegspec.setXLink(self.graph_brainstate)
        self.graph_brainstate.setXLink(self.graph_eegspec)
        self.graph_eegspec.setXLink(self.graph_treck)

        # plot EMG ampl
        self.graph_emgampl.clear()
        self.graph_emgampl.plot(self.ftime, self.EMGAmpl)
        self.graph_emgampl.setMouseEnabled(x=False, y=True)
        self.graph_emgampl.setXLink(self.graph_eegspec)
        limits = {'xMin': 0, 'xMax': self.ftime[-1]}
        self.graph_emgampl.setLimits(**limits)
        # y-axis
        ax = self.graph_emgampl.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('EMG', units='uV', **labelStyle)
        ax.enableAutoSIPrefix(enable=False)
        # x-axis        
        ax = self.graph_emgampl.getAxis(name='bottom')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('Time', units='s', **labelStyle)

        self.graph_eegspec.scene().sigMouseClicked.connect(self.mouse_pressed)
        self.graph_brainstate.scene().sigMouseClicked.connect(self.mouse_pressed)
        self.graph_treck.scene().sigMouseClicked.connect(self.mouse_pressed)
        self.graph_activity.scene().sigMouseClicked.connect(self.mouse_pressed)
        
        # setup graph_treck ###########################################################
        self.graph_treck.setMouseEnabled(x=True, y=False)
        limits = {'xMin': 0, 'xMax': self.ftime[-1]} 
        self.graph_treck.setLimits(**limits)
        self.graph_treck.setXLink(self.graph_eegspec)
                                
        ax = self.graph_treck.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('Behavior', units='', **labelStyle)
        ax = self.graph_treck.getAxis(name='bottom')
        ax.setTicks([[]])

        ax = self.graph_treck.getAxis(name='left')
        label_style = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('Behavior', units='', **label_style)
        ax.setTicks([self.ticks])
        # END setup graph_treck #######################################################

        # setup graph_activity ########################################################        
        ax = self.graph_activity.getAxis(name='left')
        limits = {'xMin': 0, 'xMax': self.ftime[-1]} 
        self.graph_activity.setLimits(**limits)
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('Activity', units='', **labelStyle)
        ax = self.graph_activity.getAxis(name='bottom')
        ax.setTicks([[]])
        #self.graph_activity.setMouseEnabled(x=False, y=True)
        self.graph_activity.setXLink(self.graph_treck)
        # END setup graph_activity #####################################################

        # Done With dock "Session" ####################################################
        
        # plot raw EEG, EMG in dock EEG/EMG (dock_eeg)
        self.graph_eeg.setXLink(self.graph_emg)
        self.graph_eeg.setMouseEnabled(x=True, y=True)
        self.graph_emg.setMouseEnabled(x=True, y=True)
        self.graph_eeg.setXRange(self.curr_time-self.twin_view/2, self.curr_time+self.twin_view/2, padding=0)

        # setup graph_eeg
        ax = self.graph_eeg.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '12pt'}
        ax.setLabel('EEG', units='uV', **labelStyle)
        ax = self.graph_eeg.getAxis(name='bottom')
        ax.setTicks([[]])                           
        ax = self.graph_eeg.getAxis(name='bottom')

        # setup graph_emg
        ax = self.graph_emg.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '12pt'}
        ax.setLabel('EMG', units='uV', **labelStyle)
        ax = self.graph_eeg.getAxis(name='bottom')
        ax.setTicks([[]])                   

        ax = self.graph_emg.getAxis(name='bottom')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('Time', units='s', **labelStyle)


        self.dock_eeg.addWidget(self.graph_eeg, row=0, col=0)
        self.dock_eeg.addWidget(self.graph_emg, row=1, col=0)


    def plot_annotation(self):
        self.graph_annotation.clear()
    
        if self.tscale_index >= 5 and self.tscale_index+5 < (len(self.tscale_ann)-1):
            y = self.symbol_array[self.tscale_index-5:self.tscale_index+6]
        elif self.tscale_index <= 5:
            tmp = self.symbol_array[0:self.tscale_index+6]
            y = np.zeros((11,))
            y[-tmp.shape[0]:] = tmp
        elif self.tscale_index > self.tscale_index -5:
            tmp = self.symbol_array[self.tscale_index-5:]
            y = np.zeros((11,))
            y[:tmp.shape[0]] = tmp
            
        x = np.arange(0,11,1)
        self.graph_annotation.plot(x, y, name = 'Ann', pen=(255,0,0), symbolPen='w')
        self.graph_annotation.plot([5.0], [y[5]], pen=(0,0,0), symbolPen=(255,0,0), symbolBrush=(255,0,0), symbolSize=8)
        
        self.plot_treck()
        
    
    def plot_treck(self):    
        self.graph_treck.clear()
        self.graph_treck.plot(self.tscale_ann, self.Kann*0.5)
        self.graph_treck.plot(self.tscale_ann, self.symbol_array, pen=(255,100,100))
        self.graph_treck.plot([self.curr_time], [0.0], pen=(0,0,0), symbolPen=(255,0,0), symbolBrush=(255,0,0), symbolSize=5)


    def plot_activity(self):
        if self.ephys:
            grp = int(self.edit_group.text())
            un  = int(self.edit_unit.text())

            if os.path.isfile(os.path.join(self.ppath, self.name, 'Spk' + str(grp) + '.npz')):
                fr = firing_rate(self.ppath, self.name, grp, un, int(self.fbin))[0]
                self.act_list.append(fr)
        elif os.path.isfile(os.path.join(self.ppath, self.name, 'DFF.mat')):
            self.act_list = [so.loadmat(os.path.join(self.ppath, self.name, 'DFF.mat'), squeeze_me=True)['dffd']]
        elif os.path.isfile(os.path.join(self.ppath, self.name, 'tone.mat')):
            self.act_list = [so.loadmat(os.path.join(self.ppath, self.name, 'tone.mat'), squeeze_me=True)['toned']]
        elif os.path.isfile(os.path.join(self.ppath, self.name, 'laser_' + self.name + '.mat')):
            self.act_list.append(self.laser_dn)
        self.graph_activity.clear()
        for fr in self.act_list:
            n = np.min((fr.shape[0], self.ftime.shape[0]))
            self.graph_activity.plot(self.ftime[0:n], fr[0:n])

            
    def plot_eeg(self):
        self.graph_eeg.clear()
        #self.graph_eeg.setRange(xRange=(t[0],t[-1]), padding=None)
        ictr = int(np.round(self.curr_time*self.SR_eeg))
        iwin = int(np.round(self.twin_eeg*self.SR_eeg))
        ii = np.arange(np.max([0,ictr-iwin]),np.min([ictr+iwin+1, self.len_eeg]), 1)
        t = ii / self.SR_eeg
        self.graph_eeg.plot(t,self.EEG[ii])
        #self.graph_eeg.plot([self.curr_time], [0.0], pen=(0,0,0), symbolPen=(255,0,0), symbolBrush=(255,0,0), symbolSize=5)

        ax = self.graph_eeg.getAxis(name='bottom')
        self.twin_view = ax.range[-1] - ax.range[0]        
        self.graph_eeg.setXRange(self.curr_time-self.twin_view/2, self.curr_time+self.twin_view/2, padding=0)
        ax = self.graph_eeg.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '12pt'}
        ax.setLabel('EEG' + ' ' + str(self.eeg_pointer+1), units='V', **labelStyle)

        self.graph_emg.clear()
        self.graph_emg.plot(t,self.EMG[ii])
        self.graph_emg.plot([self.curr_time], [0.0], pen=(0, 0, 0), symbolPen=(255, 0, 0), symbolBrush=(255, 0, 0),
                        symbolSize=5)
        ax = self.graph_emg.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '12pt'}
        ax.setLabel('EMG' + ' ' + str(self.emg_pointer+1), units='V', **labelStyle)


    def display_mode(self):
        if self.pann_mode:
            self.label_mode.setText('Annotate')
        else:
            self.label_mode.setText('View')


    def mouse_pressed(self, evt):
        mousePoint = self.graph_eegspec.getViewBox().mapToView(evt.pos())  
                
        # update current time point
        self.curr_time = mousePoint.x()
        print self.curr_time
        # shift time point to point in self.tscale_ann
        if self.pann_mode:
            i = np.argmin(np.abs(self.tscale_ann - self.curr_time))
            self.curr_time = self.tscale_ann[i]
            self.tscale_index = i

        #self.image_idx = np.argmin(np.abs(self.frame_onset-self.curr_time))
        self.set_image_idx()
        self.set_image()
        self.plot_annotation()
        self.plot_eeg()
        
        #self.index_fft = int((mousePoint.x()/(self.fdt)))
        #self.image_idx = 
        #self.index_eeg = self.index_fft*self.fbin
        #self.index_raw = self.index_eeg*self.ndec
        #self.current_t = self.index_eeg*self.dt
        #self.current_t_pressed = self.current_t
        #ax = self.graph_channels[0].getAxis(name='bottom')
        #self.twin_view = ax.range[-1] - ax.range[0]

        #self.plot_treck()
        #self.plot_channels()


    def save_annotation(self):
        # first convert symbol_array to symbol_series (list of strings)
        time = self.annotation.keys()
        time.sort()
        for (t,s) in zip(time, self.symbol_array):
            self.annotation[t] = self.num_symbol[s]
        vypro.save_behann_file(self.config_file, self.symbols, self.tann, self.annotation, self.Kann)


    def create_annotation(self):
        self.tann = 'tick'
        a = self.label_symbols.text()        

        sym = {}
        a = re.sub('\s', '', a)
        pairs = re.split(',', a)
        for p in pairs:                
            a = re.split('-', p)
            sym[a[0]] = a[1]

        idf = ''.join(sym.keys())
        if self.pconfig == False:
            vypro.create_behann_file(self.ppath, self.name, idf, sym, t=self.tann)

        self.config_file = os.path.join(self.ppath, self.name, 'vip_' + idf + '.txt')
        self.pconfig = True
        self.pann_mode = True
        self.reset_tscale()

        self.load_config()
        

    def openFileNameDialog(self):    
        fileDialog = QFileDialog(self)
        fileDialog.setOption(QFileDialog.ShowDirsOnly, True)
        name = fileDialog.getExistingDirectory(self, "Choose Recording Directory")
        (self.ppath, self.name) = os.path.split(name)        
        print "Setting base folder %s and recording %s" % (self.ppath, self.name)


    def set_tstep(self):
        self.tstep  = np.float(self.edit_tstep.text())


    def reset_tscale(self):
        """
        search for time point in self.tscale_ann that is closest to self.curr_time
        """
        i = np.argmin(np.abs(self.tscale_ann-self.curr_time))
        self.tscale_index = i
        self.index_list = [self.tscale_index]
        self.pcollect_index = False
        self.curr_time = self.tscale_ann[i]
        self.set_image_idx()
        self.set_image()
        self.pann_mode = True
        self.display_mode()
        

    def inc_tstep(self):                
        self.curr_time += self.tstep
        #self.image_idx = np.argmin(np.abs(self.frame_onset-self.curr_time))
        self.set_image_idx()


    def dec_tstep(self):
        self.curr_time -= self.tstep
        self.set_image_idx()


    def set_image_idx(self):
        # HERE could preallocated a tscale_frameidx array
        self.image_idx = np.argmin(np.abs(self.frame_onset-self.curr_time))


    def index_range(self):
        """
        for "space" processing
        """
        if len(self.index_list) == 1:
            return self.index_list
        a = self.index_list[0]
        b = self.index_list[-1]
        if a<=b:
            return range(a,b+1)
        else:
            return range(b,a+1)


    def keyPressEvent(self, event):
        k = event.key()
        #print k

        # cursor to the right
        if k == 16777236:
            if self.pann_mode == False:
                self.inc_tstep()
                self.plot_eeg()
                self.plot_treck()
            else:
                if self.tscale_index < (len(self.tscale_ann)-1):
                    self.tscale_index += 1
                    if self.pcollect_index:
                        self.index_list.append(self.tscale_index)
                    else:
                        self.index_list = [self.tscale_index]
                    self.Kann[self.tscale_index] = 1
                    self.curr_time = self.tscale_ann[self.tscale_index]
                    self.set_image_idx()
                    self.plot_annotation()
                    self.plot_eeg()
            self.set_image()
                
        #cursor to the left
        elif k == 16777234:
            if not self.pann_mode:
                self.dec_tstep() 
                self.plot_eeg()
                self.plot_treck()
            else:
                if self.tscale_index > 0:
                    self.tscale_index -= 1
                    if self.pcollect_index:
                        self.index_list.append(self.tscale_index)
                    else:
                        self.index_list = [self.tscale_index]
                    self.Kann[self.tscale_index] = 1
                    self.curr_time = self.tscale_ann[self.tscale_index]
                    self.set_image_idx()
                    self.plot_annotation()
                    self.plot_eeg()
            self.set_image()

        # cursor down
        elif event.key() == 16777237:
            self.color_max -= self.color_max / 10
            self.image_eegspec.setLevels((0, self.color_max))

        # cursor up
        elif event.key() == 16777235:
            self.color_max += self.color_max / 10
            self.image_eegspec.setLevels((0, self.color_max))

        # key /
        elif k == 47:
            if self.pann_mode==False and self.pconfig==True:            
                self.pann_mode = True
                self.reset_tscale()
                self.plot_eeg()
            else:
                self.pann_mode = False
                self.display_mode()

        # display annotation key
        elif self.pann_mode and (k in self.symbol_keys):
                key = self.symbol_dict[k]
                #self.symbol_array[self.tscale_index] = self.symbol_num[key]
                self.symbol_array[self.index_range()] = self.symbol_num[key]
                self.pcollect_index = False
                self.index_list = [self.tscale_index]
                self.plot_annotation()

        # delete - erase current behavioral annotation
        elif self.pann_mode and k == 48:
            print "deleting current behavior"
            self.symbol_array[self.tscale_index] = 0
            self.plot_annotation()

        # space
        elif event.key() == 32:
            self.pcollect_index = True
            self.index_list = [self.tscale_index]

        # "1" = save data
        elif self.pann_mode and k == QtCore.Qt.Key_1:
            self.save_annotation()

        # < - HERE
        elif event.key() == 44:
            num_eeg = len(self.EEG_list)
            if self.eeg_pointer < num_eeg-1:
                self.eeg_pointer += 1
            else:
                self.eeg_pointer = 0
            self.EEG = self.EEG_list[self.eeg_pointer]
            self.eeg_spec = self.eeg_spec_list[self.eeg_pointer]
            self.plot_eeg()
            self.image_eegspec.clear()
            self.image_eegspec.setImage(self.eeg_spec[0:self.ifreq,:].T)

        # >
        elif event.key() == 46:
            num_emg = len(self.EMG_list)
            if self.emg_pointer < num_emg-1:
                self.emg_pointer += 1
            else:
                self.emg_pointer = 0
            self.EMG = self.EMG_list[self.emg_pointer]
            self.EMGAmpl = self.emg_amp_list[self.emg_pointer]
            self.plot_eeg()
            self.graph_emgampl.clear()
            self.graph_emgampl.plot(self.ftime, self.EMGAmpl)

        elif k == QtCore.Qt.Key_H:
            self.print_help()


    def closeEvent(self, event):
        print("Closing...")
        if self.pconfig:
            self.save_annotation()


    def print_help(self):
        print """
        *** video_processor_stack.py ***
        usage

        to run start
        python video_precessor_stack.py -r path_to_recording [-c annotation_file]

        the parameter -c is optional

        The program runs in one of two modes: "View" or "Annotate" which is indicated
        in the lower right

        "View" mode allows for inspecting the video at a chosen time step (parameter tsetp in lower right)
        Make sure to press "set" to really set the time step

        In "Annotate" the user can adjust to each time point a single letter string identifying a specific behavior.
        The pool of possible strings (= behaviors) is defined in the edit box "Symbols" the corresponding
        annotation file is generated once "create" is clicked. Annotation files are within the recording folder
        and start with "vip_"

        After closing the program, the same annotation file can be reloaded using the "-c" option.

        key actions:
        :h print help
        :1 save
        :/ switch between view and annotation mode
        :0 delete annotation for current time point
        :, switch EEG channel
        :. switch EMG channel
        :cursor left, right - go to left, right
        :cursor up,down     - change color range of EEG spectrogram
        :space - starts an interval that will to annotated, once a symbol key is pressed; all states between
                 last space press and current time point will be annoated with the same symbol
        """



# some input parameter management
# we expect the following input sequence:
# python video_processor_stack -r recording/directory -c config_file                 
                
args = sys.argv[1:]
params = param_check(args)
if params.has_key('-r'):
    ddir = params['-r']
    if re.match('.*\/$', ddir):
        ddir = ddir[:-1]
    (ppath, name) = os.path.split(ddir)
else:
    ppath = ''
    name = ''
config_file = ''
if params.has_key('-c'):
    config_file = params['-c']
    a = os.path.split(config_file)
    if a[0] == '':
        config_file = os.path.join(ppath, name, config_file)


print "Starting video processing for recording %s with config file %s" % (name, config_file)
app = QtGui.QApplication([])
player = MainWindow(ppath, name, config_file)
player.show()
app.exec_()





