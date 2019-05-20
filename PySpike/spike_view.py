#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:54:17 2017

ToDo:

Check grp -> unit assignment
also make possible to click brain state and firing rate to change time point
update treck, when moving cursor
improve correspondence between clicked brainstate bin and shown EEG
plot all waveforms in view
plot autocorrelation

@author: tortugar
"""

import sys
sys.path.append('/home/lab/Documents/Data/Programming/PySleep')
#sys.path.append('/Users/tortugar/Google Drive/Penn/Programming/PySleep')

# exactly this combination of imports works with py2.7; otherwise problem importing
# pyqtgraph
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
import pyqtgraph as pg
import numpy as np
import scipy.io as so
import pyqtgraph.dockarea as dock
import os.path
import h5py
import sleepy
import spyke
import re
import pickle
from functools import reduce
import pdb


def firing_rate(ppath, name, grp, un, nbin):
    """
    y = firing_rate(ppath, name, grp, un, nbin) 
    """
    #sfile = os.path.join(ppath, name, 'Spk' + str(grp) + '.mat')
    
    #Spk = so.loadmat(sfile, struct_as_record=False, squeeze_me=True)['S']
    #train = l['S']['train'][0][un-1].toarray()[:,0]
    
    idx, train, mspike = unpack_unit(ppath, name, grp, un)
    #train = np.squeeze(Spk[un].train.toarray())
    #mspike = Spk[un].mspike
    #idx = Spk[un].idx
    
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
    files = os.listdir(os.path.join(ppath, name))
    files = [f for f in files if re.match('^Spk' + str(grp) + '-\d+\.npz$', f)]
    if len(files) > 0:
        spk_file = os.path.join(ppath, name, 'Spk' + str(grp) + '-' + str(un) + '.npz')
        Spk = np.load(spk_file, encoding='bytes')
        train = Spk['train']
        mspike = Spk['mspike']
        idx = Spk['idx']
        return idx, train, mspike

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
        #un += 1
        Spk = np.load(sfile + '.npz', encoding='bytes')
        print("Loaded %s" % sfile + '.npz')
        # explanation of why this is necessary:
        # https://stackoverflow.com/questions/22661764/storing-a-dict-with-np-savez-gives-unexpected-result/41566840
        Spk = {key:Spk[key].item() for key in Spk if Spk[key].dtype == 'O'}
        train = Spk['train'][un]
        mspike = Spk['mspike'][un]
        idx = Spk['idx'][un]
    return idx, train, mspike



def unpack_grp(ppath, name, grp):
    """
    return list holding the spike indices of each unit in group $grp
    """
    files = os.listdir(os.path.join(ppath, name))
    files = [f for f in files if re.match('^Spk' + str(grp) + '-', f)]
    if len(files ) > 0:
        units = []
        for f in files:
            a = re.findall('^Spk' + str(grp) + '-' + '(\d+)', f)
            units.append(a[0])
        #units = map(int, units)
        units = [int(i) for i in units]
        units.sort()
        idx = []
        for i in range(1, max(units)+1):
            if i in units:
                idx.append([1])
            else:
                idx.append([])
        return idx

    sfile = os.path.join(ppath, name, 'Spk' + str(grp))
    idx = []
    if os.path.isfile(sfile + '.mat'):
        Spk = so.loadmat(sfile, struct_as_record=False, squeeze_me=True)['S']
        for i in range(len(Spk)):
            idx.append(Spk[i].idx)
    elif os.path.isfile(sfile + '.npz'):
        Spk = np.load(sfile + '.npz', encoding='bytes')
        Spk = {key:Spk[key].item() for key in Spk if Spk[key].dtype == 'O'}
        SpikeIdx = Spk['idx']        
        for k in SpikeIdx:
            idx.append(SpikeIdx[k])

    return idx
    


def laser_start_end(laser, SR=1525.88, intval=5):
    """laser_start_end(laser, SR=TDT_sampling_rate, intval=5 [s]) ...
    print start and end index of laser stimulation periods
    returns the tuple (istart, iend), both indices are inclusive,
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



def downsample_matrix(X, nbin):
    """
    y = downsample_matrix(X, nbin)
    downsample the matrix X by replacing nbin consecutive 
    rows by their mean 
    @RETURN: the downsampled matrix 
    """
    n_down = int(np.floor(X.shape[0] / nbin))    
    X = X[0:n_down*nbin,:]
    X_down = np.zeros((n_down,X.shape[1]))

    # 0 1 2 | 3 4 5 | 6 7 8 
    for i in range(nbin) :
        idx = range(i, int(n_down*nbin), int(nbin))
        X_down += X[idx,:]

    return X_down / nbin
    


def laser_reliability(spikesd, laser, sr, win, iters, offs):
    
    dt = 1.0/sr
    idx,_ = laser_start_end(laser, SR=sr, intval=5)
    up_idx = np.where(np.diff(laser)>0)[0]+1
    if laser[0]>0:
        up_idx = np.concatenate(([0], up_idx))
    nwin = int(np.round(win/dt))
    # maximum length of laser pulse train
    
    M = []
    V = []
    for j in idx[offs::iters]:
        # j is start of a laser pulse train
        k = j+np.round(2/dt)
        roi = up_idx[np.where((up_idx>=j) & (up_idx<=k))[0]]
        V.append(np.diff(roi).mean()*dt)        
        m = []
        for r in roi:
            m.append(np.sum(spikesd[r:r+nwin+1]))
        M.append(m)
    
    l = min([len(i) for i in M])
    M = [i[0:l] for i in M]
    M = np.array(M)
    M[np.where(M>0)] = 1.0
    intv = np.mean(V)

    return M, intv        



# Object holding the GUI window
class MainWindow(QtGui.QMainWindow):
    
    def __init__(self, ppath, name):
        QtGui.QMainWindow.__init__(self)
        # sizing of Main Window
        self.setGeometry(QtCore.QRect(100, 0, 1500, 800))
   
        # setup names for recording folder / mouse     
        self.name = name
        self.ppath = ppath
        self.setWindowTitle(self.name)
        if self.name == '':
            self.openFileNameDialog()
        
        self.mouse = re.split('_', self.name)[0]
        
        self.load_grpfile()
        self.prune_grp()
        self.grp = list(self.grp_dict.keys())[0]
        self.un = self.grp_spike[self.grp][0]
        self.open_fid = []

        # firing rate, mean spike waveform, index of each spike, spike train in EEG time
        self.fr = []
        self.mspike = []
        self.sidx = []
        self.train = []
        
        self.twin      = 10     # duration of data loaded into memory
        self.index_fft = 20     # index in spectrogram 
        self.twin_view = 2      # shown data time range

        self.ndec = 16
        self.plot_point_mode = True

        # area is a docking area, where single docks can be added to
        self.area = dock.DockArea()      
        self.setCentralWidget(self.area)
        
        # Dock with dropdown menus and text boxes to select unit
        self.dock_unit = dock.Dock("Unit Selection & Annotation", size=(20, 10))
        self.area.addDock(self.dock_unit, 'left')
        
        # Dock with graphs to display if unit is driven
        self.dock_tagging = dock.Dock("Optogenetic Tagging", size=(20, 20))
        self.area.addDock(self.dock_tagging, 'bottom')
        
        
        self.unit_layout = pg.LayoutWidget()
        self.unit_layout.addWidget(QtGui.QLabel("Group"), row=0, col=0)      
        self.unit_layout.addWidget(QtGui.QLabel("Unit"), row=0, col=1)
        self.unit_layout.nextRow()
        
        # setup topdown menu
        self.topdown_grp = QtGui.QComboBox(self)
        #topdown.addItem("4")
        self.unit_layout.addWidget(self.topdown_grp, row=1, col=0)

        self.topdown_un = QtGui.QComboBox(self)
        #topdown.addItem("4")
        self.unit_layout.addWidget(self.topdown_un, row=1, col=1)        
             
        
        self.button_plot = QtGui.QPushButton('Plot')
        self.button_clear = QtGui.QPushButton('Clear')        
        self.unit_layout.addWidget(self.button_plot, row=2, col=0)
        self.unit_layout.addWidget(self.button_clear, row=2, col=1)
        self.button_plot.clicked.connect(self.plot_fr) 
        self.button_clear.clicked.connect(self.clear_fr) 
        
        self.topdown_trigger = QtGui.QComboBox(self)
        self.topdown_trigger.addItem("train")
        self.topdown_trigger.addItem("pulse")
        self.topdown_trigger.activated.connect(self.setup_trigger_defaults)  
        self.unit_layout.addWidget(self.topdown_trigger, row=3, col=0)
        self.button_trigger = QtGui.QPushButton('Trigger')
        self.button_trigger.clicked.connect(self.trigger_spikes)
        self.unit_layout.addWidget(self.button_trigger, row=3, col=1)
        self.unit_layout.addWidget(QtGui.QLabel("pre"), row=4, col=0)
        self.unit_layout.addWidget(QtGui.QLabel("post"), row=4, col=1)
        self.unit_layout.addWidget(QtGui.QLabel("nbin"), row=4, col=2)
        self.edit_pre = QtGui.QLineEdit('0.1')
        self.unit_layout.addWidget(self.edit_pre, row=5, col=0)
        self.edit_post = QtGui.QLineEdit('1.1')
        self.unit_layout.addWidget(self.edit_post, row=5, col=1)
        self.edit_nbin = QtGui.QLineEdit('3')
        self.unit_layout.addWidget(self.edit_nbin, row=5, col=2)
        
        self.unit_layout.addWidget(QtGui.QLabel("offs"), row=6, col=0)
        self.unit_layout.addWidget(QtGui.QLabel("iter"), row=6, col=1)
        
        self.edit_offset = QtGui.QLineEdit('0')
        self.unit_layout.addWidget(self.edit_offset, row=7, col=0)
        self.edit_iter = QtGui.QLineEdit('1')
        self.unit_layout.addWidget(self.edit_iter, row=7, col=1)
                             
        # Widgets for Unitt Annotation
        self.unit_layout.addWidget(QtGui.QLabel("Annotation"), row=8, col=0)
        self.unit_layout.addWidget(QtGui.QLabel("Qual."), row=8, col=2)
        
        self.button_add = QtGui.QPushButton('Add')               
        self.unit_layout.addWidget(self.button_add, row=9, col=0)
        self.button_add.clicked.connect(self.add_unit_annotation)

        self.button_del = QtGui.QPushButton('Del') 
        self.unit_layout.addWidget(self.button_del, row=9, col=1)
        self.button_del.clicked.connect(self.del_unit_annotation)
                
        self.topdown_qual = QtGui.QComboBox(self)
        self.topdown_qual.addItem("5")
        self.topdown_qual.addItem("4")
        self.topdown_qual.addItem("3")
        self.topdown_qual.addItem("2")
        self.topdown_qual.addItem("1")
        self.unit_layout.addWidget(self.topdown_qual, row=9, col=2)
        
        self.unit_layout.addWidget(QtGui.QLabel("Comment"), row=10, col=0)
        self.unit_layout.addWidget(QtGui.QLabel("Typ"), row=10, col=1)
        self.unit_layout.addWidget(QtGui.QLabel("Id."), row=10, col=2)

        self.edit_comment = QtGui.QLineEdit('')
        self.unit_layout.addWidget(self.edit_comment, row=11, col=0)
        
        self.edit_typ = QtGui.QLineEdit('')
        self.unit_layout.addWidget(self.edit_typ, row=11, col=1)
        
        self.topdown_driven = QtGui.QComboBox(self)
        self.topdown_driven.addItem("0")
        self.topdown_driven.addItem("1")
        self.topdown_driven.addItem("-1")
        self.unit_layout.addWidget(self.topdown_driven, row=11, col=2)
        
        
        # dock for data
        #self.dock_data = dock.Dock("", size=(200, 100))
        self.dock_eeg = dock.Dock("Whole Session", size=(100, 600))
        self.dock_raw = dock.Dock("Spiking Data", size=(100, 800))
                
        # add dock_data to the main dock 'area'
        #self.area.addDock(self.dock_data, 'left')
        # then add subdocks to dock_data
        self.area.addDock(self.dock_eeg, 'left')
        self.area.addDock(self.dock_raw, 'bottom', self.dock_eeg)
        
                
        # raw channel plots
        #self.layout_raw = pg.LayoutWidget()
        #self.dock_raw.addWidget(self.layout_raw)
        
        # Hypnogram, EEG spectrogram and EMG amplitude
        #self.layout_eeg = pg.LayoutWidget()
        
        
        # graph to display spike waveform
        self.graph_spike = pg.PlotWidget()
        self.graph_spike.hideAxis('left')
        self.graph_spike.hideAxis('bottom')

        self.dock_unit.addWidget(self.unit_layout)
        #self.dock_unit.addWidget(self.graph_spike)
        self.dock_spike = dock.Dock("Waveform", (100, 800))
        
        self.dock_spike = dock.Dock('Waveform', (100,200))
        self.dock_spike.addWidget(self.graph_spike)
        self.area.moveDock(self.dock_spike, 'above', self.dock_tagging)
        
        # show the whole GUI
        self.show()
                
        # setup data
        self.graph_channels = []
        self.load_session()
        self.load_data()

        self.plot_channels()
        
        # setup topdown menus to chose grp and unit
        self.setup_topdown_grp()
        self.setup_topdown_un()
        
        self.setup_trigger_plots()


    def openFileNameDialog(self):    
        fileDialog = QFileDialog(self)
        fileDialog.setOption(QFileDialog.ShowDirsOnly, True)
        name = fileDialog.getExistingDirectory(self, "Choose Recording Directory")
        (self.ppath, self.name) = os.path.split(name)        
        print("Setting base folder %s and recording %s" % (self.ppath, self.name))

                
    def add_unit_annotation(self):


        self.grp = int(str(self.topdown_grp.currentText()))        
        self.un = int(str(self.topdown_un.currentText()))

        spyke.del_unit_annotation(self.ppath, self.name, self.grp, self.un)


        comment  = str(self.edit_comment.text())
        driven = int(str(self.topdown_driven.currentText()))
        quality = int(str(self.topdown_qual.currentText()))
        typ =  (str(self.edit_typ.text()))
        
        spyke.add_unit_annotation(self.ppath, self.name, self.grp, self.un, typ, driven, quality, comment = comment)
        
        
    def del_unit_annotation(self):
        self.grp = int(str(self.topdown_grp.currentText()))        
        self.un = int(str(self.topdown_un.currentText()))
        
        spyke.del_unit_annotation(self.ppath, self.name, self.grp, self.un)
        

    def close_old_grp(self):
        if len(self.open_fid) > 0:
            [f.close() for f in self.open_fid]
            self.open_fid = []
        
        
    def load_data(self):
        """
        load raw channels;
        a raw channel is a (n,1) array
        """
        self.nchannels = len(self.grp_dict[self.grp])
        self.setup_raw_plots()     
        # load channels
        self.ch = {}
        for c in self.grp_dict[self.grp]:
            fid = h5py.File(os.path.join(self.ppath, self.name, 'ch_' + self.name + '_%d.mat' % c), 'r')
            self.ch[c] = fid['D']
            self.open_fid.append(fid)


                
    def setup_topdown_grp(self):
        # make sure self.load_grpfile has been called before
        for k in self.grp_dict:
            self.topdown_grp.addItem(str(k))
        #self.topdown_grp.activated.connect(self.setup_topdown_un)    
        self.topdown_grp.activated.connect(self.set_new_grp)    
    
    
    def set_new_grp(self):
        self.close_old_grp()
        self.grp = int(str(self.topdown_grp.currentText()))

        self.graph_eeg.close()
        [g.close() for g in self.graph_channels]
        self.graph_channels = []

        self.load_data()

        self.plot_channels()
        self.setup_topdown_un()
    
    
    def setup_topdown_un(self):
        # update units topdown menu
        #self.grp = int(str(self.topdown_grp.currentText()))
        self.topdown_un.clear()
        for i in self.grp_spike[self.grp]:
            self.topdown_un.addItem(str(i))
        # need to reload channel data?
        #self.load_data()
        #self.topdown_un.activated.connect(self.plot_fr)    
                    
    

    def setup_trigger_defaults(self):
        trigger = self.topdown_trigger.currentText()
        print("trigger pressed")
        if trigger == 'train':
            self.edit_pre.setText('0.1')
            self.edit_post.setText('1.1')
            self.edit_nbin.setText('3')
        else:
            self.edit_pre.setText('0.01')
            self.edit_post.setText('0.03')
            self.edit_nbin.setText('1')
            
        
    
    def mouse_pressed(self, evt):
        mousePoint = self.graph_eegspec.getViewBox().mapToView(evt.pos())  
                
        self.tscale = 1
        #pdb.set_trace()
        # update current time point
        self.index_fft = int((mousePoint.x()/(self.fdt*self.tscale)))
        self.index_eeg = self.index_fft*self.fbin
        self.index_raw = self.index_eeg*self.ndec
        self.current_t = self.index_eeg*self.dt
        self.current_t_pressed = self.current_t
        ax = self.graph_channels[0].getAxis(name='bottom')
        self.twin_view = ax.range[-1] - ax.range[0]

        self.plot_treck()
        self.plot_channels()
    
                
    def load_session(self):
        """
        load and display brsainstate, EEG spectrogram, EMG amplitude
        """
        # define sampling rates and discrete time bins
        self.SR_eeg = sleepy.get_snr(self.ppath, self.name)
        self.SR_raw = self.SR_eeg*self.ndec
        self.dt = 1/self.SR_eeg
        self.ddt = self.dt/self.ndec
        self.fbin = np.round((1/self.dt) * 2.5) # number of sampled point for one fourier bin

        self.index_eeg = self.index_fft*self.fbin
        self.index_raw = self.index_eeg*self.ndec
        self.current_t = self.index_eeg*self.dt
        self.current_t_pressed = self.current_t        

        # setup PlotWidgets
        self.graph_treck      = pg.PlotWidget()
        self.graph_brainstate = pg.PlotWidget()
        self.graph_eegspec    = pg.PlotWidget()
        self.graph_emgampl    = pg.PlotWidget()
        
        self.dock_eeg.addWidget(self.graph_treck, row=0, col=0, rowspan=1)
        self.dock_eeg.addWidget(self.graph_brainstate, row=1, col=0)
        self.dock_eeg.addWidget(self.graph_eegspec, row=2, col=0)
        self.dock_eeg.addWidget(self.graph_emgampl, row=3, col=0)
        
        # load data ###########################################################
        # load EEG/EMG        
        self.eeg_pointer = 0
        self.EEG_list = []
        EEG1 = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EEG.mat'))['EEG'])
        self.EEG_list.append(EEG1)
        # if existing, also load EEG2
        if os.path.isfile(os.path.join(self.ppath, self.name, 'EEG2.mat')):
            EEG2 = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EEG2.mat'))['EEG2'])
            self.EEG_list.append(EEG2)
        self.EEG = self.EEG_list[0]
        self.EMG = np.squeeze(so.loadmat(os.path.join(self.ppath, self.name, 'EMG.mat'))['EMG'])

        # load spectrogram / EMG amplitude
        if not(os.path.isfile(os.path.join(self.ppath, self.name, 'sp_' + self.name + '.mat'))):
            # spectrogram does not exist, generate it
            sleepy.calculate_spectrum(self.ppath, self.name, fres=0.5)
            print("Calculating spectrogram for recording %s\n" % self.name)
        
        spec = so.loadmat(os.path.join(self.ppath, self.name, 'sp_' + self.name + '.mat'))
        self.ftime = spec['t'][0]
        self.fdt = spec['dt'][0][0]
        self.eeg_spec = spec['SP']
        freq = np.squeeze(spec['freq'])
        self.ifreq = np.where(freq <= 25)[0][-1]
        self.fdx = freq[1]-freq[0]
        self.mfreq = np.where((freq>=10) & (freq <= 200))[0]
        
        self.emg_spec = so.loadmat(os.path.join(self.ppath, self.name, 'msp_' + self.name + '.mat'))['mSP']
        self.EMGAmpl = np.sqrt(self.emg_spec[self.mfreq,:].sum(axis=0))
        self.nbin = len(self.ftime) #number of bins in fourier time


        # load LFPs
        self.lfp_pointer = -1
        self.LFP_list = []
        lfp_files = [f for f in os.listdir(os.path.join(self.ppath, self.name)) if re.match('^LFP', f)]
        lfp_files.sort()
        if len(lfp_files) > 0:
            self.lfp_pointer = -1
            for f in lfp_files:
                key = re.split('\\.', f)[0]
                LFP = so.loadmat(os.path.join(self.ppath, self.name, f), squeeze_me=True)[key]
                self.LFP_list.append(LFP)

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

        # load thresholds
        self.pthres_mode = False
        self.thres_pointer = 0
        if not os.path.isfile(os.path.join(self.ppath, self.name, 'thres.p')):
            print("Calculating thresholds")
            spyke.save_threshold(self.ppath, self.name)
        with open(os.path.join(self.ppath, self.name, 'thres.p'), 'rb') as fp:
            # python 3 migration issue UnicodeDecodeError
            try:
                tmp = pickle.load(fp)
            except UnicodeDecodeError:
                # to load pickle objects created using python 2
                tmp = pickle.load(fp, encoding='latin1')
            print(tmp)
            self.thres = tmp['Thres']
            fp.close()
        ### END load data #####################################################
        
        
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
                
        self.image_brainstate = pg.ImageItem() 
        self.graph_brainstate.addItem(self.image_brainstate)                
        self.image_brainstate.setImage(self.M.T)
        
        scale = 1
        self.image_brainstate.scale(self.fdt*scale,1)
        self.graph_brainstate.setMouseEnabled(x=True, y=False)
        #self.graph_brainstate.setXLink(self.graph_eegspec)
        limits = {'xMin': -1*self.fdt*scale, 'xMax': self.ftime[-1]*scale, 'yMin': 0, 'yMax': 1}
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
        self.image_eegspec.scale(self.fdt*scale, 1.0*self.fdx)
        
        # mousescroll only allowed along x axis
        self.graph_eegspec.setMouseEnabled(x=True, y=False)
        limits = {'xMin': -1*self.fdt*scale, 'xMax': self.ftime[-1]*scale, 'yMin': 0, 'yMax': 20}
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
        self.graph_emgampl.plot(self.ftime*scale, self.EMGAmpl)
        self.graph_emgampl.setMouseEnabled(x=False, y=True)
        self.graph_emgampl.setXLink(self.graph_eegspec)
        limits = {'xMin': 0, 'xMax': self.ftime[-1]*scale}
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

        #proxy = pg.SignalProxy(self.graph_eegspec.scene().sigMouseClicked, rateLimit=60, slot=self.mouse_pressed)
        self.graph_eegspec.scene().sigMouseClicked.connect(self.mouse_pressed)
        self.graph_brainstate.scene().sigMouseClicked.connect(self.mouse_pressed)
        self.graph_treck.scene().sigMouseClicked.connect(self.mouse_pressed)

        self.plot_treck()
        


    def plot_treck(self, scale=1):     
        """
        plot laser, currently visited state and annotated states
        """
        self.graph_spike.clear()
        self.graph_spike.hideAxis('bottom')
        self.graph_spike.hideAxis('left')
        self.graph_treck.clear()
        self.graph_treck.setMouseEnabled(x=False, y=True)
 
        limits = {'xMin': 0*self.fdt*scale, 'xMax': self.ftime[-1]*scale} #, 'yMin': -0.2, 'yMax': 1.1}
        self.graph_treck.setLimits(**limits)
                                
        ax = self.graph_treck.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('FR', units='Hz', **labelStyle)
        #ax.setTicks([[(0, ''), (1, '')]])
        ax = self.graph_treck.getAxis(name='bottom')
        ax.setTicks([[]])
        self.graph_treck.plot([self.ftime[self.index_fft]*scale + 0.5*self.fdt*scale], [0.0], pen=(0,0,0), symbolPen='w')
                
        # plot firing rates
        icolor = 0
        colors = [(255,255,0), (255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        mfr = [1]
        maxfr = []
        if len(self.fr)>0:
            for (fr, mspike) in zip(self.fr, self.mspike):
                # plot firing rate
                self.graph_treck.plot(self.ftime*scale + 0.5*self.fdt*scale, fr, pen=colors[icolor])
                # plot laser
                mfr.append(fr.mean())
                maxfr.append(fr.max())
                
                # plot mean spike wave form
                spike_len = len(mspike)
                t = np.arange(-spike_len/2+1,spike_len/2+1)*(1.0/self.SR_raw)*1000
                self.graph_spike.plot(t, mspike, pen=colors[icolor], padding=0)
                self.graph_spike.showAxis('left')
                self.graph_spike.showAxis('bottom')
                ax = self.graph_spike.getAxis(name='bottom')
                labelStyle = {'color': '#FFF', 'font-size': '10pt'}
                ax.setLabel('Time', units='ms', **labelStyle)

                icolor += 1
                if icolor >= len(colors):
                    icolor=0

        self.graph_treck.plot(self.ftime*scale + 0.5*self.fdt*scale, self.laser_dn*max(mfr), pen=(0,0,255))
        if len(mfr) == 1:
            self.graph_treck.setYRange(0, 1.1)
        else:
            self.graph_treck.setYRange(0, max(maxfr))
        

    def setup_raw_plots(self):    
        """
        setup graphs and basic functionality for dock_raw
        """
        # setup PlotWidget for EEG
        self.graph_eeg = pg.PlotWidget()
        ax = self.graph_eeg.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('EEG' + ' ' + str(self.eeg_pointer+1), units='uV', **labelStyle)
        ax = self.graph_eeg.getAxis(name='bottom')
        ax.setTicks([[]])

        self.dock_raw.addWidget(self.graph_eeg)
        
        # setup a PlotWidget for each group channel
        for i in range(self.nchannels):
            tmp = pg.PlotWidget()
            ax = tmp.getAxis(name='left')
            labelStyle = {'color': '#FFF', 'font-size': '10pt'}
            ax.setLabel('Ch%d' % self.grp_dict[self.grp][i], units='uV', **labelStyle)
            #ax.enableAutoSIPrefix(enable=False)
            if i < self.nchannels-1:
                ax = tmp.getAxis(name='bottom')
                ax.setTicks([[]])

            self.dock_raw.addWidget(tmp)
            self.graph_channels.append(tmp)
                
        ax = self.graph_channels[-1].getAxis(name='bottom')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('Time', units='s', **labelStyle)
            
        for i in range(self.nchannels-1):
            self.graph_channels[i].setXLink(self.graph_channels[i+1])

        self.graph_eeg.setXLink(self.graph_channels[0])

        self.plot_thres = {c: None for c in range(self.nchannels)}



    def setup_trigger_plots(self):
        # raster plot
        self.graph_tagging_raster = pg.PlotWidget()
        self.dock_tagging.addWidget(self.graph_tagging_raster)
        
        self.graph_tagging_rel = pg.PlotWidget()
        self.dock_tagging.addWidget(self.graph_tagging_rel)
        
        # firing rate plot
        self.graph_tagging_fr = pg.PlotWidget()
        self.dock_tagging.addWidget(self.graph_tagging_fr)
        
        self.graph_tagging_raster.setXLink(self.graph_tagging_fr)
        self.graph_tagging_rel.setXLink(self.graph_tagging_fr)
        

    def plot_channels(self):
        """
        plot EEG along with raw channels of the current group
        """
        colors = [(255,255,0), (255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        
        ii_eeg = [self.index_eeg-np.round(0.5*self.twin/self.dt), self.index_eeg+np.round(0.5*self.twin/self.dt)]
        ii_eeg = [int(i) for i in ii_eeg]
        ii_raw = [int(i*self.ndec) for i in ii_eeg]
                
        for (graph,c) in zip(self.graph_channels, self.grp_dict[self.grp]):
            graph.clear()
            
            y = self.ch[c][int(ii_raw[0]):int(ii_raw[1]),0]
            t_raw = np.arange(0, len(y))*self.ddt + self.current_t-self.twin/2.0
            graph.plot(t_raw, y)
            
            # plot laser            
            z = self.laser[ii_eeg[0]:ii_eeg[1]]*0.25*self.thres[c]
            # self.t_eeg is first initialized here
            self.t_eeg = np.arange(0, len(z))*self.dt + self.current_t - self.twin/2.0
            graph.plot(self.t_eeg, z, pen=(0,0,255))
            graph.setXRange(self.current_t-self.twin_view/2, self.current_t+self.twin_view/2, padding=0)
            
            icolor=0
            if len(self.fr) > 0:
                for sidx in self.sidx:
                    # indices of spikes within the shown time window
                    sel_idx = sidx[np.where((sidx >= ii_raw[0]+10) & (sidx<ii_raw[-1]-10))[0]]-ii_raw[0]
                    if len(sel_idx) > 0:
                        if self.plot_point_mode:
                            graph.plot(t_raw[sel_idx], y[sel_idx], pen=None, symbol='o', symbolPen=colors[icolor], symbolBrush=colors[icolor], symbolSize=5)
                        else:                        
                            s = reduce(lambda x,y:x+y, [range(i-10,i+10) for i in sel_idx])
                            e = reduce(lambda x,y:x+y, [[i-10,i+9] for i in sel_idx])                        
                            graph.plot(t_raw[s], y[s], pen=colors[icolor])
                            graph.plot(t_raw[e], y[e], pen=(0,0,0))
                        icolor += 1

        self.graph_eeg.clear()
        y = self.EEG[ii_eeg[0]:ii_eeg[1]]
        #t_eeg = np.arange(0, len(y))*self.dt + self.current_t - self.twin/2.0
        self.graph_eeg.plot(self.t_eeg, y)
        
        if self.pthres_mode:
            self.plot_threshold()


    def shift_channel_view(self):             
        ax = self.graph_eeg.getAxis(name='bottom')
        self.twin_view = ax.range[-1] - ax.range[0]        
        #print self.twin_view
        self.graph_eeg.setXRange(self.current_t-self.twin_view/2, self.current_t+self.twin_view/2, padding=0)

        if self.pthres_mode:
            self.plot_threshold()



    def plot_threshold(self, pplot=True):
        """
        plot thresholds
        """
        i = 0
        for graph in self.graph_channels:
            if self.plot_thres[i]:
                graph.removeItem(self.plot_thres[i])

            if pplot:
                if i != self.thres_pointer:
                    k = self.grp_dict[self.grp][i]
                    self.plot_thres[i] = graph.plot(self.t_eeg, -1 * np.ones((self.t_eeg.shape[0],))*self.thres[k], pen=(255, 150, 150))
            i += 1

        if pplot:
            k = self.grp_dict[self.grp][self.thres_pointer]
            graph = self.graph_channels[self.thres_pointer]
            self.plot_thres[self.thres_pointer] = graph.plot(self.t_eeg,
                                                             -1 * np.ones((self.t_eeg.shape[0],)) * self.thres[k],
                                                             pen=(255, 0, 0))

    
    def trigger_spikes(self):
        """
        calculate laser pulse triggered spikes;
        """
        if sum(self.laser) == 0:
            return
        
        val = str(self.topdown_un.currentText())
        if not(re.match('^\d+$', val)):
            return

        trigger = self.topdown_trigger.currentText()

        # get pre, post, nbin
        pre  = np.float(self.edit_pre.text())
        post = np.float(self.edit_post.text())
        nbin = np.int(self.edit_nbin.text())
        offs = np.int(self.edit_offset.text())
        iters = np.int(self.edit_iter.text())        
        pre  = int(np.round(pre*self.SR_eeg))
        post = int(np.round(post*self.SR_eeg))
        
        # if there's no spike train, the plot the currently selected unit
        if len(self.train) == 0:
            self.plot_fr()
        # get currently selected spike train
        train = self.train[-1]
        len_train = len(train)
        raster = []
        if trigger == 'train':
            # get indices of each laser train onset
            idx, _ = laser_start_end(self.laser)
            for i in idx[offs::iters]:
                if (i-pre>=0) and (i+post<len_train):
                    raster.append(train[i-pre:i+post+1])
        else:
            idx = np.where(np.diff(self.laser)>0)[0]+1
            if self.laser[0]>0:
                idx = np.concatenate(([0], idx))
            for i in idx:
                if (i-pre>=0) and (i+post<len_train):
                    raster.append(train[i-pre:i+post+1])

            nwin = int(np.round(0.02*self.SR_eeg))
            M = np.array(raster)[:,pre:pre+nwin+1].sum(axis=1)                
            t = np.arange(0, M.shape[0])*(1.0/self.SR_eeg) 
            M[np.where(M>0)] = 1
        
        # time x trials
        raster = np.array(raster).T        
        raster = downsample_matrix(raster, nbin)        
        dt = nbin*1.0/self.SR_eeg
        t = np.arange(0, raster.shape[0])*dt-pre*(1.0/self.SR_eeg) 
        if trigger == 'pulse':
            delay = raster.mean(axis=1)
            rt = [t[np.argmax(delay)]]
                        
        # calculate reliability
        if trigger == 'train':
            M, intv = laser_reliability(train, self.laser, self.SR_eeg, 0.02, iters, offs)
            rt = np.arange(0, M.shape[1])*intv
                
        # plot raster
        image_raster = pg.ImageItem() 
        self.graph_tagging_raster.clear()
        self.graph_tagging_raster.addItem(image_raster)
        image_raster.setImage(raster)
        image_raster.scale(dt, 1.0)
        image_raster.setPos(-pre/self.SR_eeg, 0)
        ax = self.graph_tagging_raster.getAxis(name='bottom')
        ax.setTicks([[]])

        # y-axis
        ax = self.graph_tagging_raster.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('Trial', units='No.', **labelStyle)

        # plot reliability
        self.graph_tagging_rel.clear()
        rel_mean = M.mean(axis=0)
        if len(np.shape(rel_mean)) == 0:
            rel_mean = [rel_mean]
            self.graph_tagging_rel.plot(rt, rel_mean, symbolPen='w')            
        else:
            self.graph_tagging_rel.plot(rt, rel_mean)
            
        ax = self.graph_tagging_rel.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('Rel.', units='', **labelStyle)                       
        self.graph_tagging_rel.setXRange(t[0], t[-1])
        self.graph_tagging_rel.setYRange(0, 1)
        ax = self.graph_tagging_rel.getAxis(name='bottom')
        ax.setTicks([[]])

        # plot firing rate
        self.graph_tagging_fr.clear()
        self.graph_tagging_fr.plot(t, raster.mean(axis=1))
        # y-axis
        ax = self.graph_tagging_fr.getAxis(name='left')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('FR', units='Hz', **labelStyle)
        # x-axis        
        ax = self.graph_tagging_fr.getAxis(name='bottom')
        labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        ax.setLabel('Time', units='s', **labelStyle)
    

    def plot_fr(self):
        self.load_fr()
        self.plot_treck()
        self.plot_channels()
        

    def load_fr(self):
        val = str(self.topdown_un.currentText())
        if re.match('^\d+', val):
            self.un = int(val)
            fr, mspike, sidx, train = firing_rate(self.ppath, self.name, self.grp, self.un, int(self.fbin))
            if len(fr) != len(self.ftime):
                nd = len(self.ftime) - len(fr)
                if nd > 0:
                    fr = np.concatenate((fr, np.zeros((nd,))))
                else:
                    self.ftime = np.concatenate((self.ftime, np.zeros((int(-1.0*nd),))))
            
            self.fr.append(fr)
            self.mspike.append(mspike)
            self.sidx.append(sidx)
            self.train.append(train)        

                        
    def clear_fr(self):
        self.fr = []
        self.mspike = []
        self.sidx = []
        self.train = []
        self.plot_treck()
        self.plot_channels()

      
    def keyPressEvent(self, event):
        #print(event.key())
        
        # cursor to the right
        if event.key() == 16777236:
            self.current_t += self.twin_view/4.0                        
            if (self.current_t+self.twin_view/2) > (self.current_t_pressed+self.twin/2):
                
                # update current time point
                self.index_fft = int(np.round(self.current_t / self.fdt))
                self.index_eeg = self.index_fft*self.fbin
                self.index_raw = self.index_eeg*self.ndec
                self.current_t = self.index_eeg*self.dt
                self.current_t_pressed = self.current_t
                ax = self.graph_channels[0].getAxis(name='bottom')
                self.twin_view = ax.range[-1] - ax.range[0]
                
                self.plot_treck()
                self.plot_channels()
            
            else:
                self.shift_channel_view()
        
        #cursor to the left
        elif event.key() == 16777234:
            self.current_t -= self.twin_view/4.0
                        
            if (self.current_t-self.twin_view/2) < (self.current_t_pressed-self.twin/2):

                # update current time point
                self.index_fft = int(np.round(self.current_t / self.fdt))
                self.index_eeg = self.index_fft*self.fbin
                self.index_raw = self.index_eeg*self.ndec
                self.current_t = self.index_eeg*self.dt
                self.current_t_pressed = self.current_t
                ax = self.graph_channels[0].getAxis(name='bottom')
                self.twin_view = ax.range[-1] - ax.range[0]
                
                self.plot_treck()
                self.plot_channels()
            else:
                self.shift_channel_view()        
        
        # cursor down
        elif event.key() == 16777237:
            self.color_max -= self.color_max/10
            self.image_eegspec.setLevels((0, self.color_max))
        
        # cursor up
        elif event.key() == 16777235:
            self.color_max += self.color_max/10
            self.image_eegspec.setLevels((0, self.color_max))

        # l - turn on lfp channel
        elif event.key() == 76:
            self.eeg_pointer = -1
            if len(self.LFP_list) > 0:
                num_lfp = len(self.LFP_list)
                if self.lfp_pointer < num_lfp-1:
                    self.lfp_pointer += 1
                else:
                    self.lfp_pointer = 0
                self.EEG = self.LFP_list[self.lfp_pointer]
                self.plot_channels()
                # set correct y-label for LFP signal
                ax = self.graph_eeg.getAxis(name='left')
                labelStyle = {'color': '#FFF', 'font-size': '10pt'}
                ax.setLabel('LFP' + ' ' + str(self.eeg_pointer+1), units='uV', **labelStyle)

        #e - switch EEG channel
        elif event.key() == 69:
            self.lfp_pointer = -1
            num_eeg = len(self.EEG_list)
            if self.eeg_pointer < num_eeg - 1:
                self.eeg_pointer += 1
            else:
                self.eeg_pointer = 0

            self.EEG = self.EEG_list[self.eeg_pointer]
            self.plot_channels()
            # set correct y-label for EEG signal
            ax = self.graph_eeg.getAxis(name='left')
            labelStyle = {'color': '#FFF', 'font-size': '10pt'}
            ax.setLabel('EEG' + ' ' + str(self.eeg_pointer+1), units='uV', **labelStyle)

        # switch to threshold mode
        elif event.key() == QtCore.Qt.Key_T:
            if not self.pthres_mode:
                self.pthres_mode = True
                self.plot_threshold()
            else:
                self.pthres_mode = False
                self.plot_threshold(pplot=False)

        # change "focus" of threshold
        elif event.key() == QtCore.Qt.Key_N:
            #self.plot_thres[self.thres_pointer] = None
            if self.thres_pointer < self.nchannels-1:
                self.thres_pointer += 1
            else:
                self.thres_pointer = 0
            self.plot_threshold()

        elif event.key() == QtCore.Qt.Key_U:
            if self.pthres_mode:
                k = self.grp_dict[self.grp][self.thres_pointer]
                self.thres[k] *= 0.95
                self.plot_threshold()

        elif event.key() == QtCore.Qt.Key_D:
            if self.pthres_mode:
                k = self.grp_dict[self.grp][self.thres_pointer]
                self.thres[k] *= 1.05
                self.plot_threshold()

        # s - save threshold
        elif event.key() == QtCore.Qt.Key_S:
            self.write_thresholds()

        elif event.key() == QtCore.Qt.Key_I:
            self.print_info()

        # h - help
        elif event.key() == QtCore.Qt.Key_H:
            self.print_help()


    def load_grpfile(self):
        """
        setup dictionary mapping group number onto channels belonging to this group
        Grp -> Channels
        """        
        grpfile = os.path.join(self.ppath, self.mouse + '_grouping.txt')
        
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
                if count <= 2:
                    continue
                else:                
                    a = re.findall('(\d+)', l)         
                    a = [int(i) for i in a]
                    grp[count-2] = a                                                
        self.grp_dict = grp
        
        # dictionary mapping grp id onto unit ids        
        self.grp_spike = {}
        for g in self.grp_dict:
            #sfile = os.path.join(self.ppath, self.name, 'Spk' + str(g))
            files = os.listdir(os.path.join(self.ppath, self.name))
            files = [f for f in files if re.match('^Spk' + str(g), f)]

            if len(files) > 0:
            #if os.path.isfile(sfile + '.mat') or os.path.isfile(sfile + '.npz'):
                #Spk = so.loadmat(os.path.join(self.ppath, self.name, 'Spk' + str(g) + '.mat'), struct_as_record=False, squeeze_me=True)['S']
                #un = []
                #for i in range(len(Spk)):
                #    if len(Spk[i].idx) > 0:
                #        un.append(i)
                #    else:
                #        un.append('empty')
                #
                un = []
                # we start counting units from 1:
                # 1 - noise
                # 2 - first cluster etc.
                i = 1
                SpkIdx = unpack_grp(self.ppath, self.name, g)
                for idx in SpkIdx:
                    if len(idx) > 0:
                        un.append(i)                        
                    else:
                        un.append('empty')
                    i += 1
                
                self.grp_spike[g] = un
            else:
                # 'none' means there's really no unit in this group g
                self.grp_spike[g] = ['none']


    def prune_grp(self):
        # throw away channels in grp_dict that do not exist; 
        # Then throw away groups without channels
        new_dict = {}
        for g in self.grp_dict:
            for ch in self.grp_dict[g]:
                if not(os.path.isfile(os.path.join(self.ppath, self.name, 'ch_' + self.name + '_' + str(ch) + '.mat'))):
                    print("removing channels %d = file %s" % (ch, os.path.join(self.ppath, self.name, 'ch_' + self.name + '_' + str(ch) + '.mat')))
                else:
                    if g in new_dict:
                        new_dict[g].append(ch)
                    else:
                        new_dict[g] = [ch]
        
        self.grp_dict = new_dict


    def print_help(self):
        print ("""

        key actions:
        h       -       print help
        i       -       print info file
        e       -       switch between EEG channels
        l       -       switch between LFP channels
        t       -       show thresholds
        n       -       change focus on thresholds
        u       -       move threshold up
        d       -       move threshold down
        s       -       save thresholds

        mouse actions:
        scroll on spectrogram: zoom in / out
        scroll on EMG Ampl.: Change Amplitude of EMG
        scroll on EEG,EMG: change amplitude
        left-click on spectrum: jump to clicked time point
        """)

    def print_info(self):
        fid = open(os.path.join(self.ppath, self.name, 'info.txt'), 'rU')
        lines = fid.readlines()
        fid.close()

        print("")
        print("Content of info.txt file of recording %s." % self.name)
        for line in lines:
            if not re.match('^\s+$', line):
                print(line)


    def write_thresholds(self):
        # python 3 migration issue: changed 'w' to 'wb'
        with open(os.path.join(self.ppath, self.name, 'thres.p'), 'wb') as fp:
            print("Saving thresholds")
            pickle.dump({'Thres':self.thres}, fp)
            fp.close()


# some input parameter management
params = sys.argv[1:]
if (len(params) == 0) :
    ppath = ''
    name = ''
elif len(params) == 1:
    if re.match('.*\/$', params[0]):
        params[0] = params[0][:-1]
    (ppath, name) = os.path.split(params[0])      
else:
    ppath = params[0]
    name  = params[1]

print(name)
app = QtGui.QApplication([])
w = MainWindow(ppath, name)
w.show()
app.exec_()

