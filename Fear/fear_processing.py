#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 21:41:05 2018

@author: tortugar
"""
import matplotlib
matplotlib.use("TkAgg")
import fear
import numpy as np
from shutil import copy2
import os.path
import sev
import re
import scipy.io as so
import tkinter as Tk
import tkinter.filedialog as tkf
import vypro
from io import open

import pdb


### Functions load_grpfile and get_eegemg_ch copied from spyke,
### as there's otherwise some TK vs matplotlib conflict ##
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
    for l in lines:
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

    return grp.values()


### FUNCTIONS #########################################################################
def get_lowest_filenum(path, fname_base):
    """
    I assume that path contains files/folders with the name fname_base\d+
    find the file/folder with the highest number i at the end and then
    return the filename fname_base(i+1)
    """
    files = [f for f in os.listdir(path) if re.match(fname_base, f)]
    l = []
    for f in files:
        a = re.search('^' + fname_base + "(\d+)", f)
        if a:
            l.append(int(a.group(1)))
    if l:
        n = max(l) + 1
    else:
        n = 1

    return fname_base + str(n)


def assign_notes(params, notes):
    """
    check for each comment whether it was assigned to a specific mouse/mice using the
    @ special sign; or (if not) assign it to all mice
    """
    comment = {}

    mice = params['mouse_ID']
    for m in mice:
        comment[m] = []

    for l in notes:
        if re.match('@', l):
            for m in mice:
                if re.match('@' + m, l):
                    comment[m].append(l)
        else:
            comment[m].append(l)

    return comment


def get_infoparam(ppath, name):
    """
    name is a parameter/info text file, saving parameter values using the following
    syntax:
    field:   value

    in regular expression:
    [\D\d]+:\s+.+

    The function return the value for the given string field
    """
    fid = open(os.path.join(ppath, name), 'rU')
    lines = fid.readlines()
    params = {}
    in_note = False
    fid.close()
    for l in lines:
        if re.search("^#[nN]otes:(.*)", l):
            # a = re.search("^#\s*(.*)", l)
            # params['note'] = [a.group(1)]
            # continue
            in_note = True
            params['note'] = []
            continue
        if in_note:
            if re.match("^[A-z_\-\d]+:", l):
                in_note = False

            if in_note and not (re.search("^\s+$", l)):
                params['note'].append(l)
        if re.search("^\s+$", l):
            continue
        if re.search("^[A-z_\-\d]+:", l):

            a = re.search("^(.+):" + "\s+(.*$)", l)
            if a:
                v = a.group(2).rstrip()
                # otherwise there's a problem with the units: field
                v = re.split('\s+', v)
                params[a.group(1)] = v

                # further process 'note' entry
    tmp = [i.strip() for i in params['note']]
    tmp = [i + ' ' for i in tmp]
    if len(tmp) > 0:
        f = lambda x, y: x + y
        tmp = reduce(f, tmp)
        tmp = re.split('//', tmp)
        tmp = ['#' + i for i in tmp if len(i) > 0]

    params['note'] = assign_notes(params, tmp)

    return params



# Parameters to choose ########################################################
PPATH = '/Users/tortugar/Documents/Penn/Data/RawData/Fear'
flash_dir = '/Volumes/Transcend/New_data'
# Open File Menue to choose data for prcoessing ###############################
root = Tk.Tk()
tank_dir = tkf.askdirectory(initialdir = flash_dir)
root.update()

# I assume that tank_dir follows the following convention:
# Mouse_id-DATE-TIME
print("Selected %s" % tank_dir)
recording_dir = os.path.split(tank_dir)[1]
(mouse_id, date, time) = re.split('-', recording_dir)
yy = date[0:2]
mm = date[2:4]
dd = date[4:]
date = mm+dd+yy

params = dict()
params['date'] = [date]
params['time'] = [time]


# load info.txt
params2 = get_infoparam(tank_dir, 'info.txt')
params['note'] = params2['note']
new_keys = list(params2.keys())
new_keys.remove('note')
for k in new_keys:
    if not(k in params.keys()):
        params[k] = params2[k]

params['mouse_ID']   = params2['mouse_ID']
params['experiment'] = params2['experiment']
params['SR'] = [float(params2['SR'][0])]
params['amplifier'] = ['TDT']

mice = params['mouse_ID']
# if the experiment name contains 'Con'
# we are in a conditioning session
#pdb.set_trace()
cond_session = False
if re.search('Con', params['experiment'][0]):
    cond_session = True
# Is there a shock file?
pshock = False

nmouse = 1
for mouse in mice:
    if re.match('^X', mouse):
        nmouse += 1
        continue

    grp_file = 'common_grouping.txt'
    if os.path.isfile(os.path.join(PPATH, mouse + '_grouping.txt')):
        grp_file = mouse + '_grouping.txt'
    (eeg_list, emg_list) = get_eegemg_ch(PPATH, grp_file)
    
        # build new recording name
    fbase_name = mouse + '_' + date + 'n'
    name = get_lowest_filenum(PPATH, fbase_name)
    if not(os.path.isdir(os.path.join(PPATH,name))):
        print("Creating directory %s\n" % name)
        os.mkdir(os.path.join(PPATH,name))

    ddir = os.path.join(PPATH, name)

    if not cond_session:
        eeg_files = [f for f in os.listdir(tank_dir) if re.search('M%d_E_Ch\d+.sev$' % nmouse, f)]
        emg_files = [f for f in os.listdir(tank_dir) if re.search('M%d_M_Ch\d+.sev$' % nmouse, f)]
    else:
        eeg_files = [f for f in os.listdir(tank_dir) if re.search('EEG__Ch\d+.sev$', f)]
        emg_files = [f for f in os.listdir(tank_dir) if re.search('EMG__Ch\d+.sev$', f)]
    tone_file = [f for f in os.listdir(tank_dir) if re.search('Tone_Ch1.sev$', f)][0]
    tmp = [f for f in os.listdir(tank_dir) if re.search('Shok_Ch1.sev$', f)]
    if len(tmp) > 0:
        pshock = True
        shock_file = tmp[0]

    # process EEGs
    i = 0
    for num in eeg_list:
        eeg_file = [f for f in eeg_files if re.search('_Ch%d.sev$' % num, f)][0]
        if i==0:
            eeg_mat = 'EEG'

        else:
            eeg_mat = 'EEG%d' % (i+1)
        i += 1
        D = sev.read_sev(os.path.join(tank_dir, eeg_file))
        SR = D['fs']
        so.savemat(os.path.join(ddir, eeg_mat + '.mat'), {eeg_mat: D['data']}, do_compression=True)

    # process EMGs
    i = 0
    for num in emg_list:
        emg_file = [f for f in emg_files if re.search('_Ch%d.sev$' % num, f)][0]
        if i==0:
            emg_mat = 'EMG'
        else:
            emg_mat = 'EMG%d' % (i+1)
        i += 1
        D = sev.read_sev(os.path.join(tank_dir, emg_file))
        so.savemat(os.path.join(ddir, emg_mat + '.mat'), {emg_mat: D['data']}, do_compression=True)

    # get tone
    tone = sev.read_sev(os.path.join(tank_dir, tone_file))['data']
    so.savemat(os.path.join(ddir, 'tone.mat'), {'tone': tone})
    # get shock
    if pshock:
        shock = sev.read_sev(os.path.join(tank_dir, shock_file))['data']
        so.savemat(os.path.join(ddir, 'shock.mat'), {'tone': shock})

    # write infor file
    fid = open(os.path.join(ddir, 'info.txt'), 'w')
    # first write notes
    comments = params['note'][mouse]
    for l in comments:
        fid.write(l + os.linesep)
    # write all other info tags
    for k in params.keys():
        v = params[k]
        if k == 'note':
            continue
        if len(v) == 1:
            # shared attribute
            fid.write(k + ':' + '\t' + str(v[0]) + '\n')
        else:
            # individual attribute
            fid.write(k + ':' + '\t' + str(v[nmouse-1]) + '\n')

    # add a colleagues tag, i.e. other mice recorded together with mouse
    if len(mice)>1:
        colleagues = mice[:]
        colleagues.remove(mouse)
        fid.write('colleagues:\t' + ' '.join(colleagues) + os.linesep)
    fid.close()

    # copy video files
    if os.path.isfile(os.path.join(tank_dir, 'video_timing.mat')):
        copy2(os.path.join(tank_dir, 'video_timing.mat'), os.path.join(PPATH, name))
    if os.path.isfile(os.path.join(tank_dir, 'stack.zip')):
        copy2(os.path.join(tank_dir, 'stack.zip'), os.path.join(PPATH, name))

    nmouse += 1
    # setup time scale for freezing annotation
    vypro.tdt_video_timing(PPATH, name, dt=0.1)
    # 1s spectrogram
    fear.calculate_1s_spectrum(PPATH, name, fres=1.0)
    # for compability with sleep annotation programs
    fear.empty_annotation(PPATH, name)
    fear.downsample_tone(PPATH, name)

