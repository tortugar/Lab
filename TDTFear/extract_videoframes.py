# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:50:26 2018

@author: ChungWeberPC_04
"""
# extract movie frames
import subprocess
from shutil import make_archive, rmtree
import os
import re
import Tkinter as Tk
import tkFileDialog as tkf
import tdt
import scipy.io as so

flashdir = 'C:/Users/ChungWeberPC_04/Documents/Tank'
root = Tk.Tk()
tank_dir = tkf.askdirectory(initialdir = flashdir)
root.update()

params = dict()
params['tank'] = tank_dir

# get video timing
print("trying to get timing for video...")
# Extract video timing information
# tank = '\\'.join(re.split('\\\\', params['tank'])[0:-1])
# block = re.split('\\\\', params['tank'])[-1]
# import win32com.client
# h = win32com.client.Dispatch('matlab.application')
# cmd  =  "data = TDTbin2mat(fullfile('%s' ,'%s'),'Type',{'epocs'}); \
#         a = fields(data.epocs); cam = a{1}; \
#         onset = eval(['data.epocs.' cam '.onset']); \
#         offset= eval(['data.epocs.' cam '.offset']); \
#         save(fullfile('%s', '%s','video_timing.mat'), 'onset', 'offset')" % (tank,block, tank,block)
# print(cmd)
# while not os.path.isfile(os.path.join(params['tank'], 'video_timing.mat')):
#     print('...didnt work, trying again...')
#     try:
#         h.Execute(cmd)
#     except:
#         pass
# print("...done")
# h.quit()
tank = params['tank']
ep_data = tdt.read_block(tank, evtype=['epocs'])
onset = ep_data.epocs.Cam1.onset
offset = ep_data.epocs.Cam1.offset
so.savemat(os.path.join(tank, 'video_timing.mat'), {'onset': onset, 'offset': offset})


# extracting single movide frames
ffmpeg_path = 'C:\\Users\ChungWeberPC_04\\Documents\\Ground\\ffmpeg-20180227-fa0c9d6-win64-static\\bin\\ffmpeg'
video_file = [f for f in os.listdir(params['tank']) if re.match('^.*\.avi$', f)][0]
if not(os.path.isdir(os.path.join(params['tank'], 'Stack'))):
    os.mkdir(os.path.join(params['tank'], 'Stack'))

print("extracting single movie frames...")
cwd = os.getcwd()
os.chdir(params['tank'])
s = ffmpeg_path + ' -i ' + video_file + ' Stack\\fig%d.jpg'
subprocess.call(s, stdout=subprocess.PIPE, shell=True)

# zipping movie frames into zip stack
print("zipping movie frames...")
make_archive('stack', 'zip', params['tank'], base_dir='Stack')
if os.path.isdir('Stack'):
    rmtree('Stack')
os.chdir(cwd)
print("Everything seems to have worked fine...")

