# -*- coding: utf-8 -*-
"""
Created on Thu May 24 13:40:19 2018

@author: ChungWeberPC_04
"""
import sys
sys.path.append('C:/TDT/Synapse/SynapseAPI/Python')
import datetime
import re
import os
import time
import SynapseAPI
import zipfile
import tdt


def zipdir(path, zip_file):
    """
    zip directory $path to zip file $zip_file
    Note: only the parent directory of path is preserved (i.e.
    if the folder it /A/B/C all files and dirs within C are zipped and only
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


def current_time():
    # get curent time stamp
    ttime = str(datetime.datetime.now())
    a = re.split(' ', ttime)
    date = ''.join(re.split('-', a[0]))
    t = re.split(':', a[1])
    t[-1] = re.split('\.', t[-1])[0]
    ttime = ''.join(t)
    
    return date, ttime
    

# get curent time stamp
date, ttime = current_time()
timestamp = date+ttime 

# connect to synapse
syn = SynapseAPI.SynapseAPI()

# collect some experiment parameters
params = {}
#params['mouse_ID']   = syn.getCurrentSubject()
params['experiment'] = syn.getCurrentExperiment()
params['SR'] = str(syn.getSamplingRates().values()[0]/24.0)
print("SR: %f" % float(params['SR']))
yy = date[0:4]
mm = date[4:6]
dd = date[6:]
params['date'] = mm + '/' + dd + '/' + yy
params['amplifier'] = 'TDT'

# get input params
params['mouse_ID'] = raw_input('Mice [Mouse1 Mouse2]:'+os.linesep + '> ')
dur_min = int(raw_input('Experiment Duration? [time in min]'+os.linesep + '> '))
dur_del = int(raw_input('Do you wish some Delay till experiment should start? [time in min]'+os.linesep + '> '))

# Waiting some time
print("> Waiting for %d minutes" % dur_del)
time.sleep(dur_del*60)
params['time'] = current_time()[1]
print("> Starting recording for %d minutes" % dur_min)


# Run Experiment
syn.setMode(3)
while(syn.getMode() != 3):
    time.sleep(0.1)
time.sleep(1)
params['tank'] =  os.path.join(syn.getCurrentTank(), syn.getCurrentBlock())
print("> Saving data to %s" % params['tank'])

time.sleep(dur_min*60)
# turn off TDT
if (syn.getMode() == 3):
    syn.setMode(0)
    
# retrieve user notes
notes=syn.getExerimentMemos(params['experiment'], startTime=timestamp)
inf = []
for note in notes:
    a = re.split(':', note)[1:]
    time = a[0][8:]
    text = a[1]
    inf.append('%s: %s' % (time, text))
params['notes'] = inf


# write info file
fid = open(os.path.join(params['tank'], 'info.txt'), 'w')
# first write notes:
fid.write('#Notes:' +os.linesep)
for note in params['notes'][::-1]:
    fid.write('//' + note + os.linesep)
# Notes end with empty line
fid.write(os.linesep)
for k in params.keys():
    if k != 'notes':
        fid.write('%s:\t%s' % (k, params[k]))
        fid.write(os.linesep)
fid.close()
print("Closing...")

print("trying to get timing for video...")
# Extract video timing information
#ntrial = 0
#tank = '\\'.join(re.split('\\\\', params['tank'])[0:-1])
#block = re.split('\\\\', params['tank'])[-1]
#import win32com.client
#h = win32com.client.Dispatch('matlab.application')
#cmd  =  "data = TDTbin2mat(fullfile('%s' ,'%s'),'Type',{'epocs'}); \
#        a = fields(data.epocs); cam = a{1}; \
#        onset = eval(['data.epocs.' cam '.onset']); \
#        offset= eval(['data.epocs.' cam '.offset']); \
#        save(fullfile('%s', '%s','video_timing.mat'), 'onset', 'offset')" % (tank,block, tank,block)
#while not os.path.isfile(os.path.join(params['tank'], 'video_timing.mat')):
#    if ntrial>0:
#        print('...didnt work, trying again...')
#    try:
#        ntrial += 1
#        h.Execute(cmd)
#    except:
#        pass
#print("...done")
#h.quit()
tank = params['tank']
ep_data = tdt.read_block(tank, evtype='epocs')
onset = ep_data.epocs.Cam1.onset
offset = ep_data.epocs.Cam1.offset
so.savemat(os.path.join(tank, 'video_timing.mat'), {'onset': onset, 'offset': offset})



# extract movie frames
ffmpeg_path = 'C:\\Users\ChungWeberPC_04\\Documents\\Ground\\ffmpeg-20180227-fa0c9d6-win64-static\\bin\\ffmpeg'
video_file = [f for f in os.listdir(params['tank']) if re.match('^.*\.avi$', f)][0]
if not(os.path.isdir(os.path.join(params['tank'], 'Stack'))):
    os.mkdir(os.path.join(params['tank'], 'Stack'))

print("extracting single movie frames...")
import subprocess
from shutil import make_archive, rmtree
cwd = os.getcwd()
os.chdir(params['tank'])
s = ffmpeg_path + ' -i ' + video_file + ' Stack\\fig%d.jpg'
subprocess.call(s, stdout=subprocess.PIPE, shell=True)

print("zipping movie frames...")
zipdir(os.path.join(params['tank'], 'Stack'), 'stack.zip')
if os.path.isdir('Stack'):
    rmtree('Stack')
os.chdir(cwd)
print("Everything seems to have worked fine...")
