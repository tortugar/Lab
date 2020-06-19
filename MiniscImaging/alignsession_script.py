"""
When you run the script it will open a figure with all the reference maps. 
As first step, select reference points for alignment by left clicking. 
For reference point 1, you first click image1, then image2 etc.; 
and then for reference point 2, image1, image2 etc.  Finally click done on the lower right. 

When used for the first time, the script will write two parameters into the 
info.txt files: 
    ALIGN: and MASTER:
If you want to re-do the alignment, please delete these two parameters

The first recording is the reference recording (MASTER) according to which all 
other recordings are aligned. 
ALIGN: shows for each recording how much the image had to be rotated, moved in x and y direction
to optimally overlap with MASTER.

Example run:
    python alignsession_script.py -i ~/Documents/Penn/Data/RawData/Miniscope/ -r "J294_031720n1 J294_031920n1"
    
To see help run
    python alignsession_script.py
    
    
The result of the script is a text file MOUSEID_mapping.txt which lists 
all shared and individual ROIs. For each ROI, is indicates the actual ROI number 
for each recording sharing this ROI. 

ROI list management:
By default the script used roilist1. However, if specified in the info.txt file,
the script can use any roilist. To choose another roilist n, add the parameter
ROI_ID: n
to the corresponding info.txt file
"""

import alignsession as align
import imaging
import sys
import re
import numpy as np
import os
import scipy.io as so
import matplotlib.pylab as plt
import pandas as pd
import pdb



### SUBROUTINES ######################################
def param_check(x) :
    params = {}
    for i in range(0, len(x)) :
        a = x[i]
        if re.match('^-', a):
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
    try:
        fid = open(ifile, newline=None)
    except:
        fid = open(ifile)
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines :
        a = re.search("^" + field + ":" + "\s+(.*)", l)
        if a :
            values.append(a.group(1))
            
    return values
######################################################

plt.ion()
if len(sys.argv) == 1:
    print("Align the average (aligned) frames of the given recordings (parameter -r).\nOpen parameters are rotation angle and translation in x and y direction.\nusage: python align_session_script.py\n -i imaging directory\n -r \"recording names\"\n    the first recording is the MASTER recording used as reference for all other recordings.\n")
    sys.exit(1)

# get parameters
params     = sys.argv[1:]
args       = param_check(params)
ipath      = args['-i']
recordings = re.split('\s+', args['-r'])
master = recordings[0]

print(recordings)

# just one recordings, no alignment necessary
#if len(recordings) == 1 :
#    (ROI_coords, ROIs) = load_roilist(ipath, master, 1)
#    sys.exit(1)


# more then one recording:

theta = np.arange(0, 2*np.pi, 0.1)
ax = np.arange(-30., 30.)
bx = np.arange(-30., 30.)
thr = 0.5
amap = True

id_map = dict()
for rec in recordings:
    # get ROI_ID
    roi_id = get_infoparam(os.path.join(ipath, rec, 'info.txt'), 'ROI_ID')
    if len(roi_id) == 0:
        id_map[rec] = 1
    else:
        id_map[rec] = int(roi_id[0])

pdb.set_trace()


# get optimal rotation/translation
transf = {}
for rec in recordings :

    # test if alignment has been already done previously:
    m = get_infoparam(os.path.join(ipath,rec, 'info.txt'), 'MASTER')
    # if not, then do it:
    if len(m) == 0 :
        if not(rec == master):
            (theta_min,a_min,b_min) = align.align_recordings(ipath, master, rec, theta, ax, bx, amap, id_map[master], id_map[rec])
            try:
                s = raw_input('Does the overlap look ok? [yes|no]')
            except:
                s = input('Does the overlap look ok? [yes|no]')
        else :
            (theta_min,a_min,b_min) = (0.,0.,0.)
            s = 'yes'
    
        # write optimal rotation into info file
        if s == 'yes' :
            align.write_trans_parameters(ipath, master, rec, theta_min, a_min, b_min)
            transf[rec] = (theta_min,a_min,b_min)
    else:
        a = get_infoparam(os.path.join(ipath,rec, 'info.txt'), 'ALIGN')[0]
        (theta_min, a_min, b_min) = [float(i) for i in re.split('\s+', a)]
        transf[rec] = (theta_min, a_min, b_min)
        

im_shapes = [so.loadmat(os.path.join(ipath, rec, 'recording_' + rec + '_alignedmean.mat'))['mean'].shape for rec in recordings]



# show all the overlapping ROIs across sessions
if amap:
    master_image = so.loadmat(os.path.join(ipath, master, 'recording_' + master + '_actmap.mat'))['mean']
else:
    master_image = so.loadmat(os.path.join(ipath, master, 'recording_' + master + '_alignedmean.mat'))['mean']
master_shape = im_shapes[0]
plt.figure()
axes = plt.subplot(111)
axes.imshow(master_image, cmap='gray', vmin=0, vmax=np.percentile(master_image, 99.5))
colors = ['b', 'r', 'g', 'p']
for rec in recordings :
    if master == rec :
        (ROI_coords, ROIs_trans) = imaging.load_roilist(ipath, rec, id_map[rec])
    else :
        (ROI_coords, ROIs) = imaging.load_roilist(ipath, rec, id_map[rec])
        param = transf[rec]
        ROIs_trans = align.rottrans_rois(ROIs, param[0], param[1], param[2], master_shape[0], master_shape[1])
    align.draw_rois(ROIs_trans, axes, colors.pop(0))



# build ROI mapping for sessions
nrec = len(recordings)
roi_mapping = {}
for i in range(nrec) :
    for j in range(i,nrec) :
        im1_shape = im_shapes[i]
        im2_shape = im_shapes[j]
        rec1 = recordings[i]
        rec2 = recordings[j]

        param1 = transf[rec1]
        param2 = transf[rec2]

        (ROI_coords, ROIs1) = imaging.load_roilist(ipath, rec1, id_map[rec1])
        (ROI_coords, ROIs2) = imaging.load_roilist(ipath, rec2, id_map[rec2])

        mapping = align.roi_overlap(ROIs1, ROIs2, param1, param2, im1_shape, im2_shape, thr_ovlp=thr)
        #mapping = [(rec1 + '-' + str(ii), rec2 + '-' + str(jj)) for (ii,jj) in mapping]
        mapping = [(rec1 + '-' + str(id_map[rec1]) + '-' + str(ii), 
                    rec2 + '-' + str(id_map[rec2]) + '-' + str(jj)) for (ii,jj) in mapping]

        for m in mapping :
            #if roi_mapping.has_key(m[0]) :
            if m[0] in roi_mapping:
                roi_mapping[m[0]].append(m[1])
            else :
                if sum([m[0] in v for v in roi_mapping.values()]) == 0:
                    roi_mapping[m[0]] = [m[1]]

# mouse identifier
mouse = re.split('_', master)[0]
mapping_file = os.path.join(ipath, mouse + '_mapping.txt')
csv_file = os.path.join(ipath, mouse + '_mapping.csv')
k = list(roi_mapping.keys())
def _get_roi(s):
    return int(re.split('-', s)[2])
k.sort(key=_get_roi)
unique_id = {}
i=0
for roi in k :
    if not(roi in unique_id):
        unique_id[roi] = i
        i=i+1

recording_map = {}
i=0
for r in recordings:
    recording_map[r] = i
    i=i+1

data = []
Lines = []
Lines.append('#ROI-ID\t\t' + '\t'.join(recordings) + '\n')
for roi in k :
    roi_id = unique_id[roi]
    sessions = roi_mapping[roi]
    #line = [(re.split('-',s)[0], int(re.split('-',s)[1])) for s in sessions]
    line = [(re.split('-',s)[0], 
             re.split('-',s)[1]+'-'+re.split('-',s)[2]) for s in sessions]
    line_map = {}
    for (p,q) in line :
        line_map[p] = q


    line_str = str(roi_id) + '\t\t' + mouse + '\t\t' 
    for r in recordings :
        if r in line_map.keys() :
            line_str = line_str + str(line_map[r]) + '\t\t'
        else :
            line_str = line_str + 'X\t\t'

    Lines.append(line_str + '\n')
    data.append(line_str.split('\t\t')[0:-1])
    

fid = open(mapping_file, 'w')    
[fid.write(l) for l in Lines]
fid.close()

for l in Lines :
    l = l.rstrip('\n')
    print(l)

# write into pandas frame
df = pd.DataFrame(columns=['ID', 'mouse']+recordings, data=data)
df.to_csv(csv_file, index=False)


plt.show(block=True)

