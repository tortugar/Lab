NOTE: the imaging code is still in py27

(1) Convert intan format to our format:

python data_processing_cl.py 

The script generates a recording folder containing the EEG, EMG, info.txt file. The file laser_rec-name.mat contains the strobe signal from the camera.


(2) using normal sleep_annotation_qt.py, you can inspect now the sleep recording:
Cut out the imaging session; that means the time frame, where the
"laser" in sleep_annotation_qt was on. If the laser signal is initially 0, then 
each time the laser switches to 1 corresponds to a new video frame:

imaging.caim_snip(ppath, name)


(3) transform all the *.avi files to a hdf5 stack. The result is saved
in $ppath/$name/recording_$name_downsamp.hdf5
The parameter ndown allows for spatial downsampling. For example,
if ndown=2, the resulting image stack has only half the size. 
Since the spatial resolution is anyways not great, I would recommend ndown=2

imaging.avi2h5('/Volumes/Transcend/RawImaging/sleepRecording_JH_200117_120026/H12_M2_S44/', ppath, name, ndown=2)


(4) The miniscope often records redundant frames (i.e. two successive
frames are exactly the same. If that's the case I throw out the
second (redundant) frame and accordingly adjust the time
vector 'img_time.mat'; 
The resulting new image stack is saved in 
$ppath/$name/recording_$name_downsampcorr.hdf5

imaging.minisc_timing(ppath, name)


(5) motion correction: first spatially high pass filter each frame 
imaging.disk_filter_h5(os.path.join(ppath, name), 'recording_%s_downsampcorr.hdf5'%name, nw=2)

then align images to a reference frame using cross correlation.
Once you start imaging.align_frames, the average disk-filtered image
will appear on the screen; using the mouse (left-click), choose four corners of a 
rectangle within the image that contains a lot of contrasts, but few
ROIs; double click to select the fourth corner.

imaging.align_frames(os.path.join(ppath, name), 'recording_%s_disk.hdf5'%name)


(6) calculate activity map; This map is used by imaging_gui.py to manually
select ROIs

imaging.activity_map(os.path.join(ppath, name), 'recording_%s_aligned.hdf5' % name, nw=2)


(7) run 
python imaging_gui.py -i "base folder containing the imaging recordings"
(In Macos run pythonw imaging_gui.py)

click "view ROI" and start adding your ROIs using "A" key. Use "R" key to register your ROI.
When finished, you save your ROIs.

There's only one variable in imaging_gui that you need to adjust:
Towards the very end, there's the declaration of the variable ipath;
Set ipath to the folder where all your processed imaging recordings
are saved.


##############
Aligning ROIs across different imaging sessions
Note: This code only runs in python3

On terminal run
python alignsession_script.py -i ~/Documents/Penn/Data/RawData/Miniscope/ -r "J326_011920n2 J326_012120n1"

Using left click, select corresponding reference points on the shown activation maps. 
I would select at least three points.
Click 'done', when done.

in python:

import imaging
df_map = imaging.load_roimapping(os.path.join(path, 'J326_mapping.csv')) 
df_class = imaging.brstate_dff(path, df_map) 

That's how df_map looks like:
   mouse  ID J326_051920n2 J326_052120n1
0   J326   0           1-0           1-5
1   J326   1             X           1-0
2   J326   2           1-1           1-3
3   J326   3           1-2          1-17
4   J326   4             X           1-2
5   J326   5           1-3             X
6   J326   6           1-4           1-7

and df_class:
   mouse  ID J326_051920n2 J326_052120n1         R         W         N    F-anova        P-anova   P-tukey   Type
0   J326   0           1-0           1-5  0.293890  0.148710  0.350353   4458.089   0.000000e+00  0.001000  N-max
1   J326   1             X           1-0  0.372525  0.694640  0.386801   2992.725   0.000000e+00  0.046592  W-max
2   J326   2           1-1           1-3  0.289325  0.228649  0.124685   3462.297   0.000000e+00  0.001000  R-max
3   J326   3           1-2          1-17  0.846838  0.218281  0.221435  25079.206   0.000000e+00  0.001000  R-max
4   J326   4             X           1-2  0.167318  0.194019  0.179340     43.019   2.116806e-19  0.001000  W-max
5   J326   5           1-3             X  0.307156  0.105152  0.134125   3040.365   0.000000e+00  0.001000  R-max
6   J326   6           1-4           1-7  0.551730  0.122919  0.078187  35630.965   0.000000e+00  0.001000  R-max

mx=imaging.brstate_transitions(path, df_class[df_class['Type']=='R-max'], [[3,1], [1,2]], 60, 30, [20, 20, 20], [0, 0, 0]) 

