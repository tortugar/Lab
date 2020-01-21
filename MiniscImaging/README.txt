NOTE: the imaging code is still in py27

(1) to convert intan format to our format:

python data_processing_cl.py 


(2) using normal sleep_annotation_qt.py, you can inspect now the sleep recording:
Cut out the imaging session (that means the time frame, where the
"laser" in sleep_annotation_qt was on:

sleepy.caim_snip(ppath, name)


(3) transform all the *.avi files to a hdf5 stack; The result is saved
in $ppath/$name/recording_$name_downsamp.hdf5
The parameter ndown allows for spatial downsampling. For example,
if ndown=2, the resulting image stack has only half the size. 
Since the spatial resolution is anyways not great, I would recommend ndown=2

imaging.avi2h5('/Volumes/Transcend/RawImaging/sleepRecording_JH_200117_120026/H12_M2_S44/', ppath, name, ndown=2)


(4) The miniscope often records redundant frames (i.e. two sucessive
frames are exactly the same, if that's the case I throw out the
second (redundant) frame frame and accordingly adjust the time
vector 'img_time.mat'; 
The resulting new image stack is saved in 
$ppath/$name/recording_$name_downsampcorr.hdf5

imaging.minisc_timing(ppath, name)


(5) motion correction: first spatially high pass filter each frame 
imaging.disk_filter_h5(os.path.join(ppath, name), 'recording_%s_downsampcorr.hdf5'%name, nw=2)

then, align images to a reference image using cross correlation;
once you start imaging.align_frames, the average disk filtered image
will appear on the screen; using the mouse, choose four corners of a 
rectangle within the image that contains a lot of contrasts, but few
OIs; double click, once the four coners are selected.

imaging.align_frames(os.path.join(ppath, name), 'recording_%s_disk.hdf5'%name)


(6) calculate activity map; This map is used by imaging_gui to manually
select ROIs

imaging.activity_map(os.path.join(ppath, name), 'recording_%s_aligned.hdf5' % name, nw=2)


(7) run 
python imaging_gui.py
(In Macos run pythonw imaging_gui.py)

click "view ROI" and start adding your ROIs using "A" key. Use "R" key to register your ROI.
When finished, you save your ROIs.

There's only one variable in imaging_gui that you need to adjust:
Towards the very end, there's the declaration of the variable ipath;
Set ipath to the folder where all your processed imaging recordings
are saved.
