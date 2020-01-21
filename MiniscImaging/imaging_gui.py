import sys
sys.path.append('/Users/tortugar/Google Drive/Berkeley/Data/Programming/CommonModules')
import matplotlib
matplotlib.use('WXAgg')
import scipy.io as so
import wx
import h5py
import os
import re
import numpy as np
from roipoly import roipoly
from sleepy import get_snr, load_stateidx #added this line to test plotting stateidx

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NvigationToolbar
from Utility import *
from imaging import *

# to test whehter point lies within polygon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

### Debugger
import pdb



# for a wxpython gui you
# need to inherit the class wx.Frame
class ImageViewer(wx.Frame) :

    def __init__(self, ipath):
        wx.Frame.__init__(self, None, -1, "", size=(600,100), pos=(100,100))

        # folders
        self.ipath = ipath
        self.name = ""

        # roi management
        self.proi_mode = False
        self.pcollect_roi = False
        self.proi_delete = False
        self.pdisk = False
        self.curr_roi_x = []
        self.curr_roi_y = []
        self.roi_id = 0
        self.roi_name = ''
        self.cmap = np.zeros((1,3))
        # shape: [(xcoords, ycoords),...]
        self.ROIs = []
        self.ROI_coords = []

        self.dpi = 100
        self.icursor = 0
        self.dt = 20
        self.sr = -1
        # recording contains spectrogram
        self.sp_exists = True
        self.pcorr_stack = True

        self.create_menu()

        

    def setup_imaging(self) :
        

        # load spectrum
        if os.path.isfile(os.path.join(self.ipath, self.name, 'sp_' + self.name + '.mat')):
            P = so.loadmat(os.path.join(self.ipath, self.name, 'sp_' + self.name + '.mat'))
            self.SP = P['SP']
            self.freq = P['freq']
            self.stime = P['t'][0]
            self.sdt = self.stime[1]-self.stime[0]
            self.sp_exists = True
        
        # read image stack
        if os.path.isfile(os.path.join(self.ipath, self.name, 'recording_' + self.name + '_aligned.hdf5')):
            fid = h5py.File(os.path.join(self.ipath, self.name, 'recording_' + self.name + '_aligned.hdf5'))
            print "loaded motion corrected file (*_aligned.hdf5)"
        else:
            fid = h5py.File(os.path.join(self.ipath, self.name, 'recording_' + self.name + '_downsampcorr.hdf5'))

        
        self.stack = fid['images']
        self.nframes = self.stack.shape[0]


        # read mean of disk filtered stack
        if os.path.isfile(os.path.join(self.ipath, self.name, 'recording_' + self.name + '_actmap.mat')):
            self.disk_img = so.loadmat(os.path.join(self.ipath, self.name, 'recording_' + self.name + '_actmap.mat'))['mean']
        else:
            self.disk_img = self.stack[0,:,:]

        # read brain states
        # test if sleep state file exists
        if os.path.isfile(os.path.join(self.ipath, self.name, 'remidx_' + self.name + '.txt')): #added remidx'z' on the string to remove this dependence
            tmp = load_stateidx(self.ipath, self.name)
        else:
            tmp = np.zeros((0,))
        self.M = np.zeros((1, tmp.shape[0]))
        self.M[0,:] = tmp

        # get colorrange minimum and maximum values
        tmp = self.stack[1000:1500,20:-20,20:-20]
        self.cmin = tmp.min()
        self.cmax = tmp.max()

        self.panel = wx.Panel(self)
        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_down)

        # call functions
        self.create_image_axes()
        # called each time something is supposed to change on the figure
        self.draw_figure() 


    
    def create_menu(self) :
        """
        create menu allowing the user to select an imaging folder
        """
        self.menubar = wx.MenuBar()
        
        menu_file = wx.Menu()
        m_expt = menu_file.Append(-1, "&Open Folder\tCtrl-o", "open imaging folder")
        self.Bind(wx.EVT_MENU, self.on_open, m_expt)
        menu_file.AppendSeparator()
        m_exit = menu_file.Append(-1, "&Exit\tCtrl-X", "Exit")
        self.Bind(wx.EVT_MENU, self.on_exit, m_exit)
        
        menu_help = wx.Menu()
        m_about = menu_help.Append(-1, "&About\tF1", "Help")
        #self.Bind(wx.EVT_MENU, self.on_about, m_about)

        # Menu for ROI processing
        menu_roi = wx.Menu()
        # Extract ROIs
        m_extract = menu_roi.Append(-1, "&Extract ROI\tF2", "ROI")
        self.Bind(wx.EVT_MENU, self.on_extract, m_extract)
        # Plot ROIs
        m_plot = menu_roi.Append(-1, "&Plot ROI\tF3", "Plot")
        self.Bind(wx.EVT_MENU, self.on_plot, m_plot)
        
        
        self.menubar.Append(menu_file, "&File")
        self.menubar.Append(menu_help, "&Help")
        self.menubar.Append(menu_roi, "&ROI Processing")
        self.SetMenuBar(self.menubar)

        
    def create_image_axes(self) :
        """
        draw all the elements on the figure
        """

        self.fig = Figure((5.0, 5.0), facecolor='white', dpi=100)
        self.canvas = FigCanvas(self.panel, -1, self.fig)
        self.axes = self.fig.add_axes([0.05, 0.28, 0.9, 0.72])

        if self.sp_exists:
            img_time = imaging_timing(self.ipath, self.name)
            closest_time = lambda(it, st) : np.argmin(np.abs(it-st))    
            nf = np.min((self.nframes, img_time.shape[0]))
            last_state = closest_time((img_time[nf-1], self.stime))
            #HERE just some hack 10/6/17
            last_state = self.M.shape[1]
            self.sr = 1.0 / np.mean(np.diff(img_time))

        # axes for EEG spectrogram
        if self.sp_exists:
            self.axes_eeg = self.fig.add_axes([0.05, 0.13, 0.9, 0.14])
            SP = self.SP[0:31,0:last_state+1]
            SP = np.flipud(SP)
            tmp = self.axes_eeg.imshow(SP)
            tmp.set_clim((SP.min(), np.median(SP[:])*20))
            self.axes_eeg.axis('tight')
            self.axes_eeg.set_xticks([])
            self.axes_eeg.set_yticks([29, 19, 9])
            self.axes_eeg.yaxis.set_label_position("right")
            self.axes_eeg.set_yticklabels([0, 10, 20], fontsize=9)
            self.axes_eeg.set_ylabel('Freq (Hz)', fontsize=9)


        # show brain state
        if True:#self.sp_exists:
            self.axes2 = self.fig.add_axes([0.05, 0.08, 0.9, 0.04])
            cmap = plt.cm.jet
            my_map = cmap.from_list('ha', [[0,1,1],[1,0,1], [0.8, 0.8, 0.8]], 3)
            tmp = self.axes2.imshow(self.M)
            tmp.set_cmap(my_map)
            self.axes2.axis('tight')
            tmp.axes.get_xaxis().set_visible(False)
            tmp.axes.get_yaxis().set_visible(False)
        
        # time point tick
        self.axes3 = self.fig.add_axes([0.05, 0.08, 0.9, 0.02])
        self.axes3.set_xticks(np.arange(0, self.nframes, self.sr*600))
        labels = np.arange(0, (self.nframes/(self.sr*60.0)), 10)
        self.axes3.set_xticklabels(labels)

        #tmp.axes.get_xaxis().set_visible(False)
        self.axes3.get_yaxis().set_visible(False)


        ### Create slider to choose image frame
        self.slider_label = wx.StaticText(self.panel, -1, 
            "Bar width (%): ")
        self.slider_width = wx.Slider(self.panel, -1, size=(200,-1),
            value=0, 
            minValue=0,
            maxValue=self.nframes-1,
            style=wx.SL_AUTOTICKS | wx.SL_LABELS)
        self.slider_width.SetTickFreq(10) ## changed this (only accepts 1 arg in the new version of wx)
        self.Bind(wx.EVT_COMMAND_SCROLL_THUMBTRACK, self.on_slider_width, self.slider_width)


        ### create text box to set dt
        self.set_dt = wx.StaticText(self.panel, -1, "dt [s]")
        self.textbox = wx.TextCtrl(
            self.panel, 
            size=(50,-1),
            style=wx.TE_PROCESS_ENTER)
        self.Bind(wx.EVT_TEXT_ENTER, self.on_text_enter, self.textbox)

        #########################################################
        ### ROI management widgets ##############################
        #########################################################

        ### text label saying ROI
        self.roi_text = wx.StaticText(self.panel, -1, "ROIs")

    
        ### create check box asking for ROI mode
        self.roi_check = wx.CheckBox(self.panel, -1, 
                                   "Show ROIs",
                                   style=wx.ALIGN_RIGHT)
        self.Bind(wx.EVT_CHECKBOX, self.on_roi_check, self.roi_check)



        ### check box asking for disk image
        self.disk_check = wx.CheckBox(self.panel, -1, 
                                   "Show Disk",
                                   style=wx.ALIGN_RIGHT)
        self.Bind(wx.EVT_CHECKBOX, self.on_disk_check, self.disk_check)

        ### create NEW Button
        self.button_new = wx.Button(self.panel, -1, "NEW")
        self.Bind(wx.EVT_BUTTON, self.on_button_new, self.button_new)

        ### create LOAD Button
        self.button_load = wx.Button(self.panel, -1, "LOAD")
        self.Bind(wx.EVT_BUTTON, self.on_button_load, self.button_load)
        
        ### create SAVE Button
        self.button_save = wx.Button(self.panel, -1, "SAVE")
        self.Bind(wx.EVT_BUTTON, self.on_button_save, self.button_save)

        ### somehow position all these widgets
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        #self.vbox.Add(self.toolbar, 0, wx.EXPAND)
        #self.vbox.AddSpacer(10)


        
        #flags = wx.ALIGN_CENTER | wx.EXPAND | wx.ALIGN_CENTER_VERTICAL
        #self.hbox.Add(self.cb_grid, 0, border=3, flag=flags)
        #self.hbox.AddSpacer(0)
        #self.hbox.Add(self.slider_label, 0, flag=flags)
        #self.hbox.Add(self.slider_width, 0, border=3, flag=wx.EXPAND)

        self.bhox_slider = wx.BoxSizer(wx.HORIZONTAL)
        self.bhox_slider.AddSpacer(25)
        self.bhox_slider.Add(self.slider_width, 1, border=0, flag=wx.TOP | wx.EXPAND | wx.ALIGN_CENTER)
        self.bhox_slider.AddSpacer(25)
        self.vbox.Add(self.bhox_slider, 0, border=0, flag = wx.ALIGN_CENTER | wx.EXPAND)
        #self.vbox.Add(self.slider_width, 0, border=0, flag = wx.BOTTOM | wx.ALIGN_CENTER | wx.EXPAND)
        
        # horizontal box for time setp
        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox.Add(self.set_dt, 0, border=3, flag=wx.TOP | wx.LEFT)
        self.hbox.Add(self.textbox, 0, border=3, flag=wx.TOP | wx.LEFT)
        self.vbox.Add(self.hbox, 0, flag = wx.BOTTOM | wx.ALIGN_CENTER | wx.EXPAND)


        ### horizontal box for ROI management
        self.hbox_roi = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox_roi.Add(self.roi_text, 0, border=3, flag=wx.TOP | wx.LEFT)
        self.hbox_roi.Add(self.button_new, 0, border=3, flag=wx.TOP | wx.LEFT)
        self.hbox_roi.Add(self.button_load, 0, border=3, flag=wx.TOP | wx.LEFT)
        self.hbox_roi.Add(self.button_save, 0, border=3, flag=wx.TOP | wx.LEFT)
        self.hbox_roi.Add(self.roi_check, 0, border=3, flag=wx.TOP | wx.LEFT)
        self.hbox_roi.Add(self.disk_check, 0, border=3, flag=wx.TOP | wx.LEFT)
        # finally add to big vertical box
        self.vbox.Add(self.hbox_roi, 0, flag = wx.BOTTOM | wx.ALIGN_CENTER | wx.EXPAND)


        self.panel.SetSizer(self.vbox)
        self.vbox.Fit(self)

        self.axes.figure.canvas.mpl_connect('button_press_event', self.on_pick)


    def draw_figure(self):
        
        # plot current imaging frame
        self.axes.clear()
        if self.pdisk == True and self.proi_mode == True:
            img = self.axes.imshow(self.disk_img)
        else:
            img = self.axes.imshow(self.stack[self.icursor,:,:])
            img.set_clim((self.cmin, self.cmax))
        img.set_cmap('gray')
        # No x and y tick labels:
        img.axes.get_xaxis().set_visible(False)
        img.axes.get_yaxis().set_visible(False)
        self.axes.axis('tight')

        
        # show current time point
        self.axes3.clear()
        self.axes3.plot(self.icursor, 0, 'r^')
        self.axes3.set_xlim([0, self.nframes])
        self.axes3.set_xticks(np.arange(0, self.nframes, self.sr*600))
        labels = np.arange(0, (self.nframes/(self.sr*60.0)), 10)
        self.axes3.set_xticklabels(labels, fontsize=9)
        self.axes3.set_xlabel('Time (min)', fontsize=9)


        # draw ROIs
        if self.proi_mode == True :
            self.draw_rois()
            if self.pcollect_roi == True :
                self.draw_current_roi()
        self.canvas.draw()    

        
    ### draw ROIs
    def draw_rois(self) :
        i=0
        for (x,y) in self.ROIs :
            # NOTE: x and y coordinates are exchanged!!
            # imshow and Line2D assume different coordinate arrangements!!
            l = plt.Line2D(y+[y[0]], x+[x[0]], color=self.cmap[i,:])
            self.axes.text(np.max(y)-5, np.min(x)+7, str(i), fontsize=10, color=self.cmap[i,:],bbox=dict(facecolor='w', alpha=0.))
            self.axes.add_line(l)
            i = i+1
            

    def draw_current_roi(self) :
        # NOTE: x and y exchanged!
        l = plt.Line2D(self.curr_roi_y, self.curr_roi_x, color='r', marker='.')
        self.axes.add_line(l)
        

    
    ### CALLBACKS
    def on_slider_width(self, event):
        self.icursor = self.slider_width.GetValue()
        self.draw_figure()
        event.Skip()

    def on_key_down(self, event):
        keycode = event.GetKeyCode()
        #print keycode
        #cursor to the right
        if keycode == 316 :
            self.icursor = self.icursor+self.dt
            self.slider_width.SetValue(self.icursor)
            self.draw_figure()
        #cursor to the left
        if keycode == 314 :
            if self.icursor >= self.dt:
                self.icursor = self.icursor - self.dt
                self.slider_width.SetValue(self.icursor)
                self.draw_figure()

        # 'a' pressed: start drawing new roi
        if keycode == 65 :
            if self.proi_mode == True :
                self.pcollect_roi = True

        # 'c' pressed: stop drawing the current roi
        if keycode == 67 :
            if self.proi_mode == True :
                self.pcollect_roi = False
                self.curr_roi_x = []
                self.curr_roi_y = []
                self.draw_figure()

        # 'r' pressed: register current roi
        if keycode == 82 :
            if self.proi_mode == True and self.pcollect_roi == True :
                self.ROIs.append((self.curr_roi_x[:], self.curr_roi_y[:]))
                self.curr_roi_x = []
                self.curr_roi_y = []
                self.pcollect_roi = False
                self.set_cmap()
                self.draw_figure()

        # 'd' pressed: delete roi
        if keycode == 68 :
            # test for each roi if the mouse is within it, if so, delete this roi
            # now we are in roi delete mode
            self.proi_delete = True

        event.Skip()
        

    def on_text_enter(self, event) :
        self.dt = int( float(self.textbox.GetValue()) / (1.0 / self.sr) )
        print "dt %d" % self.dt


    # left mouse button was pressed:
    def on_pick(self, event) :
        """
        mouse has been clicked on the canvas
        """
        box_points = (event.xdata, event.ydata)
        print "coords:"
        print box_points
        print self.axes.get_xlim()
        print self.axes.get_ylim()
        print self.stack.shape
        
        if event.inaxes == self.axes :
            if (self.proi_mode == True) and (self.pcollect_roi == True) :
                # This is CRUCIAL:
                # coordinate system of imshow:
                #  0     
                #  |
                #  |
                # y|
                #  |
                #  |
                #  v 0-------->
                #        x
                # the box points I get from on_pick perfectly follow this convention.
                # When I plot (e.g. using plot) these points on the imshow figure, they appear a the right location
                # Hence the coordinate system of imshow, extracted points and replotting points on it
                # are consistent.
                # However, the matrix arrangement (according to numpy) is the following:
                #
                #  0     
                #  |
                #  |
                # x|
                #  |
                #  |
                #  v 0-------->
                #        y
                # Assume imshow shows matrix M and I have according to  imshow the coordinates (x, y). To get
                # the cooresponding element from the matrix, I have to type M(y, x). This is because
                # y corresponds to the rows (which come first) and x to the columns. So,
                # if I want to save the ROI coordinates according to the numpy matrix convention, I
                # should save x as y and y as x:
                # Again, I save the coordinates according to the way I access a matrix element: rows first, then columns
                self.curr_roi_x.append(box_points[1])
                self.curr_roi_y.append(box_points[0])

                self.draw_figure()

            if self.proi_mode == True and self.proi_delete == True :
                self.proi_delete = False
                # NOTE: change of coordinate system
                (py,px) = (box_points[0], box_points[1])
                point = Point(px,py)
                i = 0
                for r in self.ROIs :
                    (x0,y0) = (r[0][0], r[1][0])
                    polyg = Polygon(zip(r[0], r[1]) + [(x0,y0)])
                    if polyg.contains(point) :
                        # delete this point
                        self.ROIs.pop(i)
                        self.draw_figure()
                    i = i+1
                


    def on_roi_check(self, event):
        print "checked roi"
        if self.roi_check.IsChecked() :
            self.proi_mode = True
            self.draw_figure()
        else :
            self.proi_mode = False
            self.curr_roi_x = []
            self.curr_roi_y = []
            self.draw_figure()


    def on_disk_check(self, event):
        if self.disk_check.IsChecked() :
            self.pdisk = True
            self.draw_figure()
        else :
            print "disk unchecked"
            self.pdisk = False
            self.draw_figure()


    # callback for LOAD button
    def on_button_load(self, event):
        self.load_rois()
        self.draw_figure()

    def on_button_save(self, event):
        self.save_rois()

    def on_button_new(self, event):
        ddir = os.path.join(self.ipath, self.name)
        fname_base = 'recording_' + self.name + '_roilistn' 
        files = [f for f in os.listdir(ddir) if re.match(fname_base, f)]
        l = []
        for f in files :
            a = re.search('^' + fname_base + "(\d+)", f)
            if a :
                l.append(int(a.group(1)))
            
        n = 1
        if l: n = max(l) + 1

        print "creating new roilist with id %d" % n
        self.roi_id = n

            
    def on_exit(self, event):
        self.Destroy()            
            
    
    def on_open(self, event):
        dlg = wx.DirDialog(
            self, 
            message="Chose imaging folder...",
            defaultPath=self.ipath,
            style=wx.FD_SAVE)
        
        if dlg.ShowModal() == wx.ID_OK:
            self.name = os.path.split(dlg.GetPath())[-1]
            self.setup_imaging()


    def on_extract(self, event):
        msg = """ Extracting ROIs and Background
        That might take a while...
        """
        dlg = wx.MessageDialog(self, msg, "Extracting ROIs", wx.OK)
        dlg.ShowModal()        
        dlg.Destroy()
        self.extract_rois()


    def on_plot(self, event):
        """
        plot ROIs; dialog asking for correction factor
        """
        dlg = wx.TextEntryDialog(self, "Set correction factor", defaultValue='0.5')
        dlg.ShowModal()
        corr = float(dlg.GetValue())
        dlg.Destroy()

        self.plot_rois(corr)

            
    ### ROI Management related functions
    def extract_rois(self):
        """
        extract ROIs and Halo Background
        """
        ddir = os.path.join(self.ipath, self.name)
        if not self.pcorr_stack:
            arec = 'recording_' + self.name + '_downsamp.hdf5'
        else:
            arec = 'recording_' + self.name + '_downsampcorr.hdf5'
        stack = TIFFStack(ddir, arec)
        
        # get the surround of each roi for background subtraction
        Bkg, Halo = halo_subt(self.ROI_coords, 20, stack.nx, stack.ny, zonez=5)
        # extract ROIs
        print "starting to extract ROIs"
        ROI = stack.get_rois(self.ROI_coords)
        print "got ROIs"
        print "starting to extract background..."
        bROI = stack.get_rois(Halo)
        print "got surround of ROIs"

        # Finally save Ca traces for later analysis
        save_catraces(self.ipath, self.name, self.roi_id, ROI, bROI)

    # plot calcium traces along with color coded brainstates
    def plot_rois(self, corr):
        """
        plot rois by calling function in InscopixAnalyze.py
        """
        if os.path.isfile(os.path.join(self.ipath, self.name, 'remxidx_%s.txt'%self.name)):
            plot_catraces(self.ipath, self.name, self.roi_id, cf=corr)
        else:
            plot_catraces_simple(self.ipath, self.name, self.roi_id, cf=corr, SR=10)


    def save_rois(self):
        #transform gui ROI format to save format
        self.set_roicoords()
        save_roilist(self.ipath, self.name, self.ROI_coords, self.ROIs, roi_id=self.roi_id)
        print "Saved roi list"

        
    def load_rois(self):
        wildcard = "*roilist*"
        #dialog = wx.FileDialog(None, "Choose a ROI list", os.path.join(self.ipath, self.name), "", wildcard, wx.OPEN)
        #09/08/17: no idea why wildcard is not working anymore??
        dialog = wx.FileDialog(None, "Choose a ROI list", os.path.join(self.ipath, self.name), "", ".*", wx.FD_OPEN)
        if dialog.ShowModal() == wx.ID_OK:
            self.roi_name = dialog.GetFilename()
            fname_base = 'recording_' + self.name + '_roilistn'
            a = re.search('^' + fname_base + "(\d+)", self.roi_name)
            self.roi_id = int(a.group(1))

            (self.ROI_coords, self.ROIs) = load_roilist(self.ipath, self.name, self.roi_id)
            #nroi = len(self.ROIs)
            #cmap = plt.get_cmap('jet')
            #cmap = cmap(range(0, 256))[:,0:3]
            #self.cmap = downsample_matrix(cmap, int(np.floor(256/nroi)))
            self.set_cmap()

            print "ROI set chosen..."


    def set_cmap(self) :
        nroi = len(self.ROIs)
        cmap = plt.get_cmap('jet')
        cmap = cmap(range(0, 256))[:,0:3]
        self.cmap = downsample_matrix(cmap, int(np.floor(256/nroi)))
    
            
    def set_roicoords(self):
        """
        get all the pixels within an roi
        """
        Coords = []
        for r in self.ROIs:
            (X, Y) = (r[0], r[1])
            xmin = int(round(min(X)))
            xmax = int(round(max(X)))
            ymin = int(round(min(Y)))
            ymax = int(round(max(Y)))

            x, y = np.meshgrid(np.arange(xmin,xmax+1), np.arange(ymin,ymax+1))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x,y)).T
            points = [Point(i) for i in points]

            (x0,y0) = (r[0][0], r[1][0])
            polyg = Polygon(zip(r[0], r[1]) + [(x0,y0)])

            tmp = [polyg.contains(i) for i in points]
            idx = [np.nonzero(np.array(tmp)==True)][0]
            xin = x[idx]
            yin = y[idx]
            xin = [i for i in xin]
            yin = [i for i in yin]            
            Coords.append((xin,yin))

        self.ROI_coords = Coords




                        

# adding functionality to run as script:
if __name__ == '__main__':
    # adjust to your system:
    ipath = '/Volumes/BB8/Penn/Data/RawImaging'
    ipath = '/Volumes/Transcend/Miniscope/'

    ##################################3
    import sys

    def param_check(x):
        params = {}
        for i in range(0, len(x)):
            a = x[i]
            if re.match('^-', a):
                if i < len(x)-1 :
                    params[a] = x[i+1]
    
        return params

    args = sys.argv[1:]
    params = param_check(args)


    if '-i' in params:
        ipath = params['-i']        
    print("Using %s as base folder" % ipath)


    app = wx.App(False)
    app.frame = ImageViewer(ipath)
    app.frame.Show()
    app.MainLoop()


