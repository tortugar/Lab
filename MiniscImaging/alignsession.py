import imaging
from matplotlib.widgets import Button
#from PIL import Image
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import scipy.io as so
import os
import matplotlib.pylab as plt
import numpy as np
import re

import pdb


### SUBROUTINES #################

class DoneButton :
    """
    class to be assocated with a matplotlib object;
    once an action happens (e.g. button press)
    close the figure
    """
    def __init__(self, plt) :
        self.done = 0
        self.plt = plt
        
    def done_pressed(self, event):
        """
        close figure once done button bas been pressed
        """
        self.done = 1
        self.plt.close()
        


class PixelPicker:
    def __init__(self, axes):
        self.axes = axes
        self.x = []
        self.y = []
        self.cid = self.axes.figure.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        print('click', event)
        if event.inaxes==self.axes:
            self.x.append(event.xdata)
            self.y.append(event.ydata)
            self.axes.plot(self.x, self.y, 'ro')
            self.axes.figure.canvas.draw()
            print("added point")
        

    def get_points(self):
        return (self.x, self.y)


    def __repr__(self):
        print (self.x, self.y)


def rotation_matrix(theta) :
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    

def rotate_points(theta, x) :
    R = rotation_matrix(theta)


def rotation_error(X, Y, theta) :
    R = rotation_matrix(theta)
    Yr = np.dot(R, Y)
    D = X-Yr
    return np.sum(np.multiply(D, D))

    
def rottrans_matrix(theta, a, b) :
    """
    combined rotation and translation matrix
    rotate by angle theta, and translate 
    """
    R = np.array([[np.cos(theta), -np.sin(theta), a], [np.sin(theta), np.cos(theta), b], [0.0, 0.0, 1.0]],dtype='float64')
    return R


def rottrans_error(X, Y, theta, a, b) :
    """
    rotate points Y by angle theta and shift by vector (a,b);
    The calculate the mean squared error between X and the transformed Y
    """
    R = rottrans_matrix(theta, a, b)
    Ya = np.ones((3, Y.shape[1]))
    Ya[0:2,:] = Y

    #pdb.set_trace()
    Yr = np.dot(R, Ya)
    Yr = Yr[0:2,:]
    D = X - Yr

    return np.sum(np.multiply(D, D))


### draw ROIs
def draw_rois(ROIs, axes, c) :
    i=0
    for (x,y) in ROIs :
        l = plt.Line2D(y+[y[0]], x+[x[0]], color=c)
        axes.text(np.max(y)-5, np.min(x)+7, str(i), fontsize=10, color=c,bbox=dict(facecolor='w', alpha=0.))
        axes.add_line(l)
        i = i+1
            
def rottrans_rois(ROIs, theta, a, b, nx, ny):
    ROIs_trans = []
    for (y,x) in ROIs :
        (x, y) = transform_coords(nx, ny, np.array(x), np.array(y))
        X = np.ones((3, len(x)))
        X[0,:] = x
        X[1,:] = y
        R = rottrans_matrix(theta, a, b)
        Xr = np.dot(R, X)
        xr = Xr[0,:]
        yr = Xr[1,:]
        (xr, yr) = transform_coords_back(nx, ny, xr, yr)
        ROIs_trans.append((list(yr), list(xr)))

    return ROIs_trans



def transform_coords(nx, ny, x, y) :
    x = x-ny/2.0
    y = nx-y-nx/2.0
    return (x, y)



def transform_coords_back(nx, ny, x, y):
    x = x+ny/2.0
    y = (-nx+y+nx/2.0)*-1.0
    return (x, y)

    

def align_recordings(ipath, name1, name2, theta, a, b, amap=True, roi_id1=1, roi_id2=1):
    """
    align imaging sessions ipath/name1 and ipath/name2.
    try all angles specified by theta and
    all translations specified by (a,b)
    """
    if amap:
        im1 = so.loadmat(os.path.join(ipath, name1, 'recording_' + name1 + '_actmap.mat'))['mean']
        im2 = so.loadmat(os.path.join(ipath, name2, 'recording_' + name2 + '_actmap.mat'))['mean']
    else:
        im1 = so.loadmat(os.path.join(ipath, name1, 'recording_' + name1 + '_alignedmean.mat'))['mean']
        im2 = so.loadmat(os.path.join(ipath, name2, 'recording_' + name2 + '_alignedmean.mat'))['mean']
        

    plt.figure()
    axes1 = plt.subplot(121)
    img1 = axes1.imshow(im1, vmin=0, vmax=np.percentile(im1, 99.5))
    img1.set_cmap('gray')
    img1.axes.get_xaxis().set_visible(False)
    img1.axes.get_yaxis().set_visible(False)
    axes1.axis('tight')

    axes2 = plt.subplot(122)
    img2 = axes2.imshow(im2, vmin=0, vmax=np.percentile(im2, 99.5))
    img2.set_cmap('gray')
    img2.axes.get_xaxis().set_visible(False)
    img2.axes.get_yaxis().set_visible(False)
    axes2.axis('tight')

    axdone = plt.axes([0.85, 0.01, 0.05, 0.05])

    done = Button(axdone, 'Done')
    button_callback = DoneButton(plt)
    done.on_clicked(button_callback.done_pressed)
    p1 = PixelPicker(axes1)
    p2 = PixelPicker(axes2)
    plt.show(block=True)

    x1 = p1.get_points()
    x2 = p2.get_points()

    X = np.zeros((2,len(x1[0])))
    X[0,:] = np.array(x1[0])-im1.shape[1]/2.0
    X[1,:] = im1.shape[0]-np.array(x1[1])-im1.shape[0]/2.0

    Y = np.zeros((2,len(x2[0])))
    Y[0,:] = np.array(x2[0])-im2.shape[1]/2.0
    Y[1,:] = im2.shape[0]-np.array(x2[1])-im2.shape[0]/2.0


    ### Correct for rotation and translation
    Err = np.zeros((len(a), len(b), len(theta)))
    for (ai,i) in zip(a,range(len(a))) :
        for (bi,j) in zip(b, range(len(b))) :
            for (ti,k) in zip(theta, range(len(theta))) :
                Err[i,j,k] = rottrans_error(X, Y, ti, ai, bi)

    # get optimal parameters
    (ai, bi, thetai) = np.unravel_index(np.argmin(Err), (len(a), len(b), len(theta)))
    a_min = a[ai]
    b_min = b[bi]
    theta_min = theta[thetai]
    print("The optimal roation is %f, the optimal translation is (%f,%f)" % (theta_min, a_min, b_min))

    # rotate and translate reference points to new coordinate system
    Ya = np.ones((3, Y.shape[1]))
    Ya[0:2,:] = Y
    Yr = np.dot(rottrans_matrix(theta_min, a_min, b_min), Ya)[0:2]


    # plot to make sure it's allright
    plt.figure()
    plt.subplot(111)
    plt.plot(X[0], X[1], 'ro')
    plt.plot(Y[0], Y[1], 'bo')
    plt.plot(Yr[0], Yr[1], 'go')
    plt.show()

    
    # draw mean frame of master image; draw rois of master session
    # and transformed ro
    (ROI_coords, ROIs1) = imaging.load_roilist(ipath, name1, roi_id1)
    (ROI_coords, ROIs2) = imaging.load_roilist(ipath, name2, roi_id2)
    ROIs2_trans = rottrans_rois(ROIs2, theta_min, a_min, b_min, im2.shape[0], im2.shape[1])

    plt.figure()
    axes = plt.subplot(111)
    axes.imshow(im1, cmap='gray', vmin=0, vmax=np.percentile(im1, 99.5))
    draw_rois(ROIs1, axes, 'red')
    draw_rois(ROIs2_trans, axes, 'green')
    plt.show()

    return (theta_min, a_min, b_min)
    

def roi_overlap(ROIs1, ROIs2, param1, param2, im1_shape, im2_shape, thr_ovlp=0.5, pplot=False, im1=None):
    """
    rotate/translate rois ROI1 and ROI2 by parameters param1 and param2,
    and then test which ROIs overlap
    """

    ROIs1_trans = rottrans_rois(ROIs1, param1[0], param1[1], param1[2], im1_shape[0], im1_shape[1])
    ROIs2_trans = rottrans_rois(ROIs2, param2[0], param2[1], param2[2], im2_shape[0], im2_shape[1])

    # NOTE: for plotting the variable im1 is missing
    if pplot:
        plt.figure()
        axes = plt.subplot(111)
        axes.imshow(im1, cmap='gray')
        draw_rois(ROIs1, axes, 'red')
        draw_rois(ROIs2_trans, axes, 'green')
        plt.show(block=False)


    # test which ROIs overlap
    ri=0
    roi_map = {}
    for r in ROIs1_trans :
        (x0, y0) = (r[0][0], r[1][0])
        polyg1 = Polygon(list(zip(r[0], r[1])) + [(x0,y0)])

        si=0
        for s in ROIs2_trans :
            (x0, y0) = (s[0][0], s[1][0])
            polyg2 = Polygon(list(zip(s[0], s[1])) + [(x0,y0)])

            if polyg1.intersects(polyg2):
                p = polyg1.intersection(polyg2)
                if (p.area >= polyg1.area*thr_ovlp) or (p.area >= polyg2.area*thr_ovlp):
                    #if roi_map.has_key(ri):
                    if ri in roi_map:
                        roi_map[ri].append(si)
                    else :
                        roi_map[ri] = [si]
            si=si+1
            
        ri=ri+1

    for r in list(roi_map.keys()):
        if len(roi_map[r]) > 1 :
            roi_map.pop(r, None)

    roi_map = [(k,roi_map[k][0]) for k in roi_map.keys()]
            
    return roi_map



def write_trans_parameters(ipath, name1, name2, theta, a, b, pwrite=1) :
    """
    add the rotation and tranlation parameters to the info file
    """
    fid = open(os.path.join(ipath, name2, 'info.txt'), 'r')    
    lines = fid.readlines()
    fid.close()
    line_align = 'ALIGN: %f %f %f\n' % (theta, a, b)
    line_master = 'MASTER: %s\n' % name1

    if name1 == name2 :
        line_align = 'ALIGN: %f %f %f\n' % (0., 0., 0.)

    lines_new = []
    palign = 0
    for l in lines :
        a = re.search('ALIGN:', l)
        if a :
            palign = 1
            k = line_align
        else :
            k=l
        lines_new.append(k)

    if palign == 0:
        lines_new.append(line_align)
    

    lines = lines_new
    lines_new = []
    pmaster = 0
    for l in lines :
        a = re.search('MASTER:', l)
        if a :
            pmaster = 1
            k = line_master
        else :
            k=l
        lines_new.append(k)

    if pmaster == 0:
        lines_new.append(line_master)
    lines = lines_new

    # write new info file for recording name2
    if pwrite==1 :
        fid = open(os.path.join(ipath, name2, 'info.txt'), 'w')
        [fid.write(l) for l in lines]
        fid.close()

    return lines
    




##################################################


if __name__ == '__main__' :

    ### HERE the script starts
    plt.close('all')
    ipath = '/media/Imaging1/Data/RawImaging'
    name1 = '1224_090116n1'
    name2 = '1224_090516n1'

    name1 = '1187_081716n1'
    name2 = '1187_082216n1'
    #name2 = '1187_091416n1'

    im1 = so.loadmat(os.path.join(ipath, name1, 'recording_' + name1 + '_actmap.mat'))['mean']
    im2 = so.loadmat(os.path.join(ipath, name2, 'recording_' + name2 + '_actmap.mat'))['mean']

    # fig = plt.figure()
    # axes1 = plt.subplot(121)
    # img1 = axes1.imshow(im1)
    # img1.set_cmap('gray')
    # img1.axes.get_xaxis().set_visible(False)
    # img1.axes.get_yaxis().set_visible(False)
    # axes1.axis('tight')

    # axes2 = plt.subplot(122)
    # img2 = axes2.imshow(im2)
    # img2.set_cmap('gray')
    # img2.axes.get_xaxis().set_visible(False)
    # img2.axes.get_yaxis().set_visible(False)
    # axes2.axis('tight')

    # axdone = plt.axes([0.85, 0.01, 0.05, 0.05])

    # done = Button(axdone, 'Done')
    # button_callback = DoneButton(plt)
    # done.on_clicked(button_callback.done_pressed)
    # p1 = PixelPicker(axes1)
    # p2 = PixelPicker(axes2)


    # plt.show(block=True)

    # x1 = p1.get_points()
    # x2 = p2.get_points()

    # X = np.zeros((2,len(x1[0])))
    # X[0,:] = np.array(x1[0])-im1.shape[1]/2.0
    # X[1,:] = im1.shape[0]-np.array(x1[1])-im1.shape[0]/2.0

    # Y = np.zeros((2,len(x2[0])))
    # Y[0,:] = np.array(x2[0])-im2.shape[1]/2.0
    # Y[1,:] = im2.shape[0]-np.array(x2[1])-im2.shape[0]/2.0

    # ### Correct for rotation only
    # theta = np.arange(0, 2*np.pi, 0.01)
    # err = [rotation_error(X, Y, t) for t in theta]
    # theta_min = theta[np.argmin(err)]
    # theta_deg = theta_min * (180 / np.pi)
    # Yr = np.dot(rotation_matrix(theta_min), Y)

    # ### Correct for rotation and translation
    # theta = np.arange(0, 2*np.pi, 0.1)
    # a = np.arange(-30., 30.)
    # b = np.arange(-30., 30.)
    # Err = np.zeros((len(a), len(b), len(theta)))
    # for (ai,i) in zip(a,range(len(a))) :
    #     print i
    #     for (bi,j) in zip(b, range(len(b))) :
    #         for (ti,k) in zip(theta, range(len(theta))) :
    #             Err[i,j,k] = rottrans_error(X, Y, ti, ai, bi)

    # # get optimal parameters
    # (ai, bi, thetai) = np.unravel_index(np.argmin(Err), (len(a), len(b), len(theta)))
    # a_min = a[ai]
    # b_min = b[bi]
    # theta_min = theta[thetai]
    # print "The optimal roation is %f, the optimal translation is (%f,%f)" % (theta_min, a_min, b_min)


    # Alternative version:
    theta = np.arange(0, 2*np.pi, 0.1)
    a = np.arange(-30., 30.)
    b = np.arange(-30., 30.)
    (theta_min,a_min,b_min) = align_recordings(ipath, name1, name2, theta, a, b)

    # # rotate and translate reference points to new coordinate system
    # Ya = np.ones((3, Y.shape[1]))
    # Ya[0:2,:] = Y
    # Yr = np.dot(rottrans_matrix(theta_min, a_min, b_min), Ya)[0:2]


    # # plot to make sure it's allright
    # plt.figure()
    # plt.subplot(111)
    # plt.plot(X[0], X[1], 'ro')
    # plt.plot(Y[0], Y[1], 'bo')
    # plt.plot(Yr[0], Yr[1], 'go')
    # plt.show(block=False)


    A = Image.fromarray(im1)
    B = A.rotate(theta_min)
    plt.figure()
    axes = plt.subplot(221)
    img = axes.imshow(np.asarray(A), cmap='gray')
    img.set_clim((im1.min(), im1.max()))
    plt.title('reference')


    plt.subplot(222)
    plt.imshow(im2, cmap='gray')
    plt.clim((im1.min(), im1.max()))
    plt.title('2nd image')


    plt.subplot(223)
    plt.imshow(np.asarray(B), cmap='gray')
    plt.clim((im1.min(), im1.max()))
    plt.title('rotated image')
    plt.show(block=False)


    # draw mean frame of master image; draw rois of master session
    # and transformed rois
    (ROI_coords, ROIs1) = load_roilist(ipath, name1, 1)
    (ROI_coords, ROIs2) = load_roilist(ipath, name2, 1)
    ROIs2_trans = rottrans_rois(ROIs2, theta_min, a_min, b_min, im2.shape[0], im2.shape[1])

    plt.figure()
    axes = plt.subplot(111)
    axes.imshow(im1, cmap='gray')
    draw_rois(ROIs1, axes, 'red')
    draw_rois(ROIs2_trans, axes, 'green')

    plt.show(block=False)

    # test which ROIs overlap
    ri=0
    roi_map = {}
    for r in ROIs1 :
        (x0, y0) = (r[0][0], r[1][0])
        polyg1 = Polygon(zip(r[0], r[1]) + [(x0,y0)])

        si=0
        for s in ROIs2_trans :
            (x0, y0) = (s[0][0], s[1][0])
            polyg2 = Polygon(zip(s[0], s[1]) + [(x0,y0)])

            if polyg1.intersects(polyg2) :
                p = polyg1.intersection(polyg2)
                if (p.area >= polyg1.area/2.0) or (p.area >= polyg2.area/2.0) :
                    if roi_map.has_key(ri) :
                        roi_map[ri].append(si)
                    else :
                        roi_map[ri] = [si]
            si=si+1
            
        ri=ri+1

    for r in roi_map.keys() :
        if len(roi_map[r]) > 1 :
            roi_map.pop(r, None)

    print(roi_map)



### TRASH
#cid = fig.canvas.mpl_connect('button_press_event', onclick)
#cid = fig.canvas.mpl_connect('button_press_event', on_key_down)





