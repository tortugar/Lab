#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 16:34:35 2017
@author: tortugar
"""

import sys
sys.path.append('/Users/tortugar/Google Drive/Berkeley/Data/Programming/CommonModules')

import cv2
import os.path
import numpy as np
import matplotlib.pylab as plt
#import Utility as ut
import matplotlib as mpl

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib import cm
from descartes import PolygonPatch
from matplotlib.colors import LinearSegmentedColormap
import pdb



def downsample_matrix(X, nbin) :
    """
    y = downsample_matrix(X, nbin)
    downsample the matrix X by replacing nbin consecutive \
    rows by their mean \
    @RETURN: the downsampled matrix 
    """
    n_down = int(np.floor(X.shape[0] / nbin))
    X = X[0:n_down*nbin,:]
    X_down = np.zeros((n_down, X.shape[1]))

    # 0 1 2 | 3 4 5 | 6 7 8 
    for i in range(nbin) :
        idx = range(i, int(n_down*nbin), int(nbin))
        X_down += X[idx,:]

    return X_down / nbin
    


def expr_contour(ipath, name, xborder=200, yborder=300):
    
    
    ifile = os.path.join(ipath, name)
    img = cv2.imread(ifile)
    nx = img.shape[0]
    ny = img.shape[1]
    
    # get red channel
    #img_cut = img[xborder+1:-xborder,yborder+1:-yborder,2]
    img_cut = img[:,:,2]

    #b,g,r = cv2.split(img_cut)    
    #img_cut = r-b

    # threshold image
    # before 200 for lower value
    ret,thresh = cv2.threshold(img_cut,200,255,cv2.THRESH_BINARY)
    #pdb.set_trace()
    _,contours,h = cv2.findContours(thresh, 1, 2)
    
    print(len(contours))
    
    if len(contours) == 1:
        print(name)
    
    if len(contours) > 2 :
        print("Something weired going on: detected more than one patch")
    
    
    cnt = contours[0]
    # add xborder back to pixel coordinates
    for i in range(len(cnt)):
        cnt[i][0][0] += yborder
        cnt[i][0][1] += xborder
    
    #pdb.set_trace()
    
    return cnt, nx, ny
    


def expr_contour2(ipath, name):
    """
    return all red contours (polygons) in image. That is,

    :param ipath:
    :param name: file name of image with virus expression outline(s) in RED. Ideally,
                 the picture has BLACK background, the outline of the atlas section
                 is in BLUE.

    :return contours: list of polyons (outer outlines)
    :return nx, ny: size of image $name
    
    Note: In contrast to expr_contour, a histology image can have more than just one contour
    """    
    
    ifile = os.path.join(ipath, name)
    img = cv2.imread(ifile)
    nx = img.shape[0]
    ny = img.shape[1]
    
    # get red channel
    img_cut = img[:,:,2]

    # threshold image
    ret,thresh = cv2.threshold(img_cut,200,255,cv2.THRESH_BINARY)
    # only detect outer boundary of polygons (that's why cv2.RETR_EXTERNAL)
    # see https://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html for explanation
    res = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)
    
    # some version of cv2 return 3 parameters, others just 2
    if len(res) == 3:
        contours = res[1]
    else:
        contours = res[0]
    
    return contours, nx, ny
    

    
def contour2polygon(cnt, plotorder=-1, nx=0):
    #create polygon
    L = []
    P = []
    if len(cnt) >= 3:
        for i in range(len(cnt)):

            if plotorder == -1:
                L.append((cnt[i][0][1], cnt[i][0][0]))
                #L.append((cnt[0][0][1], cnt[0][0][0]))        
            elif plotorder == 1:
                L.append((cnt[i][0][0], nx-1-cnt[i][0][1]))
                #L.append((cnt[0][0][0], nx-cnt[0][0][1]))            
            else:
                L.append((cnt[i][0][0], cnt[i][0][1]))
        P = Polygon(L)
            
    return P



def contour2polygon_fast(cnt):
    #create polygon
    L = []
    for i in range(len(cnt)):
        L.append((cnt[i][0][1], cnt[i][0][0]))
    L.append((cnt[0][0][1], cnt[0][0][0]))        
    return Polygon(L)



def polygon_overlap(poly_list, nx, ny):
    
    # determine min/max x and y value
    xmin = min([int(min(x.exterior.coords.xy[0])) for x in poly_list])
    xmax = max([int(max(x.exterior.coords.xy[0])) for x in poly_list])
    
    ymin = min([int(min(x.exterior.coords.xy[1])) for x in poly_list])
    ymax = max([int(max(x.exterior.coords.xy[1])) for x in poly_list])
    
    MX = np.zeros((nx,ny))
    for i in range(xmin,xmax+1):
        for j in range(ymin, ymax+1):
            pt = Point((i,j))
            for polyg in poly_list:
                if polyg.contains(pt):
                    MX[i,j] += 1
    return MX


def hm2polygons(MX):
    
    noverlap = int(MX.max())
    polyg = []
    colors = []
    for i in range(1,noverlap+1):
        C = np.zeros(MX.shape, dtype=np.uint8)
        ii = np.where(MX==i)
        C[ii] = 255
        
        _,contours,h = cv2.findContours(C,1,2)
        for cnt in contours:
            P = contour2polygon(cnt, plotorder=1, nx=MX.shape[0])
            polyg.append(P)
            colors.append(i)
        
    return polyg, colors
    

###############################################################################
### Correlation Analysis ######################################################    
###############################################################################

def histo_grid(ipath, name, dx, xborder=0, yborder=0):
    """
    load histology picture with virus expression pattern, 
    overlay the picture with a grid and
    determine for each grid cell whether it overlap or not with virus expression
    :param ipath, name: the histology picture is named $name in folder $ipath
    :param dx: resoluation of grid (number of pixels per grid cell)
    :xborder: 0 or tuple of int allows specifing boundaries for x values within the picture
              xborder[0] is the start value for the grid along the x axis (vertical axis, starting on top),
              xborder[1] is the end value of the grid along x axis.
              Similarly, for yborder
    
    Example call: 
        dc.histo_grid(ipath, 'JS63_1.png', 15, yborder=[80, 320], xborder=[150, 300]) 
    
    Note x and y axis:
        y axis goes left to right
        x axis goes top to bottom
    """
    
    # load virus expression outline
    img = cv2.imread(os.path.join(ipath, name))
    cnt, nx, ny = expr_contour2(ipath, name, xborder=0, yborder=0) 
    poly = [contour2polygon(c, 2, nx) for c in cnt]
    
    plt.figure()
    fax = plt.subplot(111)
    [fax.add_patch(PolygonPatch(p)) for p in poly]

    if xborder == 0:
        ax = 0
        bx = nx
    else:
        (ax, bx) = xborder
    if yborder == 0:
        ay = 0
        by = ny
    else:
        (ay, by) = yborder
    
    # apply discrete grid 
    gx = np.arange(ax, bx, dx)
    gy = np.arange(ay, by, dx)
    mx = np.zeros((len(gx), len(gy)))
    
    fax.imshow(img)
    i = 0
    for x in gx:
        j = 0
        for y in gy:
            # in shapley x-axis goes down from upper left corner
            # y-axis goes to the right
            # In other words, x and y axis have to exchanged compared to 
            # matplotlib
            p = Polygon([(y, x), (y+dx, x), (y+dx, x+dx), (y, x+dx)])
            if True in [p.intersects(contour2polygon(q, 2, nx)) for q in cnt]:
                mx[i,j] = 1
                pplt = PolygonPatch(p, ec='white', fc='white', alpha=0.5)   
            else:
                pplt = PolygonPatch(p, ec='white', fc='black', alpha=0.5)   
                 
            fax.add_patch(pplt)
            j+=1
        i+=1
        
    return mx
    



if __name__ == '__main__':
    
    #mice = ['G102', 'G101', 'G93', 'G106', 'G104', 'GADVLP20', 'G153', 'GADVLP19', 'GADVLP17']
    #mice = ['G128', 'G222']
    mice = ['JS82', 'JS81']
    mice = ['G104']

    #ipath = '/Volumes/Picture1/Pictures/vlPAG_histo/Fitting'
    ipath    = '/Users/tortugar/Google Drive/Penn/Programming/VirusExpression/Data/Joe'
    name = 'JS81_1.png'

    ipath    = '/Users/tortugar/Google Drive/Penn/Programming/VirusExpression/Data'

    
    ifile = os.path.join(ipath, 'fig67_mask.png')
    img = cv2.imread(ifile)
    #(cnt, nx, ny) = expr_contour(ipath, name)
    #img = np.zeros((nx, ny,3), dtype=np.uint8)
    

    polygon_list = []
    i=0
    for m in mice:
        name = m + '_67.png'
        (cnt, nx, ny) = expr_contour(ipath, name)        
        img=cv2.drawContours(img,[cnt],-1,(0,0,100),3)
        #polygon_list.append(contour2polygon(cnt))
        i += 1
    
    
    
#    for i in range(len(cnt)):
#        img[cnt[i][0][0], cnt[i][0][1]] = 1
        
    #create polygon
#    L = []
#    for i in range(len(cnt)):
#        L.append((cnt[i][0][0], cnt[i][0][1]))
#        
#    P = Polygon(L)
#
#    z=mapping(P)['coordinates'][0]
#
#    for i in z:
#        img[i[0],i[1]] = 1

    #cv2.drawContours(img,[cnt],-1,(1),thickness=1)
    
    
    plt.figure()
    
    #img2 = img[:,:,0]
    plt.imshow(img)
    
    HM = polygon_overlap(polygon_list, nx, ny)

    plt.figure()
    plt.imshow(HM, cmap=cm.Greens)
    
    out_path = '/Users/tortugar/Documents/TTT'
    plt.savefig(os.path.join(out_path, 'fig_bla.pdf'))
    
    #%%
    fig = plt.figure()
    ax = plt.subplot(111)
    #pg = Polygon([(1,1), (2,3), (1,1)])
    #pg = Polygon(L)
    #pg = polygon_list[0]
    
    #pg = Polygon([(0, 0), (0, 2), (1, 1)])
    
    
    #p = PolygonPatch(pg, fc=[0.2, 0,0], ec='blue')
    
    

    hm_polyg, color_ids = hm2polygons(HM)
    
    
    clrs = cm.Greens(range(256))
    nd = max(color_ids)+1
    clrs = ut.downsample_matrix(clrs, int(np.floor(256/nd)))
    plt.set_cmap(cm.Greens)
    
    i=0
    for pg in hm_polyg:
        if not(pg==[]):
            p = PolygonPatch(pg, ec=clrs[color_ids[i]-1,:], fc=clrs[color_ids[i]-1,:])
            ax.add_patch(p)
        i += 1
    
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.xlim((0,ny))
    plt.ylim((0,nx))    
    axc = fig.add_axes([0.8, 0.15, 0.03, 0.18])
    cmap = mpl.cm.Greens
    norm = mpl.colors.Normalize(vmin=1, vmax=max(color_ids))
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    
    
    img = cv2.imread(ifile, 0)
    C = np.zeros(img.shape)
    C[np.nonzero(img)] = 1


    my_map = LinearSegmentedColormap.from_list('ha2', [[0,0,0],[1,1,1]], 2)
    #plt.set_cmap(my_map)
    cm.register_cmap('ha2')    
    ax.imshow(np.flipud(C), cmap=my_map)   

    
    
    out_path = '/Users/tortugar/Documents/TTT'
    plt.savefig(os.path.join(out_path, 'fig_bla.pdf'))
    
    
    
    