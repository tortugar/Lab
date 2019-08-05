#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 20:43:42 2017
@author: tortugar
"""
import detect_contour as dc
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib import cm
from descartes import PolygonPatch
from matplotlib.colors import LinearSegmentedColormap

import cv2
import numpy as np

import os.path

# location of images
ipath    = '/Users/tortugar/Google Drive/Penn/Programming/VirusExpression/Data/dPGi'
out_path = '/Users/tortugar/Google Drive/Penn/Programming/VirusExpression/Data/Result'

# mouse list
mice = ['G104', 'G222', 'G129', 'G175']
# Referenz images
refs = ['67', '69', '71']

refs = ['1']
mice = ['JS30', 'JS40', 'JS43', 'JS60', 'JS61', 'JS63', 'JS66', 'JS67', 'JS68', 'JS69', 'JS70', 'JS81', 'JS82', 'JS87', 'JS88', 'JS93', 'JS94', 'JS118']

single_polygon = False

plt.ion()
for rimg in refs:
    
    print("starting with reference image %s" % rimg)
    ifile = os.path.join(ipath, 'fig%s_mask.png' % rimg)
    polygon_list = []

    i=0
    for m in mice:
        print("processing mouse %s" % m)
        name = m + '_%s.png' % rimg
        if not single_polygon:
            (cnt, nx, ny) = dc.expr_contour2(ipath, name, xborder=0, yborder= 0)        
            #polygon_list.append(dc.contour2polygon(cnt))
            polygon_list += [dc.contour2polygon(c) for c in cnt]
        else:
            (cnt, nx, ny) = dc.expr_contour(ipath, name, xborder=0, yborder= 0)        
            #polygon_list.append(dc.contour2polygon(cnt))
            
        i += 1
    
    HM = dc.polygon_overlap(polygon_list, nx, ny)

    hm_polyg, color_ids = dc.hm2polygons(HM)
    
    fig = plt.figure()
    ax = plt.subplot(111)    
    clrs = cm.Greens(range(256))
    nd = max(color_ids)+1
    clrs = dc.downsample_matrix(clrs, int(np.floor(256/nd)))
    plt.set_cmap(cm.Greens)
    
    i=0
    for pg in hm_polyg:
        if not(pg==[]):
            p = PolygonPatch(pg, ec=clrs[color_ids[i]-1,:], fc=clrs[color_ids[i]-1,:])
            p.set_linewidth(3)       
            ax.add_patch(p)
        i += 1
    
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.xlim((0,ny))
    plt.ylim((0,nx))    
    axc = fig.add_axes([0.8, 0.15, 0.03, 0.18])
    cmap = mpl.cm.Greens
    norm = mpl.colors.Normalize(vmin=1, vmax=len(mice))
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    
    # load reference image in gray scale
    img = cv2.imread(ifile, 0)
    #C = np.zeros(img.shape)
    #C[np.nonzero(img)] = 1


    #my_map = LinearSegmentedColormap.from_list('ha2', [[0,0,0],[1,1,1]], 2)
    #cm.register_cmap('ha2')    
    #ax.imshow(np.flipud(C), cmap=my_map)   
    # plot with gray scale colormap
    
    ax.imshow(np.flipud(img), cmap='gray')
    
    plt.savefig(os.path.join(out_path, 'fig_%s-%s_%s.pdf' % (mice[0], mice[-1], rimg)))
      
    