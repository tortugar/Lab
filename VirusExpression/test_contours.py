#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:20:31 2019

@author: tortugar
"""
import cv2
import os

ipath    = '/Users/tortugar/Google Drive/Penn/Programming/VirusExpression/Data/Joe'
ifile = 'JS81_1.png'

img = cv2.imread(os.path.join(ipath, ifile))


img_cut = img[:,:,2]
ret,thresh = cv2.threshold(img_cut,200,255,cv2.THRESH_BINARY)
_,contours,h = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)

ref_img = cv2.imread(os.path.join(ipath, 'fig1_mask.png'))
i = 0
for cnt in contours:
    cv2.drawContours(ref_img,[cnt],0,(50+i*20,50+i*20,0),1) 
    i+=1

cv2.imshow('', ref_img)
