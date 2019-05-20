#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to crop a video; uses module vypro, which has to be in the same folder
"""
import vypro
import sys
import re
import os


def param_check(x):
    params = {}
    for i in range(0, len(x)):
        a = x[i]
        if re.match('^-', a):
            if i < len(x)-1:
                params[a] = x[i+1]

    return params


#### start of script ##############################################################
args = sys.argv[1:]
if len(args) == 0:
    print("""
    usage:
    python crop_movie.py -m [path_to_movie_stack or path_to_zip_folder] -z [0|1]
    """)
    sys.exit(0)

params = param_check(args)
if '-m' in params:
    movie = params['-m']

rezip = False
if '-z' in params:
    rezip = int(params['-z'])

if re.match('.*\/$', movie):
    movie = movie[:-1]

if re.match('\.zip$', movie):
    (ddir, zip_file) = os.path.split(movie)
    (ppath, name) = os.path.split(ddir)
else:
    (ddir, _) = os.path.split(movie)
    (ppath, name) = os.path.split(ddir)

if not os.path.isdir(os.path.join(ppath, name, 'Stack')):
    print("unzipping Stack.zip")
    vypro.unpack_zipstack(ppath, name)
vypro.crop_frames(ppath, name)

if rezip==1:
    zfile = os.path.join(ppath, name, 'stack.zip')
    if os.path.isfile(zfile):
        os.remove(zfile)
    vypro.zipdir(os.path.join(ppath, name, 'Stack'), zfile)




