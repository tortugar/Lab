# python 3
# please make sure to use newest version of pyphi

import sys
# to include spyke.py to python path
sys.path.append('/Users/tortugar/Google Drive/Penn/Programming/PySpike')

import spyke
import numpy as np
import matplotlib.pylab as plt 

ppath = '/Volumes/BB8/Penn/Data/RawData'
file_listing = 'REM max units.txt' 

units = spyke.load_units(ppath, file_listing)
th = [0, 100, 1000, 10000, 100000, 1000000]
ipre = 5
ipost = 5
nsmooth = 1
fmax = 20
peeg = True
phighres = True

spec = {k:[] for k in units}
unit_perf = {}
for k in units:
    nunits = 0
    perf = 0
    for rec in units[k]:
        print("Working on recording", rec)
        if phighres:
            kern, t, f, perf_max = spyke.spectralfield_highres(ppath, rec.name, rec.grp, rec.un, ipre, ipost, 
                                                   states=[1], theta=th, nsmooth=nsmooth, fmax=fmax, pzscore=True, pplot=False)
        else:
            kern, t, f, perf_max = spyke.spectralfield(ppath, rec.name, rec.grp, rec.un, ipre, ipost, peeg=peeg, perc_overlap=0.9,
                                                   states=[1], theta=th, nsmooth=nsmooth, fmax=fmax, pzscore=True, pplot=False)
            
        spec[k].append(kern)
        perf += perf_max
        nunits += 1

    unit_perf[k] = perf_max / nunits
    spec[k] = np.array(spec[k]).mean(axis=0)

keys = list(unit_perf.keys())
unit_perf_list = [unit_perf[k] for k in keys]
spec_list = [spec[k] for k in keys]


#%%
spec_mx = np.array(spec_list)

idx = np.where(np.array(unit_perf_list) > 0.1)[0]
plt.ion()
plt.figure()
if peeg:
    plt.pcolormesh(t, f, spec_mx[idx,:,:].mean(axis=0), cmap='bwr')
    plt.colorbar()
else:
    plt.plot(t, spec_mx[idx,:,:].mean(axis=0).mean(axis=0))
plt.xlabel('Time (s)')
plt.ylabel('Freq. (Hz)')
plt.show()

