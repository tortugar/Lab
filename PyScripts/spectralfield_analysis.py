import spyke
import numpy as np
import matplotlib.pylab as plt 

ppath = '/Volumes/BB8/Penn/Data/RawData'
file_listing = 'REM max units.txt' 
units = spyke.load_units(ppath, file_listing)
th = [0, 100, 1000, 10000, 100000, 1000000]
ipre = 10
ipost = 10

spec = {k:[] for k in units}
unit_perf = {}
for k in units:
    nunits = 0
    perf = 0
    for rec in units[k]:
        print("Working on recording", rec)
        kern, t, f, perf_max = spyke.spectralfield(ppath, rec.name, rec.grp, rec.un, ipre, ipost, states=[1], theta=th, nsmooth=3, fmax=20)
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
plt.pcolormesh(t, f, spec_mx[idx,:,:].mean(axis=0), cmap='bwr')
plt.xlabel('Time (s)')
plt.ylabel('Freq. (Hz)')
plt.colorbar()





        





