import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as signal
import tensorflow_io as tfio
import scipy

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2

def import_data():
    """
    Loads in data and transposes it into proper shapes
    """

    slots_list = []
    peaks_list = []
    max_list = []

    with open('/Users/joshuaneronha/Documents/Brown/Research/LeakyWaveML/comsol_results/1dconstantslots/1648578904.csv', 'rb') as file:
        results = np.loadtxt(file, delimiter=",", dtype=float)
        num_sims = int(results.shape[0] / 361)
        points = [361 * x for x in np.arange(num_sims + 1)]
        sorted_x = np.array([20*np.log10(results[i:i + 361,1]) for i in points[:-1]])

    return sorted_x

dataa = import_data()
dataa.shape
dataa[3,271:361].shape

floquet_back = [270 + (180 - x) for x in [95,99,101,103,106,109,114,117,120,122,126,128,130,132,146]]
floquet_back

floquet = [5,15,24,28,31,34,48,60,61,63,65,68,70,73,76,80,84,86,88,90] + [95,99,101,103,106,109,114,117,120,122,126,128,130,132,146]



floquet_forward = [5,15,24,28,31,34,48,60,61,63,65,68,70,73,76,80,84,86,88,90]
floquet_back
floquet_back = [270 + (180 - x) for x in [95,99,101,103,106,109,114,117,120,122,126,128,130,132,146]]
floquet_back
[450 - x for x in floquet_back]

def plot_profiles(num):

    plt.plot(np.arange(91,181),np.flip(dataa[num,271:361]),color = '#0c2c84')
    plt.plot(np.arange(0,91),np.flip(dataa[num,0:91]), color = '#0c2c84')

    plt.scatter([90 - x for x in floquet_forward], dataa[num,floquet_forward],color = '#7fcdbb')
    plt.scatter([450 - x for x in floquet_back], dataa[num,floquet_back], color = '#7fcdbb')

plot_profiles(14)

plt.plot(np.arange(0,180),dataa[14,0:180], color = '#0c2c84')
plt.scatter(floquet,dataa[14,floquet])

plt.plot(dataa[14,0:360])

floquet_back_fake = [95,99,101,103,106,109,114,117,120,122,126,128,130,132,146]


## floquet plotting

freq = 200e9
lambdaa = 3e8/freq
k0 = 2*np.pi/lambdaa
h = 1e-3;

Lambda = np.arange(1e-3, 10e-3, 1e-3)
p = np.arange(-80,81,1)

betaz = np.zeros((len(Lambda),len(p)))
neff = np.zeros((len(Lambda),len(p)))
neff_works = np.zeros((1000,4))
neff_works.shape
k=0;
for i in np.arange(len(Lambda)):
    for j in np.arange(len(p)):
        betaz[i,j] = (np.sqrt(k0**2-(np.pi/h)**2)+2*np.pi*p[j]/Lambda[i]);
        neff[i,j] = betaz[i,j]/k0;
        if (neff[i,j] > -1) & (neff[i,j] < 1):
            neff_works[k,:] = [neff[i,j], np.degrees(np.arccos(neff[i,j])), Lambda[i], p[j]];
            k=k+1;

np.round(np.unique(neff_works[0:60,1]))

neff_works[59,3]

fig, axs = plt.subplots(1,2,tight_layout=True,figsize=(8,3.5))
axy = axs[0].scatter(np.arange(59),neff_works[:59,1], c = neff_works[:59,3], cmap = 'GnBu', edgecolors = 'black')
axs[0].set_xlabel('Peak Count')
axs[0].set_ylabel('Angle')
plt.colorbar(axy)
axy.set_clim([-8,2])
axs[0].set_ylim([0, 180])


np.round(neff_works[:59,3])

np.concatenate([neff_works[:59,3],neff_works[:59,1]],axis=1)

plt.imshow()

plt.scatter(neff_works[:59,1],neff_works[:59,3],c='#ffffd9',s=200,marker='s')

plt.scatter(neff_works[:59,1],neff_works[:59,3],c='#ffffd9',s=200,marker='s')
ax = plt.axes()
ax.set_facecolor("#0c2c84")

base_array = np.zeros((151,11))

for i in np.arange(155):
    for j in np.arange(-9,4):
        for check in np.arange(60):
            if (i == np.round(neff_works[check,1])) and (j == neff_works[check,3]):
                base_array[i,j] = 1
            else:
                pass

plt.pcolor(base_array.T,cmap='YlGnBu')

base_array.sum()
base_array[i,j] = 1
plt.pcolor(np.arange(151),np.arange(-8,2),(neff_works[:59,3],neff_works[:59,1]))
ax = plt.axes()
ax.set_facecolor("orange")

plot_profiles(0,1)
axs[1].set_xlabel('Angle')
axs[1].set_ylabel('Amplitude (dB)')
plt.savefig('paper/figures/fig4.eps')

neff_works[:229,1].sort()
len(np.unique(neff_works[:229,1]))

np.unique(neff_works[:229,1])
