import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
sys.path.append('/Users/joshuaneronha/Documents/Brown/Research/LeakyWaveML/1d/peaks_to_slot/constant_power')
import pickle
import tensorflow as tf

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2

from preprocess import import_data, import_val_data

###

# slots, peaks = import_data()

# peaks.shape
# selected_peaks = peaks[peaks[:,1] <0 ]
# plt.boxplot(selected_peaks)
# plt.plot(np.mean(peaks,axis=0))

###

with open('1d/peaks_to_slot/constant_power/results/test_data.pkl','rb') as file:
    test_data = pickle.load(file)
    peaks, true, pred = test_data

val_slots, val_peaks = import_val_data('1d/peaks_to_slot/constant_power/results/validation')

grey_slots, grey_peaks = import_val_data('1d/peaks_to_slot/constant_power/results/grey_trans')

plt.imshow(pred[0])

len(grey_peaks)

plt.plot(peaks[6])
plt.plot(val_peaks[6])
plt.plot(grey_peaks[6])
plt.legend(['Reference','Binary','Greyscale'])

# correct / (len(val_slots) * 36)
# correct = 0
# for i in np.arange(len(val_slots)):
#     correct += (val_slots[i] == true[i]).sum()
# val_slots

mse_list_binary = []
mse_list_grey = []
bucket_1 = []
bucket_2 = []
bucket_3 = []
bucket_4 = []
for i in np.arange(500):
    mse_b = ((peaks[i] - val_peaks[i]) ** 2).mean()
    mse_list_binary.append(mse_b)

    mse_g = ((peaks[i] - grey_peaks[i]) ** 2).mean()
    mse_list_grey.append(mse_g)
    if mse_b < 0.0166:
        bucket_1.append(i)
    elif mse_b < 0.03236:
        bucket_2.append(i)
    elif mse_b < 0.04811:
        bucket_3.append(i)
    else:
        bucket_4.append(i)

fig, axs = plt.subplots(2,2,tight_layout=True,figsize=(6,4))

N, bins, patches = axs[0,0].hist(mse_list_binary,bins=20,ec='black',color='#7fcdbb', label='_nolegend_')
col_list = ['#7fcdbb','#7fcdbb','#1d91c0','#1d91c0','#225ea8','#225ea8','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84']

for i,thispatch in enumerate(patches):
    thispatch.set_facecolor(col_list[i])

axs[0,0].set_xlabel('Mean Square Error')
axs[0,0].set_ylabel('Count')

## bucket_1 example (22.8%)
axs[0,1].plot(peaks[22], color = 'black')
axs[0,1].plot(val_peaks[22], color = '#7fcdbb')
axs[0,1].set_xlabel('Peak Count')
axs[0,1].set_ylabel('Amplitude')

# bucket_2 example (37.4 %)

axs[1,0].plot(peaks[408], color = 'black')
axs[1,0].plot(val_peaks[408], color = '#1d91c0')
axs[1,0].set_xlabel('Peak Count')
axs[1,0].set_ylabel('Amplitude')
# plt.legend(['Objective Peaks','ML Peaks'])
# ((peaks[408] - val_peaks[408]) ** 2).mean()

# bucket_3 example (24.0%)
axs[1,1].plot(peaks[404], color = 'black')
axs[1,1].plot(val_peaks[404], color = '#225ea8')
axs[1,1].set_xlabel('Peak Count')
axs[1,1].set_ylabel('Amplitude')
# plt.legend(['Objective Peaks','ML Peaks'])
# ((peaks[404] - val_peaks[404]) ** 2).mean()

# # bucket_4 example (15.8%)
#
# plt.plot(peaks[157])
# plt.plot(val_peaks[160])
# ((peaks[157] - val_peaks[157]) ** 2).mean()
# plt.legend(['Objective Peaks','ML Peaks'])
# ((peaks[157] - val_peaks[157]) ** 2).mean()

plt.figlegend(['Reference','ML'], loc = 'lower center')

plt.savefig('paper/figures/fig2.eps')

def top_16(slot):

    median = np.sort(slot)[18]
    out = [1 if x >= median else 0 for x in slot]
    return out

fig, ax = plt.subplots(1,4,gridspec_kw={'width_ratios': [6, 1, 1, 1]})
ax[0].plot(peaks[408])
ax[0].set_xlabel('Peak Count')
ax[0].set_ylabel('Amplitude')
# ax[0].plot(val_peaks[0])
ax[1].imshow(tf.expand_dims(true[408],1),cmap='YlGnBu')
ax[1].axes.xaxis.set_visible(False)
ax[1].axes.yaxis.set_visible(False)
ax[2].imshow(tf.expand_dims(pred[408],1),cmap='YlGnBu')
ax[2].axes.xaxis.set_visible(False)
ax[2].axes.yaxis.set_visible(False)
forcmap = ax[3].imshow(tf.expand_dims(top_16(pred[408]),1),cmap='YlGnBu')
ax[3].axes.xaxis.set_visible(False)
ax[3].axes.yaxis.set_visible(False)
# cax = ax[3].inset_axes([1.04, 0.2, 0.05, 0.6], transform=ax[3].transAxes)
fig.colorbar(forcmap)

fig.savefig('paper/figures/fig3b.eps')

fig, axs = plt.subplots(1, 1, tight_layout=True)
N, bins2, patches = plt.hist(mse_list_grey,bins=bins,ec='black',color='#7fcdbb')
fig.show()

col_list = ['#7fcdbb','#7fcdbb','#1d91c0','#1d91c0','#225ea8','#225ea8','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84']

for i,thispatch in enumerate(patches):
    thispatch.set_facecolor(col_list[i])
#
#
