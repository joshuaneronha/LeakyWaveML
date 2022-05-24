import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
sys.path.append('/Users/joshuaneronha/Documents/Brown/Research/LeakyWaveML/1d/peaks_to_slot/grey')
import pickle
import tensorflow as tf
from tensorflow import losses

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2

from preprocess import import_data, import_val_data

###

slots, peaks, signals = import_data()

peaks.shape
# selected_peaks = peaks[peaks[:,1] <0 ]

fig, axs = plt.subplots(1,1,tight_layout=True,figsize=(6,4))

boxplot = axs.boxplot(peaks, sym = '', patch_artist = True)
plt.xticks(np.arange(0,36,5),np.arange(0,36,5))
plt.xlabel('Peak Count')
plt.ylabel('Amplitude (normalized)')

cmap = mpl.cm.get_cmap('GnBu')

for index, patch in enumerate(boxplot['boxes']):
    patch.set_facecolor(cmap(index / 36))

plt.savefig('paper/figures/boxgrey.eps')

###
with open('1d/peaks_to_slot/grey/results/test_data_simp.pkl','rb') as file:
    test_data = pickle.load(file)
    peaks, true, pred, waves = test_data

    predb = np.where((pred > 0) & (pred <= 0.125), 0, pred)
    predb = np.where((predb > 0.125) & (predb <= 0.375), 0.25, predb)
    predb = np.where((predb > 0.375) & (predb <= 0.625), 0.50, predb)
    predb = np.where((predb > 0.625) & (predb <= 0.875), 0.75, predb)
    predb = np.where((predb > 0.875), 1.00, predb)

grey_slots, grey_peaks, grey_wves = import_val_data('1d/peaks_to_slot/grey/results/simplelosstest')

grey_slots

with open('1d/peaks_to_slot/grey/results/test_data.pkl','rb') as file:
    test_data = pickle.load(file)
    diff_real_peaks, diff_true, diff_pred = test_data

diff_slots, diff_peaks = import_val_data('1d/peaks_to_slot/grey/results/difflosstest')

def test(i):

    plt.plot(peaks[i])
    plt.plot(grey_peaks[i+6])
    plt.plot(diff_peaks[i])
    plt.legend(['Reference','Simple','Complicated'])

    x = ((peaks[i] - diff_peaks[i]) ** 2).mean()
    y = ((peaks[i] - grey_peaks[i]) ** 2).mean()

    return(x,y)

test(23)

# correct / (len(val_slots) * 36)
# correct = 0
# for i in np.arange(len(val_slots)):
#     correct += (val_slots[i] == true[i]).sum()
# val_slots

bucket_2

# np.array(mse_list_grey).mean()
# # np.array(mse_list_diff).mean()
# np.array(mse_list_control).mean()
mse_list_grey = []
mse_list_diff = []
mse_list_control = []
bucket_1 = []
bucket_2 = []
bucket_3 = []
bucket_4 = []
#
# np.square([-1,2,-3,4,-5])
# np.array([-1,2,-3,4,-5]) ** 2
for i in np.arange(500):

    # mse_g = np.mean(np.square(peaks[i] - grey_peaks[i]))
    mse_g = losses.mse(waves[i][0:181], grey_wves[i][0:181])
    mse_list_grey.append(mse_g)
    #
    # mse_d = ((diff_real_peaks[i] - diff_peaks[i]) ** 2).mean()
    # mse_list_diff.append(mse_d)

    try:
        mse_c = losses.mse(waves[i+1][0:181], grey_wves[i][0:181])
        mse_list_control.append(mse_c)
    except:
        pass

    if mse_g < 0.0166:
        bucket_1.append(i)
    elif mse_g < 0.03236:
        bucket_2.append(i)
    elif mse_g < 0.04811:
        bucket_3.append(i)
    else:
        bucket_4.append(i)

bins = np.array([8.56330802e-25, 6.16616441e-03, 1.23323288e-02, 1.84984932e-02,
       2.46646576e-02, 3.08308220e-02, 3.69969864e-02, 4.31631508e-02,
       4.93293153e-02, 5.54954797e-02, 6.16616441e-02, 6.78278085e-02,
       7.39939729e-02, 8.01601373e-02, 8.63263017e-02, 9.24924661e-02])

fig, axs = plt.subplots(1,1,tight_layout=True,figsize=(5.5,4))

N, bins, patches = axs.hist(tf.stack(mse_list_grey).numpy(),bins=bins,ec='black',color='#7fcdbb', label='_nolegend_')
# col_list = ['#7fcdbb','#7fcdbb','#1d91c0','#1d91c0','#225ea8','#225ea8','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84']
#
for i,thispatch in enumerate(patches):
    # thispatch.set_facecolor(col_list[i])
    thispatch.set_alpha(0.75)

axs.set_xlabel('Mean Square Error')
axs.set_ylabel('Count')

N, bins, patches = axs.hist(tf.stack(mse_list_control).numpy(),bins=bins,ec='black',color='#1d91c0', label='_nolegend_')
# col_list = ['#7fcdbb','#7fcdbb','#1d91c0','#1d91c0','#225ea8','#225ea8','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84']

for i,thispatch in enumerate(patches):
    thispatch.set_alpha(0.5)

plt.ylim([0, 95])
axs.legend(['Model','Random'])
# plt.ylim([0, 100])

plt.savefig('paper/figures/grey_histogram.svg')

mse_list_grey[127]

####

def plot_instance_beta(val):

    fig, ax = plt.subplots(1,4,gridspec_kw={'width_ratios': [1, 1, 1, 10]},figsize=(7,4))
    ax[0].imshow(tf.expand_dims(true[val],1),cmap='YlGnBu')
    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)
    ax[1].imshow(tf.expand_dims(pred[val],1),cmap='YlGnBu')
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    ax[2].imshow(tf.expand_dims(predb[val],1),cmap='YlGnBu')
    ax[2].axes.xaxis.set_visible(False)
    ax[2].axes.yaxis.set_visible(False)# cax = ax[3].inset_axes([1.04, 0.2, 0.05, 0.6], transform=ax[3].transAxes)
    ax[3].plot(waves[val][0:181],color='#0c2c84')
    ax[3].plot(grey_wves[val][0:181],color='#7fcdbb')

    print('MSE = ' + str(losses.mse(peaks[val], grey_peaks[val])),5)
bucket_1
np.mean(np.square(peaks[127] - grey_peaks[127]))
mse_list_grey[127]
plot_instance_beta(71)
plt.savefig('paper/figures/grey71.eps')

bucket_3
###
###
fig, axs = plt.subplots(2,2,tight_layout=True,figsize=(6,4))

N, bins, patches = axs[0,0].hist(mse_list_grey,bins=old_bins,ec='black',color='#7fcdbb', label='_nolegend_')
col_list = ['#7fcdbb','#7fcdbb','#1d91c0','#1d91c0','#225ea8','#225ea8','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84']

for i,thispatch in enumerate(patches):
    thispatch.set_facecolor(col_list[i])

axs[0,0].set_xlabel('Mean Square Error')
axs[0,0].set_ylabel('Count')



###

old_bins = np.array([0.00086428, 0.00873865, 0.01661302, 0.02448739, 0.03236176,
       0.04023613, 0.0481105 , 0.05598487, 0.06385924, 0.07173361,
       0.07960798, 0.08748235, 0.09535672, 0.10323109, 0.11110546,
       0.11897984, 0.12685421, 0.13472858, 0.14260295, 0.15047732,
       0.15835169])

mse_list_diff

mse_list_

fig, axs = plt.subplots(1,2,tight_layout=True,figsize=(10,4))

N, bins, patches = axs[0].hist(mse_list_grey,bins=old_bins,ec='black',color='#7fcdbb', label='_nolegend_')
col_list = ['#7fcdbb','#7fcdbb','#1d91c0','#1d91c0','#225ea8','#225ea8','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84']

for i,thispatch in enumerate(patches):
    thispatch.set_facecolor(col_list[i])

axs[0].set_xlabel('Mean Square Error')
axs[0].set_ylabel('Count')
#

N, bins, patches = axs[1].hist(mse_list_control,bins=old_bins,ec='black',color='#7fcdbb', label='_nolegend_')
col_list = ['#7fcdbb','#7fcdbb','#1d91c0','#1d91c0','#225ea8','#225ea8','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84']

for i,thispatch in enumerate(patches):
    thispatch.set_facecolor(col_list[i])

axs[1].set_xlabel('Mean Square Error')
axs[1].set_ylabel('Count')




def top_16(slot):

    median = np.sort(slot)[18]
    out = [1 if x >= median else 0 for x in slot]
    return out

def plot_instance(val):



    fig, ax = plt.subplots(1,5,gridspec_kw={'width_ratios': [6, 1, 1, 1, 6]},figsize=(12,4))
    ax[0].plot(peaks[val])
    ax[0].plot(grey_peaks[val])
    ax[0].set_xlabel('Peak Count')
    ax[0].set_ylabel('Amplitude')
    # ax[0].plot(val_peaks[0])
    ax[1].imshow(tf.expand_dims(true[val],1),cmap='YlGnBu')
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    ax[2].imshow(tf.expand_dims(pred[val],1),cmap='YlGnBu')
    ax[2].axes.xaxis.set_visible(False)
    ax[2].axes.yaxis.set_visible(False)
    forcmap = ax[3].imshow(tf.expand_dims(predb[val],1),cmap='YlGnBu')

    ax[3].axes.xaxis.set_visible(False)
    ax[3].axes.yaxis.set_visible(False)# cax = ax[3].inset_axes([1.04, 0.2, 0.05, 0.6], transform=ax[3].transAxes)
    ax[4].plot(waves[val][0:181])
    ax[4].plot(grey_wves[val][0:181])
    # ax[4].scatter(floquet, waves[val][floquet],color='green')

    plt.title('MSE = ' + str(np.round(np.mean(np.square(peaks[val] - grey_peaks[val])),5)))


plot_instance(417)

bucket_4

###

## bucket_1 example (22.8%)

fig, axs = plt.subplots(1,3, gridspec_kw={'width_ratios': [4, 1, 1]})

axs[0].plot(peaks[446], color = 'black')
axs[0].plot(grey_peaks[446], color = '#7fcdbb')
axs[0].set_xlabel('Peak Count')
axs[0].set_ylabel('Amplitude')
axs[1].imshow(tf.expand_dims(true[446],1),cmap='YlGnBu')
axs[2].imshow(tf.expand_dims(pred[446],1),cmap='YlGnBu')

# bucket_2 example (37.4 %)

fig, axs = plt.subplots(1,3, gridspec_kw={'width_ratios': [4, 1, 1]})

axs[0].plot(peaks[422], color = 'black')
axs[0].plot(grey_peaks[422], color = '#1d91c0')
axs[0].set_xlabel('Peak Count')
axs[0].set_ylabel('Amplitude')
axs[1].imshow(tf.expand_dims(true[422],1),cmap='YlGnBu')
axs[2].imshow(tf.expand_dims(pred[422],1),cmap='YlGnBu')

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
ax[2].imshow(tf.expand_dims(predb[408],1),cmap='YlGnBu')
ax[2].axes.xaxis.set_visible(False)
ax[2].axes.yaxis.set_visible(False)
forcmap = ax[3].imshow(tf.expand_dims(top_16(pred[408]),1),cmap='YlGnBu')
ax[3].axes.xaxis.set_visible(False)
ax[3].axes.yaxis.set_visible(False)
# cax = ax[3].inset_axes([1.04, 0.2, 0.05, 0.6], transform=ax[3].transAxes)
fig.colorbar(forcmap)

fig.savefig('paper/figures/fig3b.eps')

fig, axs = plt.subplots(1, 1, tight_layout=True)
N, bins2, patches = plt.hist(mse_list_control,bins=bins,ec='black',color='#7fcdbb')
fig.show()

col_list = ['#7fcdbb','#7fcdbb','#1d91c0','#1d91c0','#225ea8','#225ea8','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84']

for i,thispatch in enumerate(patches):
    thispatch.set_facecolor(col_list[i])
#
#
grey_slots
