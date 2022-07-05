import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
sys.path.append('ml_model/3grey')
import pickle
import tensorflow as tf
from tensorflow import losses
from statsmodels.stats.weightstats import ztest

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2

from preprocess import import_data, import_val_data, import_data_bin

#creating box plot for the grey and binary cases

slots, peaks, signals = import_data()
slotsbin, peaksbin, signalsbin = import_data_bin()

fig, axs = plt.subplots(1,1,tight_layout=True,figsize=(6,4))
boxplot1 = axs.boxplot(peaks, sym = '', patch_artist = True)
axs2 = axs.twinx()

axs2.get_yaxis().set_visible(False)
axs.set_ylim([-0.4,1])
axs2.set_ylim(axs.get_ylim())
plt.xticks(np.arange(0,36,5),np.arange(0,36,5))
plt.xlabel('Peak Count')
axs.set_ylabel('Amplitude (normalized)')
boxplot2= axs2.boxplot(peaksbin, sym = '', patch_artist = True)

axs.set_xticks(np.arange(35)[::5])
axs2.set_xticks(np.arange(35)[::5])
axs.set_xticklabels(np.arange(35)[::5])
axs2.set_xticklabels(np.arange(35)[::5])
plt.tight_layout()

for index, patch in enumerate(boxplot1['boxes']):
    patch.set_facecolor('#0c2c84')
    patch.set_alpha(0.5)

for index, patch in enumerate(boxplot2['boxes']):
    patch.set_facecolor('#fc1303')
    patch.set_alpha(0.5)

plt.setp(boxplot1['whiskers'], color='#0c2c84')
plt.setp(boxplot2['whiskers'], color='#fc1303')

plt.legend(['Grey Model','Binary Model'],loc='lower right')

# plt.savefig('paper/figures/box3greyoverlay.svg')

###testing validation data where model generated slots put back into COMSOL
with open('analysis/3grey/test_data627.pkl','rb') as file:
    test_data = pickle.load(file)
    peaks, true, pred, waves = test_data

    predb = np.where((pred > 0) & (pred <= 0.333), 0, pred)
    predb = np.where((predb > 0.333) & (predb <= 0.666), 0.50, predb)
    predb = np.where((predb > 0.666), 1.00, predb)

val_slots, val_peaks, val_wves = import_val_data('analysis/3grey/test627')

mse_list_grey = []
mse_list_control = []

for i in np.arange(500):

    mse_g = losses.mse(waves[i][0:181], val_wves[i][0:181])
    mse_list_grey.append(mse_g)

    try:
        mse_c = losses.mse(waves[i+1][0:181], val_wves[i][0:181])
        mse_list_control.append(mse_c)
    except:
        pass

bins = np.array([8.56330802e-25, 6.16616441e-03, 1.23323288e-02, 1.84984932e-02,
       2.46646576e-02, 3.08308220e-02, 3.69969864e-02, 4.31631508e-02,
       4.93293153e-02, 5.54954797e-02, 6.16616441e-02, 6.78278085e-02,
       7.39939729e-02, 8.01601373e-02, 8.63263017e-02, 9.24924661e-02])

fig, axs = plt.subplots(1,1,tight_layout=True,figsize=(5.5,4))

N, bins, patches = axs.hist(tf.stack(mse_list_grey).numpy(),bins=bins,ec='black',color='#7fcdbb', label='_nolegend_')

for i,thispatch in enumerate(patches):
    thispatch.set_alpha(0.75)

axs.set_xlabel('Mean Square Error')
axs.set_ylabel('Count')

N, bins, patches = axs.hist(tf.stack(mse_list_control).numpy(),bins=bins,ec='black',color='#1d91c0', label='_nolegend_')

for i,thispatch in enumerate(patches):
    thispatch.set_alpha(0.5)

plt.ylim([0,95])
axs.legend(['Model','Random'])

# plt.savefig('paper/figures/3grey_histogram.svg')

#statistical tests...

np.mean(mse_list_grey)
np.mean(mse_list_control)
ztest(mse_list_grey, mse_list_control,value=0,alternative='two-sided')
ba = tf.keras.metrics.BinaryAccuracy()
ba(predb,true.astype('float32'))

#code to plot examples
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
    ax[3].plot(val_wves[val][0:181],color='#7fcdbb')

    print('MSE = ' + str(losses.mse(waves[val][0:181], val_wves[val][0:181])),5)

plot_instance_beta(34)

data34 = np.loadtxt('1d/peaks_to_slot/3grey/results/test627.csv',delimiter=',')[12274:12274+361]
np.savetxt('experiments/data34.csv',data34)
