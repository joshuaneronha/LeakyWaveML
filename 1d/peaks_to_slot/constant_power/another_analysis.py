import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
sys.path.append('/Users/joshuaneronha/Documents/Brown/Research/LeakyWaveML/1d/peaks_to_slot/constant_power')
import pickle
import tensorflow as tf
from statsmodels.stats.weightstats import ztest
stats
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
from tensorflow import losses

tensorflow.losses.mse([1],[3])

from preprocess import import_data, import_val_data

floquet = list(np.array([  6.,  16.,  24.,  29.,  32.,  34.,  49.,  60.,  62.,  63.,  66.,
        69.,  71.,  73.,  77.,  81.,  84.,  86.,  89.,  90.,  95., 100.,
       101., 104., 106., 110., 114., 118., 120., 123., 126., 129., 131.,
       132., 147.]).astype('int'))

###

slots, peaks, _ = import_data()

peaks.shape
# selected_peaks = peaks[peaks[:,1] <0 ]

fig, axs = plt.subplots(1,1,tight_layout=True,figsize=(6,4))

boxplot = axs.boxplot(peaks, sym = '', patch_artist = True)
plt.xticks(np.arange(0,36,5),np.arange(0,36,5))
plt.xlabel('Peak Count')
plt.ylabel('Amplitude (normalized)')
plt.ylim([-0.4,1])

plt.savefig('paper/figures/boxbinarymono.svg')

###

with open('1d/peaks_to_slot/constant_power/results/test_data510.pkl','rb') as file:
    test_data = pickle.load(file)
    peaks, true, pred, waves = test_data

true
top_16(pred)
# val_slots, val_peaks = import_val_data('1d/peaks_to_slot/constant_power/results/validation_old')

val510_slots, val510_peaks, val510_waves = import_val_data('1d/peaks_to_slot/constant_power/results/validation510')

plt.imshow(tf.expand_dims(pred[80] > tf.sort(pred[80])[16],1))

plt.imshow(tf.expand_dims(val51_slots[80],1))

len(grey_peaks)

plt.plot(peaks[9])
# plt.plot(val_peaks[7])
plt.plot(val59_peaks[9])
plt.legend(['Reference','Binary','Conv'])

correct / (len(val510_slots) * 36)
correct = 0
for i in np.arange(len(val510_slots)):
    correct += (val510_slots[i] == true[i]).sum()

val510_slots

mse_list_binary = []
# mse_list_grey = []
mse_list_control = []
bucket_1 = []
bucket_2 = []
bucket_3 = []
bucket_4 = []
for i in np.arange(500):
    # mse_b = ((peaks[i] - val510_peaks[i]) ** 2).mean()
    mse_b = losses.mse(waves[i][0:181], val510_waves[i][0:181])
    mse_list_binary.append(mse_b)

    # mse_g = ((peaks[i] - grey_peaks[i]) ** 2).mean()
    # mse_list_grey.append(mse_g)

    try:
        mse_c = losses.mse(waves[i+1][0:181], val510_waves[i][0:181])
        mse_list_control.append(mse_c)
    except:
        pass

    if mse_b < 0.0166:
        bucket_1.append(i)
    elif mse_b < 0.03236:
        bucket_2.append(i)
    elif mse_b < 0.04811:
        bucket_3.append(i)
    else:
        bucket_4.append(i)

# old_bins = np.array([0.00086428, 0.00873865, 0.01661302, 0.02448739, 0.03236176,
#        0.04023613, 0.0481105 , 0.05598487, 0.06385924, 0.07173361,
#        0.07960798, 0.08748235, 0.09535672, 0.10323109, 0.11110546,
#        0.11897984, 0.12685421, 0.13472858, 0.14260295, 0.15047732,
#        0.15835169])

ztest(mse_list_binary, mse_list_control,value=0,alternative='two-sided')
np.mean(mse_list_binary)
np.mean(mse_list_control)
 pred = np.array(pred)
for i in np.arange(len(pred)):
    pred[i,:] = top_16(pred[i,:])

ba = tf.keras.metrics.BinaryAccuracy()
ba(true,pred)

bins
bucket_2
tf.stack(mse_list_binary).numpy()
fig, axs = plt.subplots(1,1,tight_layout=True,figsize=(5.5,4))

N, bins, patches = axs.hist(tf.stack(mse_list_binary).numpy(),bins=15,ec='black',color='#7fcdbb', label='_nolegend_')
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

axs.legend(['Model','Random'])
# plt.ylim([0, 115])

plt.savefig('paper/figures/binary_histogramnew.svg')
bins
# axs[1].set_xlabel('Mean Square Error')
# axs[1].set_ylabel('Count')

bucket_2

####
top_16(pred)
bucket_4

fig, axs = plt.subplots(2,2,tight_layout=True,figsize=(6,4))

N, bins, patches = axs[0,0].hist(mse_list_binary,bins=old_bins,ec='black',color='#7fcdbb', label='_nolegend_')
col_list = ['#7fcdbb','#7fcdbb','#1d91c0','#1d91c0','#225ea8','#225ea8','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84']

for i,thispatch in enumerate(patches):
    thispatch.set_facecolor(col_list[i])

axs[0,0].set_xlabel('Mean Square Error')
axs[0,0].set_ylabel('Count')

## bucket_1 example (22.8%)
axs[0,1].plot(peaks[72], color = 'black')
axs[0,1].plot(val51_peaks[72], color = '#7fcdbb')
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

bucket_4

def plot_instance(val):

    fig, ax = plt.subplots(1,5,gridspec_kw={'width_ratios': [6, 1, 1, 1, 6]},figsize=(12,4))
    ax[0].plot(peaks[val])
    ax[0].plot(val510_peaks[val])
    ax[0].set_xlabel('Peak Count')
    ax[0].set_ylabel('Amplitude')
    # ax[0].plot(val_peaks[0])
    ax[1].imshow(tf.expand_dims(true[val],1),cmap='YlGnBu')
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    ax[2].imshow(tf.expand_dims(pred[val],1),cmap='YlGnBu')
    ax[2].axes.xaxis.set_visible(False)
    ax[2].axes.yaxis.set_visible(False)
    forcmap = ax[3].imshow(tf.expand_dims(top_16(pred[val]),1),cmap='YlGnBu')
    ax[3].axes.xaxis.set_visible(False)
    ax[3].axes.yaxis.set_visible(False)# cax = ax[3].inset_axes([1.04, 0.2, 0.05, 0.6], transform=ax[3].transAxes)
    ax[4].plot(waves[val][0:181])
    ax[4].plot(val510_waves[val][0:181])
    # ax[4].scatter(floquet, waves[val][floquet],color='green')

    # plt.title('MSE = ' + str(np.round(np.mean(np.square(peaks[val] - val510_peaks[val])),5)))


plt.scatter(np.arange(35),np.square(np.abs(peaks[477] - val510_peaks[477])))

plot_instance(33)
bucket_4
plot_instance(463)

def plot_instance_beta(val):

    fig, ax = plt.subplots(1,4,gridspec_kw={'width_ratios': [1, 1, 1, 10]},figsize=(7,4))
    ax[0].imshow(tf.expand_dims(true[val],1),cmap='YlGnBu')
    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)
    ax[1].imshow(tf.expand_dims(pred[val],1),cmap='YlGnBu')
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    forcmap = ax[2].imshow(tf.expand_dims(top_16(pred[val]),1),cmap='YlGnBu')
    ax[2].axes.xaxis.set_visible(False)
    ax[2].axes.yaxis.set_visible(False)# cax = ax[3].inset_axes([1.04, 0.2, 0.05, 0.6], transform=ax[3].transAxes)
    ax[3].plot(waves[val][0:181],color='#0c2c84')
    ax[3].plot(val510_waves[val][0:181],color='#7fcdbb')
    # ax[3].plot(grey_comp[0:181,1])

    print('MSE binary = ' + str(losses.mse(waves[val][0:181],val510_waves[val][0:181])))
    print('MSE grey = ' + str(losses.mse(grey_comp[0:181,1],val510_waves[val][0:181])))

bucket_3
mse_list_binary[104]
plot_instance_beta(477)



peaks[477]

grey_comp = np.loadtxt('4_grey_477_slot_results.csv')
grey_comp[:,1] = 20*np.log10(grey_comp[:,1])
grey_comp = grey_comp / grey_comp[:,1].max()


peaks[477]

plt.savefig('paper/figures/477.eps')

### trying binary tests on grey model_weights

grey_slots, grey_peaks, grey_waves = import_val_data('4greytestall')

grey_waves.shape
val510_waves.shape
waves.shape

binary_loss = []
grey_loss = []

for i in np.arange(500):
    binary_loss.append(losses.mse(waves[i][0:181], val510_waves[i][0:181]))
    grey_loss.append(losses.mse(waves[i][0:181], grey_waves[i][0:181]))

diff = np.array(binary_loss) - np.array(grey_loss)

plt.hist(diff)

# fig, ax = plt.subplots(1,4,gridspec_kw={'width_ratios': [6, 1, 1, 1]})
# ax[0].plot(peaks[313])
# ax[0].plot(val57_peaks[313])
# ax[0].set_xlabel('Peak Count')
# ax[0].set_ylabel('Amplitude')
# # ax[0].plot(val_peaks[0])
# ax[1].imshow(tf.expand_dims(true[313],1),cmap='YlGnBu')
# ax[1].axes.xaxis.set_visible(False)
# ax[1].axes.yaxis.set_visible(False)
# ax[2].imshow(tf.expand_dims(pred[313],1),cmap='YlGnBu')
# ax[2].axes.xaxis.set_visible(False)
# ax[2].axes.yaxis.set_visible(False)
# forcmap = ax[3].imshow(tf.expand_dims(top_16(pred[313]),1),cmap='YlGnBu')
# ax[3].axes.xaxis.set_visible(False)
# ax[3].axes.yaxis.set_visible(False)# cax = ax[3].inset_axes([1.04, 0.2, 0.05, 0.6], transform=ax[3].transAxes)
# # fig.colorbar(forcmap)
#
# fig.savefig('paper/figures/fig3b.eps')
#
# fig, axs = plt.subplots(1, 1, tight_layout=True)
# N, bins2, patches = plt.hist(mse_list_control,bins=bins,ec='black',color='#7fcdbb')
# fig.show()
#
# col_list = ['#7fcdbb','#7fcdbb','#1d91c0','#1d91c0','#225ea8','#225ea8','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84','#0c2c84']
#
# for i,thispatch in enumerate(patches):
#     thispatch.set_facecolor(col_list[i])
# #
# #
