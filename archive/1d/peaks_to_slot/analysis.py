import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

with open('/Users/joshuaneronha/Documents/Brown/Research/LeakyWaveML/1d/peaks_to_slot/results/generated_slots.pkl', 'rb') as file:
    data = pickle.load(file)
real_peaks = [x[0] for x in data]
true = [x[1] for x in data]
prediction = [x[2] for x in data]

floquet_forward = [5,15,24,28,31,34,48,60,61,63,65,68,70,73,76,80,84,86,88,90]
floquet_back = [270 + (180 - x) for x in [95,99,101,103,106,109,114,117,120,122,126,128,130,132,146]]
floquet_back_fake = [95,99,101,103,106,109,114,117,120,122,126,128,130,132,146]

def import_data():
    """
    Loads in data and transposes it into proper shapes
    """

    slots_list = []
    peaks_list = []
    max_list = []

    with open('/Users/joshuaneronha/Documents/Brown/Research/LeakyWaveML/comsol_results/1d/1648515434.csv', 'rb') as file:
        results = np.loadtxt(file, delimiter=",", dtype=float)
        num_sims = int(results.shape[0] / 361)
        points = [361 * x for x in np.arange(num_sims + 1)]
        sorted_x = np.array([20*np.log10(results[i:i + 361,1]) for i in points[:-1]])
        peaks = sorted_x[:,floquet_back + floquet_forward]

            # past_thresh = [index for index,x in enumerate(peaks) if x.max() > 29]

            # max = peaks[past_thresh].max(axis=1)
            # print(max.shape)
            # normalized = np.divide(peaks[past_thresh].T,np.max(peaks[past_thresh],axis=1)).T
            # normalized = (peaks[past_thresh].T / peaks[past_thresh].max(axis=1)).T# TEMP:
            # peaks_list.append(normalized)
            # peaks_list.append(normalized)
        peaks_list.append(peaks)
            # max_list.append(np.concatenate([max,max,max,max,max,max],axis=1))

    with open('/Users/joshuaneronha/Documents/Brown/Research/LeakyWaveML/comsol_results/1d/1648515434.pkl', 'rb') as file:
        slots = np.array(pickle.load(file))
            # print(slots.shape)
        slots_list.append(slots)

    return peaks_list[0], slots_list[0]

peaks, slots = import_data()

x = np.array([3,4,5])
x[[2,1]]
peaks.shape

plt.plot(real_peaks[9])
plt.plot(peaks[18])
plt.plot(peaks[19])
plt.legend(['Simulation','Prediction', 'Naive Prediction'])



plt.scatter(floquet_back_fake + floquet_forward,real_peaks[9])
plt.scatter(floquet_back_fake + floquet_forward,peaks[18])
plt.scatter(floquet_back_fake + floquet_forward,peaks[19])
plt.legend(['Simulation','Prediction', 'Naive Prediction'])

plot_rounded(9)

# tf.squeeze(prediction).shape
#
# plt.imshow(tf.expand_dims(prediction[0],1),cmap='YlGnBu')
# plt.imshow(true[0],cmap='YlGnBu')
# true[0].shape
# np.convolve(tf.squeeze(true[0]),tf.squeeze(prediction[0]))
#
# plt.imshow(tf.expand_dims(np.convolve(tf.squeeze(true[0]),tf.squeeze(true[0])),1))
# tf.expand_dims(tf.stack(prediction),2).shape
pooled = tf.nn.pool(tf.expand_dims(tf.stack(prediction),2),(2,),'AVG',(2,))
#
plt.imshow(tf.expand_dims(prediction[0],1),cmap='YlGnBu')
plt.imshow(pooled[0],cmap='YlGnBu')
pooled2 = tf.nn.pool(pooled,(2,),'AVG',(2,))
#
plt.imshow(pooled2[0],cmap='YlGnBu')
#
# x =
# truef = tf.signal.fft(tf.cast(tf.squeeze(true),tf.complex64))
# predf = tf.signal.fft(prediction)
#
#
#
# plt.plot(tf.math.real(truef[0]))
# plt.plot(tf.math.real(predf[0]))
#
# bce = tf.keras.losses.BinaryCrossentropy()
# bce(tf.math.real(truef), tf.math.real(predf))
# tf.squeeze(true)

#
# prediction = np.array([[1,2,3],[3,2,1]])
# prediction
# prediction.max(axis=1)
#
# np.argsort(testy,axis=1)
#
# gathered = tf.gather(np.argsort(testy,axis=1), [1,2],axis=1,batch_dims=1)
# gathered
# prediction
# tf.gather(prediction, gathered, axis=1,batch_dims=1)
#
# minny = tf.repeat(tf.expand_dims(tf.gather(prediction, gathered, axis=1,batch_dims=1),axis=1),axis=1,repeats = 3)
# minny
#
# tf.greater_equal(prediction,minny)


def assump_accuracy(prediction, true):
    ba = tf.keras.metrics.BinaryAccuracy()
    total_slots = 36 - tf.reduce_sum(true,axis=1)

    prediction_sorted = tf.argsort(prediction, axis=1)

    gathered = tf.gather(prediction_sorted, total_slots,axis=1,batch_dims=1)
                # print(gathered.shape)
                # print(gathered)
    #
    minny = tf.repeat(tf.expand_dims(tf.gather(prediction, gathered, axis=1,batch_dims=1),axis=1),axis=1,repeats = true.shape[1])
    #             # print(prediction.shape)
    #             # print(minny.shape)
    #
    rounded = tf.greater_equal(prediction,minny)
    #
    return ba(tf.cast(rounded,tf.float32),true)

def plot_rounded(which):
    this_data = data[which]
    true_slot = np.array(this_data[1])
    pred = np.array(this_data[2])

    total_slots = true_slot.sum()

    top_slots = list(np.argsort(pred)[36 - int(total_slots):36])
    min = pred[top_slots].min()

    pred[pred >= min] = 1
    pred[pred < min] = 0

    naive = tf.round(this_data[2])

    fig = plt.figure()
    ax = fig.subplots(1,5, gridspec_kw={'width_ratios': [3, 1, 1, 1, 1]})
    ax[0].scatter(np.arange(len(this_data[0])),this_data[0])
    ax[1].imshow(true_slot,cmap = 'GnBu')
    ax[2].imshow(tf.expand_dims(this_data[2],1),cmap = 'GnBu')
    ax[3].imshow(tf.expand_dims(pred,1),cmap = 'GnBu')
    ax[4].imshow(tf.expand_dims(naive,1),cmap = 'GnBu')

plot_rounded(3)

# assump_accuracy(tf.expand_dims(tf.stack(prediction)[0],axis=0),tf.expand_dims(tf.squeeze(tf.stack(true)[9]),axis=0))
assump_accuracy(tf.stack(prediction),tf.squeeze(tf.stack(true)))
ba = tf.keras.metrics.BinaryAccuracy()
ba(tf.round(tf.stack(prediction)),tf.squeeze(tf.stack(true)))
# ba(tf.round(tf.expand_dims(tf.stack(prediction)[9],axis=0)),tf.expand_dims(tf.stack(true)[9],axis=0))
plot_rounded(0)



min = np.array(data[0][2])[inds].min()
this_array = np.array(data[0][2])

this_array[this_array >= min] = 1
this_array[this_array < min] = 0

plt.imshow(tf.expand_dims(this_array,1))

inds
[for i,val in enumerate(data[0][2])]

plt.imshow(tf.expand_dims(data[0][1],1))
tf.reduce_sum(data[0][1])

data[0]
