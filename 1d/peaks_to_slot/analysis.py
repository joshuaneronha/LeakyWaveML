import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

with open('/Users/joshuaneronha/Documents/Brown/Research/LeakyWaveML/1d/peaks_to_slot/results/generated_slots.pkl', 'rb') as file:
    data = pickle.load(file)
true = [x[1] for x in data]
prediction = [x[2] for x in data]
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
    ax[0].plot(this_data[0])
    ax[1].imshow(true_slot,cmap = 'GnBu')
    ax[2].imshow(tf.expand_dims(this_data[2],1),cmap = 'GnBu')
    ax[3].imshow(tf.expand_dims(pred,1),cmap = 'GnBu')
    ax[4].imshow(tf.expand_dims(naive,1),cmap = 'GnBu')


# assump_accuracy(tf.expand_dims(tf.stack(prediction)[0],axis=0),tf.expand_dims(tf.squeeze(tf.stack(true)[9]),axis=0))
assump_accuracy(tf.stack(prediction),tf.squeeze(tf.stack(true)))
ba = tf.keras.metrics.BinaryAccuracy()
ba(tf.round(tf.stack(prediction)),tf.squeeze(tf.stack(true)))
# ba(tf.round(tf.expand_dims(tf.stack(prediction)[9],axis=0)),tf.expand_dims(tf.stack(true)[9],axis=0))
plot_rounded(2)



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
