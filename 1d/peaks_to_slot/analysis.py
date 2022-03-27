import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

with open('/Users/joshuaneronha/Documents/Brown/Research/LeakyWaveML/1d/peaks_to_slot/results/generated_slots.pkl', 'rb') as file:
    data = pickle.load(file)

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
    ax = fig.subplots(1,4, gridspec_kw={'width_ratios': [3, 1, 1, 1]})
    ax[0].plot(this_data[0])
    ax[1].imshow(true_slot,cmap = 'GnBu')
    ax[2].imshow(tf.expand_dims(pred,1),cmap = 'GnBu')
    ax[3].imshow(tf.expand_dims(naive,1),cmap = 'GnBu')

plot_rounded(8)



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
