import numpy as np
import tensorflow as tf
import sys
sys.path.append("1d/peaks_to_slot")
from peaks_to_slots_model import LWAPredictionModel
import matplotlib.pyplot as plt

Model = LWAPredictionModel()
Model.load_weights('1d/peaks_to_slot/results/model_weights')

Model.compile()

Model.summary()
peaks_of_interest = np.array([20,20,20,20,20,20,
                                20,20,20,20,20,20,
                                20,20,20,20,20,20,
                                20,20,20,20,20,20,
                                20,20,20,20,20,20,
                                20,20,20,20,20
])

x = tf.convert_to_tensor(peaks_of_interest)
tf.expand_dims(peaks_of_interest,0).shape
results = Model.call(tf.expand_dims(x,0))

plt.imshow(tf.transpose(results), cmap='YlGnBu')

len(peaks_of_interest)

def plot_rounded(data, total_slots):
    this_data = np.array(data[0])
    pred = np.array(data[0])

    print(pred)

    top_slots = list(np.argsort(pred)[36 - int(total_slots):36])
    min = pred[top_slots].min()

    pred[pred >= min] = 1
    pred[pred < min] = 0

    naive = tf.round(this_data)

    print(this_data)

    fig = plt.figure()
    ax = fig.subplots(1,4, gridspec_kw={'width_ratios': [3, 1, 1, 1]})
    ax[0].scatter(np.arange(len(peaks_of_interest)),peaks_of_interest)
    ax[1].imshow(tf.expand_dims(this_data,1),cmap = 'GnBu')
    ax[2].imshow(tf.expand_dims(pred,1),cmap = 'GnBu')
    ax[3].imshow(tf.expand_dims(naive,1),cmap = 'GnBu')

    return naive

plot_rounded(results,18)
