import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import sys

from preprocess import import_data

sys.path.append('1d/signal_to_slot')

shapes, efarx = import_data(['1645299750'])

floquet_list = [5,15,24,28,31,34,48,60,61,63,65,68,70,73,76,80,84,86,88,90,\
 95,99,101,103,106,109,114,117,120,122,126,128,130,132,146]

floquet_forward = [5,15,24,28,31,34,48,60,61,63,65,68,70,73,76,80,84,86,88,90]
floquet_back_index = [90-x for x in [95,99,101,103,106,109,114,117,120,122,126,128,130,132,146]]
floquet_back = [270 + (180 - x) for x in [95,99,101,103,106,109,114,117,120,122,126,128,130,132,146]]
floquet_back

len(floquet_back)

floquet_back + floquet_forward

efarx

def plot_peaks(which):
    amp_list = []
    plt.plot(np.arange(-90,0),20*np.log10(efarx[which][270:360]),color = '#0c2c84')
    plt.plot(np.arange(0,91),20*np.log10(efarx[which][0:91]), color = '#0c2c84')
    plt.scatter(floquet_back_index, 20*np.log10(efarx[which][floquet_back]),color='#7fcdbb',marker='o')
    plt.scatter(floquet_forward, 20*np.log10(efarx[which][floquet_forward]),color='#7fcdbb',marker='o')
    plt.xlim([-90,90])
    plt.xlabel('Angle')
    plt.ylabel('Amplitude (dB)')
    #         amp_list.append(efarx[which][270:360][90-i])
    #     if i <= 90:
    #         plt.scatter(i, efarx[which][0:91][i],color='orange')
    #         amp_list.append(efarx[which][0:91][i])
    #
    # normalized = amp_list / max(amp_list)
    # return normalized


plot_peaks(61)






efarx[9].argmax()

efarx[9][0:90]

plt.plot(efarx[1])
plt.plot(efarx[2])
plt.plot(efarx[3])

avg_curve = (efarx[0] + efarx[1] + efarx[2] + efarx[3])/4

from build_model_inline import *

the_model = main(['1645299750', '1645318239', '1645336863', '1645355510', '1645374240', '1645393069', '1645412004', '1645431000', '1645450180', '1645470062'])
the_model.call([1])

curve = 1.5*np.loadtxt('1d/signal_to_slot/test_curve.csv', delimiter = ',')[:,1] - 10

plt.plot(curve)
np.flip(tf.round(the_model.call(np.expand_dims(curve,0))))
plt.imshow(tf.round(the_model.call(np.expand_dims(curve,0))))
