from preprocess import *
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from scipy import signal
import tensorflow as tf
import tensorflow_probablity as tfp

shapes, efarx = import_data(['1645299750'])

np.correlate(efarx[0],efarx[0])

def findpeaks(which):
    peaks, props = signal.find_peaks(efarx[which], prominence = 8, width=8)

    plt.plot(efarx[which])
    plt.plot(peaks, efarx[which][peaks], "x")

plt.plot(signal.correlate(efarx[0], efarx[0]))

findpeaks(2)


peaks

props

mult = np.stack([efarx[0],efarx[0]])

mult.shape

efarx[0].shape

corr_vec = np.vectorize(np.correlate)

[efarx[0],efarx[0]]

np.correlate(efarx[0],efarx[0])

[compute_similarity(x[0],x[1]) for x in [[efarx[0],efarx[0]], [efarx[1],efarx[1]]]]

testvec = np.vectorize(compute_similarity)


testvec([efarx[0],efarx[0]],3)

compute_similarity(efarx[0],efarx[0])

def compute_similarity(ref_rec,input_rec,weightage=[0.33,0.33,0.33]):
    ## Time domain similarity
    ref_time = np.correlate(ref_rec,ref_rec)
    inp_time = np.correlate(ref_rec,input_rec)
    diff_time = abs(ref_time-inp_time)

    ## Freq domain similarity
    ref_freq = np.correlate(np.fft.fft(ref_rec),np.fft.fft(ref_rec))
    inp_freq = np.correlate(np.fft.fft(ref_rec),np.fft.fft(input_rec))
    diff_freq = abs(ref_freq-inp_freq)

    ## Power similarity
    ref_power = np.sum(ref_rec**2)
    inp_power = np.sum(input_rec**2)
    diff_power = abs(ref_power-inp_power)

    return float(weightage[0]*diff_time+weightage[1]*diff_freq+weightage[2]*diff_power)


tfp.stats.correlate(efarx[0])
