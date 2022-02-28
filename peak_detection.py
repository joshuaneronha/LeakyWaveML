from preprocess import *
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from scipy import signal
import tensorflow as tf
import tensorflow_probability as tfp

tf.expand_dims(efarx[0],axis=1).shape

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

np.correlate(efarx[0], efarx[0])

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

tf.Tensor(efarx[0])
1 / (tf.reduce_sum(tf.abs((prediction ** 2) - (true ** 2)), axis=1) + 1)
1 / (tf.reduce_sum(tf.abs((efarx[1] ** 2) - (efarx[0] ** 2))) + 1)

np.correlate(efarx[4],efarx[4])

np_corr(efarx, efarx)

def np_corr(first,second):
    return tf.convert_to_tensor([np.correlate(first[i],second[i])[0] for i in np.arange(len(first))])

def np_corr(first,second)

tf.numpy_function(np_corr, [efarx, efarx], tf.float32)
tf.py_function(np_corr, [efarx, efarx], tf.float32)

efarx = tf.convert_to_tensor(efarx)

efarx

np.correlate(efarx,efarx)




def my_numpy_func(x):
  # x will be a numpy array with the contents of the input to the
  # tf.function
  return np.sinh(x)
@tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
def tf_function(input):
  y = tf.numpy_function(my_numpy_func, [input], tf.float32)
  return y * y
tf_function([tf.constant(1.),tf.constant(2.)])






tf.reduce_max(tfp.stats.auto_correlation(efarx[0]))

multi = np.stack([efarx[0],efarx[2],efarx[4],efarx[6]])
multi2 = np.stack([efarx[1],efarx[3],efarx[5],efarx[6]])

tf.signal.fft(multi)

tf.linalg.tensor_diag_part(tfp.stats.correlation(multi,multi2,sample_axis=1,event_axis=0))
tf.abs(tf.linalg.tensor_diag_part(tfp.stats.correlation(tf.signal.fft(multi),tf.signal.fft(multi2),sample_axis=1,event_axis=0)))




import pickle
with open('for_comparison.pkl', 'rb') as f:
    data = np.array(pickle.load(f))

data.shape


plt.plot(data[1,0,:])
plt.plot(tf.abs(data[5,1,:]))
# plt.plot(data[5,1,:])
# plt.plot(data[0,1,:])
