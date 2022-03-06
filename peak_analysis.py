from preprocess import *
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

slots, results = import_data(['1645299750', '1645318239', '1645336863', '1645355510'])

def findpeaks(which):
    peaks, props = signal.find_peaks(which, prominence = 5, width=5)
    return peaks

def plotpeaks(i):
    peaks = findpeaks(results[i,:])
    plt.plot(results[i,:])
    plt.plot(peaks, results[i,:][peaks], "x")
    print(peaks)

plotpeaks(299)

out_dict = {}
for i in np.arange(2000):
    peaks = findpeaks(results[i,:])
    for j in peaks:
        if j in out_dict:
            out_dict[j] += 1
        else:
            out_dict[j] = 1

out_dict

out = np.array(list(out_dict.items()))

df = pd.DataFrame(out).sort_values(1, ascending = False)

df
df[0]
df[1]

df.head(25)

top_df = df.head(50)

plt.scatter(df[0], df[1])



results.shape
signal.find_peaks(results)

x = {42: 3, 45: 8}
x

42 in x
x[42] += 1
