import numpy as np
import tensorflow as tf
import pickle
import os

# floquet_forward = [5,15,24,28,31,34,48,60,61,63,65,68,70,73,76,80,84,86,88,90]
# floquet_back = [270 + (180 - x) for x in [95,99,101,103,106,109,114,117,120,122,126,128,130,132,146]]

floquet = list(np.array([  6.,  16.,  24.,  29.,  32.,  34.,  49.,  60.,  62.,  63.,  66.,
        69.,  71.,  73.,  77.,  81.,  84.,  86.,  89.,  90.,  95., 100.,
       101., 104., 106., 110., 114., 118., 120., 123., 126., 129., 131.,
       132., 147.]).astype('int'))

def import_data():
    """
    Loads in data and transposes it into proper shapes
    """

    slots_list = []
    peaks_list = []
    max_list = []
    wave_list = []


    timestamp_list = list(set([x.split('.')[0] for x in os.listdir('comsol_results/1dgrey/')]))

    for timestamp in timestamp_list:

        if timestamp == '1651155833':
            continue

        with open('comsol_results/1dgrey/' + timestamp + '.csv', 'rb') as file:
            results = np.loadtxt(file, delimiter=",", dtype=float)
            num_sims = int(results.shape[0] / 361)
            points = [361 * x for x in np.arange(num_sims + 1)]
            sorted_x = np.array([20*np.log10(results[i:i + 361,1]) for i in points[:-1]])
            peaks = sorted_x[:,floquet]

            # past_thresh = [index for index,x in enumerate(peaks) if x.max() > 29]

            # max = peaks[past_thresh].max(axis=1)
            # print(max.shape)
            # normalized = np.divide(peaks[past_thresh].T,np.max(peaks[past_thresh],axis=1)).T
            normalized = (peaks.T / peaks.max(axis=1)).T# TEMP:
            # normalized = (peaks / peaks.max(axis=0))
            peaks_list.append(normalized)
            wave_list.append((sorted_x.T / sorted_x.max(axis=1)).T)
            # peaks_list.append(normalized)
            # peaks_list.append(peaks)
            # max_list.append(np.concatenate([max,max,max,max,max,max],axis=1))

        with open('comsol_results/1dgrey/' + timestamp + '.pkl', 'rb') as file:
            slots = np.array(pickle.load(file))
            # print(slots.shape)
            slots_list.append(slots)



    # return np.concatenate(slots_list), np.concatenate([np.concatenate(peaks_list), np.concatenate(max_list)],axis=1)
    return np.concatenate(slots_list), np.concatenate(peaks_list), np.concatenate(wave_list)


def import_val_data(path):
    """
    Loads in data and transposes it into proper shapes
    """

    slots_list = []
    peaks_list = []
    max_list = []
    wave_list = []

    with open(path + '.csv', 'rb') as file:
            results = np.loadtxt(file, delimiter=",", dtype=float)
            num_sims = int(results.shape[0] / 361)
            points = [361 * x for x in np.arange(num_sims + 1)]
            sorted_x = np.array([20*np.log10(results[i:i + 361,1]) for i in points[:-1]])
            peaks = sorted_x[:,floquet]

            # past_thresh = [index for index,x in enumerate(peaks) if x.max() > 29]

            # max = peaks[past_thresh].max(axis=1)
            # print(max.shape)
            # normalized = np.divide(peaks[past_thresh].T,np.max(peaks[past_thresh],axis=1)).T
            normalized = (peaks.T / peaks.max(axis=1)).T# TEMP:
            # normalized = (peaks / peaks.max(axis=0))
            peaks_list.append(normalized)
            wave_list.append((sorted_x.T / sorted_x.max(axis=1)).T)
            # peaks_list.append(normalized)
            # peaks_list.append(peaks)
            # max_list.append(np.concatenate([max,max,max,max,max,max],axis=1))
    try:
        with open(path + '.pkl', 'rb') as file:
                slots = np.array(pickle.load(file))
                # print(slots.shape)
                slots_list.append(slots)
    except:
        slots_list.append([])

    # return np.concatenate(slots_list), np.concatenate([np.concatenate(peaks_list), np.concatenate(max_list)],axis=1)
    return np.concatenate(slots_list), np.concatenate(peaks_list), np.concatenate(wave_list)

def get_next_batch(input_array, label_array, start_index, batch_size):
    """
    Accepts an array of inputs and labels along with a starting index and batch
    size in order to separate the full array of data into batches.
    :inputs: a NumPy array of all images with shape (n x 2)
    :labels: a NumPy array of all labels with shape (n x 1)
    :start_index: the first index desired in the batch
    :batch_size: how many total images desired in the batch
    """
    return input_array[start_index: (start_index + batch_size), :], label_array[start_index: (start_index + batch_size)]
