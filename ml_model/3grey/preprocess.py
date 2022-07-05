import numpy as np
import tensorflow as tf
import pickle
import os

# list of floquet peaks, sourced from analysis/floquet.m

floquet = list(np.array([  6.,  16.,  24.,  29.,  32.,  34.,  49.,  60.,  62.,  63.,  66.,
        69.,  71.,  73.,  77.,  81.,  84.,  86.,  89.,  90.,  95., 100.,
       101., 104., 106., 110., 114., 118., 120., 123., 126., 129., 131.,
       132., 147.]).astype('int'))

def import_data():
    """
    Loads in simulation data
    """

    slots_list = []
    peaks_list = []
    max_list = []
    wave_list = []

    #get list of all files
    timestamp_list = list(set([x.split('.')[0] for x in os.listdir('simulation_model/3grey/3greydata/')]))

    for timestamp in timestamp_list:

        if timestamp == '1655498155':
            #this simulation didn't finish, so skipping it
            continue

        with open('simulation_model/3grey/3greydata/' + timestamp + '.csv', 'rb') as file:
            #load in far-field data
            #pull out only the floquet peaks and reshape
            results = np.loadtxt(file, delimiter=",", dtype=float)
            num_sims = int(results.shape[0] / 361)
            points = [361 * x for x in np.arange(num_sims + 1)]
            sorted_x = np.array([20*np.log10(results[i:i + 361,1]) for i in points[:-1]])
            peaks = sorted_x[:,floquet]

            #normalize, since power should be constant given same number of slots

            normalized = (peaks.T / peaks.max(axis=1)).T
            peaks_list.append(normalized)

            wave_list.append((sorted_x.T / sorted_x.max(axis=1)).T)

        with open('simulation_model/3grey/3greydata/' + timestamp + '.pkl', 'rb') as file:
            #also load in the slot shapes
            slots = np.array(pickle.load(file))
            slots_list.append(slots)

    return np.concatenate(slots_list), np.concatenate(peaks_list), np.concatenate(wave_list)


def import_val_data(path):
    """
    Basically the same thing as import_data, except it reads in the validation
    data from the generated slots we put back into Comsol... accepts the path
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

            normalized = (peaks.T / peaks.max(axis=1)).T

            peaks_list.append(normalized)
            wave_list.append((sorted_x.T / sorted_x.max(axis=1)).T)

    try:
        with open(path + '.pkl', 'rb') as file:
                slots = np.array(pickle.load(file))
                slots_list.append(slots)
    except:
        slots_list.append([])

    return np.concatenate(slots_list), np.concatenate(peaks_list), np.concatenate(wave_list)

def import_data_bin():
    """
    Loads in the binary data using same methods for comparison's sake
    """

    slots_list = []
    peaks_list = []
    max_list = []
    wave_list = []

    timestamp_list = list(set([x.split('.')[0] for x in os.listdir('simulation_model/binary_model/1dconstantslots/')]))

    for timestamp in timestamp_list:

        with open('simulation_model/binary_model/1dconstantslots/' + timestamp + '.csv', 'rb') as file:
            results = np.loadtxt(file, delimiter=",", dtype=float)
            num_sims = int(results.shape[0] / 361)
            points = [361 * x for x in np.arange(num_sims + 1)]
            sorted_x = np.array([20*np.log10(results[i:i + 361,1]) for i in points[:-1]])
            peaks = sorted_x[:,floquet]

            normalized = (peaks.T / peaks.max(axis=1)).T

            peaks_list.append(normalized)
            wave_list.append((sorted_x.T / sorted_x.max(axis=1)).T)

        with open('simulation_model/binary_model/1dconstantslots/' + timestamp + '.pkl', 'rb') as file:
            slots = np.array(pickle.load(file))
            slots_list.append(slots)

    return np.concatenate(slots_list), np.concatenate(peaks_list), np.concatenate(wave_list)

def get_next_batch(input_array, label_array, start_index, batch_size):
    """
    Accepts an array of inputs and labels along with a starting index and batch
    size in order to separate the full array of data into batches.
    :inputs: a NumPy array of all inputs
    :labels: a NumPy array of all labels
    :start_index: the first index desired in the batch
    :batch_size: how many total images desired in the batch

    note: function adapted from coursework in CSCI 1470 @ Brown University
    """
    return input_array[start_index: (start_index + batch_size), :], label_array[start_index: (start_index + batch_size)]
