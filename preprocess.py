import numpy as np
import tensorflow as tf
import pickle

def import_data(timestamp):
    """
    Loads in data and transposes it into proper shapes
    """
    with open('results/' + timestamp + '.pkl', 'rb') as f:
        slots = pickle.load(f)

    with open('results/' + timestamp + '.csv', 'rb') as f:
        results = np.loadtxt(f, delimiter=",", dtype=float)
        num_sims = int(results.shape[0] / 361)
        points = [361 * x for x in np.arange(num_sims + 1)]
        sorted_x = [results[i:i + 361,1] for i in points[:-1]]

    return np.expand_dims(np.array(slots),axis=3) / 1.0, np.array(sorted_x) / 1.0

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
