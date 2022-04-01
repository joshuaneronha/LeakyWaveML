import numpy as np
import tensorflow as tf
from peaks_to_slots_model import LWAPredictionModel

trained_model = tf.keras.models.load_model('1d/peaks_to_slot/results/model')
trained_model.compile()

peaks_of_interest = np.array([20,20,20,20,20,20,
                                20,20,20,20,20,20,
                                20,20,20,20,20,20,
                                20,20,20,20,20,20,
                                20,20,20,20,20,20,
                                20,20,20,20,20,20
])

results = trained_model.call(tf.expand_dims(peaks_of_interest,0))
