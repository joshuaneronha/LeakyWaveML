import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU, Flatten, LeakyReLU, Conv1D
import tensorflow_io as tfio


class LWAPredictionModel(tf.keras.Model):
    def __init__(self, lr = 1e-4):
        super(LWAPredictionModel, self).__init__()

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.batch_size = 128
        self.epochs = 35

        self.dense_layers = tf.keras.Sequential()

        self.dense_layers.add(Conv1D(128,5))
        self.dense_layers.add(Flatten())
        self.dense_layers.add(Dense(3000))

        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))
        self.dense_layers.add(Dense(3000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))

        self.dense_layers.add(Dense(3000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))

        self.dense_layers.add(Dense(3000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))

        self.dense_layers.add(Dense(3000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))

        self.dense_layers.add(Dense(3000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))

        self.dense_layers.add(Dense(36,activation='sigmoid'))

    def call(self, input):
        input = tf.expand_dims(input, 2)
        return self.dense_layers(input)

    def loss_function(self, prediction, true):

        mse =  tf.keras.losses.MeanSquaredError()

        return mse(true, prediction)

    def accuracy(self, prediction, true):
        #round to the nearest slot option and check accuracy
        ba = tf.keras.metrics.BinaryAccuracy()

        prediction = np.where((prediction > 0) & (prediction <= 0.333), 0, prediction)
        prediction = np.where((prediction > 0.333) & (prediction <= 0.666), 0.50, prediction)
        prediction = np.where((prediction > 0.666), 1.00, prediction)

        return ba(prediction, true)
