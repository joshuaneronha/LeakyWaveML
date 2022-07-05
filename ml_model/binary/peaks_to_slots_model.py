import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU, Flatten, LeakyReLU, Conv1D
import tensorflow_io as tfio


class LWAPredictionModel(tf.keras.Model):
    def __init__(self, lr = 5e-5):
        super(LWAPredictionModel, self).__init__()

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.batch_size = 64
        self.epochs = 20

        self.dense_layers = tf.keras.Sequential()

        self.dense_layers.add(Conv1D(128,5))
        self.dense_layers.add(Flatten())
        self.dense_layers.add(Dense(2000))

        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))
        self.dense_layers.add(Dense(2000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))

        self.dense_layers.add(Dense(2000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))

        self.dense_layers.add(Dense(2000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))

        self.dense_layers.add(Dense(2000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))

        self.dense_layers.add(Dense(2000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))

        self.dense_layers.add(Dense(36,activation='sigmoid'))

    def call(self, input):
        input = tf.expand_dims(input, 2)
        return self.dense_layers(input)

    def loss_function(self, prediction, true):

        bce = tf.keras.losses.BinaryCrossentropy()

        return bce(true, prediction)

    def assump_accuracy(self, prediction, true):
        #accuracy assuming we take the top 16 slots, not just probabilites over 0.5
        ba = tf.keras.metrics.BinaryAccuracy()
        total_slots = 36 - tf.reduce_sum(true,axis=1)

        prediction_sorted = tf.argsort(prediction, axis=1)

        gathered = tf.gather(prediction_sorted, total_slots,axis=1,batch_dims=1)

        minny = tf.repeat(tf.expand_dims(tf.gather(prediction, gathered, axis=1,batch_dims=1),axis=1),axis=1,repeats = true.shape[1])

        rounded = tf.greater_equal(prediction,minny)

        return ba(tf.cast(rounded,tf.float32),true)

    def accuracy(self, prediction, true):
        ba = tf.keras.metrics.BinaryAccuracy()
        prediction = tf.round(prediction)
        true = tf.round(true)
        return ba(prediction, true)
