import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU, Flatten


class LWAPredictionModel(tf.keras.Model):
    def __init__(self, lr = 0.001, b = 32):
        super(LWAPredictionModel, self).__init__()

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.batch_size = 100
        self.epochs = 50

        self.dense_layers = tf.keras.Sequential()
        self.dense_layers.add(Flatten())
        self.dense_layers.add(Dense(1000, activation = 'relu'))
        self.dense_layers.add(Dense(800, activation = 'relu'))
        self.dense_layers.add(Dense(600, activation = 'relu'))
        self.dense_layers.add(Dense(400, activation = 'relu'))
        self.dense_layers.add(Dense(200, activation = 'relu'))
        self.dense_layers.add(Dense(36, activation = 'sigmoid'))

    def call(self, input):

        return self.dense_layers(input)

    def loss_function(self, prediction, true):

        bce = tf.keras.losses.BinaryCrossentropy()

        return bce(true, prediction)

    def accuracy(self, prediction, true):
        ba = tf.keras.metrics.BinaryAccuracy()
        prediction = tf.round(prediction)
        true = tf.round(true)
        return ba(prediction, true)
