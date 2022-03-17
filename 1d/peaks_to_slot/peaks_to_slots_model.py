import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU, Flatten, LeakyReLU, Conv1DTranspose


class LWAPredictionModel(tf.keras.Model):
    def __init__(self, lr = 1e-5, b = 32):
        super(LWAPredictionModel, self).__init__()

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.batch_size = 128
        self.epochs = 50

        self.dense_layers = tf.keras.Sequential()
        self.dense_layers.add(Flatten())
        # self.dense_layers.add(Dense(1000, activation = 'relu'))
        # self.dense_layers.add(Dense(800, activation = 'relu'))
        self.dense_layers.add(Dense(1000))
        self.dense_layers.add(LeakyReLU(0.2))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(Dense(1000))
        self.dense_layers.add(LeakyReLU(0.2))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(Dense(1000))
        self.dense_layers.add(LeakyReLU(0.2))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(Dense(1000))
        self.dense_layers.add(LeakyReLU(0.2))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(Dense(1000))
        self.dense_layers.add(LeakyReLU(0.2))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(Dense(1000))
        self.dense_layers.add(LeakyReLU(0.2))
        # self.dense_layers.add(Dropout(0.1))
        # self.dense_layers.add(ReLU())
        # self.dense_layers.add(Conv1DTranspose(256, 3, 1, 'same',activation='relu'))
        # self.dense_layers.add(Conv1DTranspose(128, 4, 3, 'same',activation='relu'))
        # self.dense_layers.add(Conv1DTranspose(64, 5, 1, 'same',activation='relu'))
        # self.dense_layers.add(Conv1DTranspose(1, 6, 2, 'same', activation = 'sigmoid'))
        self.dense_layers.add(Dense(36,activation='sigmoid'))

    def call(self, input):

        return self.dense_layers(input)

    def loss_function(self, prediction, true):


        bce = tf.keras.losses.BinaryCrossentropy()
        # bfce = tf.keras.losses.BinaryFocalCrossentropy()
        total_slots = tf.math.abs(tf.math.subtract(tf.cast(tf.reduce_sum(tf.round(true), axis=1),tf.float32),tf.cast(tf.reduce_sum(tf.round(prediction), axis=1),tf.float32)))
        normalized = total_slots / tf.cast(tf.reduce_sum(tf.round(true),axis=1),tf.float32)
        # print(normalized)
        return bce(true, prediction) + (0*tf.reduce_mean(normalized))
        # return bce(true,prediction)
        # return

    def accuracy(self, prediction, true):
        ba = tf.keras.metrics.BinaryAccuracy()
        prediction = tf.round(prediction)
        true = tf.round(true)
        return ba(prediction, true)
