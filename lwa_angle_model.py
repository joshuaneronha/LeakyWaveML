import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

example = np.random.rand(50,6,36,1)


class LWAPredictionModel(tf.keras.Model):
    def __init__(self):
        super(LWAPredictionModel, self).__init__()

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.004)
        self.batch_size = 32
        self.epochs = 25

        self.conv_layers = tf.keras.Sequential()
        self.conv_layers.add(Conv2D(128, 2, 1, 'same',activation='relu'))
        self.conv_layers.add(MaxPooling2D((6,1)))
        self.conv_layers.add(Flatten())

        self.dense_layers = tf.keras.Sequential()
        self.dense_layers.add(Dense(4608, activation = 'relu'))
        self.dense_layers.add(Dense(4608 / 2, activation = 'relu'))
        self.dense_layers.add(Dense(4608 / 4, activation = 'relu'))
        self.dense_layers.add(Dense(4608 / 8, activation = 'relu'))
        self.dense_layers.add(Dense(361))

    def call(self, input):
        post_conv = self.conv_layers(input)
        post_dense = self.dense_layers(post_conv)

        return post_dense

    def loss_function(self, prediction, true):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(true, prediction)
