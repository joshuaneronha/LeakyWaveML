import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, SimpleRNN, Reshape, Dropout, BatchNormalization, ReLU

example = np.random.rand(50,6,36,1)


class LWAPredictionModel(tf.keras.Model):
    def __init__(self):
        super(LWAPredictionModel, self).__init__()

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.batch_size = 32
        self.epochs = 10

        self.conv_layers = tf.keras.Sequential()
        self.conv_layers.add(Conv2D(64, 2, 1, 'same',activation='relu'))
        # self.conv_layers.add(BatchNormalization())
        # self.conv_layers.add(MaxPooling2D((6,1)))
        # self.conv_layers.add(Conv2D(128, 2, 1, 'same',activation='relu'))
        # self.conv_layers.add(MaxPooling2D((2,2)))
        # #
        self.dense_layers = tf.keras.Sequential()
        self.dense_layers.add(Flatten())
        self.dense_layers.add(Dense(1000, activation = 'relu'))
        # self.dense_layers.add(Dropout(0.2))
        self.dense_layers.add(Dense(800, activation = 'relu'))
        # self.dense_layers.add(Dropout(0.2))
        self.dense_layers.add(Dense(600, activation = 'relu'))
        # self.dense_layers.add(Dropout(0.2))
        self.dense_layers.add(Dense(450, activation = 'relu'))
        # self.dense_layers.add(Dropout(0.2))
        self.dense_layers.add(Dense(361))

        # self.dense_only = tf.keras.Sequential()
        # self.dense_only.add(Flatten())
        # self.dense_only.add(Dense((4608), activation = 'relu'))
        # # self.dense_only.add(Dropout(0.2))
        # self.dense_only.add(Dense((4608) / 2, activation = 'relu'))
        # # self.dense_only.add(Dropout(0.2))
        # self.dense_only.add(Dense((4608) / 4, activation = 'relu'))
        # # self.dense_only.add(Dropout(0.2))
        # self.dense_only.add(Dense((4608) / 8, activation = 'relu'))
        # # self.dense_only.add(Dropout(0.2))
        # self.dense_only.add(Dense(361))

    def call(self, input):
        post_conv = self.conv_layers(input)
        post_dense = self.dense_layers(post_conv)
        # post_dense = self.dense_only(input)

        return post_dense

    def loss_function(self, prediction, true):
        mse = tf.keras.losses.MeanSquaredError()
        # huber = tf.keras.losses.Huber()
        # lcosh = tf.keras.losses.LogCosh(reduction="auto", name="log_cosh")
        # return mse(true, prediction)
        return mse(true, prediction)
