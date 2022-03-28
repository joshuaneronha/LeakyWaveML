import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU, Flatten, LeakyReLU, Conv1DTranspose


class LWAPredictionModel(tf.keras.Model):
    def __init__(self, lr = 1e-5, b = 32):
        super(LWAPredictionModel, self).__init__()

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.batch_size = 64
        self.epochs = 30

        self.dense_layers = tf.keras.Sequential()
        self.dense_layers.add(Flatten())
        # self.dense_layers.add(Dense(1000, activation = 'relu'))
        # self.dense_layers.add(Dense(800, activation = 'relu'))
        self.dense_layers.add(Dense(2000))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(LeakyReLU(0.2))
        self.dense_layers.add(Dense(2000))
        self.dense_layers.add(LeakyReLU(0.2))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(Dense(2000))
        self.dense_layers.add(LeakyReLU(0.2))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(Dense(2000))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(LeakyReLU(0.2))
        # self.dense_layers.add(Dense(2000))
        # # self.dense_layers.add(ReLU())
        # self.dense_layers.add(LeakyReLU(0.2))
        # self.dense_layers.add(Dense(2000))
        # # self.dense_layers.add(ReLU())
        # self.dense_layers.add(LeakyReLU(0.2))
        # self.dense_layers.add(Conv1DTranspose(256, 3, 1, 'same',activation='relu'))
        # self.dense_layers.add(Conv1DTranspose(128, 4, 3, 'same',activation='relu'))
        # self.dense_layers.add(Conv1DTranspose(64, 5, 1, 'same',activation='relu'))
        # self.dense_layers.add(Conv1DTranspose(1, 6, 2, 'same', activation = 'sigmoid'))
        self.dense_layers.add(Dense(36,activation='sigmoid'))

    def call(self, input):

        return self.dense_layers(input)

    def assump_accuracy(self, prediction, true):
        ba = tf.keras.metrics.BinaryAccuracy()
        total_slots = 36 - tf.reduce_sum(true,axis=1)

        prediction_sorted = tf.argsort(prediction, axis=1)

        gathered = tf.gather(prediction_sorted, total_slots,axis=1,batch_dims=1)
                # print(gathered.shape)
                # print(gathered)

        minny = tf.repeat(tf.expand_dims(tf.gather(prediction, gathered, axis=1,batch_dims=1),axis=1),axis=1,repeats = true.shape[1])
                # print(prediction.shape)
                # print(minny.shape)

        rounded = tf.greater_equal(prediction,minny)

        return ba(tf.cast(rounded,tf.float32),true)


    def loss_function(self, prediction, true):

        # print(prediction)
        bce = tf.keras.losses.BinaryCrossentropy()
        # ba = tf.keras.losses.BinaryCrossentropy()


        return bce(true,prediction)

    def accuracy(self, prediction, true):
        ba = tf.keras.metrics.BinaryAccuracy()
        prediction = tf.round(prediction)
        true = tf.round(true)
        return ba(prediction, true)
