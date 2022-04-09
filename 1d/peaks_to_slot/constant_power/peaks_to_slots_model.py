import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU, Flatten, LeakyReLU, Conv1DTranspose
import tensorflow_io as tfio


class LWAPredictionModel(tf.keras.Model):
    def __init__(self, lr = 5e-5):
        super(LWAPredictionModel, self).__init__()

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.batch_size = 64
        self.epochs = 20

        self.dense_layers = tf.keras.Sequential()
        # self.dense_layers.add(Dense(1000, activation = 'relu'))
        # self.dense_layers.add(Dense(800, activation = 'relu'))
        self.dense_layers.add(Dense(2000))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))
        self.dense_layers.add(Dense(2000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(Dense(2000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(Dense(2000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(Dense(2000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))
        # self.dense_layers.add(ReLU())
        self.dense_layers.add(Dense(2000))
        self.dense_layers.add(LeakyReLU(0.1))
        self.dense_layers.add(Dropout(0.1))

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
        bce_laplacian = tf.keras.losses.BinaryCrossentropy()
        # bce_exp2 = tf.keras.losses.BinaryCrossentropy()

        true_laplacian = tfio.experimental.filter.laplacian(tf.cast(tf.reshape(true, [-1,1,36,1]),tf.float32), ksize = [1,3])
        pred_laplacian = tfio.experimental.filter.laplacian(tf.round(tf.cast(tf.reshape(prediction, [-1,1,36,1]),tf.float32)), ksize = [1,3])


        # tf_data.shape
        # plt.imshow(tf.reshape(tfio.experimental.filter.laplacian(tf.cast(tf_data,tf.float32), ksize=[1,3]),[36,1]),cmap='YlGnBu')

        # return bce(true,prediction) + 0.5*bce_exp(true_pooled, pred_pooled) + (0*bce_exp2(true_pooled_2, pred_pooled_2))
        return bce(true,prediction) + bce_laplacian(true_laplacian, pred_laplacian)

    def accuracy(self, prediction, true):
        ba = tf.keras.metrics.BinaryAccuracy()
        prediction = tf.round(prediction)
        true = tf.round(true)
        return ba(prediction, true)
