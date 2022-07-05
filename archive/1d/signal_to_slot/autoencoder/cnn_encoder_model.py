import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Conv1DTranspose

class CNNAutoEncoder(tf.keras.Model):

    def __init__(self):
        super(CNNAutoEncoder, self).__init__()

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  #.01 originally, improved w/ .004
        self.batch_size = 32 #was 32 originally
        self.epochs = 6  #originally 25 dec5

        self.encoder = tf.keras.Sequential()
        self.encoder.add(Conv1D(32, 3, 2, 'same', activation='relu'))
        self.encoder.add(Conv1D(64, 4, 2, 'same', activation='relu'))
        self.encoder.add(Conv1D(128, 5, 3, 'same', activation='relu'))
        self.encoder.add(Conv1D(256, 6, 5, 'same', activation='relu'))

        self.dense = tf.keras.Sequential()
        self.dense.add(tf.keras.layers.Dense(512, activation = 'relu'))

        self.decoder = tf.keras.Sequential()
        # self.decoder.add(Conv1DTranspose(512, 4, 1, 'same',activation='relu'))
        self.decoder.add(Conv1DTranspose(256, 3, 1, 'same',activation='relu'))
        self.decoder.add(Conv1DTranspose(128, 4, 3, 'same',activation='relu'))
        self.decoder.add(Conv1DTranspose(64, 5, 1, 'same',activation='relu'))
        self.decoder.add(Conv1DTranspose(1, 6, 2, 'same', activation = 'sigmoid'))


    def call(self, input):

        encoded = self.encoder(input)
        densified = self.dense(encoded)
        decoded = self.decoder(densified)

        return decoded


    def loss_function(self, prediction, true):
        # mse = tf.keras.losses.MeanSquaredError()
        bce = tf.keras.losses.BinaryCrossentropy()
        # huber = tf.keras.losses.Huber()
        # lcosh = tf.keras.losses.LogCosh(reduction="auto", name="log_cosh")
        # return mse(true, prediction)
        return bce(true, prediction)

    def accuracy(self, prediction, true):
        ba = tf.keras.metrics.BinaryAccuracy()
        prediction = tf.round(prediction)
        true = tf.round(true)
        return ba(prediction, true)


    # def loss_function(self,prediction,true,mask):
    #     try:
    #         prediction = tf.squeeze(prediction,axis=3)
    #     except:
    #         pass
    #     try:
    #         mask = tf.cast(tf.squeeze(mask,axis=3),tf.float32)
    #     except:
    #         pass
    #     # rms_loss = tf.sqrt(tf.cast(tf.reduce_sum(tf.square((prediction - true) * tf.cast(mask,tf.float32)),axis=[1,2]),tf.float32) / tf.cast(tf.reduce_sum(mask,axis=[1,2]),tf.float32))
    #     # avgd = tf.reduce_mean(rms_loss)
    #     # RMS = tf.keras.metrics.RootMeanSquaredError()
    #     # RMS.update_state(true, prediction, sample_weight = mask)
    #     # print(tf.convert_to_tensor(RMS.result().numpy()))
    #     # return tf.convert_to_tensor(RMS.result().numpy())
    #     # mse = tf.keras.losses.MeanSquaredError(reduction = ttf.keras.losses.Reduction.NONE)
    #     # print(mse(true,prediction).shape)
    #
    #     a = tf.reduce_sum(tf.square(prediction - true) * mask,axis=[1,2]) / tf.reduce_sum(mask)
    #     return tf.reduce_mean(a)
    #
    # def loss_functionP(self,prediction,true,mask):
    #     try:
    #         prediction = tf.squeeze(prediction,axis=3)
    #     except:
    #         pass
    #     try:
    #         mask = tf.cast(tf.squeeze(mask,axis=3),tf.float32)
    #     except:
    #         pass
    #     # rms_loss = tf.sqrt(tf.cast(tf.reduce_sum(tf.square((prediction - true) * tf.cast(mask,tf.float32)),axis=[1,2]),tf.float32) / tf.cast(tf.reduce_sum(mask,axis=[1,2]),tf.float32))
    #     # avgd = tf.reduce_mean(rms_loss)
    #     # RMS = tf.keras.metrics.RootMeanSquaredError()
    #     # RMS.update_state(true, prediction, sample_weight = mask)
    #     # print(tf.convert_to_tensor(RMS.result().numpy()))
    #     # return tf.convert_to_tensor(RMS.result().numpy())
    #     # mse = tf.keras.losses.MeanSquaredError(reduction = ttf.keras.losses.Reduction.NONE)
    #     # print(mse(true,prediction).shape)
    #
    #     a = tf.reduce_sum(tf.math.abs(prediction - true) * mask, axis=[1, 2]) / tf.reduce_sum(mask) #
    #     return tf.reduce_mean(a)
