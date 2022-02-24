import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, SimpleRNN, Reshape, Dropout, BatchNormalization, ReLU


class LWAPredictionModel(tf.keras.Model):
    def __init__(self, lr = 0.001, f = 128, k = 3, d = 100, b = 32):
        super(LWAPredictionModel, self).__init__()

        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.batch_size = b
        self.epochs = 20

        self.conv_layers = tf.keras.Sequential()

        self.conv_layers.add(Conv1D(f, k, 1, 'same', activation='relu'))
        self.conv_layers.add(MaxPooling1D())
        self.conv_layers.add(Conv1D(f, k, 1, 'same', activation='relu'))
        self.conv_layers.add(MaxPooling1D())
        self.conv_layers.add(Conv1D(f, k, 1, 'same', activation='relu'))
        self.conv_layers.add(MaxPooling1D())


        #
        self.dense_layers = tf.keras.Sequential()
        self.dense_layers.add(Flatten())
        self.dense_layers.add(Dense(int(d*20), activation = 'relu'))
        # self.dense_layers.add(Dropout(0.2))
        self.dense_layers.add(Dense(int(d*12), activation = 'relu'))
        # self.dense_layers.add(Dropout(0.2))
        self.dense_layers.add(Dense(int(d*6), activation = 'relu'))
        # self.dense_layers.add(Dropout(0.2))
        self.dense_layers.add(Dense(int(d), activation = 'relu'))
        # self.dense_layers.add(Dropout(0.2))
        self.dense_layers.add(Dense(36, activation = 'sigmoid'))

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
        # post_dense = self.dense_layers(input)

        return post_dense

    def loss_function(self, prediction, true):
        # mse = tf.keras.losses.MeanSquaredError()
        bce = tf.keras.losses.BinaryCrossentropy()
        # huber = tf.keras.losses.Huber()
        # lcosh = tf.keras.losses.LogCosh(reduction="auto", name="log_cosh")
        # return mse(true, prediction)
        return bce(true, prediction)

    # def signal_similarity(self,prediction, true):
    #
    # ## Time domain similarity
    #     # ref_time = np.correlate(ref_rec,ref_rec)
    #     # inp_time = np.correlate(ref_rec,input_rec)
    #     # diff_time = abs(ref_time-inp_time)
    #
    #     tf.linalg.tensor_diag_part(tfp.stats.correlation(ref_rec,multi2,sample_axis=1,event_axis=0))
    #
    #     ## Freq domain similarity
    #     ref_freq = np.correlate(np.fft.fft(ref_rec),np.fft.fft(ref_rec))
    #     inp_freq = np.correlate(np.fft.fft(ref_rec),np.fft.fft(input_rec))
    #     diff_freq = abs(ref_freq-inp_freq)
    #
    #     ## Power similarity
    #     ref_power = np.sum(ref_rec**2)
    #     inp_power = np.sum(input_rec**2)
    #     diff_power = abs(ref_power-inp_power)
    #
    #     return float((0.333*diff_time) + (0.333*diff_freq) + (0.333*diff_power))

    # def signal_loss_function(self,prediction,true):
    #     prediction = tf.cast(prediction, tf.float32)
    #     true = tf.cast(true, tf.float32)
    #
    #     prediction_complex = tf.cast(prediction,tf.complex64)
    #     true_complex = tf.cast(true,tf.complex64)
    #
    #     time_corr = tf.abs(tf.linalg.tensor_diag_part(tfp.stats.correlation(prediction,true,sample_axis=1,event_axis=0)))
    #     freq_corr = tf.abs(tf.linalg.tensor_diag_part(tfp.stats.correlation(tf.signal.fft(prediction_complex),tf.signal.fft(true_complex),sample_axis=1,event_axis=0)))
    #     amp_match = 1 / (tf.reduce_sum(tf.abs((prediction ** 2) - (true ** 2)), axis=1) + 1)
    #
    #     # print(tf.reduce_sum((0.5 * time_corr) + (0.5 * freq_corr)))
    #
    #     return -1*tf.reduce_mean((0.333 * time_corr) + (0.333 * freq_corr) + (0.333 * amp_match))

    # def signal_loss_function(self,prediction,true):
    #     prediction = tf.cast(prediction, tf.float32)
    #     true = tf.cast(true, tf.float32)
    #
    #     prediction_complex = tf.cast(prediction,tf.complex64)
    #     true_complex = tf.cast(true,tf.complex64)
    #
    #     def np_corr(first,second):
    #         return tf.convert_to_tensor([np.correlate(first[i],second[i])[0] for i in np.arange(len(first))])
    #
    #     time_corr = tf.numpy_function(np_corr, [true, prediction], tf.float32)
    #     time_corr_ref = tf.numpy_function(np_corr, [true, true], tf.float32)
    #     diff_time_corr = tf.abs(time_corr - time_corr_ref)
    #
    #     fft_corr = tf.cast(tf.numpy_function(np_corr, [tf.signal.fft(true_complex), tf.signal.fft(prediction_complex)], tf.float32), tf.float32)
    #     fft_corr_ref = tf.cast(tf.numpy_function(np_corr, [tf.signal.fft(true_complex), tf.signal.fft(true_complex)], tf.float32), tf.float32)
    #     diff_fft_corr = tf.cast(tf.abs(fft_corr - fft_corr_ref), tf.float32)
    #
    #     amp_corr = tf.abs(tf.reduce_sum(prediction ** 2,axis = 1) - tf.reduce_sum(true ** 2,axis = 1))
    #
    #     # print(diff_time_corr.shape, diff_fft_corr.shape, amp_corr.shape)
    #     # print(diff_time_corr)
    #     # print(diff_fft_corr)
    #     # print(amp_corr)
    #
    #     # print(tf.reduce_sum((0.5 * time_corr) + (0.5 * freq_corr)))
    #
    #     mse = tf.keras.losses.MeanSquaredError()
    #
    #     return tf.reduce_mean((0.500 * diff_time_corr) + (0 * diff_fft_corr) + (0.500 * amp_corr) + (0 * mse(true, prediction)))
