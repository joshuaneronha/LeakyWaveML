import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Conv1DTranspose

alpha = tf.random.normal([1,360,1])

bravo = Conv1D(32, 3, 2, 'same', activation='relu')(alpha)
charlie = Conv1D(64, 5, 2, 'same', activation='relu')(bravo)
delta = Conv1D(128, 7, 3, 'same', activation='relu')(charlie)
echo = Conv1D(256, 9, 5, 'same', activation='relu')(delta)

foxtrot = Dense(512, activation = 'relu')(echo)
foxtrot.shape

golf = Conv1DTranspose(256, 4, 1, 'same',activation='relu')(foxtrot)
hotel = Conv1DTranspose(256, 4, 1, 'same',activation='relu')(golf)
india = Conv1DTranspose(128, 4, 3, 'same',activation='relu')(hotel)
juliet = Conv1DTranspose(64, 4, 1, 'same',activation='relu')(india)
kilo = Conv1DTranspose(1, 4, 2, 'same', activation = 'sigmoid')(juliet)
kilo.shape

        self.decoder = tf.keras.Sequential()
        # self.decoder.add(Conv1DTranspose(512, 4, 1, 'same',activation='relu'))
        # self.decoder.add(Conv1DTranspose(256, 4, 1, 'same',activation='relu'))
        # self.decoder.add(Conv1DTranspose(128, 4, 3, 'same',activation='relu'))
        # self.decoder.add(Conv1DTranspose(64, 4, 1, 'same',activation='relu'))
        self.decoder.add(Conv1DTranspose(1, 4, 2, 'same', activation = 'sigmoid'))
