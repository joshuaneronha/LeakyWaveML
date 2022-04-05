import numpy as np
import tensorflow as tf
from peaks_to_slots_model import LWAPredictionModel
from preprocess import import_data, get_next_batch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import pickle

parser = argparse.ArgumentParser(description='Build a deep learning model based on COMSOL simulations.')

def train(model, slots, peaks):
    completed = 0

    loss_list = []
    acc_list = []

    while completed < len(slots):

        our_size = model.batch_size

        if len(slots) - completed < model.batch_size:
            our_size = len(slots) - completed

        slot_batch, peaks_batch = get_next_batch(slots, peaks, completed, our_size)

        completed += our_size

        with tf.GradientTape() as tape:
            efarx = model.call(peaks_batch)
            efarx_loss = model.loss_function(efarx, slot_batch) #slots serves as the mask here
            # print(efarx_loss)

        gradients = tape.gradient(efarx_loss, model.trainable_variables)
        model.adam_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_list.append(efarx_loss)
        acc_list.append(model.accuracy(efarx, slot_batch))

    print('Efarx training loss: ', tf.reduce_mean(loss_list))
    print('Efarx training accuracy:', tf.reduce_mean(acc_list))

def test(model, slots, peaks):
    completed = 0

    loss_list = []
    acc_list = []
    assump_acc_list = []

    while completed < len(slots):

        our_size = model.batch_size

        if len(slots) - completed < model.batch_size:
            our_size = len(slots) - completed

        slot_batch, peaks_batch = get_next_batch(slots, peaks, completed, our_size)

        completed += our_size

        efarx = model.call(peaks_batch)
        efarx_loss = model.loss_function(efarx, slot_batch)

        loss_list.append(efarx_loss)
        acc_list.append(model.accuracy(efarx, slot_batch))
        assump_acc_list.append(model.assump_accuracy(efarx,slot_batch))


    # print('Efarx testing loss: ', tf.reduce_mean(loss_list))
    print('Efarx testing accuracy:', tf.reduce_mean(acc_list))
    print('Efarx testing assumption accuracy:', tf.reduce_mean(assump_acc_list))

def main():

    slots, peaks = import_data()

    slot_train, slot_test, peaks_train, peaks_test = train_test_split(slots, peaks, test_size=0.2)
    print(peaks_train.shape)
    print(peaks_test.shape)

    Model = LWAPredictionModel()

    for i in np.arange(Model.epochs):
        print(i)
        train(Model, slot_train, peaks_train)
        print(' ')
        test(Model, slot_test, peaks_test)

    # test(Model, slot_test, peaks_test)

    peak_sim = []

    # for random in [36, 185, 234, 368, 492, 567, 698, 722, 813, 922]:
    for i in np.arange(10):

        random = np.random.randint(0,slot_test.shape[0])

        efarx = Model.call(tf.expand_dims(peaks_test[random],0))
        # print(tf.expand_dims(peaks_test[random],0).shape)
        efarx = tf.squeeze(efarx)

        truedesign = tf.expand_dims(slot_test[random],axis = 1)

        fig = plt.figure()
        ax = fig.subplots(1,4, gridspec_kw={'width_ratios': [3, 1, 1, 1]})
        ax[0].plot(peaks_test[random])
        ax[1].imshow(truedesign)
        ax[2].imshow(tf.expand_dims(efarx,axis=1))
        ax[3].imshow(tf.round(tf.expand_dims(efarx,axis=1)))
        save_str = '1d/peaks_to_slot/constant_power/results/' + str(i) + '.png'
        fig.savefig(save_str)

        peak_sim.append([peaks_test[random],truedesign, efarx])

    with open('1d/peaks_to_slot/constant_power/results/generated_slots.pkl', 'wb') as f:
        pickle.dump(peak_sim, f)
    #
    # peaks_of_interest = np.array([20,20,20,20,20,20,
    #                             20,20,20,20,20,20,
    #                             20,20,20,20,20,20,
    #                             20,20,20,20,20,20,
    #                             20,20,20,20,20,20,
    #                             20,20,20,20,20
    #
    # ])
    # print(tf.expand_dims(peaks_of_interest,0))
    # results = Model.call(tf.expand_dims(peaks_of_interest,0))
    #
    #
    # with open('1d/peaks_to_slot/results/testing_slots.pkl', 'wb') as f:
    #     pickle.dump([peaks_of_interest, results], f)
    # Model.compute_output_shape(input_shape=(35,1))
    # Model.save_weights('1d/peaks_to_slot/results/model_weights')

    return Model



if __name__ == '__main__':
	main()
