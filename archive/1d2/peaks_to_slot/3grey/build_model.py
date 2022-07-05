import sys
sys.path.append('/Users/joshuaneronha/.conda/envs/mittleman/bin/python')
sys.path.append('1d/peaks_to_slot/3grey')
import numpy as np
import tensorflow as tf
from peaks_to_slots_model import LWAPredictionModel
from preprocess import import_data, get_next_batch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import pickle
import random

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
        # assump_acc_list.append(model.assump_accuracy(efarx,slot_batch))


    print('Efarx testing loss: ', tf.reduce_mean(loss_list))
    print('Efarx testing accuracy:', tf.reduce_mean(acc_list))
    # print('Efarx testing assumption accuracy:', tf.reduce_mean(assump_acc_list))

def main():

    slots, peaks, waves = import_data()

    state = random.randint(0,100000)

    slot_train, slot_test, peaks_train, peaks_test = train_test_split(slots, peaks, test_size=0.2, random_state = state)
    _, _, waves_train, waves_test = train_test_split(slots, waves, test_size=0.2, random_state = state)
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

    output = Model.call(peaks_test)
    saved_data = [peaks_test, slot_test, output, waves_test]

    with open('1d/peaks_to_slot/3grey/results/test_data627.pkl', 'wb') as f:
        pickle.dump(saved_data, f)

    #
    # # for random in [36, 185, 234, 368, 492, 567, 698, 722, 813, 922]:
    # for i in np.arange(10):
    #
    #     random = np.random.randint(0,slot_test.shape[0])
    #
    #     efarx = Model.call(tf.expand_dims(peaks_test[random],0))
    #     # print(tf.expand_dims(peaks_test[random],0).shape)
    #     efarx = tf.squeeze(efarx)
    #
    #     truedesign = tf.expand_dims(slot_test[random],axis = 1)
    #
    #     fig = plt.figure()
    #     ax = fig.subplots(1,4, gridspec_kw={'width_ratios': [3, 1, 1, 1]})
    #     ax[0].plot(peaks_test[random])
    #     ax[1].imshow(truedesign)
    #     ax[2].imshow(tf.expand_dims(efarx,axis=1))
    #
    #     efarx = efarx.numpy()
    #
    #     efarx = np.where((efarx > 0) & (efarx <= 0.125), 0, efarx)
    #     efarx = np.where((efarx > 0.125) & (efarx <= 0.375), 0.25, efarx)
    #     efarx = np.where((efarx > 0.375) & (efarx <= 0.625), 0.50, efarx)
    #     efarx = np.where((efarx > 0.625) & (efarx <= 0.875), 0.75, efarx)
    #     efarx = np.where((efarx > 0.875), 1.00, efarx)
    #
    #     ax[3].imshow(tf.expand_dims(efarx,axis=1))
    #     save_str = '1d/peaks_to_slot/grey/results/' + str(i) + '.png'
    #     fig.savefig(save_str)
    #
    #     peak_sim.append([peaks_test[random],truedesign, efarx])
    #
    # with open('1d/peaks_to_slot/grey/results/generated_slots.pkl', 'wb') as f:
    #     pickle.dump(peak_sim, f)
    # #
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
	thismodel = main()
# thismodel
#
# exp_test = 20*np.log10(np.loadtxt('experiments/greytestforml.csv',delimiter=','))
#
# floquet = list(np.array([  6.,  16.,  24.,  29.,  32.,  34.,  49.,  60.,  62.,  63.,  66.,
#         69.,  71.,  73.,  77.,  81.,  84.,  86.,  89.,  90.,  95., 100.,
#        101., 104., 106., 110., 114., 118., 120., 123., 126., 129., 131.,
#        132., 147.]).astype('int'))
#
# filtered = tf.reshape(exp_testy[[x-2 for x in floquet]],(1,35))
# filtered = filtered / np.max(filtered)
# out = thismodel.call(filtered)
#
#
# plt.imshow(tf.transpose(out),cmap='YlGnBu')
# with open('1d/peaks_to_slot/grey/results/test_data530.pkl','rb') as f:
#     output = pickle.load(f)
# output[0][0]
# plt.imshow(tf.expand_dims(output[2][400],1))
#
# test_array = np.array([0.44546966, 0.48853195, 0.59419797, 0.73774793, 0.81137194,
#        0.83679913, 1.        , 0.62820602, 0.53438953, 0.57906827,
#        0.59198008, 0.59870489, 0.56357657, 0.36732016, 0.62812462,
#        0.76372193, 0.61344691, 0.5473943 , 0.70663523, 0.71242345,
#        0.71085666, 0.31885805, 0.38955326, 0.5811831 , 0.57733111,
#        0.42004987, 0.54043865, 0.34543648, 0.42882175, 0.48020032,
#        0.51591963, 0.56069232, 0.56079585, 0.55677631, 0.55533863])
#
# out_array = thismodel.call(peaks[0:500])
#
# np.savetxt('4_grey_477_slot.txt',np.array(out))
# plt.imshow(tf.transpose(out),cmap='YlGnBu')
# out
#
# with open('1d/peaks_to_slot/constant_power/results/test_data510.pkl','rb') as file:
#     test_data = pickle.load(file)
#     peaks, true, pred, waves = test_data
#
# peaks.shape
#
# out_array
# np.savetxt('4_grey_all_slots.txt',np.array(out_array))
