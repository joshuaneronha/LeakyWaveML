import numpy as np
import tensorflow as tf
from lwa_angle_model import LWAPredictionModel
from preprocess import import_data, get_next_batch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import pickle

parser = argparse.ArgumentParser(description='Build a deep learning model based on COMSOL simulations.')
parser.add_argument('-t', nargs='+',
                    help='List of timestamps that you want to import data for')

args = parser.parse_args()

def train(model, slots, results):
    completed = 0

    loss_list = []
    acc_list = []

    while completed < len(slots):

        our_size = model.batch_size

        if len(slots) - completed < model.batch_size:
            our_size = len(slots) - completed

        slot_batch, results_batch = get_next_batch(slots, results, completed, our_size)

        completed += our_size

        with tf.GradientTape() as tape:
            efarx = model.call(results_batch)
            efarx_loss = model.loss_function(efarx, slot_batch) #slots serves as the mask here

        gradients = tape.gradient(efarx_loss, model.trainable_variables)
        model.adam_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_list.append(efarx_loss)
        acc_list.append(model.accuracy(efarx, slot_batch))

    print('Efarx training loss: ', tf.reduce_mean(loss_list))
    print('Efarx training accuracy:', tf.reduce_mean(acc_list))

def test(model, slots, results):
    completed = 0

    loss_list = []
    acc_list = []

    while completed < len(slots):

        our_size = model.batch_size

        if len(slots) - completed < model.batch_size:
            our_size = len(slots) - completed

        slot_batch, results_batch = get_next_batch(slots, results, completed, our_size)

        completed += our_size

        efarx = model.call(results_batch)
        efarx_loss = model.loss_function(efarx, slot_batch)

        loss_list.append(efarx_loss)
        acc_list.append(model.accuracy(efarx, slot_batch))


    print('Efarx testing loss: ', tf.reduce_mean(loss_list))
    print('Efarx testing accuracy:', tf.reduce_mean(acc_list))

def main():

    slots, outputs = import_data(args.t)

    slot_train, slot_test, results_train, results_test = train_test_split(slots, outputs, test_size=0.2)

    # Model = LWAPredictionModel(lr = 0.0008, f=512, k=5, d=100, b=8)
    Model = LWAPredictionModel(lr = 0.0008, f=128, k=5, d=100, b=32)

    for i in np.arange(Model.epochs):
        print(i)
        train(Model, slot_train, results_train)
        print(' ')
        test(Model, slot_test, results_test)

    # test(Model, slot_test, results_test)

    # peak_sim = []

    for i in np.arange(10):

        random = np.random.randint(0,slot_test.shape[0])

        efarx = Model.call(tf.expand_dims(results_test[random],0))
        efarx = tf.squeeze(efarx)

        truedesign = tf.expand_dims(slot_test[random],axis = 1)

        fig = plt.figure()
        ax = fig.subplots(1,3, gridspec_kw={'width_ratios': [3, 1, 1]})
        ax[0].plot(results_test[random])
        ax[1].imshow(truedesign)
        ax[2].imshow(tf.round(tf.expand_dims(efarx,axis=1)))
        save_str = '1d/signal_to_slot/results/' + str(i) + '.png'
        fig.savefig(save_str)

    #     peak_sim.append([truefarx, efarx])
    #
    # with open('for_comparison.pkl', 'wb') as f:
    #     pickle.dump(peak_sim, f)

    pass



if __name__ == '__main__':
	main()
