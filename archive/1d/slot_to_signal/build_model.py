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

    while completed < len(slots):

        our_size = model.batch_size

        if len(slots) - completed < model.batch_size:
            our_size = len(slots) - completed

        slot_batch, results_batch = get_next_batch(slots, results, completed, our_size)

        completed += our_size

        with tf.GradientTape() as tape:
            efarx = model.call(slot_batch)
            efarx_loss = model.loss_function(efarx, results_batch) #slots serves as the mask here

        gradients = tape.gradient(efarx_loss, model.trainable_variables)
        model.adam_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_list.append(efarx_loss)

    print('Efarx training loss: ', tf.reduce_mean(loss_list))

def test(model, slots, results):
    completed = 0

    loss_list = []

    while completed < len(slots):

        our_size = model.batch_size

        if len(slots) - completed < model.batch_size:
            our_size = len(slots) - completed

        slot_batch, results_batch = get_next_batch(slots, results, completed, our_size)

        completed += our_size

        efarx = model.call(slot_batch)
        efarx_loss = model.loss_function(efarx, results_batch)

        loss_list.append(efarx_loss)
    print('Efarx testing loss: ', tf.reduce_mean(loss_list))

def main():

    slots, outputs = import_data(args.t)
    slot_train, slot_test, results_train, results_test = train_test_split(slots, outputs, test_size=0.2)

    # Model = LWAPredictionModel(lr = 0.0008, f=512, k=5, d=100, b=8)
    Model = LWAPredictionModel(lr = 0.0008, f=128, k=5, d=100, b=32)

    for i in np.arange(Model.epochs):
        print(i)
        train(Model, slot_train, results_train)

    test(Model, slot_test, results_test)

    peak_sim = []

    for i in np.arange(10):

        random = np.random.randint(0,slot_test.shape[0])

        efarx = Model.call(tf.expand_dims(slot_test[random],0))
        efarx = tf.squeeze(efarx)

        truefarx = tf.squeeze(results_test[random])
        fig = plt.figure()
        ax = fig.subplots(1,2, gridspec_kw={'width_ratios': [1, 3]})
        ax[0].imshow(tf.squeeze(slot_test[random],axis=2))
        ax[1].plot(truefarx)
        ax[1].plot(efarx)
        ax[1].legend(['True','Prediction'])
        save_str = '1d/slot_to_signal/results/array/test_forexs_' + str(i) + '.png'
        fig.savefig(save_str)

        peak_sim.append([truefarx, efarx])

    with open('ten_examples.pkl', 'wb') as f:
        pickle.dump(peak_sim, f)

    pass



if __name__ == '__main__':
	main()
