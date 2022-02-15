import numpy as np
import tensorflow as tf
from lwa_angle_model import LWAPredictionModel
from preprocess import import_data, get_next_batch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse

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

    Model = LWAPredictionModel()

    for i in np.arange(Model.epochs):
        print(i)
        train(Model, slot_train, results_train)

    test(Model, slot_test, results_test)

    random = np.random.randint(0,slot_test.shape[0])

    efarx = Model.call(tf.expand_dims(slot_test[random],0))
    efarx = tf.squeeze(efarx)

    truefarx = tf.squeeze(results_test[random])
    fig = plt.figure()
    ax = fig.subplots(1,2, gridspec_kw={'width_ratios': [1, 3]})
    ax[0].imshow(slot_test[random])
    ax[1].plot(truefarx)
    ax[1].plot(efarx)
    ax[1].legend(['True','Prediction'])
    save_str = 'results/test.png'
    fig.savefig(save_str)

    ones = np.array([[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]])
    zeros = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
    test_array = np.concatenate([ones,zeros,ones,zeros,ones,zeros],axis=0)

    efarx = Model.call(tf.expand_dims(test_array,0))
    efarx = tf.squeeze(efarx)

    fig = plt.figure()
    ax = fig.subplots(1,2, gridspec_kw={'width_ratios': [1, 3]})
    ax[0].imshow(test_array)
    ax[1].plot(efarx)
    ax[1].legend(['True','Prediction'])
    save_str = 'results/test_periodic.png'
    fig.savefig(save_str)

    pass

if __name__ == '__main__':
	main()
