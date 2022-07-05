import sys
sys.path.append('/Users/joshuaneronha/.conda/envs/mittleman/bin/python')
sys.path.append('ml_model/3grey')
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

    #work through all the training data
    while completed < len(slots):

        our_size = model.batch_size

        if len(slots) - completed < model.batch_size:
            our_size = len(slots) - completed

        #get the next set of data
        slot_batch, peaks_batch = get_next_batch(slots, peaks, completed, our_size)

        completed += our_size

        #call the model, calculate the loss function, and update the gradient
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


    print('Efarx testing loss: ', tf.reduce_mean(loss_list))
    print('Efarx testing accuracy:', tf.reduce_mean(acc_list))

def main():

    #import the data

    slots, peaks, waves = import_data()

    state = random.randint(0,100000)
    #choose a random number for the test-train split -- this is so we can get both the same peaks AND waves
    #when we split since we need to call train_test_split twice, ensuring we can use the same kernel

    slot_train, slot_test, peaks_train, peaks_test = train_test_split(slots, peaks, test_size=0.2, random_state = state)
    _, _, waves_train, waves_test = train_test_split(slots, waves, test_size=0.2, random_state = state)
    print(peaks_train.shape)
    print(peaks_test.shape)

    #initial the model
    Model = LWAPredictionModel()

    #and train it
    for i in np.arange(Model.epochs):
        print(i)
        train(Model, slot_train, peaks_train)
        print(' ')
        test(Model, slot_test, peaks_test)

    #call the model on the test data, and dump it

    peak_sim = []

    output = Model.call(peaks_test)
    saved_data = [peaks_test, slot_test, output, waves_test]

    with open('1d/peaks_to_slot/3grey/results/test_data627.pkl', 'wb') as f:
        pickle.dump(saved_data, f)

    return Model



if __name__ == '__main__':
	main()
