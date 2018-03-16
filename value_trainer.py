from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from ConvolutionalNeuralNetwork2 import ConvolutionalNeuralNetwork2

from Othello import *

import _pickle as pickle
import numpy as np
import random
import time

def load_data():
	print('Loading Data...')
	start_time = int(round(time.time() * 1000))
	states = np.load('data/generated/new_states.npy')
	values = np.load('data/generated/new_values.npy')
	#np.save('data/generated/new_values.npy', np.reshape(values, (-1,1)), allow_pickle=False)	
	end_time = int(round(time.time() * 1000))
	print('Finished in {0} seconds.\n'.format((end_time-start_time) / 1000.0))
	print(states.shape, values.shape)
	return states, values

def main():
	np.set_printoptions(suppress=True, precision=4)
	states, values = load_data()

	cnn = ConvolutionalNeuralNetwork2(
		input_depth = states.shape[-1],
		num_layers = 6,
		num_filters = [64, 64, 128, 128, 256, 256],
		dropout_rate = 0.0,
	)
	cnn.load("models/VALUE_NETWORK/VALUE_NETWORK")

	index = 670000
	valid_size = 10000
	x_train, y_train = states[:index], values[:index]
	x_valid, y_valid = states[index:index+valid_size], values[index:index+valid_size]
	
	cnn.train(
		x_train, y_train,
		x_valid, y_valid,
		batch_size=1000,
		num_epochs=20,
		learning_rate = 1e-6,
		model_name = "models/VALUE_NETWORK/VALUE_NETWORK"
	)

if __name__ == '__main__':
    main()
