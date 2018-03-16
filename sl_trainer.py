from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from Othello import *

import _pickle as pickle
import numpy as np
import random
import time

def load_data():
	print('Loading Data...')
	start_time = int(round(time.time() * 1000))
	states = np.load('data/states.npy')
	moves = np.load('data/moves.npy')
	end_time = int(round(time.time() * 1000))
	print('Finished in {0} seconds.\n'.format((end_time-start_time) / 1000.0))

	return states, moves

def main():
	np.set_printoptions(suppress=True, precision=4)
	states, moves = load_data()

	cnn = ConvolutionalNeuralNetwork(
		input_depth = states.shape[-1],
		num_layers = 6,
		num_filters = [64, 64, 128, 128, 256, 256],
		dropout_rate = 0.0,
		regularization = 0,
	)

	index = 3000000
	valid_size = 10000
	x_train, y_train = states[:index], moves[:index]
	x_valid, y_valid = states[index:index+valid_size], moves[index:index+valid_size]
	
	cnn.train(
		x_train, y_train,
		x_valid, y_valid,
		batch_size=1000,
		num_epochs=20,
		learning_rate = 1e-4,
		model_name = "models/SL_POLICY_NETWORK/SL_POLICY_NETWORK"
	)

def main2():
	np.set_printoptions(suppress=True, precision=4)
	states, moves = load_data()

	cnn = ConvolutionalNeuralNetwork(
		input_depth = 3,
		num_layers = 6,
		num_filters = [64, 64, 128, 128, 256, 256],
		dropout_rate = 0.0,
		regularization = 0,
	)
	cnn.load("models/SL_POLICY_NETWORK/SL_POLICY_NETWORK")
	input_state = OthelloBoard([
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 1, 1, 0, 0, 0, 0],
		[0, 0,-1, 1,-1,-1,-1, 0],
		[0, 0,-1,-1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
	])
	print(cnn.get_accuracy(states[-10000:], moves[-10000:]))
	#print(cnn.predict(input_state.get_feature_matrix(OthelloConstants.BLACK)))

if __name__ == '__main__':
    #main()
    main2()