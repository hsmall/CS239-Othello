from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from Othello import *
from OthelloSimulator import *

import glob
import numpy as np
import os
import random
import time

from players.CNNPlayer import *

def get_cnn_player(model_name):
	cnn = ConvolutionalNeuralNetwork(
		input_depth = 3,
		num_layers = 6,
		num_filters = [64, 64, 128, 128, 256, 256],
		dropout_rate = 0.0,
		regularization = 0,
	)
	cnn.load(model_name)
	return CNNPlayer(cnn, greedy=False)

def main():
	print('Generating Files...')
	np.set_printoptions(suppress=True, precision=4)
	sl_player = get_cnn_player("models/SL_POLICY_NETWORK/SL_POLICY_NETWORK")
	rl_player = get_cnn_player("models/RL_POLICY_NETWORK/RL_POLICY_NETWORK")

	num_states = 1000000
	batch_size = 10000

	states, values = [], []
	start_time = int(round(time.time() * 1000))
	
	for i in range(num_states):
		if i > 0 and i % batch_size == 0:
			end_time = int(round(time.time() * 1000))
			elapsed_time = (end_time - start_time) / 1000.0
			
			np.save('data/generated/states_{0}'.format(end_time), states, allow_pickle=False)
			np.save('data/generated/values_{0}'.format(end_time), values, allow_pickle=False)
			print('Generated/Saved {0} States in {1} seconds'.format(batch_size, elapsed_time))

			start_time = int(round(time.time() * 1000))
			states, values = [], []

		while True:
			try:
				state, value = OthelloSimulator.generate_state(sl_player, rl_player)
				states.append(state)
				values.append(value)
				break
			except ValueError:
				pass

		
def main2():
	print('Merging Generated Files...')
	state_files = sorted(glob.glob('data/generated/states_*.npy'))
	value_files = sorted(glob.glob('data/generated/values_*.npy'))

	for i in range(len(state_files)):
		assert state_files[i][22:-4] == value_files[i][22:-4]

	states = np.concatenate([np.load(file) for file in state_files])
	values = np.reshape(np.concatenate([np.load(file) for file in value_files]), (-1, 1))

	print(states.shape, values.shape)
	np.save('data/generated/states.npy', states, allow_pickle=False)
	np.save('data/generated/values.npy', values, allow_pickle=False)


def to_tuple(a):
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a

def main3():
	states = np.load('data/generated/states.npy')
	values = np.load('data/generated/values.npy')

	states_index = {}
	for i in range(len(states)):
		if i % 10000 == 0: print(i)
		states_index.setdefault(to_tuple(states[i]), []).append(values[i])

	print(len(states_index))
	
	new_states, new_values = [], []
	for state in states_index:
		new_states.append(state)
		new_values.append(np.mean(states_index[state]))

	data = list(zip(new_states, new_values))
	data = random.sample(data, len(data))
	new_states, new_values = zip(*data)

	np.save('data/generated/new_states.npy', new_states, allow_pickle=False)
	np.save('data/generated/new_values.npy', new_values, allow_pickle=False)


if __name__ == '__main__':
    main3()


