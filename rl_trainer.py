from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from Othello import *
from OthelloSimulator import *
from players.CNNPlayer import *

import numpy as np
import time

def load_data():
	print('Loading Data...')
	start_time = int(round(time.time() * 1000))
	states = np.load('data/states.npy')
	moves = np.load('data/moves.npy')
	end_time = int(round(time.time() * 1000))
	print('Finished in {0} seconds.\n'.format((end_time-start_time) / 1000.0))

	return states, moves

def to_tuple(a):
    try:
        return tuple(to_tuple(i) for i in a)
    except TypeError:
        return a

def run_simulation_batch(simulator, num_simulations):
	result_states = []
	result_moves = []

	for simulation in range(num_simulations):
		scores, play_by_play = simulator.simulate_game(record=True, verbose=False)
		
		if scores[OthelloConstants.BLACK] == scores[OthelloConstants.WHITE]:
			continue
		winner = OthelloConstants.BLACK if scores[OthelloConstants.BLACK] > scores[OthelloConstants.WHITE] else OthelloConstants.WHITE

		states, players, moves = zip(*play_by_play)
		win_indicator = np.reshape(winner*np.array(players), (-1, 1))
		moves = np.array(moves) * win_indicator

		result_states.extend(states)
		result_moves.extend(moves)
	
	return np.array(result_states), np.array(result_moves)

def main():
	np.set_printoptions(suppress=True, precision=3, threshold=10000)
	states, moves = load_data()
	
	cnn = ConvolutionalNeuralNetwork(
		input_depth = states.shape[-1],
		num_layers = 6,
		num_filters = [64, 64, 128, 128, 256, 256],
		dropout_rate = 0.0,
		regularization = 0,
	)
	cnn.load('models/SL_POLICY_NETWORK/SL_POLICY_NETWORK')

	simulator = OthelloSimulator(
		CNNPlayer(cnn, greedy=False),
		CNNPlayer(cnn, greedy=False)
	)

	num_simulations = 1000000
	batch_size = 100
	for simulation_batch in range(num_simulations//batch_size):
		print('Simulation Batch #{0}'.format(simulation_batch))
		start_time = int(round(time.time() * 1000))

		sim_states, sim_moves = run_simulation_batch(simulator, batch_size)
		cnn.train(
			x_train = sim_states,
			y_train = sim_moves,
			x_valid = [],
			y_valid = [],
			batch_size=100,
			num_epochs=1,
			learning_rate = 5e-8,
		)

		end_time = int(round(time.time() * 1000))
		print('Finished in {0} seconds.\n'.format((end_time-start_time) / 1000.0))

		if simulation_batch % 10 == 0:
			cnn.save('models/RL_POLICY_NETWORK/RL_POLICY_NETWORK')


if __name__ == '__main__':
    main()