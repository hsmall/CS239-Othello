from Othello import *
from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork

import numpy as np

class PolicyPlayer:
	def __init__(self, cnn, greedy=False):
		self.cnn = cnn
		self.greedy = greedy

	def set_color(self, color):
		self.color = color

	def reset(self):
		pass

	def normalize(self, array):
		total = sum(array)
		return [elem / total for elem in array]

	def select_move(self, state, legal_moves):
		prob_distr = self.cnn.predict(state.get_feature_matrix(self.color, legal_moves))[0,:,:]
		move_probs = self.normalize([ prob_distr[row, col] for row, col in legal_moves ])
		
		if not self.greedy:
			move_index = np.random.choice(range(len(legal_moves)), 1, p=move_probs)[0]
		else:
			move_index = np.argmax(move_probs)
		
		return legal_moves[move_index], move_probs[move_index]

	def receive_opponent_move(self, move):
		pass