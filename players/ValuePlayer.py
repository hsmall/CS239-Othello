from copy import deepcopy
from Othello import *
from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork

import numpy as np

class ValuePlayer:
	def __init__(self, cnn):
		self.cnn = cnn

	def set_color(self, color):
		self.color = color

	def reset(self):
		pass

	def normalize(self, array):
		total = sum(array)
		return [elem / total for elem in array]

	def select_move(self, state, legal_moves):
		values = []
		for move in legal_moves:
			new_state = deepcopy(state)
			new_state.make_move(move, self.color)
			value = self.cnn.predict(new_state.get_feature_matrix(self.color, legal_moves))
			values.append(float(value))

		move_index = np.argmax(values)
		return legal_moves[move_index], values[move_index]

	def receive_opponent_move(self, move):
		pass