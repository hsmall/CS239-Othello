from copy import deepcopy
from Othello import *

import random

class HeuristicPlayer:
	def __init__(self, value_fn, depth=2):
		self.value_fn = value_fn
		self.depth = depth

	def set_color(self, color):
		self.color = color

	def reset(self):
		pass

	def select_move(self, state, legal_moves):
		value, move = self.minimax(state, self.color, self.depth)
		return move, value

	def minimax(self, state, player, depth):
		legal_moves = state.get_legal_moves(player)
		if depth == 0 or len(legal_moves) == 0: return (self.value_fn(state, self.color), None)

		next_depth = depth-1 if player != self.color else depth
		move_choices = []
		for move in legal_moves:
			next_state = deepcopy(state)
			move_choices.append( (self.minimax(next_state, -player, next_depth)[0], move) )

		if player == self.color:
			optimal_value = max(move_choices)[0]
		else:
			optimal_value = min(move_choices)[0]

		optimal_moves = [(value, move) for value, move in move_choices if value == optimal_value]
		result = random.choice(optimal_moves)

		return result

	def receive_opponent_move(self, move):
		pass