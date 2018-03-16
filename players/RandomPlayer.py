import random

class RandomPlayer:
	def __init__(self):
		pass
	
	def set_color(self, color):
		self.color = color

	def reset(self):
		pass

	def select_move(self, board, legal_moves):
		return random.choice(legal_moves), None

	def receive_opponent_move(self, move):
		pass