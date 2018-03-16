
class HumanPlayer:
	def __init__(self):
		pass

	def set_color(self, color):
		self.color = color

	def reset(self):
		pass

	def in_range(self, value, minimum, maximum):
		return value >= minimum and value < maximum

	def select_move(self, state, legal_moves):
		move_index = -1
		while True:
			print('Legal Moves: {0}'.format(legal_moves))
			move_index = input("Select Move: ")
			if len(move_index) == 1 and self.in_range(ord(move_index), 97, 97+len(legal_moves)):
				break
			print('Invalid move ({0}), please try again.'.format(move_index))
		
		return legal_moves[ord(move_index)-97], None

	def receive_opponent_move(self, move):
		pass