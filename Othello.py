import numpy as np

class OthelloConstants:
	BLACK, WHITE = -1, 1
	BOARD_SIZE = 8
	INITIAL_BOARD = [
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 1,-1, 0, 0, 0],
		[0, 0, 0,-1, 1, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0]
	]

class OthelloBoard:

	def __init__(self, board):
		self.player_masks = OthelloBoard.convert_board_to_masks(board)

	def __eq__(self, other):
		return self.player_masks == other.player_masks

	def __neq__(self, other):
		return not self.__eq__(other)

	@staticmethod
	def convert_board_to_masks(board):
		black = OthelloBoard.get_player_bit_mask(board, OthelloConstants.BLACK)
		white = OthelloBoard.get_player_bit_mask(board, OthelloConstants.WHITE)
		return { OthelloConstants.BLACK : black, OthelloConstants.WHITE : white }

	@staticmethod
	def get_player_bit_mask(board, player):
		mask, index = 0, 1 << 63
		for row in range(OthelloConstants.BOARD_SIZE):
			for col in range(OthelloConstants.BOARD_SIZE):
				if board[row][col] == player:
					mask |= index
				index >>= 1
		return mask

	@staticmethod
	def convert_masks_to_board(masks):
		black_mask = masks[OthelloConstants.BLACK]
		white_mask = masks[OthelloConstants.WHITE]

		board_size = OthelloConstants.BOARD_SIZE
		board = [[0 for row in range(board_size)] for col in range(board_size)]

		index = 1 << 63
		for row in range(OthelloConstants.BOARD_SIZE):
			for col in range(OthelloConstants.BOARD_SIZE):
				if black_mask & index:
					board[row][col] = OthelloConstants.BLACK
				if white_mask & index:
					board[row][col] = OthelloConstants.WHITE
				index >>= 1

		return board

	@staticmethod
	def bit_mask_to_numpy_array(bit_mask):
		return np.reshape(np.array(list(np.binary_repr(bit_mask, width=64))).astype(np.int8), (8,8))

	def get_board(self):
		return OthelloBoard.convert_masks_to_board(self.player_masks)

	def get_feature_matrix(self, player, legal_moves=None):
		if legal_moves is None:
			legal_moves = self.get_legal_moves(player)

		feature_matrix = np.zeros((
			OthelloConstants.BOARD_SIZE,
			OthelloConstants.BOARD_SIZE,
			3
		))
		feature_matrix[:,:,0] = OthelloBoard.bit_mask_to_numpy_array(self.player_masks[player]) 
		feature_matrix[:,:,1] = OthelloBoard.bit_mask_to_numpy_array(self.player_masks[-player])
		
		if len(legal_moves) > 0:
			x, y = zip(*legal_moves)
			feature_matrix[:,:,2][x, y] = 1
		
		return feature_matrix

	def get_legal_moves(self, player):
		left_right_mask = 0x7e7e7e7e7e7e7e7e  # Both most left-right edge are 0, else 1
		top_bottom_mask = 0x00ffffffffffff00  # Both most top-bottom edge are 0, else 1
		edge_mask = left_right_mask & top_bottom_mask

		own_mask = self.player_masks[player]
		opp_mask = self.player_masks[-player]		

		legal_mask = 0
		legal_mask |= self.search_offset_left(own_mask, opp_mask, left_right_mask, 1) # Left
		legal_mask |= self.search_offset_left(own_mask, opp_mask, edge_mask, 9) # Top-Left
		legal_mask |= self.search_offset_left(own_mask, opp_mask, top_bottom_mask, 8) # Top
		legal_mask |= self.search_offset_left(own_mask, opp_mask, edge_mask, 7) # Top-Right
		legal_mask |= self.search_offset_right(own_mask, opp_mask, left_right_mask, 1) # Right
		legal_mask |= self.search_offset_right(own_mask, opp_mask, edge_mask, 9) # Bottom-Right
		legal_mask |= self.search_offset_right(own_mask, opp_mask, top_bottom_mask, 8) # Bottom
		legal_mask |= self.search_offset_right(own_mask, opp_mask, edge_mask, 7) # Bottom-Left

		legal_moves = []
		index = 1 << 63
		for row in range(OthelloConstants.BOARD_SIZE):
			for col in range(OthelloConstants.BOARD_SIZE):
				if legal_mask & index:
					legal_moves.append((row, col))
				index >>= 1

		return legal_moves

	def search_offset_left(self, own_mask, opp_mask, capturable_mask, offset):
		mask = opp_mask & capturable_mask
		blank_mask = ~(own_mask | opp_mask)
		
		moves = mask & (own_mask << offset) 
		for i in range(5): # Up to six stones can be turned at once
			moves |= mask & (moves << offset)
		return blank_mask & (moves << offset) # Only the blank squares can be played

	def search_offset_right(self, own_mask, opp_mask, capturable_mask, offset):
		mask = opp_mask & capturable_mask
		blank_mask = ~(own_mask | opp_mask)
		
		moves = mask & (own_mask >> offset)
		for i in range(5): # Up to six stones can be turned at once
			moves |= mask & (moves >> offset)

		return blank_mask & (moves >> offset) # Only the blank squares can be played

	def make_move(self, move, player):
		row, col = move

		if row is None or col is None:
			print(move, player)

		pos = 64 - (row*OthelloConstants.BOARD_SIZE + col + 1)
		flipped = self.calc_flip(pos, player)

		self.player_masks[player] |= ((1 << pos) | flipped)
		self.player_masks[-player] ^= (self.player_masks[-player] & flipped)

		#self.player_masks[player] = b64(self.player_masks[player])
		#self.player_masks[-player] = b64(self.player_masks[-player])

	def calc_flip(self, pos, player):
		"""return flip stones of enemy by bitboard when I place stone at pos.
		:param pos: 0~63
		:param own: bitboard (0=top left, 63=bottom right)
		:param enemy: bitboard
		:return: flip stones of enemy when I place stone at pos.
		"""
		assert 0 <= pos <= 63, "pos={0}".format(pos)
		own_mask = self.player_masks[player]
		opp_mask = self.player_masks[-player] 

		f1 = self._calc_flip_half(pos, own_mask, opp_mask)
		f2 = self._calc_flip_half(63 - pos, self.rotate180(own_mask), self.rotate180(opp_mask))
		return f1 | self.rotate180(f2)

	def _calc_flip_half(self, pos, own, enemy):
	    el = [enemy, enemy & 0x7e7e7e7e7e7e7e7e, enemy & 0x7e7e7e7e7e7e7e7e, enemy & 0x7e7e7e7e7e7e7e7e]
	    masks = [0x0101010101010100, 0x00000000000000fe, 0x0002040810204080, 0x8040201008040200]
	    masks = [self.b64(m << pos) for m in masks]
	    flipped = 0
	    for e, mask in zip(el, masks):
	        outflank = mask & ((e | ~mask) + 1) & own
	        flipped |= (outflank - (outflank != 0)) & mask
	    return flipped

	def flip_diag_a1h8(self, x):
		k1 = 0x5500550055005500
		k2 = 0x3333000033330000
		k4 = 0x0f0f0f0f00000000
		t = k4 & (x ^ self.b64(x << 28))
		x ^= t ^ (t >> 28)
		t = k2 & (x ^ self.b64(x << 14))
		x ^= t ^ (t >> 14)
		t = k1 & (x ^ self.b64(x << 7))
		x ^= t ^ (t >> 7)
		return x

	def flip_vertical(self, x):
		k1 = 0x00FF00FF00FF00FF
		k2 = 0x0000FFFF0000FFFF
		x = ((x >> 8) & k1) | ((x & k1) << 8)
		x = ((x >> 16) & k2) | ((x & k2) << 16)
		x = (x >> 32) | self.b64(x << 32)
		return x

	def rotate90(self, x):
		return self.flip_diag_a1h8(self.flip_vertical(x))

	def rotate180(self, x):
		return self.rotate90(self.rotate90(x))

	def compute_scores(self):
		black_score = bin(self.player_masks[OthelloConstants.BLACK]).count('1')
		white_score = bin(self.player_masks[OthelloConstants.WHITE]).count('1')
		remainder = 64 - (black_score + white_score)

		if black_score > white_score:
			black_score += remainder
		elif white_score > black_score:
			white_score += remainder
		else:
			black_score += remainder//2
			white_score += remainder//2

		return { OthelloConstants.BLACK: black_score, OthelloConstants.WHITE : white_score }

	def print_board(self, label_moves=False, player=None):
		board = OthelloBoard.convert_masks_to_board(self.player_masks)
		legal_moves = None if player is None else self.get_legal_moves(player)
		print('   0 1 2 3 4 5 6 7 ')
		print('  +-+-+-+-+-+-+-+-+')
		for row in range(OthelloConstants.BOARD_SIZE):
			print('{0} |'.format(row), end='')
			for col in range(OthelloConstants.BOARD_SIZE):
				if board[row][col] == OthelloConstants.BLACK:
					char = u"\u25CF" #'B'
				elif board[row][col] == OthelloConstants.WHITE:
					char = u"\u25CB" #'W'
				elif label_moves and (row, col) in legal_moves:
					char = chr(97+legal_moves.index((row, col)))
				else:
					char = ' '
				print(char, end='|')
			print('\n  +-+-+-+-+-+-+-+-+')

	def print_mask(self, mask):
		bits = '{0:064b}'.format(mask)
		for i in range(8):
			print(bits[8*i:8*i+8])
		print("")

	def b64(self, x):
		return x & 0xFFFFFFFFFFFFFFFF


if __name__ == '__main__':
	state = OthelloBoard(OthelloConstants.INITIAL_BOARD)
	x = np.reshape(np.arange(3), (3,1))
	y = np.ones((3,3))
	print(x)
	print(y)
	print(x*y)

