from copy import deepcopy
from Othello import *

import math
import numpy as np
import random
import time

class MCTSNode:
	def __init__(self, state, player, parent=None, parent_action=None):
		self.state = state
		self.player = player
		self.children = []

		player_moves = {
			OthelloConstants.BLACK : self.state.get_legal_moves(OthelloConstants.BLACK),
			OthelloConstants.WHITE : self.state.get_legal_moves(OthelloConstants.WHITE)
		}
		self.legal_moves = player_moves[self.player]
		self.is_terminal = len(player_moves[OthelloConstants.BLACK]) == 0 and len(player_moves[OthelloConstants.WHITE]) == 0

		self.parent = parent
		self.parent_action = parent_action		

		self.visit_count = 0
		self.utility = 0

	def update_utility(self, utility):
		self.visit_count += 1
		self.utility += (utility - self.utility) / self.visit_count

	def get_best_child(self, key = None):
		if key is None:
			key = lambda x: x.utility
		
		return max(self.children, key=key)


class MCTSPlayer:
	EXPLORATION_CONSTANT = 100 * math.sqrt(2)

	def __init__(self, num_simulations=1000):
		self.player = None
		self.root = None
		self.num_simulations = num_simulations

	def set_color(self, color):
		self.player = color

	def reset(self):
		self.root = None

	def select_move(self, state, legal_moves):
		if self.root is None or self.root.state != state:
			self.root = MCTSNode(state, self.player)

		for _ in range(self.num_simulations):
			self.simulate()
		
		best_child = self.root.get_best_child(lambda node: node.utility)
		best_move = best_child.parent_action
		best_utility = best_child.utility

		self.root = best_child
		self.root.parent = None

		return best_move, best_utility

	def receive_opponent_move(self, move):
		if self.root is None: return

		for child in self.root.children:
			if child.parent_action == move:
				self.root = child
				self.root.parent = None
				return

	def simulate(self):
		start_node = self.select(self.root)
		self.expand(start_node)
		self.backpropogate(start_node, self.depth_charge(start_node))

	def select(self, node):
		if node.visit_count == 0 or node.is_terminal:
			return node
		
		for child in node.children:
			if child.visit_count == 0:
				return child

		return self.select(node.get_best_child(self.select_fn))

	def select_fn(self, node):
		return node.utility + MCTSPlayer.EXPLORATION_CONSTANT * math.sqrt(math.log(node.parent.visit_count) / node.visit_count)

	def expand(self, node):
		if node.is_terminal: return

		next_player = -1 * node.player
		
		if len(node.legal_moves) == 0:
			child_node = MCTSNode(node.state, next_player, node, (None, None))
			node.children.append(child_node)

		for move in node.legal_moves:
			next_state = deepcopy(node.state)
			next_state.make_move(move, node.player)
			child_node = MCTSNode(next_state, next_player, node, move)
			node.children.append(child_node)

	def backpropogate(self, node, scores):
		current_node = node
		while current_node.parent != None:
			current_node.update_utility(scores[current_node.parent.player])
			current_node = current_node.parent
		self.root.visit_count += 1

	def depth_charge(self, node):
		t1, t2 = 0, 0
		state = deepcopy(node.state)
		player = node.player

		while True:
			legal_moves = state.get_legal_moves(player)
			if len(legal_moves) == 0:
				player *= -1
				
				legal_moves = state.get_legal_moves(player)
				if len(legal_moves) == 0:
					break

			state.make_move(random.choice(legal_moves), player)
			player *= -1

		scores = state.compute_scores()
		black_score = np.sign(scores[OthelloConstants.BLACK] - scores[OthelloConstants.WHITE])
		return { OthelloConstants.BLACK : black_score, OthelloConstants.WHITE : -black_score }

if __name__ == '__main__':
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

	mcts_player = MCTSPlayer(OthelloBoard(INITIAL_BOARD), OthelloConstants.BLACK)
	start_time = int(round(time.time() * 1000))
	t1, t2 = 0, 0
	for i in range(1000):
		dc = mcts_player.depth_charge(mcts_player.root)
		t1 += dc[0]
		t2 += dc[1]
		#mcts_player.simulate()
	end_time = int(round(time.time() * 1000))
	print('Finished in {0} seconds.'.format((end_time-start_time) / 1000.0))
	print('Legal Moves: {0} seconds.\n'.format(t1 / 1000.0))
	print('Make Moves: {0} seconds.\n'.format(t2 / 1000.0))

	exit()

	def print_mask(mask):
		bits = '{0:064b}'.format(mask)
		for i in range(8):
			print(bits[8*i:8*i+8])
		print("")

	random.seed(1234567)
	simulations_per_turn = 5000

	mcts_player = MCTSPlayer(OthelloBoard(INITIAL_BOARD), OthelloConstants.BLACK)

	state = OthelloBoard(INITIAL_BOARD)
	current_player = OthelloConstants.BLACK

	while True:
		black_moves = state.get_legal_moves(OthelloConstants.BLACK)
		white_moves = state.get_legal_moves(OthelloConstants.WHITE)

		if len(black_moves) == 0 and len(white_moves) == 0: break

		if current_player == OthelloConstants.BLACK:
			if len(black_moves) > 0:	
				for _ in range(simulations_per_turn):
					mcts_player.simulate()
			
			move, utility = mcts_player.select_move()
			if move is not (None, None):
				state.make_move(move, current_player)

		if current_player == OthelloConstants.WHITE:
			if len(white_moves) > 0:
				print(white_moves)
				move_index = int(input("Select Move: "))
				move = white_moves[move_index]
				#move = random.choice(white_moves)
				state.make_move(move, current_player)
			else:
				move = (None, None)

			mcts_player.receive_opponent_move(move)
		
		state.print_board()
		print('Black Score: {0}'.format(bin(state.player_masks[OthelloConstants.BLACK]).count('1')))
		print('Expected Utility: {0}'.format(utility))
		#print(np.array(state))
		#print("")
		current_player *= -1


