from copy import deepcopy
from Othello import *

import math
import numpy as np
import random
import time

class AlphaNode:
	LAMBDA = 0.5

	def __init__(self, state, player, prior=0.0, parent=None, parent_action=None):
		self.state = state
		self.player = player
		self.children = []

		self.player_moves = {
			OthelloConstants.BLACK : self.state.get_legal_moves(OthelloConstants.BLACK),
			OthelloConstants.WHITE : self.state.get_legal_moves(OthelloConstants.WHITE)
		}
		self.legal_moves = self.player_moves[self.player]
		if len(self.legal_moves) == 0:
			self.player *= -1
			self.legal_moves = self.player_moves[self.player]
		
		self.is_terminal = len(self.player_moves[OthelloConstants.BLACK]) == 0 and len(self.player_moves[OthelloConstants.WHITE]) == 0

		self.parent = parent
		self.parent_action = parent_action		

		self.prior_probability = prior
		self.visit_count = 0.0
		self.W_network, self.W_rollout = 0.0, 0.0

	def get_Q_value(self):
		return ((1-AlphaNode.LAMBDA)*self.W_network + AlphaNode.LAMBDA*self.W_rollout) / self.visit_count

	def update_Q_value(self, rollout_score, network_score):
		self.visit_count += 1
		self.W_network += network_score
		self.W_rollout += rollout_score
	
	def get_best_child(self, key = None):
		if key is None:
			key = lambda x: x.get_Q_value()
		
		return max(self.children, key=key)


class AlphaPlayer:
	EXPLORATION_CONSTANT = 5 * math.sqrt(2) #100 * math.sqrt(2)

	def __init__(self, policy_network, value_network, num_simulations=1000):
		self.player = None
		self.root = None
		self.num_simulations = num_simulations
		self.policy_network = policy_network
		self.value_network = value_network

	def set_color(self, color):
		self.player = color

	def reset(self):
		self.root = None

	def select_move(self, state, legal_moves):
		if self.root is None or self.root.state != state:
			self.root = AlphaNode(state, self.player)

		for _ in range(self.num_simulations):
			self.simulate()
		
		best_child = self.root.get_best_child(lambda node: node.visit_count)
		best_move = best_child.parent_action
		best_utility = best_child.get_Q_value()

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
		constant = AlphaPlayer.EXPLORATION_CONSTANT * node.prior_probability
		return node.get_Q_value() + constant * math.sqrt(math.log(node.parent.visit_count) / node.visit_count)

	def normalize(self, vector):
		total = sum(vector.values())
		return { key: value/total for key, value in vector.items() }

	def get_move_priors(self, node):
		network_input = node.state.get_feature_matrix(node.player, node.legal_moves)
		prior_probabilities = np.reshape(self.policy_network.predict(network_input), (8,8))
		priors = { move : prior_probabilities[move[0], move[1]] for move in node.legal_moves }
		return self.normalize(priors)

	def expand(self, node):
		if node.is_terminal: return

		next_player = -1 * node.player
		
		prior_probabilities = self.get_move_priors(node)
		for move in node.legal_moves:
			next_state = deepcopy(node.state)
			next_state.make_move(move, node.player)
			child_node = AlphaNode(
				next_state,
				next_player,
				prior = prior_probabilities[move],
				parent = node,
				parent_action = move
			)
			node.children.append(child_node)

	def get_value_network_scores(self, node):
		if self.value_network is None:
			return {-1:0, 1:0}
		black_input = node.state.get_feature_matrix(OthelloConstants.BLACK, node.player_moves[OthelloConstants.BLACK])
		black_score = float(self.value_network.predict(black_input))

		white_input = node.state.get_feature_matrix(OthelloConstants.WHITE, node.player_moves[OthelloConstants.WHITE])
		white_score = float(self.value_network.predict(white_input))

		return { OthelloConstants.BLACK: black_score, OthelloConstants.WHITE: white_score }

	def backpropogate(self, node, rollout_scores):
		network_scores = self.get_value_network_scores(node)
		
		current_node = node
		while current_node.parent != None:
			current_node.update_Q_value(
				rollout_scores[current_node.parent.player],
				network_scores[current_node.parent.player]
			)
			current_node = current_node.parent
		self.root.visit_count += 1

	def depth_charge(self, node):
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

	#player = MCTSPlayer(OthelloBoard(INITIAL_BOARD), OthelloConstants.BLACK)



