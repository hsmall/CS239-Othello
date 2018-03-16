from ConvolutionalNeuralNetwork import *
from ConvolutionalNeuralNetwork2 import *
from Othello import *

import numpy as np
import random

from players.AlphaPlayer import *
from players.HeuristicPlayer import *
from players.HumanPlayer import *
from players.MCTSPlayer import *
from players.PolicyPlayer import *
from players.RandomPlayer import *
from players.ValuePlayer import *

class OthelloSimulator:
	def __init__(self, player1, player2):
		self.players = {OthelloConstants.BLACK : player1, OthelloConstants.WHITE : player2}
		player1.set_color(OthelloConstants.BLACK)
		player2.set_color(OthelloConstants.WHITE)

	def reset(self):
		player1.reset()
		player2.reset()

	def simulate_game(self, record = False, verbose = False):
		play_by_play = []

		state = OthelloBoard(OthelloConstants.INITIAL_BOARD)
		player = OthelloConstants.BLACK

		if verbose:
			state.print_board()

		while True:
			legal_moves = OthelloSimulator.get_legal_moves_index(state)

			# Game Over
			if len(legal_moves[player]) == 0 and len(legal_moves[-player]) == 0: break

			if len(legal_moves[player]) > 0:
				move, utility = self.players[player].select_move(state, legal_moves[player])
				if record:
					play_by_play.append((
						state.get_feature_matrix(player, legal_moves[player]),
						player,
						OthelloSimulator.to_one_hot_matrix(move)
					))
				state.make_move(move, player)
				self.players[-player].receive_opponent_move(move)

				if verbose:
					label_moves = isinstance(self.players[-player], HumanPlayer)
					state.print_board(label_moves, -player)
					print('Player: {0}, Move: {1}, Expected Utility: {2}\n'.format(
						'BLACK' if player == OthelloConstants.BLACK else 'WHITE',
						move,
						utility
					))

			player *= -1 # Change turns

		return state.compute_scores(), (play_by_play if record else None)

	@staticmethod
	def get_legal_moves_index(state):
		return {
			OthelloConstants.BLACK: state.get_legal_moves(OthelloConstants.BLACK),
			OthelloConstants.WHITE: state.get_legal_moves(OthelloConstants.WHITE)
		}

	@staticmethod			
	def to_one_hot_matrix(move):
		matrix = np.zeros((64,))
		matrix[move[0]*OthelloConstants.BOARD_SIZE+move[1]] = 1
		return matrix

	@staticmethod
	def simulate_moves_with_player(state, initial_color, player, num_moves):
		color = initial_color

		count = 0
		while count < num_moves:
			legal_moves = OthelloSimulator.get_legal_moves_index(state)

			if len(legal_moves[color]) == 0 and len(legal_moves[-color]) == 0:
				return state, color, True

			if len(legal_moves[color]) > 0:
				player.set_color(color)
				move, _ = player.select_move(state, legal_moves[color])
				state.make_move(move, color)
				count += 1
			
			color *= -1

		return state, color, False

	@staticmethod
	def generate_state(sl_player, rl_player):
		state = OthelloBoard(OthelloConstants.INITIAL_BOARD)
		color = OthelloConstants.BLACK

		random_player = RandomPlayer()
		U = random.randint(0, 59)

		state, color, game_over = OthelloSimulator.simulate_moves_with_player(state, color, sl_player, U)
		if game_over: raise ValueError('game terminated too early.')
		
		state, color, _ = OthelloSimulator.simulate_moves_with_player(state, color, random_player, 1)
		result_state, result_color = state.get_feature_matrix(color), color

		state, _, _ = OthelloSimulator.simulate_moves_with_player(state, color, rl_player, 60)

		scores = state.compute_scores()
		value = np.sign(scores[result_color] - scores[-result_color])
		return result_state, value


if __name__ == '__main__':
	np.set_printoptions(suppress=True, precision=4)
	initial_state = OthelloBoard(OthelloConstants.INITIAL_BOARD)

	random_player = RandomPlayer()
	heuristic_player = HeuristicPlayer(lambda state, player: np.sum(state.get_board())*player, depth=1)

	policy_network = ConvolutionalNeuralNetwork(
		input_depth = 3,
		num_layers = 6,
		num_filters = [64, 64, 128, 128, 256, 256],
		dropout_rate = 0.0,
		regularization = 0,
	)
	policy_network.load('models/SL_POLICY_NETWORK/SL_POLICY_NETWORK')
	policy_player = PolicyPlayer(policy_network, greedy=False)

	value_network = ConvolutionalNeuralNetwork2(
		input_depth = 3,
		num_layers = 6,
		num_filters = [64, 64, 128, 128, 256, 256],
		dropout_rate = 0.0,
	)
	value_network.load('models/VALUE_NETWORK/VALUE_NETWORK')
	value_player = ValuePlayer(value_network)

	mcts_player = MCTSPlayer(num_simulations = 1000)

	alpha_player = AlphaPlayer(
		policy_network,
		value_network,
		num_simulations = 1000
	)

	player1 = alpha_player
	player2 = HumanPlayer()
	simulator = OthelloSimulator(player1, player2)
	
	num_games = 1
	win_count = 0
	black_scores = []

	start_time = int(round(time.time() * 1000))
	print('Running {0} Game Simulations...'.format(num_games))
	for i in range(num_games):
		scores, play_by_play = simulator.simulate_game(record=False, verbose=True)
		print(scores)
		black_scores.append(scores[OthelloConstants.BLACK])
		if scores[OthelloConstants.BLACK] > scores[OthelloConstants.WHITE]:
			win_count += 1
		simulator.reset()
	end_time = int(round(time.time() * 1000))
	
	print('Finished in {0} seconds.'.format((end_time-start_time) / 1000.0))

	black_score = np.round(np.mean(black_scores))
	white_score = 64 - black_score
	print('BLACK won {0}/{1} matches. Average Score: {2}-{3}'.format(win_count, num_games, black_score, white_score))
	

