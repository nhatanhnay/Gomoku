import math
import random
import numpy as np
from mcts_node import MCTSNode
from tensorflow import keras
from tqdm import tqdm

class MCTSAgent:
    def __init__(self, model_path="cnn_model.keras", simulations=100000, exploration_weight=1.5):
        self.simulations = simulations
        self.exploration_weight = exploration_weight
        # Load the trained CNN model for move evaluation
        self.model = keras.models.load_model(model_path)

    def choose_move(self, state):
        root = MCTSNode(state)
        with tqdm(total=self.simulations, desc="Running Simulations", unit="sim") as pbar:
            for _ in range(self.simulations):
                node = self.selection(root)
                result = self.simulation(node.state.clone())
                self.backpropagation(node, result)
                pbar.update(1)
        best_move = root.best_child(exploration_weight=0)
        return best_move.state.last_move

    def selection(self, node):
        while node.is_fully_expanded() and node.children is not None:
            node = node.select_child(self.exploration_weight)
        if not node.is_fully_expanded():
            node = self.expand(node)
        return node

    def expand(self, node):
        moves = node.state.get_move()
        for move in moves:
            if move not in [child.state.last_move for child in node.children]:
                new_state = node.state.clone()
                new_state = new_state.make_move(move)
                child_node = MCTSNode(new_state, parent=node)
                node.children.append(child_node)
                return child_node
        return node

    def simulation(self, state):
        original_player = state.turn
        while not state.is_terminal():
            moves = state.get_move()

            # Check for a winning move for the current player
            winning_move = self.find_winning_move(state, moves, original_player)
            if winning_move:
                state = state.make_move(winning_move)
                continue

            # Use CNN to evaluate all legal moves and pick the best move
            best_move = self.get_best_move(state, moves)
            state = state.make_move(best_move)

        return 1 if state.winner == original_player else 0

    def find_winning_move(self, state, moves, player):
        """Check if there's a winning move for the current player."""
        for move in moves:
            new_state = state.make_move(move)
            if new_state.is_terminal() and new_state.winner == player:
                return move
        return None

    def get_best_move(self, state, moves):
        """Evaluate all legal moves and return the best one."""
        # Evaluate all legal moves at once using CNN
        scores = self.evaluate_moves(state, moves)
        # Choose the move with the highest score
        best_move = moves[np.argmax(scores)]
        return best_move

    def evaluate_moves(self, state, moves):
        """Evaluate a list of moves using the CNN model."""
        states = [state.make_move(move) for move in moves]
        model_inputs = np.array([s.to_model_input() for s in states])
        scores = self.model.predict(model_inputs, verbose=False).flatten()
        return scores

    def backpropagation(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent

    def random_move(self, state):
        moves = state.get_move()
        return random.choice(moves)
