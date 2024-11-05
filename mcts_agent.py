import math
import random
import numpy as np
from mcts_node import MCTSNode
from tqdm import tqdm

class MCTSAgent:
    def __init__(self, model_path="cnn_model.keras", simulations=100000, exploration_weight=math.sqrt(2)):
        self.simulations = simulations
        self.exploration_weight = exploration_weight

    def choose_move(self, state):
        root = MCTSNode(state)
        with tqdm(total=self.simulations, desc="Running Simulations", unit="sim") as pbar:
            for _ in range(self.simulations):
                node = self.selection(root)
                result = self.simulation(node.state.clone())
                self.backpropagation(node, result)
                pbar.update(1)
        for i in root.children:
            print(i.state.last_move, i.visits, i.wins)
        best_move = root.best_child(self.exploration_weight)
        return best_move.state.last_move

    def selection(self, node):
        while node.is_fully_expanded() and node.children is not None:
            node = node.select_child(self.exploration_weight)
        if not node.is_fully_expanded():
            node = self.expand(node)
        return node

    def expand(self, node):
        moves = node.state.check_3_4(-node.state.turn, node.state.last_move[0], node.state.last_move[1])
        if len(moves) == 0:
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
            moves = state.get_legal_moves()

            winning_move = self.find_winning_move(state, moves, original_player)
            if winning_move is not None:
                state = state.make_move(winning_move)
                continue
            move = self.random_move(state)
            state = state.clone().make_move(move)
        return 1 if state.winner == original_player else -1 if state.winner == -original_player else 0

    def find_winning_move(self, state, moves, player):
        """Check if there's a winning move for the current player."""
        for move in moves:
            state_new = state.clone()
            new_state = state_new.make_move(move)
            if new_state.is_terminal() and new_state.winner == player:
                return move
        return None


    def backpropagation(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent

    def random_move(self, state):
        moves = state.get_move()
        return random.choice(moves)
