import math

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # Game state
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.visits = 0  # Number of visits
        self.wins = 0  # Number of wins for the player

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_move())

    def best_child(self, exploration_weight=1.4):
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.wins / child.visits + exploration_weight * math.sqrt(math.log(self.visits) / child.visits))

    def select_child(self, exploration_weight):
        best_score = -float('inf')
        best_child = None

        for child in self.children:
            ucb_score = (
                child.wins / child.visits
                + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            )
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        return best_child

    def update(self, result):
        self.visits += 1
        self.wins += result

