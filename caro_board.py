from tkinter import *
from tkinter import messagebox
from mcts_agent import MCTSAgent
import copy
import random
import numpy as np

class State:
    def __init__(self, board, turn, win_length = 5):
        self.board = board
        self.turn = turn
        self.winner = None
        self.win_length = win_length
        self.last_move = None
        self.Board_Size = len(board)
    
    def check_have_valid_move(self):
        return 0 in [cell for row in self.board for cell in row]
    
    def to_model_input(self):
        return np.array(self.board, dtype=np.float32)

    def get_legal_moves(self):
        return [(x, y) for x in range(1, len(self.board) + 1) for y in range(1, len(self.board) + 1) if self.board[y - 1][x - 1] == 0]
    
    def check_around(self, x, y):
        min_x, max_x = max(0, x - 2), min(self.Board_Size - 1, x)
        min_y, max_y = max(0, y - 2), min(self.Board_Size - 1, y)
        for i in range(min_y, max_y + 1):
            for j in range(min_x, max_x + 1):
                if self.board[i][j] != 0:
                    return True
        return False
    def get_move(self):
        moves = self.get_legal_moves()
        ans = []
        for move in moves:
            #check around the move
            x,y = move
            if self.board[y - 1][x - 1] == 0 and self.check_around(x, y):
                ans.append(move)
        return ans
                
    def check_line(self, line, pattern):
        for i in range(len(line) - len(pattern) + 1):
            if line[i:i+len(pattern)] == pattern:
                return True

    def win_check(self, player):
        x,y = self.last_move
        column = [self.board[y - 1][i] for i in range(max(0, x - self.win_length), min(self.Board_Size, x + self.win_length - 1))]
        row = [self.board[i][x - 1] for i in range(max(0, y - self.win_length), min(self.Board_Size, y + self.win_length - 1))]
        i = min(self.win_length - 1, x - 1, y - 1)
        j = min(self.win_length , self.Board_Size - x + 1, self.Board_Size - y + 1)
        diagonal1 = [self.board[y - 1 + k][x - 1 + k] for k in range(-i, j)]
        i = min(self.win_length - 1, self.Board_Size - x, y - 1)
        j = min(self.win_length, x , self.Board_Size - y + 1)
        diagonal2 = [self.board[y + k - 1][x - k - 1] for k in range(-i, j)]
        patterns_win = [player] * self.win_length
        for pattern in [row, column, diagonal1, diagonal2]:
            if self.check_line(pattern, patterns_win):
                return True
        return False
    
    def location_validation(self, X, Y):
        if X is None or Y is None:
            return False
        return self.board[Y - 1][X - 1] == 0

    def is_terminal(self):
        return self.winner is not None or self.check_have_valid_move() is False
    
    def make_move(self, move):
        X, Y = move
        self.last_move = (X, Y)
        self.board[Y - 1][X - 1] = self.turn
        self.winner = self.turn if self.win_check(self.turn) else None
        self.turn = -self.turn
        return self.clone()

    def clone(self):
        clone_state = copy.deepcopy(self)
        clone_state.last_move = self.last_move
        return clone_state

class Gomoku:
    def __init__(self,size = 10, game_mode = 2):
        # Initialize the interface and canvas
        self.myInterface = Tk()
        self.myInterface.title("Gomoku Game")
        self.s = Canvas(self.myInterface, width=800, height=800, background="#b69b4c")
        self.s.pack()

        # Board and Game Configuration
        self.Board_Size = size
        self.Frame_Gap = 35
        self.width = 800
        self.height = 800
        self.Chess_Radius = 20  # Adjusted for visibility
        self.win_length = 5 if self.Board_Size >= 5 else 3


        # Game State
        board = [[0] * self.Board_Size for _ in range(self.Board_Size)]
        self.state = State(board, 1, self.win_length)

        # Board Arrays and Coordinates
        self.Click_Cord = [None, None]
        self.Game_CordX, self.Game_CordY, self.Actual_CordX1, self.Actual_CordY1, self.Actual_CordX2, self.Actual_CordY2 = ([] for _ in range(6))

        if game_mode == 0:
            pass

        if game_mode == 1:
            self.create_board()
            self.s.bind("<Button-1>", self.mouse_click)
            self.Turn_Text = self.score_board()
            self.agent = MCTSAgent(simulations=300)
            self.gamemode_1()

        if game_mode == 2:
            self.create_board()
            self.s.bind("<Button-1>", self.mouse_click)
            self.Turn_Text = self.score_board()
            self.gamemode_2()

    def create_circle(self, x, y, radius, fill="", outline="black", width=1):
        self.s.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline=outline, width=width)

    def mouse_click(self, event):
        X_click, Y_click = event.x, event.y
        self.Click_Cord = self.piece_location(X_click, Y_click)

    def piece_location(self, X_click, Y_click):
        X = Y = None
        for i in range(len(self.Actual_CordX1)):
            if self.Actual_CordX1[i] < X_click < self.Actual_CordX2[i]:
                X = self.Game_CordX[i]
            if self.Actual_CordY1[i] < Y_click < self.Actual_CordY2[i]:
                Y = self.Game_CordY[i]
        return X, Y

    def score_board(self):
        if self.state.winner is None:
            text = f"Turn = {'Black' if self.state.turn == -1 else 'White'}"
        else:
            text = f"{'Black' if self.state.winner == -1 else 'White'} WINS!"
        return self.s.create_text(self.width / 2, self.height - self.Frame_Gap + 15, text=text, font="Helvetica 25 bold", fill="black")

    
    def create_board(self):
        # Initialize the board UI and coordinates
        Board_X1 = self.width / 10
        Board_Y1 = self.height / 10
        Board_GapX = (self.width - Board_X1 * 2) / (self.Board_Size - 1)
        Board_GapY = (self.height - Board_Y1 * 2) / (self.Board_Size - 1)
        for i in range(self.Board_Size):
            self.Game_CordX.append(i + 1)
            self.Game_CordY.append(i + 1)
            self.Actual_CordX1.append(Board_X1 + i * Board_GapX - self.Chess_Radius)
            self.Actual_CordY1.append(Board_Y1 + i * Board_GapY - self.Chess_Radius)
            self.Actual_CordX2.append(Board_X1 + i * Board_GapX + self.Chess_Radius)
            self.Actual_CordY2.append(Board_Y1 + i * Board_GapY + self.Chess_Radius)

        # Draw the board grid
        for i in range(self.Board_Size):
            self.s.create_line(Board_X1, Board_Y1 + i * Board_GapY, Board_X1 + Board_GapX * (self.Board_Size - 1), Board_Y1 + i * Board_GapY)
            self.s.create_line(Board_X1 + i * Board_GapX, Board_Y1, Board_X1 + i * Board_GapX, Board_Y1 + Board_GapY * (self.Board_Size - 1))
    
    def show_result(self):
        if self.state.winner is not None:
            winner_text = "Black wins!" if self.state.winner == -1 else "White wins!"
            messagebox.showinfo("Game Over", winner_text)
        else:
            messagebox.showinfo("Game Over", "Draw!")

    def Turn_Text(self):
        return f"Turn = {'Black' if self.state.turn == -1 else 'White'}"

    def gamemode_2(self):
        while not self.state.is_terminal():
            self.s.update()
            X, Y = self.Click_Cord
            if self.state.location_validation(X, Y):
                self.s.delete(self.Turn_Text)
                self.create_circle(self.width / 10 + (X - 1) * (self.width - self.width / 5) / (self.Board_Size - 1),
                                   self.height / 10 + (Y - 1) * (self.height - self.height / 5) / (self.Board_Size - 1),
                                   self.Chess_Radius, fill="black" if self.state.turn == -1 else "white")
                self.state.make_move((X, Y))
                self.Turn_Text = self.score_board()

        # Show the game over popup when a player wins
        self.show_result()

    def gamemode_1(self):
        while not self.state.is_terminal():
            self.s.update()
            if self.state.turn == -1:
                state = self.state.clone()
                move = None
                while move is None:
                    move = self.agent.choose_move(state)
                print(move, 1)
                X, Y = move
            else:
                X, Y = self.Click_Cord
            if self.state.location_validation(X, Y):
                self.s.delete(self.Turn_Text)
                self.create_circle(self.width / 10 + (X - 1) * (self.width - self.width / 5) / (self.Board_Size - 1),
                                   self.height / 10 + (Y - 1) * (self.height - self.height / 5) / (self.Board_Size - 1),
                                   self.Chess_Radius, fill="black" if self.state.turn == -1 else "white")
                self.state.make_move((X, Y))
                self.Turn_Text = self.score_board()
        self.show_result()
    
    def gamemode_0(self):
        states = []
        while not self.state.is_terminal():
            states.append(self.state.to_model_input())
            moves = self.state.get_move()
            if len(moves) == 0:
                moves = self.state.get_legal_moves()
            move = random.choice(moves)
            X, Y = move
            if self.state.location_validation(X, Y):
                self.state.make_move((X, Y))
        outcome = self.state.winner
        return [(state, outcome) for state in states]

