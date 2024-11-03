from caro_board import State, Gomoku
import numpy as np
import tensorflow as tf
import random
import pickle

class CNN:
    def __init__(self, board_size=(10, 10), num_simulations=10000):
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.data = []

    def create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.board_size, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')  # Output value between -1 and 1 for outcome
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
    
    def generate_data(self):
        print("Generating data...")
        for _ in range(self.num_simulations):
            game = Gomoku(size = 10, game_mode = 0)
            game_data = game.gamemode_0()
            self.data.extend(game_data)

    def preprocess_data(self):
        X = np.array([state for state, outcome in self.data]).reshape(-1, *self.board_size, 1).astype(np.float32)
        y = np.array([outcome for state, outcome in self.data]).astype(np.float32)
        return X, y
    
    def train_model(self, epochs=100, batch_size=32):
        X, y = self.preprocess_data()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        print("Model training complete.")

    def save_model(self, filename="cnn_model.keras"):
        self.model.save(filename)
        print(f"Model saved as {filename}")

    def save_data(self, filename="game_data.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Data saved as {filename}")

if __name__ == "__main__":
    cnn_agent = CNN(board_size=(10, 10), num_simulations=10000)
    cnn_agent.create_model()      # Create CNN model
    cnn_agent.generate_data()         # Generate training data
    cnn_agent.save_data()             # Save generated data for reuse
    cnn_agent.train_model()           # Train the CNN model
    cnn_agent.save_model() 