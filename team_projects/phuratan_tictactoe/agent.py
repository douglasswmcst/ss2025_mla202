"""
Q-Learning Agent for Ultimate Tic-Tac-Toe
"""

import numpy as np
import pickle
from collections import defaultdict

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995

    def get_action(self, state, legal_moves, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return legal_moves[np.random.randint(len(legal_moves))]

        # Greedy: choose action with highest Q-value
        q_values = {move: self.q_table[state][move] for move in legal_moves}
        max_q = max(q_values.values()) if q_values else 0
        best_moves = [move for move, q in q_values.items() if q == max_q]
        return best_moves[np.random.randint(len(best_moves))]

    def update(self, state, action, reward, next_state, next_legal_moves, done):
        """Q-learning update"""
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            # Max Q-value of next state
            next_q_values = [self.q_table[next_state][a] for a in next_legal_moves]
            max_next_q = max(next_q_values) if next_q_values else 0
            target_q = reward + self.gamma * max_next_q

        # Update Q-value
        self.q_table[state][action] += self.alpha * (target_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(f))

class RandomAgent:
    """Baseline random agent"""
    def get_action(self, state, legal_moves, training=False):
        return legal_moves[np.random.randint(len(legal_moves))]
