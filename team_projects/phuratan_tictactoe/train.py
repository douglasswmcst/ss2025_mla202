"""
Training Script for Ultimate Tic-Tac-Toe Q-Learning
"""

import numpy as np
import matplotlib.pyplot as plt
from game import UltimateTicTacToe
from agent import QLearningAgent, RandomAgent

def train(episodes=10000):
    game = UltimateTicTacToe()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=1.0)

    wins = []
    draws = []
    losses = []

    print("Training via self-play...")
    print(f"Episodes: {episodes}")
    print("-" * 60)

    for episode in range(episodes):
        state = game.reset()
        done = False

        while not done:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            action = agent.get_action(state, legal_moves, training=True)
            next_state, reward, done = game.make_move(*action)

            if not done:
                next_legal_moves = game.get_legal_moves()
                agent.update(state, action, reward, next_state, next_legal_moves, done)
            else:
                agent.update(state, action, reward, next_state, [], done)

            state = next_state

        # Track results
        if game.winner == 1:
            wins.append(1)
        elif game.winner == 0:
            draws.append(1)
        else:
            losses.append(1)

        agent.decay_epsilon()

        if (episode + 1) % 1000 == 0:
            recent = 1000
            w = sum(wins[-recent:])
            d = sum(draws[-recent:])
            l = sum(losses[-recent:])
            print(f"Episode {episode + 1}: Wins={w}, Draws={d}, Losses={l}, Epsilon={agent.epsilon:.3f}")

    print("-" * 60)
    print("Training complete!")

    # Save agent
    agent.save('ultimate_ttt_agent.pkl')
    print("Agent saved to 'ultimate_ttt_agent.pkl'")

    # Plot results
    window = 100
    win_rate = [sum(wins[max(0,i-window):i+1])/min(i+1, window) for i in range(len(wins))]

    plt.figure(figsize=(10, 6))
    plt.plot(win_rate, label='Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.title('Ultimate Tic-Tac-Toe Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ultimate_ttt_training.png', dpi=300, bbox_inches='tight')
    print("Training plot saved to 'ultimate_ttt_training.png'")

if __name__ == "__main__":
    train(episodes=10000)
