"""
Training Script for Snake DQN
"""

import numpy as np
import matplotlib.pyplot as plt
from snake_game import SnakeGame
from dqn_agent import DQNAgent

def train(episodes=500):
    env = SnakeGame(grid_size=10)
    agent = DQNAgent()

    scores = []
    avg_scores = []

    print("Training Snake DQN...")
    print("-" * 60)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward

        scores.append(env.score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        if (episode + 1) % 10 == 0:
            agent.update_target_model()

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}: Score={env.score}, Avg={avg_score:.2f}, Epsilon={agent.epsilon:.3f}")

    print("-" * 60)
    print("Training complete!")

    # Save model
    agent.save('snake_dqn.pth')
    print("Model saved to 'snake_dqn.pth'")

    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.6)
    plt.plot(avg_scores, linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Scores')
    plt.legend(['Score', 'Avg (100)'])
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(avg_scores)
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title('Average Score Progress')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('snake_training.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'snake_training.png'")

if __name__ == "__main__":
    train(episodes=500)
