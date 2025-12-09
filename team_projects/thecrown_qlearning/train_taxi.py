"""
Train Q-Learning on Taxi
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from qlearning_agent import QLearningAgent

def train_taxi(episodes=5000):
    env = gym.make('Taxi-v3')
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0
    )

    rewards_per_episode = []
    avg_rewards = []

    print("Training Q-Learning on Taxi...")
    print("-" * 60)

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, training=True)
            result = env.step(action)
            
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        
        # Calculate average reward
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            avg_rewards.append(avg_reward)
            agent.decay_epsilon()
            print(f"Episode {episode+1}: Avg Reward={avg_reward:.2f}, Epsilon={agent.epsilon:.3f}")

    env.close()

    # Save agent
    agent.save('taxi_qtable.npy')
    print("Q-table saved to 'taxi_qtable.npy'")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot([i*100 for i in range(len(avg_rewards))], avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Taxi Q-Learning Training')
    plt.grid(True, alpha=0.3)
    plt.savefig('taxi_training.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'taxi_training.png'")

if __name__ == "__main__":
    train_taxi(episodes=5000)
