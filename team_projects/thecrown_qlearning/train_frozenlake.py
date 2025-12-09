"""
Train Q-Learning on FrozenLake
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from qlearning_agent import QLearningAgent

def train_frozenlake(episodes=5000):
    env = gym.make('FrozenLake-v1', is_slippery=True)
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0
    )

    rewards_per_episode = []
    success_rate = []

    print("Training Q-Learning on FrozenLake...")
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
        
        # Calculate success rate
        if (episode + 1) % 100 == 0:
            recent_success = sum(rewards_per_episode[-100:])
            success_rate.append(recent_success / 100)
            agent.decay_epsilon()
            print(f"Episode {episode+1}: Success Rate={recent_success}%, Epsilon={agent.epsilon:.3f}")

    env.close()

    # Save agent
    agent.save('frozenlake_qtable.npy')
    print("Q-table saved to 'frozenlake_qtable.npy'")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot([i*100 for i in range(len(success_rate))], success_rate)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('FrozenLake Q-Learning Training')
    plt.grid(True, alpha=0.3)
    plt.savefig('frozenlake_training.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'frozenlake_training.png'")

if __name__ == "__main__":
    train_frozenlake(episodes=5000)
