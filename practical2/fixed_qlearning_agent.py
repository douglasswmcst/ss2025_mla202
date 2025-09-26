"""
Fixed Q-Learning Agent Implementation for FrozenLake
This version uses better hyperparameters for successful learning
"""

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

def choose_action(state, q_table, epsilon, env):
    """Choose action using epsilon-greedy strategy"""
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

def test_agent(q_table, num_episodes=1000):
    """Test the trained Q-learning agent"""
    env_test = gym.make("FrozenLake-v1", render_mode=None)

    wins = 0
    total_rewards = []

    for episode in range(num_episodes):
        state, info = env_test.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, info = env_test.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        if total_reward > 0:
            wins += 1

    env_test.close()
    success_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    return success_rate, avg_reward

def main():
    """Main Q-learning training with optimized hyperparameters"""
    env = gym.make("FrozenLake-v1", render_mode=None)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    print(f"Q-table shape: {q_table.shape}")

    # Optimized Q-learning hyperparameters
    learning_rate = 0.1      # Good balance for learning speed and stability
    discount_factor = 0.99   # High value for delayed rewards
    epsilon = 1.0            # Start with full exploration
    epsilon_decay = 0.99     # Faster decay to reduce exploration over time
    min_epsilon = 0.01       # Always maintain some exploration

    # Training parameters - more episodes for better learning
    num_episodes = 5000
    rewards_per_episode = []

    print("Starting Q-learning training (optimized hyperparameters)...")
    print(f"Training for {num_episodes} episodes with α={learning_rate}, γ={discount_factor}")

    successful_episodes = 0

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        steps = 0

        while not terminated and not truncated and steps < 200:  # Prevent infinite loops
            action = choose_action(state, q_table, epsilon, env)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Q-learning update rule
            current_q = q_table[state, action]

            if terminated:
                max_future_q = 0  # No future rewards if episode ended
            else:
                max_future_q = np.max(q_table[next_state])

            target_q = reward + discount_factor * max_future_q
            error = target_q - current_q
            new_q = current_q + learning_rate * error
            q_table[state, action] = new_q

            state = next_state
            total_reward += reward
            steps += 1

            if reward > 0:
                successful_episodes += 1

        rewards_per_episode.append(total_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            recent_success = sum(1 for r in rewards_per_episode[-100:] if r > 0)
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.3f}, Recent Success: {recent_success}%, ε: {epsilon:.3f}")

    print(f"Training completed! Total successful episodes: {successful_episodes}")

    # Test the trained agent
    print("\nTesting trained agent...")
    success_rate, avg_reward = test_agent(q_table)

    print(f"\n=== RESULTS ===")
    print(f"Q-Learning Agent Performance (1000 test episodes):")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Reward: {avg_reward:.4f}")

    # Compare to random baseline
    print(f"\nComparison to Random Agent (~6% success rate):")
    if success_rate > 0:
        improvement = success_rate / 0.06
        print(f"Improvement: {improvement:.1f}x better than random!")
    else:
        print("No improvement over random agent")

    # Performance requirement check
    meets_requirement = success_rate >= 0.7
    print(f"\nMeets >70% requirement: {meets_requirement}")
    if meets_requirement:
        print("✅ SUCCESS: Agent achieves the required performance!")
    else:
        print("❌ Need more training or parameter tuning")

    # Q-table analysis
    print(f"\n=== Q-TABLE ANALYSIS ===")
    print(f"Non-zero Q-values: {np.count_nonzero(q_table)}")
    print(f"State 0 (start) Q-values: {q_table[0]}")
    best_start_action = np.argmax(q_table[0])
    actions = ['LEFT', 'DOWN', 'RIGHT', 'UP']
    print(f"Best action from start: {actions[best_start_action]} (value: {q_table[0, best_start_action]:.3f})")

    # Show some high-value Q-values
    print(f"\nHighest Q-values learned:")
    flat_indices = np.argsort(q_table.ravel())[-10:]  # Top 10 Q-values
    for flat_idx in reversed(flat_indices):
        state, action = np.unravel_index(flat_idx, q_table.shape)
        if q_table[state, action] > 0.01:  # Only show meaningful values
            print(f"  Q({state}, {actions[action]}) = {q_table[state, action]:.3f}")

    env.close()
    return success_rate, q_table

if __name__ == "__main__":
    success_rate, q_table = main()
    print(f"\nFinal Success Rate: {success_rate:.1%}")