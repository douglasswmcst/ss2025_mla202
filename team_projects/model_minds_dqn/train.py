import gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import time

def train_dqn(
    env_name='CartPole-v1',
    episodes=500,
    max_steps=500,
    target_reward=475
):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Target reward: {target_reward}")
    print("-" * 60)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=100000,
        batch_size=64,
        target_update_freq=10
    )

    episode_rewards = []
    episode_lengths = []
    losses = []

    start_time = time.time()

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        episode_reward = 0
        episode_loss = []

        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            result = env.step(action)

            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            agent.store_transition(state, action, reward, next_state, done)

            loss = agent.train()
            if loss > 0:
                episode_loss.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.end_episode()

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        if episode_loss:
            losses.append(np.mean(episode_loss))

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Reward: {episode_reward:.1f} | Avg (100): {avg_reward:.1f}")
            print(f"  Steps: {step + 1} | Avg (100): {avg_length:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            if losses:
                print(f"  Loss: {losses[-1]:.4f}")
            print(f"  Time: {time.time() - start_time:.1f}s")
            print("-" * 60)

        if len(episode_rewards) >= 100:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            if avg_reward_100 >= target_reward:
                print(f"\nEnvironment solved in {episode + 1} episodes!")
                print(f"Average reward (100 episodes): {avg_reward_100:.2f}")
                break

    agent.save('trained_dqn_model.pth')
    plot_training_results(episode_rewards, episode_lengths, losses)
    env.close()

    return agent, episode_rewards

def plot_training_results(rewards, lengths, losses):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(rewards, alpha=0.6, label='Episode Reward')
    if len(rewards) >= 100:
        moving_avg = [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
        axes[0, 0].plot(moving_avg, linewidth=2, label='Moving Average (100)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(lengths, alpha=0.6, label='Episode Length')
    if len(lengths) >= 100:
        moving_avg = [np.mean(lengths[max(0, i-99):i+1]) for i in range(len(lengths))]
        axes[0, 1].plot(moving_avg, linewidth=2, label='Moving Average (100)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if losses:
        axes[1, 0].plot(losses, alpha=0.6, label='Training Loss')
        if len(losses) >= 50:
            moving_avg = [np.mean(losses[max(0, i-49):i+1]) for i in range(len(losses))]
            axes[1, 0].plot(moving_avg, linewidth=2, label='Moving Average (50)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(rewards, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(rewards), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(rewards):.1f}')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dqn_training_results.png', dpi=300, bbox_inches='tight')
    print("\nTraining plots saved as 'dqn_training_results.png'")
    plt.close()

if __name__ == "__main__":
    train_dqn(env_name='CartPole-v1', episodes=500, max_steps=500, target_reward=475)
