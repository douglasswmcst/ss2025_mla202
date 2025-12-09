import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from environment import DynamicPricingEnv
from agent import DQNAgent

def train_agent(episodes=500, max_steps=100):
    """Train DQN agent on pricing environment"""

    # Initialize environment and agent
    env = DynamicPricingEnv(max_steps=max_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    # Training metrics
    episode_rewards = []
    episode_revenues = []
    losses = []

    print("Starting training...")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print("-" * 50)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []

        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train agent
            loss = agent.train()
            if loss > 0:
                episode_loss.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_revenues.append(info['total_revenue'])
        if episode_loss:
            losses.append(np.mean(episode_loss))

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_revenue = np.mean(episode_revenues[-50:])
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Revenue: ${avg_revenue:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            if losses:
                print(f"  Avg Loss: {np.mean(losses[-50:]):.4f}")
            print("-" * 50)

    # Save trained model
    agent.save('trained_pricing_agent.pth')
    print("\nTraining complete! Model saved as 'trained_pricing_agent.pth'")

    # Plot training curves
    plot_training_results(episode_rewards, episode_revenues, losses)

    return agent, episode_rewards, episode_revenues

def plot_training_results(rewards, revenues, losses):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Rewards
    axes[0].plot(rewards, alpha=0.6)
    axes[0].plot(moving_average(rewards, 50), linewidth=2)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True)

    # Revenues
    axes[1].plot(revenues, alpha=0.6)
    axes[1].plot(moving_average(revenues, 50), linewidth=2)
    axes[1].set_title('Episode Revenues')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Revenue ($)')
    axes[1].grid(True)

    # Losses
    if losses:
        axes[2].plot(losses, alpha=0.6)
        axes[2].plot(moving_average(losses, 50), linewidth=2)
        axes[2].set_title('Training Loss')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("Training plots saved as 'training_results.png'")
    plt.close()

def moving_average(data, window):
    """Calculate moving average"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    train_agent(episodes=500, max_steps=100)
