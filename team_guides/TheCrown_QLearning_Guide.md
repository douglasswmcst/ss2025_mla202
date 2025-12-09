# The Crown - Q-Learning Fundamentals
## Combined Term Paper & Mini-Project Implementation Guide

**Team Members**: 3 students
**Research Paper**: [Q-learning](https://arxiv.org/abs/2108.02827)
**Submission Deadline**: Day 6 (Saturday, 11:59 PM)

---

## Overview

Since your paper is on Q-learning fundamentals, implement and compare classic Q-learning across multiple environments (FrozenLake, Taxi, CliffWalking) to demonstrate the algorithm's versatility and core concepts.

---

## Part 1: Term Paper Report (10 marks - 1000-1500 words)

### Structure

**1. Introduction (2 marks)**
- What is Q-learning and why is it fundamental to RL?
- Temporal Difference learning background
- Off-policy vs on-policy learning
- Problem: Demonstrate Q-learning on multiple benchmark environments

**2. Methodology (3 marks)**

**Q-Learning Algorithm**:
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

where:
- α: learning rate (step size)
- γ: discount factor (future reward importance)
- r: immediate reward
- s,a: current state-action
- s',a': next state-action
```

**Key Concepts to Explain**:
- Q-table representation
- Epsilon-greedy exploration
- Convergence properties
- Hyperparameter effects (α, γ, ε)

**Environments**:
1. **FrozenLake**: Navigate 4×4 grid avoiding holes
2. **Taxi**: Pick up and drop off passengers
3. **CliffWalking**: Reach goal avoiding cliff edge

**3. Findings (3 marks)**
- Learning curves for each environment
- Convergence rates comparison
- Hyperparameter sensitivity analysis
- Q-learning vs SARSA comparison (if time permits)
- Success rates and optimal policies found

**4. Organization & References (2 marks)**

---

## Part 2: Implementation (15 marks)

### Quick Setup

```bash
mkdir qlearning_fundamentals
cd qlearning_fundamentals
python -m venv venv
source venv/bin/activate
pip install numpy gym matplotlib pandas seaborn
```

### File Structure
```
qlearning_fundamentals/
├── qlearning_agent.py   # Core Q-learning implementation
├── train_frozenlake.py  # FrozenLake experiments
├── train_taxi.py        # Taxi experiments
├── train_cliff.py       # CliffWalking experiments
├── compare.py           # Compare across environments
├── utils.py             # Plotting utilities
└── README.md
```

### Implementation: `qlearning_agent.py`

```python
import numpy as np
import pickle

class QLearningAgent:
    """
    Classic Q-Learning Agent

    Implements the tabular Q-learning algorithm for discrete
    state and action spaces.
    """

    def __init__(self, n_states, n_actions, learning_rate=0.1,
                 discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        """
        Args:
            n_states: Number of states
            n_actions: Number of actions
            learning_rate: α - learning rate
            discount_factor: γ - discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum epsilon value
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))

    def get_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            training: If True, use epsilon-greedy; else use greedy

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action based on Q-values
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning update rule

        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Current Q-value
        current_q = self.q_table[state, action]

        # Target Q-value
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q

        # Q-learning update
        self.q_table[state, action] += self.alpha * (target_q - current_q)

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save Q-table"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath):
        """Load Q-table"""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)

    def get_policy(self):
        """Return greedy policy: best action for each state"""
        return np.argmax(self.q_table, axis=1)

    def get_q_table(self):
        """Return Q-table"""
        return self.q_table.copy()
```

### Implementation: `train_frozenlake.py`

```python
import gym
import numpy as np
import matplotlib.pyplot as plt
from qlearning_agent import QLearningAgent

def train_frozenlake(episodes=10000, render_freq=1000):
    """Train Q-learning on FrozenLake-v1"""

    # Create environment (slippery by default)
    env = gym.make('FrozenLake-v1', is_slippery=True)

    # Create agent
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01
    )

    # Training metrics
    rewards_per_episode = []
    success_rate = []

    print(f"Training Q-Learning on FrozenLake-v1")
    print(f"State space: {env.observation_space.n}")
    print(f"Action space: {env.action_space.n}")
    print("-" * 60)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.get_action(state, training=True)

            # Take action
            next_state, reward, done, _ = env.step(action)

            # Update Q-table
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        # Decay epsilon
        agent.decay_epsilon()

        # Track metrics
        rewards_per_episode.append(total_reward)

        # Calculate success rate (moving window)
        if episode >= 100:
            recent_success = np.mean(rewards_per_episode[-100:])
            success_rate.append(recent_success)

        # Print progress
        if (episode + 1) % render_freq == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}: Avg Reward (100) = {avg_reward:.3f}, "
                  f"Epsilon = {agent.epsilon:.3f}")

    # Save agent
    agent.save('frozenlake_agent.pkl')
    print(f"\nFinal Success Rate: {np.mean(rewards_per_episode[-100:]):.3f}")

    # Plot results
    plot_results(rewards_per_episode, 'FrozenLake-v1', 'frozenlake_results.png')

    # Display learned policy
    display_policy(agent, env)

    env.close()
    return agent, rewards_per_episode

def plot_results(rewards, env_name, filename):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Raw rewards
    ax1.plot(rewards, alpha=0.3)
    # Moving average
    window = 100
    if len(rewards) >= window:
        moving_avg = [np.mean(rewards[max(0, i-window+1):i+1])
                     for i in range(len(rewards))]
        ax1.plot(moving_avg, linewidth=2, label=f'MA({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'{env_name} - Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Success rate over time
    if len(rewards) >= 100:
        success_rates = [np.mean(rewards[max(0, i-99):i+1])
                        for i in range(100, len(rewards))]
        ax2.plot(success_rates, linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate (100-episode window)')
        ax2.set_title(f'{env_name} - Learning Progress')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Results saved to {filename}")
    plt.close()

def display_policy(agent, env):
    """Display learned policy for FrozenLake"""
    policy = agent.get_policy()
    action_symbols = ['←', '↓', '→', '↑']

    print("\nLearned Policy:")
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            print(action_symbols[policy[state]], end='  ')
        print()

if __name__ == "__main__":
    train_frozenlake(episodes=10000)
```

### Implementation: `train_taxi.py`

```python
import gym
import numpy as np
import matplotlib.pyplot as plt
from qlearning_agent import QLearningAgent

def train_taxi(episodes=10000):
    """Train Q-learning on Taxi-v3"""

    env = gym.make('Taxi-v3')

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01
    )

    rewards_per_episode = []
    steps_per_episode = []

    print(f"Training Q-Learning on Taxi-v3")
    print(f"State space: {env.observation_space.n}")
    print(f"Action space: {env.action_space.n}")
    print("-" * 60)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            avg_steps = np.mean(steps_per_episode[-100:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, "
                  f"Avg Steps = {avg_steps:.1f}, Epsilon = {agent.epsilon:.3f}")

    agent.save('taxi_agent.pkl')

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Rewards
    window = 100
    moving_avg = [np.mean(rewards_per_episode[max(0, i-window+1):i+1])
                 for i in range(len(rewards_per_episode))]
    ax1.plot(moving_avg, linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward (100 episodes)')
    ax1.set_title('Taxi-v3 - Training Rewards')
    ax1.grid(True, alpha=0.3)

    # Steps
    moving_avg_steps = [np.mean(steps_per_episode[max(0, i-window+1):i+1])
                       for i in range(len(steps_per_episode))]
    ax2.plot(moving_avg_steps, linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Steps (100 episodes)')
    ax2.set_title('Taxi-v3 - Steps to Complete')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('taxi_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    env.close()
    return agent, rewards_per_episode

if __name__ == "__main__":
    train_taxi(episodes=10000)
```

### Implementation: `train_cliff.py`

```python
import gym
import numpy as np
import matplotlib.pyplot as plt
from qlearning_agent import QLearningAgent

def train_cliffwalking(episodes=500):
    """Train Q-learning on CliffWalking-v0"""

    env = gym.make('CliffWalking-v0')

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1,  # Lower epsilon for cliff
        epsilon_decay=0.999,
        epsilon_min=0.01
    )

    rewards_per_episode = []

    print(f"Training Q-Learning on CliffWalking-v0")
    print("-" * 60)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_per_episode[-50:])
            print(f"Episode {episode + 1}: Avg Reward (50) = {avg_reward:.2f}")

    agent.save('cliff_agent.pkl')

    # Plot
    plt.figure(figsize=(10, 6))
    window = 50
    moving_avg = [np.mean(rewards_per_episode[max(0, i-window+1):i+1])
                 for i in range(len(rewards_per_episode))]
    plt.plot(moving_avg, linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (50 episodes)')
    plt.title('CliffWalking-v0 - Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('cliff_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    env.close()
    return agent, rewards_per_episode

if __name__ == "__main__":
    train_cliffwalking(episodes=500)
```

### Implementation: `compare.py` - Hyperparameter Analysis

```python
import gym
import numpy as np
import matplotlib.pyplot as plt
from qlearning_agent import QLearningAgent

def compare_learning_rates():
    """Compare different learning rates"""
    env = gym.make('FrozenLake-v1')
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    episodes = 5000

    results = {}

    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        agent = QLearningAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            learning_rate=lr,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.999
        )

        rewards = []
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.get_action(state, training=True)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            agent.decay_epsilon()
            rewards.append(total_reward)

        results[lr] = rewards

    # Plot comparison
    plt.figure(figsize=(12, 6))
    for lr, rewards in results.items():
        window = 100
        moving_avg = [np.mean(rewards[max(0, i-window+1):i+1])
                     for i in range(len(rewards))]
        plt.plot(moving_avg, label=f'α={lr}', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Success Rate (100-episode window)')
    plt.title('Effect of Learning Rate on Q-Learning Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison saved to learning_rate_comparison.png")
    plt.close()

    env.close()

if __name__ == "__main__":
    compare_learning_rates()
```

---

## Deliverables Checklist

### Code
- [ ] `qlearning_agent.py` - Core Q-learning
- [ ] `train_frozenlake.py` - FrozenLake training
- [ ] `train_taxi.py` - Taxi training
- [ ] `train_cliff.py` - CliffWalking training
- [ ] `compare.py` - Hyperparameter analysis
- [ ] Training plots for all environments
- [ ] `README.md`

### Report (1000-1500 words)
- [ ] Q-learning algorithm explanation
- [ ] Mathematical formulation
- [ ] Results across 3 environments
- [ ] Hyperparameter analysis
- [ ] Convergence discussion

### Video (5 minutes)
- [ ] Q-learning fundamentals
- [ ] Code demonstration
- [ ] Results across environments
- [ ] Key insights

---

## Assessment (15 marks)

- **Code (5 marks)**: Correct Q-learning, works on multiple environments
- **Functionality (4 marks)**: Converges, achieves good performance
- **Documentation (3 marks)**: Clear explanations, good report
- **Presentation (2 marks)**: Effective video
- **Q&A (1 mark)**: Deep understanding of Q-learning

---

## Tips

1. **FrozenLake is hard**: Success rate of 0.7+ is good!
2. **Taxi trains faster**: Should reach positive rewards quickly
3. **Visualize Q-tables**: Heatmaps help understand learning
4. **Try SARSA**: Compare on-policy vs off-policy
5. **Experiment**: Learning rate and epsilon are crucial

Good luck, The Crown! 👑
