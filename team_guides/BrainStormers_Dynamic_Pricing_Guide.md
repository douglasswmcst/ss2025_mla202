# BrainStormers - Dynamic Pricing with Deep RL
## Combined Term Paper & Mini-Project Implementation Guide

**Team Members**: 3 students
**Research Paper**: [Dynamic Pricing on E-Commerce Platforms with DRL](https://arxiv.org/pdf/1912.02572)
**Submission Deadline**: Day 6 (Saturday, 11:59 PM)

---

## Overview

You will implement a simplified version of dynamic pricing using Deep Reinforcement Learning. This guide walks you through creating a working system that demonstrates how RL can optimize pricing strategies in e-commerce.

---

## Part 1: Term Paper Report (10 marks)

### Required Sections (1000-1500 words)

#### 1. Introduction & Problem Statement (2 marks)
Write about:
- What is dynamic pricing and why is it important in e-commerce?
- Traditional pricing strategies vs. RL-based approaches
- Your specific problem: "How can we use DRL to maximize revenue while maintaining customer satisfaction?"

**Key points to cover**:
- Market competition and demand fluctuations
- Need for real-time adaptive pricing
- Business metrics: revenue, conversion rate, customer retention

#### 2. Methodology & Literature Review (3 marks)
Based on your paper, explain:
- **RL Framework**: State space, action space, reward function
- **Algorithm Used**: DQN or Policy Gradient methods
- **State representation**: Customer features, inventory, competitor prices, time
- **Actions**: Price adjustments (e.g., ±5%, ±10%, no change)
- **Rewards**: Revenue - cost penalties for extreme prices

**Technical approach**:
```
State = [current_price, inventory_level, demand_indicator, competitor_price, time_of_day]
Actions = {decrease_10%, decrease_5%, maintain, increase_5%, increase_10%}
Reward = units_sold × (price - cost) - customer_dissatisfaction_penalty
```

#### 3. Findings & Discussion (3 marks)
Discuss:
- Performance comparison: RL vs. fixed pricing vs. rule-based pricing
- Convergence behavior of your RL agent
- Impact on revenue and customer behavior
- Limitations and challenges

#### 4. Organization & References (2 marks)
- Proper citations (APA format)
- Clear structure with sections and subsections
- At least 5 references including your main paper
- Figures and tables with captions

---

## Part 2: Technical Implementation (15 marks)

### Step-by-Step Implementation Guide

#### Setup (Day 1)

```bash
# Create project directory
mkdir dynamic_pricing_rl
cd dynamic_pricing_rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required libraries
pip install numpy pandas matplotlib gym torch scikit-learn
```

#### File Structure
```
dynamic_pricing_rl/
├── environment.py          # E-commerce environment
├── agent.py               # DQN agent
├── train.py              # Training script
├── evaluate.py           # Evaluation and visualization
├── utils.py              # Helper functions
├── requirements.txt
└── README.md
```

---

### Implementation: environment.py (Day 2)

Create a simplified e-commerce pricing environment:

```python
import numpy as np
import gym
from gym import spaces

class DynamicPricingEnv(gym.Env):
    """
    E-commerce Dynamic Pricing Environment

    State: [current_price, inventory, demand_level, competitor_price, hour_of_day]
    Action: 0-4 representing price changes: [-10%, -5%, 0%, +5%, +10%]
    Reward: Profit with customer satisfaction consideration
    """

    def __init__(self, base_price=100, max_inventory=1000, max_steps=100):
        super(DynamicPricingEnv, self).__init__()

        self.base_price = base_price
        self.cost_per_unit = base_price * 0.6  # 60% of base price
        self.max_inventory = max_inventory
        self.max_steps = max_steps

        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # State space: [price, inventory, demand, competitor_price, hour]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([500, max_inventory, 100, 500, 23]),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_price = self.base_price
        self.inventory = self.max_inventory
        self.total_revenue = 0
        self.total_units_sold = 0

        # Random initial conditions
        self.demand_level = np.random.uniform(30, 70)
        self.competitor_price = np.random.uniform(80, 120)
        self.hour = np.random.randint(0, 24)

        return self._get_state()

    def _get_state(self):
        """Return current state"""
        return np.array([
            self.current_price,
            self.inventory,
            self.demand_level,
            self.competitor_price,
            self.hour
        ], dtype=np.float32)

    def _calculate_demand(self, price):
        """
        Calculate demand based on price elasticity
        Demand decreases with higher prices and increases with competitor advantage
        """
        # Price elasticity: demand decreases as price increases
        price_factor = max(0, (150 - price) / 150)

        # Competitor factor: demand increases if we're cheaper
        competitor_factor = 1.0 + 0.3 * (self.competitor_price - price) / price
        competitor_factor = np.clip(competitor_factor, 0.5, 2.0)

        # Time factor: peak hours have higher demand
        time_factor = 1.0 + 0.3 * np.sin(self.hour * np.pi / 12)

        # Base demand with noise
        base_demand = self.demand_level + np.random.normal(0, 5)

        # Final demand calculation
        demand = base_demand * price_factor * competitor_factor * time_factor
        demand = max(0, int(demand))

        return min(demand, self.inventory)

    def step(self, action):
        """
        Execute action and return next state, reward, done, info

        Actions:
        0: Decrease price by 10%
        1: Decrease price by 5%
        2: Maintain price
        3: Increase price by 5%
        4: Increase price by 10%
        """
        # Apply action to price
        price_changes = [-0.10, -0.05, 0.0, 0.05, 0.10]
        price_change = price_changes[action]
        self.current_price = self.current_price * (1 + price_change)

        # Ensure price stays in reasonable range
        self.current_price = np.clip(self.current_price, 50, 200)

        # Calculate demand and units sold
        units_sold = self._calculate_demand(self.current_price)

        # Calculate revenue and profit
        revenue = units_sold * self.current_price
        cost = units_sold * self.cost_per_unit
        profit = revenue - cost

        # Update inventory
        self.inventory -= units_sold

        # Calculate reward with penalties
        # Penalty for extreme pricing
        price_penalty = 0
        if self.current_price < 70 or self.current_price > 150:
            price_penalty = -50

        # Penalty for stockout
        stockout_penalty = -100 if self.inventory <= 0 else 0

        # Reward is profit with penalties
        reward = profit + price_penalty + stockout_penalty

        # Update tracking variables
        self.total_revenue += revenue
        self.total_units_sold += units_sold

        # Update environment dynamics
        self.current_step += 1
        self.hour = (self.hour + 1) % 24
        self.demand_level = np.clip(
            self.demand_level + np.random.normal(0, 3), 20, 80
        )
        self.competitor_price = np.clip(
            self.competitor_price + np.random.normal(0, 5), 80, 120
        )

        # Check if episode is done
        done = self.current_step >= self.max_steps or self.inventory <= 0

        # Info for logging
        info = {
            'revenue': revenue,
            'units_sold': units_sold,
            'profit': profit,
            'total_revenue': self.total_revenue,
            'total_units_sold': self.total_units_sold
        }

        return self._get_state(), reward, done, info

    def render(self, mode='human'):
        """Print current state"""
        print(f"Step: {self.current_step}")
        print(f"Price: ${self.current_price:.2f}, Inventory: {self.inventory}")
        print(f"Demand Level: {self.demand_level:.1f}, Competitor: ${self.competitor_price:.2f}")
        print(f"Total Revenue: ${self.total_revenue:.2f}, Units Sold: {self.total_units_sold}")
        print("-" * 50)

# Test the environment
if __name__ == "__main__":
    env = DynamicPricingEnv()
    state = env.reset()
    print("Initial State:", state)

    for _ in range(5):
        action = env.action_space.sample()  # Random action
        state, reward, done, info = env.step(action)
        env.render()
        print(f"Reward: {reward:.2f}\n")
        if done:
            break
```

**Task**: Run this code and verify the environment works correctly.

---

### Implementation: agent.py (Day 3)

Create a DQN agent for learning optimal pricing:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    """Deep Q-Network for pricing decisions"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent for dynamic pricing"""

    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        # Networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Training parameters
        self.batch_size = 64
        self.update_target_every = 10
        self.train_step = 0

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        """Train the agent using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
```

---

### Implementation: train.py (Day 4)

```python
import numpy as np
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
    return np.convolve(data, np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    train_agent(episodes=500, max_steps=100)
```

---

### Implementation: evaluate.py (Day 5)

```python
import numpy as np
import matplotlib.pyplot as plt
from environment import DynamicPricingEnv
from agent import DQNAgent

def evaluate_agent(agent, episodes=10, max_steps=100):
    """Evaluate trained agent"""

    env = DynamicPricingEnv(max_steps=max_steps)

    total_rewards = []
    total_revenues = []
    price_histories = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        prices = []

        for step in range(max_steps):
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)

            episode_reward += reward
            prices.append(env.current_price)

            if done:
                break

        total_rewards.append(episode_reward)
        total_revenues.append(info['total_revenue'])
        price_histories.append(prices)

        print(f"Episode {episode + 1}: Revenue = ${info['total_revenue']:.2f}, "
              f"Units Sold = {info['total_units_sold']}")

    print(f"\nAverage Revenue: ${np.mean(total_revenues):.2f}")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")

    # Compare with baseline strategies
    compare_strategies(env, agent)

    # Plot price evolution
    plot_price_evolution(price_histories)

    return total_rewards, total_revenues

def compare_strategies(env, rl_agent, episodes=20):
    """Compare RL agent with baseline strategies"""

    strategies = {
        'RL Agent': lambda state: rl_agent.select_action(state, training=False),
        'Fixed Price': lambda state: 2,  # Always maintain price
        'Random': lambda state: np.random.randint(0, 5),
        'High Price': lambda state: 4,  # Always increase
        'Low Price': lambda state: 0,   # Always decrease
    }

    results = {name: [] for name in strategies}

    for name, strategy in strategies.items():
        for _ in range(episodes):
            state = env.reset()
            total_revenue = 0

            for _ in range(100):
                action = strategy(state)
                state, reward, done, info = env.step(action)
                if done:
                    total_revenue = info['total_revenue']
                    break

            results[name].append(total_revenue)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = range(len(strategies))
    means = [np.mean(results[name]) for name in strategies]
    stds = [np.std(results[name]) for name in strategies]

    ax.bar(positions, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(strategies.keys(), rotation=45, ha='right')
    ax.set_ylabel('Average Revenue ($)')
    ax.set_title('Pricing Strategy Comparison')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("\nStrategy comparison saved as 'strategy_comparison.png'")
    plt.close()

    # Print results
    print("\nStrategy Comparison Results:")
    for name in strategies:
        print(f"{name:15s}: ${np.mean(results[name]):8.2f} ± ${np.std(results[name]):6.2f}")

def plot_price_evolution(price_histories):
    """Plot how prices evolve over episodes"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, prices in enumerate(price_histories[:5]):  # Plot first 5 episodes
        ax.plot(prices, label=f'Episode {i+1}', alpha=0.7)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price ($)')
    ax.set_title('Price Evolution Over Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('price_evolution.png', dpi=300, bbox_inches='tight')
    print("Price evolution saved as 'price_evolution.png'")
    plt.close()

if __name__ == "__main__":
    # Load trained agent
    env = DynamicPricingEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    agent.load('trained_pricing_agent.pth')

    # Evaluate
    evaluate_agent(agent, episodes=10)
```

---

### Final Steps (Day 6)

#### 1. Create README.md

```markdown
# Dynamic Pricing with Deep Reinforcement Learning

Implementation of DQN-based dynamic pricing for e-commerce platforms.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train agent
python train.py

# Evaluate agent
python evaluate.py
```

## Results
- RL agent achieves X% higher revenue than fixed pricing
- Converges after ~300 episodes
- See visualizations in generated PNG files
```

#### 2. Create requirements.txt

```
numpy==1.24.0
torch==2.0.0
gym==0.26.0
matplotlib==3.7.0
pandas==2.0.0
```

#### 3. Run Complete Pipeline

```bash
# Train
python train.py

# Evaluate
python evaluate.py
```

---

## Deliverables Checklist

### Code (GitHub Repository)
- [ ] `environment.py` - Working e-commerce environment
- [ ] `agent.py` - DQN implementation
- [ ] `train.py` - Training script with results
- [ ] `evaluate.py` - Evaluation and comparison
- [ ] `README.md` - Clear documentation
- [ ] `requirements.txt` - Dependencies

### Report (PDF, 1000-1500 words)
- [ ] Introduction with problem statement
- [ ] Methodology and RL framework explanation
- [ ] Results with figures (training curves, comparison charts)
- [ ] Discussion and limitations
- [ ] References (APA format)

### Video Demonstration (5 minutes)
- [ ] Introduction to the problem (1 min)
- [ ] Code walkthrough (2 min)
- [ ] Results demonstration (1.5 min)
- [ ] Conclusion (0.5 min)

---

## Assessment Rubric

### Code Quality & Implementation (5 marks)
- Environment correctly implements state/action/reward
- DQN agent properly implemented with replay buffer
- Training converges and shows improvement
- Code is well-commented and organized

### Functionality & Correctness (4 marks)
- All scripts run without errors
- Results are reproducible
- RL agent outperforms baselines
- Visualizations are clear and informative

### Results Analysis & Documentation (3 marks)
- README clearly explains project
- Report discusses findings and limitations
- Comparison with baseline strategies
- Professional presentation of results

### Video Presentation (2 marks)
- Clear explanation of approach
- Effective demonstration of results
- Professional delivery
- Answers potential questions

### Q&A Readiness (1 mark)
- Understanding of RL concepts
- Ability to explain design decisions
- Knowledge of limitations and improvements

---

## Tips for Success

1. **Start Early**: Don't wait until Day 5 to start coding
2. **Test Incrementally**: Run each file as you complete it
3. **Document as You Go**: Add comments while coding
4. **Simple First**: Get basic version working before optimization
5. **Visualize Everything**: Plots make results more convincing
6. **Compare Baselines**: Show RL is better than simple strategies

---

## Common Issues & Solutions

**Environment not converging**: Adjust reward function, check state normalization

**Training too slow**: Reduce episodes or max_steps for faster testing

**Poor performance**: Tune hyperparameters (learning rate, epsilon decay)

**Code errors**: Check tensor dimensions, numpy/torch conversions

---

## Questions? Contact Module Tutors

Good luck, BrainStormers! 🚀
