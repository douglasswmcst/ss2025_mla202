# Practical 3: Deep Q-Networks - From Tables to Neural Networks

Welcome to Practical 3! In Practical 2, you implemented Q-learning and achieved >70% success on FrozenLake. But what happens when we face environments with millions of states, or worse, continuous state spaces? A Q-table becomes impossible.

Today, we'll solve this by replacing the Q-table with a neural network - creating a Deep Q-Network (DQN). This breakthrough algorithm, published by DeepMind in 2015, learned to play Atari games at superhuman levels.

### Learning Objectives

* ✅ Understand why tabular Q-learning fails with large state spaces
* ✅ Implement neural networks as Q-value function approximators
* ✅ Master experience replay for stable learning
* ✅ Understand the role of target networks
* ✅ Train a DQN agent to solve CartPole (avg reward >195)
* ✅ Extend DQN to visual domains (Atari games)
* ✅ Compare DQN performance against tabular methods

-----

## 1. Why We Need DQN: The Limits of Q-Tables

### The Problem with Large State Spaces

In Practical 2, our FrozenLake environment had only **16 discrete states** and **4 actions**, giving us a Q-table of size 16×4 = 64 entries. That's tiny and manageable.

But what about CartPole? Let's analyze its state space:

**CartPole State Space:**
- Cart Position: continuous value roughly in [-4.8, 4.8]
- Cart Velocity: continuous value roughly in [-∞, ∞]
- Pole Angle: continuous value roughly in [-0.418, 0.418] radians
- Pole Angular Velocity: continuous value roughly in [-∞, ∞]

**The Q-Table Size Problem:**

If we tried to discretize each continuous value into just 10 bins:
- Q-table size = 10 × 10 × 10 × 10 × 2 actions = **20,000 entries**

If we wanted more precision with 100 bins per dimension:
- Q-table size = 100 × 100 × 100 × 100 × 2 = **200,000,000 entries**

And for Atari games with visual inputs (210×160×3 RGB pixels):
- State space dimensionality = **100,800 dimensions**
- A Q-table would be impossibly large!

### 🔑 Key Concept: The Curse of Dimensionality

**The curse of dimensionality** refers to how the state space explodes exponentially as we add more dimensions:

- **1 dimension**, 100 bins → 100 states
- **2 dimensions**, 100 bins each → 10,000 states
- **3 dimensions**, 100 bins each → 1,000,000 states
- **4 dimensions**, 100 bins each → 100,000,000 states

**Why this matters:**
1. **Memory**: We can't store tables with billions of entries
2. **Learning**: We'd need to visit every state-action pair many times
3. **Generalization**: Similar states should have similar Q-values, but Q-tables treat each state independently

**Real-world analogy**:
Imagine learning to drive. You don't need to memorize the exact action for every possible combination of speed, steering angle, and road position (that would be a Q-table). Instead, you learn general patterns: "If going fast and approaching a turn, slow down" - this is **function approximation**.

### The Solution: Function Approximation

Instead of storing Q(s,a) in a table, we approximate it with a **function**:

**Q-table approach:**
```
Q(state_0, action_0) = 0.5
Q(state_0, action_1) = 0.8
Q(state_1, action_0) = 0.3
... (store all values explicitly)
```

**Function approximation approach:**
```
Q(s,a) ≈ Q(s,a; θ)
```

Where **θ** represents the parameters of our function approximator (a neural network).

**Why this works:**
- **Generalization**: Similar states produce similar Q-values automatically
- **Scalability**: Number of parameters doesn't depend on state space size
- **Continuous states**: No need to discretize - handle continuous values directly

-----

## 2. Setup and Installation

### Step 1: Verify Previous Setup

You should already have Python and Gymnasium from Practicals 1 & 2. Let's verify:

```bash
python3 --version  # Should be 3.8 or higher
python3 -c "import gymnasium; print(f'Gymnasium version: {gymnasium.__version__}')"
```

### Step 2: Install PyTorch

PyTorch is a deep learning framework that makes building and training neural networks easy. Install it with:

```bash
# For CPU-only (sufficient for this practical)
pip install torch torchvision

# For GPU (optional, if you have NVIDIA GPU with CUDA)
# This will make training faster but is not required
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> **Note**: The CPU version is perfectly fine for CartPole. The GPU version helps with Atari training but isn't necessary for completing the practical.

### Step 3: Install Additional Dependencies

```bash
# For Atari games support
pip install gymnasium[atari]
pip install ale-py

# For image processing (Atari)
pip install opencv-python

# For visualization (already have from previous practicals)
pip install matplotlib numpy
```

### Step 4: Verify Installation

Create a test script to verify everything works:

```python
import torch
import gymnasium as gym
import numpy as np

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ Gymnasium version: {gym.__version__}")
print(f"✅ NumPy version: {np.__version__}")

# Test creating a simple neural network
model = torch.nn.Linear(4, 2)
print(f"✅ Can create neural networks!")

# Test CartPole environment
env = gym.make("CartPole-v1")
print(f"✅ CartPole environment ready!")

print("\n🎉 All dependencies installed correctly!")
```

Run it with:
```bash
python3 test_setup.py
```

-----

## 3. From Q-Tables to Q-Networks

### 🔑 Key Concept: Neural Networks as Function Approximators

A neural network is a **universal function approximator** - it can learn to approximate almost any function, including our Q-function.

**What is a neural network?**

Think of it as a series of mathematical transformations:

```
Input (state) → Layer 1 → Layer 2 → ... → Output (Q-values)
```

Each layer transforms the data using:
1. **Linear transformation**: Multiply by weights, add bias
2. **Non-linear activation**: Apply a function like ReLU (max(0, x))

**Example: Q-Network for CartPole**

```
Input: [cart_pos, cart_vel, pole_angle, pole_vel]  # 4 numbers
   ↓ (Linear: 4 → 128)
Hidden Layer 1: [128 neurons with ReLU activation]
   ↓ (Linear: 128 → 128)
Hidden Layer 2: [128 neurons with ReLU activation]
   ↓ (Linear: 128 → 2)
Output: [Q(s, left), Q(s, right)]  # 2 Q-values
```

**In PyTorch code:**

```python
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(QNetwork, self).__init__()
        # Layer 1: state → hidden
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # Layer 2: hidden → hidden
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Layer 3: hidden → actions
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        # Forward pass through the network
        x = torch.relu(self.fc1(state))  # First hidden layer
        x = torch.relu(self.fc2(x))      # Second hidden layer
        q_values = self.fc3(x)           # Output Q-values (no activation)
        return q_values
```

**Key differences from Q-table:**

| Aspect | Q-Table (Practical 2) | Q-Network (Practical 3) |
|--------|----------------------|-------------------------|
| **Lookup** | `q_table[state, action]` | `q_network(state)[action]` |
| **Update** | Modify single entry | Gradient descent on network weights |
| **Memory** | Size = num_states × num_actions | Size = network parameters (constant) |
| **Generalization** | None (each entry independent) | Automatic (similar states → similar Q-values) |
| **State space** | Only discrete states | Continuous or discrete |

### 🔑 Key Concept: How Neural Networks Generalize

This is the magic that makes DQN work!

**Example: Learning CartPole**

Suppose the network sees these experiences:
- State: [cart_pos=0.5, cart_vel=0.1, angle=0.02, angular_vel=0.0] → Action: right → Reward: good
- State: [cart_pos=0.51, cart_vel=0.09, angle=0.021, angular_vel=0.01] → Action: right → Reward: good

**With a Q-table**: These would be two completely separate entries (after discretization). We learn nothing about one from the other.

**With a Q-network**: The network learns that states with similar values should have similar Q-values. When we update the network for the first state, the second state **automatically** gets a better Q-value too!

**Mathematical notation:**
- Q-table: Q(s,a) - just a lookup table
- Q-network: Q(s,a; θ) - a function parameterized by θ (network weights)

**The learning process:**
1. Experience a transition: (s, a, r, s')
2. Calculate target Q-value: target = r + γ max Q(s', a'; θ)
3. Calculate current Q-value: current = Q(s, a; θ)
4. Calculate error: loss = (target - current)²
5. **Update weights θ using gradient descent** (not just one table entry!)

-----

## 4. The Three Pillars of DQN

Deep Q-Network (DQN) isn't just "Q-learning with neural networks." It requires three critical innovations to work stably:

### 🔑 Key Concept: Pillar 1 - The Q-Network Architecture

**What it is**: A neural network that takes a state as input and outputs Q-values for all possible actions.

**Why we need it**: To handle large or continuous state spaces through function approximation.

**Architecture choices matter**:
- **Too small**: Can't represent complex Q-functions, underfitting
- **Too large**: Slow training, overfitting, high memory usage
- **Just right**: CartPole uses 2 hidden layers with 128 neurons each

**Implementation pattern**:
```python
# Input: state (e.g., [0.5, 0.1, 0.02, 0.0] for CartPole)
state_tensor = torch.FloatTensor(state)

# Forward pass
q_values = q_network(state_tensor)
# Output: [Q(s, action_0), Q(s, action_1)]

# Choose best action
action = q_values.argmax().item()
```

### 🔑 Key Concept: Pillar 2 - Experience Replay

**The Problem without Replay:**

In Practical 2, we updated Q-values immediately after each experience:
```python
experience = (s, a, r, s')
q_table[s, a] = q_table[s, a] + α * (r + γ * max_q_next - q_table[s, a])
```

With neural networks, this causes **catastrophic problems**:

1. **Sequential Correlation**: Consecutive experiences are highly correlated
   - Frame 1: cart moving right with pole tilting right
   - Frame 2: cart moving right with pole tilting right
   - Frame 3: cart moving right with pole tilting right
   - The network only sees similar situations and overfits to recent experiences

2. **Rapid Forgetting**: Training on new data makes the network forget old data

**The Solution: Experience Replay Buffer**

Store experiences in a buffer and sample randomly for training:

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store experience
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Sample random batch
        batch = random.sample(self.buffer, batch_size)
        return batch
```

**How it works**:
1. **Collect**: Store every experience (s, a, r, s', done) in the buffer
2. **Store**: Keep up to N most recent experiences (e.g., N=10,000)
3. **Sample**: Randomly select a batch (e.g., 64 experiences)
4. **Train**: Update network using this random batch

**Why this works**:
- **Breaks correlation**: Random sampling means consecutive training samples are uncorrelated
- **Data efficiency**: Each experience can be used for training multiple times
- **Stability**: Network sees diverse experiences, not just recent trajectory

**Real-world analogy**: Instead of studying one chapter repeatedly until you master it (then forgetting it when you study the next chapter), you review random topics from all chapters in each study session.

### 🔑 Key Concept: Pillar 3 - Target Network

**The Moving Target Problem:**

In Q-learning, we update Q(s,a) using:
```
target = r + γ * max Q(s', a')
```

But with neural networks, **both the current Q-value AND the target depend on the same network parameters θ**:
```
loss = (r + γ * max Q(s', a'; θ) - Q(s, a; θ))²
```

**The problem**: As we update θ to minimize loss, the target also changes! It's like trying to hit a moving dartboard.

**The Solution: Target Network**

Maintain two networks:
1. **Q-Network (θ)**: Updated every step, used for choosing actions
2. **Target Network (θ⁻)**: Updated slowly, used only for computing targets

**Implementation:**
```python
# Initialize both networks
q_network = QNetwork(state_dim, action_dim)
target_network = QNetwork(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())  # Copy weights

# During training:
# 1. Compute target using target_network (frozen)
with torch.no_grad():  # Don't compute gradients for target
    next_q_values = target_network(next_states)
    target_q = rewards + gamma * next_q_values.max(1)[0]

# 2. Compute current Q using q_network
current_q = q_network(states).gather(1, actions)

# 3. Update q_network only
loss = (target_q - current_q).pow(2).mean()
loss.backward()  # Only q_network gets updated

# 4. Periodically update target_network
if episode % target_update_freq == 0:
    target_network.load_state_dict(q_network.state_dict())
```

**Update frequency matters**:
- **Every step (no target network)**: Very unstable, moving target problem
- **Every 1-10 episodes**: Good balance for CartPole
- **Every 1000+ episodes**: Better for Atari with more complex environments

**Why this works**:
- **Stable targets**: The target Q-values don't change while we're training
- **Smooth learning**: Gradual updates prevent wild oscillations
- **Convergence**: The network has time to learn before targets shift

**Mathematical view**:
- Without target network: minimizing (Q(s,a; θ) - [r + γ max Q(s',a'; θ)])²
  - Both terms depend on θ - target moves as we learn!
- With target network: minimizing (Q(s,a; θ) - [r + γ max Q(s',a'; θ⁻)])²
  - Only first term depends on θ - target is stable!

### 🔑 Key Concept: The DQN Loss Function

Now we can put it all together. The loss function for DQN is:

**Loss = (Target Q-value - Current Q-value)²**

More precisely:

**L(θ) = E[(r + γ max Q(s', a'; θ⁻) - Q(s, a; θ))²]**

Where:
- **θ**: Parameters of the Q-network (being updated)
- **θ⁻**: Parameters of the target network (frozen)
- **E**: Expectation over random batch from replay buffer

**Breaking it down:**

1. **Target Q-value**: r + γ max Q(s', a'; θ⁻)
   - Computed using target network
   - Represents our best estimate of what Q(s,a) should be

2. **Current Q-value**: Q(s, a; θ)
   - Computed using Q-network
   - Our current estimate

3. **Temporal Difference Error**: (target - current)
   - How wrong our current estimate is
   - Same concept as in tabular Q-learning!

4. **Squared Error**: Error²
   - Penalizes large errors more than small errors
   - Smooth gradient for optimization

**In code:**
```python
# Sample batch from replay buffer
states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

# Compute current Q-values
current_q = q_network(states).gather(1, actions.unsqueeze(1))

# Compute target Q-values
with torch.no_grad():
    next_q = target_network(next_states).max(1)[0]
    target_q = rewards + (1 - dones) * gamma * next_q

# Compute loss
loss = nn.MSELoss()(current_q.squeeze(), target_q)

# Update network
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

-----

## 5. Implementing DQN for CartPole

Now let's put everything together in a complete implementation!

### Task 1: Create the Q-Network

First, we define our neural network architecture:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """
    Neural network that approximates Q(s,a) for all actions

    Architecture for CartPole:
    - Input: State (4 values)
    - Hidden Layer 1: 128 neurons with ReLU
    - Hidden Layer 2: 128 neurons with ReLU
    - Output: Q-values for each action (2 values)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Forward pass: state → Q-values

        Args:
            state: Tensor of shape (batch_size, state_dim)

        Returns:
            q_values: Tensor of shape (batch_size, action_dim)
        """
        x = torch.relu(self.fc1(state))  # Apply first layer + ReLU
        x = torch.relu(self.fc2(x))      # Apply second layer + ReLU
        q_values = self.fc3(x)           # Output layer (no activation)
        return q_values
```

**Key points:**
- **No activation on output**: Q-values can be any real number (positive or negative)
- **ReLU activation**: Most common choice, prevents vanishing gradients
- **Architecture size**: 128 neurons is a good starting point for CartPole

### Task 2: Implement the Replay Buffer

```python
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions

    Why we need this:
    - Breaks correlation between consecutive samples
    - Improves sample efficiency by reusing experiences
    - Stabilizes learning
    """
    def __init__(self, capacity=10000):
        """
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state reached
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random batch of transitions

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        # Sample random batch
        batch = random.sample(self.buffer, batch_size)

        # Unzip batch into separate lists
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)
```

### Task 3: Create the DQN Agent

Now we combine everything into a complete agent:

```python
class DQNAgent:
    """
    DQN Agent that learns to play CartPole

    Implements:
    - Q-network and target network
    - Experience replay
    - Epsilon-greedy exploration
    - Training via gradient descent
    """
    def __init__(self, state_dim, action_dim):
        """
        Initialize the DQN agent

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        # Q-network (the one we train)
        self.q_network = QNetwork(state_dim, action_dim)

        # Target network (slowly tracks q_network)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer (Adam is good default)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

        # Hyperparameters
        self.gamma = 0.99           # Discount factor
        self.epsilon = 1.0          # Exploration rate (start at 100%)
        self.epsilon_decay = 0.995  # Decay exploration over time
        self.epsilon_min = 0.01     # Minimum exploration (always explore 1%)
        self.batch_size = 64        # Training batch size
        self.target_update_freq = 10  # Update target network every N episodes

        # For tracking
        self.steps = 0

    def choose_action(self, state, env):
        """
        Choose action using epsilon-greedy policy

        Args:
            state: Current state
            env: Environment (for action space)

        Returns:
            action: Action to take
        """
        if random.random() < self.epsilon:
            # Explore: random action
            return env.action_space.sample()
        else:
            # Exploit: choose action with highest Q-value
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():  # No gradients needed for inference
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """
        Perform one training step (gradient descent)

        Returns:
            loss: Training loss (or None if not enough samples)
        """
        # Need enough samples to train
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Compute current Q-values: Q(s,a) for the actions we took
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values: r + γ max Q(s',a') using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            # If episode ended (done=1), there's no next state, so target is just reward
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss (Mean Squared Error)
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Gradient descent
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update weights

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Reduce exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

### Task 4: The Complete Training Loop

Now we can train our agent:

```python
import gymnasium as gym
import matplotlib.pyplot as plt

def train_dqn(num_episodes=500):
    """
    Train DQN agent on CartPole

    Target: Average reward >195 over 100 consecutive episodes
    Expected training time: 5-10 minutes on CPU

    Args:
        num_episodes: Number of episodes to train

    Returns:
        agent: Trained DQN agent
        rewards_per_episode: List of rewards per episode
        losses: List of training losses
    """
    # Create environment
    env = gym.make("CartPole-v1")

    # Get dimensions
    state_dim = env.observation_space.shape[0]  # 4 for CartPole
    action_dim = env.action_space.n             # 2 for CartPole

    # Create agent
    agent = DQNAgent(state_dim, action_dim)

    # Tracking
    rewards_per_episode = []
    losses = []

    print("Starting DQN training on CartPole...")
    print(f"Target: Average reward >195 over 100 episodes\n")

    # Training loop
    for episode in range(num_episodes):
        # Reset environment
        state, info = env.reset()
        total_reward = 0
        done = False

        # Episode loop
        while not done:
            # Choose and take action
            action = agent.choose_action(state, env)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, float(done))

            # Train on a batch
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            # Move to next state
            state = next_state
            total_reward += reward

        # End of episode
        rewards_per_episode.append(total_reward)

        # Decay exploration
        agent.decay_epsilon()

        # Update target network periodically
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:]) if len(rewards_per_episode) >= 100 else np.mean(rewards_per_episode)
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Avg Loss: {np.mean(losses[-100:]) if losses else 0:.4f}")

        # Check if solved
        if len(rewards_per_episode) >= 100:
            avg_last_100 = np.mean(rewards_per_episode[-100:])
            if avg_last_100 >= 195:
                print(f"\n🎉 Solved! Average reward {avg_last_100:.2f} >= 195")
                print(f"Solved in {episode + 1} episodes!")
                break

    env.close()
    return agent, rewards_per_episode, losses
```

### Task 5: Visualization and Testing

Let's add functions to visualize and test our agent:

```python
def plot_training_results(rewards, losses):
    """
    Plot learning curves

    Args:
        rewards: List of rewards per episode
        losses: List of training losses
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot rewards
    ax1.plot(rewards, alpha=0.3, label='Episode Reward')

    # Calculate and plot moving average
    window = 100
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg,
                label='Moving Average (100 episodes)', linewidth=2, color='orange')

    ax1.axhline(y=195, color='r', linestyle='--', label='Target (195)', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('DQN Training Progress on CartPole')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot losses
    if losses:
        ax2.plot(losses, alpha=0.5)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss (MSE)')
        ax2.set_title('Training Loss Over Time')
        ax2.set_yscale('log')  # Log scale often clearer for loss
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('practical3/dqn_cartpole_results.png', dpi=150)
    print("\n📊 Plots saved to practical3/dqn_cartpole_results.png")
    plt.show()

def test_agent(agent, num_episodes=100):
    """
    Test the trained agent

    Args:
        agent: Trained DQN agent
        num_episodes: Number of test episodes

    Returns:
        test_rewards: List of rewards from test episodes
    """
    env = gym.make("CartPole-v1")
    test_rewards = []

    print(f"\n Testing agent for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Pure exploitation (no exploration)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            action = q_values.argmax().item()

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        test_rewards.append(total_reward)

    env.close()

    # Calculate statistics
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    min_reward = min(test_rewards)
    max_reward = max(test_rewards)

    print(f"\n=== TEST RESULTS ===")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {min_reward:.0f}")
    print(f"Max Reward: {max_reward:.0f}")
    print(f"Success Rate: {sum(1 for r in test_rewards if r >= 195) / len(test_rewards) * 100:.1f}%")
    print(f"Solved: {'✅ YES' if avg_reward >= 195 else '❌ NO'}")

    return test_rewards
```

### Task 6: Putting It All Together

Create the main execution script:

```python
if __name__ == "__main__":
    import torch

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("DQN Training on CartPole-v1")
    print("=" * 60)

    # Train the agent
    agent, rewards, losses = train_dqn(num_episodes=500)

    # Plot results
    plot_training_results(rewards, losses)

    # Test the agent
    test_rewards = test_agent(agent, num_episodes=100)

    # Save the model
    torch.save(agent.q_network.state_dict(), 'practical3/dqn_cartpole_model.pth')
    print("\n💾 Model saved to practical3/dqn_cartpole_model.pth")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
```

**Expected Output:**

```
============================================================
DQN Training on CartPole-v1
============================================================
Starting DQN training on CartPole...
Target: Average reward >195 over 100 episodes

Episode 50/500, Avg Reward (last 100): 22.50, Epsilon: 0.778, Avg Loss: 0.4521
Episode 100/500, Avg Reward (last 100): 35.80, Epsilon: 0.606, Avg Loss: 0.3215
Episode 150/500, Avg Reward (last 100): 62.40, Epsilon: 0.471, Avg Loss: 0.2543
Episode 200/500, Avg Reward (last 100): 112.30, Epsilon: 0.367, Avg Loss: 0.1987
Episode 250/500, Avg Reward (last 100): 165.20, Epsilon: 0.285, Avg Loss: 0.1654

🎉 Solved! Average reward 197.85 >= 195
Solved in 267 episodes!

📊 Plots saved to practical3/dqn_cartpole_results.png

Testing agent for 100 episodes...

=== TEST RESULTS ===
Average Reward: 241.50 ± 68.32
Min Reward: 108
Max Reward: 500
Success Rate: 72.0%
Solved: ✅ YES

💾 Model saved to practical3/dqn_cartpole_model.pth

============================================================
Training Complete!
============================================================
```

-----

## 6. Understanding What DQN Learned

### Analyzing the Q-Values

Let's visualize what our Q-network learned. We'll create a heatmap showing Q-values across different states:

```python
def visualize_q_values(agent):
    """
    Visualize Q-values across the state space

    This helps us understand what the network learned
    """
    import matplotlib.pyplot as plt

    # Sample positions and angles
    cart_positions = np.linspace(-2.4, 2.4, 20)
    pole_angles = np.linspace(-0.2, 0.2, 20)

    # Initialize grid for max Q-values
    q_value_grid = np.zeros((len(pole_angles), len(cart_positions)))

    # For each combination of position and angle
    for i, angle in enumerate(pole_angles):
        for j, pos in enumerate(cart_positions):
            # Create state (set velocities to 0 for visualization)
            state = torch.FloatTensor([pos, 0, angle, 0]).unsqueeze(0)

            # Get Q-values
            with torch.no_grad():
                q_values = agent.q_network(state)

            # Store max Q-value (value of being in this state)
            q_value_grid[i, j] = q_values.max().item()

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(q_value_grid, cmap='RdYlGn', aspect='auto',
                    extent=[-2.4, 2.4, -0.2, 0.2], origin='lower')
    plt.colorbar(im, label='Max Q-Value')
    plt.xlabel('Cart Position')
    plt.ylabel('Pole Angle (radians)')
    plt.title('Learned Q-Values Across State Space\n(Higher values = Better states)')
    plt.axhline(y=0, color='blue', linestyle='--', alpha=0.5, label='Balanced (angle=0)')
    plt.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Centered (pos=0)')
    plt.legend()
    plt.savefig('practical3/q_value_heatmap.png', dpi=150)
    print("📊 Q-value heatmap saved to practical3/q_value_heatmap.png")
    plt.show()

# After training:
visualize_q_values(agent)
```

**What to look for in the heatmap:**
- **High Q-values (green)**: States the agent considers good (centered cart, balanced pole)
- **Low Q-values (red)**: States the agent considers bad (edge of track, pole falling)
- **Gradient pattern**: Should see smooth transitions, showing the network generalizes

### Comparing DQN vs Tabular Q-Learning

Let's reflect on the key differences:

| Aspect | Tabular Q-Learning (Practical 2) | DQN (Practical 3) |
|--------|----------------------------------|-------------------|
| **State Space** | Discrete (16 states in FrozenLake) | Continuous (infinite states in CartPole) |
| **Memory** | Q-table: 16×4 = 64 entries | Network: ~17,000 parameters |
| **Generalization** | None (each entry independent) | Automatic (similar states → similar Q-values) |
| **Update Method** | Direct lookup and update | Gradient descent on loss function |
| **Training Speed** | Fast (5000 episodes in ~1 min) | Moderate (500 episodes in ~5 min) |
| **Complexity** | Simple to understand | Requires understanding of neural networks |
| **Scalability** | Fails with large state spaces | Handles continuous and high-dimensional spaces |

**When to use each:**
- **Tabular Q-Learning**: Small, discrete state spaces (games, grid worlds)
- **DQN**: Large, continuous, or visual state spaces (robotics, games with images)

-----

## 7. Extending DQN to Atari: Visual State Spaces

Now that you understand DQN on CartPole, let's tackle a much harder problem: learning from raw pixels!

### The Challenge of Visual Inputs

**CartPole state**: 4 numbers [position, velocity, angle, angular velocity]
**Atari state**: 210×160×3 = 100,800 pixels!

If we used a fully-connected network like we did for CartPole:
- Input layer: 100,800 neurons
- Hidden layer: 128 neurons
- Connections: 100,800 × 128 = **12,902,400 parameters**

This would be:
1. **Too slow**: Training would take forever
2. **Too memory-intensive**: Can't fit in RAM
3. **Too many parameters**: Would overfit badly
4. **Ignores spatial structure**: Doesn't use the fact that nearby pixels are related

**The solution**: Convolutional Neural Networks (CNNs)

### 🔑 Key Concept: Convolutional Neural Networks

CNNs are designed for image processing. Instead of connecting every pixel to every neuron, they use **filters** that scan across the image.

**Key idea**: The same features (edges, shapes) appear at different positions in an image, so we can share the same weights across positions.

**Architecture for Atari DQN**:
```
Input: 84×84×4 stacked grayscale frames
   ↓
Conv Layer 1: 32 filters, 8×8, stride 4 → 20×20×32
   ↓
Conv Layer 2: 64 filters, 4×4, stride 2 → 9×9×64
   ↓
Conv Layer 3: 64 filters, 3×3, stride 1 → 7×7×64
   ↓
Flatten: 7×7×64 = 3136 values
   ↓
Fully Connected: 512 neurons
   ↓
Output: Q-values for each action
```

**Why this is better:**
- Parameters reduced from ~13M to ~1.7M
- Uses spatial structure of images
- Can detect visual patterns (edges, shapes, objects)

### Frame Preprocessing

Raw Atari frames are 210×160×3 RGB images, which is:
1. **Too large**: 100,800 dimensions
2. **Color unnecessary**: Most games don't need color
3. **Single frame insufficient**: Can't tell direction/velocity from one frame

**Preprocessing steps**:

1. **Grayscale conversion**: 3 channels → 1 channel
2. **Downsampling**: 210×160 → 84×84
3. **Frame stacking**: Stack 4 consecutive frames

**Why frame stacking?**
- Provides **motion information**: Can see if ball is moving left/right, up/down
- Single frame is **ambiguous**: Can't tell velocity or direction
- 4 frames is the **sweet spot**: Enough for motion, not too much memory

**Implementation**:

```python
import cv2
from collections import deque

class FramePreprocessor:
    """
    Preprocess Atari frames for DQN

    Steps:
    1. Convert RGB to grayscale
    2. Resize to 84×84
    3. Normalize to [0, 1]
    4. Stack 4 consecutive frames
    """
    def __init__(self):
        self.frame_stack = deque(maxlen=4)

    def preprocess_frame(self, frame):
        """
        Process a single frame

        Args:
            frame: RGB image (210, 160, 3)

        Returns:
            processed: Grayscale image (84, 84)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize to 84×84
        resized = cv2.resize(gray, (84, 84))

        # Normalize to [0, 1]
        normalized = resized / 255.0

        return normalized

    def reset(self, initial_frame):
        """
        Initialize frame stack with first frame repeated 4 times

        Args:
            initial_frame: First frame of episode

        Returns:
            stacked: Numpy array (4, 84, 84)
        """
        processed = self.preprocess_frame(initial_frame)

        # Fill stack with same frame
        for _ in range(4):
            self.frame_stack.append(processed)

        return np.array(self.frame_stack)

    def step(self, frame):
        """
        Add new frame to stack

        Args:
            frame: New frame

        Returns:
            stacked: Numpy array (4, 84, 84)
        """
        processed = self.preprocess_frame(frame)
        self.frame_stack.append(processed)
        return np.array(self.frame_stack)
```

### CNN-based Q-Network for Atari

```python
class CNNQNetwork(nn.Module):
    """
    Convolutional Neural Network for Atari DQN

    Architecture from original DQN paper (Mnih et al., 2015):
    - Conv1: 32 filters, 8×8 kernel, stride 4
    - Conv2: 64 filters, 4×4 kernel, stride 2
    - Conv3: 64 filters, 3×3 kernel, stride 1
    - FC1: 512 neurons
    - Output: Q-values for each action
    """
    def __init__(self, num_actions):
        super(CNNQNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, state):
        """
        Forward pass through CNN

        Args:
            state: Tensor of shape (batch_size, 4, 84, 84)

        Returns:
            q_values: Tensor of shape (batch_size, num_actions)
        """
        # Convolutional layers with ReLU
        x = torch.relu(self.conv1(state))  # → (batch, 32, 20, 20)
        x = torch.relu(self.conv2(x))      # → (batch, 64, 9, 9)
        x = torch.relu(self.conv3(x))      # → (batch, 64, 7, 7)

        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)          # → (batch, 3136)

        # Fully connected layers
        x = torch.relu(self.fc1(x))        # → (batch, 512)
        q_values = self.fc2(x)             # → (batch, num_actions)

        return q_values
```

### Training DQN on Atari

The training loop is similar to CartPole, but with frame preprocessing:

```python
def train_atari_dqn(game="PongNoFrameskip-v4", num_episodes=1000):
    """
    Train DQN on Atari game

    Note: Full training to human-level requires 10M+ frames (hours/days).
    This demonstrates learning in a shorter timeframe.

    Args:
        game: Atari game name
        num_episodes: Number of episodes to train

    Returns:
        agent: Trained agent
        rewards: List of rewards per episode
    """
    # Create environment
    env = gym.make(game, render_mode="rgb_array")

    # Get action dimension
    num_actions = env.action_space.n

    # Create agent with CNN network
    agent = AtariDQNAgent(num_actions)

    # Tracking
    rewards_per_episode = []

    print(f"Training DQN on {game}")
    print("Note: Expect gradual improvement over many episodes\n")

    for episode in range(num_episodes):
        # Reset environment and preprocessor
        frame, info = env.reset()
        state = agent.frame_preprocessor.reset(frame)

        total_reward = 0
        done = False
        frame_count = 0

        while not done:
            # Choose action
            action = agent.choose_action(state)

            # Take action
            next_frame, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Preprocess next frame
            next_state = agent.frame_preprocessor.step(next_frame)

            # Clip rewards to [-1, 1] for stability (standard for Atari)
            clipped_reward = np.clip(reward, -1, 1)

            # Store transition
            agent.replay_buffer.push(state, action, clipped_reward, next_state, float(done))

            # Train every 4 frames (standard for Atari)
            if frame_count % 4 == 0:
                agent.train_step()

            state = next_state
            total_reward += reward
            frame_count += 1

        # End of episode
        rewards_per_episode.append(total_reward)
        agent.decay_epsilon()

        # Update target network less frequently for Atari
        if episode % 100 == 0 and episode > 0:
            agent.update_target_network()

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_per_episode[-50:])
            print(f"Episode {episode + 1}, Avg Reward (last 50): {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent, rewards_per_episode
```

**Key differences from CartPole:**
- **Frame preprocessing**: Convert pixels to 84×84×4 stacked frames
- **CNN network**: Use convolutional layers instead of fully connected
- **Reward clipping**: Clip to [-1, 1] for stable gradients across different games
- **Training frequency**: Train every 4 frames, not every frame
- **Slower target updates**: Update target network every 100 episodes instead of 10
- **Slower epsilon decay**: More exploration needed for complex visual patterns

**Expected performance:**
- **CartPole**: Solves in ~300 episodes (~5 minutes)
- **Atari Pong**: Shows improvement in ~500 episodes (~30 minutes)
- **Human-level Atari**: Requires 10M+ frames (~1-2 days even with GPU)

**Note**: We're demonstrating learning, not achieving state-of-the-art performance!

-----

## 8. Exercises 🧠

Now it's time to deepen your understanding through hands-on experimentation.

### Exercise 1: Network Architecture Exploration

The Q-network architecture significantly impacts learning performance.

1. **Hidden Layer Size Experiment**:
   - Modify the CartPole Q-network to use `hidden_dim=64` instead of 128
   - Train and compare performance against the original
   - **Questions**:
     * Does the smaller network learn slower or faster?
     * What is the final performance compared to the original?
     * Why might smaller networks be desirable despite any performance loss?
     * Calculate the number of parameters for both networks

2. **Network Depth Experiment**:
   - Add a third hidden layer to the Q-network (128 → 128 → 128 → 2)
   - **Questions**:
     * Does the deeper network improve performance?
     * What is the impact on training time?
     * When might deeper networks be beneficial in RL?
     * Does the deeper network converge faster or slower?

3. **Activation Function Experiment**:
   - Try replacing ReLU with Tanh or LeakyReLU
   - **Questions**:
     * How does this affect learning stability?
     * Which activation function works best for CartPole?
     * Why do we typically use ReLU in deep RL?
     * Look at the Q-value ranges - how do they differ with different activations?

**Key Insight**: Network architecture is a crucial hyperparameter in deep RL. There's always a trade-off between capacity (ability to represent complex functions) and efficiency (training speed, memory).

### Exercise 2: Understanding Experience Replay

Experience replay is crucial for DQN's success.

1. **Buffer Size Impact**:
   - Train agents with buffer sizes: 1000, 10000, 100000
   - Track training stability and final performance
   - **Questions**:
     * How does buffer size affect sample diversity?
     * What happens with a very small buffer (1000)?
     * Is there a point of diminishing returns?
     * How does buffer size affect memory usage?

2. **Batch Size Experiment**:
   - Try batch sizes: 16, 32, 64, 128
   - Compare training speed and stability
   - **Questions**:
     * How does batch size affect training stability?
     * What is the trade-off between batch size and training speed?
     * Why is batch training better than single-sample updates?
     * What happens if batch size is too large or too small?

3. **Remove Replay Buffer** (Advanced):
   - Modify the code to train without experience replay
   - Update immediately on each new experience (like tabular Q-learning)
   - **Questions**:
     * Does the agent still learn?
     * How stable is the training?
     * Why is experience replay so important for neural networks?
     * Compare the learning curve with and without replay

**Key Insight**: Experience replay breaks temporal correlation and improves sample efficiency. Without it, neural network-based RL becomes highly unstable.

### Exercise 3: Target Network Analysis

Target networks provide stability in DQN training.

1. **Update Frequency Experiment**:
   - Try target update frequencies: 1, 10, 50, 100 episodes
   - Observe training stability and convergence speed
   - **Questions**:
     * What happens with very frequent updates (every episode)?
     * What happens with very infrequent updates (every 100 episodes)?
     * What is the optimal update frequency for CartPole?
     * How does this relate to the "moving target" problem?

2. **Remove Target Network** (Advanced):
   - Modify code to use the same network for both current and target Q-values
   - **Questions**:
     * Does training become unstable?
     * Can the agent still learn at all?
     * Why do target networks help?
     * Compare loss curves with and without target network

3. **Soft Updates** (Very Advanced):
   - Instead of copying weights every N episodes, try soft updates:
     ```python
     tau = 0.001  # Soft update parameter
     for target_param, param in zip(target_network.parameters(), q_network.parameters()):
         target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
     ```
   - **Questions**:
     * How does this compare to hard updates?
     * What values of tau work well?
     * What are the advantages of soft updates?

**Key Insight**: Target networks prevent the "moving target" problem in Q-learning. They're essential for stable training with function approximation.

### Exercise 4: DQN vs Q-Learning Comparison

Reflect on the differences between Practical 2 and Practical 3.

1. **Conceptual Comparison**:
   Create a detailed table comparing:
   - State representation (discrete vs continuous)
   - Memory requirements (Q-table size vs network parameters)
   - Generalization capability
   - Training time
   - When each method is preferable
   - Ease of implementation
   - Interpretability

2. **Performance Analysis**:
   - **Question**: Why can't we use tabular Q-learning on CartPole?
     * Try to estimate the Q-table size needed
     * What precision would we lose with discretization?
   - **Question**: How does DQN generalize across similar states?
     * Look at your Q-value heatmap
     * Are transitions smooth or abrupt?
   - **Question**: What are the trade-offs of using neural networks?
     * Sample efficiency
     * Computational cost
     * Stability
     * Convergence guarantees

3. **Extension Question**:
   - Could we use DQN on FrozenLake (from Practical 2)?
   - What would be the advantages/disadvantages?
   - When is the added complexity of DQN worth it?
   - Would you expect DQN to perform better or worse than tabular Q-learning on FrozenLake?

**Key Insight**: DQN extends Q-learning to complex environments but adds computational cost. Choose your method based on your problem's state space.

### Exercise 5: Hyperparameter Sensitivity (Advanced)

Explore how sensitive DQN is to hyperparameters.

1. **Learning Rate**:
   - Try: 0.0001, 0.001, 0.01
   - **Questions**:
     * How does learning rate affect convergence speed?
     * Can the learning rate be too high or too low?
     * How does this relate to the neural network loss landscape?

2. **Discount Factor**:
   - Try: 0.9, 0.95, 0.99, 0.999
   - **Questions**:
     * For CartPole, which works best?
     * Why does CartPole need a high discount factor?
     * How would this change for a task with immediate rewards?

3. **Exploration Schedule**:
   - Try different epsilon decay rates: 0.99, 0.995, 0.999
   - Try different minimum epsilons: 0.01, 0.05, 0.1
   - **Questions**:
     * What happens if we stop exploring too quickly?
     * What happens if we explore too much?
     * How can you tell if exploration is the bottleneck?

**Key Insight**: DQN performance is sensitive to hyperparameters. In practice, hyperparameter tuning is crucial for success.

-----

## 9. Common Pitfalls and Debugging Tips

### Issue 1: Agent Not Learning (Reward stays low)

**Symptoms**: Rewards don't improve after many episodes, stay near random performance

**Possible causes and solutions**:

1. **Learning rate too high or too low**
   - Try: 0.0001, 0.001 (default), 0.01
   - Check loss values - if exploding, reduce learning rate

2. **Network too small**
   - Try larger hidden layers: 256 or 512 neurons
   - Add more layers

3. **Not enough exploration**
   - Slow down epsilon decay
   - Increase minimum epsilon

4. **Replay buffer too small**
   - Try 50,000 or 100,000 capacity
   - Make sure buffer fills before training starts

5. **Target network updated too frequently**
   - Try updating every 100 steps instead of 10

**Debugging steps**:
```python
# Add print statements
print(f"Loss: {loss:.4f}, Q-values: {q_values.mean():.4f}, Epsilon: {epsilon:.4f}")

# Plot Q-values over time
plt.plot(q_value_history)
plt.title("Average Q-value over training")
```

### Issue 2: Training is Unstable (Reward fluctuates wildly)

**Symptoms**: Performance improves then suddenly drops, very high variance

**Possible causes and solutions**:

1. **Learning rate too high**
   - Reduce to 0.0001 or 0.0005
   - Use gradient clipping: `nn.utils.clip_grad_norm_(network.parameters(), 10)`

2. **Target network updated too frequently**
   - Update every 50-100 episodes instead

3. **Batch size too small**
   - Try 64 or 128 instead of 32

4. **No experience replay**
   - Make sure replay buffer is being used

**Debugging**:
```python
# Monitor loss and Q-value statistics
print(f"Loss: {loss:.4f}")
print(f"Q-values mean: {q_values.mean():.4f}, std: {q_values.std():.4f}")
print(f"Target Q mean: {target_q.mean():.4f}")
```

### Issue 3: Out of Memory Errors

**Symptoms**: Program crashes with memory error

**Possible causes and solutions**:

1. **Replay buffer too large**
   - Reduce capacity to 10,000

2. **Batch size too large**
   - Reduce to 32 or even 16

3. **Not using `with torch.no_grad()` during inference**
   ```python
   # Correct:
   with torch.no_grad():
       q_values = network(state)

   # Wrong (stores gradients unnecessarily):
   q_values = network(state)
   ```

4. **Storing too much training history**
   - Don't store all Q-values, just plot periodically

### Issue 4: Code Runs But Results Don't Match Expected

**Debugging checklist**:

1. **Check tensor shapes**:
   ```python
   print(f"State shape: {states.shape}")
   print(f"Q-values shape: {q_values.shape}")
   print(f"Actions shape: {actions.shape}")
   ```

2. **Verify Q-network output**:
   ```python
   test_state = torch.randn(1, 4)  # Random state
   output = q_network(test_state)
   print(f"Output shape: {output.shape}, values: {output}")
   ```

3. **Check if target network is updating**:
   ```python
   # Before update
   old_weight = target_network.fc1.weight.data.clone()

   # After update
   target_network.load_state_dict(q_network.state_dict())
   new_weight = target_network.fc1.weight.data

   print(f"Weights changed: {not torch.equal(old_weight, new_weight)}")
   ```

4. **Verify epsilon decay**:
   ```python
   print(f"Epsilon at episode {episode}: {agent.epsilon}")
   ```

-----

## 10. Next Steps and Advanced Topics

Congratulations on implementing DQN! You now understand one of the most important algorithms in deep reinforcement learning.

### What You've Learned

- ✅ Why tabular methods fail with large state spaces
- ✅ How neural networks approximate Q-functions
- ✅ The role of experience replay in breaking correlations
- ✅ Why target networks stabilize training
- ✅ How to implement DQN from scratch in PyTorch
- ✅ How to extend DQN to visual domains (Atari)

### Extensions and Advanced Topics

If you want to go deeper, explore these improvements to DQN:

1. **Double DQN** (2016)
   - Addresses overestimation bias in DQN
   - Uses Q-network for action selection, target network for evaluation
   - Paper: "Deep Reinforcement Learning with Double Q-learning"

2. **Dueling DQN** (2016)
   - Separates state value and advantage functions
   - Better for states where action choice doesn't matter much
   - Paper: "Dueling Network Architectures for Deep RL"

3. **Prioritized Experience Replay** (2016)
   - Sample important transitions more frequently
   - Speeds up learning significantly
   - Paper: "Prioritized Experience Replay"

4. **Rainbow DQN** (2017)
   - Combines 6 extensions including the above
   - State-of-the-art performance on Atari
   - Paper: "Rainbow: Combining Improvements in Deep RL"

5. **Policy Gradient Methods**
   - Different approach: directly optimize policy instead of Q-values
   - Algorithms: A3C, PPO, SAC
   - Better for continuous action spaces

### Applying DQN to Your Own Problems

To use DQN on a new environment:

1. **Define state and action spaces**
   - What information does the agent need?
   - What actions can it take?

2. **Choose network architecture**
   - Vector inputs → Fully connected layers
   - Image inputs → Convolutional layers
   - Sequential inputs → Recurrent layers (LSTM/GRU)

3. **Set hyperparameters**
   - Start with our CartPole values
   - Adjust learning rate, buffer size, batch size as needed

4. **Monitor training**
   - Plot rewards and losses
   - Visualize Q-values
   - Watch agent behavior

5. **Debug and iterate**
   - If not learning, try smaller network first
   - If unstable, reduce learning rate
   - If forgetting, check replay buffer size

-----

## 11. Submission Instructions 📝

For your work to be graded, please follow these instructions carefully.

### Part 1: Code Implementation

1. **Create or Update Your Repository**
   - Use the same repository from Practicals 1 & 2, or create a new one
   - Create a `practical3` folder

2. **Required Files**:
   - `dqn_cartpole.py` - Your complete CartPole DQN implementation
   - `dqn_atari.py` - Your Atari DQN implementation (if attempted)
   - Any modified versions for exercises

3. **Required Outputs**:
   - Training plots showing learning curves
   - Q-value heatmap visualization
   - Screenshots of final test results

### Part 2: Written Report

Update your `README.md` with a new section for Practical 3:

#### Required Sections:

1. **Implementation Summary** (1-2 paragraphs)
   - Brief description of your DQN implementation
   - What worked well, what was challenging

2. **Performance Results**
   - Final average reward on CartPole (must be >195)
   - Number of episodes to solve
   - Training time
   - Comparison to random agent baseline

3. **Exercise Responses**
   - **Exercise 1 (Architecture)**: Your findings about network size, depth, and activations
   - **Exercise 2 (Replay Buffer)**: How buffer size and batch size affected performance
   - **Exercise 3 (Target Network)**: Impact of update frequency
   - **Exercise 4 (Comparison)**: DQN vs tabular Q-learning analysis

4. **Key Insights** (minimum 3-4 points)
   - What surprised you most about DQN?
   - How does function approximation change RL?
   - Why is DQN more complex than Q-learning?
   - What's the biggest challenge in deep RL?

5. **Challenges and Solutions**
   - What was the hardest part of this practical?
   - What bugs did you encounter?
   - How did you debug them?

6. **Atari Extension** (if attempted)
   - Which game did you try?
   - What performance did you achieve?
   - How does it compare to CartPole?

### Part 3: Performance Requirements

Your submission must demonstrate:
- ✅ CartPole DQN achieving average reward >195
- ✅ Clear learning curves showing improvement
- ✅ Proper implementation of all three DQN components:
  - Neural network Q-function approximator
  - Experience replay buffer
  - Target network
- ✅ At least 3 exercises completed with thoughtful analysis

### Part 4: Submit

1. **Make your repository public**

2. **Submit the URL** on the course Google Sheet

3. **Include in README**:
   - Link to your training plots
   - Link to code files
   - Your test results

### Grading Criteria

- **Implementation (40%)**:
  - Code runs without errors
  - Achieves performance target (>195 on CartPole)
  - Proper use of PyTorch
  - Clean, well-commented code

- **Exercises (30%)**:
  - Completed at least 3 exercises
  - Thoughtful analysis and insights
  - Evidence of experimentation

- **Understanding (20%)**:
  - Can explain DQN components
  - Insightful comparison to Q-learning
  - Identifies key challenges

- **Documentation (10%)**:
  - Clear README
  - Good visualizations
  - Complete submission

-----

## 12. References and Further Reading

### Original Papers

- **DQN (2015)**: Mnih et al., "Human-level control through deep reinforcement learning", Nature
  - Link: https://www.nature.com/articles/nature14236
  - The paper that started the deep RL revolution

- **Double DQN (2016)**: van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning"
  - Addresses overestimation bias

- **Dueling DQN (2016)**: Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning"
  - Separates state value and advantages

- **Prioritized Replay (2016)**: Schaul et al., "Prioritized Experience Replay"
  - Makes experience replay more efficient

- **Rainbow (2017)**: Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning"
  - Combines multiple DQN improvements

### Tutorials and Resources

- **PyTorch DQN Tutorial**: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- **OpenAI Spinning Up**: https://spinningup.openai.com/en/latest/
- **DeepMind's DQN Code**: https://github.com/deepmind/dqn
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/ (library with DQN implementation)

### Textbooks

- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (2nd edition)
  - Chapter 9-10: Prediction and Control with Function Approximation
  - Free online: http://incompleteideas.net/book/the-book-2nd.html

- **Graesser & Keng**: "Foundations of Deep Reinforcement Learning"
  - Practical implementation guide

### Related Course Materials

- Practical 1: `practicals/practical1.md` - Gymnasium basics
- Practical 2: `practicals/practical2.md` - Tabular Q-learning
- Study Guide: `studyguide/midtermguide.md` - RL fundamentals

### Environments

- **Gymnasium Documentation**: https://gymnasium.farama.org/
- **CartPole**: https://gymnasium.farama.org/environments/classic_control/cart_pole/
- **Atari**: https://gymnasium.farama.org/environments/atari/

-----

## Appendix: Complete Code Listing

The complete, runnable code is available in:
- `practical3/dqn_cartpole.py` - Full CartPole implementation
- `practical3/dqn_atari.py` - Full Atari implementation

Both files are standalone and can be run directly:

```bash
# Run CartPole DQN
cd practical3
python3 dqn_cartpole.py

# Run Atari DQN (takes longer)
python3 dqn_atari.py
```

Expected runtime:
- CartPole: 5-10 minutes to solve
- Atari: 30+ minutes to show improvement

**Happy Learning!** 🎉

If you have questions or run into issues, refer back to the debugging section or consult the references above.
