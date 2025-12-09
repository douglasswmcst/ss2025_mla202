# MLA202 Reinforcement Learning - Guided Take-Home Practical Test
## 24-Hour Assessment (10 marks)

**Submission Deadline**: Wednesday, 12:00 PM
**Format**: Take-home practical examination
**Submission**: Submit Jupyter notebook + Python files via GITHUB

---

## Important Instructions

1. **Individual Work**: This is an individual assessment. No collaboration allowed.
2. **Time Limit**: 24 hours from release to submission
3. **Submission Format**:
   - Main Jupyter notebook: `studentID_practical_test.ipynb`
   - Supporting Python files (if any): `studentID_*.py`
   - README: `studentID_README.txt` (explaining how to run)
4. **Environment**: Python 3.8+, numpy, torch, gym, matplotlib
5. **Academic Integrity**: Your code will be checked for plagiarism

---

## Assessment Structure

The test consists of 3 problems with increasing difficulty:

- **Problem 1**: Q-Learning Implementation (3 marks)
- **Problem 2**: Deep Q-Network Completion (4 marks)
- **Problem 3**: Analysis and Debugging (3 marks)

**Total**: 10 marks

---

## Problem 1: Q-Learning on GridWorld (3 marks)

### Task Description

Implement Q-learning to solve a simple 5×5 GridWorld environment.

### GridWorld Specification

```
S . . . .
. # . # .
. . . # .
. # . . .
. . . . G

S = Start (0,0)
G = Goal (4,4)
# = Wall (cannot pass)
. = Empty cell
```

**State Space**: 21 valid positions (25 - 4 walls)
**Action Space**: 4 actions {UP, DOWN, LEFT, RIGHT}
**Rewards**:
- Reach goal: +10
- Hit wall: -1 (stay in place)
- Each step: -0.1 (encourage efficiency)

### Implementation Requirements

**Part A (1.5 marks)**: Implement the GridWorld environment

Complete the following class:

```python
import numpy as np

class GridWorld:
    def __init__(self):
        self.grid_size = 5
        self.walls = [(1,1), (1,3), (2,3), (3,1)]
        self.start = (0, 0)
        self.goal = (4, 4)
        self.current_pos = self.start

    def reset(self):
        """Reset to start position. Return initial state."""
        # TODO: Implement this
        pass

    def step(self, action):
        """
        Take action and return (next_state, reward, done)
        Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        # TODO: Implement this
        pass

    def state_to_index(self, state):
        """Convert (row, col) to unique state index"""
        # TODO: Implement this
        pass

    def index_to_state(self, index):
        """Convert state index back to (row, col)"""
        # TODO: Implement this
        pass
```

**Part B (1.5 marks)**: Implement Q-learning algorithm

Complete the following function:

```python
def train_qlearning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Train Q-learning agent

    Args:
        env: GridWorld environment
        episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate

    Returns:
        Q: Q-table (numpy array)
        rewards: List of total rewards per episode
    """
    n_states = 21  # Valid positions
    n_actions = 4

    # TODO: Initialize Q-table

    # TODO: Training loop for each episode
    #   - Reset environment
    #   - For each step:
    #     - Choose action (epsilon-greedy)
    #     - Take action
    #     - Update Q-value using Q-learning rule:
    #       Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
    #   - Track total reward

    # TODO: Return Q-table and rewards
    pass
```

**Testing**: Your implementation should:
- Converge to optimal policy within 1000 episodes
- Achieve average reward > 8.0 in final 100 episodes
- Find shortest path from start to goal

**Deliverables**:
- Completed GridWorld class
- Completed train_qlearning function
- Plot of rewards over episodes
- Visualization of learned policy (arrows showing best action per state)

---

## Problem 2: Deep Q-Network Completion (4 marks)

### Task Description

You are given a partially implemented DQN for CartPole-v1. Your task is to complete the missing parts and get it working.

### Provided Code (with gaps)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # TODO: Define network architecture
        # Requirements:
        # - Input: state_dim
        # - Hidden layers: 2 layers of size 128 with ReLU
        # - Output: action_dim (Q-values)
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # TODO: Implement random sampling
        # Return: (states, actions, rewards, next_states, dones) as numpy arrays
        pass

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # TODO: Initialize Q-network and target network
        # TODO: Initialize optimizer (Adam, lr=0.001)
        # TODO: Initialize replay buffer (capacity=10000)

        pass

    def select_action(self, state, training=True):
        # TODO: Implement epsilon-greedy action selection
        pass

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return 0

        # TODO: Sample batch from replay buffer
        # TODO: Compute current Q-values
        # TODO: Compute target Q-values using target network
        # TODO: Compute loss (MSE between current and target Q)
        # TODO: Backpropagate and update network
        # TODO: Decay epsilon

        pass

    def update_target_network(self):
        # TODO: Copy weights from Q-network to target network
        pass

# Training function
def train_cartpole(episodes=500):
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_dim=4, action_dim=2)

    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # TODO: Select action
            # TODO: Take step in environment
            # TODO: Store transition in replay buffer
            # TODO: Train agent
            # TODO: Update state and reward

            pass

        # TODO: Update target network every 10 episodes
        # TODO: Track rewards

        pass

    return agent, rewards

# TODO: Run training and plot results
```

### Requirements (4 marks breakdown)

1. **Network Architecture (1 mark)**:
   - Correct DQN class with 2 hidden layers
   - Proper forward pass

2. **Replay Buffer (0.5 marks)**:
   - Correct sampling implementation

3. **Training Loop (2 marks)**:
   - Correct DQN update rule
   - Target network usage
   - Epsilon-greedy exploration

4. **Results (0.5 marks)**:
   - Agent achieves average reward > 400 over 100 episodes
   - Plot showing learning progress

**Testing**: Run for 500 episodes. Should solve CartPole (reward > 475) within 400 episodes.

**Deliverables**:
- Completed code with all TODOs filled
- Training plot (rewards vs episodes)
- Final performance statistics

---

## Problem 3: Analysis and Debugging (3 marks)

### Task Description

You are given a buggy implementation of Q-learning for FrozenLake. The agent is not learning effectively. Your task is to identify and fix the issues.

### Buggy Code

```python
import gym
import numpy as np

def buggy_train_frozenlake(episodes=10000):
    env = gym.make('FrozenLake-v1', is_slippery=True)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Bug 1: Something wrong here
    Q = np.zeros((n_states, n_actions))

    alpha = 0.8  # Learning rate
    gamma = 0.95  # Discount factor
    epsilon = 0.3  # Exploration rate

    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Bug 2: Action selection issue
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmin(Q[state])  # BUG HERE?

            next_state, reward, done, _ = env.step(action)

            # Bug 3: Update rule issue
            Q[state, action] = reward + gamma * np.max(Q[next_state])

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        # Bug 4: Epsilon handling
        # No epsilon decay

    success_rate = np.mean(rewards[-100:])
    print(f"Success rate: {success_rate}")

    return Q, rewards

# Run
Q, rewards = buggy_train_frozenlake()
```

### Your Tasks (3 marks)

**Part A (1.5 marks)**: Identify and Describe Bugs

For each bug (there are 4):
1. Describe what the bug is
2. Explain why it prevents learning
3. Provide the fix

Write your answers in this format:

```
Bug 1: [Line number and description]
Impact: [How this affects learning]
Fix: [Corrected code]

Bug 2: ...
```

**Part B (1.5 marks)**: Implement Fixes and Demonstrate Improvement

- Provide corrected code
- Run for 10,000 episodes
- Show that success rate improves to > 0.70
- Plot comparison: buggy vs fixed implementation

**Hint**: Bugs are in:
1. Q-table initialization or dimensions
2. Action selection (argmin vs argmax?)
3. Q-learning update rule (missing something?)
4. Exploration strategy (epsilon decay?)

**Deliverables**:
- Written analysis of each bug
- Corrected code
- Comparative plot showing improvement
- Final success rate statistics

---

## Submission Checklist

Before submitting, ensure you have:

- [ ] Problem 1: Completed GridWorld + Q-learning
- [ ] Problem 1: Rewards plot and policy visualization
- [ ] Problem 2: Completed DQN implementation
- [ ] Problem 2: Training results and plot
- [ ] Problem 3: Bug analysis written
- [ ] Problem 3: Fixed code and comparison plot
- [ ] All code runs without errors
- [ ] Jupyter notebook with markdown explanations
- [ ] studentID_README.txt with instructions
- [ ] File naming follows format: studentID_*

---

## Grading Rubric

### Problem 1 (3 marks)
- **GridWorld implementation (1 mark)**: Correct state/action handling, reward logic
- **Q-learning algorithm (1.5 marks)**: Correct update rule, epsilon-greedy, convergence
- **Visualization (0.5 marks)**: Clear plots and policy display

### Problem 2 (4 marks)
- **Network architecture (1 mark)**: Correct DQN structure
- **Training components (2 marks)**: Replay buffer, target network, updates
- **Performance (0.5 marks)**: Achieves target performance
- **Code quality (0.5 marks)**: Clean, commented, runs properly

### Problem 3 (3 marks)
- **Bug identification (1 mark)**: Correctly identifies all 4 bugs
- **Analysis (0.5 marks)**: Clear explanation of impacts
- **Fix implementation (1 mark)**: Correct fixes, code works
- **Demonstration (0.5 marks)**: Shows improvement with plots

---

## Submission

Submit via Github before Wednesday 12:00 PM:
1. Main notebook: `studentID_practical_test.ipynb`
2. Supporting files: `studentID_*.py` (if any)
3. README: `studentID_README.txt`

**Late submissions**: 10% penalty per hour, maximum 24 hours late.

---

Good luck! This test assesses your practical RL skills developed throughout the semester. Trust your preparation! 🎯
