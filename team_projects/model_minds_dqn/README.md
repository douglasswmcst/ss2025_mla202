# Deep Q-Network (DQN) Implementation

Classic DQN algorithm implementation based on the Nature 2015 paper.

## Team: Model Minds

## Features
- Experience Replay Buffer
- Target Network
- Epsilon-greedy exploration
- Training on CartPole-v1

## Installation
```bash
pip install torch gym numpy matplotlib
```

## Usage
```bash
# Train agent
python train.py

# Test network
python network.py

# Test replay buffer
python replay_buffer.py
```

## Key Results
- Solves CartPole-v1 in ~300 episodes
- Achieves average reward of 475+ over 100 episodes
- Demonstrates importance of experience replay and target networks

## References
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
