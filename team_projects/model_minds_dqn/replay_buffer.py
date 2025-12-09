import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Experience Replay Buffer

    Stores transitions and samples random mini-batches for training
    This breaks correlations between consecutive samples and improves stability
    """

    def __init__(self, capacity=100000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample random mini-batch

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        # Randomly sample transitions
        batch = random.sample(self.buffer, batch_size)

        # Unpack transitions
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer)


# Test the replay buffer
if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=1000)

    # Add some dummy transitions
    for i in range(10):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = False

        buffer.push(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)}")

    # Sample a batch
    if len(buffer) >= 5:
        states, actions, rewards, next_states, dones = buffer.sample(5)
        print(f"\nSampled batch:")
        print(f"States shape: {states.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Rewards: {rewards}")
