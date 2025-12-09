import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):
    """
    Deep Q-Network

    Takes state as input and outputs Q-values for each action
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dim: Size of hidden layers
        """
        super(DQNetwork, self).__init__()

        # Three-layer network as in original DQN paper
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: State tensor

        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on output (Q-values can be any real number)
        return x

    def get_action(self, state):
        """
        Get greedy action for given state

        Args:
            state: Current state (numpy array)

        Returns:
            Action with highest Q-value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state_tensor)
        return q_values.argmax().item()


# Test the network
if __name__ == "__main__":
    # Create network
    state_dim = 4  # CartPole has 4 state variables
    action_dim = 2  # CartPole has 2 actions (left, right)

    network = DQNetwork(state_dim, action_dim)
    print(network)

    # Test forward pass
    dummy_state = torch.randn(1, state_dim)
    q_values = network(dummy_state)
    print(f"\nInput state shape: {dummy_state.shape}")
    print(f"Output Q-values: {q_values}")
    print(f"Selected action: {q_values.argmax().item()}")
