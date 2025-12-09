"""
Autoregressive Image Generation Model
Simplified version for educational purposes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelCNN(nn.Module):
    """Simplified PixelCNN for autoregressive image generation"""
    def __init__(self, in_channels=1, hidden_dim=64, n_classes=256):
        super(PixelCNN, self).__init__()
        
        # Masked convolutions for autoregressive property
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Simple forward pass (not fully autoregressive for simplicity)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class SimplePolicyNetwork(nn.Module):
    """Policy network for GRPO"""
    def __init__(self, state_dim=128, action_dim=256):
        super(SimplePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class AR_GRPO_Model:
    """Autoregressive model with GRPO training"""
    def __init__(self, img_size=28):
        self.img_size = img_size
        self.pixel_cnn = PixelCNN()
        self.policy_net = SimplePolicyNetwork()
        
        self.pixel_optimizer = torch.optim.Adam(self.pixel_cnn.parameters(), lr=0.001)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0001)
        
    def generate_image(self, noise=None):
        """Generate image autoregressively"""
        if noise is None:
            noise = torch.randn(1, 1, self.img_size, self.img_size)
        
        with torch.no_grad():
            logits = self.pixel_cnn(noise)
            probs = F.softmax(logits, dim=1)
            # Sample from distribution
            pixels = torch.argmax(probs, dim=1)
        
        return pixels.float() / 255.0
    
    def compute_reward(self, generated_img, target_img=None):
        """Simple reward based on image quality"""
        if target_img is None:
            # Reward for diversity and structure
            variance = generated_img.var()
            mean_intensity = generated_img.mean()
            reward = variance + (0.5 - abs(mean_intensity - 0.5))
        else:
            # Reward based on similarity to target
            mse = F.mse_loss(generated_img, target_img)
            reward = -mse.item()
        
        return reward
    
    def train_step(self, batch):
        """Single training step"""
        self.pixel_optimizer.zero_grad()
        
        # Forward pass through PixelCNN
        logits = self.pixel_cnn(batch)
        
        # Compute loss (simplified cross-entropy)
        target = (batch * 255).long().squeeze(1)
        loss = F.cross_entropy(logits, target)
        
        loss.backward()
        self.pixel_optimizer.step()
        
        return loss.item()
    
    def save(self, path):
        torch.save({
            'pixel_cnn': self.pixel_cnn.state_dict(),
            'policy_net': self.policy_net.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.pixel_cnn.load_state_dict(checkpoint['pixel_cnn'])
        self.policy_net.load_state_dict(checkpoint['policy_net'])
