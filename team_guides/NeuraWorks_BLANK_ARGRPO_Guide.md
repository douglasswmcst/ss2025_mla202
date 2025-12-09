# NeuraWorks & BLANK Teams - AR-GRPO Image Generation
## Combined Term Paper & Mini-Project Implementation Guide

**Team**: NeuraWorks + 2 BLANK teams (Total: ~7-9 students across 3 teams)
**Research Paper**: [AR-GRPO: Training Autoregressive Image Generation Models via RL](https://arxiv.org/abs/2508.06924)
**Submission Deadline**: Day 6 (Saturday, 11:59 PM)

---

## ⚠️ IMPORTANT NOTE

This topic is **advanced** and involves autoregressive image generation with RL - very cutting-edge research! Given the one-week timeline, we recommend a **simplified demonstration approach** rather than full implementation.

---

## Two Implementation Options

### Option A: Simplified Demonstration (RECOMMENDED for 1 week)

Implement a simplified version using MNIST digit generation with policy gradients

### Option B: Literature Review + Conceptual Implementation

Focus on thorough paper analysis with conceptual code framework

---

## Part 1: Term Paper Report (10 marks - 1000-1500 words)

### Structure

**1. Introduction (2 marks)**
- What is autoregressive image generation?
- Why use RL instead of supervised learning?
- GRPO (Group Relative Policy Optimization) overview
- Your approach: Simplified demonstration on MNIST

**2. Methodology (3 marks)**

**Autoregressive Generation**:
- Generate image pixel-by-pixel or patch-by-patch
- Each step conditioned on previous steps
- Model: P(x₁, x₂, ..., xₙ) = ∏ P(xᵢ | x₁, ..., xᵢ₋₁)

**RL Framework**:
- State: Partially generated image
- Action: Next pixel/patch values
- Reward: Quality metrics (image coherence, class recognition)
- Policy: Neural network predicting next pixel distribution

**GRPO Concept** (from paper):
- Group-based policy optimization
- Reduces variance in policy gradients
- More sample-efficient than standard REINFORCE

**3. Findings (3 marks)**
- Results from simplified implementation
- Quality of generated images
- Comparison with random generation
- Challenges and limitations
- How this relates to the original paper

**4. Organization & References (2 marks)**

---

## Part 2: Implementation Option A - Simplified MNIST Generation (15 marks)

### Setup

```bash
mkdir ar_image_generation
cd ar_image_generation
python -m venv venv
source venv/bin/activate
pip install torch torchvision numpy matplotlib
```

### Simplified Approach

Instead of full AR-GRPO, implement:
1. Autoregressive MNIST generation (8×8 downsampled)
2. Policy gradient training
3. Reward based on classifier confidence

### File: `model.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoregressiveGenerator(nn.Module):
    """
    Simplified autoregressive image generator
    Generates 8x8 images pixel by pixel
    """

    def __init__(self, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(256, hidden_dim)  # Pixel value embedding
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.output = nn.Linear(hidden_dim, 256)  # Predict next pixel (0-255)

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len) - previous pixels
            hidden: LSTM hidden state

        Returns:
            logits: (batch, seq_len, 256) - pixel value probabilities
            hidden: updated hidden state
        """
        embedded = self.embedding(x)  # (batch, seq_len, hidden_dim)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.output(output)  # (batch, seq_len, 256)
        return logits, hidden

    def generate(self, batch_size=1, seq_len=64, temperature=1.0):
        """
        Generate images autoregressively

        Args:
            batch_size: Number of images to generate
            seq_len: Number of pixels (64 for 8x8 image)
            temperature: Sampling temperature

        Returns:
            images: (batch_size, seq_len) generated pixel values
        """
        device = next(self.parameters()).device
        images = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        hidden = None

        for t in range(seq_len):
            if t == 0:
                # Start token (black pixel)
                x = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            else:
                x = images[:, t-1:t]

            logits, hidden = self.forward(x, hidden)
            logits = logits[:, -1, :] / temperature  # (batch, 256)

            # Sample next pixel
            probs = F.softmax(logits, dim=-1)
            next_pixel = torch.multinomial(probs, 1).squeeze(-1)
            images[:, t] = next_pixel

        return images


class SimpleCNN Classifier(nn.Module):
    """Simple classifier for reward computation"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### File: `train.py`

```python
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from model import AutoregressiveGenerator, SimpleCNNClassifier

def train_classifier():
    """Train a simple classifier for reward computation"""
    transform = transforms.Compose([
        transforms.Resize(8),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    classifier = SimpleCNNClassifier()
    optimizer = torch.optim.Adam(classifier.parameters())

    print("Training classifier...")
    for epoch in range(3):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    torch.save(classifier.state_dict(), 'classifier.pth')
    return classifier

def compute_reward(images, target_class, classifier):
    """
    Compute reward for generated images

    Args:
        images: (batch, 64) pixel values
        target_class: Target digit class
        classifier: Trained classifier

    Returns:
        rewards: (batch,) reward values
    """
    # Reshape to 8x8 and normalize
    imgs = images.float().view(-1, 1, 8, 8) / 255.0

    with torch.no_grad():
        logits = classifier(imgs)
        probs = F.softmax(logits, dim=-1)
        target_probs = probs[:, target_class]

    # Reward is confidence in target class
    rewards = target_probs

    return rewards

def train_generator(epochs=100, batch_size=16):
    """Train autoregressive generator with policy gradients"""

    # Load or train classifier
    try:
        classifier = SimpleCNNClassifier()
        classifier.load_state_dict(torch.load('classifier.pth'))
        classifier.eval()
    except:
        classifier = train_classifier()
        classifier.eval()

    # Create generator
    generator = AutoregressiveGenerator(hidden_dim=128)
    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

    target_digit = 7  # Try to generate digit 7

    rewards_history = []

    print(f"\nTraining generator to create digit {target_digit}...")
    print("-" * 60)

    for epoch in range(epochs):
        # Generate images
        images = generator.generate(batch_size=batch_size, seq_len=64, temperature=1.0)

        # Compute rewards
        rewards = compute_reward(images, target_digit, classifier)
        rewards_history.append(rewards.mean().item())

        # Compute loss (REINFORCE algorithm)
        # We need log probabilities of generated sequence
        logits, _ = generator(images[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs of chosen actions
        chosen_log_probs = log_probs.gather(2, images[:, 1:].unsqueeze(-1)).squeeze(-1)
        chosen_log_probs = chosen_log_probs.sum(dim=1)  # Sum over sequence

        # Policy gradient loss
        loss = -(chosen_log_probs * rewards).mean()

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Reward = {rewards.mean().item():.4f}, "
                  f"Loss = {loss.item():.4f}")

            # Visualize samples
            if (epoch + 1) % 50 == 0:
                visualize_samples(generator, epoch + 1, target_digit)

    # Save model
    torch.save(generator.state_dict(), 'generator.pth')

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.savefig('training_progress.png')
    print("\nTraining complete!")

def visualize_samples(generator, epoch, target_digit):
    """Visualize generated samples"""
    with torch.no_grad():
        images = generator.generate(batch_size=16, seq_len=64, temperature=0.8)
        images = images.cpu().numpy().reshape(-1, 8, 8)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(f'Epoch {epoch} - Generating Digit {target_digit}')

    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx], cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'samples_epoch_{epoch}.png')
    plt.close()

if __name__ == "__main__":
    train_generator(epochs=100, batch_size=16)
```

---

## Part 2: Implementation Option B - Conceptual Framework (15 marks)

If full implementation is too complex, create:

1. **Detailed Literature Review** (5 pages)
   - Deep dive into AR-GRPO paper
   - Comparison with other image generation methods (GANs, VAEs, Diffusion)
   - RL formulation for image generation

2. **Conceptual Code Framework** (Python pseudocode)
   - Architecture diagrams
   - Algorithm pseudocode
   - Data flow diagrams

3. **Small-Scale Demonstration**
   - Simple autoregressive model (even 4×4 images)
   - Show the concept works
   - Discuss scaling challenges

---

## Deliverables Checklist

### For All Teams

**Code** (Choose Option A or B):
- [ ] Option A: Working autoregressive generator + training
- [ ] Option B: Conceptual framework + small demo
- [ ] README with clear explanation
- [ ] Training/results visualizations

**Report** (1000-1500 words):
- [ ] Introduction to autoregressive generation
- [ ] RL formulation explanation
- [ ] GRPO concepts from paper
- [ ] Implementation approach and results
- [ ] Limitations and future work discussion

**Video** (5 minutes):
- [ ] Problem explanation
- [ ] Approach overview
- [ ] Demo/results
- [ ] Connection to original paper

---

## Assessment (15 marks)

- **Code (5 marks)**: Working implementation OR thorough conceptual framework
- **Functionality (4 marks)**: Demonstrates understanding of autoregressive + RL
- **Documentation (3 marks)**: Clear explanation of complex topic
- **Presentation (2 marks)**: Good video given complexity
- **Q&A (1 mark)**: Deep understanding of concepts

---

## Team Coordination

Since you have 3 teams working on the same topic:

**Division of Labor**:
- **NeuraWorks**: Core generator implementation
- **BLANK Team 1**: Classifier training + reward computation
- **BLANK Team 2**: Evaluation metrics + visualization

**Collaboration**:
- Shared GitHub repository
- Weekly sync meetings
- Shared report with individual contributions noted

---

## Tips

1. **This is hard!**: Don't expect state-of-the-art results in 1 week
2. **Focus on concepts**: Understanding > perfect implementation
3. **Use small images**: 8×8 trains much faster than 28×28
4. **Simplify reward**: Classifier confidence is simpler than full GRPO
5. **Document well**: Explain what you tried and why
6. **Be honest**: Discuss limitations openly

---

## Alternative: Switch Topics?

If this proves too difficult, consider switching to a simpler application:
- Reinforcement Learning for game playing (easier)
- Basic image classification with RL
- Any of the other team topics

**Contact tutors immediately if you want to switch!**

---

Good luck, NeuraWorks & BLANK Teams! This is challenging but you've got this! 🧠🎨
