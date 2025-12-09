"""
Training Script for AR-GRPO Image Generation
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import AR_GRPO_Model

def create_simple_dataset(n_samples=1000, img_size=28):
    """Create simple synthetic dataset"""
    data = []
    for _ in range(n_samples):
        # Create simple patterns: circles, lines, etc.
        img = torch.zeros(1, img_size, img_size)
        
        # Random pattern
        pattern_type = np.random.randint(0, 3)
        if pattern_type == 0:
            # Circle
            center = img_size // 2
            y, x = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing='ij')
            dist = ((x - center)**2 + (y - center)**2).sqrt()
            img[0] = (dist < img_size // 3).float()
        elif pattern_type == 1:
            # Horizontal lines
            img[0, ::4, :] = 1.0
        else:
            # Vertical lines
            img[0, :, ::4] = 1.0
        
        data.append(img)
    
    return torch.stack(data)

def train(epochs=100, batch_size=32):
    print("Training AR-GRPO Model...")
    print("-" * 60)
    
    model = AR_GRPO_Model(img_size=28)
    dataset = create_simple_dataset(n_samples=1000, img_size=28)
    
    losses = []
    rewards = []
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_rewards = []
        
        # Shuffle data
        indices = torch.randperm(len(dataset))
        
        for i in range(0, len(dataset), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = dataset[batch_indices]
            
            # Training step
            loss = model.train_step(batch)
            epoch_losses.append(loss)
            
            # Generate and evaluate
            if i % 100 == 0:
                generated = model.generate_image()
                reward = model.compute_reward(generated)
                epoch_rewards.append(reward)
        
        avg_loss = np.mean(epoch_losses)
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        
        losses.append(avg_loss)
        rewards.append(avg_reward)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}")
    
    print("-" * 60)
    print("Training complete!")
    
    # Save model
    model.save('ar_grpo_model.pth')
    print("Model saved to 'ar_grpo_model.pth'")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(rewards)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Average Reward')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ar_grpo_training.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'ar_grpo_training.png'")
    
    # Generate and save sample images
    print("\nGenerating sample images...")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img = model.generate_image()
        ax.imshow(img.squeeze().numpy(), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig('generated_samples.png', dpi=300, bbox_inches='tight')
    print("Samples saved to 'generated_samples.png'")

if __name__ == "__main__":
    train(epochs=100, batch_size=32)
