# The Innovators - FlapAI Bird (Flappy Bird with RL)
## Combined Term Paper & Mini-Project Implementation Guide

**Team Members**: 3 students
**Research Paper**: [FlapAI Bird: Training an Agent to Play Flappy Bird Using RL](https://arxiv.org/abs/2003.09579)
**Submission Deadline**: Day 6 (Saturday, 11:59 PM)

---

## Overview

Implement DQN or Policy Gradient methods to train an agent to play Flappy Bird. This classic RL application demonstrates continuous decision-making under time pressure.

---

## Part 1: Term Paper Report (10 marks - 1000-1500 words)

### Structure

**1. Introduction (2 marks)**
- Flappy Bird game mechanics and challenge
- Why this is a good RL testbed (simple rules, difficult execution)
- Problem statement: Train agent to maximize score

**2. Methodology (3 marks)**
- **State**: Bird y-position, velocity, pipe distances, pipe gaps
- **Actions**: Flap (jump) or do nothing
- **Rewards**: +1 for surviving, +10 for passing pipe, -100 for collision
- **Algorithm**: DQN with experience replay
- **Network Architecture**: State → FC layers → Q-values for 2 actions

**3. Findings (3 marks)**
- Learning curve (score over episodes)
- Survival time improvement
- Strategies learned (timing, height management)
- Comparison with human performance

**4. Organization & References (2 marks)**

---

## Part 2: Implementation (15 marks)

### Quick Setup

```bash
mkdir flappy_bird_rl
cd flappy_bird_rl
python -m venv venv
source venv/bin/activate
pip install numpy pygame torch matplotlib
```

### Simplified Implementation Using Flappy Bird Environment

### File: `flappy_game.py`

```python
import pygame
import numpy as np
import random

class FlappyBird:
    """Simplified Flappy Bird game for RL"""

    def __init__(self, screen_width=400, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.bird_x = 50
        self.pipe_width = 70
        self.pipe_gap = 200
        self.gravity = 1
        self.flap_strength = -10
        self.pipe_velocity = 5

        self.reset()

    def reset(self):
        """Reset game state"""
        self.bird_y = self.screen_height // 2
        self.bird_velocity = 0
        self.pipes = [self.create_pipe()]
        self.score = 0
        self.game_over = False
        return self.get_state()

    def create_pipe(self):
        """Create new pipe"""
        gap_y = random.randint(150, self.screen_height - 150)
        return {
            'x': self.screen_width,
            'gap_y': gap_y,
            'passed': False
        }

    def get_state(self):
        """Return state representation"""
        if not self.pipes:
            return np.array([self.bird_y, 0, 300, 300, self.bird_velocity])

        # Get next pipe
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_x:
                next_pipe = pipe
                break

        if next_pipe is None:
            next_pipe = self.pipes[0]

        # State: [bird_y, bird_velocity, horizontal_distance, vertical_distance_top, vertical_distance_bottom]
        horizontal_dist = next_pipe['x'] - self.bird_x
        vertical_dist_top = self.bird_y - (next_pipe['gap_y'] - self.pipe_gap // 2)
        vertical_dist_bottom = (next_pipe['gap_y'] + self.pipe_gap // 2) - self.bird_y

        state = np.array([
            self.bird_y / self.screen_height,  # Normalize
            self.bird_velocity / 10,
            horizontal_dist / self.screen_width,
            vertical_dist_top / self.screen_height,
            vertical_dist_bottom / self.screen_height
        ])

        return state

    def step(self, action):
        """
        Take action: 0 = do nothing, 1 = flap

        Returns: (state, reward, done, info)
        """
        # Apply action
        if action == 1:
            self.bird_velocity = self.flap_strength

        # Update bird position
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity

        # Update pipes
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_velocity

        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if p['x'] > -self.pipe_width]

        # Add new pipes
        if len(self.pipes) == 0 or self.pipes[-1]['x'] < self.screen_width - 200:
            self.pipes.append(self.create_pipe())

        # Check collisions
        reward = 0.1  # Small reward for surviving

        # Hit ground or ceiling
        if self.bird_y <= 0 or self.bird_y >= self.screen_height:
            self.game_over = True
            reward = -100
            return self.get_state(), reward, True, {'score': self.score}

        # Check pipe collision
        for pipe in self.pipes:
            if (self.bird_x + 34 > pipe['x'] and
                self.bird_x < pipe['x'] + self.pipe_width):
                # Bird is horizontally aligned with pipe
                gap_top = pipe['gap_y'] - self.pipe_gap // 2
                gap_bottom = pipe['gap_y'] + self.pipe_gap // 2

                if self.bird_y < gap_top or self.bird_y + 24 > gap_bottom:
                    self.game_over = True
                    reward = -100
                    return self.get_state(), reward, True, {'score': self.score}

                # Passed pipe
                if not pipe['passed'] and pipe['x'] + self.pipe_width < self.bird_x:
                    pipe['passed'] = True
                    self.score += 1
                    reward = 10

        return self.get_state(), reward, False, {'score': self.score}

    def render(self, screen=None):
        """Render game (optional)"""
        if screen is None:
            return

        screen.fill((135, 206, 235))  # Sky blue

        # Draw bird
        pygame.draw.rect(screen, (255, 255, 0), (self.bird_x, int(self.bird_y), 34, 24))

        # Draw pipes
        for pipe in self.pipes:
            gap_top = pipe['gap_y'] - self.pipe_gap // 2
            gap_bottom = pipe['gap_y'] + self.pipe_gap // 2

            # Top pipe
            pygame.draw.rect(screen, (0, 255, 0),
                           (pipe['x'], 0, self.pipe_width, gap_top))
            # Bottom pipe
            pygame.draw.rect(screen, (0, 255, 0),
                           (pipe['x'], gap_bottom, self.pipe_width,
                            self.screen_height - gap_bottom))

        # Draw score
        font = pygame.font.Font(None, 36)
        text = font.render(f'Score: {self.score}', True, (0, 0, 0))
        screen.blit(text, (10, 10))

        pygame.display.flip()
```

### File: `dqn_agent.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size=5, action_size=2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size=5, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])

        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, filename='flappy_agent.pth'):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename='flappy_agent.pth'):
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()
```

### File: `train.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from flappy_game import FlappyBird
from dqn_agent import DQNAgent

def train(episodes=1000, target_update_freq=10):
    game = FlappyBird()
    agent = DQNAgent(state_size=5, action_size=2)

    scores = []
    avg_scores = []
    max_score = 0

    print("Training Flappy Bird Agent...")
    print("-" * 60)

    for episode in range(episodes):
        state = game.reset()
        score = 0
        done = False

        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = game.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()

            state = next_state
            score = info['score']

        scores.append(score)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)

        if score > max_score:
            max_score = score
            agent.save('best_flappy_agent.pth')

        if (episode + 1) % target_update_freq == 0:
            agent.update_target_model()

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}: Score={score}, "
                  f"Avg(100)={avg_score:.2f}, Max={max_score}, "
                  f"Eps={agent.epsilon:.3f}")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.3, label='Score')
    plt.plot(avg_scores, linewidth=2, label='Avg Score (100)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(scores, bins=50, edgecolor='black')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('flappy_training_results.png', dpi=300)
    print(f"\nTraining complete! Best score: {max_score}")

if __name__ == "__main__":
    train(episodes=1000)
```

### File: `play.py` - Watch Trained Agent

```python
import pygame
from flappy_game import FlappyBird
from dqn_agent import DQNAgent
import time

def play_game(episodes=5):
    pygame.init()
    screen = pygame.display.set_mode((400, 600))
    pygame.display.set_caption('Flappy Bird RL')
    clock = pygame.time.Clock()

    game = FlappyBird()
    agent = DQNAgent()
    agent.load('best_flappy_agent.pth')
    agent.epsilon = 0  # No exploration

    for episode in range(episodes):
        state = game.reset()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.get_action(state, training=False)
            state, reward, done, info = game.step(action)

            game.render(screen)
            clock.tick(30)

        print(f"Episode {episode + 1}: Score = {info['score']}")
        time.sleep(1)

    pygame.quit()

if __name__ == "__main__":
    play_game(episodes=5)
```

---

## Deliverables Checklist

### Code
- [ ] `flappy_game.py` - Game environment
- [ ] `dqn_agent.py` - DQN agent
- [ ] `train.py` - Training script
- [ ] `play.py` - Visualization
- [ ] Training plots
- [ ] `README.md`

### Report (1000-1500 words)
- [ ] Flappy Bird challenge explanation
- [ ] DQN methodology
- [ ] Training results
- [ ] Analysis of learned behavior

### Video (5 minutes)
- [ ] Game demonstration
- [ ] Training explanation
- [ ] Trained agent playing
- [ ] Discussion

---

## Assessment (15 marks)

- **Code (5 marks)**: Working game, DQN, training
- **Functionality (4 marks)**: Agent learns, scores improve
- **Documentation (3 marks)**: Clear code, good report
- **Presentation (2 marks)**: Engaging video
- **Q&A (1 mark)**: Understanding

---

## Tips

1. **Start with easier version**: Reduce pipe speed initially
2. **Reward shaping**: Small positive rewards for staying alive
3. **State representation matters**: Include velocity and distances
4. **Be patient**: May take 500+ episodes to see good results
5. **Save best model**: Keep model with highest score

Good luck, The Innovators! 🐦
