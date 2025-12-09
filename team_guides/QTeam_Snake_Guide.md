# Q-Team - Snake Game with Reinforcement Learning
## Combined Term Paper & Mini-Project Implementation Guide

**Team Members**: 3 students
**Research Paper**: [Exploration of Reinforcement Learning to SNAKE](https://cs229.stanford.edu/proj2016spr/report/060.pdf)
**Submission Deadline**: Day 6 (Saturday, 11:59 PM)

---

## Overview

Implement Deep Q-Learning or Policy Gradient methods to train an agent to play the classic Snake game. This demonstrates RL in a dynamic environment with sparse rewards and self-collision challenges.

---

## Part 1: Term Paper Report (10 marks - 1000-1500 words)

### Structure

**1. Introduction (2 marks)**
- Snake game mechanics and challenges
- Why RL is suitable (delayed rewards, strategy learning)
- Problem: Maximize score while avoiding collisions

**2. Methodology (3 marks)**
- **State**: Snake head position, food position, danger indicators (4 directions), snake length
- **Actions**: 4 directions (up, down, left, right) or 3 relative (straight, left, right)
- **Rewards**: +10 for food, -10 for death, -0.01 per step (encourage efficiency)
- **Algorithm**: DQN with experience replay
- **Network**: State → Hidden layers → Q-values for actions

**3. Findings (3 marks)**
- Learning progression (score over episodes)
- Survival time improvement
- Strategies learned (food seeking, self-avoidance)
- Challenges encountered

**4. Organization & References (2 marks)**

---

## Part 2: Implementation (15 marks)

### Quick Setup

```bash
mkdir snake_rl
cd snake_rl
python -m venv venv
source venv/bin/activate
pip install numpy pygame torch matplotlib
```

### File Structure
```
snake_rl/
├── snake_game.py    # Game environment
├── agent.py         # DQN agent
├── train.py         # Training
├── play.py          # Play with trained agent
└── README.md
```

### Implementation: `snake_game.py`

```python
import numpy as np
import pygame
from enum import Enum
from collections import namedtuple

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGame:
    def __init__(self, w=640, h=480, block_size=20):
        self.w = w
        self.h = h
        self.block_size = block_size
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head,
                      Point(self.head.x - self.block_size, self.head.y),
                      Point(self.head.x - 2 * self.block_size, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self.get_state()

    def _place_food(self):
        x = np.random.randint(0, (self.w - self.block_size) // self.block_size) * self.block_size
        y = np.random.randint(0, (self.h - self.block_size) // self.block_size) * self.block_size
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def get_state(self):
        """Return state as numpy array"""
        head = self.snake[0]

        # Points around head
        point_l = Point(head.x - self.block_size, head.y)
        point_r = Point(head.x + self.block_size, head.y)
        point_u = Point(head.x, head.y - self.block_size)
        point_d = Point(head.x, head.y + self.block_size)

        # Current direction
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # State: 11 boolean values
        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l, dir_r, dir_u, dir_d,

            # Food location
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y   # food down
        ]

        return np.array(state, dtype=int)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def step(self, action):
        """
        Action: [1,0,0] = straight, [0,1,0] = right turn, [0,0,1] = left turn
        Returns: state, reward, done, score
        """
        self.frame_iteration += 1

        # Move
        self._move(action)
        self.snake.insert(0, self.head)

        # Check game over
        reward = 0
        done = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            return self.get_state(), reward, done, self.score

        # Check food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        return self.get_state(), reward, done, self.score

    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)
```

### Implementation: `agent.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class DQNAgent:
    def __init__(self, state_size=11, action_size=3, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate

        self.memory = deque(maxlen=100000)
        self.model = DQN(state_size, 256, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def train_step(self, batch_size=1000):
        if len(self.memory) < batch_size:
            return 0

        minibatch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])

        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, filename='snake_model.pth'):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename='snake_model.pth'):
        self.model.load_state_dict(torch.load(filename))
```

### Implementation: `train.py`

```python
import matplotlib.pyplot as plt
from snake_game import SnakeGame
from agent import DQNAgent
import numpy as np

def train(episodes=500):
    game = SnakeGame(w=200, h=200, block_size=10)
    agent = DQNAgent(state_size=11, action_size=3)

    scores = []
    mean_scores = []

    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        done = False

        while not done:
            action_idx = agent.get_action(state, training=True)
            action = [0, 0, 0]
            action[action_idx] = 1

            next_state, reward, done, score = game.step(action)
            agent.remember(state, action_idx, reward, next_state, done)

            state = next_state
            total_reward += reward

        loss = agent.train_step()

        scores.append(game.score)
        mean_score = np.mean(scores[-100:])
        mean_scores.append(mean_score)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: Score={game.score}, "
                  f"Mean(100)={mean_score:.2f}, Eps={agent.epsilon:.3f}")

    agent.save('snake_agent.pth')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(scores, alpha=0.6, label='Score')
    plt.plot(mean_scores, linewidth=2, label='Mean Score (100)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_results.png')
    print("Training complete! Model saved.")

if __name__ == "__main__":
    train(episodes=500)
```

---

## Deliverables Checklist

### Code
- [ ] `snake_game.py` - Game environment
- [ ] `agent.py` - DQN implementation
- [ ] `train.py` - Training script
- [ ] Training results plot
- [ ] `README.md`

### Report (1000-1500 words)
- [ ] Introduction to Snake game challenge
- [ ] RL methodology
- [ ] Results and learning curves
- [ ] Discussion

### Video (5 minutes)
- [ ] Problem demonstration
- [ ] Code walkthrough
- [ ] Trained agent playing

---

## Assessment (15 marks)

- **Code (5 marks)**: Working game, DQN agent, proper training
- **Functionality (4 marks)**: Agent improves, achieves reasonable scores
- **Documentation (3 marks)**: Clear code, good report
- **Presentation (2 marks)**: Effective video
- **Q&A (1 mark)**: Understanding concepts

---

## Tips

1. **Start small**: 10×10 grid to train faster
2. **Monitor rewards**: Should increase over time
3. **Adjust epsilon decay**: Balance exploration/exploitation
4. **Tune rewards**: Food reward vs death penalty ratio matters
5. **Frame skipping**: Can speed up training

Good luck, Q-Team! 🐍
