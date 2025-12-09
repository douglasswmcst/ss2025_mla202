# PhuRaTan - Ultimate Tic-Tac-Toe with RL
## Combined Term Paper & Mini-Project Implementation Guide

**Team Members**: 3 students
**Research Paper**: [Using Reinforcement Learning to play Ultimate Tic-Tac-Toe](https://arxiv.org/pdf/2212.12252)
**Submission Deadline**: Day 6 (Saturday, 11:59 PM)

---

## Overview

Implement Q-learning or SARSA to play Ultimate Tic-Tac-Toe - a complex variant where each cell contains another tic-tac-toe board. This demonstrates how RL handles combinatorial complexity.

---

## Part 1: Term Paper Report (10 marks - 1000-1500 words)

### Structure

**1. Introduction (2 marks)**
- What is Ultimate Tic-Tac-Toe and why is it challenging?
- State space complexity vs regular Tic-Tac-Toe
- Why RL is suitable for this problem

**2. Methodology (3 marks)**
- State representation: 9×9 board + active board indicator
- Action space: 81 possible moves (with legal move filtering)
- Reward structure: +1 win, -1 loss, 0 draw, small penalties for illegal moves
- Algorithm: Q-learning or SARSA with function approximation
- Training approach: Self-play

**3. Findings (3 marks)**
- Learning curve and convergence
- Win rate against random/rule-based opponents
- Strategic insights learned by agent
- Limitations and challenges

**4. Organization & References (2 marks)**

---

## Part 2: Implementation (15 marks)

### Quick Setup

```bash
mkdir ultimate_tictactoe_rl
cd ultimate_tictactoe_rl
python -m venv venv
source venv/bin/activate
pip install numpy matplotlib pickle5
```

### Implementation: `game.py` - Game Logic

```python
import numpy as np

class UltimateTicTacToe:
    def __init__(self):
        # 9x9 board: 0=empty, 1=X, 2=O
        self.board = np.zeros((9, 9), dtype=int)
        # 3x3 meta-board tracking won local boards
        self.meta_board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.active_board = None  # None means any board
        self.game_over = False
        self.winner = 0

    def reset(self):
        self.board = np.zeros((9, 9), dtype=int)
        self.meta_board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.active_board = None
        self.game_over = False
        self.winner = 0
        return self.get_state()

    def get_state(self):
        """Return state tuple"""
        return (
            tuple(self.board.flatten()),
            tuple(self.meta_board.flatten()),
            self.active_board,
            self.current_player
        )

    def get_legal_moves(self):
        """Return list of legal (board_idx, cell_idx) moves"""
        if self.game_over:
            return []

        moves = []
        if self.active_board is None:
            # Can play in any board that's not won
            for board_idx in range(9):
                if self.meta_board[board_idx // 3, board_idx % 3] == 0:
                    for cell_idx in range(9):
                        if self.board[board_idx, cell_idx] == 0:
                            moves.append((board_idx, cell_idx))
        else:
            # Must play in active board
            board_idx = self.active_board
            if self.meta_board[board_idx // 3, board_idx % 3] == 0:
                for cell_idx in range(9):
                    if self.board[board_idx, cell_idx] == 0:
                        moves.append((board_idx, cell_idx))
            else:
                # Active board is won, can play anywhere
                return self.get_legal_moves_any()

        return moves

    def get_legal_moves_any(self):
        """Get moves from any non-won board"""
        moves = []
        for board_idx in range(9):
            if self.meta_board[board_idx // 3, board_idx % 3] == 0:
                for cell_idx in range(9):
                    if self.board[board_idx, cell_idx] == 0:
                        moves.append((board_idx, cell_idx))
        return moves

    def make_move(self, board_idx, cell_idx):
        """Make a move and return (next_state, reward, done)"""
        if self.game_over:
            return self.get_state(), 0, True

        if (board_idx, cell_idx) not in self.get_legal_moves():
            # Illegal move - penalty
            return self.get_state(), -10, True

        # Place piece
        self.board[board_idx, cell_idx] = self.current_player

        # Check if local board is won
        local_board = self.board[board_idx].reshape(3, 3)
        local_winner = self.check_winner(local_board)
        if local_winner:
            self.meta_board[board_idx // 3, board_idx % 3] = local_winner

        # Check if game is won
        game_winner = self.check_winner(self.meta_board)
        if game_winner:
            self.game_over = True
            self.winner = game_winner
            reward = 1 if game_winner == self.current_player else -1
            return self.get_state(), reward, True

        # Check for draw
        if np.all(self.meta_board != 0) or len(self.get_legal_moves_any()) == 0:
            self.game_over = True
            return self.get_state(), 0, True

        # Set next active board
        self.active_board = cell_idx if self.meta_board[cell_idx // 3, cell_idx % 3] == 0 else None

        # Switch player
        self.current_player = 3 - self.current_player

        return self.get_state(), 0, False

    def check_winner(self, board):
        """Check 3x3 board for winner"""
        # Rows
        for i in range(3):
            if board[i, 0] == board[i, 1] == board[i, 2] != 0:
                return board[i, 0]
        # Columns
        for j in range(3):
            if board[0, j] == board[1, j] == board[2, j] != 0:
                return board[0, j]
        # Diagonals
        if board[0, 0] == board[1, 1] == board[2, 2] != 0:
            return board[0, 0]
        if board[0, 2] == board[1, 1] == board[2, 0] != 0:
            return board[0, 2]
        return 0

    def render(self):
        """Print board"""
        print("\nMeta Board:")
        print(self.meta_board)
        print("\nFull Board (9x9):")
        print(self.board)
        print(f"Current Player: {self.current_player}")
        print(f"Active Board: {self.active_board}")
```

### Implementation: `agent.py` - Q-Learning Agent

```python
import numpy as np
import pickle
from collections import defaultdict

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995

    def get_action(self, state, legal_moves, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return legal_moves[np.random.randint(len(legal_moves))]

        # Greedy: choose action with highest Q-value
        q_values = {move: self.q_table[state][move] for move in legal_moves}
        max_q = max(q_values.values()) if q_values else 0
        best_moves = [move for move, q in q_values.items() if q == max_q]
        return best_moves[np.random.randint(len(best_moves))]

    def update(self, state, action, reward, next_state, next_legal_moves, done):
        """Q-learning update"""
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            # Max Q-value of next state
            next_q_values = [self.q_table[next_state][a] for a in next_legal_moves]
            max_next_q = max(next_q_values) if next_q_values else 0
            target_q = reward + self.gamma * max_next_q

        # Update Q-value
        self.q_table[state][action] += self.alpha * (target_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(f))

class RandomAgent:
    """Baseline random agent"""
    def get_action(self, state, legal_moves, training=False):
        return legal_moves[np.random.randint(len(legal_moves))]
```

### Implementation: `train.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from game import UltimateTicTacToe
from agent import QLearningAgent, RandomAgent

def train(episodes=10000):
    game = UltimateTicTacToe()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=1.0)

    wins = []
    draws = []
    losses = []

    print("Training via self-play...")

    for episode in range(episodes):
        state = game.reset()
        done = False

        while not done:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            action = agent.get_action(state, legal_moves, training=True)
            next_state, reward, done = game.make_move(*action)

            if not done:
                next_legal_moves = game.get_legal_moves()
                agent.update(state, action, reward, next_state, next_legal_moves, done)
            else:
                agent.update(state, action, reward, next_state, [], done)

            state = next_state

        # Track results
        if game.winner == 1:
            wins.append(1)
        elif game.winner == 0:
            draws.append(1)
        else:
            losses.append(1)

        agent.decay_epsilon()

        if (episode + 1) % 1000 == 0:
            recent = 1000
            w = sum(wins[-recent:])
            d = sum(draws[-recent:])
            l = sum(losses[-recent:])
            print(f"Episode {episode + 1}: W:{w} D:{d} L:{l} Eps:{agent.epsilon:.3f}")

    agent.save('trained_agent.pkl')
    plot_results(wins, draws, losses)
    return agent

def plot_results(wins, draws, losses):
    window = 100
    x = range(window, len(wins) + 1)
    w = [sum(wins[i-window:i])/window for i in x]
    d = [sum(draws[i-window:i])/window for i in x]
    l = [sum(losses[i-window:i])/window for i in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, w, label='Wins')
    plt.plot(x, d, label='Draws')
    plt.plot(x, l, label='Losses')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (100-episode window)')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    train(episodes=10000)
```

### Implementation: `evaluate.py`

```python
from game import UltimateTicTacToe
from agent import QLearningAgent, RandomAgent

def play_match(agent1, agent2, games=100):
    """Play matches between two agents"""
    game = UltimateTicTacToe()
    results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}

    for _ in range(games):
        state = game.reset()
        done = False

        while not done:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            if game.current_player == 1:
                action = agent1.get_action(state, legal_moves, training=False)
            else:
                action = agent2.get_action(state, legal_moves, training=False)

            state, reward, done = game.make_move(*action)

        if game.winner == 1:
            results['agent1_wins'] += 1
        elif game.winner == 2:
            results['agent2_wins'] += 1
        else:
            results['draws'] += 1

    return results

def evaluate():
    # Load trained agent
    trained_agent = QLearningAgent()
    trained_agent.load('trained_agent.pkl')
    trained_agent.epsilon = 0  # No exploration

    # Test against random
    random_agent = RandomAgent()

    print("Trained Agent (Player 1) vs Random (Player 2):")
    results = play_match(trained_agent, random_agent, games=100)
    print(f"Wins: {results['agent1_wins']}, Losses: {results['agent2_wins']}, Draws: {results['draws']}")
    print(f"Win Rate: {results['agent1_wins']/100*100:.1f}%")

    print("\nRandom (Player 1) vs Trained Agent (Player 2):")
    results = play_match(random_agent, trained_agent, games=100)
    print(f"Wins: {results['agent1_wins']}, Losses: {results['agent2_wins']}, Draws: {results['draws']}")
    print(f"Trained Win Rate: {results['agent2_wins']/100*100:.1f}%")

if __name__ == "__main__":
    evaluate()
```

---

## Deliverables Checklist

### Code
- [ ] `game.py` - Game logic
- [ ] `agent.py` - Q-learning agent
- [ ] `train.py` - Training script
- [ ] `evaluate.py` - Evaluation
- [ ] `README.md`

### Report (1000-1500 words)
- [ ] Introduction to Ultimate Tic-Tac-Toe complexity
- [ ] RL methodology (state, action, reward)
- [ ] Results and learning curves
- [ ] Discussion of strategy learned

### Video (5 minutes)
- [ ] Problem explanation
- [ ] Code demonstration
- [ ] Results and game playthrough

---

## Assessment Rubric (15 marks)

- **Code (5 marks)**: Correct game logic, Q-learning implementation, working training
- **Functionality (4 marks)**: Agent learns and improves, beats random player
- **Documentation (3 marks)**: Clear README, well-commented code, report quality
- **Presentation (2 marks)**: Clear video, good demonstration
- **Q&A (1 mark)**: Understanding of Q-learning and game strategy

---

## Tips

1. **Start with regular Tic-Tac-Toe** if Ultimate is too complex initially
2. **Use state hashing** for efficient Q-table storage
3. **Self-play** is key - agent plays against itself
4. **Monitor learning** - win rate should increase over time
5. **Simplify state** representation if needed (symmetries, rotations)

Good luck, PhuRaTan! 🎮
