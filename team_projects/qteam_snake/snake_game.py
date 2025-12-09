"""
Snake Game Environment for RL
"""

import numpy as np
import random

class SnakeGame:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """Reset game to initial state"""
        # Snake starts in middle, length 3
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid, mid-1), (mid, mid-2)]
        self.direction = (0, 1)  # Moving right
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.done = False
        return self._get_state()

    def _place_food(self):
        """Place food at random empty position"""
        while True:
            food = (random.randint(0, self.grid_size-1), 
                   random.randint(0, self.grid_size-1))
            if food not in self.snake:
                return food

    def _get_state(self):
        """Get state representation"""
        head = self.snake[0]
        
        # Danger detection
        danger_straight = self._is_collision((head[0] + self.direction[0], 
                                             head[1] + self.direction[1]))
        danger_left = self._is_collision((head[0] + self.direction[1], 
                                         head[1] - self.direction[0]))
        danger_right = self._is_collision((head[0] - self.direction[1], 
                                          head[1] + self.direction[0]))

        # Food direction
        food_up = self.food[0] < head[0]
        food_down = self.food[0] > head[0]
        food_left = self.food[1] < head[1]
        food_right = self.food[1] > head[1]

        # Current direction
        dir_up = self.direction == (-1, 0)
        dir_down = self.direction == (1, 0)
        dir_left = self.direction == (0, -1)
        dir_right = self.direction == (0, 1)

        state = np.array([
            danger_straight, danger_left, danger_right,
            dir_up, dir_down, dir_left, dir_right,
            food_up, food_down, food_left, food_right
        ], dtype=int)

        return state

    def _is_collision(self, pos):
        """Check if position causes collision"""
        return (pos[0] < 0 or pos[0] >= self.grid_size or
                pos[1] < 0 or pos[1] >= self.grid_size or
                pos in self.snake)

    def step(self, action):
        """
        Take action: 0=straight, 1=left, 2=right
        Returns: state, reward, done
        """
        self.steps += 1

        # Update direction based on action
        if action == 1:  # Turn left
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2:  # Turn right
            self.direction = (-self.direction[1], self.direction[0])

        # Move snake
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Check collision
        if self._is_collision(new_head):
            self.done = True
            return self._get_state(), -10, True

        # Add new head
        self.snake.insert(0, new_head)

        # Check food
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()

        # Check timeout
        if self.steps > 100 * len(self.snake):
            self.done = True

        return self._get_state(), reward, self.done

    def render(self):
        """Print game board"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        
        for pos in self.snake[1:]:
            grid[pos] = 'o'
        grid[self.snake[0]] = 'O'
        grid[self.food] = 'F'

        print("\nScore:", self.score)
        for row in grid:
            print(' '.join(row))
