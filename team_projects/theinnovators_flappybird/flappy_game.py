"""
Flappy Bird Game Environment for RL
"""

import numpy as np
import random

class FlappyBird:
    def __init__(self):
        # Game parameters
        self.bird_y = 50
        self.bird_velocity = 0
        self.gravity = 2
        self.flap_strength = -10
        
        self.pipe_x = 100
        self.pipe_gap = 30
        self.pipe_y = random.randint(20, 60)
        
        self.score = 0
        self.done = False

    def reset(self):
        """Reset game to initial state"""
        self.bird_y = 50
        self.bird_velocity = 0
        self.pipe_x = 100
        self.pipe_y = random.randint(20, 60)
        self.score = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """Get state representation"""
        return np.array([
            self.bird_y,
            self.bird_velocity,
            self.pipe_x,
            self.pipe_y,
            self.pipe_y + self.pipe_gap
        ], dtype=np.float32)

    def step(self, action):
        """
        Take action: 0=do nothing, 1=flap
        Returns: state, reward, done
        """
        # Apply action
        if action == 1:
            self.bird_velocity = self.flap_strength

        # Apply physics
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity

        # Move pipe
        self.pipe_x -= 5

        # Check if passed pipe
        if self.pipe_x < 0:
            self.pipe_x = 100
            self.pipe_y = random.randint(20, 60)
            self.score += 1

        # Check collisions
        reward = 0.1  # Small reward for surviving
        
        # Hit ground or ceiling
        if self.bird_y < 0 or self.bird_y > 100:
            self.done = True
            return self._get_state(), -10, True

        # Hit pipe
        if 40 <= self.pipe_x <= 60:  # Bird in pipe zone
            if self.bird_y < self.pipe_y or self.bird_y > self.pipe_y + self.pipe_gap:
                self.done = True
                return self._get_state(), -10, True

        # Passed pipe
        if self.pipe_x == 35:  # Just passed
            reward = 10

        return self._get_state(), reward, False

    def render(self):
        """Simple text rendering"""
        print(f"\nScore: {self.score}")
        print(f"Bird Y: {self.bird_y:.1f}, Velocity: {self.bird_velocity}")
        print(f"Pipe X: {self.pipe_x}, Gap: [{self.pipe_y}, {self.pipe_y + self.pipe_gap}]")
