"""
Play Flappy Bird with trained agent
"""

import time
from flappy_game import FlappyBird
from dqn_agent import DQNAgent

def play(model_path='flappybird_dqn.pth', games=10):
    env = FlappyBird()
    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0  # No exploration

    print(f"Playing Flappy Bird with trained agent ({games} games)...")
    print("-" * 60)

    scores = []

    for game in range(games):
        state = env.reset()
        done = False
        steps = 0

        while not done:
            action = agent.act(state)
            state, reward, done = env.step(action)
            steps += 1

            # Optional: visualize
            # env.render()
            # time.sleep(0.05)

        scores.append(env.score)
        print(f"Game {game+1}: Score = {env.score}, Steps = {steps}")

    print("-" * 60)
    print(f"Average Score: {sum(scores)/len(scores):.2f}")
    print(f"Best Score: {max(scores)}")

if __name__ == "__main__":
    play(games=10)
