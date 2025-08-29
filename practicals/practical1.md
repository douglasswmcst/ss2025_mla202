# Practical 1: An Introduction to Gymnasium Environments

Welcome to the first practical session for the Reinforcement Learning module\! Today, we will dive into **Gymnasium** (the maintained successor to OpenAI's Gym), which is the standard toolkit for developing and comparing RL algorithms.

The goal of this session is to build a solid understanding of the fundamental agent-environment interaction loop. This loop is the practical implementation of the theoretical concepts of **Markov Decision Processes (MDPs)** we discuss in lectures. By the end of this session, you will be able to confidently set up any Gymnasium environment and control an agent within it.

### Learning Objectives

  * ✅ Install the Gymnasium library and set up a proper Python environment.
  * ✅ Instantiate a simulated environment and query its properties.
  * ✅ Understand the **observation space** and **action space**.
  * ✅ Implement the full agent-environment interaction loop using `reset()` and `step()`.
  * ✅ Distinguish between a `terminated` and `truncated` episode.
  * ✅ Evaluate the performance of a baseline random agent.

-----

## 2\. Setup and Installation

Before we can use Gymnasium, we need to set up our Python environment correctly. Following these steps will prevent many common issues.

### Step 1: Install Python (if you haven't already)

This module requires **Python 3.8 or newer**. If you don't have it installed, follow these instructions.

1.  **Download Python**: Visit the official Python website at [python.org/downloads/](https://python.org/downloads/). Download a recent, stable version (e.g., Python 3.11).
2.  **Run the Installer**:
      * **On Windows**: Run the installer. **This is very important:** On the first screen of the installer, make sure you check the box that says **"Add Python to PATH"** before clicking "Install Now". This will allow you to run Python from your command prompt.
      * **On macOS**: Run the downloaded installer package. The installer will handle the setup and path configuration for you.
      * **On Linux**: Python is usually pre-installed. You can check your version with `python3 --version`. If you need to install it, use your distribution's package manager, for example: `sudo apt-get update && sudo apt-get install python3`.
3.  **Verify the Installation**: Open a **new** terminal or command prompt and type `python --version` (or `python3 --version` on macOS/Linux). You should see the version number you just installed.

### Step 2: Create a Virtual Environment (Recommended)

A virtual environment is a private workspace for each project that keeps its dependencies separate from others. This is a crucial best practice.

1.  **Create a Project Folder**: Open your terminal or command prompt.
    ```bash
    # 1. Create a folder for your RL practicals and navigate into it
    mkdir rl-practicals
    cd rl-practicals
    ```
2.  **Create the Virtual Environment**:
    ```bash
    # 2. Create a Python virtual environment named 'venv'
    python -m venv venv
    ```
3.  **Activate the Environment**: You must activate the environment every time you work on the project.
    ```bash
    # On Windows:
    venv\Scripts\activate

    # On macOS/Linux:
    source venv/bin/activate
    ```
    You'll know it's active because your command prompt will be prefixed with `(venv)`.

### Step 3: Install Gymnasium

Now, with your virtual environment active, install the library using `pip`.

```bash
pip install gymnasium
```

To verify the installation was successful, you can run `pip show gymnasium`. You are now ready to start coding\!

-----

## 3\. Interacting with an Environment: "FrozenLake"

Let's write a Python script to interact with the `"FrozenLake-v1"` environment. This environment is a classic grid world where an agent controls a character who must navigate from a starting tile ('S') to a goal tile ('G') across a frozen lake. Some tiles are walkable ('F'), but others are holes ('H') that end the game.

### Task 1: Create and Inspect the Environment

First, we import `gymnasium` and use `gym.make()` to load our environment.

```python
import gymnasium as gym

# Create the FrozenLake environment
# 'render_mode="human"' will pop up a window to visualize the agent's actions.
# We use "rgb_array" when we need to capture the state as pixel data for training.
env = gym.make("FrozenLake-v1", render_mode="human")
```

> **Note on Versioning:** The `"v1"` in the environment name refers to the version. Using specific versions ensures your code is reproducible, as environment dynamics can change between versions.

Next, let's inspect its core properties: the **action space** (what the agent can do) and the **observation space** (what the agent can see).

```python
# ACTION SPACE
# This tells us the set of all valid actions.
print(f"Action Space: {env.action_space}")
# The output `Discrete(4)` means actions are numbered from 0 to 3.
# For FrozenLake, these correspond to directions:
# 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP

# OBSERVATION SPACE
# This tells us the set of all possible states.
print(f"Observation Space: {env.observation_space}")
# The output `Discrete(16)` means states are numbered from 0 to 15.
# This corresponds to the 16 tiles in the 4x4 grid, indexed from left-to-right, top-to-bottom.
# State 0 is the top-left 'S' tile, and state 15 is the bottom-right tile.
```

### Task 2: Implement the Agent-Environment Loop

Now we'll write the code for a single episode where our agent takes **random actions**. This is a crucial baseline—any intelligent agent we build later must perform better than this\!

```python
import gymnasium as gym
import time

env = gym.make("FrozenLake-v1", render_mode="human")

print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

# --- THE MAIN LOOP ---

# 1. RESET the environment
# This function is called at the beginning of every new episode.
# It returns the initial observation and some optional info.
observation, info = env.reset()

# Initialize variables to track the episode's progress
terminated = False
truncated = False
total_reward = 0.0

# The loop continues as long as the episode is not finished.
while not terminated and not truncated:
    # 2. RENDER the environment (optional)
    # This displays the current state in the popup window.
    env.render()

    # 3. CHOOSE an action
    # For now, we select a completely random action from the action space.
    action = env.action_space.sample()
    print(f"Taking action: {action} (0:L, 1:D, 2:R, 3:U)")

    # 4. STEP the environment
    # This is the most important function. It executes the chosen action.
    # It returns five crucial pieces of information:
    #   next_observation: The agent's new location.
    #   reward: The reward obtained from the last action. For FrozenLake, this is 1.0 if we reach the goal 'G', and 0.0 otherwise.
    #   terminated: A boolean. True if the episode ended because of a terminal state (reaching 'G' or falling in 'H').
    #   truncated: A boolean. True if the episode ended for another reason (e.g., a time limit was reached).
    #   info: A dictionary for debugging info, which we can ignore for now.
    next_observation, reward, terminated, truncated, info = env.step(action)

    # Update our tracking variables
    total_reward += reward
    observation = next_observation

    # Add a small delay to make the visualization easier to follow
    time.sleep(0.5)

# After the loop, the episode is finished.
print(f"\nEpisode finished! Total Reward: {total_reward}")

# 5. CLOSE the environment
# This is important for cleaning up resources, especially the rendering window.
env.close()
```

Run this script. You'll see your agent (the red rectangle) moving about randomly. Most of the time, it will end by falling into a hole ('H'), resulting in a total reward of 0.0.

-----

## 4\. Exercises 🧠

Time to apply what you've learned.

### Exercise 1: CartPole Challenge

The `"CartPole-v1"` environment is another classic. The goal is to balance a pole on a moving cart by pushing the cart left or right.

1.  Modify your script to use `gym.make("CartPole-v1", render_mode="human")`.
2.  Print the `action_space` and `observation_space`.
3.  **Answer these questions in comments in your code:**
      * What type of space is the action space? How many actions are there?
      * What type of space is the observation space? The output is `Box(4,)`. This represents a continuous space with 4 numbers. Based on the problem, what could these four numbers possibly represent?
      * Run the random agent for one episode. What does the reward seem to represent in this environment? (Hint: you get a reward for every step the pole remains balanced).

### Exercise 2: Performance Evaluation of the Random Agent

A single episode doesn't tell us much. To properly evaluate an agent, we need to average its performance over many episodes.

1.  Using your `"FrozenLake-v1"` script as a base, remove the `render()`, `print()`, and `time.sleep()` calls from the main loop to speed up execution.
2.  Create an outer loop to run **1000** episodes.
3.  Store the `total_reward` from each episode in a list.
4.  After the outer loop finishes, calculate and print the **average reward** over the 1000 episodes. This value represents the success rate of your random agent.

> **Code Structure Hint:**
>
> ```python
> num_episodes = 1000
> rewards_per_episode = []
> ```

> for episode in range(num\_episodes):
> \# Reset the environment for a new episode
> observation, info = env.reset()
> terminated, truncated = False, False
> episode\_reward = 0

> ```
> while not terminated and not truncated:
>     # ... take a random step ...
>     # ... update episode_reward ...
> ```

> ```
> # Append the final reward for this episode to the list
> rewards_per_episode.append(episode_reward)
> ```

> # After all episodes, calculate the average
>
> average\_reward = sum(rewards\_per\_episode) / num\_episodes
> print(f"Average reward over {num\_episodes} episodes: {average\_reward:.4f}")
>
> ```
> ```

-----

## Next Steps

Excellent work\! You now have the fundamental skills to interact with and test agents in any Gymnasium environment. You've seen that the core API—`reset()` and `step()`—is consistent across different problems, which is what makes this framework so powerful.

You've also proven that a random agent performs very poorly on "FrozenLake". The agent has no memory and doesn't learn from its rewards. In the next practical, we will fix this by implementing **Q-Learning**, our first true Reinforcement Learning algorithm, which will learn a policy to reliably solve this environment.

-----

## 5\. Submission Instructions 📝

For your work to be graded, please follow these instructions carefully.

1.  **Create a Public Git Repository**: Create a new **public** repository on a platform like GitHub or GitLab. Name it something descriptive, like `rl-module-practicals`.

2.  **Upload Your Work**: Push the following files to your repository:

      * The Python script(s) you wrote to complete the main walkthrough and the exercises. Please ensure your code is clean and commented where necessary.
      * A `README.md` file in the root of your repository.

3.  **Write Your Learning Retrospective**: Your `README.md` file is a critical part of your submission. It should contain a reflection on your experience with this practical. Please include the following:

      * A brief summary of the tasks you completed.
      * The answers to the questions from **Exercise 1** about the CartPole environment's action and observation spaces.
      * The final **average reward** you calculated for the random agent in **Exercise 2**.
      * A section on challenges: What was the most difficult part of this practical for you? (e.g., setting up the environment, understanding the `step` function's return values, structuring the loops).
      * A section on key takeaways: What is the most important or surprising thing you learned?

4.  **Submit Your Repository URL**: Once your repository is public and contains all the required files, submit the **URL** to your repository on the submission portal Moodle.