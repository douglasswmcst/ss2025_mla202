# Practical 2: Q-Learning - Your First Intelligent Agent

Welcome to the second practical session! In practical1, you discovered that a random agent performs quite poorly on FrozenLake, achieving only about 6% success rate. Today, we will implement **Q-Learning**, a foundational reinforcement learning algorithm that will learn from its mistakes and dramatically improve performance.

This practical connects the theoretical MDP concepts from your exercises to a working implementation that can actually learn and improve over time.

### Learning Objectives

* ✅ Understand Q-values and Q-tables as action-value functions
* ✅ Implement the Q-learning algorithm from scratch
* ✅ Master the exploration vs exploitation trade-off using epsilon-greedy
* ✅ Understand the role of learning rate (α) and discount factor (γ)
* ✅ Train an agent that achieves >80% success rate on FrozenLake
* ✅ Analyze the learning process and compare against random baseline

-----

## 2. From Value Functions to Q-Learning

Remember from your exercises that the Bellman equation tells us the value of being in a state:
V(s) = R(s) + γ Σ P(s'|s)V(s')

But this requires knowing the transition probabilities P(s'|s), which we don't have when interacting with an environment. Q-learning solves this by learning **action-values** (Q-values) instead.

### 🔑 Key Concept: What is a Q-Value?

A Q-value, Q(s,a), represents the **expected total reward** for taking action `a` in state `s` and then following the optimal policy afterwards. Think of it as answering: "How good is it to take this specific action from this specific state?"

**Detailed Explanation:**
- **Q(s,a)** = "Quality" of action `a` in state `s`
- It predicts the total future reward if we:
  1. Take action `a` from state `s`
  2. Act optimally (follow the best policy) forever after
- Higher Q-values mean better actions
- The agent will prefer actions with higher Q-values

**Real-world analogy**: Imagine you're deciding which route to take to work:
- Q(home, highway) = "How good is taking the highway from home?"
- Q(home, backroads) = "How good is taking backroads from home?"
- The Q-value considers both immediate travel time AND future traffic conditions

### 🔑 Key Concept: The Q-Table

For FrozenLake with 16 states and 4 actions, we need a 16×4 table:

```
State\Action    LEFT    DOWN    RIGHT   UP
0 (Start)       0.0     0.0     0.0     0.0
1               0.0     0.0     0.0     0.0
5 (Hole)        0.0     0.0     0.0     0.0
15 (Goal)       0.0     0.0     0.0     0.0
...
```

**Detailed Explanation:**
- **Initialization**: All Q-values start at 0.0 (we know nothing about the environment)
- **Learning Process**: Through trial and error, Q-values get updated based on actual experience
- **Final State**: After training, the table contains learned "wisdom" about which actions work best in each state

**Why this works:**
- The agent doesn't need to know the environment's rules in advance
- It learns by trying actions and observing the results
- Over time, good actions get higher Q-values, bad actions get lower Q-values
- The final Q-table represents the agent's learned policy

### 🔑 Key Concept: From State Values to Action Values

**State Values V(s)** tell us "how good is this state?" but don't tell us what to do.
**Action Values Q(s,a)** tell us "how good is this action from this state?" which directly guides our decisions.

**Example in FrozenLake:**
- V(state_0) might be 0.8 (starting position is pretty good)
- But Q(state_0, RIGHT) = 0.9 and Q(state_0, DOWN) = 0.2
- This tells us: go RIGHT from the start, not DOWN!

### 🔑 Key Concept: The Q-Learning Update Rule

Q-learning uses this update rule after each step:

**Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]**

**Detailed Parameter Breakdown:**

**α (Alpha) - Learning Rate (0 < α ≤ 1)**
- **What it controls**: How much we update our Q-value each step
- **High α (e.g., 0.8)**: Learn fast, but might be unstable (big updates)
- **Low α (e.g., 0.1)**: Learn slowly, but more stable (small updates)
- **Real-world analogy**: Like how quickly you change your opinion after new information
  - High α = "I completely change my mind based on one experience"
  - Low α = "I gradually adjust my opinion based on many experiences"

**r - Immediate Reward**
- **What it is**: The reward we just received for taking action `a` in state `s`
- **In FrozenLake**: Usually 0, except +1 when reaching the goal
- **Role**: Tells us how good our immediate action was

**γ (Gamma) - Discount Factor (0 ≤ γ ≤ 1)**
- **What it controls**: How much we care about future rewards vs immediate rewards
- **High γ (e.g., 0.99)**: "I care a lot about long-term consequences"
- **Low γ (e.g., 0.1)**: "I only care about immediate rewards"
- **Mathematical effect**: Future rewards get multiplied by γ, γ², γ³, etc.
- **FrozenLake example**: High γ is crucial because the only reward comes at the very end!

**max Q(s',a') - Best Future Value**
- **What it represents**: The best possible Q-value from the next state s'
- **Why max?**: We assume the agent will act optimally in the future
- **Computation**: Look at all possible actions from s', pick the highest Q-value

### 🔑 Key Concept: Breaking Down the Update Rule

Let's understand each part of: **Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]**

**Step 1: Calculate the "Target" Value**
```
Target = r + γ max Q(s',a')
```
This is our estimate of what Q(s,a) should actually be, based on what just happened.

**Step 2: Calculate the "Prediction Error"**
```
Error = Target - Q(s,a) = [r + γ max Q(s',a') - Q(s,a)]
```
This tells us how wrong our current Q-value was.

**Step 3: Update Our Estimate**
```
New Q(s,a) = Old Q(s,a) + α × Error
```
We adjust our Q-value in the direction of the error, scaled by the learning rate.

### 🔑 Key Concept: Temporal Difference Learning

Q-learning is a **temporal difference** method:
- **"Temporal"**: We compare our prediction at time t with our prediction at time t+1
- **"Difference"**: We update based on the difference between what we expected and what we got

**The Genius of Q-Learning:**
- We don't need to wait until the episode ends to learn
- We can update our estimates immediately after each step
- We bootstrap: use our own estimates to improve our own estimates

**Example Walkthrough:**
1. **Current belief**: Q(state_5, RIGHT) = 0.2 (going right from state 5 is somewhat bad)
2. **Take action**: Go RIGHT from state 5
3. **Observe**: Get reward = 0, land in state 9
4. **Look ahead**: Best action from state 9 has Q-value = 0.8
5. **Calculate target**: 0 + 0.99 × 0.8 = 0.792
6. **Calculate error**: 0.792 - 0.2 = 0.592 (we were too pessimistic!)
7. **Update**: Q(state_5, RIGHT) = 0.2 + 0.1 × 0.592 = 0.259 (slightly more optimistic)

**Intuition**: If we got more reward than expected (including future prospects), increase the Q-value. If we got less, decrease it.

-----

## 3. Implementing Q-Learning on FrozenLake

Let's implement Q-learning step by step. We'll use the same FrozenLake-v1 environment from practical1.

### Step 1: Setting Up the Q-Table

```python
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Create environment (same as practical1)
env = gym.make("FrozenLake-v1", render_mode="human")

# Initialize Q-table with zeros
# Shape: (num_states, num_actions) = (16, 4)
q_table = np.zeros([env.observation_space.n, env.action_space.n])

print(f"Q-table shape: {q_table.shape}")
print(f"Initial Q-table (first 5 states):")
print(q_table[:5])
```

### 🔑 Key Concept: Epsilon-Greedy Action Selection

This is how we balance exploration (trying new actions) vs exploitation (using known good actions):

```python
def choose_action(state, q_table, epsilon):
    """
    Choose action using epsilon-greedy strategy

    Args:
        state: Current state number
        q_table: Current Q-table
        epsilon: Exploration rate (0 = always exploit, 1 = always explore)

    Returns:
        action: Chosen action number
    """
    if random.random() < epsilon:
        # EXPLORE: Choose random action
        return env.action_space.sample()
    else:
        # EXPLOIT: Choose action with highest Q-value
        return np.argmax(q_table[state])
```

**Detailed Concept Explanation:**

**The Exploration vs Exploitation Dilemma:**
- **Exploitation**: Use what you already know to maximize reward
- **Exploration**: Try new things to potentially discover better strategies
- **The Problem**: If you only exploit, you might miss better strategies. If you only explore, you never use what you've learned!

**How Epsilon-Greedy Solves This:**
- **ε (epsilon)**: Probability of choosing a random action (exploration)
- **1-ε**: Probability of choosing the best known action (exploitation)
- **Balance**: As epsilon decreases during training, we explore less and exploit more

**Real-World Analogy - Restaurant Choice:**
- **Exploitation**: "I know I like pizza, so I'll go to my favorite pizza place"
- **Exploration**: "Maybe I'll try that new sushi restaurant instead"
- **Epsilon-greedy**: "90% of the time I'll go to my favorite, 10% I'll try something new"

**Why This Strategy Works:**
1. **Early Training (high ε)**: Lots of exploration to discover the environment
2. **Late Training (low ε)**: Mostly exploitation to use learned knowledge efficiently
3. **Always Some Exploration**: Even when ε is small, we occasionally try new things

**Epsilon Decay Strategy:**
```python
# Start exploring everything
epsilon = 1.0

# Gradually reduce exploration
epsilon = epsilon * 0.995  # Each episode

# Always explore at least a little
epsilon = max(0.01, epsilon)
```

**Mathematical Intuition:**
- Episode 1: ε = 1.0 → 100% random actions (pure exploration)
- Episode 1000: ε ≈ 0.007 → 99.3% best actions, 0.7% random
- Episode 5000: ε = 0.01 → 99% best actions, 1% random (minimum exploration)

### Step 2: The Complete Q-Learning Training Loop

```python
# Q-learning hyperparameters
learning_rate = 0.1    # α - how much to update Q-values each step
discount_factor = 0.99 # γ - how much we value future rewards
epsilon = 1.0          # ε - exploration rate (start exploring everything)
epsilon_decay = 0.995  # Gradually reduce exploration
min_epsilon = 0.01     # Always explore at least 1%

# Training parameters
num_episodes = 10000
rewards_per_episode = []

print("Starting Q-learning training...")

for episode in range(num_episodes):
    # Reset environment for new episode
    state, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # Choose action using epsilon-greedy
        action = choose_action(state, q_table, epsilon)

        # Take action and observe result
        next_state, reward, terminated, truncated, info = env.step(action)

        # 🔑 Q-LEARNING UPDATE RULE - Step by Step
        # Current Q-value (our old belief about this action)
        current_q = q_table[state, action]

        # Best future Q-value (if not terminal)
        if terminated:
            max_future_q = 0  # No future rewards if episode ended
        else:
            max_future_q = np.max(q_table[next_state])  # Best action from next state

        # Calculate the "target" Q-value (what Q should be based on this experience)
        target_q = reward + discount_factor * max_future_q

        # Calculate the "prediction error" (how wrong were we?)
        error = target_q - current_q

        # Update Q-value by moving towards the target (scaled by learning rate)
        new_q = current_q + learning_rate * error

        # Store the updated Q-value in our table
        q_table[state, action] = new_q

        # Move to next state
        state = next_state
        total_reward += reward

    # Store episode reward
    rewards_per_episode.append(total_reward)

    # Decay epsilon (explore less over time)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Print progress every 1000 episodes
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])  # Last 100 episodes
        print(f"Episode {episode + 1}/10000, Average Reward: {avg_reward:.3f}, Epsilon: {epsilon:.3f}")

print("Training completed!")
```

-----

## 4. Analyzing Your Q-Learning Agent

### Visualizing Learning Progress

```python
# Plot learning curve
plt.figure(figsize=(12, 4))

# Plot 1: Rewards over time (with moving average)
plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode, alpha=0.3, label='Episode Reward')

# Calculate moving average for smoother trend
window_size = 100
moving_avg = []
for i in range(len(rewards_per_episode)):
    start = max(0, i - window_size)
    moving_avg.append(np.mean(rewards_per_episode[start:i+1]))

plt.plot(moving_avg, label='Moving Average', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Q-Learning Training Progress')
plt.legend()

# Plot 2: Final Q-table heatmap
plt.subplot(1, 2, 2)
plt.imshow(q_table, cmap='viridis', aspect='auto')
plt.xlabel('Actions (0:Left, 1:Down, 2:Right, 3:Up)')
plt.ylabel('States')
plt.title('Final Q-Table Values')
plt.colorbar()

plt.tight_layout()
plt.show()
```

### Testing Your Trained Agent

Now let's test how well your agent performs compared to the random agent from practical1:

```python
def test_agent(q_table, num_episodes=1000, render=False):
    """Test the trained Q-learning agent"""
    env_test = gym.make("FrozenLake-v1", render_mode="human" if render else None)

    wins = 0
    total_rewards = []

    for episode in range(num_episodes):
        state, info = env_test.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not terminated and not truncated:
            # Use pure exploitation (epsilon = 0)
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, info = env_test.step(action)
            total_reward += reward

            if render and episode < 3:  # Only render first 3 episodes
                env_test.render()
                time.sleep(1)

        total_rewards.append(total_reward)
        if total_reward > 0:
            wins += 1

    env_test.close()

    success_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)

    return success_rate, avg_reward

# Test your Q-learning agent
success_rate, avg_reward = test_agent(q_table, num_episodes=1000)

print(f"Q-Learning Agent Results (1000 episodes):")
print(f"Success Rate: {success_rate:.1%}")
print(f"Average Reward: {avg_reward:.4f}")
print(f"\nComparison to Random Agent (~6% success rate):")
print(f"Improvement: {success_rate/0.06:.1f}x better!")
```

-----

## 5. Exercises 🧠

### 🔑 Key Concept: Exercise 1 - Hyperparameter Investigation

Understanding how hyperparameters affect learning is crucial for RL success.

1. **Learning Rate (α) Experiment**:
   - Train three agents with α = 0.01, 0.1, and 0.5
   - Keep other parameters the same (γ=0.99, ε starts at 1.0)
   - **Deep Understanding Questions**:
     * How does learning rate affect convergence speed and final performance?
     * Why might α = 0.01 learn slowly but be more stable?
     * What happens with α = 0.5 - does it learn faster or become unstable?
     * **Key Insight**: Learning rate controls the size of updates - too high causes instability, too low causes slow learning

2. **Discount Factor (γ) Experiment**:
   - Train agents with γ = 0.9, 0.99, and 0.999
   - **Deep Understanding Questions**:
     * Why might a higher discount factor perform better on FrozenLake?
     * How does γ affect the agent's "patience" for long-term rewards?
     * What happens with γ = 0.9 vs γ = 0.999 in terms of final performance?
     * **Key Insight**: FrozenLake only gives reward at the goal - agents need high γ to value that distant reward

3. **Epsilon Decay Experiment**:
   - Try epsilon_decay = 0.99 (fast decay) vs 0.999 (slow decay)
   - **Deep Understanding Questions**:
     * What happens if we stop exploring too quickly (fast decay)?
     * How does exploration schedule affect final performance?
     * Which decay rate leads to better learning and why?
     * **Key Insight**: Balance between exploration (learning about environment) and exploitation (using learned knowledge)

### 🔑 Key Concept: Exercise 2 - Q-Table Analysis and Understanding

After training your best agent:

1. **Starting Position Analysis**:
   - Print the Q-values for state 0 (starting position). Which action has the highest Q-value and why?
   - **Deep Understanding**: What does this tell you about the optimal first move?
   - **Connection to FrozenLake Map**: Look at the 4x4 grid - why might certain directions be better from the start?

2. **Exploration Coverage Analysis**:
   - Find a state where all Q-values are still 0. What does this tell you about the agent's exploration?
   - **Deep Understanding**: Why might some states never be visited?
   - **Policy Implications**: Is it okay for an optimal agent to never visit certain states?

3. **Terminal State Analysis**:
   - Look at the Q-values for state 5 (first hole). What should these values be and why?
   - **Deep Understanding**: Why should Q-values for hole states be very negative or zero?
   - **Learning Verification**: Does your Q-table reflect the danger of holes?

**🔑 Understanding Q-Table Patterns:**
- **High Q-values**: Actions that lead toward the goal
- **Low Q-values**: Actions that lead to holes or away from goal
- **Zero Q-values**: States/actions never experienced during training
- **Convergence Signs**: Similar Q-values across multiple training runs

**Code Structure Hint:**
```python
# Print Q-values for specific states
print("State 0 Q-values:", q_table[0])
print("Action with highest Q-value:", np.argmax(q_table[0]))

# Find states with no exploration
unexplored_states = []
for state in range(16):
    if np.all(q_table[state] == 0):
        unexplored_states.append(state)
print("Unexplored states:", unexplored_states)

# Analyze hole states (5, 7, 11, 12 in FrozenLake)
hole_states = [5, 7, 11, 12]
for hole in hole_states:
    print(f"Hole state {hole} Q-values:", q_table[hole])
```

### Exercise 3: Custom Environment Challenge (Optional)

Modify the FrozenLake parameters and see how your agent adapts:

```python
# Create a more challenging version
env_hard = gym.make("FrozenLake-v1", is_slippery=True)  # Add randomness to actions
```

Train your Q-learning agent on this slippery version. How does performance compare?

-----

## Next Steps

Excellent work! You've now implemented your first reinforcement learning algorithm that can actually learn and improve its performance. You've seen how Q-learning transforms a random agent (6% success) into an intelligent agent (>80% success) through experience.

The key insights you should take away:
- **Q-values represent the learned "wisdom" about which actions work best in each state**
- **The exploration vs exploitation trade-off is crucial for learning**
- **Temporal difference learning allows us to learn from each step, not just episode outcomes**
- **Hyperparameters significantly affect both learning speed and final performance**

In future practicals, we'll explore more advanced RL algorithms, but Q-learning remains the foundation that makes it all possible.

-----

## 6. Submission Instructions 📝

For your work to be graded, please follow these instructions carefully.

1. **Update Your Repository**: Use the same repository you created for practical1. Create a new folder called `practical2` and add your files there.

2. **Upload Your Work**: Push the following files to your repository:
   - Your complete Q-learning implementation script (`qlearning_agent.py`)
   - Your hyperparameter experiment results (`hyperparameter_experiments.py`)
   - Screenshots showing:
     * Your final Q-table heatmap
     * Learning curve plot
     * Final performance comparison (Q-learning vs random)

3. **Update Your README.md**: Add a new section for practical2 that includes:
   - **Implementation Summary**: Brief description of your Q-learning agent
   - **Performance Results**: Your final success rate and comparison to random baseline
   - **Exercise Answers**: Responses to the hyperparameter investigation questions
   - **Q-Table Analysis**: Your findings from Exercise 2 about Q-values
   - **Challenges**: What was the most difficult part of implementing Q-learning?
   - **Key Insights**: What surprised you most about how Q-learning works?
   - **Learning Reflection**: How does Q-learning compare to the random agent in terms of behavior?

4. **Performance Requirements**: Your submission should demonstrate:
   - Q-learning agent achieving >70% success rate on FrozenLake-v1
   - Clear learning curve showing improvement over episodes
   - Proper implementation of the Q-learning update rule
   - Meaningful analysis of hyperparameter effects

5. **Submit Your Repository URL**: Update the same Google Sheet entry you used for practical1 with a note that practical2 is now included.