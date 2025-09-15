# Study Guide: MLA202 Reinforcement Learning Midterm

This guide covers the key concepts and problem-solving skills you'll need for your midterm exam.

## Part A: Core Concepts

---

### 1. Markov Decision Processes (MDPs)

An MDP is the mathematical framework for modeling decision-making. You **must** know its five components.

- **Definition**: An MDP is a 5-tuple $(S, A, P, R, \gamma)$:

  - $S$: A finite set of **states**. (e.g., 'High Inventory', 'Low Inventory')
  - $A$: A finite set of **actions**. (e.g., 'Order More', 'Do Nothing')
  - $P$: The **transition probability model**, $P(s' | s, a)$. This is the probability of ending in state $s'$ if you take action $a$ in state $s$.
  - $R$: The **reward function**, $R(s, a, s')$. This is the immediate reward for transitioning from $s$ to $s'$ by taking action $a$.
  - $\gamma$: The **discount factor**, a number between 0 and 1.

- **The Markov Property**: This is a crucial assumption for MDPs. It states that the future is independent of the past, given the present. In other words, the next state and reward only depend on the current state and action, not on the entire history of previous states and actions.

---

### 2. Policies and Value Functions

These concepts are about how an agent acts and how it evaluates its situation.

- **Policy ($\pi$)**: A policy is the agent's strategy. It's a rule that specifies what **action** the agent will take for every possible **state**.
- **Value Function ($V(s)$)**: The value function, $V(s)$, represents the expected total future reward an agent can get starting from a state $s$ and following a specific policy. It tells you "how good" a state is.

---

### 3. The Bellman Equation

The Bellman equation is the most important formula in this section. It's fundamental because it **connects the value of a state to the value of the states that might follow it**.

- **Core Idea**: The value of your current state is the **immediate reward** you get plus the **discounted value** of whatever state you land in next.
- **MRP Formula**: The value of a state $s$ in a Markov Reward Process is the immediate reward for being in that state plus the discounted expected value of its successor states.
  $$V(s) = R(s) + \gamma \sum_{s'} P(s'|s)V(s')$$
  - $R(s)$ is the immediate reward for being in state $s$.
  - $\gamma$ is the discount factor.
  - $P(s'|s)$ is the probability of transitioning to state $s'$.
  - $V(s')$ is the value of the next state.

---

### 4. Key RL Concepts

- **Discount Factor ($\gamma$)**: The discount factor has two primary purposes:

  1.  It prioritizes **immediate rewards** over future rewards.
  2.  It ensures that the total reward remains a **finite value** in problems that could go on forever (infinite horizons).

- **Exploration vs. Exploitation Trade-off**: This is a central dilemma in RL.

  - **Exploitation**: Taking the action you currently believe is the best to maximize reward. (e.g., Going to your favorite restaurant).
  - **Exploration**: Taking a different action to see if it might lead to a better long-term reward. (e.g., Trying a new restaurant).
    > **Why do both?** An agent must exploit to use what it knows, but it must also explore to discover better strategies and avoid getting stuck with a sub-optimal choice.

- **Planning vs. Learning**:
  - **Planning Algorithms** (like Value Iteration) require a complete model of the MDP. The agent must know the transition probabilities ($P$) and reward function ($R$) in advance to "plan" its optimal policy.
  - **Learning Algorithms** do not need a model. They learn the best actions by directly interacting with the environment through trial and error.

## Part B: Problem-Solving Guide

This section provides a general method for tackling the formulation and calculation problems you will see in the exam.

---

### How to Formulate an MDP from a Word Problem 📝

Many problems will describe a scenario and ask you to model it as an MDP. Follow these steps:

1.  **Identify the States (S)**: Read the problem and find the distinct situations, statuses, or locations the agent can be in. Ask yourself: "What are the different conditions described?".
2.  **Identify the Actions (A)**: Find the choices the agent can make. These are the decisions that influence the state. Ask yourself: "What can the agent _do_?".
3.  **Determine the Reward Function (R)**: Look for any mention of profits, costs, points, or penalties. The reward is the immediate value gained or lost from an action.
    - If an outcome is uncertain (e.g., depends on customer demand), you may need to calculate the **expected reward**.
    - **Expected Reward = (Prob of Outcome 1 × Reward 1) + (Prob of Outcome 2 × Reward 2) + ...**.
4.  **Define the Transition Model (P)**: The problem will usually state the probabilities of moving between states. Clearly list them for each action. For example: "If in State X and taking Action Y, the probability of moving to State Z is 50%".

---

### How to Apply the Bellman Equation 🧮

If you are asked to calculate the value of a state ($V(s)$), use this checklist:

1.  **Write Down the Formula**: Start with the Bellman equation for an MRP:
    $$V(s) = R(s) + \gamma \sum_{s'} P(s'|s)V(s')$$
2.  **List All Knowns**: Identify and list every value given in the problem:
    - The immediate reward for your target state, $R(s)$.
    - The values of all possible successor states, $V(s')$.
    - The transition probabilities from your state to each successor state, $P(s'|s)$.
    - The discount factor, $\gamma$.
3.  **Substitute and Solve**: Carefully plug all the known values into the formula.
    - First, calculate the sum inside the brackets: the discounted expected value of the next states.
    - Then, add the immediate reward to get the final answer.
    - **Show your work clearly**, step by step, to avoid calculation errors.

---

### Sample Problem & Solution

Here is a new problem to practice these skills.

**Scenario**: A cleaning robot can be in one of three states: 'Charging', 'Working', or 'Broken' (a terminal state). An engineer wants to calculate the value of the 'Working' state.

**Given Information**:

- The immediate reward for being in the 'Working' state is +5.
- The robot's policy leads to the following transitions from the 'Working' state:
  - Probability of returning to 'Charging' is 80%.
  - Probability of moving to 'Broken' is 20%.
- The value of the 'Charging' state is known: $V(\text{Charging}) = 10$.
- The 'Broken' state is terminal, so its value is its reward: $V(\text{Broken}) = 0$.
- The discount factor is $\gamma = 0.9$.

**Task**: Using the Bellman equation, calculate the value of being in the 'Working' state, $V(\text{Working})$.

**Solution**:

1.  **Formula**:
    $$V(s) = R(s) + \gamma \sum_{s'} P(s'|s)V(s')$$

2.  **Known Values**:

    - $s = \text{Working}$
    - $R(\text{Working}) = +5$
    - $V(\text{Charging}) = 10$
    - $V(\text{Broken}) = 0$
    - $P(\text{Charging}|\text{Working}) = 0.8$
    - $P(\text{Broken}|\text{Working}) = 0.2$
    - $\gamma = 0.9$

3.  **Substitute and Solve**:
    $$V(\text{Working}) = R(\text{Working}) + \gamma \left[ P(\text{Charging}|\text{Working})V(\text{Charging}) + P(\text{Broken}|\text{Working})V(\text{Broken}) \right]$$ $$V(\text{Working}) = 5 + 0.9 \left[ (0.8 \times 10) + (0.2 \times 0) \right]$$ $$V(\text{Working}) = 5 + 0.9 \left[ 8 + 0 \right]$$ $$V(\text{Working}) = 5 + 0.9 \times 8$$ $$V(\text{Working}) = 5 + 7.2$$ $$V(\text{Working}) = 12.2$$

The value of the 'Working' state is **12.2**.
