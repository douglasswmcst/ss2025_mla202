# MLA202 Reinforcement Learning
## Concised Study Guide

---

## Exam Overview
| Detail | Information |
|--------|-------------|
| Duration | 2 Hours |
| Maximum Marks | 50 |
| Part A | 20 marks (All questions compulsory) |
| Part B | 30 marks (Q5 compulsory + ONE from Q6/Q7) |

---

# Domain 1: Foundations of Reinforcement Learning

## 1.1 What is Reinforcement Learning?

Reinforcement Learning (RL) is a computational approach to learning from interaction with an environment. Unlike supervised learning, RL agents learn through trial and error, receiving rewards or penalties based on their actions.

### The RL Framework

The agent-environment interaction follows this cycle:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    ┌─────────┐         Action (a_t)        ┌─────────┐ │
│    │         │ ────────────────────────────▶│         │ │
│    │  Agent  │                              │ Environ-│ │
│    │         │◀──────────────────────────── │  ment   │ │
│    └─────────┘   State (s_{t+1}), Reward    └─────────┘ │
│                        (r_{t+1})                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Definitions

| Term | Definition | Mathematical Notation |
|------|------------|----------------------|
| **State** | Complete description of the world | s ∈ S |
| **Action** | Decision made by the agent | a ∈ A |
| **Reward** | Scalar feedback signal | r ∈ ℝ |
| **Policy** | Mapping from states to actions | π(a\|s) |
| **Value Function** | Expected cumulative reward | V(s), Q(s,a) |
| **Return** | Cumulative discounted reward | G_t = Σ γ^k r_{t+k+1} |

### The Goal of RL

The agent's objective is to find a policy π that maximizes the expected return:

```
J(π) = E_π[G_0] = E_π[Σ_{t=0}^∞ γ^t R_{t+1}]
```

**Reference:** Sutton & Barto (2018), Chapter 1, "The Reinforcement Learning Problem"

---

## 1.2 Markov Decision Processes (MDPs)

An MDP provides the mathematical framework for modeling sequential decision-making problems where outcomes are partly random and partly under the control of the agent.

### Formal Definition

An MDP is defined by the 5-tuple **(S, A, P, R, γ)**:

| Component | Symbol | Description | Requirements |
|-----------|--------|-------------|--------------|
| **State Space** | S | Set of all possible states | Finite or continuous |
| **Action Space** | A | Set of all possible actions | May depend on state: A(s) |
| **Transition Function** | P(s'\|s,a) | Probability of next state given current state-action | Σ_{s'} P(s'\|s,a) = 1 |
| **Reward Function** | R(s,a,s') | Expected reward for transition | Bounded: \|R\| < R_max |
| **Discount Factor** | γ ∈ [0,1] | Weight for future rewards | γ < 1 for infinite horizons |

### The Markov Property

The key assumption underlying MDPs is the **Markov Property**:

```
P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0, A_0) = P(S_{t+1} | S_t, A_t)
```

**In plain English:** The future depends only on the present state and action, not on the history of how we got there.

### Detailed Example: Robot Navigation in 4×4 Grid

Consider a robot navigating a 4×4 grid world:

```
┌─────┬─────┬─────┬─────┐
│  0  │  1  │  2  │  3  │
├─────┼─────┼─────┼─────┤
│  4  │  5  │ OBS │  7  │
├─────┼─────┼─────┼─────┤
│  8  │  9  │ 10  │ 11  │
├─────┼─────┼─────┼─────┤
│ 12  │ 13  │ 14  │GOAL │
└─────┴─────┴─────┴─────┘
```

**State Space (S):**
```
S = {s_0, s_1, s_2, ..., s_15} 
or equivalently:
S = {(x,y) | x ∈ {0,1,2,3}, y ∈ {0,1,2,3}}

Additional state features might include:
- Robot orientation (if turning is an action)
- Battery level (for realistic scenarios)
- Carrying status (for pickup/delivery tasks)
```

**Action Space (A):**
```
A = {UP, DOWN, LEFT, RIGHT}
or numerically: A = {0, 1, 2, 3}

Movement vectors:
- UP:    Δ = (0, -1)
- DOWN:  Δ = (0, +1)
- LEFT:  Δ = (-1, 0)
- RIGHT: Δ = (+1, 0)
```

**Transition Function P(s'|s,a):**

*Deterministic Version:*
```
P(s'|s,a) = {
    1  if s' = result of applying action a in state s
    0  otherwise
}

Special cases:
- At boundary: P(s|s,a) = 1 (stay in place)
- At obstacle: P(s|s,toward_obstacle) = 1 (stay in place)
```

*Stochastic Version (more realistic):*
```
P(intended_direction | s, a) = 0.8
P(left_perpendicular | s, a) = 0.1
P(right_perpendicular | s, a) = 0.1

Example: Action = RIGHT
- P(move right) = 0.8
- P(move up) = 0.1
- P(move down) = 0.1
```

**Reward Function R(s,a,s'):**
```
R(s, a, s') = {
    +100   if s' = GOAL                    (task completion)
    -100   if s' = OBSTACLE                (collision penalty)
    -1     otherwise                        (step penalty for efficiency)
}

Alternative reward shaping:
R(s, a, s') = -1 + 100·𝟙[s'=GOAL] - 100·𝟙[s'=OBS] - 0.1·d(s', GOAL)
where d(s', GOAL) is Manhattan distance to goal
```

**Discount Factor (γ):**
```
γ = 0.95 (typical for navigation tasks)

Interpretation:
- Reward 10 steps away is worth: 0.95^10 × R ≈ 0.60 × R
- Reward 50 steps away is worth: 0.95^50 × R ≈ 0.08 × R

Choosing γ:
- γ close to 1: Long-term planning (patient agent)
- γ close to 0: Short-term focus (myopic agent)
- γ = 1: Only valid for finite-horizon episodic tasks
```

**Reference:** Sutton & Barto (2018), Chapter 3, "Finite Markov Decision Processes"

---

## 1.3 Value Functions

Value functions estimate "how good" it is for the agent to be in a given state or to perform a given action in a given state.

### State-Value Function V^π(s)

The state-value function V^π(s) gives the expected return when starting from state s and following policy π:

```
V^π(s) = E_π[G_t | S_t = s]
       = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]
```

**Recursive Form (Bellman Expectation Equation):**
```
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
```

### Action-Value Function Q^π(s,a)

The action-value function Q^π(s,a) gives the expected return when starting from state s, taking action a, then following policy π:

```
Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
         = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s, A_t = a]
```

**Recursive Form:**
```
Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ Σ_{a'} π(a'|s')Q^π(s',a')]
```

### Relationship Between V and Q

```
V^π(s) = Σ_a π(a|s) Q^π(s,a)    [V is weighted average of Q]

Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')   [Q is reward + discounted V]
```

### Optimal Value Functions

The optimal state-value function V*(s) is the maximum value achievable from state s:
```
V*(s) = max_π V^π(s) = max_a Q*(s,a)
```

The optimal action-value function Q*(s,a) is the maximum value achievable from state s taking action a:
```
Q*(s,a) = max_π Q^π(s,a)
```

**Reference:** Sutton & Barto (2018), Chapter 3.5, "Value Functions"

---

# Domain 2: Bellman Equations

## 2.1 The Bellman Expectation Equations

The Bellman equations express a fundamental relationship: the value of a state equals the immediate reward plus the discounted value of successor states.

### For State-Value Function:
```
V^π(s) = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
       = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
```

### For Action-Value Function:
```
Q^π(s,a) = E[R_{t+1} + γQ^π(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
         = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ Σ_{a'} π(a'|s')Q^π(s',a')]
```

## 2.2 The Bellman Optimality Equations

### State-Value Bellman Optimality Equation:
```
V*(s) = max_a E[R_{t+1} + γV*(S_{t+1}) | S_t = s, A_t = a]
      = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]
```

**Interpretation:** The optimal value of a state is the expected value of the best action.

### Action-Value Bellman Optimality Equation:
```
Q*(s,a) = E[R_{t+1} + γ max_{a'} Q*(S_{t+1}, a') | S_t = s, A_t = a]
        = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ max_{a'} Q*(s',a')]
```

**Interpretation:** The optimal value of an action is the immediate reward plus the value of the best action from the next state.

## 2.3 Backup Diagrams

Visual representation of value function relationships:

```
State-Value Backup (V):              Action-Value Backup (Q):

        V(s)                                 Q(s,a)
         │                                     │
    ┌────┴────┐                               │
    ▼         ▼                               ▼
  π(a₁|s)   π(a₂|s)              ┌────────────┼────────────┐
    │         │                   │            │            │
    ▼         ▼                   ▼            ▼            ▼
Q(s,a₁)   Q(s,a₂)             P(s'₁)      P(s'₂)       P(s'₃)
                                  │            │            │
                                  ▼            ▼            ▼
                              R+γV(s'₁)   R+γV(s'₂)   R+γV(s'₃)
```

## 2.4 Role in Solving MDPs

| Application | How Bellman Equations Are Used |
|-------------|-------------------------------|
| **Value Iteration** | Iteratively apply Bellman optimality operator until convergence |
| **Policy Iteration** | Alternate between policy evaluation (Bellman expectation) and improvement |
| **TD Learning** | Sample-based approximation of Bellman update |
| **Q-Learning** | Sample-based Bellman optimality update for Q-values |

**Contraction Property:**
The Bellman operator T is a contraction mapping with factor γ:
```
||TV - TV'||_∞ ≤ γ||V - V'||_∞
```

This guarantees convergence to a unique fixed point V*.

**Reference:** Sutton & Barto (2018), Chapter 3.6-3.8; Bertsekas & Tsitsiklis (1996), Chapter 2

---

# Domain 3: Temporal Difference Learning

## 3.1 TD(0) Algorithm

TD learning combines ideas from Monte Carlo (learning from experience) and Dynamic Programming (bootstrapping from estimates).

### The TD(0) Update Rule

```
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
                   └──────────────────────────────┘
                            TD Error (δ_t)
```

Where:
- **α**: Learning rate (step size), typically 0.01 - 0.3
- **γ**: Discount factor
- **δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)**: The TD error

### Derivation from Bellman Equation

**Step 1: Start with Bellman Expectation Equation**
```
V^π(s) = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
```

**Step 2: Replace expectation with sample**
Instead of computing the full expectation, use a single sample:
```
V^π(s) ≈ r + γV(s')
```
where r and s' are observed from taking action under policy π.

**Step 3: Move current estimate toward sample**
```
V(s) ← V(s) + α[r + γV(s') - V(s)]
```

This is stochastic gradient descent on the squared TD error.

### TD(0) Algorithm Pseudocode

```
Algorithm: TD(0) for Policy Evaluation
─────────────────────────────────────────────────────
Input: Policy π to be evaluated
Initialize: V(s) arbitrarily for all s ∈ S
            V(terminal) = 0
Parameters: α ∈ (0, 1], γ ∈ [0, 1]

Repeat for each episode:
    Initialize S
    Repeat for each step of episode:
        A ← action given by π for S
        Take action A, observe R, S'
        
        V(S) ← V(S) + α[R + γV(S') - V(S)]
        
        S ← S'
    Until S is terminal
─────────────────────────────────────────────────────
```

### Why TD Can Learn from Incomplete Episodes

**Key Insight: Bootstrapping**

| Aspect | Monte Carlo | TD Learning |
|--------|-------------|-------------|
| **Target** | G_t (actual return) | R_{t+1} + γV(S_{t+1}) |
| **When computed** | End of episode | Every step |
| **Requires** | Complete episode | Single transition |
| **Works for** | Episodic tasks only | Episodic and continuing |

TD "bootstraps" by using the current estimate V(S') as a stand-in for the true expected future return. This allows:
1. Updates after every single step
2. Learning in continuing (non-episodic) tasks
3. Faster learning in long episodes

**Reference:** Sutton & Barto (2018), Chapter 6, "Temporal-Difference Learning"

---

## 3.2 Monte Carlo Methods

Monte Carlo methods learn from complete episodes of experience.

### First-Visit MC

```
Algorithm: First-Visit MC Prediction
─────────────────────────────────────────────────────
Initialize: V(s) arbitrarily, Returns(s) = empty list

Repeat forever:
    Generate episode using π: S₀, A₀, R₁, S₁, A₁, R₂, ..., S_T
    G ← 0
    
    For t = T-1, T-2, ..., 0:
        G ← γG + R_{t+1}
        
        If S_t not in {S₀, S₁, ..., S_{t-1}}:  # First visit check
            Append G to Returns(S_t)
            V(S_t) ← average(Returns(S_t))
─────────────────────────────────────────────────────
```

### Every-Visit MC

```
Same as First-Visit, but remove the "If S_t not in..." check.
Update V(S_t) for EVERY occurrence of S_t in the episode.
```

### Comparison: First-Visit vs Every-Visit

| Property | First-Visit MC | Every-Visit MC |
|----------|---------------|----------------|
| **Samples per episode** | At most one per state | Multiple possible |
| **Bias** | Unbiased | Slight bias for finite samples |
| **Variance** | Generally lower | Generally higher |
| **Convergence** | Converges to V^π | Converges to V^π |

**Reference:** Sutton & Barto (2018), Chapter 5, "Monte Carlo Methods"

---

## 3.3 TD vs Monte Carlo: Detailed Comparison

### Bias-Variance Tradeoff

```
                    ┌─────────────────────────────────────┐
                    │              High Variance          │
                    │                                     │
                    │    Monte Carlo                      │
                    │    • Uses actual returns            │
                    │    • Many random variables          │
                    │    • Unbiased but noisy             │
                    │                                     │
    Bias ◄─────────┼─────────────────────────────────────┤
                    │                                     │
                    │    TD Learning                      │
                    │    • Bootstraps from estimates      │
                    │    • Fewer random variables         │
                    │    • Biased but lower variance      │
                    │                                     │
                    │              Low Variance           │
                    └─────────────────────────────────────┘
```

### Mathematical Comparison

**Monte Carlo Target:**
```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... + γ^{T-t-1}R_T
```
- Involves T-t random variables (rewards)
- Unbiased: E[G_t] = V^π(S_t)
- High variance due to many random terms

**TD(0) Target:**
```
R_{t+1} + γV(S_{t+1})
```
- Involves only 2 random variables (R_{t+1} and S_{t+1})
- Biased: E[R_{t+1} + γV(S_{t+1})] ≠ V^π(S_t) (unless V = V^π)
- Lower variance due to fewer random terms

### Comprehensive Comparison Table

| Property | Monte Carlo | TD(0) |
|----------|-------------|-------|
| **Bias** | Unbiased estimate | Biased (due to bootstrapping) |
| **Variance** | High (full trajectory) | Low (single step) |
| **Data efficiency** | Low (needs complete episodes) | High (learns every step) |
| **Episodic tasks** | Required | Not required |
| **Continuing tasks** | Cannot handle | Handles naturally |
| **Convergence** | Guaranteed with enough samples | Guaranteed under conditions |
| **Sensitivity to initial values** | None | Present |
| **Online learning** | Not possible | Natural |
| **Markov property** | Doesn't require | Exploits it |

### When to Use Each

**Use Monte Carlo when:**
- Episodes are short
- Environment is non-Markovian
- You need unbiased estimates
- Sample efficiency is not critical

**Use TD when:**
- Episodes are long or infinite
- Environment is Markovian
- Online learning is needed
- Sample efficiency matters

**Reference:** Sutton & Barto (2018), Chapter 6.1-6.2; Chapter 7 for n-step methods

---

# Domain 4: Q-Learning and SARSA

## 4.1 Q-Learning (Off-Policy TD Control)

Q-learning learns the optimal action-value function Q* directly, regardless of the policy being followed.

### The Q-Learning Update Rule

```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
                             └────────────────────────────────────────────┘
                                           TD Target
```

**Key Feature:** Uses **max** over next actions (greedy with respect to Q)

### Complete Q-Learning Algorithm

```
Algorithm: Q-Learning (Off-policy TD Control)
─────────────────────────────────────────────────────────────────────
Initialize: Q(s,a) arbitrarily for all s ∈ S, a ∈ A
            Q(terminal, ·) = 0
Parameters: α ∈ (0, 1], γ ∈ [0, 1], ε ∈ (0, 1]

Repeat for each episode:
    Initialize S (starting state)
    
    Repeat for each step of episode:
        ┌─────────────────────────────────────────────────────────┐
        │ Action Selection (ε-greedy):                            │
        │   With probability ε:                                   │
        │       A ← random action from A                          │
        │   Else:                                                 │
        │       A ← argmax_a Q(S, a)                              │
        └─────────────────────────────────────────────────────────┘
        
        Take action A, observe R, S'
        
        ┌─────────────────────────────────────────────────────────┐
        │ Q-Value Update:                                         │
        │   Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]    │
        └─────────────────────────────────────────────────────────┘
        
        S ← S'
    Until S is terminal
─────────────────────────────────────────────────────────────────────
```

### Why Q-Learning is Off-Policy

The behavior policy (ε-greedy) differs from the target policy (greedy/optimal):
- **Behavior policy:** What we actually do (includes exploration)
- **Target policy:** What we're learning about (always greedy)

```
Behavior: π_b(a|s) = {  ε/|A|           for all actions
                      { 1 - ε + ε/|A|   for greedy action

Target:   π_t(a|s) = { 1   if a = argmax_a' Q(s,a')
                     { 0   otherwise
```

**Reference:** Watkins (1989), "Learning from Delayed Rewards" (PhD Thesis); Sutton & Barto (2018), Chapter 6.5

---

## 4.2 SARSA (On-Policy TD Control)

SARSA learns the value of the policy being followed, including exploration.

### The SARSA Update Rule

```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

**Name origin:** Uses (S, A, R, S', A') - hence "SARSA"

### Complete SARSA Algorithm

```
Algorithm: SARSA (On-policy TD Control)
─────────────────────────────────────────────────────────────────────
Initialize: Q(s,a) arbitrarily for all s ∈ S, a ∈ A
            Q(terminal, ·) = 0
Parameters: α ∈ (0, 1], γ ∈ [0, 1], ε ∈ (0, 1]

Repeat for each episode:
    Initialize S
    Choose A from S using policy derived from Q (e.g., ε-greedy)
    
    Repeat for each step of episode:
        Take action A, observe R, S'
        Choose A' from S' using policy derived from Q (e.g., ε-greedy)
        
        ┌─────────────────────────────────────────────────────────┐
        │ SARSA Update:                                           │
        │   Q(S,A) ← Q(S,A) + α[R + γ Q(S',A') - Q(S,A)]         │
        │                              │                          │
        │                    Uses actual next action A'           │
        └─────────────────────────────────────────────────────────┘
        
        S ← S'
        A ← A'
    Until S is terminal
─────────────────────────────────────────────────────────────────────
```

**Reference:** Rummery & Niranjan (1994), "On-Line Q-Learning Using Connectionist Systems"

---

## 4.3 Expected SARSA

Expected SARSA takes the expectation over next actions instead of sampling.

### The Expected SARSA Update Rule

```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ Σ_a π(a|S_{t+1})Q(S_{t+1},a) - Q(S_t, A_t)]
                                         └───────────────────────────┘
                                           Expected value over policy
```

### Advantages of Expected SARSA

| Comparison | Expected SARSA Advantage |
|------------|-------------------------|
| vs SARSA | Lower variance (expectation vs sample) |
| vs Q-learning | Can be on-policy, safer in risky environments |
| Flexibility | With greedy policy, becomes Q-learning |

**Reference:** van Seijen et al. (2009), "A Theoretical and Empirical Analysis of Expected SARSA"

---

## 4.4 SARSA vs Q-Learning: Critical Comparison

### Side-by-Side Update Rules

```
SARSA:      Q(s,a) ← Q(s,a) + α[r + γ Q(s', a')         - Q(s,a)]
                                     └──────┘
                                   Actual next action

Q-learning: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s', a') - Q(s,a)]
                                     └───────────────┘
                                    Best possible action
```

### Behavioral Differences

```
                     Cliff Walking Example
    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    │Start│     │     │     │     │     │     │ Goal│
    ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
    │CLIFF│CLIFF│CLIFF│CLIFF│CLIFF│CLIFF│CLIFF│CLIFF│
    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
    
    Q-learning path (optimal but risky):
    Start → → → → → → → Goal
              ↑ walks along cliff edge
    
    SARSA path (safer):
    Start → ↑ → → → → → ↓ Goal
            walks away from cliff
```

### Comprehensive Comparison

| Aspect | SARSA | Q-Learning |
|--------|-------|------------|
| **Policy Type** | On-policy | Off-policy |
| **Update Uses** | Actual next action a' | Best action max_a' |
| **Learns** | Value of current policy | Optimal policy value |
| **During Learning** | Policy is reasonable | May be dangerous |
| **Exploration Effect** | Accounts for exploration | Ignores exploration risk |
| **Convergence** | To Q^π (policy value) | To Q* (optimal value) |
| **Sample Efficiency** | Lower | Higher (can reuse data) |
| **Safety** | Safer | May take risks |

### When to Choose Each

**Choose SARSA when:**
- Safety during learning is critical (robotics, autonomous vehicles)
- The agent must perform well while learning (online systems)
- Exploration has real costs or risks
- You want to learn the value of a stochastic policy

**Choose Q-Learning when:**
- Finding the optimal policy is the goal
- You can learn from historical data (offline RL)
- Exploration mistakes are acceptable
- Sample efficiency matters (experience replay)

**Reference:** Sutton & Barto (2018), Chapter 6.4-6.5

---

# Domain 5: Exploration vs Exploitation

## 5.1 The Exploration-Exploitation Dilemma

The fundamental tension in RL:
- **Exploitation:** Use current knowledge to maximize reward
- **Exploration:** Try new actions to gain information

```
             Exploitation                    Exploration
                 │                               │
    "Go to your favorite               "Try that new restaurant
     restaurant"                        you've never been to"
                 │                               │
                 ▼                               ▼
    ┌───────────────────────┐     ┌───────────────────────┐
    │ • Guaranteed decent   │     │ • Might find better   │
    │   reward              │     │   options             │
    │ • No new information  │     │ • Risk of bad outcome │
    │ • May miss better     │     │ • Reduces uncertainty │
    │   options             │     │                       │
    └───────────────────────┘     └───────────────────────┘
```

## 5.2 Exploration Strategies

### ε-Greedy

The simplest and most common exploration strategy.

```
Action Selection:
    With probability ε:
        A ← random action from A(s)      # Explore
    With probability 1-ε:
        A ← argmax_a Q(s,a)              # Exploit
```

**Probability Distribution:**
```
π(a|s) = {
    1 - ε + ε/|A|    if a = argmax_a' Q(s,a')  (greedy action)
    ε/|A|            otherwise                   (non-greedy actions)
}
```

**ε Decay Schedule:**
```
Exponential decay: ε_t = max(ε_min, ε_0 × decay^t)
Linear decay:      ε_t = max(ε_min, ε_0 - decay × t)
1/t decay:         ε_t = max(ε_min, ε_0 / (1 + decay × t))

Typical values:
- ε_0 = 1.0 (start fully random)
- ε_min = 0.01 or 0.05 (always some exploration)
- decay = 0.995-0.9999 (for exponential)
```

### Boltzmann (Softmax) Exploration

Actions selected with probability proportional to their estimated value.

```
π(a|s) = exp(Q(s,a)/τ) / Σ_a' exp(Q(s,a')/τ)
```

**Temperature Parameter τ:**
- τ → ∞: Uniform random (maximum exploration)
- τ → 0: Greedy (maximum exploitation)
- τ = 1: Probabilities proportional to exp(Q)

**Advantage:** Rarely-chosen actions with high Q still get chosen
**Disadvantage:** Sensitive to Q-value scale; requires tuning τ

### Upper Confidence Bound (UCB)

Select actions based on optimistic estimates that account for uncertainty.

```
A_t = argmax_a [Q(s,a) + c√(ln(t)/N(s,a))]
                └──────┘   └────────────────┘
                 Exploit     Exploration bonus
```

Where:
- N(s,a) = number of times action a taken in state s
- c = exploration constant (typically c = 2)
- t = total timesteps

**Key Idea:** "Optimism in the face of uncertainty" - favor uncertain actions.

### Thompson Sampling

Bayesian approach: sample from posterior distribution of Q-values.

```
For each action a:
    Sample Q̃(s,a) ~ P(Q(s,a) | history)
Select A = argmax_a Q̃(s,a)
```

**Advantage:** Naturally balances exploration and exploitation based on uncertainty.

### Comparison Table

| Strategy | Key Feature | Pros | Cons |
|----------|------------|------|------|
| **ε-greedy** | Fixed probability ε | Simple, robust | All non-greedy actions equal |
| **Boltzmann** | Probability ∝ Q-value | Considers action quality | Sensitive to τ and Q scale |
| **UCB** | Uncertainty bonus | Theoretical guarantees | Requires action counts |
| **Thompson** | Posterior sampling | Optimal exploration | Computationally expensive |

**Reference:** Sutton & Barto (2018), Chapter 2; Auer et al. (2002) for UCB

---

# Domain 6: Maximization Bias and Double Q-Learning

## 6.1 The Maximization Bias Problem

### Understanding the Problem

When we use the **same samples** to both **select** the best action AND **evaluate** its value, we systematically **overestimate** Q-values.

### Mathematical Explanation

Suppose we have noisy estimates of Q-values:
```
Q̂(s,a) = Q*(s,a) + noise_a
```

When we take the maximum:
```
max_a Q̂(s,a) = max_a [Q*(s,a) + noise_a]
              ≥ Q*(s, a*) + noise_{a*}    (usually with strict inequality)
              = Q̂(s, a*) where a* = argmax_a Q*(s,a)
```

The max operator tends to select actions where noise is positive, leading to:
```
E[max_a Q̂(s,a)] ≥ max_a E[Q̂(s,a)] = max_a Q*(s,a)
```

### Concrete Example

```
True Q-values:    Q*(s, a1) = 0,  Q*(s, a2) = 0,  Q*(s, a3) = 0

Estimates (Run 1): Q̂(s, a1) = -1, Q̂(s, a2) = +2, Q̂(s, a3) = -1
                   max = +2 (overestimate of true max = 0)

Estimates (Run 2): Q̂(s, a1) = +1, Q̂(s, a2) = -2, Q̂(s, a3) = +3
                   max = +3 (overestimate of true max = 0)

Average of max estimates: (+2 + +3) / 2 = +2.5
True value: 0
Overestimation: +2.5
```

## 6.2 Double Q-Learning Solution

### Key Insight

**Decouple action selection from action evaluation** by maintaining two Q-functions.

### Algorithm

```
Algorithm: Double Q-Learning
─────────────────────────────────────────────────────────────────────
Initialize: Q_A(s,a) and Q_B(s,a) arbitrarily for all s, a
Parameters: α ∈ (0, 1], γ ∈ [0, 1], ε ∈ (0, 1]

Repeat for each episode:
    Initialize S
    
    Repeat for each step of episode:
        # Use average of Q_A and Q_B for action selection
        Choose A from S using ε-greedy on (Q_A(S,·) + Q_B(S,·))/2
        
        Take action A, observe R, S'
        
        # Randomly update one Q-function
        With probability 0.5:
            ┌───────────────────────────────────────────────────┐
            │ Update Q_A:                                       │
            │   a* = argmax_a Q_A(S', a)    ← Q_A selects      │
            │   Q_A(S,A) ← Q_A(S,A) + α[R + γ Q_B(S',a*) - Q_A(S,A)]
            │                                   └────────────┘  │
            │                                   Q_B evaluates   │
            └───────────────────────────────────────────────────┘
        Else:
            ┌───────────────────────────────────────────────────┐
            │ Update Q_B:                                       │
            │   a* = argmax_a Q_B(S', a)    ← Q_B selects      │
            │   Q_B(S,A) ← Q_B(S,A) + α[R + γ Q_A(S',a*) - Q_B(S,A)]
            │                                   └────────────┘  │
            │                                   Q_A evaluates   │
            └───────────────────────────────────────────────────┘
        
        S ← S'
    Until S is terminal
─────────────────────────────────────────────────────────────────────
```

### Why This Works

```
Standard Q-learning:
    Target = R + γ max_a Q(S',a)
                   └─────────────┘
                   Same Q for select AND evaluate → Maximization bias

Double Q-learning:
    a* = argmax_a Q_A(S',a)    ← Q_A selects best action
    Target = R + γ Q_B(S', a*)
                   └──────────┘
                   Q_B evaluates → No maximization bias
```

Because Q_A and Q_B have **independent noise**, Q_B's estimate of Q_A's selected action is unbiased.

**Reference:** Hasselt (2010), "Double Q-learning"; Hasselt et al. (2016), "Deep Reinforcement Learning with Double Q-learning"

---

# Domain 7: Deep Q-Networks (DQN)

## 7.1 Motivation: Why Deep Learning for RL?

### The Curse of Dimensionality

Tabular methods store Q(s,a) for every state-action pair:
- Atari games: ~10^67 possible states
- Go: ~10^170 board positions
- Robotics: Continuous state spaces (infinite)

**Solution:** Use function approximation to generalize across states.

## 7.2 DQN Architecture

### Network Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DQN Architecture                              │
│                                                                      │
│  ┌──────────┐    ┌─────────────────────────┐    ┌──────────────┐   │
│  │  Input   │    │   Convolutional Layers  │    │    Fully     │   │
│  │  State   │───▶│   (Feature Extraction)  │───▶│   Connected  │   │
│  │ 84×84×4  │    │                         │    │    Layers    │   │
│  └──────────┘    │  Conv1: 32 8×8, stride 4│    │              │   │
│                  │  Conv2: 64 4×4, stride 2│    │  FC1: 512    │   │
│                  │  Conv3: 64 3×3, stride 1│    │  FC2: |A|    │   │
│                  └─────────────────────────┘    └──────────────┘   │
│                                                        │            │
│                                                        ▼            │
│                                               ┌────────────────┐   │
│                                               │ Q(s,a) for all │   │
│                                               │    actions     │   │
│                                               │ [Q(s,a₁), ..., │   │
│                                               │  Q(s,aₙ)]      │   │
│                                               └────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Input Preprocessing (for Atari)

```
Raw frame (210×160 RGB) → Grayscale → Resize (84×84) → Stack 4 frames
```

Why 4 frames? To capture temporal information (velocity, direction).

## 7.3 Key Innovations

### Innovation 1: Experience Replay

**Problem:** Neural networks assume i.i.d. training data, but RL experiences are:
- Temporally correlated (s' follows s)
- Non-stationary (policy changes during training)

**Solution: Replay Buffer**

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Experience Replay                              │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    Replay Buffer D                          │  │
│   │  ┌────────────────────────────────────────────────────────┐ │  │
│   │  │ (s₁,a₁,r₁,s'₁,done₁), (s₂,a₂,r₂,s'₂,done₂), ...      │ │  │
│   │  │ (s₃,a₃,r₃,s'₃,done₃), (s₄,a₄,r₄,s'₄,done₄), ...      │ │  │
│   │  │ ...                   (capacity: ~1,000,000)           │ │  │
│   │  └────────────────────────────────────────────────────────┘ │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              │ Random sample minibatch              │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  Minibatch of 32 transitions (random, not sequential)       │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  Train neural network on this minibatch                     │  │
│   └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Benefits:**
1. Breaks temporal correlation through random sampling
2. Reuses experiences multiple times (sample efficiency)
3. Avoids catastrophic forgetting of rare experiences

### Innovation 2: Target Network

**Problem: Moving Target**
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
                         └─────────────┘
                         This changes as we update Q!
```

The target depends on Q, which we're updating → oscillations/divergence.

**Solution: Separate Target Network**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Target Network                                │
│                                                                      │
│   Online Network (θ)              Target Network (θ⁻)               │
│   ┌─────────────────┐             ┌─────────────────┐               │
│   │ Updated every   │             │ Updated every   │               │
│   │ gradient step   │             │ C steps         │               │
│   │                 │   Copy θ    │                 │               │
│   │    Q(s,a;θ)    │────────────▶│   Q(s,a;θ⁻)    │               │
│   │                 │  every C    │                 │               │
│   └─────────────────┘   steps     └─────────────────┘               │
│          │                               │                          │
│          │ Used for                      │ Used for                 │
│          │ action selection              │ computing targets        │
│          ▼                               ▼                          │
│   a = argmax_a Q(s,a;θ)          y = r + γ max_a' Q(s',a';θ⁻)      │
└─────────────────────────────────────────────────────────────────────┘
```

**Typical value:** C = 10,000 steps

### DQN Loss Function

```
L(θ) = E[(y - Q(s,a;θ))²]

where y = r + γ max_a' Q(s',a';θ⁻)   [target, computed with θ⁻]
```

## 7.4 Complete DQN Algorithm

```
Algorithm: Deep Q-Network (DQN)
─────────────────────────────────────────────────────────────────────
Initialize: 
    - Replay buffer D with capacity N
    - Action-value network Q with random weights θ
    - Target network Q̂ with weights θ⁻ = θ

For episode = 1 to M:
    Initialize state s₁ (preprocessed sequence of frames)
    
    For t = 1 to T:
        # ε-greedy action selection
        With probability ε:
            a_t = random action
        Else:
            a_t = argmax_a Q(s_t, a; θ)
        
        # Execute action and observe
        Execute a_t, observe r_t and next state s_{t+1}
        
        # Store transition in replay buffer
        Store (s_t, a_t, r_t, s_{t+1}, done) in D
        
        # Sample random minibatch
        Sample minibatch of transitions (s_j, a_j, r_j, s'_j, done_j) from D
        
        # Compute targets
        For each transition j:
            If done_j:
                y_j = r_j
            Else:
                y_j = r_j + γ max_a' Q̂(s'_j, a'; θ⁻)
        
        # Gradient descent step
        Perform gradient descent on (y_j - Q(s_j, a_j; θ))²
        
        # Update target network
        Every C steps: θ⁻ ← θ
─────────────────────────────────────────────────────────────────────
```

**Reference:** Mnih et al. (2015), "Human-level control through deep reinforcement learning", Nature

---

## 7.5 DQN Variants

### Double DQN

**Problem:** DQN still suffers from maximization bias (uses max in target).

**Solution:** Use online network to select action, target network to evaluate.

```
Standard DQN target:
    y = r + γ max_a' Q(s', a'; θ⁻)
              └─────────────────┘
              Target network selects AND evaluates

Double DQN target:
    a* = argmax_a' Q(s', a'; θ)     ← Online network selects
    y = r + γ Q(s', a*; θ⁻)         ← Target network evaluates
```

### Dueling DQN

**Key Insight:** Some states are valuable regardless of action taken.

**Architecture:**

```
                    ┌───────────────────────────┐
                    │                           │
     State ────────▶│   Shared Convolutional   │
                    │        Layers             │
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
          ┌─────────────────┐       ┌─────────────────┐
          │  Value Stream   │       │ Advantage Stream│
          │     V(s)        │       │     A(s,a)      │
          │   (scalar)      │       │   (|A| values)  │
          └────────┬────────┘       └────────┬────────┘
                   │                         │
                   └────────────┬────────────┘
                                │
                                ▼
                    ┌───────────────────────────┐
                    │  Q(s,a) = V(s) + A(s,a)  │
                    │        - mean_a A(s,a)   │
                    └───────────────────────────┘
```

**Combining Function:**
```
Q(s,a) = V(s) + [A(s,a) - (1/|A|) Σ_a' A(s,a')]
```

The mean subtraction ensures identifiability (unique decomposition).

### Comparison of DQN Variants

| Variant | Innovation | Benefit |
|---------|------------|---------|
| **DQN** | Experience replay + Target network | Stable training |
| **Double DQN** | Decoupled selection/evaluation | Reduces overestimation |
| **Dueling DQN** | Separate V and A streams | Better for many similar-valued actions |
| **Prioritized ER** | Sample important transitions more | Faster learning |
| **Rainbow** | Combines all improvements | State-of-the-art |

**Reference:** Hasselt et al. (2016); Wang et al. (2016), "Dueling Network Architectures"

---

# Domain 8: Monte Carlo Control

## 8.1 Monte Carlo Control with ε-Soft Policies

### The Algorithm

```
Algorithm: On-Policy First-Visit MC Control (ε-soft)
─────────────────────────────────────────────────────────────────────
Initialize:
    Q(s,a) ← arbitrary for all s ∈ S, a ∈ A
    Returns(s,a) ← empty list for all s, a
    π ← ε-soft policy (e.g., ε-greedy with respect to Q)

Repeat forever:
    # Generate episode using π
    Generate episode: S₀, A₀, R₁, S₁, A₁, R₂, ..., S_{T-1}, A_{T-1}, R_T
    
    G ← 0
    
    # Process episode backwards
    For t = T-1, T-2, ..., 0:
        G ← γG + R_{t+1}
        
        # First-visit check
        If (S_t, A_t) not in {(S₀,A₀), (S₁,A₁), ..., (S_{t-1},A_{t-1})}:
            
            # Update Q-value
            Append G to Returns(S_t, A_t)
            Q(S_t, A_t) ← average(Returns(S_t, A_t))
            
            # Policy improvement (ε-greedy)
            A* ← argmax_a Q(S_t, a)
            
            For all a ∈ A(S_t):
                If a = A*:
                    π(a|S_t) ← 1 - ε + ε/|A(S_t)|
                Else:
                    π(a|S_t) ← ε/|A(S_t)|
─────────────────────────────────────────────────────────────────────
```

### Key Properties

- **ε-soft policy:** π(a|s) ≥ ε/|A| for all actions (ensures exploration)
- **Convergence:** Converges to best ε-soft policy (not optimal greedy policy)
- **Exploration:** Maintained throughout learning (never becomes fully greedy)

**Reference:** Sutton & Barto (2018), Chapter 5.4

---

# Domain 9: Business Applications of RL

## 9.1 Dynamic Pricing in E-Commerce

### Problem Formulation

Design an RL system that automatically adjusts prices to maximize revenue while considering market conditions and competition.

### MDP Design

**State Space (Multi-dimensional):**
```
s = (demand_level,           # Current demand: {low, medium, high}
     competitor_price,       # Discretized: {cheaper, similar, expensive}
     inventory_level,        # Stock: {critical, low, medium, high}
     time_features,          # Hour of day, day of week, season
     customer_segment,       # {budget, regular, premium}
     price_elasticity)       # Historical responsiveness to price changes
```

**Action Space:**
```
Discrete: A = {$9.99, $14.99, $19.99, $24.99, $29.99}

Or relative: A = {-20%, -10%, -5%, 0%, +5%, +10%, +20%}

Or continuous: a ∈ [price_min, price_max]
```

**Reward Function:**
```
R = Revenue - Costs + λ₁·Customer_Satisfaction + λ₂·Long_Term_Value
  = (Price × Quantity_Sold) - (COGS + Holding_Costs) 
    + λ₁·(1 - Stockout_Rate) + λ₂·Customer_Lifetime_Value
```

**Considerations:**
- Balance short-term revenue with long-term customer relationships
- Account for inventory costs and stockouts
- Consider competitive dynamics

### Real-World Example: Uber/Lyft Surge Pricing

- **State:** Demand/supply ratio per geographic zone, time, weather, events
- **Action:** Surge multiplier (1.0×, 1.5×, 2.0×, etc.)
- **Reward:** Total rides completed, driver utilization, customer satisfaction
- **Challenge:** Multi-agent system with interdependent zones

## 9.2 Portfolio Management

### RL Formulation

**State Space:**
```
s = (asset_prices,           # Current prices of all assets
     portfolio_weights,      # Current allocation
     technical_indicators,   # Moving averages, momentum, volatility
     fundamental_data,       # P/E ratios, earnings, etc.
     market_regime,          # Bull/bear/volatile
     macro_indicators)       # Interest rates, GDP, unemployment
```

**Action Space:**
```
a = (w₁, w₂, ..., wₙ) where Σwᵢ = 1 and wᵢ ≥ 0
    (portfolio weights for n assets)
```

**Reward Function:**
```
R = Portfolio_Return - λ·Risk_Penalty - Transaction_Costs

Common objectives:
- Sharpe Ratio: (Return - Risk_Free_Rate) / Volatility
- Maximum Drawdown: Largest peak-to-trough decline
- Risk-adjusted return: Return / VaR
```

### Challenges

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Non-stationarity** | Market regimes change | Regime detection, meta-learning |
| **Transaction costs** | Frequent trading expensive | Penalize turnover in reward |
| **Partial observability** | Hidden market factors | Use LSTMs, attention mechanisms |
| **High stakes** | Real money at risk | Paper trading, robust policies |
| **Sample efficiency** | Limited historical data | Transfer learning, simulation |

### Ethical Considerations

1. **Transparency:** Clients must understand how decisions are made
2. **Fairness:** Equal treatment across client portfolios
3. **Risk disclosure:** Clear communication of potential losses
4. **Regulatory compliance:** Follow SEC, FINRA regulations
5. **Market manipulation:** Avoid strategies that could destabilize markets

## 9.3 Case Study: Amazon Inventory Management

**Problem:** Optimize stock levels across thousands of warehouses for millions of products.

**RL Approach:**
- State: Current inventory, demand forecasts, lead times, storage costs
- Action: Order quantities for each product-warehouse pair
- Reward: Revenue - Holding costs - Stockout costs - Ordering costs

**Results:**
- Reduced holding costs by optimizing safety stock
- Decreased stockouts through better demand anticipation
- Improved delivery times by strategic pre-positioning

**Reference:** Various industry papers and blog posts; academic: Gijsbrechts et al. (2022)

---

# Domain 10: Advanced Topics

## 10.1 Eligibility Traces and TD(λ)

### Unifying MC and TD

Eligibility traces provide a mechanism to bridge between TD(0) and Monte Carlo:

```
TD(0):           Uses immediate reward + next state estimate
                 Updates only predecessor state
                 
Monte Carlo:     Uses complete return
                 Updates all states in trajectory equally
                 
TD(λ):           Interpolates between these extremes
                 λ controls the blend
```

### The λ-Return

```
G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} G_t^{(n)}

where G_t^{(n)} is the n-step return:
G_t^{(n)} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γⁿV(S_{t+n})
```

**Special Cases:**
- λ = 0: G_t^λ = G_t^{(1)} = R_{t+1} + γV(S_{t+1}) [TD(0)]
- λ = 1: G_t^λ = G_t [Monte Carlo]

### Eligibility Traces (Backward View)

```
e_t(s) = {
    γλ·e_{t-1}(s) + 1   if s = S_t
    γλ·e_{t-1}(s)       otherwise
}

Update: δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
        V(s) ← V(s) + α·δ_t·e_t(s)  for all s
```

**Reference:** Sutton & Barto (2018), Chapter 12

---

# Quick Reference: Formula Sheet

## Value Functions
```
V^π(s) = E_π[Σ γᵗR_{t+1} | S₀=s]
Q^π(s,a) = E_π[Σ γᵗR_{t+1} | S₀=s, A₀=a]
```

## Bellman Equations
```
V*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]
Q*(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ max_{a'} Q*(s',a')]
```

## Update Rules
```
TD(0):        V(s) ← V(s) + α[r + γV(s') - V(s)]
Q-learning:   Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
SARSA:        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
Expected SARSA: Q(s,a) ← Q(s,a) + α[r + γ Σπ(a'|s')Q(s',a') - Q(s,a)]
```

## Exploration
```
ε-greedy: π(a|s) = 1-ε+ε/|A| if greedy, else ε/|A|
Boltzmann: π(a|s) = exp(Q(s,a)/τ) / Σexp(Q(s,a')/τ)
UCB: A = argmax[Q(s,a) + c√(ln(t)/N(s,a))]
```

## DQN
```
Target: y = r + γ max_a' Q(s',a';θ⁻)
Loss: L = E[(y - Q(s,a;θ))²]
Double DQN: y = r + γ Q(s', argmax_a' Q(s',a';θ); θ⁻)
Dueling: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```

---

# References

## Primary Textbook
- **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
  - Available free online: http://incompleteideas.net/book/RLbook2020.pdf

## Foundational Papers

### Q-Learning & TD Methods
- **Watkins, C. J. C. H. (1989).** Learning from Delayed Rewards. PhD Thesis, Cambridge University.
- **Rummery, G. A., & Niranjan, M. (1994).** On-Line Q-Learning Using Connectionist Systems. Technical Report CUED/F-INFENG/TR 166.

### Deep RL
- **Mnih, V., et al. (2015).** Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
- **van Hasselt, H., Guez, A., & Silver, D. (2016).** Deep Reinforcement Learning with Double Q-learning. *AAAI*.
- **Wang, Z., et al. (2016).** Dueling Network Architectures for Deep Reinforcement Learning. *ICML*.

### Exploration
- **Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).** Finite-time Analysis of the Multiarmed Bandit Problem. *Machine Learning*, 47(2-3), 235-256.

### Theoretical Foundations
- **Bertsekas, D. P., & Tsitsiklis, J. N. (1996).** *Neuro-Dynamic Programming*. Athena Scientific.

## Additional Resources

### Online Courses
- David Silver's RL Course (UCL/DeepMind): https://www.davidsilver.uk/teaching/
- Berkeley CS285 Deep RL: https://rail.eecs.berkeley.edu/deeprlcourse/
- Stanford CS234: http://web.stanford.edu/class/cs234/

### Tutorials & Surveys
- **Li, Y. (2017).** Deep Reinforcement Learning: An Overview. *arXiv:1701.07274*.
- **François-Lavet, V., et al. (2018).** An Introduction to Deep Reinforcement Learning. *Foundations and Trends in Machine Learning*.

---

# Exam Preparation Checklist

## Conceptual Understanding
- [ ] Can explain the RL agent-environment loop
- [ ] Understand the difference between V(s) and Q(s,a)
- [ ] Can derive Bellman equations from definitions
- [ ] Understand bias-variance tradeoff in TD vs MC
- [ ] Know when to use on-policy vs off-policy methods

## Algorithm Knowledge
- [ ] Can write pseudocode for Q-learning and SARSA
- [ ] Understand DQN architecture and innovations
- [ ] Know Double Q-learning and why it helps
- [ ] Can explain experience replay and target networks

## Problem Solving
- [ ] Can formulate any problem as an MDP
- [ ] Choose appropriate algorithms for different scenarios
- [ ] Design reward functions for business applications
- [ ] Analyze tradeoffs between different approaches

## Mathematical Skills
- [ ] Write and manipulate Bellman equations
- [ ] Derive update rules from first principles
- [ ] Calculate expected returns and values
- [ ] Understand convergence conditions

---

**Good luck with your exam! Remember: Understanding the "why" is as important as knowing the "what."** 🎓