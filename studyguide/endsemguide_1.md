# MLA202 Reinforcement Learning - End Semester Study Guide

## 📚 Course Information

| Detail | Information |
|--------|-------------|
| **Course Code** | MLA202 |
| **Course Title** | Reinforcement Learning |
| **Program** | Common Foundation (2BBI) |
| **Exam Duration** | 2 Hours |
| **Total Marks** | 50 |

---

## 📋 Exam Structure Overview

| Section | Marks | Requirements |
|---------|-------|--------------|
| **Part A** | 20 | Answer ALL questions |
| **Part B** | 30 | Q5 compulsory + ONE from Q6/Q7 |

---

## 📺 Video Lectures & Resources

### Primary Video Courses

#### 1. DeepMind x UCL - David Silver's Reinforcement Learning Course (Highly Recommended)
The gold standard for learning RL fundamentals, taught by the co-creator of AlphaGo.

| Lecture | Topic | Link |
|---------|-------|------|
| Lecture 1 | Introduction to RL | [YouTube](https://www.youtube.com/watch?v=2pWv7GOvuf0) |
| Lecture 2 | Markov Decision Processes | [YouTube](https://www.youtube.com/watch?v=lfHX2hHRMVQ) |
| Lecture 3 | Planning by Dynamic Programming | [YouTube](https://www.youtube.com/watch?v=Nd1-UUMVfz4) |
| Lecture 4 | Model-Free Prediction (MC & TD) | [YouTube](https://www.youtube.com/watch?v=PnHCvfgC_ZA) |
| Lecture 5 | Model-Free Control (SARSA, Q-learning) | [YouTube](https://www.youtube.com/watch?v=0g4j2k_Ggc4) |
| Lecture 6 | Value Function Approximation | [YouTube](https://www.youtube.com/watch?v=UoPei5o4fps) |
| Lecture 7 | Policy Gradient Methods | [YouTube](https://www.youtube.com/watch?v=KHZVXao4qXs) |

🔗 **Full Course Page:** [DeepMind Learning Resources](https://www.deepmind.com/learning-resources/introduction-to-reinforcement-learning-with-david-silver)

🔗 **Course Slides & Materials:** [UCL Course Page](https://www.davidsilver.uk/teaching/)

#### 2. Stanford CS234: Reinforcement Learning (Emma Brunskill)
More recent course with excellent coverage of modern RL techniques.

🔗 **YouTube Playlist:** [Stanford CS234 Winter 2019](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)

🔗 **Course Website:** [Stanford CS234](http://web.stanford.edu/class/cs234/)

#### 3. Hugging Face Deep RL Course (Free, Hands-On)
Excellent practical course with coding exercises.

🔗 **Course Link:** [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction)

### Topic-Specific Video Resources

| Topic | Resource | Link |
|-------|----------|------|
| **DQN Explained** | DeepLizard Series | [YouTube](https://deeplizard.com/learn/video/wrBUkpiRvCA) |
| **DQN Training Process** | DeepLizard | [YouTube](https://deeplizard.com/learn/video/0bt0SjbS3xc) |
| **Q-Learning Tutorial** | PyTorch Official | [Tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) |
| **SARSA vs Q-Learning** | Built In Guide | [Article](https://builtin.com/machine-learning/sarsa) |
| **Experience Replay** | TensorFlow Agents | [Tutorial](https://www.tensorflow.org/agents/tutorials/0_intro_rl) |

---

## 📖 Textbooks & Reading Materials

### Primary Textbook
**Reinforcement Learning: An Introduction (2nd Edition)**
*Richard S. Sutton & Andrew G. Barto*

🔗 **Free PDF (Official):** [incompleteideas.net](http://incompleteideas.net/book/the-book-2nd.html)

🔗 **Alternative PDF:** [Stanford Mirror](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

| Chapter | Topic | Exam Relevance |
|---------|-------|----------------|
| Chapter 3 | Finite MDPs | ⭐⭐⭐⭐⭐ |
| Chapter 4 | Dynamic Programming | ⭐⭐⭐⭐ |
| Chapter 5 | Monte Carlo Methods | ⭐⭐⭐⭐⭐ |
| Chapter 6 | Temporal-Difference Learning | ⭐⭐⭐⭐⭐ |
| Chapter 9-10 | Function Approximation & DQN | ⭐⭐⭐⭐ |
| Chapter 13 | Policy Gradient Methods | ⭐⭐⭐ |

### Additional Resources

| Resource | Description | Link |
|----------|-------------|------|
| **OpenAI Spinning Up** | Practical Deep RL guide | [spinningup.openai.com](https://spinningup.openai.com/en/latest/) |
| **Hugging Face Deep RL** | Interactive course | [huggingface.co](https://huggingface.co/learn/deep-rl-course/) |
| **Lilian Weng's Blog** | Excellent RL overview | [lilianweng.github.io](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) |
| **Gymnasium Docs** | Environment library | [gymnasium.farama.org](https://gymnasium.farama.org/) |

---

## Part A: Core Concepts (20 Marks)

### 1. Q-Learning Fundamentals

#### 1.1 The Q-Learning Update Rule

The Q-learning algorithm learns optimal action-value functions using the following update equation:

```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

**Key Components:**
- `Q(s,a)`: Current estimate of action-value for state `s` and action `a`
- `α` (alpha): Learning rate (typically 0.1 to 0.5)
- `r`: Immediate reward received
- `γ` (gamma): Discount factor (typically 0.9 to 0.99)
- `max_a' Q(s',a')`: Maximum Q-value achievable from the next state

**Critical Identifier:** The **max operator** over next actions distinguishes Q-learning from other algorithms like SARSA.

📺 **Watch:** David Silver Lecture 5 - Model-Free Control **(refer above)**

---

### 2. Exploration Strategies

#### 2.1 ε-Greedy Exploration

The most common exploration strategy that balances exploration and exploitation:

```
With probability ε:    Select random action (exploration)
With probability 1-ε:  Select argmax_a Q(s,a) (exploitation)
```

**Typical Values:** ε starts at 1.0 and decays to 0.01-0.1

#### 2.2 Other Exploration Methods

| Method | Description | Key Feature |
|--------|-------------|-------------|
| **Boltzmann/Softmax** | P(a) ∝ exp(Q(s,a)/τ) | Temperature parameter τ |
| **UCB** | a = argmax[Q(s,a) + c√(log(t)/N(s,a))] | Confidence bound |
| **Thompson Sampling** | Sample from posterior distributions | Bayesian approach |
| **Optimistic Initialization** | Start Q-values high | Natural exploration |

📺 **Watch:** David Silver Lecture 9 - Exploration and Exploitation **[LINK](https://www.youtube.com/watch?v=sGuiWX07sKw&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=9)**

---

### 3. TD Learning vs Monte Carlo Methods

#### 3.1 Primary Advantage of TD Learning

**TD can learn from incomplete episodes** through bootstrapping - updating estimates based on other estimates without waiting for episode termination.

| Property | TD Learning | Monte Carlo |
|----------|-------------|-------------|
| **Learning** | Online, every step | After episode ends |
| **Bias** | Biased (bootstrapping) | Unbiased |
| **Variance** | Lower | Higher |
| **Episode Requirement** | Not required | Required |

📺 **Watch:** David Silver Lecture 4 - Model-Free Prediction. **(refer above)**

📖 **Read:** Sutton & Barto Chapter 6

---

### 4. Deep Q-Networks (DQN) Components

#### 4.1 Experience Replay Buffer

**Purpose:** Break correlation between consecutive samples

**Why It's Necessary:**
- Sequential experiences are highly correlated
- Violates i.i.d. assumption for neural network training
- Enables reuse of experiences (improved sample efficiency)

📺 **Watch:** [DeepLizard - DQN Training](https://deeplizard.com/learn/video/0bt0SjbS3xc)

📖 **Read:** [Hugging Face - Deep Q Algorithm](https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-algorithm)

---

### 5. RL Applications in Finance

**Policy Gradient methods** are preferred for portfolio management because:
- Handle continuous action spaces naturally (portfolio weights)
- Can optimize directly for expected returns
- Flexible reward shaping for risk management

📖 **Read:** Sutton & Barto Chapter 13 - Policy Gradient Methods

---

## 📖 Key Concepts Deep Dive

### 6. On-Policy vs Off-Policy Learning

#### 6.1 On-Policy Methods

**Definition:** Methods where the behavior policy (used for action selection) and target policy (being evaluated/improved) are **identical**.

**Examples:** SARSA, Monte Carlo ES, REINFORCE

**When to Use:**
- Safety-critical applications (robots, autonomous vehicles)
- Online learning where policy must be reasonable during training

#### 6.2 Off-Policy Methods

**Definition:** Methods where the target policy differs from the behavior policy used to generate data.

**Examples:** Q-learning, DQN, Expected SARSA (configurable)

**When to Use:**
- Learning from demonstrations or historical data
- Finding optimal deterministic policies
- When sample efficiency through experience replay is needed

📺 **Watch:** [Stack Overflow - SARSA vs Q-Learning Explained](https://stackoverflow.com/questions/6848828/what-is-the-difference-between-q-learning-and-sarsa)

📖 **Read:** [Towards Data Science - Q-Learning and SARSA](https://towardsdatascience.com/q-learning-and-sasar-with-python-3775f86bd178/)

---

### 7. Bellman Equations

#### 7.1 Conceptual Foundation

Bellman equations express a fundamental recursive relationship:

> **The value of a state equals the immediate reward plus the discounted value of successor states.**

#### 7.2 State-Value Bellman Optimality Equation

```
V*(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV*(s')]
```

#### 7.3 Action-Value Bellman Optimality Equation

```
Q*(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γ max_a' Q*(s',a')]
```

📺 **Watch:** David Silver Lecture 2 - Markov Decision Processes

📖 **Read:** Sutton & Barto Chapter 3.5-3.6

---

### 8. Maximization Bias and Double Q-Learning

#### 8.1 The Problem

**Maximization bias** occurs when the same samples are used to both **select** and **evaluate** an action.

**Mathematical Basis (Jensen's Inequality):**
```
E[max_a Q(s,a)] ≥ max_a E[Q(s,a)]
```

#### 8.2 Double Q-Learning Solution

Maintain **two Q-functions** (Q_A and Q_B):

```
Action Selection:  a* = argmax_a Q_A(s',a)   [Use Q_A]
Value Evaluation:  Q_B(s',a*)                [Use Q_B]
```

📖 **Read:** [Hugging Face - Double DQN](https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-algorithm#double-dqn)

---

## Part B: Applied Topics (30 Marks)

### 9. MDP Formulation (Question 5 - Compulsory)

#### 9.1 Robot Navigation Problem

**Scenario:** 4×4 grid world, robot must reach goal while avoiding obstacles.

#### 9.2 MDP Components

| Component | Definition | Example |
|-----------|------------|---------|
| **States** | S = {(i,j) \| i,j ∈ {0,1,2,3}} | 16 grid positions |
| **Actions** | A = {up, down, left, right} | 4 directional moves |
| **Transitions** | P(s'\|s,a) | Deterministic or stochastic |
| **Rewards** | R(s,a,s') | Goal: +10, Obstacle: -5, Step: -1 |
| **Discount** | γ ∈ [0.8, 0.99] | Typically 0.9 or 0.95 |

#### 9.3 Q-Learning Implementation

```python
# Algorithm: Q-Learning for Grid World

Initialize Q(s,a) = 0 for all s,a
Set α = 0.1, ε = 0.1, γ = 0.9

For episode = 1 to max_episodes:
    Initialize s to start position
    
    While s is not terminal:
        Choose a using ε-greedy from Q(s,·)
        Execute a, observe r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        s ← s'
```

📺 **Watch:** [Coder One - RL Tutorial with OpenAI Gym](https://www.gocoder.one/blog/rl-tutorial-with-openai-gym/)

📖 **Practice:** [Gymnasium FrozenLake Environment](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)

---

### 10. Deep Reinforcement Learning (Question 6a)

#### 10.1 DQN Architecture

```
┌─────────────────┐
│  State Input    │  (e.g., 84×84×4 for Atari)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Convolutional   │  (for image inputs)
│    Layers       │  or Fully Connected
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hidden Layers  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Layer   │  Q-values for each action
│   (size = |A|)  │
└─────────────────┘
```

#### 10.2 DQN Variants Comparison

| Variant | Problem Addressed | Solution |
|---------|-------------------|----------|
| **Double DQN** | Overestimation bias | Separate selection and evaluation |
| **Dueling DQN** | Value vs Advantage | Separate V(s) and A(s,a) streams |

📺 **Watch:** [DeepLizard - Deep Q-Learning](https://deeplizard.com/learn/video/wrBUkpiRvCA)

📖 **Read:** [GeeksforGeeks - Deep Q-Learning](https://www.geeksforgeeks.org/deep-learning/deep-q-learning/)

📖 **Tutorial:** [PyTorch DQN Tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

---

### 11. Monte Carlo Methods (Question 6b)

#### 11.1 First-Visit vs Every-Visit MC

| Method | Description | Convergence |
|--------|-------------|-------------|
| **First-Visit** | Use return from first visit to state per episode | Unbiased |
| **Every-Visit** | Use returns from all visits to state | May be biased |

#### 11.2 MC vs TD Comparison

| Property | Monte Carlo | TD Learning |
|----------|-------------|-------------|
| **Bias** | Unbiased | Biased (bootstrapping) |
| **Variance** | High | Low |
| **Speed** | Slower | Faster in practice |

📺 **Watch:** David Silver Lecture 4 - Model-Free Prediction **(refer above)**

📖 **Read:** Sutton & Barto Chapter 5

---

### 12. Business Applications of RL (Question 7a)

#### 12.1 Dynamic Pricing System Design

**State Space:**
- Current demand level/volume
- Competitor prices
- Time features (hour, day, season)
- Inventory/stock levels

**Action Space:**
- Discrete: Price levels {$10, $15, $20, ...}
- Continuous: Price ∈ [min_price, max_price]

**Reward Function:**
```
R = Revenue - Costs + λ × Customer_Satisfaction
```

#### 12.2 Real-World Case Studies

| Company | Application | RL Approach |
|---------|-------------|-------------|
| **Uber/Lyft** | Dynamic Pricing | Multi-agent optimization by zone |
| **Amazon** | Inventory Management | Ordering policies from demand patterns |
| **DeepMind** | Data Center Cooling | Energy optimization |

---

### 13. TD Learning Algorithms (Question 7b)

#### 13.1 TD(0) Derivation

**Step 1: Bellman Equation**
```
V^π(s) = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
```

**Step 2: Update Rule**
```
V(s) ← V(s) + α[r + γV(s') - V(s)]
              └────────────────┘
                   TD Error (δ)
```

#### 13.2 SARSA vs Q-Learning

| Aspect | SARSA | Q-Learning |
|--------|-------|------------|
| **Type** | On-policy | Off-policy |
| **Next Action** | Uses actual a' | Uses max over a' |
| **Learns** | Policy being followed | Optimal policy |
| **Safety** | Safer in risky environments | May be riskier |

📺 **Watch:** David Silver Lecture 5 - Model-Free Control **(refer above)**

📖 **Read:** [Introduction to RL: TD, SARSA, Q-Learning](https://towardsdatascience.com/introduction-to-reinforcement-learning-temporal-difference-sarsa-q-learning-e8f22669c366/)

---

## 🎯 Quick Reference Formulas

```
┌────────────────────────────────────────────────────────────────┐
│                    ESSENTIAL EQUATIONS                         │
├────────────────────────────────────────────────────────────────┤
│ Q-Learning:                                                    │
│   Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]         │
│                                                                │
│ SARSA:                                                         │
│   Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]                 │
│                                                                │
│ Expected SARSA:                                                │
│   Q(s,a) ← Q(s,a) + α[r + γ Σ_a' π(a'|s')Q(s',a') - Q(s,a)]  │
│                                                                │
│ TD(0):                                                         │
│   V(s) ← V(s) + α[r + γV(s') - V(s)]                          │
│                                                                │
│ Bellman Optimality (V*):                                       │
│   V*(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV*(s')]           │
│                                                                │
│ Bellman Optimality (Q*):                                       │
│   Q*(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γ max_a' Q*(s',a')]    │
│                                                                │
│ DQN Loss:                                                      │
│   L = E[(r + γ max_a' Q_target(s',a') - Q(s,a))²]             │
│                                                                │
│ Double DQN Target:                                             │
│   y = r + γ Q_target(s', argmax_a Q(s',a))                    │
│                                                                │
│ Dueling DQN:                                                   │
│   Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))                    │
└────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Practice Questions

> **Note:** These questions are designed to test similar concepts to the exam but are **not identical** to actual exam questions. Use them for self-assessment and practice.

### Section A: Multiple Choice Practice

**Q1.** Which update rule correctly represents the SARSA algorithm?

a) Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]  
b) Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]  
c) Q(s,a) ← r + γV(s')  
d) V(s) ← V(s) + α[r + γV(s') - V(s)]

<details>
<summary>Click for Answer</summary>

**Answer: (b)**

Explanation: SARSA uses the actual next action a' chosen by the policy, not the maximum. Option (a) is Q-learning, (c) is incomplete, and (d) is TD(0) for state values.
</details>

---

**Q2.** What problem does the target network in DQN primarily solve?

a) Reduces memory usage  
b) Speeds up training  
c) Stabilizes training by keeping targets fixed  
d) Increases exploration  

<details>
<summary>Click for Answer</summary>

**Answer: (c)**

Explanation: The target network provides stable Q-value targets during training. Without it, the targets would constantly shift as the network updates, leading to unstable training (chasing a moving target).
</details>

---

**Q3.** In the context of exploration strategies, what does the temperature parameter τ control in Boltzmann exploration?

a) The learning rate  
b) The randomness of action selection  
c) The discount factor  
d) The size of the replay buffer  

<details>
<summary>Click for Answer</summary>

**Answer: (b)**

Explanation: Higher temperature τ leads to more uniform (random) action selection, while lower temperature makes the policy more deterministic (greedy). As τ→0, Boltzmann becomes fully greedy; as τ→∞, it becomes uniform random.
</details>

---

**Q4.** Which of the following is NOT a characteristic of Monte Carlo methods?

a) Requires complete episodes  
b) Has high variance  
c) Uses bootstrapping  
d) Provides unbiased estimates  

<details>
<summary>Click for Answer</summary>

**Answer: (c)**

Explanation: Monte Carlo methods do NOT use bootstrapping - they wait for actual returns from complete episodes. TD methods use bootstrapping (updating estimates based on other estimates).
</details>

---

**Q5.** For a continuous action space like portfolio allocation, which method is most suitable?

a) Tabular Q-learning  
b) Value Iteration  
c) Policy Gradient methods  
d) Monte Carlo Tree Search  

<details>
<summary>Click for Answer</summary>

**Answer: (c)**

Explanation: Policy Gradient methods can naturally handle continuous action spaces by directly parameterizing the policy. Tabular methods and Value Iteration require discrete action spaces.
</details>

---

### Section B: Short Answer Practice

**Q6.** Explain why experience replay is important for training Deep Q-Networks. What two main problems does it address?

<details>
<summary>Click for Answer</summary>

**Answer:**

Experience replay addresses two main problems:

1. **Correlation between consecutive samples:** Sequential experiences from an agent interacting with an environment are highly correlated (e.g., consecutive frames in a game). Neural networks assume i.i.d. (independent and identically distributed) training data. Random sampling from the replay buffer breaks this temporal correlation.

2. **Sample efficiency:** Without replay, each experience is used only once and then discarded. The replay buffer allows reusing experiences multiple times, making better use of collected data and improving sample efficiency.

Additional benefits include smoother learning (by averaging over many past experiences) and the ability to prioritize important experiences (in Prioritized Experience Replay).
</details>

---

**Q7.** Compare and contrast on-policy and off-policy learning. Give one example algorithm for each and describe a scenario where each would be preferred.

<details>
<summary>Click for Answer</summary>

**Answer:**

**On-Policy Learning:**
- Definition: Learns the value of the policy currently being used for action selection
- Behavior policy = Target policy
- Example: SARSA
- Preferred when: Safety is important during learning (e.g., robotics), or when we want to learn a specific stochastic policy

**Off-Policy Learning:**
- Definition: Learns a target policy different from the behavior policy used to collect data
- Behavior policy ≠ Target policy  
- Example: Q-learning
- Preferred when: Learning from historical data/demonstrations, maximizing sample efficiency with experience replay, or when seeking the optimal deterministic policy

**Key Trade-off:** On-policy is safer but less sample efficient; Off-policy can reuse data but may have stability issues.
</details>

---

**Q8.** Derive the TD(0) update rule starting from the Bellman equation. Show each step clearly.

<details>
<summary>Click for Answer</summary>

**Answer:**

**Step 1: Start with Bellman Expectation Equation**
```
V^π(s) = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
```

**Step 2: Replace expectation with a single sample**

Instead of computing the full expectation, we approximate using a single sample (s, r, s'):
```
V(s) ≈ r + γV(s')
```

**Step 3: Form the TD target and TD error**

TD Target: `r + γV(s')`
TD Error: `δ = r + γV(s') - V(s)`

**Step 4: Move current estimate toward sample**

Use the TD error to update, scaled by learning rate α:
```
V(s) ← V(s) + α[r + γV(s') - V(s)]
        ↑                ↑
   old estimate    TD error (sample - old estimate)
```

This is the TD(0) update rule. The "0" indicates we're using a 1-step return (looking only one step ahead).
</details>

---

### Section C: Problem-Solving Practice

**Q9.** Consider a 3×3 grid world where an agent starts at position (0,0) and must reach the goal at (2,2). Obstacles are at (1,1). The agent can move up, down, left, right.

a) Define all five components of the MDP  
b) Write the Q-learning update that would occur if the agent is at (0,1), takes action "right", receives reward -1, and ends up at (1,1) hitting an obstacle  

<details>
<summary>Click for Answer</summary>

**Answer:**

**a) MDP Components:**

1. **States:** S = {(i,j) | i,j ∈ {0,1,2}} = 9 positions
   - Terminal states: (2,2) goal, (1,1) obstacle

2. **Actions:** A = {up, down, left, right}

3. **Transition Function:** P(s'|s,a)
   - Deterministic: P(s'|s,a) = 1 if s' is the intended destination, 0 otherwise
   - At boundaries: stay in place if action would move outside grid

4. **Reward Function:** 
   - R = +10 for reaching goal (2,2)
   - R = -10 for hitting obstacle (1,1)
   - R = -1 for each step (encourages efficiency)

5. **Discount Factor:** γ = 0.9

**b) Q-learning Update:**

Given:
- Current state s = (0,1)
- Action a = "right"
- Reward r = -10 (hit obstacle at (1,1))
- Next state s' = (1,1) [terminal - obstacle]

Assuming α = 0.1, γ = 0.9, and current Q((0,1), right) = 0:

```
Q((0,1), right) ← Q((0,1), right) + α[r + γ max_a' Q((1,1), a') - Q((0,1), right)]
Q((0,1), right) ← 0 + 0.1[-10 + 0.9 × 0 - 0]
Q((0,1), right) ← 0 + 0.1[-10]
Q((0,1), right) ← -1.0
```

Note: Since (1,1) is terminal (obstacle), the future Q-values are 0.
</details>

---

**Q10.** You are designing a reinforcement learning system for an automated trading agent. The agent must decide how much of each asset to hold in a portfolio.

a) Why might policy gradient methods be more suitable than Q-learning for this problem?  
b) Design an appropriate state space (list at least 5 relevant features)  
c) What ethical considerations should be addressed?  

<details>
<summary>Click for Answer</summary>

**Answer:**

**a) Why Policy Gradient over Q-learning:**

1. **Continuous action space:** Portfolio weights are continuous (e.g., 40% stocks, 30% bonds, 30% cash). Q-learning requires discretizing this space, which becomes impractical with many assets.

2. **Stochastic policies:** Policy gradients naturally output probability distributions over actions, useful for managing risk through diversification.

3. **Differentiable objectives:** Can directly optimize for financial metrics like Sharpe ratio.

4. **No need for argmax:** Q-learning requires finding max over actions, which is computationally expensive in continuous spaces.

**b) State Space Features (5+):**

1. Current portfolio holdings/weights
2. Recent price movements (returns over different horizons)
3. Volatility indicators (e.g., VIX, rolling standard deviation)
4. Trading volume
5. Technical indicators (moving averages, RSI)
6. Interest rates / macroeconomic indicators
7. Time features (day of week, month, quarter-end)
8. Current cash position
9. Transaction costs/fees
10. Market sentiment indicators

**c) Ethical Considerations:**

1. **Transparency:** Can decisions be explained to clients? Black-box AI trading raises accountability issues.

2. **Fairness:** Does the system treat all clients equally? Could it favor certain groups?

3. **Risk disclosure:** Clear communication about potential losses and that past performance doesn't guarantee future results.

4. **Market impact:** High-frequency trading by many AI agents could destabilize markets.

5. **Regulatory compliance:** Must follow financial regulations (SEC, etc.)

6. **Bias:** Training data may contain historical biases that could perpetuate unfair outcomes.

7. **Human oversight:** Should there be human approval for large trades?
</details>

---

**Q11.** Compare DQN, Double DQN, and Dueling DQN architectures. For each, explain the problem it addresses and how it solves it.

<details>
<summary>Click for Answer</summary>

**Answer:**

**DQN (Deep Q-Network):**
- **Problem:** Q-learning fails with high-dimensional state spaces (can't use tables)
- **Solution:** Uses neural network to approximate Q(s,a)
- **Key innovations:** 
  - Experience replay (breaks correlation)
  - Target network (stabilizes training)
- **Update:** L = E[(r + γ max_a' Q_target(s',a') - Q(s,a))²]

**Double DQN:**
- **Problem:** DQN overestimates Q-values because max operator amplifies noise
- **Mathematical cause:** E[max Q] ≥ max E[Q]
- **Solution:** Decouple action selection from action evaluation
  - Use main network to SELECT: a* = argmax_a Q(s',a)
  - Use target network to EVALUATE: Q_target(s', a*)
- **Update:** y = r + γ Q_target(s', argmax_a Q(s',a))
- **Benefit:** More accurate value estimates, faster convergence

**Dueling DQN:**
- **Problem:** For many states, the value is similar regardless of action taken
- **Solution:** Separate the Q-function into two streams:
  - Value stream V(s): How good is this state overall?
  - Advantage stream A(s,a): How much better is this action than average?
- **Combination:** Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
- **Benefit:** Better learning when actions don't affect outcomes much (e.g., can't avoid crash regardless of action)

**Combined:** All three improvements can be used together for even better performance (Rainbow DQN includes these and more).
</details>

---

### Section D: Algorithm Implementation Practice

**Q12.** Write pseudocode for the SARSA algorithm with ε-greedy exploration. Include initialization, episode loop, and update step.

<details>
<summary>Click for Answer</summary>

```
Algorithm: SARSA with ε-greedy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Initialize:
    Q(s,a) ← 0 for all s ∈ S, a ∈ A
    Set parameters: α (learning rate), γ (discount), ε (exploration)

For each episode:
    Initialize state s
    Choose action a from s using ε-greedy policy:
        With probability ε: a ← random action
        With probability 1-ε: a ← argmax_a Q(s,a)
    
    While s is not terminal:
        Take action a, observe reward r and next state s'
        
        Choose next action a' from s' using ε-greedy policy:
            With probability ε: a' ← random action
            With probability 1-ε: a' ← argmax_a Q(s',a)
        
        # SARSA Update (uses actual next action a')
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        
        s ← s'
        a ← a'  # This is the key difference from Q-learning

Return Q
```

**Key differences from Q-learning:**
1. Action a' is chosen BEFORE the update (not after)
2. Update uses Q(s',a') not max_a' Q(s',a')
3. The same action a' is used for both update and next step
</details>

---

**Q13.** Given the following Q-table and ε = 0.1, what is the probability of selecting each action in state S2?

| State | Left | Right | Up | Down |
|-------|------|-------|-----|------|
| S1 | 2.5 | 3.0 | 1.0 | 0.5 |
| S2 | 4.0 | 2.0 | 4.0 | 1.0 |
| S3 | 1.5 | 2.5 | 3.0 | 2.0 |

<details>
<summary>Click for Answer</summary>

**Answer:**

In state S2, the Q-values are:
- Left: 4.0
- Right: 2.0  
- Up: 4.0
- Down: 1.0

The maximum Q-value is 4.0, achieved by both Left and Up (tie).

With ε-greedy (ε = 0.1):
- Random action probability: ε = 0.1, distributed equally among 4 actions
- Greedy action probability: 1 - ε = 0.9

For tied maximum values, the greedy choice is typically broken randomly among the tied actions.

**Probabilities:**

For **Left** (tied max):
```
P(Left) = ε/4 + (1-ε)/2 = 0.1/4 + 0.9/2 = 0.025 + 0.45 = 0.475
```

For **Up** (tied max):
```
P(Up) = ε/4 + (1-ε)/2 = 0.1/4 + 0.9/2 = 0.025 + 0.45 = 0.475
```

For **Right** (not max):
```
P(Right) = ε/4 = 0.1/4 = 0.025
```

For **Down** (not max):
```
P(Down) = ε/4 = 0.1/4 = 0.025
```

**Verification:** 0.475 + 0.475 + 0.025 + 0.025 = 1.0 ✓
</details>

---

**Q14.** An agent is learning to play a cliff-walking game where falling off the cliff gives a large negative reward. Explain why SARSA might learn a safer policy than Q-learning in this environment.

<details>
<summary>Click for Answer</summary>

**Answer:**

In the cliff-walking environment, there's typically an optimal but risky path right along the cliff edge, and a safer but longer path further away.

**Why SARSA learns a safer policy:**

1. **SARSA is on-policy:** It evaluates the policy it's actually following (ε-greedy). This means it accounts for the fact that the agent will occasionally take random exploratory actions.

2. **Risk awareness:** When SARSA evaluates states near the cliff, it considers what happens when the ε-greedy policy randomly chooses an action. Near the cliff, a random action has a significant probability of falling off.

3. **Value updates reflect actual behavior:** The Q-values near the cliff will be lower because SARSA "sees" that following the ε-greedy policy from these states leads to falling off some percentage of the time.

**Why Q-learning learns a riskier policy:**

1. **Q-learning is off-policy:** It learns the optimal policy assuming it will always act greedily in the future (via the max operator).

2. **Ignores exploration risk:** Q-learning doesn't account for the fact that the agent will sometimes explore. It assumes the agent will always take the best action.

3. **Values assume greedy behavior:** The path along the cliff has high Q-values because IF you always act optimally, you won't fall. But in practice, ε-greedy exploration means you sometimes will fall.

**Practical implication:** If training must be done in a real environment (not simulation) where falling off the cliff has real consequences (e.g., damaging a robot), SARSA is preferred because it learns to avoid risky states during training.
</details>

---

**Q15.** Explain the concept of bootstrapping in reinforcement learning. How does TD learning use bootstrapping, and what are the trade-offs compared to Monte Carlo methods?

<details>
<summary>Click for Answer</summary>

**Answer:**

**Bootstrapping Definition:**
Bootstrapping means updating estimates based partly on other estimates, rather than waiting for the final outcome. It's like "pulling yourself up by your bootstraps" - using your current knowledge to improve your current knowledge.

**How TD Learning Uses Bootstrapping:**

TD(0) update rule:
```
V(s) ← V(s) + α[r + γV(s') - V(s)]
```

The term `V(s')` is itself an estimate - we're using our current estimate of the next state's value to update the current state's value. We don't wait to see the actual return from state s'; we bootstrap from our current estimate.

**Comparison with Monte Carlo (No Bootstrapping):**

MC update rule:
```
V(s) ← V(s) + α[G_t - V(s)]
```

MC waits for the actual return `G_t` (sum of all future rewards in the episode), which requires completing the episode.

**Trade-offs:**

| Aspect | TD (Bootstrapping) | MC (No Bootstrapping) |
|--------|-------------------|----------------------|
| **Bias** | Biased (estimates depend on other estimates which may be wrong) | Unbiased (uses actual returns) |
| **Variance** | Lower (updates based on single transition) | Higher (return depends on entire trajectory) |
| **Data efficiency** | More efficient (learns from each step) | Less efficient (needs complete episodes) |
| **Online learning** | Can learn online, during episodes | Must wait for episode end |
| **Continuing tasks** | Works for non-episodic tasks | Only for episodic tasks |
| **Convergence** | Converges faster in practice | May need more samples |
| **Markov property** | Exploits Markov structure | Doesn't assume Markov |

**Key insight:** Bootstrapping trades bias for lower variance and faster learning, which is often a good trade-off in practice.
</details>

---

## 🔧 Hands-On Practice Environments

### Recommended Practice Environments

| Environment | Complexity | Best For | Link |
|-------------|------------|----------|------|
| **FrozenLake** | Easy | Q-learning, SARSA basics | [Gymnasium](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) |
| **CartPole** | Easy | DQN introduction | [Gymnasium](https://gymnasium.farama.org/environments/classic_control/cart_pole/) |
| **CliffWalking** | Easy | SARSA vs Q-learning comparison | [Gymnasium](https://gymnasium.farama.org/environments/toy_text/cliff_walking/) |
| **MountainCar** | Medium | Exploration challenges | [Gymnasium](https://gymnasium.farama.org/environments/classic_control/mountain_car/) |
| **LunarLander** | Medium | Deep RL practice | [Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/) |
| **Atari Games** | Hard | Full DQN implementation | [Gymnasium](https://gymnasium.farama.org/environments/atari/) |

### Getting Started Code

```python
import gymnasium as gym

# Create environment
env = gym.make('FrozenLake-v1', render_mode='human')

# Reset to get initial state
state, info = env.reset()

# Take random actions
for _ in range(100):
    action = env.action_space.sample()  # Random action
    next_state, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        state, info = env.reset()
    else:
        state = next_state

env.close()
```

🔗 **More tutorials:** [Gymnasium Official Tutorials](https://gymnasium.farama.org/tutorials/training_agents/)

---

### Common Exam Mistakes to Avoid

1. **Confusing SARSA with Q-learning** - Remember: SARSA uses actual next action, Q-learning uses max
2. **Forgetting the max operator** in Q-learning update
3. **Mixing up bias/variance** for MC vs TD methods
4. **Incomplete MDP specifications** - Always include all 5 components
5. **Forgetting ε-decay** in exploration strategies
6. **Confusing target network** with experience replay purposes

---

*Last Updated: November 2025*

*Good luck with your exam! 🎓*