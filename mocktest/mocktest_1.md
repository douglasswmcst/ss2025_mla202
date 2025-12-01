# MLA202 Reinforcement Learning - Mock Test
## With Answers and Explanations

---

## Exam Information
| Detail | Information |
|--------|-------------|
| Duration | 2 Hours |
| Maximum Marks | 50 |
| Structure | Part A (20 marks) + Part B (30 marks) |

### Instructions
- Part A: Answer ALL questions
- Part B: Question 5 is compulsory. Answer ONE from Question 6 or 7
- The first 15 minutes is for reading the questions

---

# Part A (20 Marks)
*Answer ALL questions*

---

## Question 1: Multiple Choice Questions (5×1=5 marks)

### Q1.1
In the SARSA algorithm, which equation represents the action-value function update?

(a) Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

(b) Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

(c) V(s) ← V(s) + α[r + γ V(s') - V(s)]

(d) Q(s,a) ← r + γ max_a' Q(s',a')

---

#### ✅ Answer: (b)

**Explanation:** SARSA is an **on-policy** algorithm that uses the actual next action a' taken by the policy. The name SARSA comes from the tuple (S, A, R, S', A') used in the update.

**Key Identifier:** SARSA uses `Q(s',a')` where a' is the **action actually chosen** by the policy, NOT the maximum.

**Why others are wrong:**
- **(a)** This is **Q-learning** - notice the `max` operator, which makes it off-policy
- **(c)** This is **TD(0) for state-value function** V(s), not action-value Q(s,a)
- **(d)** This is missing the learning rate α and doesn't have the incremental update form

**Exam Tip:** Look for the `max` operator - if it's there, it's Q-learning; if it uses the actual next action a', it's SARSA.

---

### Q1.2
Which component of Deep Q-Networks (DQN) helps stabilize training by keeping target values fixed?

(a) Experience replay buffer

(b) Target network

(c) Convolutional layers

(d) ε-greedy policy

---

#### ✅ Answer: (b)

**Explanation:** The **target network** is a separate copy of the Q-network with parameters θ⁻ that are periodically updated (typically every C = 10,000 steps). 

**How it stabilizes training:**
- In the loss L = (y - Q(s,a;θ))², the target y = r + γ max_a' Q(s',a';θ⁻)
- θ⁻ stays fixed while θ is updated, preventing the "moving target" problem
- Without this, both the prediction and target change simultaneously → oscillations/divergence

**Why others are wrong:**
- **(a)** Experience replay breaks **correlation**, not target instability
- **(c)** Convolutional layers are for **feature extraction** from images
- **(d)** ε-greedy is for **exploration**, not training stability

---

### Q1.3
In Monte Carlo methods, what is the key difference between First-Visit and Every-Visit?

(a) First-Visit uses bootstrapping, Every-Visit does not

(b) First-Visit counts only the first occurrence of a state per episode

(c) Every-Visit requires a model of the environment

(d) First-Visit has higher variance

---

#### ✅ Answer: (b)

**Explanation:**
- **First-Visit MC:** Only uses the return from the **FIRST** time state s is visited in an episode
- **Every-Visit MC:** Uses returns from **EVERY** time state s is visited in an episode

**Example:** If state s appears at timesteps 3, 7, and 12 in an episode:
- First-Visit: Only uses G₃ (return from timestep 3)
- Every-Visit: Uses G₃, G₇, and G₁₂ (all three returns)

**Why others are wrong:**
- **(a)** Neither MC method uses bootstrapping - that's TD learning
- **(c)** Neither requires a model - both are model-free
- **(d)** Every-Visit typically has **higher** variance (more samples, but correlated)

---

### Q1.4
What is the primary purpose of the discount factor γ in reinforcement learning?

(a) To increase exploration

(b) To balance immediate vs. future rewards

(c) To speed up learning

(d) To reduce memory usage

---

#### ✅ Answer: (b)

**Explanation:** The discount factor γ ∈ [0,1] determines how much the agent values future rewards compared to immediate ones.

**Mathematical Interpretation:**
- A reward r received k steps in the future is worth γᵏ × r today
- γ = 0: Only immediate reward matters (myopic agent)
- γ = 1: All future rewards equally important (can cause infinite returns)
- γ = 0.9: Reward 10 steps away worth 0.9¹⁰ ≈ 0.35 of immediate reward

**Why others are wrong:**
- **(a)** Exploration is controlled by ε or exploration strategies, not γ
- **(c)** γ doesn't directly affect learning speed
- **(d)** γ has no relation to memory usage

---

### Q1.5
Which exploration strategy samples actions based on a probability distribution proportional to their estimated values?

(a) ε-greedy

(b) Upper Confidence Bound

(c) Boltzmann (Softmax) exploration

(d) Random exploration

---

#### ✅ Answer: (c)

**Explanation:** Boltzmann (Softmax) exploration selects actions with probability proportional to their exponentiated Q-values:

```
π(a|s) = exp(Q(s,a)/τ) / Σ_a' exp(Q(s,a')/τ)
```

**Temperature parameter τ:**
- τ → ∞: Uniform random (all actions equally likely)
- τ → 0: Greedy (highest Q-value always chosen)
- Middle values: Actions with higher Q more likely, but not guaranteed

**Why others are wrong:**
- **(a)** ε-greedy uses **fixed probability ε**, not proportional to values
- **(b)** UCB adds an **uncertainty bonus**, doesn't use softmax distribution
- **(d)** Random exploration completely **ignores** Q-values

---

## Question 2 (5 marks)

**Question:** Explain the difference between model-based and model-free reinforcement learning. Provide one example algorithm for each approach and discuss when you would prefer to use each.

---

### ✅ Model Answer:

**Model-Based RL (1.5 marks):**

Model-based methods either learn or are given a model of the environment's dynamics: the transition function P(s'|s,a) and reward function R(s,a). They use this model to plan by:
- Generating simulated experience
- Computing optimal policies through dynamic programming
- Looking ahead to evaluate actions

**Example:** **Dyna-Q** - learns a model from real experience and uses it to generate simulated transitions for additional Q-value updates. Also: MCTS (Monte Carlo Tree Search), Model Predictive Control.

**Model-Free RL (1.5 marks):**

Model-free methods learn value functions or policies directly from experience without explicitly learning environment dynamics. They rely solely on sampled transitions (s, a, r, s').

**Example:** **Q-learning** - updates Q-values using actual observed rewards and transitions without any model. Also: SARSA, Policy Gradient methods, DQN.

**When to Use Each (2 marks):**

| Use Model-Based When: | Use Model-Free When: |
|-----------------------|---------------------|
| Environment dynamics are known or easy to learn | Environment is too complex to model accurately |
| Sample efficiency is critical (expensive real data) | Model errors could compound and hurt performance |
| Accurate simulation is possible | Sufficient real experience is available |
| Planning horizons are long | Computational simplicity is needed |
| Environment is deterministic/low-noise | Environment is highly stochastic |

---

**Marking Scheme:**
- Definition of model-based with key points (1-1.5 marks)
- Definition of model-free with key points (1-1.5 marks)
- Valid example for each (0.5 marks each)
- When to use each with valid reasoning (1-2 marks)

---

## Question 3 (5 marks)

**Question:** Describe the TD(0) algorithm for policy evaluation. Derive the update rule from the Bellman equation and explain why TD methods can learn from incomplete episodes while Monte Carlo cannot.

---

### ✅ Model Answer:

**Step 1: Starting from the Bellman Expectation Equation (1.5 marks)**

The Bellman equation for the state-value function under policy π:

```
V^π(s) = E_π[R_{t+1} + γV^π(S_{t+1}) | S_t = s]
```

This says: the value of state s equals the expected immediate reward plus the discounted value of the next state.

**Step 2: Sampling Approximation (1.5 marks)**

Instead of computing the full expectation (which requires knowing the environment model), we use a **sample**:
- Execute action according to policy π
- Observe actual reward r and next state s'
- Use sample estimate: `r + γV(s')`

This replaces the expectation with a single sample from the distribution.

**Step 3: The TD(0) Update Rule (1 mark)**

Move the current estimate toward the sample using gradient descent:

```
V(s) ← V(s) + α[r + γV(s') - V(s)]
              └────────────────────┘
                    TD Error (δ)
```

Where:
- α is the learning rate (step size)
- `r + γV(s')` is the TD target (sample-based estimate)
- `r + γV(s') - V(s)` is the TD error

**Why TD Can Learn from Incomplete Episodes (1 mark)**

TD methods use **bootstrapping** - they update estimates based on other estimates (V(s')). This means:

1. **Only one transition needed:** We can update after every (s, a, r, s') transition
2. **No episode termination required:** Don't need to wait until the end
3. **Works for continuing tasks:** Can handle non-episodic environments

**Monte Carlo**, in contrast, needs the complete return G = R₁ + γR₂ + γ²R₃ + ... from state s until episode termination. This is impossible if the episode hasn't ended.

---

**Marking Scheme:**
- Starting Bellman equation correctly stated (1-1.5 marks)
- Sampling approximation explained (1-1.5 marks)
- Correct TD(0) update rule with terms identified (1 mark)
- Explanation of bootstrapping and why MC needs complete episodes (1 mark)

---

## Question 4 (5 marks)

**Question:** What is experience replay in Deep Q-Networks (DQN)? Explain its mechanism and discuss at least two benefits it provides for training neural network-based value function approximators.

---

### ✅ Model Answer:

**Mechanism of Experience Replay (2 marks):**

Experience replay stores the agent's experiences in a **replay buffer** D of fixed capacity (typically ~1 million transitions). Each stored experience is a tuple (s, a, r, s', done).

**How it works:**
1. **Collect:** Agent interacts with environment, stores transition (s, a, r, s', done) in buffer D
2. **Sample:** Randomly sample a minibatch of N transitions (e.g., N=32) from D
3. **Train:** Compute loss and update neural network using this minibatch
4. **Repeat:** Continue collecting and sampling throughout training

```
Buffer: [(s₁,a₁,r₁,s'₁), (s₂,a₂,r₂,s'₂), ..., (sₙ,aₙ,rₙ,s'ₙ)]
                                    ↓
                          Random sample minibatch
                                    ↓
                            Train network
```

**Benefit 1: Breaking Temporal Correlation (1.5 marks)**

Sequential experiences are highly correlated because s' from one transition becomes s for the next. Neural networks assume **i.i.d. (independent and identically distributed)** training data.

Without replay: Consecutive samples like (s₁, s₂, s₃, s₄) are correlated
With replay: Random samples like (s₄₇, s₁₂₃, s₈, s₉₉₂) are approximately i.i.d.

This prevents the network from:
- Overfitting to recent experience
- Oscillating due to correlated gradients
- Forgetting important past experiences

**Benefit 2: Sample Efficiency (1.5 marks)**

Each experience can be reused multiple times for training updates. Instead of using each transition once and discarding it:
- Same transition can be sampled many times over training
- Rare but important transitions remain available
- Greatly improves **sample efficiency** - crucial when environment interaction is expensive

**Additional benefits (for extra credit):**
- Enables off-policy learning (stored experiences from old policies)
- Provides diverse training distribution (old and new experiences mixed)
- Avoids catastrophic forgetting of early experiences

---

**Marking Scheme:**
- Clear explanation of mechanism with storage and sampling (2 marks)
- Benefit 1 with explanation of why correlation is problematic (1.5 marks)
- Benefit 2 with explanation of sample reuse (1.5 marks)
- Additional valid benefits (bonus consideration)

---

# Part B (30 Marks)
*Question 5 is compulsory. Answer ONE from Question 6 or 7.*

---

## Question 5 (Compulsory) - 10 marks

Consider a warehouse robot that must navigate a 5×5 grid to pick up packages and deliver them to a shipping zone. The robot can move in four directions and can pick up or drop packages.

### Part (a): MDP Formulation (3 marks)

**Question:** Formulate this problem as a Markov Decision Process (MDP) by defining all five components.

---

#### ✅ Answer:

**States S (0.5 marks):**
```
S = {(x, y, has_package, package_locations)}
where:
  x, y ∈ {0, 1, 2, 3, 4} (robot position)
  has_package ∈ {True, False} (carrying status)
  package_locations = set of (x,y) coordinates where packages remain
```

Alternative simpler formulation:
```
S = (robot_position, carrying_package, packages_remaining)
```

**Actions A (0.5 marks):**
```
A = {UP, DOWN, LEFT, RIGHT, PICK_UP, DROP_OFF}
```
Or numerically: A = {0, 1, 2, 3, 4, 5}

**Transition Function P(s'|s,a) (0.5 marks):**

*Deterministic version:*
- Movement: Robot moves to adjacent cell if valid, stays if wall
- PICK_UP: Succeeds if at package location and not carrying; fails otherwise
- DROP_OFF: Succeeds if at shipping zone and carrying package

*Stochastic version (more realistic):*
- P(intended direction) = 0.85
- P(slip to adjacent direction) = 0.075 each
- P(stay in place at boundary) = 1.0

**Reward Function R(s,a,s') (1 mark):**
```
R(s, a, s') = {
    +50   if successfully delivered package (dropped at shipping zone)
    +5    if successfully picked up package
    -1    per step (encourages efficiency)
    -10   if collision with wall/obstacle
    -5    if invalid action (pick up when no package, drop when not carrying)
}
```

**Discount Factor γ (0.5 marks):**
```
γ = 0.95 
```
Rationale: Moderate discounting balances immediate step penalties with long-term delivery rewards.

---

### Part (b): SARSA Pseudocode (4 marks)

**Question:** Write the pseudocode for implementing SARSA to solve this problem. Include initialization, episode loop, and update rule.

---

#### ✅ Answer:

```
Algorithm: SARSA for Warehouse Robot
────────────────────────────────────────────────────────────────

# INITIALIZATION (0.5 marks)
Initialize Q(s,a) = 0 for all s ∈ S, a ∈ A
Set Q(terminal_states, ·) = 0
Set hyperparameters: α = 0.1, γ = 0.95, ε = 0.3

# EPISODE LOOP (0.5 marks)
For episode = 1 to max_episodes:
    
    # Initialize episode
    s = initial_state  # robot at start, packages at locations
    
    # Choose initial action using ε-greedy (1 mark)
    If random() < ε:
        a = random action from A
    Else:
        a = argmax_a' Q(s, a')
    
    # Step loop
    While s is not terminal:
        
        # Execute action and observe
        Execute action a
        Observe reward r and next state s'
        
        # Choose NEXT action using ε-greedy (KEY FOR SARSA) (1 mark)
        If random() < ε:
            a' = random action from A
        Else:
            a' = argmax_a'' Q(s', a'')
        
        # SARSA UPDATE (1 mark)
        Q(s, a) ← Q(s, a) + α × [r + γ × Q(s', a') - Q(s, a)]
        
        # Transition
        s ← s'
        a ← a'
    
    # Optional: Decay ε
    ε = max(0.01, ε × 0.995)

────────────────────────────────────────────────────────────────
```

**Critical Elements:**
- Initialization of Q-table (0.5 marks)
- Episode structure with state initialization (0.5 marks)
- ε-greedy action selection for BOTH current and next action (1 mark)
- Correct SARSA update rule with Q(s',a') not max (1 mark)

---

### Part (c): SARSA vs Q-learning Comparison (3 marks)

**Question:** Compare how SARSA and Q-learning would behave differently in this environment if there are dangerous zones the robot should avoid. Which algorithm would you recommend and why?

---

#### ✅ Answer:

**Behavioral Differences (1 mark):**

**SARSA (On-Policy):**
- Learns the value of the policy it actually follows
- If using ε-greedy, the Q-values reflect the possibility of random exploratory actions
- Will learn to avoid states NEAR dangerous zones because random exploration might accidentally step into danger

**Q-Learning (Off-Policy):**
- Learns optimal Q-values assuming greedy behavior
- The max operator always considers the best action, ignoring exploration risk
- May learn a policy that walks very close to danger, underestimating the risk

**Illustration:**
```
    ┌─────┬─────┬─────┬─────┬─────┐
    │Start│     │     │     │     │
    ├─────┼─────┼─────┼─────┼─────┤
    │     │ XXX │ XXX │ XXX │Goal │
    ├─────┼─────┼─────┼─────┼─────┤  XXX = Danger Zone
    │     │     │     │     │     │
    └─────┴─────┴─────┴─────┴─────┘
    
    Q-learning: May learn path right along danger edge (optimal but risky)
    SARSA: Learns safer path that stays further from danger
```

**Impact in Dangerous Environments (1 mark):**

- **SARSA accounts for exploration noise:** Because it uses the actual next action a' (which might be random), the learned Q-values incorporate the risk of accidental dangerous moves
- **Q-learning ignores exploration risk:** Assumes perfect greedy execution, which doesn't happen during learning with ε-greedy
- **During training:** Q-learning robot may suffer more collisions with dangerous zones

**Recommendation (1 mark):**

**I recommend SARSA** for this warehouse robot because:

1. **Safety-critical application:** Warehouse robots are expensive; collisions have real costs
2. **Continuous operation:** The robot must perform reasonably DURING learning, not just after
3. **Acceptable suboptimality:** A slightly longer path that avoids danger is preferable to an optimal but risky path
4. **Exploration is ongoing:** Even after initial training, some exploration may continue for adaptation

**When Q-learning might be preferred:** If you can train in simulation first, then deploy a fully greedy policy, Q-learning's optimal policy is better.

---

## Question 6 - Option (a): Policy Gradient Methods (10 marks)

### Part (i): Fundamental Idea (3 marks)

**Question:** Explain the fundamental idea behind policy gradient methods. Why do we parameterize the policy directly instead of learning a value function?

---

#### ✅ Answer:

**Core Concept (1.5 marks):**

Policy gradient methods parameterize the policy directly as π_θ(a|s) - a probability distribution over actions given states, controlled by parameters θ (typically neural network weights).

Instead of learning value functions and deriving a policy (as in Q-learning), we directly optimize the policy parameters θ to maximize expected return:

```
J(θ) = E_π_θ[G_0] = E_π_θ[Σ_{t=0}^∞ γᵗ R_{t+1}]

Goal: Find θ* = argmax_θ J(θ)
```

We use gradient ascent: θ ← θ + α∇_θJ(θ)

**Why Parameterize Policy Directly (1.5 marks):**

| Reason | Explanation |
|--------|-------------|
| **Continuous action spaces** | Value-based methods need argmax_a Q(s,a), which is intractable for continuous actions. Policy gradient naturally outputs continuous actions. |
| **Stochastic policies** | Can directly represent and learn stochastic policies, useful for partially observable environments or games requiring randomization. |
| **Smooth optimization** | Small changes in θ → small changes in policy → stable learning. Value-based methods can have discontinuous policy changes. |
| **Simpler representation** | Sometimes the optimal policy has a simpler structure than the optimal value function. |
| **Better convergence** | Guaranteed to converge to at least a local optimum (with appropriate step sizes). |

---

### Part (ii): Policy Gradient Theorem (4 marks)

**Question:** Write the policy gradient theorem and explain each term. Discuss why we use log probabilities in the gradient.

---

#### ✅ Answer:

**The Policy Gradient Theorem (2 marks):**

```
∇_θ J(θ) = E_π_θ [∇_θ log π_θ(a|s) · Q^π(s,a)]
```

Or equivalently in trajectory form:
```
∇_θ J(θ) = E_τ~π_θ [Σ_t ∇_θ log π_θ(a_t|s_t) · G_t]
```

**Term-by-Term Explanation (1 mark):**

| Term | Meaning |
|------|---------|
| **∇_θ J(θ)** | Gradient of expected return with respect to policy parameters |
| **E_π_θ[·]** | Expectation under trajectories generated by policy π_θ |
| **∇_θ log π_θ(a\|s)** | Score function - direction to increase probability of action a in state s |
| **Q^π(s,a)** | How good action a is in state s under current policy |
| **G_t** | Return from timestep t (can replace Q in practice) |

**Interpretation:** Move policy parameters in the direction that increases probability of actions that led to high returns.

**Why Log Probabilities (1 mark):**

1. **Likelihood Ratio Trick:**
   ```
   ∇_θ π_θ(a|s) = π_θ(a|s) · ∇_θ log π_θ(a|s)
   ```
   This allows us to write the gradient as an expectation we can sample.

2. **Mathematical Convenience:**
   - For trajectory probability: P(τ) = Π_t π(a_t|s_t)
   - Log converts product to sum: log P(τ) = Σ_t log π(a_t|s_t)
   - Gradients of sums are easier than gradients of products

3. **Numerical Stability:**
   - Products of many small probabilities → vanishingly small numbers
   - Log scale prevents underflow issues

---

### Part (iii): REINFORCE and Baseline (3 marks)

**Question:** Describe the REINFORCE algorithm and explain the high variance problem. How does adding a baseline help?

---

#### ✅ Answer:

**REINFORCE Algorithm (1 mark):**

```
For each episode:
    Generate trajectory τ = (s_0, a_0, r_1, s_1, a_1, ..., s_T) using π_θ
    
    For t = 0 to T-1:
        G_t = Σ_{k=t}^{T-1} γ^{k-t} r_{k+1}    # Return from step t
        
        θ ← θ + α · γᵗ · G_t · ∇_θ log π_θ(a_t|s_t)
```

REINFORCE is a Monte Carlo policy gradient method - it uses complete episode returns.

**High Variance Problem (1 mark):**

REINFORCE suffers from high variance because:

1. **Full trajectory dependence:** Return G_t depends on many random variables (all future rewards)
2. **Credit assignment:** ALL actions get credit/blame based on total episode return, even if only one action was good/bad
3. **Noisy gradients:** Single episode samples are very noisy estimates of the true gradient
4. **Return magnitude:** Absolute return values can vary widely between episodes

This leads to:
- Slow learning (need many samples to average out noise)
- Unstable training (gradients fluctuate wildly)

**Baseline Solution (1 mark):**

Subtract a baseline b(s) from returns:

```
θ ← θ + α · (G_t - b(s_t)) · ∇_θ log π_θ(a_t|s_t)
```

**Common baseline:** b(s) = V(s) (learned state value function)

**Why this works:**
1. **Unbiased:** E[∇_θ log π · b(s)] = 0 because b(s) is independent of action choice
2. **Variance reduction:** Var(G_t - b(s)) < Var(G_t) when b(s) ≈ E[G_t|s]

**Intuition:** Instead of "increase probability of actions with high return," we get "increase probability of actions with BETTER THAN AVERAGE return."

---

## Question 6 - Option (b): Value Function Approximation (10 marks)

### Part (i): Why Approximation & Linear Methods (3 marks)

**Question:** Explain why tabular methods become impractical for large state spaces. Describe linear function approximation for value functions.

---

#### ✅ Answer:

**Why Tabular Methods Fail (1 mark):**

Tabular methods store V(s) or Q(s,a) for every state (or state-action pair) in a table.

**Problems with large state spaces:**
1. **Memory:** Atari games have ~10^67 possible states - impossible to store
2. **Sample complexity:** Must visit each state multiple times to learn its value
3. **No generalization:** Learning V(s) tells us nothing about V(s') for similar s'
4. **Continuous states:** Infinite states → infinite table size

**Linear Function Approximation (2 marks):**

Instead of a table, approximate value as a linear combination of features:

```
V̂(s; w) = w^T φ(s) = Σ_{i=1}^d w_i φ_i(s)
```

Where:
- **φ(s)** is a feature vector of dimension d representing state s
- **w** is the weight vector (learned parameters)
- **d << |S|** (much fewer parameters than states)

**Examples of features φ(s):**
- Tile coding (discretized overlapping regions)
- Radial basis functions
- Polynomial features
- Hand-crafted domain features

**Update Rule (Semi-gradient TD):**
```
w ← w + α · δ · φ(s)
where δ = r + γV̂(s';w) - V̂(s;w)  [TD error]
```

**Benefits:**
- **Generalization:** Similar states have similar feature vectors → similar values
- **Memory efficiency:** Only d parameters instead of |S|
- **Sample efficiency:** One update affects all states with similar features

---

### Part (ii): The Deadly Triad (4 marks)

**Question:** Discuss the deadly triad problem in reinforcement learning. What are its three components and why is it problematic?

---

#### ✅ Answer:

**The Three Components (2 marks):**

| Component | Description |
|-----------|-------------|
| **1. Function Approximation** | Using a parameterized function (linear, neural network) instead of a table to represent values |
| **2. Bootstrapping** | Updating value estimates using other value estimates (as in TD learning), rather than waiting for complete returns |
| **3. Off-Policy Learning** | Learning about a target policy (e.g., optimal/greedy) while following a different behavior policy (e.g., exploratory) |

**Why It's Problematic (2 marks):**

When all three elements are combined, learning can become unstable or even **diverge** (values grow unboundedly):

1. **Function approximation introduces errors:**
   - The true value function may not be representable by our function class
   - Approximation errors in one state affect estimated values of other states

2. **Bootstrapping propagates errors:**
   - TD updates: V(s) ← ... + γV(s')
   - Errors in V(s') feed into updates for V(s)
   - Errors can compound and amplify over time

3. **Off-policy amplifies in rarely-visited states:**
   - Behavior policy samples some states rarely
   - Target policy may prefer different states
   - Errors accumulate in states the target policy visits but behavior policy doesn't

**Classic Example: Baird's Counterexample**
A simple MDP where semi-gradient TD with linear function approximation diverges when learning off-policy.

**Solutions:**
- Experience replay (DQN)
- Target networks (DQN)
- Gradient TD methods (TDC, GTD2)
- Careful feature design
- Regularization

---

### Part (iii): How DQN Addresses Challenges (3 marks)

**Question:** Explain how DQN addresses the challenges of combining neural networks with Q-learning. Focus on the key architectural decisions.

---

#### ✅ Answer:

**Challenge 1: Correlated Samples → Experience Replay (1 mark)**

**Problem:** Sequential RL experiences are highly correlated; neural networks need i.i.d. data.

**DQN Solution - Experience Replay:**
- Store transitions (s, a, r, s', done) in replay buffer D
- Sample random minibatches for training
- Breaks temporal correlation
- Enables data reuse (sample efficiency)

**Challenge 2: Moving Targets → Target Network (1 mark)**

**Problem:** In Q-learning, the target y = r + γ max Q(s',a') changes as Q is updated, causing oscillations.

**DQN Solution - Target Network:**
- Maintain separate network Q_target with parameters θ⁻
- Target: y = r + γ max_a' Q(s', a'; θ⁻)
- Update θ⁻ ← θ only every C steps (e.g., C=10,000)
- Keeps targets stable during training windows

**Challenge 3: Scale Sensitivity → Reward Clipping (0.5 marks)**

**Problem:** Different games have vastly different reward scales.

**DQN Solution:**
- Clip all rewards to [-1, +1]
- Ensures consistent gradient magnitudes across games

**Challenge 4: Temporal Information → Frame Stacking (0.5 marks)**

**Problem:** Single frame doesn't capture motion/velocity.

**DQN Solution:**
- Stack 4 consecutive frames as input (84×84×4)
- Allows network to infer motion direction and speed

---

## Question 7 - Option (a): Multi-Agent RL (10 marks)

### Part (i): Cooperative vs Competitive (3 marks)

**Question:** Define cooperative and competitive multi-agent settings. Provide one real-world example of each.

---

#### ✅ Answer:

**Cooperative Setting (1 mark):**

In cooperative multi-agent RL, all agents share a common goal and work together to maximize a joint reward signal. Agents may need to coordinate, communicate, and divide tasks.

**Characteristics:**
- Shared reward function: R_team = R₁ = R₂ = ... = Rₙ
- Success depends on coordination
- Agents benefit from helping each other

**Real-World Example:** 
**Warehouse robots (Amazon):** Multiple robots coordinate to fulfill orders efficiently. They must avoid collisions, share pathways, and collectively minimize delivery time. One robot helping another reach a package benefits the overall system.

**Competitive Setting (1 mark):**

In competitive multi-agent RL, agents have opposing or conflicting goals. One agent's gain typically comes at another's expense (zero-sum or general-sum games).

**Characteristics:**
- Opposing rewards: Often R₁ = -R₂ (zero-sum)
- Agents try to outperform/defeat opponents
- Modeling opponent behavior is crucial

**Real-World Example:**
**Algorithmic trading:** Trading agents compete for profits in financial markets. If one agent buys low and sells high, another agent likely sold low. Agents must model and exploit competitors' strategies.

**Mixed Settings (0.5 marks):**
Many real scenarios are mixed - like team sports where players cooperate with teammates while competing against opponents.

---

### Part (ii): Multi-Agent Challenges (4 marks)

**Question:** Explain the challenges that arise in multi-agent RL compared to single-agent settings. Focus on non-stationarity and credit assignment.

---

#### ✅ Answer:

**Challenge 1: Non-Stationarity (2 marks)**

**The Problem:**
From each agent's perspective, the environment appears to change over time because other agents are also learning and updating their policies.

**Formally:**
- Agent i sees transition: P(s'|s, aᵢ) 
- But the true dynamics depend on ALL agents: P(s'|s, a₁, a₂, ..., aₙ)
- As other agents change policies, P(s'|s, aᵢ) effectively changes

**Consequences:**
- Violates the stationarity assumption underlying most RL theory
- Convergence guarantees from single-agent RL don't apply
- Agents may "chase" each other's policies without converging
- Q-values become non-stationary (Q(s,a) changes even for fixed (s,a))

**Example:** In a game, if agent 2 suddenly switches from aggressive to defensive play, the optimal response for agent 1 changes completely.

**Challenge 2: Credit Assignment (2 marks)**

**The Problem:**
When multiple agents act simultaneously, determining which agent's actions caused the observed outcomes is difficult.

**Specific Issues:**

| Issue | Description |
|-------|-------------|
| **Shared reward** | If all agents receive R_team, who deserves credit when it's high? |
| **Free-riding** | Agents might learn to let others do the work while still receiving reward |
| **Noise** | Good actions by agent 1 might be masked by bad actions from agent 2 |
| **Delayed effects** | Agent 1's action at t=5 might only matter because of agent 2's action at t=10 |

**Solutions:**
- **Difference rewards:** R_i = R_team - R_team_without_agent_i
- **Counterfactual credit:** What would have happened with default action?
- **Attention mechanisms:** Learn which agents' actions mattered

---

### Part (iii): Independent Q-Learning (3 marks)

**Question:** Describe how independent Q-learning works in multi-agent settings and discuss its limitations.

---

#### ✅ Answer:

**How Independent Q-Learning Works (1.5 marks):**

Each agent i maintains its own Q-function Q_i(s, a_i) and learns as if it were alone in the environment:

```
For each agent i:
    Observe state s (might be local observation o_i)
    Choose action a_i using ε-greedy on Q_i(s, ·)
    Execute a_i, observe reward r_i and next state s'
    Update: Q_i(s, a_i) ← Q_i(s, a_i) + α[r_i + γ max_{a'} Q_i(s', a') - Q_i(s, a_i)]
```

**Key Features:**
- Each agent ignores other agents' actions
- No explicit modeling of other agents
- Simple and scalable (no joint action space)

**Limitations (1.5 marks):**

| Limitation | Explanation |
|------------|-------------|
| **Non-stationarity** | Environment appears non-stationary from each agent's view because other agents' policies change. Q-values oscillate. |
| **No coordination** | Cannot learn strategies requiring joint actions. If success needs (a₁=left, a₂=right), independent learners may never discover this. |
| **No convergence guarantees** | Unlike single-agent Q-learning, no theoretical guarantee of convergence in general games. |
| **Ignores information** | Other agents' actions contain useful information that's being discarded. |
| **Suboptimal equilibria** | May converge to Nash equilibria that are Pareto-dominated (everyone could be better off). |

**When It Works:**
- Agents are truly independent (no interaction)
- Coordination is not required
- As a baseline before trying complex methods

---

## Question 7 - Option (b): Advanced TD Methods (10 marks)

### Part (i): Eligibility Traces (3 marks)

**Question:** Explain the concept of eligibility traces and how they unify MC and TD methods. What does the parameter λ control?

---

#### ✅ Answer:

**Concept of Eligibility Traces (2 marks):**

Eligibility traces are a mechanism that maintains a decaying memory of which states have been recently visited, allowing credit assignment to states encountered many steps before a reward.

**How they work:**
```
e_t(s) = {
    γλ·e_{t-1}(s) + 1   if s = S_t  (just visited)
    γλ·e_{t-1}(s)       otherwise   (decay)
}
```

**Intuition:** Each state's eligibility indicates how much "credit" or "blame" it should receive for the current TD error. Recently visited states have high eligibility; distant states have low eligibility.

**Unifying MC and TD:**

| λ Value | Behavior | Equivalent To |
|---------|----------|---------------|
| λ = 0 | Only update predecessor state | TD(0) |
| λ = 1 | Update all states in episode equally | Monte Carlo |
| 0 < λ < 1 | Update all states with exponential decay | Interpolation |

**What λ Controls (1 mark):**

λ ∈ [0,1] controls the **trace decay rate** or equivalently the **bootstrapping degree**:

- **λ = 0:** Immediate credit only (pure TD, maximum bootstrapping)
- **λ = 1:** Full trajectory credit (pure MC, no bootstrapping)
- **Intermediate λ:** Geometric weighting of n-step returns

Higher λ means:
- More states receive credit for each TD error
- Longer effective backup length
- More like Monte Carlo (higher variance, lower bias)

---

### Part (ii): TD(λ) Derivation (4 marks)

**Question:** Derive and explain the TD(λ) algorithm. How does it interpolate between TD(0) and Monte Carlo?

---

#### ✅ Answer:

**The λ-Return (2 marks):**

TD(λ) uses the **λ-return** as its target, which is a weighted average of n-step returns:

```
G_t^λ = (1-λ) Σ_{n=1}^∞ λ^{n-1} G_t^{(n)}
```

Where the n-step return is:
```
G_t^{(n)} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γⁿV(S_{t+n})
```

**Weighting Interpretation:**
- 1-step return G_t^{(1)} gets weight (1-λ)
- 2-step return G_t^{(2)} gets weight (1-λ)λ
- n-step return G_t^{(n)} gets weight (1-λ)λ^{n-1}
- Weights sum to 1: (1-λ)(1 + λ + λ² + ...) = 1

**Interpolation Between TD(0) and MC (2 marks):**

**Case: λ = 0**
```
G_t^{λ=0} = (1-0) × G_t^{(1)} + 0 = G_t^{(1)} = R_{t+1} + γV(S_{t+1})
```
This is exactly the TD(0) target!

**Case: λ = 1**
```
G_t^{λ=1} = lim_{λ→1} G_t^λ

As λ→1, all weight shifts to G_t^{(∞)} = G_t (full MC return)
```
This is exactly the Monte Carlo return!

**Case: 0 < λ < 1**
```
G_t^λ combines all n-step returns:
- Short-term returns (small n) get higher weight
- Long-term returns (large n) get lower weight  
- Smooth interpolation between TD and MC
```

**Geometric Weighting Diagram:**
```
Weight
  ▲
  │  ╭─(1-λ)        ← 1-step
  │  │  ╭─(1-λ)λ    ← 2-step
  │  │  │  ╭─(1-λ)λ² ← 3-step
  │  │  │  │
  └──┴──┴──┴────────────▶ n-step returns
```

---

### Part (iii): Forward vs Backward View (3 marks)

**Question:** Compare forward-view and backward-view implementations of TD(λ). Which is used in practice and why?

---

#### ✅ Answer:

**Forward View (1 mark):**

The forward view defines what TD(λ) is trying to compute:
- At each timestep t, look **forward** to compute the λ-return G_t^λ
- Update: V(S_t) ← V(S_t) + α[G_t^λ - V(S_t)]

**Characteristics:**
- Conceptually clean and theoretically motivated
- **Not implementable online:** Need to wait until episode ends to compute G_t^λ
- Requires storing entire trajectory
- Used for understanding, not implementation

**Backward View (1 mark):**

The backward view provides an **incremental, online** algorithm:

```
At each step:
    δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)     # TD error
    
    e(S_t) ← e(S_t) + 1                       # Increment trace for current state
    
    For all states s:
        V(s) ← V(s) + α·δ_t·e(s)              # Update proportional to trace
        e(s) ← γλ·e(s)                         # Decay all traces
```

**Characteristics:**
- Updates happen every step
- No need to store trajectory
- Memory: Only need current traces (size |S|)
- Computationally efficient

**Why Backward View is Used in Practice (1 mark):**

| Reason | Explanation |
|--------|-------------|
| **Online learning** | Can update after every transition, not just at episode end |
| **Memory efficient** | Only stores eligibility trace vector, not full trajectory |
| **Works for continuing tasks** | No episode termination needed |
| **Computationally tractable** | O(|S|) per step instead of O(T×|S|) at episode end |
| **Equivalent to forward view** | Mathematically produces same updates (for linear function approximation) |

**Equivalence Result:** For linear function approximation, the sum of backward-view updates over an episode exactly equals the forward-view λ-return update.

---

# End of Mock Test

---

## Scoring Summary

| Question | Marks | Key Assessment Criteria |
|----------|-------|------------------------|
| Q1 (MCQs) | 5 | Correct answer only |
| Q2 | 5 | Definitions, examples, when-to-use reasoning |
| Q3 | 5 | Bellman derivation, update rule, bootstrapping explanation |
| Q4 | 5 | Mechanism clear, two benefits well-explained |
| Q5(a) | 3 | All 5 MDP components present and reasonable |
| Q5(b) | 4 | Correct SARSA pseudocode with key elements |
| Q5(c) | 3 | Clear comparison, justified recommendation |
| Q6 or Q7 | 10 | Depth, accuracy, examples, mathematical rigor |
| **Total** | **50** | |

---

## Final Exam Tips

1. **Read questions carefully** - note exactly what's asked
2. **Show your work** - partial credit for correct reasoning
3. **Use diagrams** - especially for DQN architecture, backup diagrams
4. **Write equations clearly** - define all symbols
5. **Manage time** - Part A should take ~45 min, Part B ~75 min
6. **Answer what you know first** - then return to difficult questions

**Good luck!** 🎓