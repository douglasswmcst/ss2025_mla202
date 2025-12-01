# MLA202 Reinforcement Learning - Mock Test 2
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
Which of the following correctly describes the relationship between V*(s) and Q*(s,a)?

(a) V*(s) = Σ_a Q*(s,a)

(b) V*(s) = min_a Q*(s,a)

(c) V*(s) = max_a Q*(s,a)

(d) V*(s) = average_a Q*(s,a)

---

#### ✅ Answer: (c)

**Explanation:** The optimal state-value function V*(s) equals the value of taking the best possible action from state s:

```
V*(s) = max_a Q*(s,a)
```

**Intuition:** If you're in state s and act optimally, you'll take the action with the highest Q-value. Therefore, the value of being in state s (under optimal policy) equals the maximum Q-value achievable.

**Why others are wrong:**
- **(a)** Summing all Q-values makes no sense - we pick the best action, not all actions
- **(b)** Taking minimum would give the worst action's value, not optimal
- **(d)** Averaging is for stochastic policies, not optimal (greedy) policy

**Related equation:** Q*(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')

---

### Q1.2
In the context of DQN, what problem does Double DQN specifically address?

(a) Slow training speed

(b) Maximization bias leading to overestimation

(c) Catastrophic forgetting

(d) Exploration inefficiency

---

#### ✅ Answer: (b)

**Explanation:** Double DQN addresses **maximization bias** - the systematic overestimation of Q-values that occurs when the same network is used to both select and evaluate actions.

**The Problem in Standard DQN:**
```
Target: y = r + γ max_a' Q(s', a'; θ⁻)
                 └─────────────────────┘
                 Same network selects AND evaluates
```

When Q-estimates have noise, max picks actions where noise is positive → overestimation.

**Double DQN Solution:**
```
a* = argmax_a' Q(s', a'; θ)      ← Online network SELECTS
y = r + γ Q(s', a*; θ⁻)          ← Target network EVALUATES
```

**Why others are wrong:**
- **(a)** Double DQN doesn't speed up training; it improves accuracy
- **(c)** Experience replay addresses catastrophic forgetting
- **(d)** Exploration strategies (ε-greedy, UCB) address exploration

---

### Q1.3
What is the key characteristic that makes an environment satisfy the Markov property?

(a) All rewards are positive

(b) The state space is finite

(c) The future depends only on the current state, not the history

(d) Actions are deterministic

---

#### ✅ Answer: (c)

**Explanation:** The Markov property (also called the "memoryless" property) states:

```
P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0, A_0) = P(S_{t+1} | S_t, A_t)
```

**In plain English:** Given the current state and action, the probability distribution over next states is independent of all previous states and actions. The current state contains all relevant information.

**Why others are wrong:**
- **(a)** Reward signs are irrelevant to the Markov property
- **(b)** Both finite and continuous state spaces can be Markov or non-Markov
- **(d)** Markov environments can have stochastic transitions

**Example:** Chess is Markov (board position is sufficient). Poker is non-Markov (past cards dealt matter but aren't in current state).

---

### Q1.4
Which statement about the discount factor γ is FALSE?

(a) γ = 0 makes the agent completely myopic

(b) γ = 1 is always safe to use in any environment

(c) Higher γ values make the agent more far-sighted

(d) γ helps ensure finite returns in infinite-horizon problems

---

#### ✅ Answer: (b)

**Explanation:** γ = 1 is **NOT always safe**. In infinite-horizon continuing tasks, γ = 1 can lead to infinite returns:

```
G = R_1 + R_2 + R_3 + ... = Σ_{t=1}^∞ R_t → ∞ (if rewards don't decay)
```

With γ < 1:
```
G = R_1 + γR_2 + γ²R_3 + ... ≤ R_max / (1-γ) < ∞
```

**When γ = 1 IS safe:** Finite-horizon episodic tasks where episodes always terminate.

**Why others are correct:**
- **(a)** TRUE: γ = 0 → G = R_1 only (ignores all future rewards)
- **(c)** TRUE: Higher γ means future rewards matter more
- **(d)** TRUE: γ < 1 ensures geometric sum converges

---

### Q1.5
In Monte Carlo policy evaluation, what does the algorithm estimate?

(a) The transition probabilities P(s'|s,a)

(b) The optimal policy π*

(c) The value function V^π for a given policy π

(d) The reward function R(s,a,s')

---

#### ✅ Answer: (c)

**Explanation:** Monte Carlo **policy evaluation** (also called MC prediction) estimates the state-value function V^π for a **given, fixed policy** π.

```
V^π(s) = E_π[G_t | S_t = s]
       ≈ average of returns observed when starting from state s
```

**Process:**
1. Follow policy π to generate episodes
2. For each state visited, record the return G
3. V(s) = average of all returns from state s

**Why others are wrong:**
- **(a)** MC doesn't estimate transition probabilities (that would be model learning)
- **(b)** MC evaluation doesn't find optimal policy; MC **control** does
- **(d)** Rewards are observed directly, not estimated

---

## Question 2 (5 marks)

**Question:** Explain the concept of bootstrapping in reinforcement learning. Compare how TD(0), Monte Carlo, and n-step TD methods use bootstrapping, and discuss the trade-offs involved.

---

### ✅ Model Answer:

**Definition of Bootstrapping (1.5 marks):**

Bootstrapping means updating value estimates using other value estimates, rather than waiting for actual complete returns.

```
Bootstrapping:     V(s) ← ... + γV(s')     [uses estimate V(s')]
Non-bootstrapping: V(s) ← ... + G          [uses actual return G]
```

**Key insight:** Bootstrapping allows learning before knowing the final outcome by substituting current estimates for unknown future values.

**Comparison of Methods (2.5 marks):**

| Method | Bootstrapping? | Target | Update After |
|--------|---------------|--------|--------------|
| **Monte Carlo** | No | G_t (complete return) | Episode end |
| **TD(0)** | Yes (1 step) | R + γV(s') | Every step |
| **n-step TD** | Yes (n steps) | R_1 + γR_2 + ... + γ^{n-1}R_n + γⁿV(s_{t+n}) | Every n steps |

**TD(0) - Maximum Bootstrapping:**
```
V(s) ← V(s) + α[R + γV(s') - V(s)]
                   └────┘
                   Bootstrap from next state
```

**Monte Carlo - No Bootstrapping:**
```
V(s) ← V(s) + α[G - V(s)]
               └─┘
               Actual complete return (no estimates used)
```

**n-step TD - Partial Bootstrapping:**
```
V(s) ← V(s) + α[R_1 + γR_2 + ... + γⁿV(s_{t+n}) - V(s)]
                                   └──────────┘
                                   Bootstrap from n steps ahead
```

**Trade-offs (1 mark):**

| Aspect | More Bootstrapping (TD) | Less Bootstrapping (MC) |
|--------|------------------------|------------------------|
| **Bias** | Higher (estimates may be wrong) | Lower (uses actual returns) |
| **Variance** | Lower (fewer random variables) | Higher (full trajectory noise) |
| **Speed** | Faster (updates every step) | Slower (waits for episode end) |
| **Data efficiency** | Higher | Lower |
| **Continuing tasks** | Works | Doesn't work |

**Optimal choice:** Often intermediate n-step methods or TD(λ) balance these trade-offs best.

---

## Question 3 (5 marks)

**Question:** Describe the Dueling DQN architecture. Draw a diagram showing the network structure and explain why separating value and advantage streams improves learning.

---

### ✅ Model Answer:

**Architecture Description (2 marks):**

Dueling DQN separates the Q-function into two components:
1. **Value stream V(s):** How good is it to be in state s (scalar)
2. **Advantage stream A(s,a):** How much better is action a than average (|A| values)

These are combined to produce Q-values:
```
Q(s,a) = V(s) + [A(s,a) - (1/|A|) Σ_{a'} A(s,a')]
                           └─────────────────────┘
                           Mean subtraction for identifiability
```

**Network Diagram (1.5 marks):**

```
                        DUELING DQN ARCHITECTURE
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ┌──────────┐     ┌─────────────────────────┐                     │
│   │  Input   │     │   Shared Convolutional  │                     │
│   │  State   │────▶│      Feature Layers     │                     │
│   │ (84×84×4)│     │                         │                     │
│   └──────────┘     └───────────┬─────────────┘                     │
│                                │                                    │
│                    ┌───────────┴───────────┐                       │
│                    │                       │                        │
│                    ▼                       ▼                        │
│         ┌─────────────────┐     ┌─────────────────┐                │
│         │  Value Stream   │     │ Advantage Stream│                │
│         │                 │     │                 │                │
│         │   FC layers     │     │   FC layers     │                │
│         │       ↓         │     │       ↓         │                │
│         │    V(s)         │     │   A(s,a₁)       │                │
│         │   (scalar)      │     │   A(s,a₂)       │                │
│         │                 │     │     ...         │                │
│         │                 │     │   A(s,aₙ)       │                │
│         └────────┬────────┘     └────────┬────────┘                │
│                  │                       │                          │
│                  └───────────┬───────────┘                         │
│                              │                                      │
│                              ▼                                      │
│               ┌──────────────────────────────┐                     │
│               │  Q(s,a) = V(s) + A(s,a)      │                     │
│               │           - mean(A(s,·))      │                     │
│               │                              │                      │
│               │  Output: [Q(s,a₁), Q(s,a₂),  │                     │
│               │           ..., Q(s,aₙ)]      │                     │
│               └──────────────────────────────┘                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Why Separation Improves Learning (1.5 marks):**

**1. Better State Value Learning:**
- In many states, the choice of action doesn't matter much
- Standard DQN must learn similar Q-values for all actions separately
- Dueling learns V(s) once, then small A(s,a) differences
- Example: In a hallway with no enemies, all movement actions have similar value

**2. More Efficient Updates:**
- When updating Q(s,a), both V(s) and A(s,a) are updated
- V(s) affects ALL Q-values for state s simultaneously
- Faster propagation of state value information

**3. Better Generalization:**
- Many states share similar values but different advantages
- Shared V stream generalizes state value learning
- Advantage stream focuses on action-specific differences

**4. Improved Performance:**
- Particularly beneficial when many actions have similar values
- Empirically shows better performance on Atari games

---

## Question 4 (5 marks)

**Question:** What is the "deadly triad" in reinforcement learning? Explain each component and provide an example of an algorithm that suffers from this problem and one that avoids it.

---

### ✅ Model Answer:

**The Deadly Triad Definition (1.5 marks):**

The deadly triad refers to the combination of three elements that, when present together, can cause learning instability or divergence in RL algorithms:

```
┌─────────────────────────────────────────────────────────────┐
│                     THE DEADLY TRIAD                        │
│                                                             │
│    ┌──────────────────┐                                    │
│    │    Function      │                                    │
│    │  Approximation   │──────┐                             │
│    └──────────────────┘      │                             │
│                              │                              │
│    ┌──────────────────┐      │      ┌─────────────────┐   │
│    │  Bootstrapping   │──────┼─────▶│   INSTABILITY   │   │
│    │                  │      │      │   or DIVERGENCE │   │
│    └──────────────────┘      │      └─────────────────┘   │
│                              │                              │
│    ┌──────────────────┐      │                             │
│    │   Off-Policy     │──────┘                             │
│    │    Learning      │                                    │
│    └──────────────────┘                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Component 1: Function Approximation (1 mark)**

Using parameterized functions (neural networks, linear models) instead of tables to represent value functions.

- **Why problematic:** Approximation errors in one state affect others
- **Example:** V̂(s;w) = w^T φ(s)

**Component 2: Bootstrapping (1 mark)**

Updating estimates based on other estimates (as in TD learning).

- **Why problematic:** Errors propagate through the bootstrap chain
- **Example:** V(s) ← V(s) + α[r + γV(s') - V(s)]

**Component 3: Off-Policy Learning (1 mark)**

Learning about a target policy different from the behavior policy generating data.

- **Why problematic:** Importance sampling ratios can amplify errors
- **Example:** Q-learning (learns greedy while following ε-greedy)

**Algorithm Examples (0.5 marks):**

**Suffers from Deadly Triad:**
- **Q-learning with linear function approximation** - Has all three elements and can diverge (Baird's counterexample)
- **Naive DQN without stabilization techniques**

**Avoids the Deadly Triad:**
- **Monte Carlo with function approximation** - No bootstrapping (removes one element)
- **Tabular Q-learning** - No function approximation
- **On-policy SARSA with function approximation** - No off-policy learning
- **DQN** - Uses target networks and experience replay to stabilize (doesn't remove elements but mitigates instability)

---

# Part B (30 Marks)
*Question 5 is compulsory. Answer ONE from Question 6 or 7.*

---

## Question 5 (Compulsory) - 10 marks

A ride-sharing company wants to use reinforcement learning to optimize driver positioning. Drivers can be directed to different city zones to pick up passengers. The goal is to minimize passenger wait times while maximizing driver utilization.

### Part (a): MDP Formulation (3 marks)

**Question:** Formulate this problem as a Markov Decision Process (MDP) by defining all five components.

---

#### ✅ Answer:

**States S (0.5 marks):**
```
S = (zone_demands, driver_positions, time_of_day, day_of_week, weather, special_events)

Where:
- zone_demands[i] = estimated demand in zone i (e.g., {low, medium, high})
- driver_positions = distribution of drivers across zones
- time_of_day = hour (affects demand patterns)
- day_of_week = weekday/weekend (affects demand)
- weather = {clear, rain, snow} (affects demand and travel)
- special_events = nearby concerts, games, etc.
```

Or simplified per-driver view:
```
S = (current_zone, local_demand, nearby_demands, time_features)
```

**Actions A (0.5 marks):**
```
A = {stay_in_zone, move_to_zone_1, move_to_zone_2, ..., move_to_zone_n}

Or continuous: A = target_zone ∈ {1, 2, ..., n}
```

**Transition Function P(s'|s,a) (0.5 marks):**

Stochastic transitions based on:
- Driver movement time between zones
- Probability of getting a ride in each zone
- Demand evolution over time

```
P(s'|s, move_to_zone_i) depends on:
  - Travel time to zone i
  - Demand in zone i at arrival time
  - Other drivers' movements (multi-agent)
```

**Reward Function R(s,a,s') (1 mark):**
```
R = α × fare_revenue 
    - β × passenger_wait_time
    - γ × driver_idle_time  
    - δ × fuel_cost

Where:
- fare_revenue: Money earned from completed rides
- passenger_wait_time: Penalty for long waits (customer satisfaction)
- driver_idle_time: Penalty for unproductive time
- fuel_cost: Cost of repositioning
```

Alternative simpler reward:
```
R = +fare if pickup passenger
    -0.1 per minute of passenger waiting
    -0.05 per minute driver idle
```

**Discount Factor γ (0.5 marks):**
```
γ = 0.99 (high discount since continuous operation)
```
Rationale: Ride-sharing is a continuing task; drivers work long shifts. High γ encourages long-term positioning strategy over short-term gains.

---

### Part (b): Algorithm Selection (4 marks)

**Question:** Compare Q-learning and Policy Gradient methods for this problem. Which would you recommend and why? Discuss at least three factors in your comparison.

---

#### ✅ Answer:

**Factor 1: Action Space Nature (1 mark)**

| Aspect | Q-Learning | Policy Gradient |
|--------|------------|-----------------|
| **Strength** | Natural for discrete actions | Handles continuous actions naturally |
| **For this problem** | Works if zones are discrete | Better if fine-grained positioning needed |

**Analysis:** If we discretize the city into zones (discrete actions), Q-learning works well. If we want continuous GPS coordinates, policy gradient is more natural.

**Factor 2: Multi-Agent Considerations (1 mark)**

| Aspect | Q-Learning | Policy Gradient |
|--------|------------|-----------------|
| **Multi-agent** | Independent Q-learning struggles | Can be extended to multi-agent (MAPPO) |
| **Coordination** | Hard to coordinate drivers | Can learn coordinated policies |

**Analysis:** Ride-sharing involves many drivers. Policy gradient methods (especially actor-critic) extend better to multi-agent settings where drivers must implicitly coordinate.

**Factor 3: Sample Efficiency vs Stability (1 mark)**

| Aspect | Q-Learning | Policy Gradient |
|--------|------------|-----------------|
| **Sample efficiency** | More sample efficient (off-policy, replay) | Less sample efficient |
| **Stability** | Can be unstable with function approx | More stable gradients |
| **Real-world training** | Can learn from historical data | Needs on-policy data |

**Analysis:** Q-learning can learn from historical ride data (off-policy). Policy gradient needs fresh on-policy data, which is expensive in real deployment.

**Recommendation (1 mark):**

**I recommend Q-Learning (specifically DQN or Rainbow)** for the following reasons:

1. **Historical data available:** Ride-sharing companies have massive historical datasets. Q-learning's off-policy nature allows learning from this data.

2. **Discrete zone decisions:** Most practical implementations discretize the city into zones, making the action space naturally discrete.

3. **Sample efficiency:** Real-world experimentation is expensive; Q-learning's replay buffer makes efficient use of each experience.

4. **Proven track record:** Similar applications (Uber, Lyft pricing) have used value-based methods successfully.

**However, consider Policy Gradient if:**
- Continuous fine-grained positioning is needed
- Multi-agent coordination is critical
- You have a good simulator for on-policy training

---

### Part (c): Exploration Strategy (3 marks)

**Question:** Design an exploration strategy suitable for this real-world application. Consider that poor exploration could result in lost revenue and unhappy customers.

---

#### ✅ Answer:

**Recommended Strategy: Decayed ε-greedy with Safety Constraints (1.5 marks)**

```
Exploration Policy:
────────────────────────────────────────────────────────
1. Base ε-greedy with decay:
   ε_t = max(ε_min, ε_0 × decay^t)
   
   Initial: ε_0 = 0.3 (30% exploration initially)
   Minimum: ε_min = 0.05 (always 5% exploration)
   Decay: Per day of operation
   
2. Safety constraints on exploration:
   - Never explore to zones with 0 historical demand
   - Limit exploration distance (max 2 zones from current)
   - During peak hours: reduce ε by 50%
   
3. Smart exploration selection:
   When exploring, don't choose uniformly random:
   - Weight by historical demand in zones
   - Prefer under-explored zones (UCB-style bonus)
────────────────────────────────────────────────────────
```

**Handling Real-World Constraints (1 mark):**

| Constraint | Solution |
|------------|----------|
| **Lost revenue** | Limit exploration during peak demand hours |
| **Customer satisfaction** | Ensure minimum driver coverage per zone |
| **Driver acceptance** | Make exploration moves seem reasonable to drivers |
| **Safety** | Never send drivers to dangerous/closed areas |

**Implementation Details (0.5 marks):**

```python
def select_action(state, Q, epsilon, constraints):
    if random() < epsilon and not is_peak_hour(state):
        # Safe exploration
        valid_zones = get_nearby_zones(state.current_zone, max_distance=2)
        valid_zones = filter_by_minimum_demand(valid_zones)
        weights = get_historical_demand_weights(valid_zones)
        return weighted_random_choice(valid_zones, weights)
    else:
        # Exploitation
        return argmax_a Q(state, a)
```

**Alternative Approaches (0.5 marks):**
- **Thompson Sampling:** Sample from posterior distribution of Q-values (natural exploration-exploitation balance)
- **UCB:** Add uncertainty bonus: Q(s,a) + c√(log(t)/N(s,a))
- **Boltzmann:** Softmax over Q-values with temperature

---

## Question 6 - Option (a): Actor-Critic Methods (10 marks)

### Part (i): Actor-Critic Architecture (3 marks)

**Question:** Explain the actor-critic architecture. What are the roles of the actor and critic, and how do they interact during learning?

---

#### ✅ Answer:

**Architecture Overview (1.5 marks):**

Actor-Critic methods combine policy-based (actor) and value-based (critic) approaches:

```
┌──────────────────────────────────────────────────────────────┐
│                    ACTOR-CRITIC ARCHITECTURE                  │
│                                                              │
│  ┌─────────────────┐              ┌─────────────────┐       │
│  │     ACTOR       │              │     CRITIC      │       │
│  │   π_θ(a|s)      │              │    V_w(s)       │       │
│  │                 │              │    or Q_w(s,a)  │       │
│  │  Policy Network │              │  Value Network  │       │
│  │  (parameters θ) │              │  (parameters w) │       │
│  └────────┬────────┘              └────────┬────────┘       │
│           │                                │                 │
│           │ selects action                 │ evaluates       │
│           ▼                                ▼                 │
│      ┌─────────┐                    ┌───────────┐           │
│      │ Action  │───────────────────▶│ TD Error  │           │
│      │    a    │                    │    δ      │           │
│      └────┬────┘                    └─────┬─────┘           │
│           │                               │                  │
│           │                               │ guides actor     │
│           │                               │ update           │
│           ▼                               ▼                  │
│  ┌─────────────────────────────────────────────────┐        │
│  │              ENVIRONMENT                         │        │
│  │         reward r, next state s'                  │        │
│  └─────────────────────────────────────────────────┘        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Role of the Actor (0.75 marks):**
- Parameterized policy π_θ(a|s) that selects actions
- Updated using policy gradient, guided by critic's evaluation
- Goal: Learn the best policy
- Update: θ ← θ + α_actor · δ · ∇_θ log π_θ(a|s)

**Role of the Critic (0.75 marks):**
- Estimates value function V_w(s) or Q_w(s,a)
- Evaluates how good the actor's actions are
- Computes TD error δ to guide actor updates
- Update: w ← w + α_critic · δ · ∇_w V_w(s)
- Acts as a learned baseline to reduce variance

**Interaction:**
1. Actor selects action a ~ π_θ(·|s)
2. Environment returns reward r and next state s'
3. Critic computes TD error: δ = r + γV_w(s') - V_w(s)
4. Critic updates its value estimate using δ
5. Actor updates policy using δ as the advantage estimate

---

### Part (ii): Advantage Function (4 marks)

**Question:** Define the advantage function A(s,a). Explain why using advantages instead of Q-values reduces variance in policy gradient methods. Derive how A(s,a) relates to Q(s,a) and V(s).

---

#### ✅ Answer:

**Definition of Advantage Function (1 mark):**

The advantage function measures how much better action a is compared to the average action in state s:

```
A^π(s,a) = Q^π(s,a) - V^π(s)
```

**Interpretation:**
- A(s,a) > 0: Action a is better than average
- A(s,a) < 0: Action a is worse than average  
- A(s,a) = 0: Action a is exactly average

**Derivation of Relationship (1 mark):**

Starting from definitions:
```
V^π(s) = E_a~π [Q^π(s,a)]        [V is expected Q over policy]
       = Σ_a π(a|s) Q^π(s,a)

Therefore:
A^π(s,a) = Q^π(s,a) - V^π(s)
         = Q^π(s,a) - Σ_{a'} π(a'|s) Q^π(s,a')
```

**Property:** The expected advantage under the policy is zero:
```
E_a~π [A^π(s,a)] = E_a~π [Q^π(s,a)] - V^π(s) = V^π(s) - V^π(s) = 0
```

**Why Advantages Reduce Variance (2 marks):**

**Standard Policy Gradient (high variance):**
```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) · Q^π(s,a)]
```

Problem: Q-values can be large and vary significantly across states.

**Policy Gradient with Advantage (lower variance):**
```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) · A^π(s,a)]
```

**Why this reduces variance:**

1. **Centering:** A(s,a) is centered around 0 (E[A] = 0), while Q(s,a) can be large positive values. Centered values have lower variance in gradient estimates.

2. **Same expected gradient:** Subtracting baseline V(s) doesn't change the expected gradient:
   ```
   E[∇ log π · V(s)] = V(s) · E[∇ log π] = V(s) · 0 = 0
   ```
   
3. **Removes state-dependent offset:** Different states have different base values. Advantage removes this, focusing only on action-dependent differences.

**Numerical Example:**
```
State s with V(s) = 100
Q-values: Q(s,a₁) = 102, Q(s,a₂) = 98, Q(s,a₃) = 100

Using Q-values: gradients scaled by ~100 (high variance)
Using Advantages: A(s,a₁) = +2, A(s,a₂) = -2, A(s,a₃) = 0
                  gradients scaled by ~2 (much lower variance)
```

---

### Part (iii): A2C vs A3C (3 marks)

**Question:** Compare Advantage Actor-Critic (A2C) and Asynchronous Advantage Actor-Critic (A3C). What are the benefits of asynchronous training?

---

#### ✅ Answer:

**A2C (Synchronous) (1 mark):**

```
┌─────────────────────────────────────────────────────────┐
│                         A2C                              │
│                                                          │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│   │Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker n │      │
│   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘      │
│        │           │           │           │            │
│        │    Collect experiences (synchronously)         │
│        │           │           │           │            │
│        └───────────┴───────────┴───────────┘            │
│                         │                                │
│                         ▼                                │
│              ┌────────────────────┐                     │
│              │  Aggregate & Update │                     │
│              │   (single update)   │                     │
│              └────────────────────┘                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

- Multiple workers collect experiences in parallel
- Workers synchronize and aggregate gradients
- Single update step with averaged gradients
- Wait for slowest worker (synchronization barrier)

**A3C (Asynchronous) (1 mark):**

```
┌─────────────────────────────────────────────────────────┐
│                         A3C                              │
│                                                          │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│   │Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker n │      │
│   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘      │
│        │           │           │           │            │
│        ▼           ▼           ▼           ▼            │
│   ┌─────────────────────────────────────────────┐      │
│   │         Global Shared Parameters            │      │
│   │                                             │      │
│   │  Workers update asynchronously (no waiting) │      │
│   └─────────────────────────────────────────────┘      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

- Workers run independently without synchronization
- Each worker updates global parameters asynchronously
- No waiting for other workers
- Updates may use slightly stale parameters

**Benefits of Asynchronous Training (A3C) (1 mark):**

| Benefit | Explanation |
|---------|-------------|
| **Speed** | No synchronization barriers; no waiting for slowest worker |
| **Exploration diversity** | Different workers explore different parts of state space simultaneously |
| **Implicit regularization** | Stale gradients act like noise, preventing overfitting |
| **Hardware efficiency** | Better CPU utilization; no need for GPU |
| **Robustness** | If one worker fails, others continue |

**Trade-offs:**
- A3C: Faster but gradient updates slightly stale
- A2C: Slower but more stable updates; often achieves similar final performance with modern hardware

**Modern Preference:** A2C is often preferred now because:
- GPUs make synchronous updates fast
- More reproducible results
- Easier to debug

---

## Question 6 - Option (b): Reward Shaping (10 marks)

### Part (i): Reward Shaping Concept (3 marks)

**Question:** Explain what reward shaping is and why it might be needed. Discuss the risks of poorly designed reward shaping.

---

#### ✅ Answer:

**What is Reward Shaping (1 mark):**

Reward shaping adds additional reward signals F(s,a,s') to the environment's natural reward R to guide learning:

```
R_shaped(s,a,s') = R(s,a,s') + F(s,a,s')
```

**Purpose:** Provide denser feedback to speed up learning when natural rewards are sparse or delayed.

**Why It's Needed (1 mark):**

| Problem | Example | How Shaping Helps |
|---------|---------|-------------------|
| **Sparse rewards** | Robot navigation: +1 only at goal | Add distance-to-goal bonus |
| **Delayed rewards** | Chess: +1 only for winning | Add material advantage bonus |
| **Credit assignment** | Complex tasks with many steps | Intermediate milestones |
| **Exploration guidance** | Large state spaces | Encourage visiting new states |

**Risks of Poor Reward Shaping (1 mark):**

1. **Reward hacking:** Agent finds ways to maximize shaped reward without achieving the true goal
   - Example: Robot repeatedly approaching but not entering goal zone to collect approach bonus

2. **Suboptimal policies:** Shaping changes the optimal policy if not designed carefully
   - Example: Excessive step penalty makes agent take shortcuts that fail

3. **Local optima:** Agent gets stuck maximizing intermediate rewards
   - Example: Collecting all bonuses before attempting the actual task

4. **Unintended behaviors:** Agent exploits loopholes in shaping function
   - Example: AI boat racing game - agent learned to collect bonuses by spinning in circles rather than finishing race

---

### Part (ii): Potential-Based Reward Shaping (4 marks)

**Question:** Describe potential-based reward shaping. Prove that it preserves the optimal policy and explain why this property is important.

---

#### ✅ Answer:

**Potential-Based Reward Shaping Definition (1 mark):**

A shaping function F is potential-based if there exists a potential function Φ(s) such that:

```
F(s, a, s') = γΦ(s') - Φ(s)
```

The total shaped reward becomes:
```
R_shaped = R(s,a,s') + γΦ(s') - Φ(s)
```

**Proof of Policy Invariance (2 marks):**

**Goal:** Show that the optimal policy under R_shaped is the same as under R.

**Step 1:** Consider the shaped return from state s₀:
```
G_shaped = Σ_{t=0}^∞ γᵗ [R_t + γΦ(s_{t+1}) - Φ(s_t)]
```

**Step 2:** Expand the shaping terms:
```
Shaping terms = [γΦ(s₁) - Φ(s₀)] + γ[γΦ(s₂) - Φ(s₁)] + γ²[γΦ(s₃) - Φ(s₂)] + ...
              = -Φ(s₀) + γΦ(s₁) - γΦ(s₁) + γ²Φ(s₂) - γ²Φ(s₂) + ...
              = -Φ(s₀) + lim_{t→∞} γᵗΦ(s_t)
              = -Φ(s₀) + 0     [since γ < 1]
              = -Φ(s₀)
```

**Step 3:** Therefore:
```
G_shaped = G_original - Φ(s₀)
```

**Step 4:** For any two policies π₁ and π₂:
```
V_shaped^{π₁}(s) - V_shaped^{π₂}(s) = [V^{π₁}(s) - Φ(s)] - [V^{π₂}(s) - Φ(s)]
                                     = V^{π₁}(s) - V^{π₂}(s)
```

Since the difference in values is preserved, the ordering of policies is preserved, and the optimal policy remains optimal. ∎

**Why This Property is Important (1 mark):**

1. **Safety:** Guaranteed not to change what the agent ultimately learns
2. **No reward hacking:** Agent can't exploit shaping to get infinite reward
3. **Design freedom:** Can add any potential-based shaping without worry
4. **Theoretical grounding:** Formal guarantee unlike arbitrary shaping
5. **Practical guidance:** Natural choice: Φ(s) = estimate of V(s) or distance to goal

---

### Part (iii): Designing Reward Functions (3 marks)

**Question:** For a robot learning to walk, design a reward function that balances multiple objectives: forward progress, energy efficiency, and stability. Explain your design choices.

---

#### ✅ Answer:

**Proposed Reward Function:**

```
R(s,a,s') = α·R_progress + β·R_energy + γ·R_stability + δ·R_alive + R_termination

Where:
────────────────────────────────────────────────────────────────

R_progress = v_forward                    # Forward velocity (m/s)
           - λ₁·|v_lateral|              # Penalize sideways motion
           - λ₂·|ω_rotation|             # Penalize spinning

R_energy = -Σᵢ |τᵢ · ωᵢ|                 # Sum of joint power usage
                                          # (torque × angular velocity)

R_stability = -||a_body||                 # Penalize body acceleration
            - max(|roll|, |pitch|)·λ₃    # Penalize tilting

R_alive = +1 per timestep                # Survival bonus

R_termination = {
    -100  if fallen (height < threshold)
    0     otherwise
}
────────────────────────────────────────────────────────────────
```

**Design Choices and Rationale (2 marks):**

| Component | Design Choice | Rationale |
|-----------|---------------|-----------|
| **Forward velocity** | Positive coefficient (main objective) | Primary goal is to walk forward |
| **Lateral velocity** | Small negative penalty | Prevent crab-walking or drifting |
| **Energy usage** | Moderate negative penalty | Encourage efficient, natural gaits |
| **Body acceleration** | Negative penalty | Smooth motion, avoid jerky movements |
| **Tilt angles** | Negative penalty | Keep robot upright |
| **Alive bonus** | Constant +1 | Encourage staying upright longer |
| **Fall penalty** | Large negative | Strong signal to avoid falling |

**Weight Selection Strategy (1 mark):**

```
Suggested weights:
α = 1.0     # Progress (primary objective)
β = 0.001   # Energy (minor consideration)
γ = 0.1     # Stability (moderate importance)
δ = 0.01    # Alive bonus (small but continuous)
λ₁ = 0.5    # Lateral penalty
λ₂ = 0.1    # Rotation penalty
λ₃ = 0.5    # Tilt penalty
```

**Key Principles:**
1. **Dominant term:** Forward progress should dominate so robot actually moves
2. **Bounded penalties:** Use terms that can't grow unboundedly
3. **Dense feedback:** Most terms provide signal every timestep
4. **Interpretable:** Each term has clear physical meaning
5. **Tunable:** Weights can be adjusted based on desired behavior

---

## Question 7 - Option (a): Inverse Reinforcement Learning (10 marks)

### Part (i): IRL Concept (3 marks)

**Question:** Explain what Inverse Reinforcement Learning (IRL) is and how it differs from standard RL. Provide a real-world scenario where IRL would be more appropriate than standard RL.

---

#### ✅ Answer:

**What is Inverse Reinforcement Learning (1.5 marks):**

IRL is the problem of inferring the reward function from observed behavior (demonstrations), rather than learning a policy from a given reward.

```
Standard RL:    Given R → Learn π*
Inverse RL:     Given π* (demonstrations) → Learn R
```

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   STANDARD RL                      INVERSE RL                │
│   ───────────                      ──────────                │
│                                                              │
│   Known: Environment + Reward      Known: Environment +      │
│                                           Expert demos       │
│   Learn: Optimal policy π*         Learn: Reward function R  │
│                                                              │
│   ┌─────────┐                      ┌─────────────────┐      │
│   │ Reward  │                      │ Expert behavior │      │
│   │    R    │                      │      π_E        │      │
│   └────┬────┘                      └────────┬────────┘      │
│        │                                    │                │
│        ▼                                    ▼                │
│   ┌─────────┐                      ┌─────────────────┐      │
│   │ Learn   │                      │ Infer reward    │      │
│   │ Policy  │                      │ that explains   │      │
│   └────┬────┘                      │ expert behavior │      │
│        │                                    │                │
│        ▼                                    ▼                │
│   ┌─────────┐                      ┌─────────────────┐      │
│   │ π*      │                      │ R (then can     │      │
│   └─────────┘                      │ learn π* from R)│      │
│                                    └─────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight:** The reward function is often the most compact and transferable representation of a task.

**Real-World Scenario for IRL (1.5 marks):**

**Autonomous Driving:**

**Why IRL is appropriate:**
1. **Reward is hard to specify:** What's the reward for safe, comfortable driving? It's a complex combination of:
   - Safety (don't crash)
   - Comfort (smooth acceleration)
   - Efficiency (reach destination quickly)
   - Rule following (stop at red lights)
   - Social norms (don't cut people off)
   
2. **Expert demonstrations are plentiful:** Millions of hours of human driving data available

3. **Humans can't articulate their reward:** Drivers can't explicitly state how they weight all factors

4. **Transferability:** Once reward is learned, can train new policies (e.g., for different vehicles)

**Alternative scenarios:**
- Robot manipulation (learn from human demonstrations)
- Game AI (learn to play like a specific player style)
- Animation (learn natural motion from motion capture)

---

### Part (ii): IRL Challenges (4 marks)

**Question:** Discuss the key challenges in Inverse Reinforcement Learning. Explain the ambiguity problem and describe one approach to address it.

---

#### ✅ Answer:

**Challenge 1: Reward Ambiguity (1.5 marks)**

**The Problem:** Multiple reward functions can explain the same observed behavior.

```
Example: Expert walks from A to B in a straight line

Possible reward functions that explain this:
- R₁(s) = -distance_to_B           (minimize distance to goal)
- R₂(s) = -distance_to_B - energy  (also minimize energy)
- R₃(s) = 0 everywhere except +1 at B
- R₄(s) = constant (any policy is optimal, including the observed one!)
```

**Formally:** The set of reward functions consistent with optimal behavior is infinite.

**Degenerate solutions:**
- R = 0 (constant): Every policy is optimal
- R = c (any constant): Same problem

**Challenge 2: Suboptimal Demonstrations (1 mark)**

Real experts aren't perfectly optimal:
- Human mistakes and lapses in attention
- Different skill levels among demonstrators
- Demonstrations may be near-optimal, not optimal

IRL typically assumes demonstrations are optimal, leading to errors when they're not.

**Challenge 3: State Space Coverage (0.5 marks)**

Demonstrations may not cover all states:
- Expert only shows some situations
- Can't infer reward for unvisited states
- Need to generalize reward function

**Challenge 4: Computational Cost (0.5 marks)**

Many IRL methods require solving RL in inner loop:
- For each candidate R, compute optimal π
- Check if π matches demonstrations
- Very expensive for complex environments

**Approach to Address Ambiguity: Maximum Entropy IRL (1.5 marks)**

**Key Idea:** Among all reward functions explaining the demonstrations, choose the one that makes the expert's behavior have maximum entropy (be as random as possible while still matching observed statistics).

**Principle:** Don't assume more structure than necessary (Occam's razor for distributions).

**Formulation:**
```
max_R H(π_R) - D_KL(π_R || π_demo)

Subject to: Feature expectations match
            E_π[φ(s,a)] = E_demo[φ(s,a)]
```

**Result:** Unique reward function (up to scaling) that:
- Explains demonstrations
- Makes no unnecessary assumptions
- Generalizes well to new situations

---

### Part (iii): Behavioral Cloning vs IRL (3 marks)

**Question:** Compare behavioral cloning with inverse reinforcement learning. When would you prefer one over the other?

---

#### ✅ Answer:

**Behavioral Cloning (BC) (1 mark):**

Directly learns a policy by supervised learning on demonstrations:

```
Training: π_θ = argmin_θ Σᵢ L(π_θ(sᵢ), aᵢ)
          (minimize loss between predicted and expert actions)
```

**Characteristics:**
- Simple supervised learning problem
- Fast to train
- No reward function involved
- Learns π directly

**Inverse RL (IRL) (1 mark):**

Infers reward function from demonstrations, then derives policy:

```
Step 1: R* = IRL(demonstrations)
Step 2: π* = RL(R*)
```

**Characteristics:**
- More complex (RL in inner loop)
- Slower to train
- Learns transferable reward function
- Can generalize beyond demonstrations

**Comparison:**

| Aspect | Behavioral Cloning | Inverse RL |
|--------|-------------------|------------|
| **What's learned** | Policy π(a\|s) | Reward R(s,a) |
| **Training complexity** | Simple (supervised) | Complex (RL in loop) |
| **Data efficiency** | Lower | Higher |
| **Compounding errors** | Severe problem | Mitigated |
| **Generalization** | Poor to new states | Better |
| **Transferability** | Must re-train | Transfer R to new settings |

**When to Prefer Each (1 mark):**

**Prefer Behavioral Cloning when:**
- Lots of demonstration data available
- Task is simple with limited state space
- Quick solution needed
- Demonstrator covers most situations
- No need for transfer

**Prefer IRL when:**
- Limited demonstrations
- Need to generalize beyond demonstrated states
- Want to transfer to different environments
- Need interpretable reward function
- Compounding errors are a concern
- Demonstrations are near-optimal but not perfect

**Example Decision:**
- Teaching a robot a specific manufacturing task with fixed setup → BC
- Learning driving behavior to transfer across different cities → IRL

---

## Question 7 - Option (b): Model-Based RL (10 marks)

### Part (i): Model-Based vs Model-Free (3 marks)

**Question:** Compare model-based and model-free reinforcement learning. What are the key trade-offs between these approaches?

---

#### ✅ Answer:

**Model-Based RL (1 mark):**

Learns or uses a model of environment dynamics:
```
Model: P̂(s'|s,a) and R̂(s,a)
Use: Planning, generating synthetic experience, look-ahead
```

**Examples:** Dyna-Q, MBPO, World Models, AlphaZero

**Model-Free RL (1 mark):**

Learns value functions or policies directly from experience without explicit model:
```
Learn: Q(s,a) or π(a|s) directly from (s,a,r,s') transitions
```

**Examples:** Q-learning, DQN, PPO, SAC

**Key Trade-offs (1 mark):**

| Aspect | Model-Based | Model-Free |
|--------|-------------|------------|
| **Sample efficiency** | Higher (simulate many experiences) | Lower (need real experiences) |
| **Computational cost** | Higher (model learning + planning) | Lower (just policy/value learning) |
| **Model errors** | Compound over planning horizon | N/A |
| **Asymptotic performance** | Limited by model accuracy | Can achieve optimal |
| **Applicability** | Requires learnable dynamics | Works for any environment |
| **Transfer** | Model may transfer | Policy usually task-specific |

---

### Part (ii): Dyna Architecture (4 marks)

**Question:** Describe the Dyna architecture and explain how it combines real and simulated experience. Provide the Dyna-Q algorithm.

---

#### ✅ Answer:

**Dyna Architecture Overview (1.5 marks):**

Dyna integrates learning and planning by:
1. Learning a model from real experience
2. Using the model to generate simulated experience
3. Updating value function from both real and simulated experience

```
┌─────────────────────────────────────────────────────────────────┐
│                      DYNA ARCHITECTURE                           │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                     REAL EXPERIENCE                      │   │
│   │   Agent ←→ Real Environment → (s, a, r, s')             │   │
│   └────────────────────────┬────────────────────────────────┘   │
│                            │                                     │
│              ┌─────────────┴─────────────┐                      │
│              ▼                           ▼                       │
│   ┌────────────────────┐      ┌────────────────────┐           │
│   │   Model Learning   │      │   Direct RL        │           │
│   │                    │      │   (Q-learning)     │           │
│   │   Learn P̂(s'|s,a)  │      │   Update Q(s,a)    │           │
│   │   Learn R̂(s,a)     │      │                    │           │
│   └─────────┬──────────┘      └────────────────────┘           │
│             │                            ▲                       │
│             │                            │                       │
│             ▼                            │                       │
│   ┌────────────────────┐                │                       │
│   │    PLANNING        │                │                       │
│   │                    │                │                       │
│   │  Generate simulated│────────────────┘                       │
│   │  experience using  │   (update Q from                       │
│   │  learned model     │    simulated experience)               │
│   │                    │                                        │
│   │  (s, a) → P̂,R̂ →   │                                        │
│   │  (s, a, r̂, ŝ')    │                                        │
│   └────────────────────┘                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Dyna-Q Algorithm (2.5 marks):**

```
Algorithm: Dyna-Q
──────────────────────────────────────────────────────────────

Initialize:
    Q(s,a) = 0 for all s, a
    Model(s,a) = empty for all s, a
    
Parameters: α (learning rate), γ (discount), n (planning steps)

Repeat forever:
    
    (a) s ← current state
    
    (b) a ← ε-greedy action from Q(s,·)
    
    (c) Execute a, observe r, s'
    
    ┌──────────────────────────────────────────────────────┐
    │ (d) DIRECT RL: Update Q from real experience         │
    │     Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    └──────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────────────┐
    │ (e) MODEL LEARNING: Update model                     │
    │     Model(s,a) ← (r, s')    [deterministic model]   │
    │                                                      │
    │     For stochastic: update P̂(s'|s,a) and R̂(s,a)     │
    └──────────────────────────────────────────────────────┘
    
    ┌──────────────────────────────────────────────────────┐
    │ (f) PLANNING: n iterations of Q-learning on          │
    │     simulated experience                             │
    │                                                      │
    │     Repeat n times:                                  │
    │         s̃ ← random previously observed state        │
    │         ã ← random action previously taken in s̃     │
    │         (r̃, s̃') ← Model(s̃, ã)                      │
    │         Q(s̃,ã) ← Q(s̃,ã) + α[r̃ + γ max Q(s̃',·) - Q(s̃,ã)]
    └──────────────────────────────────────────────────────┘
    
    (g) s ← s'

──────────────────────────────────────────────────────────────
```

**Key Insight:** Parameter n controls the ratio of planning to real experience. More planning (higher n) = more sample efficient but more computation.

---

### Part (iii): Model Errors and Solutions (3 marks)

**Question:** Discuss the problem of model errors in model-based RL. Describe one technique to mitigate the effects of inaccurate models.

---

#### ✅ Answer:

**The Model Error Problem (1.5 marks):**

When the learned model P̂ differs from true dynamics P, errors compound during planning:

```
True trajectory:    s₀ → s₁ → s₂ → s₃ → s₄ → ...
Model trajectory:   s₀ → ŝ₁ → ŝ₂ → ŝ₃ → ŝ₄ → ...
                         ↑     ↑↑    ↑↑↑   ↑↑↑↑
                      small  medium large  huge
                      error  error  error  error
```

**Why errors compound:**
1. Small error at step 1: ŝ₁ ≈ s₁
2. Planning from ŝ₁ (not s₁) adds more error
3. By step k, model may be in completely wrong region of state space
4. Policies optimized for model may fail catastrophically in reality

**Consequences:**
- Overconfident policies that exploit model errors
- Poor real-world performance despite good simulated performance
- Potentially dangerous in safety-critical applications

**Mitigation Technique: Model Ensemble with Uncertainty (1.5 marks)**

**Approach:** Train multiple models and use disagreement as uncertainty estimate.

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL ENSEMBLE                            │
│                                                              │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐      ┌─────────┐   │
│   │ Model 1 │  │ Model 2 │  │ Model 3 │ ...  │ Model K │   │
│   │  P̂₁     │  │  P̂₂     │  │  P̂₃     │      │  P̂_K    │   │
│   └────┬────┘  └────┬────┘  └────┬────┘      └────┬────┘   │
│        │            │            │                 │        │
│        └────────────┴────────────┴─────────────────┘        │
│                            │                                 │
│                            ▼                                 │
│              ┌───────────────────────────┐                  │
│              │  Prediction: mean(P̂ᵢ)     │                  │
│              │  Uncertainty: var(P̂ᵢ)     │                  │
│              └───────────────────────────┘                  │
│                            │                                 │
│                            ▼                                 │
│              ┌───────────────────────────┐                  │
│              │  When uncertainty high:   │                  │
│              │  - Shorter planning       │                  │
│              │  - More real experience   │                  │
│              │  - Penalize uncertain     │                  │
│              │    regions                │                  │
│              └───────────────────────────┘                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**How it helps:**
1. **Uncertainty quantification:** High disagreement = unreliable prediction
2. **Pessimistic planning:** Avoid states where models disagree
3. **Adaptive horizon:** Plan shorter when uncertain
4. **Targeted exploration:** Collect data in uncertain regions

**Other techniques:**
- **Short planning horizons:** Don't plan too far (model-predictive control)
- **Model validation:** Check model predictions against reality
- **Conservative estimates:** Use lower-bound on value estimates

---

# End of Mock Test 2

---

## Scoring Summary

| Question | Marks | Topics Covered |
|----------|-------|----------------|
| Q1 (MCQs) | 5 | V*/Q* relationship, Double DQN, Markov property, discount factor, MC evaluation |
| Q2 | 5 | Bootstrapping, TD vs MC vs n-step |
| Q3 | 5 | Dueling DQN architecture |
| Q4 | 5 | Deadly triad |
| Q5(a) | 3 | MDP formulation (ride-sharing) |
| Q5(b) | 4 | Algorithm comparison (Q-learning vs Policy Gradient) |
| Q5(c) | 3 | Exploration strategy design |
| Q6/Q7 | 10 | Actor-Critic OR Reward Shaping / IRL OR Model-Based RL |
| **Total** | **50** | |

---

## Final Exam Tips

1. **Practice both mock tests** - they cover complementary topics
2. **Understand trade-offs** - most questions ask you to compare approaches
3. **Draw diagrams** - especially for architectures (Dueling DQN, Actor-Critic, Dyna)
4. **Know real-world applications** - ride-sharing, robotics, games
5. **Manage time** - allocate ~8 minutes per 5-mark question
6. **Show reasoning** - partial credit available for correct approach

**Good luck with your exam!** 🎓