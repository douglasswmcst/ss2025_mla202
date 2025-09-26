# MLA202 - Machine Learning Applications Course

This repository contains my work for the MLA202 Machine Learning Applications course, including practicals, exercises, and study materials.

## Repository Structure

- `practical2/` - Q-Learning implementation and experiments
- `practicals/` - Practical instructions and walkthroughs
- `exercises/` - Course exercises and solutions
- `studyguide/` - Study materials and guides
- `thoughts/` - Additional notes and research

## Practical 2: Q-Learning - Your First Intelligent Agent

### Implementation Summary

I implemented a Q-Learning agent that learns to navigate the FrozenLake-v1 environment through trial and error. The implementation includes:

- **Core Q-learning algorithm** with the temporal difference update rule
- **Epsilon-greedy action selection** for exploration vs exploitation balance
- **Comprehensive training loop** with 5,000 episodes of learning
- **Performance testing and analysis** capabilities
- **Q-table visualization and interpretation** tools

### Performance Results

**Final Q-Learning Agent Performance:**
- **Success Rate: 72.9%** (exceeds the required >70% threshold)
- **Average Reward: 0.729** per episode
- **Improvement: 12.2x better** than random baseline (~6% success rate)
- **Training Episodes: 5,000** with optimized hyperparameters

**Key Hyperparameters Used:**
- Learning Rate (α): 0.1 - balanced learning speed and stability
- Discount Factor (γ): 0.99 - high value for delayed reward scenarios
- Epsilon Decay: 0.99 - faster exploration reduction for efficient learning

### Exercise Answers: Hyperparameter Investigation

#### 1. Learning Rate (α) Analysis
**Tested values:** 0.01, 0.1, 0.5

**Findings:**
- **α = 0.01:** Learns slowly but provides stable, consistent updates. Good for final convergence but requires more episodes.
- **α = 0.1:** Optimal balance - learns efficiently while maintaining stability. Best overall performance.
- **α = 0.5:** Learns very quickly initially but can become unstable with large Q-value updates, potentially overshooting optimal values.

**Key Insight:** Learning rate controls update magnitude - too high causes instability and oscillation, too low causes slow convergence.

#### 2. Discount Factor (γ) Analysis
**Tested values:** 0.9, 0.99, 0.999

**Findings:**
- **Higher γ values (0.99, 0.999) significantly outperform lower values** in FrozenLake
- **γ = 0.9:** Agent becomes more "impatient" and struggles to value the distant goal reward
- **γ = 0.99-0.999:** Agent properly values future rewards, leading to much better path planning

**Key Insight:** FrozenLake only rewards at the goal state - agents need high γ to properly value that distant reward and plan multi-step paths.

#### 3. Epsilon Decay Analysis
**Tested strategies:** Fast decay (0.99) vs Slow decay (0.999)

**Findings:**
- **Fast decay (0.99):** Quickly transitions from exploration to exploitation, good when environment is simple
- **Slow decay (0.999):** Maintains exploration longer, better for complex environments but may waste episodes

**Key Insight:** The exploration schedule must balance learning about the environment (exploration) with using learned knowledge effectively (exploitation). FrozenLake benefits from moderate decay allowing sufficient initial exploration.

### Q-Table Analysis

#### Starting Position Analysis
- **State 0 Q-values:** [0.476, 0.469, 0.461, 0.454]
- **Best action from start:** LEFT (Q-value: 0.476)
- **Insight:** The optimal policy learned that going LEFT from the starting position provides the highest expected reward, suggesting an optimal path that begins leftward.

#### Exploration Coverage Analysis
- **Non-zero Q-values:** 44 out of 64 possible state-action pairs
- **Insight:** The agent successfully explored most of the reachable state space, with some state-action combinations remaining unexplored (likely because they're suboptimal or lead to immediate failure).

#### Terminal State Analysis
- **Goal state (14) Q-values:** DOWN=0.799, UP=0.717, RIGHT=0.694, LEFT=0.625
- **Insight:** All actions from the goal state have high positive Q-values since they all lead to the reward. The slight differences reflect the learning process and the stochastic nature of action selection during training.

### Challenges

The most difficult part of implementing Q-learning was:

1. **Balancing exploration vs exploitation** - Finding the right epsilon decay schedule that allows sufficient exploration while transitioning to effective exploitation
2. **Understanding delayed rewards** - Recognizing that FrozenLake's sparse reward structure (only at goal) requires high discount factors for effective learning
3. **Debugging learning issues** - Initially the agent wasn't learning due to insufficient training episodes and poor hyperparameter choices

### Key Insights

**What surprised me most about Q-learning:**

1. **The power of temporal difference learning** - How the agent can learn effectively by bootstrapping from its own estimates, updating beliefs immediately after each step rather than waiting for episode completion

2. **Hyperparameter sensitivity** - Small changes in learning rate or discount factor had dramatic effects on performance, highlighting the importance of proper tuning

3. **Emergent behavior** - Watching random exploration gradually transform into intelligent, goal-directed behavior through the accumulation of Q-values was fascinating

### Learning Reflection: Q-Learning vs Random Agent

**Behavioral Differences:**

**Random Agent:**
- Takes completely random actions regardless of situation
- Shows no improvement over time
- Achieves ~6% success rate through pure chance
- No pattern recognition or learning capability

**Q-Learning Agent:**
- Starts random but gradually develops intelligent behavior
- Learns to avoid holes and seek the goal through experience
- Achieves 72.9% success rate through learned strategy
- Develops consistent, optimal policy based on accumulated knowledge
- Shows clear improvement throughout training episodes

**The transformation from random to intelligent behavior through Q-learning demonstrates the fundamental power of reinforcement learning - the ability to discover optimal strategies through experience rather than explicit programming.**

## Files Included

### Core Implementation
- `practical2/fixed_qlearning_agent.py` - Main Q-learning implementation achieving >72% success
- `practical2/hyperparameter_experiments.py` - Comprehensive hyperparameter analysis
- `practical2/debug_qlearning.py` - Debug version for understanding learning process

### Analysis and Results
- `practical2/test_qlearning_quick.py` - Quick testing implementation
- Performance results and Q-table analysis (generated during execution)

All code achieves the required >70% success rate on FrozenLake-v1 and includes proper implementation of the Q-learning update rule with meaningful hyperparameter analysis.