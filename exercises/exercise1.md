### ## Exercise 1: MDP Formulation (Business Scenario) 🤖

You are managing an autonomous warehouse robot. The robot's goal is to retrieve items efficiently while managing its battery life. You need to create a policy for what the robot should do when its battery is low.

**Scenario Details:**

* **States:** The robot can be in one of two states at the start of a task: **'High Battery'** or **'Low Battery'**.
* **Actions:** When in the **'Low Battery'** state, you can choose one of two actions:
    * **'Continue Task':** The robot attempts to finish its current job.
    * **'Go Charge':** The robot immediately returns to the charging station.
* **Rewards & Costs:**
    * Successfully completing a task earns a profit of **$20**.
    * The action **'Go Charge'** costs **$5** in lost time and productivity.
    * If the robot attempts to **'Continue Task'** on a low battery, there is a **50%** chance it succeeds (earning $20) and a **50%** chance its battery dies mid-task, which costs **$40** to recover the robot and reset the task.
* **Transitions (Next State):**
    * If the robot chooses **'Go Charge'**, it will always start the next task in the **'High Battery'** state.
    * If the robot successfully completes a task after choosing **'Continue Task'**, it will start the next task in the **'High Battery'** state (as it gets a quick charge after).
    * If the robot's battery dies, it is recharged fully during recovery and starts the next task in the **'High Battery'** state.

**Your Task:**

1.  For the **'Low Battery'** state, calculate the immediate expected reward for taking the action **'Continue Task'**.
2.  What is the immediate reward for taking the action **'Go Charge'**?
3.  Describe the transition probabilities for the robot starting its *next* task. For example, "If the state is 'Low Battery' and the action is 'Go Charge', the probability of the next state being 'High Battery' is X%."

---

### ## Exercise 2: Bellman Equation Calculation 📚

Consider a simplified model of a software development sprint, represented as a Markov Reward Process (MRP). Your goal is to find the value of starting a new sprint.

**Process Details:**

* **States & Rewards:**
    * The reward for starting in the **'Planning'** state is **-5** (cost of meetings).
    * The reward for transitioning into the **'Development'** state is **+10**.
    * The **'Release'** state is a terminal state, and the reward for reaching it is **+50**.
* **Transition Probabilities from 'Planning':**
    * Probability of going from 'Planning' to **'Development'** = 0.7
    * Probability of going from 'Planning' to **'Release'** = 0.3
* **Known State Values:**
    * The value of being in the 'Development' state is known: $V(\text{Development}) = 25$.
    * The 'Release' state is terminal, so its value is its immediate reward: $V(\text{Release}) = 50$.
* **Discount Factor:** $\gamma = 0.9$

**Your Task:**

Using the Bellman equation for an MRP, calculate the value of being in the **'Planning'** state, $V(\text{Planning})$. You must show all your calculations.

### ## Solution Guide: Exercise 1 (MDP Formulation) 🤖

This exercise tests your ability to translate a business problem into the formal components of a Markov Decision Process (MDP). Let's break it down into its three parts.

#### **Part 1: Calculate the Immediate Expected Reward for 'Continue Task'**

The key here is the phrase **"expected reward."** Because the outcome is uncertain (50% success, 50% failure), we can't use a single value. We must calculate the average reward you would expect to get over many attempts.

* **Step 1: Identify all possible outcomes and their values.**
    * **Outcome A (Success):** The robot completes the task. The problem states this earns a profit of **+$20**.
    * **Outcome B (Failure):** The robot's battery dies. The problem states this results in a cost of **-$40**.

* **Step 2: Identify the probability of each outcome.**
    * Probability of Outcome A (Success) = **50% or 0.5**.
    * Probability of Outcome B (Failure) = **50% or 0.5**.

* **Step 3: Use the formula for expected value.**
    The formula is:
    `Expected Reward = (Probability of Outcome A × Value of Outcome A) + (Probability of Outcome B × Value of Outcome B)`

* **Step 4: Substitute the values and calculate.**
    `Expected Reward = (0.5 × $20) + (0.5 × -$40)`
    `Expected Reward = $10 + (-$20)`
    `Expected Reward = -$10`

**Answer:** The immediate expected reward for taking the action **'Continue Task'** from the 'Low Battery' state is **-$10**.

***

#### **Part 2: Calculate the Immediate Reward for 'Go Charge'**

This part is simpler because the outcome is certain (deterministic). There is no probability involved.

* **Step 1: Identify the value associated with the action.**
    The problem states that the action 'Go Charge' costs **$5** in lost time.

* **Step 2: Convert the cost to a reward value.**
    In reinforcement learning, a cost is simply a negative reward.

**Answer:** The immediate reward for taking the action **'Go Charge'** is **-$5**.

***

#### **Part 3: Describe the Transition Probabilities**

This asks for the probability of what state the robot will be in for the *next* task, given its current state ('Low Battery') and the action taken. This is written as `P(s' | s, a)`, where `s'` is the next state.

* **Case 1: Action is 'Go Charge'**
    * **Current State (s):** 'Low Battery'
    * **Action (a):** 'Go Charge'
    * **Outcome:** The problem states the robot will **always** start the next task in the 'High Battery' state.
    * **Probability:** Since this outcome is certain, the probability is 100%.

    **Answer:** If the state is 'Low Battery' and the action is 'Go Charge', the probability of the next state being **'High Battery' is 100%**.

* **Case 2: Action is 'Continue Task'**
    * **Current State (s):** 'Low Battery'
    * **Action (a):** 'Continue Task'
    * **Outcome 1 (Success):** The robot succeeds and gets a quick charge, starting the next task in 'High Battery'.
    * **Outcome 2 (Failure):** The robot's battery dies, is recovered, and is fully recharged, starting the next task in 'High Battery'.
    * **Conclusion:** In *both* possible outcomes (success or failure), the robot starts its next task in the 'High Battery' state.
    * **Probability:** Because every possible outcome leads to the same next state, the probability is 100%.

    **Answer:** If the state is 'Low Battery' and the action is 'Continue Task', the probability of the next state being **'High Battery' is 100%**.

---

### ## Solution Guide: Exercise 2 (Bellman Equation) 📚

This exercise tests your ability to apply the Bellman equation to calculate the value of a state. We will follow the formula step-by-step.

#### **Step 1: Write Down the Bellman Equation**

First, always start with the formula to guide your work. For a state `s`:
$V(s) = R(s) + \gamma \sum_{s'} P(s'|s)V(s')$

#### **Step 2: Identify All Components from the Problem Description**

Let's break down the problem and assign each piece of information to a variable in our formula. We are solving for $V(\text{Planning})$.

* **State (s):** The state we want to find the value of is **'Planning'**.
* **Immediate Reward R(s):** The reward for starting in 'Planning' is given as **-5**.
* **Discount Factor ($\gamma$):** Given as **0.9**.
* **Successor States (s'):** These are the states we can go to from 'Planning'. They are **'Development'** and **'Release'**.
* **Transition Probabilities P(s'|s):**
    * The probability of moving from 'Planning' to 'Development' is **0.7**.
    * The probability of moving from 'Planning' to 'Release' is **0.3**.
* **Values of Successor States V(s'):**
    * The value of 'Development' is given: $V(\text{Development}) = 25$.
    * The value of 'Release' is given: $V(\text{Release}) = 50$.

#### **Step 3: Substitute the Values into the Equation**

Now, we plug all the numbers from Step 2 into the Bellman equation:
$V(\text{Planning}) = -5 + 0.9 \times [ (P(\text{Development}|\text{Planning}) \times V(\text{Development})) + (P(\text{Release}|\text{Planning}) \times V(\text{Release})) ]$

$V(\text{Planning}) = -5 + 0.9 \times [ (0.7 \times 25) + (0.3 \times 50) ]$

#### **Step 4: Solve the Equation Following the Order of Operations**

1.  **Calculate the value inside the brackets first.** This part represents the expected value of the successor states.
    * $(0.7 \times 25) = 17.5$
    * $(0.3 \times 50) = 15.0$
    * Sum them: $17.5 + 15.0 = 32.5$

2.  **Now, update the equation with this new value.**
    $V(\text{Planning}) = -5 + 0.9 \times [32.5]$

3.  **Apply the discount factor.**
    * $0.9 \times 32.5 = 29.25$

4.  **Update the equation again.**
    $V(\text{Planning}) = -5 + 29.25$

5.  **Perform the final calculation.**
    $V(\text{Planning}) = 24.25$

**Answer:** The calculated value of being in the **'Planning'** state is **24.25**. This number represents the total future reward the agent can expect to receive, discounted back to the present moment, if it starts in the 'Planning' state.