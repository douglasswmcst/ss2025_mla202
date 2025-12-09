import numpy as np
import gym
from gym import spaces

class DynamicPricingEnv(gym.Env):
    """
    E-commerce Dynamic Pricing Environment

    State: [current_price, inventory, demand_level, competitor_price, hour_of_day]
    Action: 0-4 representing price changes: [-10%, -5%, 0%, +5%, +10%]
    Reward: Profit with customer satisfaction consideration
    """

    def __init__(self, base_price=100, max_inventory=1000, max_steps=100):
        super(DynamicPricingEnv, self).__init__()

        self.base_price = base_price
        self.cost_per_unit = base_price * 0.6  # 60% of base price
        self.max_inventory = max_inventory
        self.max_steps = max_steps

        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # State space: [price, inventory, demand, competitor_price, hour]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([500, max_inventory, 100, 500, 23]),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.current_price = self.base_price
        self.inventory = self.max_inventory
        self.total_revenue = 0
        self.total_units_sold = 0

        # Random initial conditions
        self.demand_level = np.random.uniform(30, 70)
        self.competitor_price = np.random.uniform(80, 120)
        self.hour = np.random.randint(0, 24)

        return self._get_state()

    def _get_state(self):
        """Return current state"""
        return np.array([
            self.current_price,
            self.inventory,
            self.demand_level,
            self.competitor_price,
            self.hour
        ], dtype=np.float32)

    def _calculate_demand(self, price):
        """
        Calculate demand based on price elasticity
        Demand decreases with higher prices and increases with competitor advantage
        """
        # Price elasticity: demand decreases as price increases
        price_factor = max(0, (150 - price) / 150)

        # Competitor factor: demand increases if we're cheaper
        competitor_factor = 1.0 + 0.3 * (self.competitor_price - price) / price
        competitor_factor = np.clip(competitor_factor, 0.5, 2.0)

        # Time factor: peak hours have higher demand
        time_factor = 1.0 + 0.3 * np.sin(self.hour * np.pi / 12)

        # Base demand with noise
        base_demand = self.demand_level + np.random.normal(0, 5)

        # Final demand calculation
        demand = base_demand * price_factor * competitor_factor * time_factor
        demand = max(0, int(demand))

        return min(demand, self.inventory)

    def step(self, action):
        """
        Execute action and return next state, reward, done, info

        Actions:
        0: Decrease price by 10%
        1: Decrease price by 5%
        2: Maintain price
        3: Increase price by 5%
        4: Increase price by 10%
        """
        # Apply action to price
        price_changes = [-0.10, -0.05, 0.0, 0.05, 0.10]
        price_change = price_changes[action]
        self.current_price = self.current_price * (1 + price_change)

        # Ensure price stays in reasonable range
        self.current_price = np.clip(self.current_price, 50, 200)

        # Calculate demand and units sold
        units_sold = self._calculate_demand(self.current_price)

        # Calculate revenue and profit
        revenue = units_sold * self.current_price
        cost = units_sold * self.cost_per_unit
        profit = revenue - cost

        # Update inventory
        self.inventory -= units_sold

        # Calculate reward with penalties
        # Penalty for extreme pricing
        price_penalty = 0
        if self.current_price < 70 or self.current_price > 150:
            price_penalty = -50

        # Penalty for stockout
        stockout_penalty = -100 if self.inventory <= 0 else 0

        # Reward is profit with penalties
        reward = profit + price_penalty + stockout_penalty

        # Update tracking variables
        self.total_revenue += revenue
        self.total_units_sold += units_sold

        # Update environment dynamics
        self.current_step += 1
        self.hour = (self.hour + 1) % 24
        self.demand_level = np.clip(
            self.demand_level + np.random.normal(0, 3), 20, 80
        )
        self.competitor_price = np.clip(
            self.competitor_price + np.random.normal(0, 5), 80, 120
        )

        # Check if episode is done
        done = self.current_step >= self.max_steps or self.inventory <= 0

        # Info for logging
        info = {
            'revenue': revenue,
            'units_sold': units_sold,
            'profit': profit,
            'total_revenue': self.total_revenue,
            'total_units_sold': self.total_units_sold
        }

        return self._get_state(), reward, done, info

    def render(self, mode='human'):
        """Print current state"""
        print(f"Step: {self.current_step}")
        print(f"Price: ${self.current_price:.2f}, Inventory: {self.inventory}")
        print(f"Demand Level: {self.demand_level:.1f}, Competitor: ${self.competitor_price:.2f}")
        print(f"Total Revenue: ${self.total_revenue:.2f}, Units Sold: {self.total_units_sold}")
        print("-" * 50)

# Test the environment
if __name__ == "__main__":
    env = DynamicPricingEnv()
    state = env.reset()
    print("Initial State:", state)

    for _ in range(5):
        action = env.action_space.sample()  # Random action
        state, reward, done, info = env.step(action)
        env.render()
        print(f"Reward: {reward:.2f}\n")
        if done:
            break
