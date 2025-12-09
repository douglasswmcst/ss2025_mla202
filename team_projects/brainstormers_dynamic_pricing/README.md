# Dynamic Pricing with Deep Reinforcement Learning

Implementation of DQN-based dynamic pricing for e-commerce platforms.

## Team: BrainStormers

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train agent
python train.py

# Test environment
python environment.py
```

## Project Structure
- `environment.py` - E-commerce pricing environment
- `agent.py` - DQN agent with experience replay
- `train.py` - Training script
- `requirements.txt` - Dependencies

## Results
- RL agent learns to optimize pricing dynamically
- Balances revenue maximization with customer satisfaction
- Adapts to competitor prices and demand fluctuations
