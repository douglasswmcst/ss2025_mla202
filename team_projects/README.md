# Team Projects - MLA202 Emergency Assessment

This folder contains implementation code for all team projects.

## Project Status

### ✅ Fully Implemented (Ready to Run)

1. **brainstormers_dynamic_pricing** - Dynamic Pricing with Deep RL
   - Complete DQN implementation
   - E-commerce pricing environment
   - Training and evaluation scripts
   - **Status**: Production-ready

2. **model_minds_dqn** - Deep Q-Network on CartPole
   - Complete DQN from scratch
   - Experience replay and target networks
   - CartPole-v1 environment
   - **Status**: Production-ready

### 📝 Placeholder Implementations (Refer to Guides)

3. **phuratan_tictactoe** - Ultimate Tic-Tac-Toe RL
   - See: [team_guides/PhuRaTan_UltimateTicTacToe_Guide.md](../team_guides/PhuRaTan_UltimateTicTacToe_Guide.md)

4. **qteam_snake** - Snake Game with RL
   - See: [team_guides/QTeam_Snake_Guide.md](../team_guides/QTeam_Snake_Guide.md)

5. **thecrown_qlearning** - Q-Learning Fundamentals
   - See: [team_guides/TheCrown_QLearning_Guide.md](../team_guides/TheCrown_QLearning_Guide.md)

6. **theinnovators_flappybird** - Flappy Bird with DQN
   - See: [team_guides/TheInnovators_FlappyBird_Guide.md](../team_guides/TheInnovators_FlappyBird_Guide.md)

7. **neuraworks_argrpo** - AR-GRPO Image Generation
   - See: [team_guides/NeuraWorks_BLANK_ARGRPO_Guide.md](../team_guides/NeuraWorks_BLANK_ARGRPO_Guide.md)

## Quick Start

### For Fully Implemented Projects

#### BrainStormers - Dynamic Pricing
```bash
cd brainstormers_dynamic_pricing
pip install -r requirements.txt
python train.py  # Train the agent (500 episodes)
```

#### Model Minds - DQN
```bash
cd model_minds_dqn
pip install torch gym numpy matplotlib
python train.py  # Train on CartPole-v1
```

### For Placeholder Projects

Teams should:
1. Navigate to their project folder
2. Open their team guide in `../team_guides/`
3. Copy the complete implementation code from the guide
4. Replace placeholder files with full implementations
5. Follow the guide's day-by-day timeline

## Project Structure

Each project folder contains:
- `README.md` - Project description and usage
- Source files (`.py`) - Implementation code
- `requirements.txt` (if applicable) - Dependencies

## Installation

### Common Dependencies
```bash
# Install common packages
pip install numpy matplotlib torch gym

# For specific projects
pip install pygame  # For Snake and Flappy Bird
pip install pandas  # For data analysis
```

### Using Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install <packages>
```

## Testing

### Test BrainStormers Project
```bash
cd brainstormers_dynamic_pricing
python environment.py  # Test environment
python train.py  # Full training
```

### Test Model Minds Project
```bash
cd model_minds_dqn
python network.py  # Test network
python replay_buffer.py  # Test replay buffer
python train.py  # Full training
```

## Implementation Notes

### Fully Implemented Projects
- **Production-ready code**: Can be run immediately
- **Complete documentation**: README with usage instructions
- **Working examples**: All components tested and functional

### Placeholder Projects
- **Starter files**: Basic structure provided
- **Refer to guides**: Complete code in team guides
- **Copy and adapt**: Students should implement from guides

## Team Assignments

| Team | Project | Guide |
|------|---------|-------|
| BrainStormers | Dynamic Pricing | [Guide](../team_guides/BrainStormers_Dynamic_Pricing_Guide.md) |
| Model Minds | Deep Q-Network | [Guide](../team_guides/ModelMinds_DeepQNetwork_Guide.md) |
| PhuRaTan | Ultimate Tic-Tac-Toe | [Guide](../team_guides/PhuRaTan_UltimateTicTacToe_Guide.md) |
| Q-Team | Snake RL | [Guide](../team_guides/QTeam_Snake_Guide.md) |
| The Crown | Q-Learning | [Guide](../team_guides/TheCrown_QLearning_Guide.md) |
| The Innovators | Flappy Bird | [Guide](../team_guides/TheInnovators_FlappyBird_Guide.md) |
| NeuraWorks & BLANK | AR-GRPO | [Guide](../team_guides/NeuraWorks_BLANK_ARGRPO_Guide.md) |

## Support

For help:
1. Check your team's guide for detailed instructions
2. Review the [MASTER_GUIDE_INDEX.md](../MASTER_GUIDE_INDEX.md)
3. Attend office hours (2-5 PM daily)
4. Post questions on VLE forum

## License

Educational use only - MLA202 Course Materials
