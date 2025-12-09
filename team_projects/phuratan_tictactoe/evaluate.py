"""
Evaluation Script for Ultimate Tic-Tac-Toe
"""

from game import UltimateTicTacToe
from agent import QLearningAgent, RandomAgent

def evaluate(agent_path='ultimate_ttt_agent.pkl', games=100):
    """Evaluate trained agent against random player"""
    print(f"Evaluating agent against random player ({games} games)...")
    print("-" * 60)

    game = UltimateTicTacToe()
    agent = QLearningAgent()
    agent.load(agent_path)
    random_agent = RandomAgent()

    wins = 0
    draws = 0
    losses = 0

    for i in range(games):
        state = game.reset()
        done = False

        while not done:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            # Agent plays as player 1, random as player 2
            if game.current_player == 1:
                action = agent.get_action(state, legal_moves, training=False)
            else:
                action = random_agent.get_action(state, legal_moves)

            state, reward, done = game.make_move(*action)

        if game.winner == 1:
            wins += 1
        elif game.winner == 0:
            draws += 1
        else:
            losses += 1

    print(f"Results:")
    print(f"  Wins: {wins} ({wins/games*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/games*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/games*100:.1f}%)")
    print("-" * 60)

if __name__ == "__main__":
    evaluate(games=100)
