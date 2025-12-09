"""
Ultimate Tic-Tac-Toe Game Implementation

A 9x9 grid of 9 individual 3x3 tic-tac-toe boards.
Win a small board to claim it in the meta-board.
Win the meta-board to win the game!
"""

import numpy as np

class UltimateTicTacToe:
    def __init__(self):
        # 9 boards, each with 9 cells
        # 0 = empty, 1 = player 1, 2 = player 2
        self.board = np.zeros((9, 9), dtype=int)

        # Meta-board: which player won each of the 9 boards
        self.meta_board = np.zeros((3, 3), dtype=int)

        self.current_player = 1
        self.active_board = None  # Which board must be played in (None = any)
        self.game_over = False
        self.winner = 0

    def reset(self):
        """Reset game to initial state"""
        self.board = np.zeros((9, 9), dtype=int)
        self.meta_board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.active_board = None
        self.game_over = False
        self.winner = 0
        return self.get_state()

    def get_state(self):
        """Return state tuple"""
        return (
            tuple(self.board.flatten()),
            tuple(self.meta_board.flatten()),
            self.active_board,
            self.current_player
        )

    def get_legal_moves(self):
        """Return list of legal (board_idx, cell_idx) moves"""
        if self.game_over:
            return []

        moves = []
        if self.active_board is None:
            # Can play in any board that's not won
            for board_idx in range(9):
                if self.meta_board[board_idx // 3, board_idx % 3] == 0:
                    for cell_idx in range(9):
                        if self.board[board_idx, cell_idx] == 0:
                            moves.append((board_idx, cell_idx))
        else:
            # Must play in active board
            board_idx = self.active_board
            if self.meta_board[board_idx // 3, board_idx % 3] == 0:
                for cell_idx in range(9):
                    if self.board[board_idx, cell_idx] == 0:
                        moves.append((board_idx, cell_idx))
            else:
                # Active board is won, can play anywhere
                return self.get_legal_moves_any()

        return moves

    def get_legal_moves_any(self):
        """Get moves from any non-won board"""
        moves = []
        for board_idx in range(9):
            if self.meta_board[board_idx // 3, board_idx % 3] == 0:
                for cell_idx in range(9):
                    if self.board[board_idx, cell_idx] == 0:
                        moves.append((board_idx, cell_idx))
        return moves

    def make_move(self, board_idx, cell_idx):
        """Make a move and return (next_state, reward, done)"""
        if self.game_over:
            return self.get_state(), 0, True

        if (board_idx, cell_idx) not in self.get_legal_moves():
            # Illegal move - penalty
            return self.get_state(), -10, True

        # Place piece
        self.board[board_idx, cell_idx] = self.current_player

        # Check if local board is won
        local_board = self.board[board_idx].reshape(3, 3)
        local_winner = self.check_winner(local_board)
        if local_winner:
            self.meta_board[board_idx // 3, board_idx % 3] = local_winner

        # Check if game is won
        game_winner = self.check_winner(self.meta_board)
        if game_winner:
            self.game_over = True
            self.winner = game_winner
            reward = 1 if game_winner == self.current_player else -1
            return self.get_state(), reward, True

        # Check for draw
        if np.all(self.meta_board != 0) or len(self.get_legal_moves_any()) == 0:
            self.game_over = True
            return self.get_state(), 0, True

        # Set next active board
        self.active_board = cell_idx if self.meta_board[cell_idx // 3, cell_idx % 3] == 0 else None

        # Switch player
        self.current_player = 3 - self.current_player

        return self.get_state(), 0, False

    def check_winner(self, board):
        """Check 3x3 board for winner"""
        # Rows
        for i in range(3):
            if board[i, 0] == board[i, 1] == board[i, 2] != 0:
                return board[i, 0]
        # Columns
        for j in range(3):
            if board[0, j] == board[1, j] == board[2, j] != 0:
                return board[0, j]
        # Diagonals
        if board[0, 0] == board[1, 1] == board[2, 2] != 0:
            return board[0, 0]
        if board[0, 2] == board[1, 1] == board[2, 0] != 0:
            return board[0, 2]
        return 0

    def render(self):
        """Print board"""
        print("\nMeta Board:")
        print(self.meta_board)
        print("\nFull Board (9x9):")
        for i in range(9):
            print(self.board[i])
        print(f"Current Player: {self.current_player}")
        print(f"Active Board: {self.active_board}")
