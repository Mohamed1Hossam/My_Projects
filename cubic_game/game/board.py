"""
Game board representation and basic operations
"""

import numpy as np
from config import BOARD_SIZE, EMPTY_CELL

class Board:
    """
    Represents the 4x4x4 Cubic game board
    """

    def __init__(self):
        """Initialize empty board"""
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.move_history = []

    def reset(self):
        """Reset the board to empty state"""
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.move_history = []

    def make_move(self, x, y, z, player):
        """
        Make a move on the board

        Args:
            x, y, z: Position coordinates
            player: Player identifier (1 or 2)

        Returns:
            True if move was successful, False otherwise
        """
        if not self.is_valid_move(x, y, z):
            return False

        self.grid[x, y, z] = player
        self.move_history.append((x, y, z, player))
        return True

    def undo_move(self, x, y, z):
        """Undo a move at given position"""
        self.grid[x, y, z] = EMPTY_CELL
        if self.move_history and self.move_history[-1][:3] == (x, y, z):
            self.move_history.pop()

    def is_valid_move(self, x, y, z):
        """Check if move is valid"""
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and 0 <= z < BOARD_SIZE):
            return False
        return self.grid[x, y, z] == EMPTY_CELL

    def get_valid_moves(self):
        """Get all valid moves (empty positions)"""
        moves = []
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                for z in range(BOARD_SIZE):
                    if self.grid[x, y, z] == EMPTY_CELL:
                        moves.append((x, y, z))
        return moves

    def get_cell(self, x, y, z):
        """Get value at cell"""
        return self.grid[x, y, z]

    def is_full(self):
        """Check if board is full"""
        return len(self.get_valid_moves()) == 0

    def copy(self):
        """Create a copy of the board"""
        new_board = Board()
        new_board.grid = np.copy(self.grid)
        new_board.move_history = self.move_history.copy()
        return new_board

    def __str__(self):
        """String representation of board"""
        result = []
        for z in range(BOARD_SIZE):
            result.append(f"Layer {z}:")
            result.append(str(self.grid[:, :, z]))
        return '\n'.join(result)