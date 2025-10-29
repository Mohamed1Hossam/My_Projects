"""
Helper functions for the Cubic game
"""

import numpy as np
from typing import Tuple, List
from config import BOARD_SIZE


def is_valid_position(x: int, y: int, z: int) -> bool:
    """Check if position is within board bounds"""
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE and 0 <= z < BOARD_SIZE


def get_center_distance(x: int, y: int, z: int) -> float:
    """Calculate distance from center of the board"""
    center = (BOARD_SIZE - 1) / 2
    return abs(x - center) + abs(y - center) + abs(z - center)


def count_adjacent_pieces(board: np.ndarray, x: int, y: int, z: int) -> int:
    """Count number of adjacent pieces (non-empty cells)"""
    count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                if is_valid_position(nx, ny, nz) and board[nx, ny, nz] != 0:
                    count += 1
    return count


def format_time(seconds: float) -> str:
    """Format time in seconds to readable string"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def board_to_hash(board: np.ndarray) -> int:
    """Convert board state to hash for transposition table"""
    return hash(board.tobytes())