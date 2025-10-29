"""
Constants for the Cubic game
"""

# Direction vectors for checking winning lines
DIRECTIONS = {
    'rows': [(1, 0, 0)],
    'columns': [(0, 1, 0)],
    'pillars': [(0, 0, 1)],
    'xy_diagonals': [(1, 1, 0), (1, -1, 0)],
    'xz_diagonals': [(1, 0, 1), (1, 0, -1)],
    'yz_diagonals': [(0, 1, 1), (0, 1, -1)],
    'space_diagonals': [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)]
}

# All possible starting points for winning lines
WINNING_LINE_STARTS = {
    'rows': [(0, y, z) for y in range(4) for z in range(4)],
    'columns': [(x, 0, z) for x in range(4) for z in range(4)],
    'pillars': [(x, y, 0) for x in range(4) for y in range(4)],
    'xy_diagonals': [(0, 0, z) for z in range(4)] + [(0, 3, z) for z in range(4)],
    'xz_diagonals': [(0, y, 0) for y in range(4)] + [(0, y, 3) for y in range(4)],
    'yz_diagonals': [(x, 0, 0) for x in range(4)] + [(x, 0, 3) for x in range(4)],
    'space_diagonals': [(0, 0, 0), (0, 0, 3), (0, 3, 0), (3, 0, 0)]
}