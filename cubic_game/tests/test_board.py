"""
Unit tests for game board
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game.board import Board
from config import PLAYER_HUMAN, PLAYER_AI


class TestBoard(unittest.TestCase):
    """Test board functionality"""

    def setUp(self):
        self.board = Board()

    def test_initialization(self):
        """Test board initializes correctly"""
        self.assertEqual(self.board.grid.shape, (4, 4, 4))
        self.assertTrue((self.board.grid == 0).all())

    def test_valid_move(self):
        """Test making valid moves"""
        self.assertTrue(self.board.make_move(0, 0, 0, PLAYER_HUMAN))
        self.assertEqual(self.board.get_cell(0, 0, 0), PLAYER_HUMAN)

    def test_invalid_move(self):
        """Test invalid moves are rejected"""
        self.board.make_move(0, 0, 0, PLAYER_HUMAN)
        self.assertFalse(self.board.make_move(0, 0, 0, PLAYER_AI))

    def test_undo_move(self):
        """Test move undo"""
        self.board.make_move(1, 1, 1, PLAYER_AI)
        self.board.undo_move(1, 1, 1)
        self.assertEqual(self.board.get_cell(1, 1, 1), 0)

    def test_get_valid_moves(self):
        """Test getting valid moves"""
        self.assertEqual(len(self.board.get_valid_moves()), 64)
        self.board.make_move(0, 0, 0, PLAYER_HUMAN)
        self.assertEqual(len(self.board.get_valid_moves()), 63)


if __name__ == '__main__':
    unittest.main()