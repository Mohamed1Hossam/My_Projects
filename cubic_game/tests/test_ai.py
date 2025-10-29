"""
Unit tests for AI components
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game.board import Board
from game.rules import GameRules
from ai.heuristics import HeuristicEvaluator
from ai.alpha_beta import AlphaBetaPruning
from config import PLAYER_HUMAN, PLAYER_AI


class TestAI(unittest.TestCase):
    """Test AI functionality"""

    def setUp(self):
        self.board = Board()
        self.rules = GameRules()
        self.evaluator = HeuristicEvaluator(self.rules)
        self.ai = AlphaBetaPruning(self.rules, self.evaluator, max_depth=2)

    def test_ai_makes_valid_move(self):
        """Test AI makes valid moves"""
        move = self.ai.get_best_move(self.board)
        self.assertIsNotNone(move)
        x, y, z = move
        self.assertTrue(self.board.is_valid_move(x, y, z))

    def test_ai_blocks_win(self):
        """Test AI blocks opponent's winning move"""
        # Set up player about to win
        self.board.make_move(0, 0, 0, PLAYER_HUMAN)
        self.board.make_move(1, 0, 0, PLAYER_HUMAN)
        self.board.make_move(2, 0, 0, PLAYER_HUMAN)

        # AI should block at (3, 0, 0)
        move = self.ai.get_best_move(self.board)
        self.assertEqual(move, (3, 0, 0))

    def test_ai_takes_win(self):
        """Test AI takes winning move"""
        # Set up AI about to win
        self.board.make_move(0, 1, 1, PLAYER_AI)
        self.board.make_move(1, 1, 1, PLAYER_AI)
        self.board.make_move(2, 1, 1, PLAYER_AI)

        # AI should win at (3, 1, 1)
        move = self.ai.get_best_move(self.board)
        self.assertEqual(move, (3, 1, 1))


if __name__ == '__main__':
    unittest.main()