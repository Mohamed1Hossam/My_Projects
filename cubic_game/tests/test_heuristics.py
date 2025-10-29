"""
Unit tests for heuristic evaluation
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game.board import Board
from game.rules import GameRules
from ai.heuristics import HeuristicEvaluator
from config import PLAYER_HUMAN, PLAYER_AI


class TestHeuristics(unittest.TestCase):
    """Test heuristic evaluation"""

    def setUp(self):
        self.board = Board()
        self.rules = GameRules()
        self.evaluator = HeuristicEvaluator(self.rules)

    def test_empty_board(self):
        """Empty board should evaluate to 0"""
        score = self.evaluator.evaluate_board(self.board)
        self.assertEqual(score, 0)

    def test_ai_advantage(self):
        """Board with AI pieces should have positive score"""
        self.board.make_move(0, 0, 0, PLAYER_AI)
        self.board.make_move(1, 0, 0, PLAYER_AI)
        score = self.evaluator.evaluate_board(self.board)
        self.assertGreater(score, 0)

    def test_player_advantage(self):
        """Board with player pieces should have negative score"""
        self.board.make_move(0, 0, 0, PLAYER_HUMAN)
        self.board.make_move(1, 0, 0, PLAYER_HUMAN)
        score = self.evaluator.evaluate_board(self.board)
        self.assertLess(score, 0)

    def test_win_detection(self):
        """Winning position should have maximum score"""
        for i in range(4):
            self.board.make_move(i, 0, 0, PLAYER_AI)
        score = self.evaluator.evaluate_board(self.board, PLAYER_AI)
        self.assertEqual(score, 10000)


if __name__ == '__main__':
    unittest.main()