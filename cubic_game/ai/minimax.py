"""
Minimax algorithm implementation
"""

from typing import Tuple, Optional
from game.board import Board
from game.rules import GameRules
from ai.heuristics import HeuristicEvaluator
from config import PLAYER_AI, PLAYER_HUMAN


class MinimaxAlgorithm:
    """
    Basic Minimax algorithm without pruning
    (Used for comparison and testing)
    """

    def __init__(self, rules: GameRules, evaluator: HeuristicEvaluator, max_depth: int = 3):
        self.rules = rules
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.nodes_evaluated = 0

    def get_best_move(self, board: Board) -> Optional[Tuple[int, int, int]]:
        """
        Find the best move using Minimax algorithm

        Args:
            board: Current board state

        Returns:
            Best move as (x, y, z) tuple
        """
        self.nodes_evaluated = 0
        _, best_move = self._minimax(board, self.max_depth, True)
        return best_move

    def _minimax(self, board: Board, depth: int, maximizing: bool) -> Tuple[int, Optional[Tuple[int, int, int]]]:
        """
        Minimax recursive function

        Args:
            board: Current board state
            depth: Remaining search depth
            maximizing: True if maximizing player (AI), False if minimizing (human)

        Returns:
            Tuple of (score, best_move)
        """
        self.nodes_evaluated += 1

        # Check terminal condition
        winner = self.rules.check_winner(board)
        if winner is not None or depth == 0:
            score = self.evaluator.evaluate_board(board, winner)
            return score, None

        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return 0, None

        best_move = valid_moves[0]

        if maximizing:
            max_score = float('-inf')
            for move in valid_moves:
                x, y, z = move
                board.make_move(x, y, z, PLAYER_AI)
                score, _ = self._minimax(board, depth - 1, False)
                board.undo_move(x, y, z)

                if score > max_score:
                    max_score = score
                    best_move = move

            return max_score, best_move
        else:
            min_score = float('inf')
            for move in valid_moves:
                x, y, z = move
                board.make_move(x, y, z, PLAYER_HUMAN)
                score, _ = self._minimax(board, depth - 1, True)
                board.undo_move(x, y, z)

                if score < min_score:
                    min_score = score
                    best_move = move

            return min_score, best_move