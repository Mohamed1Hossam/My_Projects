"""
Move ordering strategies for better alpha-beta pruning
"""

from typing import List, Tuple
import numpy as np
from game.board import Board
from utils.helpers import get_center_distance, count_adjacent_pieces


class MoveOrderer:
    """
    Orders moves to improve alpha-beta pruning efficiency
    """

    def __init__(self, heuristic_evaluator):
        self.evaluator = heuristic_evaluator

    def order_moves(self, board: Board, moves: List[Tuple[int, int, int]],
                    player: int) -> List[Tuple[int, int, int]]:
        """
        Order moves from most to least promising

        Args:
            board: Current board state
            moves: List of valid moves
            player: Current player

        Returns:
            Ordered list of moves
        """
        if len(moves) <= 1:
            return moves

        # Score each move
        move_scores = []
        for move in moves:
            score = self._score_move(board, move, player)
            move_scores.append((move, score))

        # Sort by score (descending for maximizing, ascending for minimizing)
        move_scores.sort(key=lambda x: x[1], reverse=(player == 2))

        return [move for move, score in move_scores]

    def _score_move(self, board: Board, move: Tuple[int, int, int],
                    player: int) -> float:
        """
        Score a move for ordering purposes

        Args:
            board: Current board state
            move: Move to score
            player: Current player

        Returns:
            Score for the move
        """
        x, y, z = move
        score = 0.0

        # 1. Center position bonus (positions closer to center are better)
        center_dist = get_center_distance(x, y, z)
        score -= center_dist * 2  # Lower distance = higher score

        # 2. Adjacent pieces bonus (positions near existing pieces are better)
        adjacent = count_adjacent_pieces(board.grid, x, y, z)
        score += adjacent * 5

        # 3. Quick evaluation of resulting position
        board.make_move(x, y, z, player)
        position_score = self.evaluator.evaluate_board(board)
        board.undo_move(x, y, z)

        score += position_score * 0.1

        return score