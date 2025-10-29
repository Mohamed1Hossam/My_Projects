"""
Alpha-Beta Pruning implementation
"""

from typing import Tuple, Optional, Dict
from game.board import Board
from game.rules import GameRules
from ai.heuristics import HeuristicEvaluator
from ai.move_ordering import MoveOrderer
from utils.helpers import board_to_hash
from config import (
    PLAYER_AI, PLAYER_HUMAN, USE_TRANSPOSITION_TABLE,
    USE_MOVE_ORDERING, USE_ADAPTIVE_DEPTH,
    EARLY_GAME_DEPTH, MID_GAME_DEPTH, LATE_GAME_DEPTH,
    EARLY_GAME_THRESHOLD, MID_GAME_THRESHOLD
)
import time


class AlphaBetaPruning:
    """
    Minimax with Alpha-Beta Pruning optimization
    """

    def __init__(self, rules: GameRules, evaluator: HeuristicEvaluator, max_depth: int = 3):
        self.rules = rules
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.move_orderer = MoveOrderer(evaluator)

        # Statistics
        self.nodes_evaluated = 0
        self.pruning_count = 0
        self.search_time = 0.0

        # Transposition table: board_hash -> (depth, score, best_move)
        self.transposition_table: Dict[int, Tuple[int, int, Optional[Tuple[int, int, int]]]] = {}

    def get_best_move(self, board: Board) -> Optional[Tuple[int, int, int]]:
        """
        Find the best move using Alpha-Beta Pruning

        Args:
            board: Current board state

        Returns:
            Best move as (x, y, z) tuple
        """
        # Reset statistics
        self.nodes_evaluated = 0
        self.pruning_count = 0
        start_time = time.time()

        # Adaptive depth
        depth = self._get_adaptive_depth(board)

        # Run alpha-beta search
        _, best_move = self._alpha_beta(
            board, depth, float('-inf'), float('inf'), True
        )

        self.search_time = time.time() - start_time

        # Print statistics
        self._print_statistics(depth)

        return best_move

    def _get_adaptive_depth(self, board: Board) -> int:
        """Determine search depth based on game state"""
        if not USE_ADAPTIVE_DEPTH:
            return self.max_depth

        moves_remaining = len(board.get_valid_moves())

        if moves_remaining > EARLY_GAME_THRESHOLD:
            return EARLY_GAME_DEPTH
        elif moves_remaining > MID_GAME_THRESHOLD:
            return MID_GAME_DEPTH
        else:
            return LATE_GAME_DEPTH

    def _alpha_beta(self, board: Board, depth: int, alpha: float, beta: float,
                    maximizing: bool) -> Tuple[float, Optional[Tuple[int, int, int]]]:
        """
        Alpha-Beta pruning recursive function

        Args:
            board: Current board state
            depth: Remaining search depth
            alpha: Alpha value (best for maximizer)
            beta: Beta value (best for minimizer)
            maximizing: True if maximizing player

        Returns:
            Tuple of (score, best_move)
        """
        self.nodes_evaluated += 1

        # Check transposition table
        if USE_TRANSPOSITION_TABLE:
            board_hash = board_to_hash(board.grid)
            if board_hash in self.transposition_table:
                cached_depth, cached_score, cached_move = self.transposition_table[board_hash]
                if cached_depth >= depth:
                    return cached_score, cached_move

        # Terminal condition
        winner = self.rules.check_winner(board)
        if winner is not None or depth == 0:
            score = self.evaluator.evaluate_board(board, winner)
            return score, None

        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return 0, None

        # Move ordering
        if USE_MOVE_ORDERING:
            player = PLAYER_AI if maximizing else PLAYER_HUMAN
            valid_moves = self.move_orderer.order_moves(board, valid_moves, player)

        best_move = valid_moves[0]

        if maximizing:
            max_score = float('-inf')

            for move in valid_moves:
                x, y, z = move
                board.make_move(x, y, z, PLAYER_AI)
                score, _ = self._alpha_beta(board, depth - 1, alpha, beta, False)
                board.undo_move(x, y, z)

                if score > max_score:
                    max_score = score
                    best_move = move

                alpha = max(alpha, score)

                # Beta cutoff (pruning)
                if beta <= alpha:
                    self.pruning_count += 1
                    break

            # Store in transposition table
            if USE_TRANSPOSITION_TABLE:
                self.transposition_table[board_hash] = (depth, max_score, best_move)

            return max_score, best_move
        else:
            min_score = float('inf')

            for move in valid_moves:
                x, y, z = move
                board.make_move(x, y, z, PLAYER_HUMAN)
                score, _ = self._alpha_beta(board, depth - 1, alpha, beta, True)
                board.undo_move(x, y, z)

                if score < min_score:
                    min_score = score
                    best_move = move

                beta = min(beta, score)

                # Alpha cutoff (pruning)
                if beta <= alpha:
                    self.pruning_count += 1
                    break

            # Store in transposition table
            if USE_TRANSPOSITION_TABLE:
                self.transposition_table[board_hash] = (depth, min_score, best_move)

            return min_score, best_move

    def _print_statistics(self, depth: int):
        """Print search statistics"""
        print(f"\n{'=' * 60}")
        print(f"AI Search Statistics:")
        print(f"  Depth: {depth}")
        print(f"  Nodes evaluated: {self.nodes_evaluated:,}")
        print(f"  Pruning cutoffs: {self.pruning_count:,}")
        print(f"  Search time: {self.search_time:.3f}s")
        print(f"  Nodes/second: {self.nodes_evaluated / self.search_time:,.0f}")
        print(f"{'=' * 60}\n")

    def clear_cache(self):
        """Clear transposition table"""
        self.transposition_table.clear()