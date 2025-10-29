"""
Heuristic evaluation functions for board positions
"""

from typing import List, Tuple
from game.board import Board
from game.rules import GameRules
from config import (
    WEIGHT_WIN, WEIGHT_THREE_IN_ROW, WEIGHT_TWO_IN_ROW,
    WEIGHT_ONE_IN_ROW, PLAYER_AI, PLAYER_HUMAN, EMPTY_CELL
)


class HeuristicEvaluator:
    """
    Evaluates board positions using heuristic functions
    """

    def __init__(self, rules: GameRules):
        self.rules = rules

    def evaluate_board(self, board: Board, winner: int = None) -> int:
        """
        Evaluate the current board state

        Args:
            board: Current board state
            winner: Winner if game is over (optional)

        Returns:
            Evaluation score (positive favors AI, negative favors player)
        """
        # Terminal state evaluation
        if winner == PLAYER_AI:
            return WEIGHT_WIN
        elif winner == PLAYER_HUMAN:
            return -WEIGHT_WIN
        elif winner == 0:  # Draw
            return 0

        # Non-terminal state: evaluate all lines
        total_score = 0

        for line in self.rules.winning_lines:
            line_score = self._evaluate_line(board, line)
            total_score += line_score

        return total_score

    def _evaluate_line(self, board: Board, line: List[Tuple[int, int, int]]) -> int:
        """
        Evaluate a single line

        Args:
            board: Current board state
            line: List of 4 positions forming a line

        Returns:
            Score for this line
        """
        player_count, ai_count, empty_count = self.rules.get_line_value(board, line)

        # Line is blocked if both players have pieces
        if player_count > 0 and ai_count > 0:
            return 0

        # Evaluate AI potential
        if ai_count > 0:
            if ai_count == 4:
                return WEIGHT_WIN
            elif ai_count == 3:
                return WEIGHT_THREE_IN_ROW
            elif ai_count == 2:
                return WEIGHT_TWO_IN_ROW
            elif ai_count == 1:
                return WEIGHT_ONE_IN_ROW

        # Evaluate player threats
        if player_count > 0:
            if player_count == 4:
                return -WEIGHT_WIN
            elif player_count == 3:
                return -WEIGHT_THREE_IN_ROW
            elif player_count == 2:
                return -WEIGHT_TWO_IN_ROW
            elif player_count == 1:
                return -WEIGHT_ONE_IN_ROW

        return 0

    def find_threats(self, board: Board, player: int) -> List[Tuple[int, int, int]]:
        """
        Find positions where player can create threats

        Args:
            board: Current board state
            player: Player to find threats for

        Returns:
            List of threatening positions
        """
        threats = []

        for line in self.rules.winning_lines:
            player_count, ai_count, empty_count = self.rules.get_line_value(board, line)

            # Check if player has 3 in this line with 1 empty
            if player == PLAYER_AI:
                if ai_count == 3 and empty_count == 1 and player_count == 0:
                    # Find the empty position
                    for pos in line:
                        if board.get_cell(*pos) == EMPTY_CELL:
                            threats.append(pos)
            else:
                if player_count == 3 and empty_count == 1 and ai_count == 0:
                    for pos in line:
                        if board.get_cell(*pos) == EMPTY_CELL:
                            threats.append(pos)

        return threats