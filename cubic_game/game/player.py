"""
Player classes
"""

from abc import ABC, abstractmethod

class Player(ABC):
    """Abstract base class for players"""

    def __init__(self, player_id, name):
        self.player_id = player_id
        self.name = name

    @abstractmethod
    def get_move(self, board):
        """Get next move from player"""
        pass


class HumanPlayer(Player):
    """Human player implementation"""

    def __init__(self, player_id):
        super().__init__(player_id, "Human")
        self.pending_move = None

    def get_move(self, board):
        """Human move is set externally via GUI"""
        move = self.pending_move
        self.pending_move = None
        return move

    def set_move(self, x, y, z):
        """Set the move chosen by human via GUI"""
        self.pending_move = (x, y, z)


class AIPlayerInterface(Player):
    """Interface for AI player"""

    def __init__(self, player_id):
        super().__init__(player_id, "AI")
        self.ai_engine = None

    def set_ai_engine(self, engine):
        """Set the AI engine"""
        self.ai_engine = engine

    def get_move(self, board):
        """Get move from AI engine"""
        if self.ai_engine:
            return self.ai_engine.get_best_move(board)
        return None