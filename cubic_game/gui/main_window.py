"""
Main game window
"""

import tkinter as tk
from tkinter import messagebox
import threading
import time
from typing import Optional

from game.board import Board
from game.rules import GameRules
from ai.heuristics import HeuristicEvaluator
from ai.alpha_beta import AlphaBetaPruning
from gui.controls import ControlPanel
from gui.board_display import BoardDisplay
from gui.info_panel import InfoPanel
from gui.styles import StyleManager
from config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, PLAYER_HUMAN, PLAYER_AI,
    DEFAULT_MAX_DEPTH
)


class MainWindow:
    """Main application window"""

    def __init__(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Intelligent Cubic Player - 4x4x4 Tic-Tac-Toe")

        # Make window fullscreen
        self.root.attributes('-fullscreen', True)
        self.root.state('zoomed')

        # Game components
        self.board = Board()
        self.rules = GameRules()
        self.evaluator = HeuristicEvaluator(self.rules)
        self.ai = AlphaBetaPruning(self.rules, self.evaluator, DEFAULT_MAX_DEPTH)

        # Game state
        self.current_player = PLAYER_HUMAN
        self.game_over = False
        self.ai_thinking = False
        self.move_count = 0

        # Setup GUI
        self._setup_gui()

        # Read player name from control panel entry (inline)
        self.player_name = self.control_panel.get_player_name()

        # Print welcome message
        self._print_welcome()

        # Update initial status with player name
        self.info_panel.update_status(f"{self.player_name} turn", StyleManager.COLORS['player'])

    def _setup_gui(self):
        """Setup all GUI components"""
        # Control panel at top
        # Control panel with name change callback
        self.control_panel = ControlPanel(self.root, {
            'new_game': self._on_new_game,
            'exit': self._on_exit,
            'name_change': self._on_name_change
        })
        self.control_panel.pack(side=tk.TOP, fill=tk.X)

        # Info panel
        self.info_panel = InfoPanel(self.root)
        self.info_panel.pack(side=tk.TOP, fill=tk.X)

        # Board display in center
        self.board_display = BoardDisplay(self.root, self._on_cell_click)
        self.board_display.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

    def _print_welcome(self):
        """Print welcome message to console"""
        print("\n" + "=" * 70)
        print("INTELLIGENT CUBIC PLAYER - 4x4x4 TIC-TAC-TOE")
        print("=" * 70)
        print("\nProject Features:")
        print("  * Minimax Algorithm with Alpha-Beta Pruning")
        print("  * Advanced Heuristic Evaluation (76 winning lines)")
        print("  * Transposition Table for Position Caching")
        print("  * Move Ordering for Better Pruning")
        print("  * Adaptive Search Depth")
        print("  * User-Friendly 3D Visualization")
        print("=" * 70 + "\n")

    def _on_cell_click(self, x: int, y: int, z: int):
        """Handle cell click"""
        if self.game_over or self.ai_thinking or self.current_player != PLAYER_HUMAN:
            return

        # Try to make move
        if self.board.make_move(x, y, z, PLAYER_HUMAN):
            self.move_count += 1
            self.info_panel.update_move_count(self.move_count)
            # Update the specific cell across displayed layers
            self.board_display.update_cell(x, y, z, PLAYER_HUMAN, False)

            # Check if game over
            winner = self.rules.check_winner(self.board)
            if winner is not None:
                self._handle_game_over(winner)
                return

            # Switch to AI turn
            self.current_player = PLAYER_AI
            self.info_panel.update_status("AI is thinking...",
                                          StyleManager.COLORS['ai'])
            self.ai_thinking = True
            self.board_display.set_all_cells_state(False)

            # Run AI in separate thread
            threading.Thread(target=self._ai_make_move, daemon=True).start()

    def _ai_make_move(self):
        """AI makes a move (runs in separate thread)"""
        time.sleep(0.3)  # Brief pause for better UX

        # Get best move from AI
        move = self.ai.get_best_move(self.board)

        if move:
            x, y, z = move
            self.board.make_move(x, y, z, PLAYER_AI)
            self.move_count += 1

            # Update GUI in main thread
            self.root.after(0, lambda: self._after_ai_move(x, y, z))

    def _after_ai_move(self, x: int, y: int, z: int):
        """Update GUI after AI move"""
        self.info_panel.update_move_count(self.move_count)
        self.info_panel.update_ai_time(self.ai.search_time)
        # Update all visible layers
        self.board_display.refresh_all_cells(self.board, True)

        # Check if game over
        winner = self.rules.check_winner(self.board)
        if winner is not None:
            self._handle_game_over(winner)
            return

        # Switch back to human
        self.current_player = PLAYER_HUMAN
        self.ai_thinking = False
        self.info_panel.update_status(f"{self.player_name} turn",
                                      StyleManager.COLORS['player'])
        self.board_display.set_all_cells_state(True)

    def _handle_game_over(self, winner: int):
        """Handle game over"""
        self.game_over = True
        self.board_display.set_all_cells_state(False)

        if winner == PLAYER_HUMAN:
            message = "*** Congratulations! You won! ***"
            self.info_panel.update_status("You Won!",
                                          StyleManager.COLORS['success'])
        elif winner == PLAYER_AI:
            message = "### AI wins! Better luck next time! ###"
            self.info_panel.update_status("AI Won!",
                                          StyleManager.COLORS['danger'])
        else:
            message = "=== It's a draw! Well played! ==="
            self.info_panel.update_status("Draw!",
                                          StyleManager.COLORS['neutral'])

        messagebox.showinfo("Game Over", message)

    def _on_name_change(self, name: str):
        """Handle player name change"""
        self.player_name = name
        if self.current_player == PLAYER_HUMAN and not self.game_over:
            self.info_panel.update_status(f"{self.player_name} turn",
                                      StyleManager.COLORS['player'])

    def _on_new_game(self):
        """Start new game"""
        # Reset game state
        self.board.reset()
        self.ai.clear_cache()
        self.current_player = PLAYER_HUMAN
        self.game_over = False
        self.ai_thinking = False
        self.move_count = 0
        # Reset GUI
        self.info_panel.update_status(f"{self.player_name} turn",
                                      StyleManager.COLORS['player'])
        self.info_panel.update_move_count(0)
        self.info_panel.update_ai_time(0)
        self.board_display.refresh_all_cells(self.board, True)

        print("\n" + "=" * 70)
        print("NEW GAME STARTED")
        print("=" * 70 + "\n")

    # Player name is entered in the control panel entry; dialog removed.

    def _on_exit(self):
        """Exit application"""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.root.quit()

    def run(self):
        """Run the application"""
        self.root.mainloop()