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
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)

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

        # Print welcome message
        self._print_welcome()

    def _setup_gui(self):
        """Setup all GUI components"""
        # Control panel at top
        self.control_panel = ControlPanel(self.root, {
            'layer_change': self._on_layer_change,
            'new_game': self._on_new_game,
            'exit': self._on_exit
        })
        self.control_panel.pack(side=tk.TOP, fill=tk.X)

        # Info panel
        self.info_panel = InfoPanel(self.root)
        self.info_panel.pack(side=tk.TOP, fill=tk.X)

        # Board display in center
        self.board_display = BoardDisplay(self.root, self._on_cell_click)
        self.board_display.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        # Instructions at bottom
        self._create_instructions()

    def _create_instructions(self):
        """Create instruction panel"""
        inst_frame = tk.Frame(self.root, bg=StyleManager.COLORS['bg_medium'],
                              padx=10, pady=5)
        inst_frame.pack(side=tk.BOTTOM, fill=tk.X)

        instructions = (
            "HOW TO PLAY: Click on any empty cell to place your piece (Blue ●). "
            "Switch layers using the radio buttons. "
            "Get 4 in a row in any direction to win!"
        )

        tk.Label(
            inst_frame,
            text=instructions,
            font=StyleManager.FONT_NORMAL,
            bg=StyleManager.COLORS['bg_medium'],
            fg=StyleManager.COLORS['white'],
            wraplength=WINDOW_WIDTH - 40
        ).pack()

    def _print_welcome(self):
        """Print welcome message to console"""
        print("\n" + "=" * 70)
        print("INTELLIGENT CUBIC PLAYER - 4x4x4 TIC-TAC-TOE")
        print("=" * 70)
        print("\nProject Features:")
        print("  ✓ Minimax Algorithm with Alpha-Beta Pruning")
        print("  ✓ Advanced Heuristic Evaluation (76 winning lines)")
        print("  ✓ Transposition Table for Position Caching")
        print("  ✓ Move Ordering for Better Pruning")
        print("  ✓ Adaptive Search Depth")
        print("  ✓ User-Friendly 3D Visualization")
        print("\nGame Rules:")
        print("  - Players alternate placing pieces on a 4x4x4 grid")
        print("  - First to get 4 in a row (any direction) wins")
        print("  - 76 possible winning lines")
        print("=" * 70 + "\n")

    def _on_cell_click(self, x: int, y: int, z: int):
        """Handle cell click"""
        if self.game_over or self.ai_thinking or self.current_player != PLAYER_HUMAN:
            return

        # Try to make move
        if self.board.make_move(x, y, z, PLAYER_HUMAN):
            self.move_count += 1
            self.info_panel.update_move_count(self.move_count)
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

        # Update all cells on current layer
        self.board_display.refresh_all_cells(self.board, True)

        # Check if game over
        winner = self.rules.check_winner(self.board)
        if winner is not None:
            self._handle_game_over(winner)
            return

        # Switch back to human
        self.current_player = PLAYER_HUMAN
        self.ai_thinking = False
        self.info_panel.update_status("Your Turn!",
                                      StyleManager.COLORS['player'])
        self.board_display.set_all_cells_state(True)

    def _handle_game_over(self, winner: int):
        """Handle game over"""
        self.game_over = True
        self.board_display.set_all_cells_state(False)

        if winner == PLAYER_HUMAN:
            message = "🎉 Congratulations! You won! 🎉"
            self.info_panel.update_status("You Won!",
                                          StyleManager.COLORS['success'])
        elif winner == PLAYER_AI:
            message = "🤖 AI wins! Better luck next time! 🤖"
            self.info_panel.update_status("AI Won!",
                                          StyleManager.COLORS['danger'])
        else:
            message = "🤝 It's a draw! Well played! 🤝"
            self.info_panel.update_status("Draw!",
                                          StyleManager.COLORS['neutral'])

        messagebox.showinfo("Game Over", message)

    def _on_layer_change(self, layer: int):
        """Handle layer change"""
        self.board_display.set_layer(layer)
        self.board_display.refresh_all_cells(self.board, not self.ai_thinking and not self.game_over)

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
        self.info_panel.update_status("Your Turn!",
                                      StyleManager.COLORS['player'])
        self.info_panel.update_move_count(0)
        self.info_panel.update_ai_time(0)
        self.board_display.set_layer(0)
        self.board_display.refresh_all_cells(self.board, True)

        print("\n" + "=" * 70)
        print("NEW GAME STARTED")
        print("=" * 70 + "\n")

    def _on_exit(self):
        """Exit application"""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.root.quit()

    def run(self):
        """Run the application"""
        self.root.mainloop()