"""
Board display widget
"""

import tkinter as tk
from typing import Callable, Optional
from gui.styles import StyleManager
from config import BOARD_SIZE, BUTTON_WIDTH, BUTTON_HEIGHT


class BoardDisplay:
    """Displays the 4x4x4 game board layer by layer"""

    def __init__(self, parent, on_cell_click: Callable):
        """
        Args:
            parent: Parent widget
            on_cell_click: Callback for cell click (x, y, z)
        """
        self.on_cell_click = on_cell_click
        self.current_layer = 0

        # Main frame
        self.frame = tk.Frame(parent, bg=StyleManager.COLORS['bg_light'],
                              padx=20, pady=20)

        # Layer title
        self.layer_label = tk.Label(
            self.frame,
            text="Layer 0 (Z = 0)",
            font=StyleManager.FONT_HEADING,
            bg=StyleManager.COLORS['bg_light']
        )
        self.layer_label.grid(row=0, column=0, columnspan=BOARD_SIZE, pady=10)

        # Create button grid
        self.buttons = {}
        self._create_grid()

    def _create_grid(self):
        """Create the button grid for current layer"""
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                btn = tk.Button(
                    self.frame,
                    text="",
                    width=BUTTON_WIDTH,
                    height=BUTTON_HEIGHT,
                    command=lambda x=x, y=y: self._cell_clicked(x, y)
                )
                StyleManager.configure_button(btn, 'cell')
                btn.grid(row=x + 1, column=y, padx=2, pady=2)
                self.buttons[(x, y, self.current_layer)] = btn

    def _cell_clicked(self, x: int, y: int):
        """Handle cell click"""
        self.on_cell_click(x, y, self.current_layer)

    def pack(self, **kwargs):
        """Pack the frame"""
        self.frame.pack(**kwargs)

    def set_layer(self, layer: int):
        """Change displayed layer"""
        if 0 <= layer < BOARD_SIZE:
            self.current_layer = layer
            self.layer_label.config(text=f"Layer {layer} (Z = {layer})")
            self._refresh_grid()

    def _refresh_grid(self):
        """Refresh the button grid"""
        # Clear existing buttons
        for widget in self.frame.winfo_children():
            if widget != self.layer_label:
                widget.destroy()

        self.buttons.clear()
        self._create_grid()

    def update_cell(self, x: int, y: int, z: int, player: int, enabled: bool = True):
        """
        Update a cell's appearance

        Args:
            x, y, z: Position
            player: Player ID (0=empty, 1=human, 2=AI)
            enabled: Whether cell is clickable
        """
        if z != self.current_layer:
            return

        btn = self.buttons.get((x, y, z))
        if btn:
            if player == 0:
                btn.config(
                    text="",
                    bg=StyleManager.COLORS['white'],
                    state=tk.NORMAL if enabled else tk.DISABLED
                )
            else:
                color = StyleManager.get_player_color(player)
                btn.config(
                    text="●",
                    bg=color,
                    fg=StyleManager.COLORS['white'],
                    state=tk.DISABLED
                )

    def set_all_cells_state(self, enabled: bool):
        """Enable or disable all cells"""
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in self.buttons.values():
            if btn.cget('text') == "":  # Only affect empty cells
                btn.config(state=state)

    def refresh_all_cells(self, board, enabled: bool = True):
        """Refresh all cells from board state"""
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                player = board.get_cell(x, y, self.current_layer)
                self.update_cell(x, y, self.current_layer, player, enabled)