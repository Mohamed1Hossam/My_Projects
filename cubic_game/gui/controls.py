"""
Control panel with game controls
"""

import tkinter as tk
from tkinter import ttk
from gui.styles import StyleManager


class ControlPanel:
    """Game control panel"""

    def __init__(self, parent, callbacks):
        """
        Args:
            parent: Parent widget
            callbacks: Dict of callback functions
        """
        self.callbacks = callbacks

        # Main frame
        self.frame = tk.Frame(parent, bg=StyleManager.COLORS['bg_dark'],
                              padx=10, pady=10)

        # Title
        title = tk.Label(
            self.frame,
            text="Intelligent Cubic Player",
            font=StyleManager.FONT_TITLE,
            bg=StyleManager.COLORS['bg_dark'],
            fg=StyleManager.COLORS['white']
        )
        title.pack(pady=5)

        # Layer selection frame
        layer_frame = tk.Frame(self.frame, bg=StyleManager.COLORS['bg_medium'],
                               padx=10, pady=10)
        layer_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            layer_frame,
            text="Select Layer (Z-axis):",
            font=StyleManager.FONT_NORMAL,
            bg=StyleManager.COLORS['bg_medium'],
            fg=StyleManager.COLORS['white']
        ).pack(side=tk.LEFT, padx=5)

        self.layer_var = tk.IntVar(value=0)
        for i in range(4):
            rb = tk.Radiobutton(
                layer_frame,
                text=f"Layer {i}",
                variable=self.layer_var,
                value=i,
                command=self._on_layer_change,
                font=StyleManager.FONT_NORMAL,
                bg=StyleManager.COLORS['bg_medium'],
                fg=StyleManager.COLORS['white'],
                selectcolor=StyleManager.COLORS['bg_dark']
            )
            rb.pack(side=tk.LEFT, padx=5)

        # Button frame
        button_frame = tk.Frame(self.frame, bg=StyleManager.COLORS['bg_dark'])
        button_frame.pack(pady=10)

        # New Game button
        new_game_btn = tk.Button(
            button_frame,
            text="New Game",
            command=self._on_new_game,
            padx=20,
            pady=5
        )
        StyleManager.configure_button(new_game_btn, 'primary')
        new_game_btn.pack(side=tk.LEFT, padx=5)

        # Exit button
        exit_btn = tk.Button(
            button_frame,
            text="Exit",
            command=self._on_exit,
            padx=20,
            pady=5
        )
        StyleManager.configure_button(exit_btn, 'danger')
        exit_btn.pack(side=tk.LEFT, padx=5)

    def pack(self, **kwargs):
        """Pack the frame"""
        self.frame.pack(**kwargs)

    def get_current_layer(self) -> int:
        """Get currently selected layer"""
        return self.layer_var.get()

    def _on_layer_change(self):
        """Handle layer change"""
        if 'layer_change' in self.callbacks:
            self.callbacks['layer_change'](self.layer_var.get())

    def _on_new_game(self):
        """Handle new game button"""
        if 'new_game' in self.callbacks:
            self.callbacks['new_game']()

    def _on_exit(self):
        """Handle exit button"""
        if 'exit' in self.callbacks:
            self.callbacks['exit']()