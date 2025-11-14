"""
Control panel with game controls
"""

import tkinter as tk
from typing import Dict, Callable, Union, Any
from gui.styles import StyleManager

class ControlPanel:
    """Game control panel"""

    def __init__(self, parent: Union[tk.Tk, tk.Frame], callbacks: Dict[str, Callable[..., None]]):
        """
        Args:
            parent: Parent widget
            callbacks: Dict of callback functions (new_game, exit, name_change)
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

        # Player name entry (inline in control panel)
        name_frame = tk.Frame(self.frame, bg=StyleManager.COLORS['bg_dark'])
        name_frame.pack(pady=(0, 8))

        tk.Label(
            name_frame,
            text="Player Name:",
            font=StyleManager.FONT_NORMAL,
            bg=StyleManager.COLORS['bg_dark'],
            fg=StyleManager.COLORS['white']
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.name_var = tk.StringVar(value="Player")
        # Create a container frame for entry and button
        entry_container = tk.Frame(name_frame, bg=StyleManager.COLORS['bg_dark'])
        entry_container.pack(side=tk.LEFT)

        # Name entry with adjusted height
        name_entry = tk.Entry(
            entry_container,
            textvariable=self.name_var,
            font=StyleManager.FONT_NORMAL,
            width=16
        )
        name_entry.pack(side=tk.LEFT, ipady=2)  # Add internal padding to match button height

        # Apply name button with matching height
        apply_btn = tk.Button(
            entry_container,
            text="Enter",
            command=self._on_name_change,
            font=StyleManager.FONT_NORMAL
        )
        StyleManager.configure_button(apply_btn, 'primary')
        apply_btn.pack(side=tk.LEFT, padx=(5, 0), ipady=2)  # Match the entry height

        # Bind Enter key to both entry and button
        name_entry.bind('<Return>', lambda e: self._on_name_change())

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

    def pack(self, **kwargs: Any) -> None:
        """Pack the frame"""
        self.frame.pack(**kwargs)

    def _on_name_change(self):
        """Handle name change button click"""
        if 'name_change' in self.callbacks:
            self.callbacks['name_change'](self.get_player_name())

    def get_player_name(self) -> str:
        """Return the current player name from the entry"""
        return self.name_var.get().strip() or "Player"


    def _on_new_game(self):
        """Handle new game button"""
        if 'new_game' in self.callbacks:
            self.callbacks['new_game']()

    def _on_exit(self):
        """Handle exit button"""
        if 'exit' in self.callbacks:
            self.callbacks['exit']()