"""
Information panel showing game status
"""

import tkinter as tk
from gui.styles import StyleManager


class InfoPanel:
    """Displays game information and status"""

    def __init__(self, parent):
        self.frame = tk.Frame(parent, bg=StyleManager.COLORS['bg_medium'],
                              padx=10, pady=10)

        # Status label
        self.status_label = tk.Label(
            self.frame,
            text="Game Ready - Your Turn!",
            font=StyleManager.FONT_HEADING,
            bg=StyleManager.COLORS['bg_medium'],
            fg=StyleManager.COLORS['player']
        )
        self.status_label.pack(pady=5)

        # Statistics frame
        stats_frame = tk.Frame(self.frame, bg=StyleManager.COLORS['bg_medium'])
        stats_frame.pack(pady=5)

        self.move_count_label = tk.Label(
            stats_frame,
            text="Moves: 0",
            font=StyleManager.FONT_NORMAL,
            bg=StyleManager.COLORS['bg_medium'],
            fg=StyleManager.COLORS['white']
        )
        self.move_count_label.pack(side=tk.LEFT, padx=10)

        self.ai_time_label = tk.Label(
            stats_frame,
            text="AI Time: -",
            font=StyleManager.FONT_NORMAL,
            bg=StyleManager.COLORS['bg_medium'],
            fg=StyleManager.COLORS['white']
        )
        self.ai_time_label.pack(side=tk.LEFT, padx=10)

    def pack(self, **kwargs):
        """Pack the frame"""
        self.frame.pack(**kwargs)

    def update_status(self, text: str, color: str = None):
        """Update status message"""
        self.status_label.config(text=text)
        if color:
            self.status_label.config(fg=color)

    def update_move_count(self, count: int):
        """Update move counter"""
        self.move_count_label.config(text=f"Moves: {count}")

    def update_ai_time(self, time: float):
        """Update AI thinking time"""
        self.ai_time_label.config(text=f"AI Time: {time:.2f}s")