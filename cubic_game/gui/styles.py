"""
UI styling and theming
"""

from typing import Dict, Union, Literal, TypedDict
import tkinter as tk
from config import (
    COLOR_BG_DARK, COLOR_BG_MEDIUM, COLOR_BG_LIGHT,
    COLOR_PLAYER, COLOR_AI, COLOR_SUCCESS, COLOR_DANGER,
    COLOR_NEUTRAL, COLOR_WARNING
)

ColorDict = Dict[str, str]
ButtonStyle = Literal['normal', 'primary', 'danger', 'cell']


class ButtonStyleConfig(TypedDict):
    bg: str
    fg: str
    font: tuple[str, int] | tuple[str, int, str]

class StyleManager:
    """Manages UI styles and themes"""

    # Font configurations
    FONT_TITLE = ('Arial', 18, 'bold')
    FONT_HEADING = ('Arial', 14, 'bold')
    FONT_NORMAL = ('Arial', 12)
    FONT_BUTTON = ('Arial', 12, 'bold')
    FONT_CELL = ('Arial', 16, 'bold')

    # Color scheme
    COLORS: ColorDict = {
        'bg_dark': COLOR_BG_DARK,
        'bg_medium': COLOR_BG_MEDIUM,
        'bg_light': COLOR_BG_LIGHT,
        'player': COLOR_PLAYER,
        'ai': COLOR_AI,
        'success': COLOR_SUCCESS,
        'danger': COLOR_DANGER,
        'neutral': COLOR_NEUTRAL,
        'warning': COLOR_WARNING,
        'white': '#ffffff',
        'black': '#000000'
    }

    @staticmethod
    def get_player_color(player_id: int) -> str:
        """Get color for player"""
        if player_id == 1:
            return StyleManager.COLORS['player']
        elif player_id == 2:
            return StyleManager.COLORS['ai']
        return StyleManager.COLORS['white']

    @staticmethod
    def configure_button(button: tk.Button, style: ButtonStyle = 'normal') -> None:
        """Apply style to button"""
        styles: Dict[str, ButtonStyleConfig] = {
            'normal': {
                'bg': StyleManager.COLORS['white'],
                'fg': StyleManager.COLORS['black'],
                'font': StyleManager.FONT_NORMAL
            },
            'primary': {
                'bg': StyleManager.COLORS['success'],
                'fg': StyleManager.COLORS['white'],
                'font': StyleManager.FONT_BUTTON
            },
            'danger': {
                'bg': StyleManager.COLORS['danger'],
                'fg': StyleManager.COLORS['white'],
                'font': StyleManager.FONT_BUTTON
            },
            'cell': {
                'bg': StyleManager.COLORS['white'],
                'fg': StyleManager.COLORS['black'],
                'font': StyleManager.FONT_CELL
            }
        }

        if style in styles:
            button.config(**styles[style])