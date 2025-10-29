"""
Configuration file for Cubic Game
Contains all game settings and AI parameters
"""

# Game Configuration
BOARD_SIZE = 4
PLAYER_HUMAN = 1
PLAYER_AI = 2
EMPTY_CELL = 0

# AI Configuration
DEFAULT_MAX_DEPTH = 3
EARLY_GAME_DEPTH = 2
MID_GAME_DEPTH = 3
LATE_GAME_DEPTH = 4

EARLY_GAME_THRESHOLD = 50  # moves remaining
MID_GAME_THRESHOLD = 30

# Heuristic Weights
WEIGHT_WIN = 10000
WEIGHT_THREE_IN_ROW = 100
WEIGHT_TWO_IN_ROW = 10
WEIGHT_ONE_IN_ROW = 1

# GUI Configuration
WINDOW_WIDTH = 1920  # Full HD width
WINDOW_HEIGHT = 1080  # Full HD height
BUTTON_WIDTH = 4  # Smaller button width
BUTTON_HEIGHT = 2  # Smaller button height
LAYERS_TO_SHOW = 4  # Number of layers to show simultaneously

# Colors
COLOR_BG_DARK = '#2c3e50'
COLOR_BG_MEDIUM = '#34495e'
COLOR_BG_LIGHT = '#ecf0f1'
COLOR_PLAYER = '#3498db'
COLOR_AI = '#e74c3c'
COLOR_SUCCESS = '#27ae60'
COLOR_DANGER = '#e74c3c'
COLOR_NEUTRAL = '#95a5a6'
COLOR_WARNING = '#f39c12'

# Optimization Settings
USE_TRANSPOSITION_TABLE = True
USE_MOVE_ORDERING = True
USE_ADAPTIVE_DEPTH = True