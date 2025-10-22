"""
Configuration constants for NFL play animation.

This module contains styling and configuration constants used for animating plays.
"""

from typing import Dict, Tuple, List

# Field dimensions (in yards)
FIELD_LENGTH = 120  # Including 10-yard end zones on each side
FIELD_WIDTH = 53.3
END_ZONE_LENGTH = 10
PLAYING_FIELD_START = 10
PLAYING_FIELD_END = 110

# Default figure size
DEFAULT_FIGSIZE = (14, 7)

# Field colors
FIELD_COLOR = '#2E8B57'  # Sea green
END_ZONE_COLOR = 'green'
YARD_LINE_COLOR = 'white'

# Animation parameters
DEFAULT_INTERVAL = 200  # milliseconds between frames
DEFAULT_FPS = 5  # frames per second for saving
DEFAULT_PAUSE_TIME = 30  # seconds to display animation

# Player role color and marker formatting
# Format: {role: [color, marker]}
PLAYER_MARKER_FORMAT: Dict[str, List[str]] = {
    "Defensive Coverage": ['red', 'x'],
    "Other Route Runner": ['blue', 'o'],
    "Passer": ['navy', 'o'],
    "Targeted Receiver": ['cyan', 'o'],
    "Ball": ['black', 'X']
}

# Player role trajectory line formatting
# Format: {role: [color, linestyle]}
PLAYER_PATH_FORMAT: Dict[str, List[str]] = {
    "Defensive Coverage": ['red', '-'],
    "Other Route Runner": ['blue', '-'],
    "Passer": ['navy', '-'],
    "Targeted Receiver": ['cyan', '-']
}

# Column definitions for data loading
PLAY_KEYS = ['game_id', 'play_id']
PLAYER_KEYS = PLAY_KEYS + ['nfl_id']
PLAYER_DETAILS = [
    'player_name', 'player_height', 'player_weight',
    'player_birth_date', 'player_position', 'player_side', 'player_role'
]
PLAYER_MOVEMENT = ['frame_id', 'x', 'y']

# Yard line positions for field markings
YARD_LINE_POSITIONS = list(range(10, 111, 10))

# Yard line labels (mirrored for both sides)
YARD_LINE_LABELS = ['', '', 10, 20, 30, 40, 50, 40, 30, 20, 10, '', '']

# Field styling
YARD_LINE_WIDTH = 0.5
YARD_LINE_ALPHA = 0.5
END_ZONE_ALPHA = 0.2

# Text positioning
INFO_TEXT_X = 0.07
INFO_TEXT_Y = 0.015
INFO_TEXT_FONTSIZE = 10
TITLE_WRAP_WIDTH = 100  # characters

# Legend configuration
LEGEND_LOCATION = 'upper left'
