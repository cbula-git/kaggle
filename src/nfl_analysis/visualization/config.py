"""
Configuration constants for NFL visualization.

This module contains styling and configuration constants used for all
visualization types including animations, coverage analysis, and field rendering.
"""

from typing import Dict, Tuple, List

# ============================================================================
# FIELD DIMENSIONS
# ============================================================================

FIELD_LENGTH = 120  # Including 10-yard end zones on each side
FIELD_WIDTH = 53.3
END_ZONE_LENGTH = 10
PLAYING_FIELD_START = 10
PLAYING_FIELD_END = 110

# ============================================================================
# ANIMATION PARAMETERS
# ============================================================================

DEFAULT_INTERVAL = 200  # milliseconds between frames
DEFAULT_FPS = 5  # frames per second for saving
DEFAULT_PAUSE_TIME = 30  # seconds to display animation

# ============================================================================
# FIGURE SIZES (width, height in inches)
# ============================================================================

DEFAULT_FIGSIZE = (14, 7)
FIGSIZE_DOUBLE_WIDE = (18, 8)
FIGSIZE_GRID_2X2 = (16, 12)
FIGSIZE_GRID_3X3 = (16, 10)

# ============================================================================
# FIELD COLORS
# ============================================================================

FIELD_COLOR = '#2E8B57'  # Sea green
END_ZONE_COLOR = 'green'
YARD_LINE_COLOR = 'white'

# ============================================================================
# PLAYER MARKER SIZES
# ============================================================================

MARKER_SIZE_LARGE = 200
MARKER_SIZE_MEDIUM = 150
MARKER_SIZE_SMALL = 100

# ============================================================================
# PLAYER COLORS AND MARKERS
# ============================================================================

# Basic colors
OFFENSIVE_COLOR = 'blue'
DEFENSIVE_COLOR = 'red'
QB_COLOR = 'darkblue'
BALL_COLOR = 'yellow'

# Basic markers
OFFENSIVE_MARKER = '^'  # Triangle up
DEFENSIVE_MARKER = 'o'  # Circle
QB_MARKER = 's'  # Square
BALL_MARKER = '*'  # Star

# Player role color and marker formatting for animations
# Format: {role: [color, marker]}
PLAYER_MARKER_FORMAT: Dict[str, List[str]] = {
    "Defensive Coverage": ['red', 'x'],
    "Other Route Runner": ['blue', 'o'],
    "Passer": ['navy', 'o'],
    "Targeted Receiver": ['cyan', 'o'],
    "Ball": ['black', 'X']
}

# Player role trajectory line formatting for animations
# Format: {role: [color, linestyle]}
PLAYER_PATH_FORMAT: Dict[str, List[str]] = {
    "Defensive Coverage": ['red', '-'],
    "Other Route Runner": ['blue', '-'],
    "Passer": ['navy', '-'],
    "Targeted Receiver": ['cyan', '-']
}

# ============================================================================
# LINE STYLES
# ============================================================================

LOS_COLOR = 'yellow'
LOS_LINEWIDTH = 2
BOUNDARY_COLOR = 'white'
BOUNDARY_LINEWIDTH = 2
YARD_LINE_WIDTH = 0.5
YARD_LINE_ALPHA = 0.5
END_ZONE_ALPHA = 0.2

# ============================================================================
# ZONE VISUALIZATION
# ============================================================================

ZONE_ALPHA = 0.3
ZONE_EDGE_COLOR = 'black'
ZONE_EDGE_WIDTH = 1

# ============================================================================
# STRESS VISUALIZATION
# ============================================================================

STRESS_COLORMAP = 'YlOrRd'
STRESS_THRESHOLD_LOW = 0.5
STRESS_THRESHOLD_MED = 1.0
STRESS_THRESHOLD_HIGH = 1.5

# Stress level colors
STRESS_COLORS = {
    'low': 'green',
    'medium': 'orange',
    'high': 'darkred'
}

# ============================================================================
# BURDEN VISUALIZATION
# ============================================================================

BURDEN_COLORMAP = 'Reds'
BURDEN_LOW_THRESHOLD = 0.3
BURDEN_HIGH_THRESHOLD = 0.7

# Burden level colors
BURDEN_COLORS = {
    'low': 'lightgreen',
    'medium': 'yellow',
    'high': 'salmon'
}

# ============================================================================
# COVERAGE RADIUS
# ============================================================================

DEFENDER_COVERAGE_RADIUS = 7  # yards

# ============================================================================
# TEXT STYLING
# ============================================================================

TEXT_FONTSIZE_LARGE = 14
TEXT_FONTSIZE_MEDIUM = 12
TEXT_FONTSIZE_SMALL = 8
TEXT_FONTSIZE_TINY = 7
LABEL_OFFSET_VERTICAL = 1.5
LABEL_OFFSET_HORIZONTAL = 1.8

# Text positioning for animations
INFO_TEXT_X = 0.07
INFO_TEXT_Y = 0.015
INFO_TEXT_FONTSIZE = 10
TITLE_WRAP_WIDTH = 100  # characters

# ============================================================================
# PLOT STYLING
# ============================================================================

GRID_ALPHA = 0.3
LINE_ALPHA_LOW = 0.3
LINE_ALPHA_MED = 0.5
LINE_ALPHA_HIGH = 0.7

# ============================================================================
# PLOT LIMITS
# ============================================================================

FIELD_VIEW_OFFSET_X = 5
FIELD_VIEW_DEPTH = 35
FIELD_VIEW_OFFSET_Y = 2

# ============================================================================
# ZONE STRESS TIMELINE
# ============================================================================

TIMELINE_FRAME_STEP = 2
MAX_TIMELINE_FRAMES = 25

# ============================================================================
# ROUTE SYNERGY COLORS
# ============================================================================

SYNERGY_COLORS = {
    'spacing': '#FF6B6B',
    'coverage': '#4ECDC4',
    'horizontal': '#45B7D1',
    'vertical': '#96CEB4',
    'concept': '#FFEAA7'
}

# ============================================================================
# PLAYER POSITION FILTERS
# ============================================================================

ROUTE_RUNNER_POSITIONS = ['WR', 'TE', 'RB']

# ============================================================================
# FIELD MARKINGS
# ============================================================================

# Yard line positions for field markings
YARD_LINE_POSITIONS = list(range(10, 111, 10))

# Yard line labels (mirrored for both sides)
YARD_LINE_LABELS = ['', '', 10, 20, 30, 40, 50, 40, 30, 20, 10, '', '']

# ============================================================================
# DATA COLUMN DEFINITIONS
# ============================================================================

PLAY_KEYS = ['game_id', 'play_id']
PLAYER_KEYS = PLAY_KEYS + ['nfl_id']
PLAYER_DETAILS = [
    'player_name', 'player_height', 'player_weight',
    'player_birth_date', 'player_position', 'player_side', 'player_role'
]
PLAYER_MOVEMENT = ['frame_id', 'x', 'y']

# ============================================================================
# LEGEND CONFIGURATION
# ============================================================================

LEGEND_LOCATION = 'upper left'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_field_xlim(los: float, offset: float = FIELD_VIEW_OFFSET_X,
                   depth: float = FIELD_VIEW_DEPTH) -> Tuple[float, float]:
    """Get default x-axis limits for field view."""
    return (los - offset, los + depth)

def get_field_ylim(width: float = FIELD_WIDTH,
                   offset: float = FIELD_VIEW_OFFSET_Y) -> Tuple[float, float]:
    """Get default y-axis limits for field view."""
    return (-offset, width + offset)
