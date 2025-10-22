"""
Configuration constants for coverage visualization.

This module contains styling, colors, and configuration constants used
for visualizing defensive coverage and offensive route concepts.
"""

from typing import Dict, Tuple

# Field dimensions (in yards)
FIELD_WIDTH = 53.3
FIELD_LENGTH = 120
END_ZONE_LENGTH = 10

# Player marker sizes
MARKER_SIZE_LARGE = 200
MARKER_SIZE_MEDIUM = 150
MARKER_SIZE_SMALL = 100

# Player colors and markers
OFFENSIVE_COLOR = 'blue'
DEFENSIVE_COLOR = 'red'
QB_COLOR = 'darkblue'
BALL_COLOR = 'yellow'

OFFENSIVE_MARKER = '^'  # Triangle up
DEFENSIVE_MARKER = 'o'  # Circle
QB_MARKER = 's'  # Square
BALL_MARKER = '*'  # Star

# Line styles
LOS_COLOR = 'yellow'
LOS_LINEWIDTH = 2
BOUNDARY_COLOR = 'white'
BOUNDARY_LINEWIDTH = 2

# Zone visualization colors
ZONE_ALPHA = 0.3
ZONE_EDGE_COLOR = 'black'
ZONE_EDGE_WIDTH = 1

# Stress visualization
STRESS_COLORMAP = 'YlOrRd'
STRESS_THRESHOLD_LOW = 0.5
STRESS_THRESHOLD_MED = 1.0
STRESS_THRESHOLD_HIGH = 1.5

# Burden colors
BURDEN_COLORMAP = 'Reds'
BURDEN_LOW_THRESHOLD = 0.3
BURDEN_HIGH_THRESHOLD = 0.7

# Coverage radius for defenders (yards)
DEFENDER_COVERAGE_RADIUS = 7

# Text styling
TEXT_FONTSIZE_LARGE = 14
TEXT_FONTSIZE_MEDIUM = 12
TEXT_FONTSIZE_SMALL = 8
TEXT_FONTSIZE_TINY = 7
LABEL_OFFSET_VERTICAL = 1.5
LABEL_OFFSET_HORIZONTAL = 1.8

# Plot styling
GRID_ALPHA = 0.3
LINE_ALPHA_LOW = 0.3
LINE_ALPHA_MED = 0.5
LINE_ALPHA_HIGH = 0.7

# Figure sizes (width, height in inches)
FIGSIZE_DOUBLE_WIDE = (18, 8)
FIGSIZE_GRID_2X2 = (16, 12)
FIGSIZE_GRID_3X3 = (16, 10)

# Plot limits
FIELD_VIEW_OFFSET_X = 5
FIELD_VIEW_DEPTH = 35
FIELD_VIEW_OFFSET_Y = 2

# Zone stress timeline
TIMELINE_FRAME_STEP = 2
MAX_TIMELINE_FRAMES = 25

# Route synergy colors
SYNERGY_COLORS = {
    'spacing': '#FF6B6B',
    'coverage': '#4ECDC4',
    'horizontal': '#45B7D1',
    'vertical': '#96CEB4',
    'concept': '#FFEAA7'
}

# Burden level colors
BURDEN_COLORS = {
    'low': 'lightgreen',
    'medium': 'yellow',
    'high': 'salmon'
}

# Stress level colors
STRESS_COLORS = {
    'low': 'green',
    'medium': 'orange',
    'high': 'darkred'
}

# Player position filters for route runners
ROUTE_RUNNER_POSITIONS = ['WR', 'TE', 'RB']

# Default view ranges
def get_field_xlim(los: float, offset: float = FIELD_VIEW_OFFSET_X,
                   depth: float = FIELD_VIEW_DEPTH) -> Tuple[float, float]:
    """Get default x-axis limits for field view."""
    return (los - offset, los + depth)

def get_field_ylim(width: float = FIELD_WIDTH,
                   offset: float = FIELD_VIEW_OFFSET_Y) -> Tuple[float, float]:
    """Get default y-axis limits for field view."""
    return (-offset, width + offset)
