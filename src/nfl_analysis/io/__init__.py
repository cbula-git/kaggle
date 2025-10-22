"""
Input/Output module for NFL tracking data.

This module provides functionality to load and query consolidated NFL data.
"""

from nfl_analysis.io.loader import NFLDataLoader, load_input, load_output, load_play_level, load_player_analysis

__all__ = [
    "NFLDataLoader",
    "load_input",
    "load_output",
    "load_play_level",
    "load_player_analysis",
]
