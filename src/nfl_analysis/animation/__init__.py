"""
Animation module for NFL tracking data.

This module provides functionality to animate and visualize plays.
"""

from .animator import PlayAnimator
from .field_renderer import FieldRenderer

__all__ = [
    'PlayAnimator',
    'FieldRenderer',
]
