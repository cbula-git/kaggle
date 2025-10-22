"""
Utility functions for NFL analysis metrics and calculations.
"""

import numpy as np
import pandas as pd
import math
from typing import Tuple, Union


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        x1, y1: Coordinates of first point
        x2, y2: Coordinates of second point

    Returns:
        Euclidean distance between the points
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_separation(
    offensive_pos: Tuple[float, float],
    defensive_pos: Tuple[float, float]
) -> float:
    """
    Calculate separation distance between offensive and defensive players.

    Args:
        offensive_pos: (x, y) coordinates of offensive player
        defensive_pos: (x, y) coordinates of defensive player

    Returns:
        Separation distance in yards
    """
    off_x, off_y = offensive_pos
    def_x, def_y = defensive_pos
    return calculate_distance(off_x, off_y, def_x, def_y)


def calculate_velocity(
    x1: float, y1: float,
    x2: float, y2: float,
    time_delta: float = 0.1
) -> float:
    """
    Calculate velocity between two positions.

    Args:
        x1, y1: Starting position
        x2, y2: Ending position
        time_delta: Time elapsed (default 0.1 seconds per frame)

    Returns:
        Velocity in yards per second
    """
    distance = calculate_distance(x1, y1, x2, y2)
    return distance / time_delta if time_delta > 0 else 0.0


def calculate_displacement(
    trajectory_df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y'
) -> float:
    """
    Calculate total displacement along a trajectory.

    Args:
        trajectory_df: DataFrame with player trajectory (must be sorted by time)
        x_col: Name of x coordinate column
        y_col: Name of y coordinate column

    Returns:
        Total displacement in yards
    """
    if len(trajectory_df) < 2:
        return 0.0

    start_x, start_y = trajectory_df.iloc[0][[x_col, y_col]]
    end_x, end_y = trajectory_df.iloc[-1][[x_col, y_col]]

    return calculate_distance(start_x, start_y, end_x, end_y)


def calculate_path_length(
    trajectory_df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y'
) -> float:
    """
    Calculate total path length along a trajectory.

    Args:
        trajectory_df: DataFrame with player trajectory (must be sorted by time)
        x_col: Name of x coordinate column
        y_col: Name of y coordinate column

    Returns:
        Total path length in yards
    """
    if len(trajectory_df) < 2:
        return 0.0

    positions = trajectory_df[[x_col, y_col]].values
    distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    return float(np.sum(distances))


def calculate_angle(
    x1: float, y1: float,
    x2: float, y2: float,
    degrees: bool = True
) -> float:
    """
    Calculate angle from point 1 to point 2.

    Args:
        x1, y1: Coordinates of first point (origin)
        x2, y2: Coordinates of second point (target)
        degrees: Return angle in degrees (True) or radians (False)

    Returns:
        Angle in degrees or radians
    """
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    return float(np.degrees(angle_rad)) if degrees else float(angle_rad)


def calculate_relative_position(
    player_x: float, player_y: float,
    reference_x: float, reference_y: float
) -> Tuple[float, float, float]:
    """
    Calculate relative position metrics from a reference point.

    Args:
        player_x, player_y: Player coordinates
        reference_x, reference_y: Reference point coordinates

    Returns:
        Tuple of (distance, relative_x, relative_y)
    """
    distance = calculate_distance(player_x, player_y, reference_x, reference_y)
    relative_x = player_x - reference_x
    relative_y = player_y - reference_y

    return distance, relative_x, relative_y


def calculate_acceleration(
    v1: float, v2: float,
    time_delta: float = 0.1
) -> float:
    """
    Calculate acceleration between two velocities.

    Args:
        v1: Initial velocity (yards/second)
        v2: Final velocity (yards/second)
        time_delta: Time elapsed (default 0.1 seconds per frame)

    Returns:
        Acceleration in yards/second^2
    """
    return (v2 - v1) / time_delta if time_delta > 0 else 0.0


def normalize_direction(direction: float) -> float:
    """
    Normalize direction to range [0, 360).

    Args:
        direction: Direction in degrees

    Returns:
        Normalized direction in range [0, 360)
    """
    return direction % 360
