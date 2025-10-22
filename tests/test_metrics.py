"""
Tests for nfl_analysis.utils.metrics module.
"""

import pytest
import pandas as pd
import numpy as np
from nfl_analysis.utils.metrics import (
    calculate_distance,
    calculate_separation,
    calculate_velocity,
    calculate_displacement,
    calculate_path_length,
    calculate_angle,
    calculate_relative_position,
    calculate_acceleration,
    normalize_direction
)


class TestCalculateDistance:
    """Tests for calculate_distance function."""

    def test_zero_distance(self):
        """Test distance calculation for same point."""
        assert calculate_distance(0, 0, 0, 0) == 0.0

    def test_horizontal_distance(self):
        """Test distance calculation for horizontal line."""
        assert calculate_distance(0, 0, 3, 0) == 3.0

    def test_vertical_distance(self):
        """Test distance calculation for vertical line."""
        assert calculate_distance(0, 0, 0, 4) == 4.0

    def test_diagonal_distance(self):
        """Test distance calculation for diagonal line (3-4-5 triangle)."""
        assert calculate_distance(0, 0, 3, 4) == 5.0


class TestCalculateSeparation:
    """Tests for calculate_separation function."""

    def test_separation(self):
        """Test separation calculation."""
        offensive_pos = (10, 20)
        defensive_pos = (13, 24)
        assert calculate_separation(offensive_pos, defensive_pos) == 5.0


class TestCalculateVelocity:
    """Tests for calculate_velocity function."""

    def test_zero_velocity(self):
        """Test velocity when no movement."""
        assert calculate_velocity(0, 0, 0, 0, 0.1) == 0.0

    def test_constant_velocity(self):
        """Test velocity calculation."""
        # Move 1 yard in 0.1 seconds = 10 yards/second
        assert calculate_velocity(0, 0, 1, 0, 0.1) == 10.0


class TestCalculateDisplacement:
    """Tests for calculate_displacement function."""

    def test_displacement_single_point(self):
        """Test displacement with single point."""
        df = pd.DataFrame({'x': [0], 'y': [0]})
        assert calculate_displacement(df) == 0.0

    def test_displacement_straight_line(self):
        """Test displacement along straight path."""
        df = pd.DataFrame({
            'x': [0, 1, 2, 3],
            'y': [0, 0, 0, 0]
        })
        assert calculate_displacement(df) == 3.0


class TestCalculatePathLength:
    """Tests for calculate_path_length function."""

    def test_path_length_single_point(self):
        """Test path length with single point."""
        df = pd.DataFrame({'x': [0], 'y': [0]})
        assert calculate_path_length(df) == 0.0

    def test_path_length_straight(self):
        """Test path length along straight path."""
        df = pd.DataFrame({
            'x': [0, 1, 2, 3],
            'y': [0, 0, 0, 0]
        })
        assert calculate_path_length(df) == 3.0

    def test_path_length_zigzag(self):
        """Test path length along zigzag path."""
        df = pd.DataFrame({
            'x': [0, 1, 1, 2],
            'y': [0, 0, 1, 1]
        })
        expected = 1.0 + 1.0 + 1.0  # Three segments of length 1
        assert abs(calculate_path_length(df) - expected) < 0.001


class TestCalculateAngle:
    """Tests for calculate_angle function."""

    def test_angle_right(self):
        """Test angle calculation for right direction."""
        assert calculate_angle(0, 0, 1, 0, degrees=True) == 0.0

    def test_angle_up(self):
        """Test angle calculation for upward direction."""
        assert calculate_angle(0, 0, 0, 1, degrees=True) == 90.0

    def test_angle_left(self):
        """Test angle calculation for left direction."""
        assert abs(calculate_angle(0, 0, -1, 0, degrees=True) - 180.0) < 0.001

    def test_angle_radians(self):
        """Test angle calculation in radians."""
        angle = calculate_angle(0, 0, 1, 0, degrees=False)
        assert angle == 0.0


class TestCalculateRelativePosition:
    """Tests for calculate_relative_position function."""

    def test_relative_position(self):
        """Test relative position calculation."""
        distance, rel_x, rel_y = calculate_relative_position(10, 20, 5, 10)
        assert distance == pytest.approx(11.1803, rel=0.01)
        assert rel_x == 5.0
        assert rel_y == 10.0


class TestCalculateAcceleration:
    """Tests for calculate_acceleration function."""

    def test_zero_acceleration(self):
        """Test acceleration when velocity is constant."""
        assert calculate_acceleration(10, 10, 0.1) == 0.0

    def test_positive_acceleration(self):
        """Test positive acceleration."""
        # Velocity increases from 0 to 1 yards/s in 0.1 s = 10 yards/s^2
        assert calculate_acceleration(0, 1, 0.1) == 10.0

    def test_negative_acceleration(self):
        """Test negative acceleration (deceleration)."""
        assert calculate_acceleration(1, 0, 0.1) == -10.0


class TestNormalizeDirection:
    """Tests for normalize_direction function."""

    def test_normalize_positive(self):
        """Test normalization of positive angle."""
        assert normalize_direction(45) == 45.0

    def test_normalize_over_360(self):
        """Test normalization of angle over 360."""
        assert normalize_direction(400) == 40.0

    def test_normalize_negative(self):
        """Test normalization of negative angle."""
        assert normalize_direction(-45) == 315.0
