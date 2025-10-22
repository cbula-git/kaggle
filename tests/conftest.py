"""
Pytest configuration and fixtures for nfl_analysis tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_input_data():
    """Create sample input tracking data for testing."""
    return pd.DataFrame({
        'game_id': [2023090700] * 10,
        'play_id': [1] * 10,
        'nfl_id': range(10),
        'frame_id': [1] * 10,
        'x': np.random.uniform(0, 120, 10),
        'y': np.random.uniform(0, 53.3, 10),
        's': np.random.uniform(0, 10, 10),
        'a': np.random.uniform(-5, 5, 10),
        'o': np.random.uniform(0, 360, 10),
        'dir': np.random.uniform(0, 360, 10),
        'player_position': ['QB', 'WR', 'WR', 'TE', 'DB', 'DB', 'DB', 'LB', 'DL', 'DL'],
        'player_side': ['offense', 'offense', 'offense', 'offense', 'defense', 'defense', 'defense', 'defense', 'defense', 'defense'],
        'player_role': ['Passer', 'Targeted Receiver', 'Other Route Runner', 'Other Route Runner',
                       'Defensive Coverage', 'Defensive Coverage', 'Defensive Coverage',
                       'Defensive Coverage', 'Defensive Coverage', 'Defensive Coverage'],
        'player_to_predict': [False, True, True, True, True, True, True, True, False, False],
        'player_name': [f'Player_{i}' for i in range(10)],
        'ball_land_x': [60.0] * 10,
        'ball_land_y': [26.7] * 10,
        'week': [1] * 10,
        'play_direction': ['right'] * 10,
        'absolute_yardline_number': [50] * 10,
        'num_frames_output': [10] * 10
    })


@pytest.fixture
def sample_output_data():
    """Create sample output tracking data for testing."""
    return pd.DataFrame({
        'game_id': [2023090700] * 7,
        'play_id': [1] * 7,
        'nfl_id': range(1, 8),
        'frame_id': [1] * 7,
        'x': np.random.uniform(0, 120, 7),
        'y': np.random.uniform(0, 53.3, 7),
        'week': [1] * 7
    })


@pytest.fixture
def sample_supplementary_data():
    """Create sample supplementary data for testing."""
    return pd.DataFrame({
        'game_id': [2023090700],
        'play_id': [1],
        'pass_result': ['C'],
        'pass_length': [25.0],
        'yards_gained': [22],
        'team_coverage_type': ['Cover 2'],
        'offense_formation': ['Shotgun'],
        'route_of_targeted_receiver': ['Go'],
        'season': [2023],
        'week': [1]
    })


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_consolidated_dir(temp_data_dir, sample_input_data, sample_output_data, sample_supplementary_data):
    """Create a temporary directory with sample consolidated data files."""
    consolidated_dir = temp_data_dir / "consolidated"
    consolidated_dir.mkdir()

    # Save sample data as parquet files
    sample_input_data.to_parquet(consolidated_dir / "master_input.parquet")
    sample_output_data.to_parquet(consolidated_dir / "master_output.parquet")
    sample_supplementary_data.to_parquet(consolidated_dir / "supplementary.parquet")

    return consolidated_dir
