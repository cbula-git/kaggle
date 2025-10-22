"""
Tests for nfl_analysis.io.loader module.
"""

import pytest
import pandas as pd
from nfl_analysis.io.loader import NFLDataLoader


class TestNFLDataLoader:
    """Tests for NFLDataLoader class."""

    def test_init_with_valid_dir(self, sample_consolidated_dir):
        """Test initialization with valid directory."""
        loader = NFLDataLoader(str(sample_consolidated_dir))
        assert loader.data_dir == sample_consolidated_dir

    def test_init_with_invalid_dir(self):
        """Test initialization with invalid directory."""
        with pytest.raises(FileNotFoundError):
            NFLDataLoader("/nonexistent/directory")

    def test_load_input(self, sample_consolidated_dir, sample_input_data):
        """Test loading input data."""
        loader = NFLDataLoader(str(sample_consolidated_dir))
        df = loader.load_input()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_input_data)

    def test_load_output(self, sample_consolidated_dir, sample_output_data):
        """Test loading output data."""
        loader = NFLDataLoader(str(sample_consolidated_dir))
        df = loader.load_output()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_output_data)

    def test_load_supplementary(self, sample_consolidated_dir, sample_supplementary_data):
        """Test loading supplementary data."""
        loader = NFLDataLoader(str(sample_consolidated_dir))
        df = loader.load_supplementary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_supplementary_data)

    def test_get_play_data(self, sample_consolidated_dir):
        """Test getting data for specific play."""
        loader = NFLDataLoader(str(sample_consolidated_dir))
        play_data = loader.get_play_data(game_id=2023090700, play_id=1)

        assert 'input' in play_data
        assert 'output' in play_data
        assert 'supplementary' in play_data
        assert len(play_data['input']) > 0
        assert len(play_data['output']) > 0
        assert len(play_data['supplementary']) == 1

    def test_get_player_data(self, sample_consolidated_dir):
        """Test getting data for specific player."""
        loader = NFLDataLoader(str(sample_consolidated_dir))
        player_data = loader.get_player_data(nfl_id=1)

        assert 'input' in player_data
        assert 'output' in player_data
        assert len(player_data['input']) == 1  # Player 1 appears once in sample data
        assert len(player_data['output']) == 1

    def test_list_available_datasets(self, sample_consolidated_dir):
        """Test listing available datasets."""
        loader = NFLDataLoader(str(sample_consolidated_dir))
        datasets = loader.list_available_datasets()

        assert 'master_input' in datasets
        assert 'master_output' in datasets
        assert 'supplementary' in datasets

    def test_get_dataset_info(self, sample_consolidated_dir):
        """Test getting dataset information."""
        loader = NFLDataLoader(str(sample_consolidated_dir))
        info = loader.get_dataset_info()

        assert isinstance(info, pd.DataFrame)
        assert 'dataset' in info.columns
        assert 'rows' in info.columns
        assert 'columns' in info.columns
        assert 'memory_mb' in info.columns
