"""
NFL Data Loader Utilities
=========================
Convenient functions for loading and working with consolidated NFL data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
import warnings


class NFLDataLoader:
    """Utility class for loading consolidated NFL tracking data."""

    def __init__(self, data_dir: str = 'data/consolidated'):
        """
        Initialize data loader.

        Args:
            data_dir: Path to consolidated data directory
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                "Please run consolidation first."
            )

    def load_input(self) -> pd.DataFrame:
        """Load master input (pre-pass) data."""
        return pd.read_parquet(self.data_dir / 'master_input.parquet')

    def load_output(self) -> pd.DataFrame:
        """Load master output (post-pass) data."""
        return pd.read_parquet(self.data_dir / 'master_output.parquet')

    def load_supplementary(self) -> pd.DataFrame:
        """Load supplementary play context data."""
        return pd.read_parquet(self.data_dir / 'supplementary.parquet')

    def load_play_level(self) -> pd.DataFrame:
        """Load play-level aggregated data."""
        return pd.read_parquet(self.data_dir / 'play_level.parquet')

    def load_trajectories(self) -> pd.DataFrame:
        """Load complete player trajectories (pre + post pass)."""
        return pd.read_parquet(self.data_dir / 'trajectories.parquet')

    def load_player_analysis(self) -> pd.DataFrame:
        """Load player-centric analysis data."""
        return pd.read_parquet(self.data_dir / 'player_analysis.parquet')

    def load_spatial_features(self) -> Optional[pd.DataFrame]:
        """Load spatial relationship features (if available)."""
        spatial_path = self.data_dir / 'spatial_features.parquet'
        if spatial_path.exists():
            return pd.read_parquet(spatial_path)
        else:
            warnings.warn("Spatial features not found. Run consolidation with skip_spatial=False.")
            return None

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets.

        Returns:
            Dictionary with dataset names as keys and DataFrames as values
        """
        datasets = {
            'input': self.load_input(),
            'output': self.load_output(),
            'supplementary': self.load_supplementary(),
            'play_level': self.load_play_level(),
            'trajectories': self.load_trajectories(),
            'player_analysis': self.load_player_analysis(),
        }

        spatial = self.load_spatial_features()
        if spatial is not None:
            datasets['spatial'] = spatial

        return datasets

    def get_play_data(self, game_id: int, play_id: int) -> Dict[str, pd.DataFrame]:
        """
        Get all data for a specific play.

        Args:
            game_id: Game identifier
            play_id: Play identifier

        Returns:
            Dictionary with input, output, and supplementary data for the play
        """
        input_df = self.load_input()
        output_df = self.load_output()
        supp_df = self.load_supplementary()

        play_data = {
            'input': input_df[
                (input_df['game_id'] == game_id) &
                (input_df['play_id'] == play_id)
            ].copy(),
            'output': output_df[
                (output_df['game_id'] == game_id) &
                (output_df['play_id'] == play_id)
            ].copy(),
            'supplementary': supp_df[
                (supp_df['game_id'] == game_id) &
                (supp_df['play_id'] == play_id)
            ].copy()
        }

        return play_data

    def get_player_data(self, nfl_id: int, limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Get all data for a specific player.

        Args:
            nfl_id: Player identifier
            limit: Optional limit on number of plays to return

        Returns:
            Dictionary with input and output data for the player
        """
        input_df = self.load_input()
        output_df = self.load_output()

        player_input = input_df[input_df['nfl_id'] == nfl_id].copy()
        player_output = output_df[output_df['nfl_id'] == nfl_id].copy()

        if limit:
            # Get unique plays
            plays = player_input[['game_id', 'play_id']].drop_duplicates().head(limit)
            player_input = player_input.merge(plays, on=['game_id', 'play_id'])
            player_output = player_output.merge(plays, on=['game_id', 'play_id'])

        return {
            'input': player_input,
            'output': player_output
        }

    def get_week_data(self, week: int) -> Dict[str, pd.DataFrame]:
        """
        Get all data for a specific week.

        Args:
            week: Week number (1-18)

        Returns:
            Dictionary with input and output data for the week
        """
        input_df = self.load_input()
        output_df = self.load_output()

        return {
            'input': input_df[input_df['week'] == week].copy(),
            'output': output_df[output_df['week'] == week].copy()
        }

    def list_available_datasets(self) -> List[str]:
        """List all available consolidated datasets."""
        datasets = []
        for file in self.data_dir.glob('*.parquet'):
            if not file.name.endswith('_sample.csv'):
                datasets.append(file.stem)
        return sorted(datasets)

    def get_dataset_info(self) -> pd.DataFrame:
        """Get information about all available datasets."""
        info = []
        for dataset in self.list_available_datasets():
            file_path = self.data_dir / f"{dataset}.parquet"
            df = pd.read_parquet(file_path)
            info.append({
                'dataset': dataset,
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
            })

        return pd.DataFrame(info)


# Convenience functions
def load_input(data_dir: str = 'data/consolidated') -> pd.DataFrame:
    """Quick load master input data."""
    return NFLDataLoader(data_dir).load_input()


def load_output(data_dir: str = 'data/consolidated') -> pd.DataFrame:
    """Quick load master output data."""
    return NFLDataLoader(data_dir).load_output()


def load_play_level(data_dir: str = 'data/consolidated') -> pd.DataFrame:
    """Quick load play-level data."""
    return NFLDataLoader(data_dir).load_play_level()


def load_player_analysis(data_dir: str = 'data/consolidated') -> pd.DataFrame:
    """Quick load player analysis data."""
    return NFLDataLoader(data_dir).load_player_analysis()
