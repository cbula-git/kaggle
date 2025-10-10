
"""
NFL Tracking Data Consolidation Script
======================================
Consolidates weekly NFL tracking data into unified datasets for analysis.

Usage:
    python consolidate_nfl_data.py

Output:
    consolidated_data/
        ├── master_input.parquet
        ├── master_output.parquet
        ├── supplementary.parquet
        ├── play_level.parquet
        ├── trajectories.parquet
        ├── player_analysis.parquet
        └── spatial_features.parquet
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from typing import Dict, Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NFLDataConsolidator:
    """Consolidates NFL tracking data from multiple weekly files."""
    
    def __init__(self, data_dir: str, output_dir: str = 'consolidated_data'):
        """
        Initialize the consolidator.
        
        Args:
            data_dir: Path to the directory containing NFL data
            output_dir: Path where consolidated data will be saved
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.master_input = None
        self.master_output = None
        self.supplementary = None
        
    def consolidate_input_data(self) -> pd.DataFrame:
        """
        Combine all weekly input files into one master input dataset.
        
        Returns:
            DataFrame with all input tracking data
        """
        logger.info("Consolidating input data...")
        
        input_files = sorted(glob.glob(str(self.data_dir / "train" / "input_2023_w*.csv")))
        
        if not input_files:
            raise FileNotFoundError(f"No input files found in {self.data_dir / 'train'}")
        
        logger.info(f"Found {len(input_files)} input files")
        
        dfs = []
        for file in input_files:
            week = int(Path(file).stem.split('_w')[-1])
            logger.info(f"  Reading week {week}...")
            df = pd.read_csv(file)
            df['week'] = week
            dfs.append(df)
        
        master_input = pd.concat(dfs, ignore_index=True)
        logger.info(f"  Combined {len(master_input):,} rows")
        
        return master_input
    
    def consolidate_output_data(self) -> pd.DataFrame:
        """
        Combine all weekly output files into one master output dataset.
        
        Returns:
            DataFrame with all output tracking data
        """
        logger.info("Consolidating output data...")
        
        output_files = sorted(glob.glob(str(self.data_dir / "train" / "output_2023_w*.csv")))
        
        if not output_files:
            raise FileNotFoundError(f"No output files found in {self.data_dir / 'train'}")
        
        logger.info(f"Found {len(output_files)} output files")
        
        dfs = []
        for file in output_files:
            week = int(Path(file).stem.split('_w')[-1])
            logger.info(f"  Reading week {week}...")
            df = pd.read_csv(file)
            df['week'] = week
            dfs.append(df)
        
        master_output = pd.concat(dfs, ignore_index=True)
        logger.info(f"  Combined {len(master_output):,} rows")
        
        return master_output
    
    def load_supplementary_data(self) -> pd.DataFrame:
        """
        Load supplementary play context data.
        
        Returns:
            DataFrame with supplementary data
        """
        logger.info("Loading supplementary data...")
        
        supp_file = self.data_dir / "supplementary_data.csv"
        if not supp_file.exists():
            raise FileNotFoundError(f"Supplementary data not found: {supp_file}")
        
        supplementary = pd.read_csv(supp_file)
        logger.info(f"  Loaded {len(supplementary):,} plays")
        
        return supplementary
    
    def create_play_level_dataset(self) -> pd.DataFrame:
        """
        Aggregate input data at play level with statistical features.
        
        Returns:
            DataFrame with play-level aggregated features
        """
        logger.info("Creating play-level dataset...")
        
        # Group by play and calculate aggregate features
        agg_dict = {
            'frame_id': 'max',
            'x': ['mean', 'std', 'min', 'max'],
            'y': ['mean', 'std', 'min', 'max'],
            's': ['mean', 'max', 'std'],
            'a': ['mean', 'max', 'std'],
            'player_to_predict': 'sum',
            'ball_land_x': 'first',
            'ball_land_y': 'first',
            'absolute_yardline_number': 'first',
            'play_direction': 'first',
            'nfl_id': 'nunique'  # Count unique players
        }
        
        play_features = self.master_input.groupby(['game_id', 'play_id']).agg(agg_dict).reset_index()
        
        # Flatten column names
        play_features.columns = ['_'.join(col).strip('_') for col in play_features.columns.values]
        
        # Rename for clarity
        rename_dict = {
            'game_id_': 'game_id',
            'play_id_': 'play_id',
            'frame_id_max': 'max_input_frames',
            'player_to_predict_sum': 'num_players_to_predict',
            'nfl_id_nunique': 'num_unique_players',
            'ball_land_x_first': 'ball_land_x',
            'ball_land_y_first': 'ball_land_y',
            'absolute_yardline_number_first': 'absolute_yardline_number',
            'play_direction_first': 'play_direction'
        }
        play_features.rename(columns=rename_dict, inplace=True)
        
        # Merge with supplementary data
        play_dataset = play_features.merge(
            self.supplementary,
            on=['game_id', 'play_id'],
            how='left'
        )
        
        logger.info(f"  Created dataset with {len(play_dataset):,} plays")
        
        return play_dataset
    
    def create_trajectory_dataset(self) -> pd.DataFrame:
        """
        Create complete player trajectories combining input and output data.
        
        Returns:
            DataFrame with complete player trajectories
        """
        logger.info("Creating trajectory dataset...")
        
        # Add phase indicator
        input_data = self.master_input.copy()
        input_data['phase'] = 'pre_pass'
        
        output_data = self.master_output.copy()
        output_data['phase'] = 'post_pass'
        
        # Get max input frame for each play/player to make frame_ids continuous
        max_input_frames = input_data.groupby(
            ['game_id', 'play_id', 'nfl_id']
        )['frame_id'].max().reset_index()
        max_input_frames.columns = ['game_id', 'play_id', 'nfl_id', 'max_input_frame']
        
        # Adjust output frame_ids to be continuous with input
        output_data_adj = output_data.merge(
            max_input_frames,
            on=['game_id', 'play_id', 'nfl_id'],
            how='left'
        )
        output_data_adj['frame_id'] = output_data_adj['frame_id'] + output_data_adj['max_input_frame']
        output_data_adj.drop('max_input_frame', axis=1, inplace=True)
        
        # Select columns for stacking
        common_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y', 'phase', 'week']
        
        # Keep additional features from input
        input_cols = common_cols + ['s', 'a', 'o', 'dir', 
                                      'player_position', 'player_side', 
                                      'player_role', 'player_to_predict',
                                      'player_name', 'ball_land_x', 'ball_land_y']
        
        # Filter to available columns
        input_cols = [col for col in input_cols if col in input_data.columns]
        output_cols = [col for col in common_cols if col in output_data_adj.columns]
        
        # Combine
        full_trajectories = pd.concat([
            input_data[input_cols],
            output_data_adj[output_cols]
        ], ignore_index=True).sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
        
        logger.info(f"  Created dataset with {len(full_trajectories):,} frames")
        
        return full_trajectories
    
    def create_player_analysis_dataset(self) -> pd.DataFrame:
        """
        Create player-centric analysis dataset focusing on players needing predictions.
        
        Returns:
            DataFrame with player-level analysis features
        """
        logger.info("Creating player analysis dataset...")
        
        # Filter to players needing predictions
        predict_players = self.master_input[
            self.master_input['player_to_predict'] == True
        ].copy()
        
        # Get last frame before pass for each player
        last_pre_pass = predict_players.loc[
            predict_players.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].idxmax()
        ].copy()
        
        # Get first post-pass position
        first_post_pass = self.master_output.loc[
            self.master_output.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].idxmin()
        ][['game_id', 'play_id', 'nfl_id', 'x', 'y', 'frame_id']].rename(
            columns={'x': 'x_first_post', 'y': 'y_first_post', 'frame_id': 'frame_first_post'}
        )
        
        # Get last post-pass position
        last_post_pass = self.master_output.loc[
            self.master_output.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].idxmax()
        ][['game_id', 'play_id', 'nfl_id', 'x', 'y', 'frame_id']].rename(
            columns={'x': 'x_last_post', 'y': 'y_last_post', 'frame_id': 'frame_last_post'}
        )
        
        # Merge everything
        player_dataset = last_pre_pass.merge(
            first_post_pass,
            on=['game_id', 'play_id', 'nfl_id'],
            how='left'
        ).merge(
            last_post_pass,
            on=['game_id', 'play_id', 'nfl_id'],
            how='left'
        ).merge(
            self.supplementary,
            on=['game_id', 'play_id'],
            how='left'
        )
        
        # Calculate displacement features
        player_dataset['displacement_x'] = player_dataset['x_last_post'] - player_dataset['x']
        player_dataset['displacement_y'] = player_dataset['y_last_post'] - player_dataset['y']
        player_dataset['total_displacement'] = np.sqrt(
            player_dataset['displacement_x']**2 + player_dataset['displacement_y']**2
        )
        
        # Distance to ball landing
        player_dataset['dist_to_ball_pre_pass'] = np.sqrt(
            (player_dataset['x'] - player_dataset['ball_land_x'])**2 +
            (player_dataset['y'] - player_dataset['ball_land_y'])**2
        )
        
        player_dataset['dist_to_ball_post_pass'] = np.sqrt(
            (player_dataset['x_last_post'] - player_dataset['ball_land_x'])**2 +
            (player_dataset['y_last_post'] - player_dataset['ball_land_y'])**2
        )
        
        # Calculate average velocity during post-pass phase
        player_dataset['avg_velocity_post_pass'] = (
            player_dataset['total_displacement'] / player_dataset['num_frames_output']
        )
        
        logger.info(f"  Created dataset with {len(player_dataset):,} player-plays")
        
        return player_dataset
    
    def create_spatial_features(self) -> pd.DataFrame:
        """
        Create spatial context features with relative positions and distances.
        
        Returns:
            DataFrame with spatial relationship features
        """
        logger.info("Creating spatial features...")
        logger.info("  This may take a while for large datasets...")
        
        spatial_features = []
        
        # Process in batches by play for efficiency
        grouped = self.master_input.groupby(['game_id', 'play_id', 'frame_id'])
        total_groups = len(grouped)
        
        for i, ((game_id, play_id, frame_id), frame_data) in enumerate(grouped):
            if i % 1000 == 0:
                logger.info(f"    Processing frame {i:,} of {total_groups:,}")
            
            # Get passer position
            passer = frame_data[frame_data['player_role'] == 'Passer']
            if len(passer) == 0:
                continue
            passer_x, passer_y = passer.iloc[0][['x', 'y']]
            
            # Get ball landing position
            ball_x = frame_data.iloc[0]['ball_land_x']
            ball_y = frame_data.iloc[0]['ball_land_y']
            
            # Get targeted receiver position if available
            target = frame_data[frame_data['player_role'] == 'Targeted Receiver']
            if len(target) > 0:
                target_x, target_y = target.iloc[0][['x', 'y']]
            else:
                target_x, target_y = None, None
            
            # Calculate distances for each player
            for idx, player in frame_data.iterrows():
                features = {
                    'game_id': game_id,
                    'play_id': play_id,
                    'frame_id': frame_id,
                    'nfl_id': player['nfl_id'],
                    'dist_to_passer': np.sqrt((player['x'] - passer_x)**2 + (player['y'] - passer_y)**2),
                    'dist_to_ball_landing': np.sqrt((player['x'] - ball_x)**2 + (player['y'] - ball_y)**2),
                    'relative_x_to_ball': player['x'] - ball_x,
                    'relative_y_to_ball': player['y'] - ball_y,
                    'angle_to_ball': np.arctan2(player['y'] - ball_y, player['x'] - ball_x) * 180 / np.pi,
                }
                
                if target_x is not None:
                    features['dist_to_target'] = np.sqrt((player['x'] - target_x)**2 + (player['y'] - target_y)**2)
                
                spatial_features.append(features)
        
        spatial_df = pd.DataFrame(spatial_features)
        logger.info(f"  Created dataset with {len(spatial_df):,} player-frames")
        
        return spatial_df
    
    def save_dataset(self, df: pd.DataFrame, name: str):
        """Save dataset to parquet format."""
        output_path = self.output_dir / f"{name}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"  Saved to {output_path}")
        
        # Also save a small CSV sample for inspection
        sample_path = self.output_dir / f"{name}_sample.csv"
        df.head(1000).to_csv(sample_path, index=False)
        logger.info(f"  Saved sample to {sample_path}")
    
    def generate_summary_report(self) -> Dict:
        """
        Generate summary statistics about the consolidated datasets.
        
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Generating summary report...")
        
        summary = {
            'total_weeks': self.master_input['week'].nunique(),
            'total_games': self.master_input['game_id'].nunique(),
            'total_plays': self.master_input.groupby(['game_id', 'play_id']).ngroups,
            'total_players': self.master_input['nfl_id'].nunique(),
            'total_input_frames': len(self.master_input),
            'total_output_frames': len(self.master_output),
            'players_to_predict': self.master_input[
                self.master_input['player_to_predict'] == True
            ].groupby(['game_id', 'play_id', 'nfl_id']).ngroups,
            'avg_frames_per_play_input': self.master_input.groupby(['game_id', 'play_id'])['frame_id'].max().mean(),
            'avg_frames_per_play_output': self.master_output.groupby(['game_id', 'play_id'])['frame_id'].max().mean(),
        }
        
        # Player position distribution
        position_dist = self.master_input.groupby('player_position')['nfl_id'].nunique().to_dict()
        summary['player_positions'] = position_dist
        
        # Player role distribution
        role_dist = self.master_input['player_role'].value_counts().to_dict()
        summary['player_roles'] = role_dist
        
        # Pass result distribution
        if 'pass_result' in self.supplementary.columns:
            pass_result_dist = self.supplementary['pass_result'].value_counts().to_dict()
            summary['pass_results'] = pass_result_dist
        
        return summary
    
    def save_summary_report(self, summary: Dict):
        """Save summary report to file."""
        report_path = self.output_dir / "consolidation_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("NFL TRACKING DATA CONSOLIDATION SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Weeks:              {summary['total_weeks']}\n")
            f.write(f"Total Games:              {summary['total_games']:,}\n")
            f.write(f"Total Plays:              {summary['total_plays']:,}\n")
            f.write(f"Total Unique Players:     {summary['total_players']:,}\n")
            f.write(f"Total Input Frames:       {summary['total_input_frames']:,}\n")
            f.write(f"Total Output Frames:      {summary['total_output_frames']:,}\n")
            f.write(f"Players to Predict:       {summary['players_to_predict']:,}\n")
            f.write(f"Avg Input Frames/Play:    {summary['avg_frames_per_play_input']:.1f}\n")
            f.write(f"Avg Output Frames/Play:   {summary['avg_frames_per_play_output']:.1f}\n\n")
            
            if 'player_positions' in summary:
                f.write("PLAYER POSITIONS\n")
                f.write("-" * 70 + "\n")
                for pos, count in sorted(summary['player_positions'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {pos:20s}: {count:4d} unique players\n")
                f.write("\n")
            
            if 'player_roles' in summary:
                f.write("PLAYER ROLES\n")
                f.write("-" * 70 + "\n")
                for role, count in sorted(summary['player_roles'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {role:25s}: {count:8,} occurrences\n")
                f.write("\n")
            
            if 'pass_results' in summary:
                f.write("PASS RESULTS\n")
                f.write("-" * 70 + "\n")
                for result, count in sorted(summary['pass_results'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {result:15s}: {count:5,} plays\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
        
        logger.info(f"  Saved summary report to {report_path}")
    
    def consolidate_all(self, skip_spatial: bool = False):
        """
        Execute complete consolidation pipeline.
        
        Args:
            skip_spatial: If True, skip spatial features (time-consuming)
        """
        logger.info("=" * 70)
        logger.info("STARTING NFL DATA CONSOLIDATION")
        logger.info("=" * 70)
        start_time = datetime.now()
        
        try:
            # Step 1: Load and consolidate raw data
            logger.info("\n[1/7] Consolidating input data...")
            self.master_input = self.consolidate_input_data()
            self.save_dataset(self.master_input, 'master_input')
            
            logger.info("\n[2/7] Consolidating output data...")
            self.master_output = self.consolidate_output_data()
            self.save_dataset(self.master_output, 'master_output')
            
            logger.info("\n[3/7] Loading supplementary data...")
            self.supplementary = self.load_supplementary_data()
            self.save_dataset(self.supplementary, 'supplementary')
            
            # Step 2: Create derived datasets
            logger.info("\n[4/7] Creating play-level dataset...")
            play_dataset = self.create_play_level_dataset()
            self.save_dataset(play_dataset, 'play_level')
            
            logger.info("\n[5/7] Creating trajectory dataset...")
            trajectory_dataset = self.create_trajectory_dataset()
            self.save_dataset(trajectory_dataset, 'trajectories')
            
            logger.info("\n[6/7] Creating player analysis dataset...")
            player_dataset = self.create_player_analysis_dataset()
            self.save_dataset(player_dataset, 'player_analysis')
            
            # Step 3: Create spatial features (optional - can be slow)
            if not skip_spatial:
                logger.info("\n[7/7] Creating spatial features...")
                spatial_features = self.create_spatial_features()
                self.save_dataset(spatial_features, 'spatial_features')
            else:
                logger.info("\n[7/7] Skipping spatial features (use skip_spatial=False to include)")
            
            # Step 4: Generate summary
            logger.info("\n" + "=" * 70)
            logger.info("GENERATING SUMMARY REPORT")
            logger.info("=" * 70)
            summary = self.generate_summary_report()
            self.save_summary_report(summary)
            
            # Print summary
            logger.info("\nCONSOLIDATION COMPLETE!")
            logger.info("-" * 70)
            logger.info(f"Total Games:          {summary['total_games']:,}")
            logger.info(f"Total Plays:          {summary['total_plays']:,}")
            logger.info(f"Total Players:        {summary['total_players']:,}")
            logger.info(f"Input Frames:         {summary['total_input_frames']:,}")
            logger.info(f"Output Frames:        {summary['total_output_frames']:,}")
            logger.info(f"Players to Predict:   {summary['players_to_predict']:,}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"\nTotal Time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            logger.info(f"Output Location: {self.output_dir.absolute()}")
            logger.info("=" * 70)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error during consolidation: {str(e)}", exc_info=True)
            raise


def main():
    """Main execution function."""
    # Configuration
    DATA_DIR = "114239_nfl_competition_files_published_analytics_final"
    OUTPUT_DIR = "consolidated_data"
    
    # Check if data directory exists
    if not Path(DATA_DIR).exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        logger.error("Please ensure the data directory is in the current working directory.")
        return
    
    # Create consolidator
    consolidator = NFLDataConsolidator(DATA_DIR, OUTPUT_DIR)
    
    # Run consolidation
    # Set skip_spatial=True to skip time-consuming spatial feature calculation
    summary = consolidator.consolidate_all(skip_spatial=False)
    
    logger.info("\nConsolidated datasets are ready for analysis!")
    logger.info(f"Load them with: pd.read_parquet('{OUTPUT_DIR}/[dataset_name].parquet')")


if __name__ == "__main__":
    main()
