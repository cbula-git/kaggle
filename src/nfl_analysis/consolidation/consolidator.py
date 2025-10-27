"""
NFL Tracking Data Consolidation
================================
Consolidates weekly NFL tracking data into unified datasets for analysis.
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

    def __init__(self, data_dir: str, output_dir: str = 'data/consolidated'):
        """
        Initialize the consolidator.

        Args:
            data_dir: Path to the directory containing NFL data
            output_dir: Path where consolidated data will be saved
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def create_route_analysis_dataset(self) -> pd.DataFrame:
        """
        Create route analysis dataset for receiver route visualization and success analysis.

        One row per targeted receiver per play with:
        - Route information (type, result)
        - Position at pass release
        - Position at ball landing
        - Distance metrics
        - Route characteristics

        Returns:
            DataFrame with route analysis features
        """
        logger.info("Creating route analysis dataset...")

        # Get targeted receivers at pass release (last input frame)
        targeted_receivers = self.master_input[
            self.master_input['player_role'] == 'Targeted Receiver'
        ].copy()

        # Get last frame (pass release) for each targeted receiver
        route_data = targeted_receivers.loc[
            targeted_receivers.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].idxmax()
        ].copy()

        # Get final post-pass position for each receiver
        final_positions = self.master_output.loc[
            self.master_output.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].idxmax()
        ][['game_id', 'play_id', 'nfl_id', 'x', 'y']].rename(
            columns={'x': 'x_at_landing', 'y': 'y_at_landing'}
        )

        # Count frames in post-pass trajectory
        frames_count = self.master_output.groupby(
            ['game_id', 'play_id', 'nfl_id']
        ).size().reset_index(name='frames_post_pass')

        # Merge everything
        route_dataset = route_data.merge(
            final_positions,
            on=['game_id', 'play_id', 'nfl_id'],
            how='left'
        ).merge(
            frames_count,
            on=['game_id', 'play_id', 'nfl_id'],
            how='left'
        ).merge(
            self.supplementary[['game_id', 'play_id', 'route_of_targeted_receiver',
                               'pass_result', 'team_coverage_type', 'team_coverage_man_zone',
                               'offense_formation', 'receiver_alignment']],
            on=['game_id', 'play_id'],
            how='left'
        )

        # Rename columns for clarity
        route_dataset = route_dataset.rename(columns={
            'x': 'x_at_release',
            'y': 'y_at_release',
            's': 'speed_at_release',
            'a': 'acceleration_at_release',
            'dir': 'direction_at_release',
            'o': 'orientation_at_release',
            'route_of_targeted_receiver': 'route_type'
        })

        # Calculate distance to ball at landing
        route_dataset['dist_to_ball_at_landing'] = np.sqrt(
            (route_dataset['x_at_landing'] - route_dataset['ball_land_x'])**2 +
            (route_dataset['y_at_landing'] - route_dataset['ball_land_y'])**2
        )

        # Calculate route metrics (raw, before normalization)
        route_dataset['route_depth'] = route_dataset['x_at_landing'] - route_dataset['x_at_release']
        route_dataset['route_width'] = route_dataset['y_at_landing'] - route_dataset['y_at_release']

        # Normalize route metrics based on play direction
        # When play_direction is 'left', offense moves toward lower X values
        # So we need to flip the sign to get positive depth for downfield routes
        left_plays = route_dataset['play_direction'] == 'left'
        route_dataset.loc[left_plays, 'route_depth'] = -route_dataset.loc[left_plays, 'route_depth']
        route_dataset.loc[left_plays, 'route_width'] = -route_dataset.loc[left_plays, 'route_width']

        # Calculate total route distance (using normalized depth/width)
        route_dataset['route_distance'] = np.sqrt(
            route_dataset['route_depth']**2 + route_dataset['route_width']**2
        )

        # Calculate distance to ball at release
        route_dataset['dist_to_ball_at_release'] = np.sqrt(
            (route_dataset['x_at_release'] - route_dataset['ball_land_x'])**2 +
            (route_dataset['y_at_release'] - route_dataset['ball_land_y'])**2
        )

        # Success indicator (completion)
        route_dataset['is_completion'] = (route_dataset['pass_result'] == 'C').astype(int)

        logger.info(f"  Created dataset with {len(route_dataset):,} routes")
        logger.info(f"  Route types: {route_dataset['route_type'].nunique()}")
        logger.info(f"  Completion rate: {route_dataset['is_completion'].mean()*100:.1f}%")

        return route_dataset

    def create_route_trajectories_dataset(self) -> pd.DataFrame:
        """
        Create normalized route trajectories for visualization and comparison.

        Normalizes all routes to start at origin (0, 0) with play direction standardized,
        allowing direct comparison and overlay of route shapes across players and plays.

        One row per frame per targeted receiver route with:
        - Normalized coordinates (relative to route start)
        - Original coordinates preserved
        - Frame sequence within route
        - Distance traveled metrics
        - Route metadata for filtering

        Returns:
            DataFrame with normalized route trajectories
        """
        logger.info("Creating route trajectories dataset...")
        logger.info("  This will normalize all targeted receiver routes to common coordinates...")

        # First, get the list of targeted receiver routes with play direction
        targeted_routes = self.master_input[
            self.master_input['player_role'] == 'Targeted Receiver'
        ][['game_id', 'play_id', 'nfl_id', 'play_direction']].drop_duplicates()

        # Merge with supplementary data to get route type and other metadata
        route_metadata = targeted_routes.merge(
            self.supplementary[['game_id', 'play_id', 'route_of_targeted_receiver',
                               'pass_result', 'team_coverage_man_zone',
                               'offense_formation']],
            on=['game_id', 'play_id'],
            how='left'
        )
        route_metadata = route_metadata.rename(columns={
            'route_of_targeted_receiver': 'route_type'
        })

        logger.info(f"  Processing {len(route_metadata):,} targeted receiver routes...")

        # Now get trajectories for each route
        all_trajectories = []

        for idx, route_info in route_metadata.iterrows():
            if idx % 1000 == 0 and idx > 0:
                logger.info(f"    Processed {idx:,} / {len(route_metadata):,} routes...")

            game_id = route_info['game_id']
            play_id = route_info['play_id']
            nfl_id = route_info['nfl_id']
            route_type = route_info['route_type']
            play_direction = route_info['play_direction']
            pass_result = route_info['pass_result']

            # Get pre-pass trajectory
            pre_pass = self.master_input[
                (self.master_input['game_id'] == game_id) &
                (self.master_input['play_id'] == play_id) &
                (self.master_input['nfl_id'] == nfl_id)
            ][['frame_id', 'x', 'y', 's', 'a', 'dir', 'o', 'player_name', 'player_position']].copy().sort_values('frame_id')
            pre_pass['phase'] = 'pre_pass'

            # Get post-pass trajectory (need to adjust frame_ids to be continuous)
            post_pass = self.master_output[
                (self.master_output['game_id'] == game_id) &
                (self.master_output['play_id'] == play_id) &
                (self.master_output['nfl_id'] == nfl_id)
            ][['frame_id', 'x', 'y']].copy().sort_values('frame_id')

            # Adjust post-pass frame IDs to be continuous with pre-pass
            if len(pre_pass) > 0 and len(post_pass) > 0:
                max_pre_pass_frame = pre_pass['frame_id'].max()
                post_pass['frame_id'] = post_pass['frame_id'] + max_pre_pass_frame

            post_pass['phase'] = 'post_pass'

            # Combine trajectories (they're already sorted within phase, now concat in order)
            full_traj = pd.concat([pre_pass, post_pass], ignore_index=True)

            if len(full_traj) == 0:
                continue

            # Get starting position (first frame)
            start_x = full_traj.iloc[0]['x']
            start_y = full_traj.iloc[0]['y']

            # Normalize coordinates to start at origin
            full_traj['x_norm'] = full_traj['x'] - start_x
            full_traj['y_norm'] = full_traj['y'] - start_y

            # Apply play direction normalization
            # If play goes left, flip coordinates so all routes go "right"
            if play_direction == 'left':
                full_traj['x_norm'] = -full_traj['x_norm']
                full_traj['y_norm'] = -full_traj['y_norm']

            # Add frame sequence (0-indexed within this route)
            full_traj['frame_sequence'] = range(len(full_traj))

            # Calculate cumulative distance traveled
            full_traj['dist_from_start'] = np.sqrt(
                full_traj['x_norm']**2 + full_traj['y_norm']**2
            )

            # Add route metadata
            full_traj['game_id'] = game_id
            full_traj['play_id'] = play_id
            full_traj['nfl_id'] = nfl_id
            full_traj['route_type'] = route_type
            full_traj['pass_result'] = pass_result
            full_traj['play_direction'] = play_direction
            full_traj['is_completion'] = 1 if pass_result == 'C' else 0
            full_traj['team_coverage_man_zone'] = route_info['team_coverage_man_zone']
            full_traj['offense_formation'] = route_info['offense_formation']

            # Preserve original coordinates
            full_traj = full_traj.rename(columns={'x': 'x_original', 'y': 'y_original'})

            # Get player info from first frame (where it exists)
            if 'player_name' in full_traj.columns:
                player_name = full_traj['player_name'].iloc[0] if pd.notna(full_traj['player_name'].iloc[0]) else None
                player_position = full_traj['player_position'].iloc[0] if pd.notna(full_traj['player_position'].iloc[0]) else None
                full_traj['player_name'] = player_name
                full_traj['player_position'] = player_position

            all_trajectories.append(full_traj)

        # Combine all routes
        route_trajectories = pd.concat(all_trajectories, ignore_index=True)

        # Select and order columns
        columns_order = [
            'game_id', 'play_id', 'nfl_id', 'frame_id', 'frame_sequence',
            'player_name', 'player_position', 'route_type', 'pass_result', 'is_completion',
            'play_direction', 'phase',
            'x_norm', 'y_norm', 'x_original', 'y_original',
            'dist_from_start', 's', 'a', 'dir', 'o',
            'team_coverage_man_zone', 'offense_formation'
        ]

        # Filter to available columns
        columns_order = [col for col in columns_order if col in route_trajectories.columns]
        route_trajectories = route_trajectories[columns_order]

        logger.info(f"  Created dataset with {len(route_trajectories):,} frames")
        logger.info(f"  Total routes: {route_trajectories.groupby(['game_id', 'play_id', 'nfl_id']).ngroups:,}")
        logger.info(f"  Avg frames per route: {len(route_trajectories) / route_trajectories.groupby(['game_id', 'play_id', 'nfl_id']).ngroups:.1f}")

        return route_trajectories

    def _calculate_zone_boundaries(self, LOS: float, play_direction: str) -> Dict:
        """
        Calculate 15-zone boundaries relative to line of scrimmage.

        Creates a 3x5 grid:
        - 3 depth zones: shallow (0-5), intermediate (5-15), deep (15+) yards past LOS
        - 5 lateral zones based on exact NFL hash marks

        Args:
            LOS: Line of scrimmage (absolute yardline number)
            play_direction: Direction of play ('left' or 'right')

        Returns:
            Dictionary mapping zone_id to {x_min, x_max, y_min, y_max, center_x, center_y, area}
        """
        # Field dimensions
        FIELD_WIDTH = 53.3
        LEFT_HASH = 23.36  # Distance from left sideline to left hash
        RIGHT_HASH = 29.64  # Distance from left sideline to right hash (53.3 - 23.36)

        # Lateral zone boundaries (y-coordinates, constant across all plays)
        # Based on hash marks
        HASH_BUFFER = 5.0  # Yards around hash marks for hash zones

        lateral_zones = {
            'far_left': {
                'y_min': 0.0,
                'y_max': LEFT_HASH - HASH_BUFFER,  # 18.36
                'y_center': (0.0 + LEFT_HASH - HASH_BUFFER) / 2  # ~9.18
            },
            'left_hash': {
                'y_min': LEFT_HASH - HASH_BUFFER,  # 18.36
                'y_max': LEFT_HASH,  # 23.36
                'y_center': LEFT_HASH - HASH_BUFFER / 2  # ~20.86
            },
            'middle': {
                'y_min': LEFT_HASH,  # 23.36
                'y_max': RIGHT_HASH,  # 29.64
                'y_center': (LEFT_HASH + RIGHT_HASH) / 2  # 26.5
            },
            'right_hash': {
                'y_min': RIGHT_HASH,  # 29.64
                'y_max': RIGHT_HASH + HASH_BUFFER,  # 35.64
                'y_center': RIGHT_HASH + HASH_BUFFER / 2  # ~32.14
            },
            'far_right': {
                'y_min': RIGHT_HASH + HASH_BUFFER,  # 35.64
                'y_max': FIELD_WIDTH,  # 53.3
                'y_center': (RIGHT_HASH + HASH_BUFFER + FIELD_WIDTH) / 2  # ~44.47
            }
        }

        # Depth zone boundaries (x-coordinates, relative to LOS)
        # Depends on play direction
        if play_direction == 'right':
            # Play moving toward higher X values
            depth_zones = {
                'shallow': {
                    'x_min': LOS,
                    'x_max': LOS + 5,
                    'x_center': LOS + 2.5
                },
                'intermediate': {
                    'x_min': LOS + 5,
                    'x_max': LOS + 15,
                    'x_center': LOS + 10
                },
                'deep': {
                    'x_min': LOS + 15,
                    'x_max': 120.0,  # End zone
                    'x_center': LOS + 25  # Approximate center for deep zone
                }
            }
        else:  # play_direction == 'left'
            # Play moving toward lower X values
            depth_zones = {
                'shallow': {
                    'x_min': LOS - 5,
                    'x_max': LOS,
                    'x_center': LOS - 2.5
                },
                'intermediate': {
                    'x_min': LOS - 15,
                    'x_max': LOS - 5,
                    'x_center': LOS - 10
                },
                'deep': {
                    'x_min': 0.0,  # End zone
                    'x_max': LOS - 15,
                    'x_center': LOS - 25  # Approximate center for deep zone
                }
            }

        # Combine depth and lateral to create 15 zones
        zone_boundaries = {}
        for depth_name, depth_bounds in depth_zones.items():
            for lateral_name, lateral_bounds in lateral_zones.items():
                zone_id = f"{depth_name}_{lateral_name}"

                x_min = depth_bounds['x_min']
                x_max = depth_bounds['x_max']
                y_min = lateral_bounds['y_min']
                y_max = lateral_bounds['y_max']

                # Calculate area
                width = x_max - x_min
                height = y_max - y_min
                area = width * height

                zone_boundaries[zone_id] = {
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'center_x': depth_bounds['x_center'],
                    'center_y': lateral_bounds['y_center'],
                    'area': area,
                    'depth_category': depth_name,
                    'lateral_category': lateral_name
                }

        return zone_boundaries

    def _assign_player_to_zone(self, x: float, y: float, zone_boundaries: Dict,
                                find_nearest: bool = False) -> str:
        """
        Assign player coordinates to a zone.

        Args:
            x: Player x-coordinate
            y: Player y-coordinate
            zone_boundaries: Dictionary of zone boundaries from _calculate_zone_boundaries
            find_nearest: If True and coordinates are out of bounds, find nearest zone by
                         Euclidean distance to zone centers. Used for ball landings that go
                         out of bounds (e.g., sideline passes). Approximately 3.3% of plays
                         have out-of-bounds ball landings (Y < 0 or Y > 53.3).

        Returns:
            Zone ID string (e.g., "shallow_middle") or None if out of bounds and find_nearest=False
        """
        # Check each zone for exact match
        for zone_id, bounds in zone_boundaries.items():
            if (bounds['x_min'] <= x < bounds['x_max'] and
                bounds['y_min'] <= y < bounds['y_max']):
                return zone_id

        # Player is outside all zones (e.g., far sideline, end zone)
        if find_nearest:
            # Find the nearest zone by calculating distance to each zone's center
            min_dist = float('inf')
            nearest_zone = None

            for zone_id, bounds in zone_boundaries.items():
                zone_center_x = bounds['center_x']
                zone_center_y = bounds['center_y']
                dist = np.sqrt((x - zone_center_x)**2 + (y - zone_center_y)**2)

                if dist < min_dist:
                    min_dist = dist
                    nearest_zone = zone_id

            return nearest_zone

        return None

    def _calculate_zone_metrics(self, frame_data: pd.DataFrame, zone_boundaries: Dict,
                                ball_land_x: float, ball_land_y: float) -> pd.DataFrame:
        """
        Calculate vulnerability metrics for all 15 zones in a single frame.

        Args:
            frame_data: DataFrame with all players in this frame
            zone_boundaries: Dictionary of zone boundaries
            ball_land_x: Ball landing x-coordinate
            ball_land_y: Ball landing y-coordinate

        Returns:
            DataFrame with one row per zone (15 rows total)
        """
        # Assign each player to a zone
        frame_data = frame_data.copy()
        frame_data['zone_id'] = frame_data.apply(
            lambda row: self._assign_player_to_zone(row['x'], row['y'], zone_boundaries),
            axis=1
        )

        # Separate defenders and receivers
        defenders = frame_data[frame_data['player_side'] == 'Defense'].copy()
        receivers = frame_data[frame_data['player_role'].isin([
            'Targeted Receiver', 'Other Route Runner'
        ])].copy()

        # Sanity check: there should always be defenders on the field
        if len(defenders) == 0:
            logger.warning(f"WARNING: No defenders found in frame! This indicates a data issue.")
            logger.warning(f"  Available player_side values: {frame_data['player_side'].unique()}")

        # Calculate metrics for each zone
        zone_metrics = []

        for zone_id, bounds in zone_boundaries.items():
            # Defenders in this zone
            zone_defenders = defenders[defenders['zone_id'] == zone_id]
            defender_count = len(zone_defenders)
            defender_ids = ','.join(zone_defenders['nfl_id'].astype(str).tolist()) if defender_count > 0 else ''

            # Receivers in this zone
            zone_receivers = receivers[receivers['zone_id'] == zone_id]
            receiver_count = len(zone_receivers)
            route_runner_count = receiver_count  # All receivers we filtered are route runners
            receiver_ids = ','.join(zone_receivers['nfl_id'].astype(str).tolist()) if receiver_count > 0 else ''

            # Calculate nearest defender distance to zone center
            zone_center_x = bounds['center_x']
            zone_center_y = bounds['center_y']

            if defender_count > 0:
                # Distance from each defender to zone center
                distances = np.sqrt(
                    (zone_defenders['x'] - zone_center_x)**2 +
                    (zone_defenders['y'] - zone_center_y)**2
                )
                nearest_defender_dist = distances.min()
            else:
                # No defenders in zone - use large distance
                # Calculate from all defenders on field
                if len(defenders) > 0:
                    distances = np.sqrt(
                        (defenders['x'] - zone_center_x)**2 +
                        (defenders['y'] - zone_center_y)**2
                    )
                    nearest_defender_dist = distances.min()
                else:
                    nearest_defender_dist = 50.0  # Maximum field dimension

            # Coverage density (defenders per zone area)
            coverage_density = defender_count / (bounds['area'] / 100) if bounds['area'] > 0 else 0

            # Vulnerability score: higher = more vulnerable
            # Combines distance and count - empty zones with far defenders = most vulnerable
            zone_void_score = nearest_defender_dist * (1.0 / (defender_count + 0.1))

            # Check if this is the target zone
            # Use find_nearest=True to handle out-of-bounds ball landings (e.g., sidelines)
            target_zone_id = self._assign_player_to_zone(ball_land_x, ball_land_y, zone_boundaries,
                                                          find_nearest=True)
            is_target_zone = (zone_id == target_zone_id)

            zone_metrics.append({
                'zone_id': zone_id,
                'zone_depth_category': bounds['depth_category'],
                'zone_lateral_category': bounds['lateral_category'],
                'zone_center_x': zone_center_x,
                'zone_center_y': zone_center_y,
                'zone_area': bounds['area'],
                'defender_count': defender_count,
                'defender_ids': defender_ids,
                'nearest_defender_dist': nearest_defender_dist,
                'coverage_density': coverage_density,
                'zone_void_score': zone_void_score,
                'receiver_count': receiver_count,
                'route_runner_count': route_runner_count,
                'receiver_ids': receiver_ids,
                'is_target_zone': is_target_zone
            })

        return pd.DataFrame(zone_metrics)

    def create_zone_vulnerability_timeseries(self) -> pd.DataFrame:
        """
        Create zone vulnerability timeseries dataset.

        Analyzes defensive zone occupation and vulnerability across 15 LOS-relative zones
        for every frame of every play. Tracks how zones evolve from pre-snap through
        route development to post-throw.

        Returns:
            DataFrame with zone vulnerability metrics over time
            One row per zone per frame per play (~5-10M rows)
        """
        logger.info("Creating zone vulnerability timeseries dataset...")
        logger.info("  Analyzing 15-zone grid across all plays and frames...")

        # Get list of all plays
        plays = self.master_input[['game_id', 'play_id', 'absolute_yardline_number',
                                    'play_direction', 'ball_land_x', 'ball_land_y',
                                    'week']].drop_duplicates()

        logger.info(f"  Processing {len(plays):,} plays...")

        all_zone_data = []
        plays_processed = 0

        for _, play_info in plays.iterrows():
            if plays_processed % 1000 == 0 and plays_processed > 0:
                logger.info(f"    Processed {plays_processed:,} / {len(plays):,} plays...")

            game_id = play_info['game_id']
            play_id = play_info['play_id']
            LOS = play_info['absolute_yardline_number']
            play_direction = play_info['play_direction']
            ball_land_x = play_info['ball_land_x']
            ball_land_y = play_info['ball_land_y']
            week = play_info['week']

            # Calculate zone boundaries for this play
            zone_boundaries = self._calculate_zone_boundaries(LOS, play_direction)

            # Get all frames for this play (input only - pre-snap through throw)
            play_frames = self.master_input[
                (self.master_input['game_id'] == game_id) &
                (self.master_input['play_id'] == play_id)
            ].copy()

            if len(play_frames) == 0:
                continue

            # Get throw frame (last input frame)
            throw_frame = play_frames['frame_id'].max()

            # Process each frame
            for frame_id in play_frames['frame_id'].unique():
                frame_data = play_frames[play_frames['frame_id'] == frame_id]

                # Determine phase
                if frame_id == 1:
                    phase = 'pre_snap'
                elif frame_id < throw_frame:
                    phase = 'route_development'
                elif frame_id == throw_frame:
                    phase = 'at_throw'
                else:
                    phase = 'post_throw'  # Shouldn't occur in input data

                # Calculate zone metrics for this frame
                zone_metrics_df = self._calculate_zone_metrics(
                    frame_data, zone_boundaries, ball_land_x, ball_land_y
                )

                # Add play context
                zone_metrics_df['game_id'] = game_id
                zone_metrics_df['play_id'] = play_id
                zone_metrics_df['frame_id'] = frame_id
                zone_metrics_df['LOS_position'] = LOS
                zone_metrics_df['play_direction'] = play_direction
                zone_metrics_df['phase'] = phase
                zone_metrics_df['ball_land_x'] = ball_land_x
                zone_metrics_df['ball_land_y'] = ball_land_y
                zone_metrics_df['week'] = week

                all_zone_data.append(zone_metrics_df)

            plays_processed += 1

        # Combine all zone data
        logger.info("  Combining zone data from all plays...")
        zone_timeseries = pd.concat(all_zone_data, ignore_index=True)

        # Reorder columns for clarity
        columns_order = [
            'game_id', 'play_id', 'frame_id', 'week',
            'zone_id', 'zone_depth_category', 'zone_lateral_category',
            'zone_center_x', 'zone_center_y', 'zone_area',
            'LOS_position', 'play_direction', 'phase',
            'ball_land_x', 'ball_land_y',
            'defender_count', 'defender_ids', 'nearest_defender_dist',
            'coverage_density', 'zone_void_score',
            'receiver_count', 'route_runner_count', 'receiver_ids',
            'is_target_zone'
        ]

        zone_timeseries = zone_timeseries[columns_order]

        # Summary statistics
        total_zones = len(zone_timeseries)
        total_plays = zone_timeseries[['game_id', 'play_id']].drop_duplicates().shape[0]
        avg_frames_per_play = zone_timeseries.groupby(['game_id', 'play_id'])['frame_id'].nunique().mean()
        avg_void_score = zone_timeseries['zone_void_score'].mean()
        max_void_score = zone_timeseries['zone_void_score'].max()

        logger.info(f"  Created dataset with {total_zones:,} zone-frame records")
        logger.info(f"  Total plays: {total_plays:,}")
        logger.info(f"  Avg frames per play: {avg_frames_per_play:.1f}")
        logger.info(f"  Avg vulnerability score: {avg_void_score:.2f}")
        logger.info(f"  Max vulnerability score: {max_void_score:.2f}")

        # Most vulnerable zones
        avg_by_zone = zone_timeseries.groupby('zone_id')['zone_void_score'].mean().sort_values(ascending=False)
        logger.info(f"  Top 3 most vulnerable zones (avg):")
        for zone, score in avg_by_zone.head(3).items():
            logger.info(f"    {zone:20s}: {score:.2f}")

        return zone_timeseries

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
            logger.info("\n[1/10] Consolidating input data...")
            self.master_input = self.consolidate_input_data()
            self.save_dataset(self.master_input, 'master_input')

            logger.info("\n[2/10] Consolidating output data...")
            self.master_output = self.consolidate_output_data()
            self.save_dataset(self.master_output, 'master_output')

            logger.info("\n[3/10] Loading supplementary data...")
            self.supplementary = self.load_supplementary_data()
            self.save_dataset(self.supplementary, 'supplementary')

            # Step 2: Create derived datasets
            logger.info("\n[4/10] Creating play-level dataset...")
            play_dataset = self.create_play_level_dataset()
            self.save_dataset(play_dataset, 'play_level')

            logger.info("\n[5/10] Creating trajectory dataset...")
            trajectory_dataset = self.create_trajectory_dataset()
            self.save_dataset(trajectory_dataset, 'trajectories')

            logger.info("\n[6/10] Creating player analysis dataset...")
            player_dataset = self.create_player_analysis_dataset()
            self.save_dataset(player_dataset, 'player_analysis')

            logger.info("\n[7/10] Creating route analysis dataset...")
            route_dataset = self.create_route_analysis_dataset()
            self.save_dataset(route_dataset, 'route_analysis')

            logger.info("\n[8/10] Creating route trajectories dataset...")
            route_trajectories = self.create_route_trajectories_dataset()
            self.save_dataset(route_trajectories, 'route_trajectories')

            logger.info("\n[9/10] Creating zone vulnerability timeseries dataset...")
            zone_vulnerability = self.create_zone_vulnerability_timeseries()
            self.save_dataset(zone_vulnerability, 'zone_vulnerability_timeseries')

            # Step 3: Create spatial features (optional - can be slow)
            if not skip_spatial:
                logger.info("\n[10/10] Creating spatial features...")
                spatial_features = self.create_spatial_features()
                self.save_dataset(spatial_features, 'spatial_features')
            else:
                logger.info("\n[10/10] Skipping spatial features (use skip_spatial=False to include)")

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
