"""
Zone Vulnerability Visualization
==================================
Visualization tools for analyzing defensive zone vulnerability over time.

This module provides tools to visualize:
- Zone occupation heatmaps
- Vulnerability evolution over time
- Defender positioning across 15 LOS-relative zones
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import seaborn as sns

from . import config


class ZoneVulnerabilityVisualizer:
    """
    Visualizes zone vulnerability and defensive positioning.

    Provides methods to visualize zone occupation, vulnerability scores,
    and defensive evolution across 15 LOS-relative zones.
    """

    def __init__(self, data_dir: str = 'data/consolidated'):
        """
        Initialize the ZoneVulnerabilityVisualizer.

        Args:
            data_dir: Path to consolidated data directory
        """
        self.data_dir = Path(data_dir)
        self.zone_data = None

    def load_zone_data(self) -> pd.DataFrame:
        """
        Load zone vulnerability timeseries dataset.

        Returns:
            DataFrame with zone vulnerability data
        """
        zone_file = self.data_dir / 'zone_vulnerability_timeseries.parquet'
        if not zone_file.exists():
            raise FileNotFoundError(
                f"Zone vulnerability data not found: {zone_file}\n"
                "Run consolidation first to generate this dataset."
            )

        self.zone_data = pd.read_parquet(zone_file)
        return self.zone_data

    def plot_zone_grid(
        self,
        game_id: int,
        play_id: int,
        frame_id: int,
        metric: str = 'zone_void_score',
        figsize: Tuple[float, float] = (14, 10),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot 15-zone heatmap for a single frame.

        Creates a grid visualization showing the specified metric across
        all 15 zones for one moment in time.

        Args:
            game_id: Game ID to visualize
            play_id: Play ID to visualize
            frame_id: Frame ID to visualize
            metric: Metric to display ('zone_void_score', 'defender_count', 'nearest_defender_dist')
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure

        Returns:
            Tuple of (figure, axes)
        """
        if self.zone_data is None:
            self.load_zone_data()

        # Filter to specific frame
        frame_zones = self.zone_data[
            (self.zone_data['game_id'] == game_id) &
            (self.zone_data['play_id'] == play_id) &
            (self.zone_data['frame_id'] == frame_id)
        ].copy()

        if len(frame_zones) == 0:
            raise ValueError(f"No data found for game {game_id}, play {play_id}, frame {frame_id}")

        # Create pivot table for heatmap
        # Rows: depth categories (shallow, intermediate, deep)
        # Columns: lateral categories (far_left, left_hash, middle, right_hash, far_right)

        # Define order
        depth_order = ['shallow', 'intermediate', 'deep']
        lateral_order = ['far_left', 'left_hash', 'middle', 'right_hash', 'far_right']

        # Pivot the data
        heatmap_data = frame_zones.pivot_table(
            index='zone_depth_category',
            columns='zone_lateral_category',
            values=metric,
            aggfunc='first'
        )

        # Reorder to match field orientation
        heatmap_data = heatmap_data.reindex(index=depth_order, columns=lateral_order)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        cmap = 'YlOrRd' if metric == 'zone_void_score' else 'Blues'
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            cbar_kws={'label': metric.replace('_', ' ').title()},
            linewidths=2,
            linecolor='black',
            ax=ax,
            vmin=0 if metric == 'defender_count' else None
        )

        # Get play info
        play_direction = frame_zones['play_direction'].iloc[0]
        phase = frame_zones['phase'].iloc[0]
        LOS = frame_zones['LOS_position'].iloc[0]

        # Title
        title = f"Zone {metric.replace('_', ' ').title()}\n"
        title += f"Game {game_id}, Play {play_id}, Frame {frame_id}\n"
        title += f"Phase: {phase.replace('_', ' ').title()}, "
        title += f"Direction: {play_direction}, LOS: {LOS:.1f}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Axis labels
        ax.set_xlabel('Lateral Zone (sideline to sideline)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Depth Zone (LOS to deep)', fontsize=12, fontweight='bold')

        # Rotate x labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    def plot_zone_evolution(
        self,
        game_id: int,
        play_id: int,
        zone_id: str,
        figsize: Tuple[float, float] = (12, 6),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot vulnerability over time for one zone.

        Shows how a single zone's vulnerability score changes from pre-snap
        through route development to throw.

        Args:
            game_id: Game ID to visualize
            play_id: Play ID to visualize
            zone_id: Zone to track (e.g., "deep_middle", "shallow_left_hash")
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure

        Returns:
            Tuple of (figure, axes)
        """
        if self.zone_data is None:
            self.load_zone_data()

        # Filter to specific zone across all frames
        zone_evolution = self.zone_data[
            (self.zone_data['game_id'] == game_id) &
            (self.zone_data['play_id'] == play_id) &
            (self.zone_data['zone_id'] == zone_id)
        ].sort_values('frame_id').copy()

        if len(zone_evolution) == 0:
            raise ValueError(f"No data found for zone {zone_id} in game {game_id}, play {play_id}")

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot 1: Vulnerability score
        ax1.plot(zone_evolution['frame_id'], zone_evolution['zone_void_score'],
                 marker='o', linewidth=2, markersize=8, color='red', label='Void Score')
        ax1.set_ylabel('Vulnerability Score', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # Highlight phases
        phases = zone_evolution.groupby('phase')['frame_id'].agg(['min', 'max'])
        for phase_name, (min_frame, max_frame) in phases.iterrows():
            if phase_name == 'pre_snap':
                color = 'lightblue'
            elif phase_name == 'route_development':
                color = 'lightyellow'
            else:  # at_throw
                color = 'lightcoral'
            ax1.axvspan(min_frame - 0.5, max_frame + 0.5, alpha=0.2, color=color)

        # Plot 2: Defender count
        ax2.plot(zone_evolution['frame_id'], zone_evolution['defender_count'],
                 marker='s', linewidth=2, markersize=8, color='blue', label='Defenders')
        ax2.set_xlabel('Frame ID', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Defender Count', fontsize=11, fontweight='bold')
        ax2.set_yticks(range(int(zone_evolution['defender_count'].max()) + 2))
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        # Highlight is_target_zone
        if zone_evolution['is_target_zone'].any():
            ax1.axhline(y=zone_evolution.loc[zone_evolution['is_target_zone'], 'zone_void_score'].iloc[0],
                        color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target Zone')
            ax1.legend(loc='upper right')

        # Title
        title = f"Zone Evolution: {zone_id.replace('_', ' ').title()}\n"
        title += f"Game {game_id}, Play {play_id}"
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, (ax1, ax2)

    def plot_target_zone_comparison(
        self,
        game_ids: List[int],
        play_ids: List[int],
        figsize: Tuple[float, float] = (14, 8),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Compare vulnerability of target zones across multiple plays.

        Args:
            game_ids: List of game IDs
            play_ids: List of play IDs (must match length of game_ids)
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Tuple of (figure, axes)
        """
        if len(game_ids) != len(play_ids):
            raise ValueError("game_ids and play_ids must have same length")

        if self.zone_data is None:
            self.load_zone_data()

        target_data = []

        for game_id, play_id in zip(game_ids, play_ids):
            # Get target zone at throw
            play_zones = self.zone_data[
                (self.zone_data['game_id'] == game_id) &
                (self.zone_data['play_id'] == play_id) &
                (self.zone_data['phase'] == 'at_throw') &
                (self.zone_data['is_target_zone'] == True)
            ]

            if len(play_zones) > 0:
                target_data.append({
                    'game_id': game_id,
                    'play_id': play_id,
                    'zone_id': play_zones['zone_id'].iloc[0],
                    'void_score': play_zones['zone_void_score'].iloc[0],
                    'defender_count': play_zones['defender_count'].iloc[0],
                    'nearest_defender_dist': play_zones['nearest_defender_dist'].iloc[0]
                })

        target_df = pd.DataFrame(target_data)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Void score by play
        x_pos = range(len(target_df))
        colors = ['green' if score < 20 else 'orange' if score < 40 else 'red'
                  for score in target_df['void_score']]

        ax1.bar(x_pos, target_df['void_score'], color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Play Index', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Vulnerability Score', fontsize=11, fontweight='bold')
        ax1.set_title('Target Zone Vulnerability at Throw', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Zone distribution
        zone_counts = target_df['zone_id'].value_counts()
        ax2.barh(range(len(zone_counts)), zone_counts.values, color='steelblue', edgecolor='black')
        ax2.set_yticks(range(len(zone_counts)))
        ax2.set_yticklabels([z.replace('_', ' ').title() for z in zone_counts.index])
        ax2.set_xlabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Most Targeted Zones', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, (ax1, ax2)
