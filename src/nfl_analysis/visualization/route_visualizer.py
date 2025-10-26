"""
Route visualization for receiver route analysis.

This module provides visualization tools for analyzing receiver routes,
including route frequency, success rates, and separation metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import seaborn as sns

from . import config


class RouteVisualizer:
    """
    Visualizes receiver route analysis data.

    This class provides methods to visualize route frequency, success rates,
    and separation metrics for receivers.
    """

    def __init__(self, data_dir: str = 'data/consolidated'):
        """
        Initialize the RouteVisualizer.

        Args:
            data_dir: Path to consolidated data directory
        """
        self.data_dir = Path(data_dir)
        self.route_data = None

    def load_route_data(self) -> pd.DataFrame:
        """
        Load route analysis dataset.

        Returns:
            DataFrame with route analysis data
        """
        route_file = self.data_dir / 'route_analysis.parquet'
        if not route_file.exists():
            raise FileNotFoundError(
                f"Route analysis data not found: {route_file}\n"
                "Run consolidation first to generate this dataset."
            )

        self.route_data = pd.read_parquet(route_file)
        return self.route_data

    def plot_route_frequency(
        self,
        player_name: Optional[str] = None,
        nfl_id: Optional[int] = None,
        figsize: Tuple[float, float] = (12, 6)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot route frequency distribution.

        Args:
            player_name: Optional player name to filter by (partial match)
            nfl_id: Optional NFL ID to filter by (exact match)
            figsize: Figure size as (width, height)

        Returns:
            Tuple of (figure, axes)
        """
        if self.route_data is None:
            self.load_route_data()

        # Filter data if player specified
        data = self._filter_player_data(player_name, nfl_id)

        # Count routes
        route_counts = data['route_type'].value_counts().sort_index()

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create bar chart
        bars = ax.bar(
            range(len(route_counts)),
            route_counts.values,
            color=config.OFFENSIVE_COLOR,
            edgecolor='black',
            linewidth=1.5
        )

        # Customize plot
        ax.set_xticks(range(len(route_counts)))
        ax.set_xticklabels(route_counts.index, rotation=45, ha='right')
        ax.set_xlabel('Route Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, route_counts.values)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{int(count)}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        # Title
        if player_name or nfl_id:
            title = f"Route Frequency - {self._get_player_title(player_name, nfl_id, data)}"
        else:
            title = "Route Frequency Distribution - All Players"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        return fig, ax

    def plot_route_success_rate(
        self,
        player_name: Optional[str] = None,
        nfl_id: Optional[int] = None,
        min_attempts: int = 5,
        figsize: Tuple[float, float] = (12, 6)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot success rate (completion percentage) by route type.

        Args:
            player_name: Optional player name to filter by
            nfl_id: Optional NFL ID to filter by
            min_attempts: Minimum number of attempts to include a route type
            figsize: Figure size as (width, height)

        Returns:
            Tuple of (figure, axes)
        """
        if self.route_data is None:
            self.load_route_data()

        # Filter data if player specified
        data = self._filter_player_data(player_name, nfl_id)

        # Calculate success rate by route
        route_success = data.groupby('route_type').agg({
            'is_completion': ['mean', 'count']
        })
        route_success.columns = ['success_rate', 'count']

        # Filter by minimum attempts
        route_success = route_success[route_success['count'] >= min_attempts]
        route_success = route_success.sort_values('success_rate', ascending=False)

        # Convert to percentage
        route_success['success_pct'] = route_success['success_rate'] * 100

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create bar chart with color gradient
        colors = plt.cm.RdYlGn(route_success['success_rate'])
        bars = ax.bar(
            range(len(route_success)),
            route_success['success_pct'],
            color=colors,
            edgecolor='black',
            linewidth=1.5
        )

        # Customize plot
        ax.set_xticks(range(len(route_success)))
        ax.set_xticklabels(route_success.index, rotation=45, ha='right')
        ax.set_xlabel('Route Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Completion Rate (%)', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)

        # Add value labels on bars
        for i, (bar, pct, count) in enumerate(zip(bars, route_success['success_pct'], route_success['count'])):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{pct:.1f}%\n(n={int(count)})',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

        # Add reference line at average
        avg_success = route_success['success_pct'].mean()
        ax.axhline(avg_success, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: {avg_success:.1f}%')
        ax.legend()

        # Title
        if player_name or nfl_id:
            title = f"Route Success Rate - {self._get_player_title(player_name, nfl_id, data)}"
        else:
            title = f"Route Success Rate - All Players (min {min_attempts} attempts)"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        return fig, ax

    def plot_route_separation(
        self,
        player_name: Optional[str] = None,
        nfl_id: Optional[int] = None,
        min_attempts: int = 5,
        figsize: Tuple[float, float] = (12, 6)
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot average separation (distance to ball at landing) by route type.

        Args:
            player_name: Optional player name to filter by
            nfl_id: Optional NFL ID to filter by
            min_attempts: Minimum number of attempts to include a route type
            figsize: Figure size as (width, height)

        Returns:
            Tuple of (figure, axes)
        """
        if self.route_data is None:
            self.load_route_data()

        # Filter data if player specified
        data = self._filter_player_data(player_name, nfl_id)

        # Calculate separation stats by route
        route_separation = data.groupby('route_type').agg({
            'dist_to_ball_at_landing': ['mean', 'std', 'count']
        })
        route_separation.columns = ['mean_sep', 'std_sep', 'count']

        # Filter by minimum attempts
        route_separation = route_separation[route_separation['count'] >= min_attempts]
        route_separation = route_separation.sort_values('mean_sep')

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create bar chart with color scale (lower is better)
        norm_sep = (route_separation['mean_sep'] - route_separation['mean_sep'].min()) / \
                   (route_separation['mean_sep'].max() - route_separation['mean_sep'].min())
        colors = plt.cm.RdYlGn_r(norm_sep)  # Reverse colormap (green = low separation)

        bars = ax.bar(
            range(len(route_separation)),
            route_separation['mean_sep'],
            yerr=route_separation['std_sep'],
            color=colors,
            edgecolor='black',
            linewidth=1.5,
            capsize=5,
            error_kw={'linewidth': 2}
        )

        # Customize plot
        ax.set_xticks(range(len(route_separation)))
        ax.set_xticklabels(route_separation.index, rotation=45, ha='right')
        ax.set_xlabel('Route Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Distance to Ball at Landing (yards)', fontsize=12, fontweight='bold')

        # Add value labels on bars
        for i, (bar, sep, count) in enumerate(zip(bars, route_separation['mean_sep'], route_separation['count'])):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{sep:.2f}y\n(n={int(count)})',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

        # Add reference line at average
        avg_sep = route_separation['mean_sep'].mean()
        ax.axhline(avg_sep, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: {avg_sep:.2f}y')
        ax.legend()

        # Title
        if player_name or nfl_id:
            title = f"Route Separation - {self._get_player_title(player_name, nfl_id, data)}"
        else:
            title = f"Route Separation - All Players (min {min_attempts} attempts)"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        return fig, ax

    def plot_route_dashboard(
        self,
        player_name: Optional[str] = None,
        nfl_id: Optional[int] = None,
        min_attempts: int = 5,
        figsize: Tuple[float, float] = (18, 12)
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Create a comprehensive dashboard with multiple route visualizations.

        Args:
            player_name: Optional player name to filter by
            nfl_id: Optional NFL ID to filter by
            min_attempts: Minimum number of attempts for success/separation plots
            figsize: Figure size as (width, height)

        Returns:
            Tuple of (figure, list of axes)
        """
        if self.route_data is None:
            self.load_route_data()

        # Create subplot layout
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        axes = [
            fig.add_subplot(gs[0, :]),  # Frequency - full width
            fig.add_subplot(gs[1, 0]),  # Success rate
            fig.add_subplot(gs[1, 1]),  # Separation
            fig.add_subplot(gs[2, :])   # Combined scatter plot
        ]

        # Filter data
        data = self._filter_player_data(player_name, nfl_id)

        # 1. Route Frequency
        route_counts = data['route_type'].value_counts().sort_index()
        axes[0].bar(range(len(route_counts)), route_counts.values,
                    color=config.OFFENSIVE_COLOR, edgecolor='black', linewidth=1.5)
        axes[0].set_xticks(range(len(route_counts)))
        axes[0].set_xticklabels(route_counts.index, rotation=45, ha='right')
        axes[0].set_title('Route Frequency', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count')
        axes[0].grid(axis='y', alpha=0.3)

        # 2. Success Rate
        route_success = data.groupby('route_type').agg({
            'is_completion': ['mean', 'count']
        })
        route_success.columns = ['success_rate', 'count']
        route_success = route_success[route_success['count'] >= min_attempts]
        route_success = route_success.sort_values('success_rate', ascending=False)
        route_success['success_pct'] = route_success['success_rate'] * 100

        colors = plt.cm.RdYlGn(route_success['success_rate'])
        axes[1].bar(range(len(route_success)), route_success['success_pct'],
                    color=colors, edgecolor='black', linewidth=1.5)
        axes[1].set_xticks(range(len(route_success)))
        axes[1].set_xticklabels(route_success.index, rotation=45, ha='right')
        axes[1].set_title(f'Success Rate (min {min_attempts} attempts)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Completion %')
        axes[1].set_ylim(0, 100)
        axes[1].grid(axis='y', alpha=0.3)

        # 3. Separation
        route_separation = data.groupby('route_type').agg({
            'dist_to_ball_at_landing': ['mean', 'count']
        })
        route_separation.columns = ['mean_sep', 'count']
        route_separation = route_separation[route_separation['count'] >= min_attempts]
        route_separation = route_separation.sort_values('mean_sep')

        norm_sep = (route_separation['mean_sep'] - route_separation['mean_sep'].min()) / \
                   (route_separation['mean_sep'].max() - route_separation['mean_sep'].min())
        colors = plt.cm.RdYlGn_r(norm_sep)

        axes[2].bar(range(len(route_separation)), route_separation['mean_sep'],
                    color=colors, edgecolor='black', linewidth=1.5)
        axes[2].set_xticks(range(len(route_separation)))
        axes[2].set_xticklabels(route_separation.index, rotation=45, ha='right')
        axes[2].set_title(f'Avg Separation (min {min_attempts} attempts)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Distance (yards)')
        axes[2].grid(axis='y', alpha=0.3)

        # 4. Combined scatter: Success vs Separation
        combined = data.groupby('route_type').agg({
            'is_completion': 'mean',
            'dist_to_ball_at_landing': 'mean',
            'route_type': 'count'
        })
        combined.columns = ['success_rate', 'avg_separation', 'count']
        combined = combined[combined['count'] >= min_attempts]

        scatter = axes[3].scatter(
            combined['avg_separation'],
            combined['success_rate'] * 100,
            s=combined['count'] * 10,
            c=combined['success_rate'],
            cmap='RdYlGn',
            alpha=0.7,
            edgecolor='black',
            linewidth=2
        )

        # Add labels for each point
        for route, row in combined.iterrows():
            axes[3].annotate(
                route,
                (row['avg_separation'], row['success_rate'] * 100),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold'
            )

        axes[3].set_xlabel('Average Separation (yards)', fontsize=11, fontweight='bold')
        axes[3].set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
        axes[3].set_title('Success Rate vs Separation by Route Type', fontsize=12, fontweight='bold')
        axes[3].grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[3])
        cbar.set_label('Success Rate', rotation=270, labelpad=20)

        # Overall title
        if player_name or nfl_id:
            suptitle = f"Route Analysis Dashboard - {self._get_player_title(player_name, nfl_id, data)}"
        else:
            suptitle = "Route Analysis Dashboard - All Players"
        fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)

        return fig, axes

    def _filter_player_data(
        self,
        player_name: Optional[str] = None,
        nfl_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Filter route data by player.

        Args:
            player_name: Player name (partial match)
            nfl_id: NFL ID (exact match)

        Returns:
            Filtered DataFrame
        """
        data = self.route_data.copy()

        if nfl_id is not None:
            data = data[data['nfl_id'] == nfl_id]
        elif player_name is not None:
            data = data[data['player_name'].str.contains(player_name, case=False, na=False)]

        if len(data) == 0:
            raise ValueError(f"No data found for player: {player_name or nfl_id}")

        return data

    def _get_player_title(
        self,
        player_name: Optional[str],
        nfl_id: Optional[int],
        data: pd.DataFrame
    ) -> str:
        """
        Generate player title for plots.

        Args:
            player_name: Player name filter
            nfl_id: NFL ID filter
            data: Filtered data

        Returns:
            Title string
        """
        if nfl_id is not None:
            name = data['player_name'].iloc[0]
            return f"{name} (ID: {nfl_id})"
        elif player_name is not None:
            unique_players = data['player_name'].unique()
            if len(unique_players) == 1:
                return unique_players[0]
            else:
                return f"{len(unique_players)} players matching '{player_name}'"
        return "All Players"

    def plot_player_routes_overlay(
        self,
        game_id: int,
        player_name: Optional[str] = None,
        nfl_id: Optional[int] = None,
        figsize: Tuple[float, float] = (14, 10),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot all routes for a player in a specific game with completion color coding.

        Uses normalized route trajectories to overlay all routes starting from origin (0,0).
        Completed routes shown in green, incomplete routes in red.

        Args:
            game_id: Game ID to visualize
            player_name: Player name (partial match)
            nfl_id: NFL ID (exact match)
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure

        Returns:
            Tuple of (figure, axes)
        """
        # Load route trajectories dataset
        traj_file = self.data_dir / 'route_trajectories.parquet'
        if not traj_file.exists():
            raise FileNotFoundError(
                f"Route trajectories data not found: {traj_file}\n"
                "Run consolidation with create_route_trajectories_dataset() first."
            )

        route_traj = pd.read_parquet(traj_file)

        # Filter by game
        game_routes = route_traj[route_traj['game_id'] == game_id].copy()

        # Filter by player
        if nfl_id is not None:
            player_routes = game_routes[game_routes['nfl_id'] == nfl_id]
        elif player_name is not None:
            player_routes = game_routes[
                game_routes['player_name'].str.contains(player_name, case=False, na=False)
            ]
        else:
            raise ValueError("Must specify either player_name or nfl_id")

        if len(player_routes) == 0:
            raise ValueError(f"No routes found for player in game {game_id}")

        # Get player info
        actual_player_name = player_routes['player_name'].iloc[0]
        actual_nfl_id = player_routes['nfl_id'].iloc[0]

        # Get unique routes (one per play)
        unique_plays = player_routes[['play_id', 'route_type', 'is_completion']].drop_duplicates()

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Count completions
        completions = 0
        incompletions = 0

        # Plot each route
        for _, route_info in unique_plays.iterrows():
            play_id = route_info['play_id']
            route_type = route_info['route_type']
            is_complete = route_info['is_completion']

            # Get trajectory for this route
            route_data = player_routes[player_routes['play_id'] == play_id].sort_values('frame_sequence')

            # Set color based on completion
            if is_complete == 1:
                color = 'green'
                completions += 1
            else:
                color = 'red'
                incompletions += 1

            # Plot the route
            ax.plot(route_data['x_norm'], route_data['y_norm'],
                    color=color, linewidth=2.5, alpha=0.7)

            # Mark the end point
            end_x = route_data['x_norm'].iloc[-1]
            end_y = route_data['y_norm'].iloc[-1]
            ax.scatter(end_x, end_y, color=color, s=120, edgecolor='black',
                       linewidth=1.5, zorder=5, alpha=0.9)

            # Add route type label at end point
            if pd.notna(route_type):
                ax.annotate(route_type, (end_x, end_y),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      alpha=0.7, edgecolor=color))

        # Mark the starting point
        ax.scatter([0], [0], color='blue', s=400, marker='*',
                   edgecolor='black', linewidth=2, zorder=10)

        # Add grid and styling
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.grid(True, alpha=0.2)

        # Labels and title
        ax.set_xlabel('Yards Downfield (normalized)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Yards Across Field (normalized)', fontsize=13, fontweight='bold')

        total_targets = completions + incompletions
        catch_rate = (completions / total_targets * 100) if total_targets > 0 else 0

        title = f"{actual_player_name} - All Routes in Game {game_id}\n"
        title += f"{completions}/{total_targets} Completions ({catch_rate:.0f}%) | "
        title += f"Green = Complete, Red = Incomplete"
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', label=f'Complete ({completions})'),
            Patch(facecolor='red', edgecolor='black', label=f'Incomplete ({incompletions})'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue',
                       markersize=15, markeredgecolor='black', label='Start Position')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)

        # Equal aspect ratio for accurate field representation
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax
