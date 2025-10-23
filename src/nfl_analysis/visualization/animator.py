"""
Play animation for NFL tracking data.

This module provides the PlayAnimator class for creating animated visualizations
of NFL plays using tracking data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from typing import Dict, Optional, Any
from pathlib import Path

from nfl_analysis.io.loader import NFLDataLoader
from nfl_analysis.utils.metrics import calculate_distance
from .field_renderer import FieldRenderer
from . import config


class PlayAnimator:
    """
    Animates NFL plays using tracking data.

    This class handles loading play data, preparing it for animation,
    and creating animated visualizations showing player movements during plays.

    Attributes:
        data_loader: NFLDataLoader instance for accessing data
        field_renderer: FieldRenderer instance for drawing the field
    """

    def __init__(
        self,
        data_dir: str,
        field_renderer: Optional[FieldRenderer] = None
    ):
        """
        Initialize the PlayAnimator.

        Args:
            data_dir: Path to consolidated data directory
            field_renderer: Optional FieldRenderer instance (creates default if None)
        """
        self.data_loader = NFLDataLoader(data_dir)
        self.field_renderer = field_renderer or FieldRenderer()
        self._path_tracers = {}  # Storage for animation line objects

    def load_play_data(
        self,
        game_id: int,
        play_id: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all data for a specific play.

        Args:
            game_id: Game identifier
            play_id: Play identifier

        Returns:
            Dictionary containing 'input', 'output', and 'supplementary' DataFrames
        """
        # Use the data loader to get play data
        play_data = self.data_loader.get_play_data(game_id, play_id)

        # Load full datasets to merge player details
        input_df = self.data_loader.load_input()
        output_df = self.data_loader.load_output()

        # Filter for this specific play
        play_input = play_data['input']
        play_output = play_data['output']
        play_supp = play_data['supplementary']

        # Merge output data with player details from input
        player_details_cols = config.PLAYER_KEYS + config.PLAYER_DETAILS
        player_details = play_input[player_details_cols].drop_duplicates()

        output_with_details = pd.merge(
            play_output,
            player_details,
            on=config.PLAYER_KEYS,
            how='left'
        )

        return {
            'input': play_input,
            'output': output_with_details,
            'supplementary': play_supp
        }

    def prepare_animation_data(
        self,
        play_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Prepare play data for animation by aligning frames and grouping players.

        Args:
            play_data: Dictionary with 'input', 'output', 'supplementary' DataFrames

        Returns:
            Dictionary containing prepared data for animation
        """
        input_df = play_data['input'].copy()
        output_df = play_data['output'].copy()
        supp_df = play_data['supplementary']

        # Align frame IDs so output continues from input
        num_input_frames = input_df['frame_id'].max()
        num_output_frames = output_df['frame_id'].max()

        # Adjust output frame IDs to be continuous with input
        output_df['frame_id'] = output_df['frame_id'] + num_input_frames

        # Combine input and output for complete trajectories
        common_cols = list(set(input_df.columns) & set(output_df.columns))
        union_df = pd.concat([
            input_df[common_cols],
            output_df[common_cols]
        ]).sort_values(by=['nfl_id', 'frame_id'])

        # Group by player
        grouped = union_df.groupby('nfl_id')
        output_grouped = output_df.groupby('nfl_id')

        # Calculate total frames
        total_frames = num_input_frames + num_output_frames

        return {
            'input': input_df,
            'output': output_df,
            'supplementary': supp_df,
            'union': union_df,
            'grouped': grouped,
            'output_grouped': output_grouped,
            'num_frames': total_frames,
            'num_input_frames': num_input_frames,
            'num_output_frames': num_output_frames
        }

    def create_animation(
        self,
        game_id: int,
        play_id: int,
        interval: int = config.DEFAULT_INTERVAL,
        repeat: bool = False
    ) -> FuncAnimation:
        """
        Create an animated visualization of a play.

        Args:
            game_id: Game identifier
            play_id: Play identifier
            interval: Milliseconds between frames
            repeat: Whether to repeat the animation

        Returns:
            matplotlib FuncAnimation object
        """
        # Load and prepare data
        play_data = self.load_play_data(game_id, play_id)
        prepared = self.prepare_animation_data(play_data)

        # Create figure and field
        fig, ax = self.field_renderer.create_figure()
        self.field_renderer.setup_field(ax)

        # Add title from play description
        supp_df = prepared['supplementary']
        if len(supp_df) > 0 and 'play_description' in supp_df.columns:
            self.field_renderer.add_title(ax, supp_df['play_description'].iloc[0])

        # Add info text
        info_text = self._create_info_text(prepared)
        self.field_renderer.add_info_text(fig, info_text)

        # Plot ball landing position
        input_df = prepared['input']
        if len(input_df) > 0 and 'ball_land_x' in input_df.columns:
            ball_x = input_df['ball_land_x'].iloc[0]
            ball_y = input_df['ball_land_y'].iloc[0]
            self.field_renderer.plot_ball_position(ax, ball_x, ball_y)

        # Plot starting positions for all players
        self._plot_starting_positions(ax, prepared)

        # Create animation functions
        def init():
            return self._init_animation(ax, prepared)

        def update(frame):
            return self._update_animation(frame, prepared)

        # Create the animation
        ani = FuncAnimation(
            fig=fig,
            func=update,
            init_func=init,
            frames=prepared['num_frames'],
            interval=interval,
            blit=True,
            repeat=repeat
        )

        # Add legend
        self.field_renderer.add_legend(ax)

        return ani

    def _create_info_text(self, prepared: Dict[str, Any]) -> str:
        """
        Create informational text about the play.

        Args:
            prepared: Prepared animation data

        Returns:
            Formatted info text string
        """
        supp_df = prepared['supplementary']
        input_df = prepared['input']

        if len(supp_df) == 0:
            return ""

        # Extract play information
        info_lines = []

        # Team info
        if 'home_team_abbr' in supp_df.columns:
            home = f"home = {supp_df['home_team_abbr'].iloc[0]}"
            away = f"away = {supp_df['visitor_team_abbr'].iloc[0]}"
            info_lines.append(f"{home} ; {away}")

        # Route and pass information
        route_info = []
        if 'route_of_targeted_receiver' in supp_df.columns:
            route_info.append(f"route = {supp_df['route_of_targeted_receiver'].iloc[0]}")

        if 'pass_length' in supp_df.columns and 'dropback_distance' in supp_df.columns:
            total_length = round(
                supp_df['pass_length'].iloc[0] + supp_df['dropback_distance'].iloc[0],
                4
            )
            route_info.append(f"pass_length = {total_length}")

        if 'yards_gained' in supp_df.columns:
            route_info.append(f"yards_gained = {supp_df['yards_gained'].iloc[0]}")

        if route_info:
            info_lines.append(" ; ".join(route_info))

        # Coverage info
        if 'team_coverage_type' in supp_df.columns:
            info_lines.append(f"coverage = {supp_df['team_coverage_type'].iloc[0]}")

        return "\n".join(info_lines)

    def _plot_starting_positions(self, ax, prepared: Dict[str, Any]) -> None:
        """
        Plot starting positions for all players.

        Args:
            ax: Matplotlib axes
            prepared: Prepared animation data
        """
        grouped = prepared['grouped']
        output_grouped = prepared['output_grouped']

        # Plot initial positions from combined data (pre-pass)
        for player_id, player_df in grouped:
            role = player_df['player_role'].iloc[0]
            if role in config.PLAYER_MARKER_FORMAT:
                color, marker = config.PLAYER_MARKER_FORMAT[role]
                ax.plot(
                    player_df['x'].iloc[0],
                    player_df['y'].iloc[0],
                    color=color,
                    marker=marker
                )

        # Plot starting positions for output phase (post-pass) with dot marker
        for player_id, player_df in output_grouped:
            role = player_df['player_role'].iloc[0]
            if role in config.PLAYER_MARKER_FORMAT:
                color = config.PLAYER_MARKER_FORMAT[role][0]
                ax.plot(
                    player_df['x'].iloc[0],
                    player_df['y'].iloc[0],
                    color=color,
                    marker='.'
                )

    def _init_animation(self, ax, prepared: Dict[str, Any]) -> list:
        """
        Initialize animation by creating path tracers for each player.

        Args:
            ax: Matplotlib axes
            prepared: Prepared animation data

        Returns:
            List of line objects for animation
        """
        self._path_tracers = {}
        grouped = prepared['grouped']

        for player_id, player_df in grouped:
            role = player_df['player_role'].iloc[0]
            if role in config.PLAYER_PATH_FORMAT:
                color, linestyle = config.PLAYER_PATH_FORMAT[role]
                player_name = player_df['player_name'].iloc[0]
                player_weight = player_df['player_weight'].iloc[0]

                line, = ax.plot(
                    player_df['x'].iloc[0],
                    player_df['y'].iloc[0],
                    color=color,
                    linestyle=linestyle,
                    label=f"{player_name}; Wt:{player_weight} lbs"
                )
                self._path_tracers[player_id] = line

        return list(self._path_tracers.values())

    def _update_animation(self, frame: int, prepared: Dict[str, Any]) -> list:
        """
        Update animation for the given frame.

        Args:
            frame: Current frame number
            prepared: Prepared animation data

        Returns:
            List of updated line objects
        """
        grouped = prepared['grouped']

        for player_id, player_df in grouped:
            if player_id in self._path_tracers:
                # Update the path up to current frame
                self._path_tracers[player_id].set_data(
                    player_df['x'].iloc[:frame+1],
                    player_df['y'].iloc[:frame+1]
                )

                # Ensure colors and styles are maintained
                role = player_df['player_role'].iloc[0]
                if role in config.PLAYER_PATH_FORMAT:
                    color, linestyle = config.PLAYER_PATH_FORMAT[role]
                    self._path_tracers[player_id].set_color(color)
                    self._path_tracers[player_id].set_linestyle(linestyle)

        return list(self._path_tracers.values())

    def show(self, animation_obj: FuncAnimation, pause_time: int = config.DEFAULT_PAUSE_TIME) -> None:
        """
        Display the animation.

        Args:
            animation_obj: FuncAnimation object to display
            pause_time: How long to keep the window open (seconds)
        """
        plt.show(block=False)
        plt.pause(pause_time)

    def save(
        self,
        animation_obj: FuncAnimation,
        filename: str,
        fps: int = config.DEFAULT_FPS,
        **kwargs
    ) -> None:
        """
        Save the animation to a file.

        Args:
            animation_obj: FuncAnimation object to save
            filename: Output filename (e.g., 'play.mp4', 'play.gif')
            fps: Frames per second
            **kwargs: Additional arguments passed to save()
        """
        animation_obj.save(filename, fps=fps, **kwargs)
