"""
Plotting utilities for coverage visualization.

This module provides reusable helper functions for plotting players,
zones, and field elements in coverage visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.axes import Axes
from typing import Dict, List, Optional, Tuple

from . import config
from ..animation.field_renderer import FieldRenderer


class CoveragePlotHelper:
    """
    Helper class for coverage visualization plotting operations.

    This class provides reusable methods for plotting common elements
    in coverage visualizations, reducing code duplication.
    """

    def __init__(self, field_renderer: Optional[FieldRenderer] = None):
        """
        Initialize the plot helper.

        Args:
            field_renderer: Optional FieldRenderer instance for field drawing
        """
        self.field_renderer = field_renderer or FieldRenderer()

    def plot_offensive_players(
        self,
        ax: Axes,
        players_df: pd.DataFrame,
        show_labels: bool = True,
        label_offset: float = config.LABEL_OFFSET_VERTICAL,
        marker_size: int = config.MARKER_SIZE_MEDIUM
    ) -> None:
        """
        Plot offensive players on the field.

        Args:
            ax: Matplotlib axes to plot on
            players_df: DataFrame with player data (must have x, y, player_position, player_name)
            show_labels: Whether to show player name labels
            label_offset: Vertical offset for labels
            marker_size: Size of player markers
        """
        for _, player in players_df.iterrows():
            position = player['player_position']

            if position in config.ROUTE_RUNNER_POSITIONS:
                # Route runners (WR, TE, RB)
                ax.scatter(
                    player['x'], player['y'],
                    s=marker_size,
                    c=config.OFFENSIVE_COLOR,
                    marker=config.OFFENSIVE_MARKER,
                    edgecolor='white',
                    linewidth=2,
                    zorder=5
                )

                if show_labels:
                    name = player['player_name'].split()[-1]  # Last name only
                    ax.text(
                        player['x'], player['y'] - label_offset,
                        name,
                        ha='center',
                        fontsize=config.TEXT_FONTSIZE_SMALL,
                        fontweight='bold'
                    )

            elif position == 'QB':
                # Quarterback
                ax.scatter(
                    player['x'], player['y'],
                    s=marker_size,
                    c=config.QB_COLOR,
                    marker=config.QB_MARKER,
                    edgecolor='white',
                    linewidth=2,
                    zorder=5
                )

    def plot_defensive_players(
        self,
        ax: Axes,
        players_df: pd.DataFrame,
        show_labels: bool = True,
        label_offset: float = config.LABEL_OFFSET_HORIZONTAL,
        marker_size: int = config.MARKER_SIZE_MEDIUM,
        burden_scores: Optional[Dict[int, float]] = None
    ) -> None:
        """
        Plot defensive players on the field.

        Args:
            ax: Matplotlib axes to plot on
            players_df: DataFrame with player data
            show_labels: Whether to show position labels
            label_offset: Vertical offset for labels
            marker_size: Size of player markers
            burden_scores: Optional dict mapping nfl_id to burden scores for coloring
        """
        for _, player in players_df.iterrows():
            nfl_id = player['nfl_id']

            # Determine color based on burden score if provided
            if burden_scores and nfl_id in burden_scores:
                burden = burden_scores[nfl_id]
                color_intensity = plt.cm.get_cmap(config.BURDEN_COLORMAP)(
                    0.3 + 0.7 * burden
                )
            else:
                color_intensity = config.DEFENSIVE_COLOR

            ax.scatter(
                player['x'], player['y'],
                s=marker_size,
                c=[color_intensity] if isinstance(color_intensity, tuple) else color_intensity,
                marker=config.DEFENSIVE_MARKER,
                edgecolor='white' if not burden_scores else 'black',
                linewidth=2,
                zorder=4
            )

            if show_labels:
                if 'player_position' in player:
                    label = player['player_position']
                elif 'player_name' in player:
                    label = player['player_name'].split()[-1]
                else:
                    label = ''

                if label:
                    ax.text(
                        player['x'], player['y'] + label_offset,
                        label,
                        ha='center',
                        fontsize=config.TEXT_FONTSIZE_TINY,
                        fontweight='bold' if not burden_scores else 'normal'
                    )

    def draw_coverage_radius(
        self,
        ax: Axes,
        x: float,
        y: float,
        radius: float = config.DEFENDER_COVERAGE_RADIUS,
        color: str = config.DEFENSIVE_COLOR,
        alpha: float = config.LINE_ALPHA_MED
    ) -> None:
        """
        Draw coverage radius circle around a defender.

        Args:
            ax: Matplotlib axes
            x: X coordinate of defender
            y: Y coordinate of defender
            radius: Coverage radius in yards
            color: Circle edge color
            alpha: Transparency
        """
        circle = Circle(
            (x, y), radius,
            fill=False,
            edgecolor=color,
            linewidth=1,
            alpha=alpha
        )
        ax.add_patch(circle)

    def draw_zone(
        self,
        ax: Axes,
        zone_def: Dict,
        color: str,
        alpha: float = config.ZONE_ALPHA,
        label: Optional[str] = None,
        show_label: bool = True
    ) -> None:
        """
        Draw a zone on the field.

        Args:
            ax: Matplotlib axes
            zone_def: Zone definition dict with 'x_range' and 'y_range'
            color: Zone fill color
            alpha: Transparency
            label: Optional zone label
            show_label: Whether to show zone name in center
        """
        x_range = zone_def['x_range']
        y_range = zone_def['y_range']

        rect = Rectangle(
            (x_range[0], y_range[0]),
            x_range[1] - x_range[0],
            y_range[1] - y_range[0],
            alpha=alpha,
            facecolor=color,
            edgecolor=config.ZONE_EDGE_COLOR,
            linewidth=config.ZONE_EDGE_WIDTH
        )
        ax.add_patch(rect)

        if show_label and label:
            cx = (x_range[0] + x_range[1]) / 2
            cy = (y_range[0] + y_range[1]) / 2
            ax.text(
                cx, cy,
                label.replace('_', '\n'),
                ha='center',
                va='center',
                fontsize=config.TEXT_FONTSIZE_SMALL,
                fontweight='bold'
            )

    def draw_field_boundaries(
        self,
        ax: Axes,
        los: float,
        field_width: float = config.FIELD_WIDTH,
        show_los: bool = True,
        los_label: str = 'Line of Scrimmage'
    ) -> None:
        """
        Draw field boundaries and line of scrimmage.

        Args:
            ax: Matplotlib axes
            los: Line of scrimmage position
            field_width: Width of field in yards
            show_los: Whether to draw line of scrimmage
            los_label: Label for LOS line
        """
        # Horizontal boundaries
        ax.axhline(
            y=0,
            color=config.BOUNDARY_COLOR,
            linewidth=config.BOUNDARY_LINEWIDTH
        )
        ax.axhline(
            y=field_width,
            color=config.BOUNDARY_COLOR,
            linewidth=config.BOUNDARY_LINEWIDTH
        )

        # Line of scrimmage
        if show_los:
            ax.axvline(
                x=los,
                color=config.LOS_COLOR,
                linewidth=config.LOS_LINEWIDTH,
                alpha=config.LINE_ALPHA_HIGH,
                label=los_label
            )

    def plot_ball_landing(
        self,
        ax: Axes,
        ball_x: float,
        ball_y: float,
        marker_size: int = config.MARKER_SIZE_SMALL,
        label: str = 'Ball Landing'
    ) -> None:
        """
        Plot ball landing position.

        Args:
            ax: Matplotlib axes
            ball_x: Ball landing x coordinate
            ball_y: Ball landing y coordinate
            marker_size: Size of ball marker
            label: Label for legend
        """
        ax.scatter(
            ball_x, ball_y,
            s=marker_size,
            c=config.BALL_COLOR,
            marker=config.BALL_MARKER,
            edgecolor='black',
            linewidth=2,
            zorder=6,
            label=label
        )

    def setup_field_axes(
        self,
        ax: Axes,
        los: float,
        field_width: float = config.FIELD_WIDTH,
        title: Optional[str] = None
    ) -> None:
        """
        Setup standard axes configuration for field plots.

        Args:
            ax: Matplotlib axes
            los: Line of scrimmage position
            field_width: Field width in yards
            title: Optional plot title
        """
        if title:
            ax.set_title(title, fontsize=config.TEXT_FONTSIZE_LARGE, fontweight='bold')

        xlim = config.get_field_xlim(los)
        ylim = config.get_field_ylim(field_width)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Field Position (yards)', fontsize=11)
        ax.set_ylabel('Field Width (yards)', fontsize=11)
        ax.grid(True, alpha=config.GRID_ALPHA)

    def draw_spacing_lines(
        self,
        ax: Axes,
        positions: np.ndarray,
        show_distances: bool = True,
        line_alpha: float = config.LINE_ALPHA_MED
    ) -> None:
        """
        Draw spacing lines between all receiver positions.

        Args:
            ax: Matplotlib axes
            positions: Nx2 array of (x, y) positions
            show_distances: Whether to show distance labels
            line_alpha: Line transparency
        """
        from itertools import combinations

        for (i, p1), (j, p2) in combinations(enumerate(positions), 2):
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                'g--',
                alpha=line_alpha,
                linewidth=1
            )

            if show_distances:
                mid_x, mid_y = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
                ax.text(
                    mid_x, mid_y,
                    f'{dist:.1f}y',
                    fontsize=config.TEXT_FONTSIZE_SMALL,
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='yellow',
                        alpha=0.7
                    )
                )
