"""
Field renderer for NFL play visualization.

This module provides the FieldRenderer class for drawing football fields
with proper dimensions, markings, and styling.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Tuple, Optional
from textwrap import wrap

from . import config


class FieldRenderer:
    """
    Renders an NFL football field with proper dimensions and markings.

    This class handles all field-related visualization including:
    - Field dimensions and boundaries
    - Yard lines and end zones
    - Field coloring and styling
    - Text and annotations

    Attributes:
        figsize: Tuple of (width, height) for the figure in inches
    """

    def __init__(self, figsize: Tuple[float, float] = config.DEFAULT_FIGSIZE):
        """
        Initialize the FieldRenderer.

        Args:
            figsize: Figure size as (width, height) in inches
        """
        self.figsize = figsize

    def create_figure(self) -> Tuple[Figure, Axes]:
        """
        Create a new figure and axes for the field.

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        return fig, ax

    def setup_field(self, ax: Axes) -> Axes:
        """
        Setup the football field with all markings and styling.

        Args:
            ax: Matplotlib axes to draw on

        Returns:
            The configured axes
        """
        # Set field dimensions
        ax.set(xlim=[0, config.FIELD_LENGTH], ylim=[0, config.FIELD_WIDTH])

        # Set field color
        ax.set_facecolor(config.FIELD_COLOR)

        # Add yard lines
        self.add_yard_lines(ax)

        # Add end zones
        self.add_end_zones(ax)

        # Configure x-axis ticks (yard markers)
        ax.set_xticks(
            ticks=[x for x in range(0, 130, 10)],
            labels=config.YARD_LINE_LABELS
        )

        return ax

    def add_yard_lines(self, ax: Axes) -> None:
        """
        Add yard line markings to the field.

        Args:
            ax: Matplotlib axes to draw on
        """
        for yard in config.YARD_LINE_POSITIONS:
            ax.axvline(
                x=yard,
                color=config.YARD_LINE_COLOR,
                linestyle='-',
                linewidth=config.YARD_LINE_WIDTH,
                alpha=config.YARD_LINE_ALPHA
            )

    def add_end_zones(self, ax: Axes) -> None:
        """
        Add end zone rectangles to the field.

        Args:
            ax: Matplotlib axes to draw on
        """
        # Left end zone
        ax.add_patch(Rectangle(
            (0, 0),
            config.END_ZONE_LENGTH,
            config.FIELD_WIDTH,
            facecolor=config.END_ZONE_COLOR,
            alpha=config.END_ZONE_ALPHA
        ))

        # Right end zone
        ax.add_patch(Rectangle(
            (config.PLAYING_FIELD_END, 0),
            config.END_ZONE_LENGTH,
            config.FIELD_WIDTH,
            facecolor=config.END_ZONE_COLOR,
            alpha=config.END_ZONE_ALPHA
        ))

    def add_title(self, ax: Axes, title: str, wrap_width: int = config.TITLE_WRAP_WIDTH) -> None:
        """
        Add a wrapped title to the field.

        Args:
            ax: Matplotlib axes to draw on
            title: Title text to display
            wrap_width: Maximum characters per line before wrapping
        """
        wrapped_title = "\n".join(wrap(title, wrap_width))
        ax.set_title(wrapped_title)

    def add_info_text(
        self,
        fig: Figure,
        text: str,
        x: float = config.INFO_TEXT_X,
        y: float = config.INFO_TEXT_Y,
        fontsize: int = config.INFO_TEXT_FONTSIZE
    ) -> None:
        """
        Add informational text to the figure (outside the plot area).

        Args:
            fig: Matplotlib figure to add text to
            text: Text to display
            x: X position in figure coordinates (0-1)
            y: Y position in figure coordinates (0-1)
            fontsize: Font size for the text
        """
        fig.text(
            x, y,
            text,
            wrap=False,
            horizontalalignment='left',
            fontsize=fontsize
        )

    def plot_ball_position(
        self,
        ax: Axes,
        x: float,
        y: float,
        color: str = config.PLAYER_MARKER_FORMAT["Ball"][0],
        marker: str = config.PLAYER_MARKER_FORMAT["Ball"][1]
    ) -> None:
        """
        Plot the ball landing position on the field.

        Args:
            ax: Matplotlib axes to draw on
            x: X coordinate of ball
            y: Y coordinate of ball
            color: Marker color
            marker: Marker style
        """
        ax.plot(x, y, color=color, marker=marker)

    def add_legend(self, ax: Axes, loc: str = config.LEGEND_LOCATION) -> None:
        """
        Add a legend to the plot.

        Args:
            ax: Matplotlib axes to draw on
            loc: Legend location (e.g., 'upper left', 'lower right')
        """
        ax.legend(loc=loc)
