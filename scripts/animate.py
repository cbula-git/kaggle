#!/usr/bin/env python
"""
CLI script for animating NFL plays.

This script uses the PlayAnimator class to create animated visualizations
of NFL plays from tracking data.
"""

import argparse
import sys
from pathlib import Path

from nfl_analysis.animation import PlayAnimator


def main():
    """Main CLI entry point for play animation."""
    parser = argparse.ArgumentParser(
        description="Animate NFL plays from tracking data"
    )
    parser.add_argument(
        "game_id",
        type=int,
        help="Game ID to animate"
    )
    parser.add_argument(
        "play_id",
        type=int,
        help="Play ID to animate"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/consolidated",
        help="Path to consolidated data directory (default: data/consolidated)"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save animation to file (e.g., play.mp4, play.gif)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for saved animation (default: 5)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=200,
        help="Milliseconds between frames (default: 200)"
    )
    parser.add_argument(
        "--pause",
        type=int,
        default=30,
        help="Seconds to display animation (default: 30)"
    )

    args = parser.parse_args()

    # Validate data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1

    print(f"Creating animation for Game {args.game_id}, Play {args.play_id}")
    print(f"Using data from: {args.data_dir}")

    try:
        # Create animator
        animator = PlayAnimator(data_dir=args.data_dir)

        # Create animation
        ani = animator.create_animation(
            game_id=args.game_id,
            play_id=args.play_id,
            interval=args.interval
        )

        # Save or show
        if args.save:
            print(f"Saving animation to: {args.save}")
            animator.save(ani, args.save, fps=args.fps)
            print("Animation saved successfully!")
        else:
            print(f"Displaying animation for {args.pause} seconds...")
            animator.show(ani, pause_time=args.pause)
            print("Animation complete!")

        return 0

    except FileNotFoundError as e:
        print(f"Error: Could not find required data files: {e}")
        return 1
    except Exception as e:
        print(f"Error creating animation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
