#!/usr/bin/env python
"""
CLI script for animating NFL plays.

This is a wrapper around the animation functionality from notebook/animation.py
"""

import argparse
import sys
from pathlib import Path

# This script is a placeholder for future animation CLI functionality
# The current implementation is in notebook/animation.py

def main():
    """Main CLI entry point for play animation."""
    parser = argparse.ArgumentParser(
        description="Animate NFL plays from tracking data"
    )
    parser.add_argument(
        "game_id",
        type=str,
        help="Game ID to animate"
    )
    parser.add_argument(
        "play_id",
        type=str,
        help="Play ID to animate"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/consolidated",
        help="Path to consolidated data directory (default: data/consolidated)"
    )

    args = parser.parse_args()

    print(f"Animation for Game {args.game_id}, Play {args.play_id}")
    print("\nNote: Animation functionality is currently in notebook/animation.py")
    print(f"Run: python notebook/animation.py {args.game_id} {args.play_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
