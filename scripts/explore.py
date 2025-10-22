#!/usr/bin/env python
"""
CLI script for NFL data exploration.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nfl_analysis.exploration import NFLDataExplorer


def main():
    """Main CLI entry point for exploration."""
    parser = argparse.ArgumentParser(
        description="Explore and validate consolidated NFL tracking data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/consolidated",
        help="Path to consolidated data directory (default: data/consolidated)"
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation"
    )

    args = parser.parse_args()

    # Create explorer
    explorer = NFLDataExplorer(args.data_dir)

    try:
        # Load datasets
        explorer.load_datasets()

        # Run exploration
        explorer.show_dataset_info()
        explorer.validate_data_quality()
        explorer.explore_play_level_data()
        explorer.explore_player_analysis()

        if not args.skip_viz:
            explorer.create_visualizations()

        explorer.generate_report()

        print("\n" + "=" * 80)
        print("EXPLORATION COMPLETE!")
        print("=" * 80)
        return 0

    except FileNotFoundError:
        print("\nError: Consolidated data not found.")
        print(f"Please run 'nfl-consolidate' or 'python scripts/consolidate.py' first.")
        return 1
    except Exception as e:
        print(f"\nError during exploration: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
