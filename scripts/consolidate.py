#!/usr/bin/env python
"""
CLI script for NFL data consolidation.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nfl_analysis.consolidation import NFLDataConsolidator


def main():
    """Main CLI entry point for consolidation."""
    parser = argparse.ArgumentParser(
        description="Consolidate NFL tracking data from weekly CSV files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/114239_nfl_competition_files_published_analytics_final",
        help="Path to raw data directory (default: data/raw/114239_nfl_competition_files_published_analytics_final)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/consolidated",
        help="Path to output directory (default: data/consolidated)"
    )
    parser.add_argument(
        "--skip-spatial",
        action="store_true",
        help="Skip spatial features calculation (faster)"
    )

    args = parser.parse_args()

    # Check if data directory exists
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Please ensure the data directory exists.")
        return 1

    # Create consolidator
    consolidator = NFLDataConsolidator(args.data_dir, args.output_dir)

    # Run consolidation
    try:
        consolidator.consolidate_all(skip_spatial=args.skip_spatial)
        print("\nConsolidated datasets are ready for analysis!")
        print(f"Load them with: pd.read_parquet('{args.output_dir}/[dataset_name].parquet')")
        return 0
    except Exception as e:
        print(f"\nError during consolidation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
