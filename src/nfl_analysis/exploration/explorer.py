"""
NFL Data Explorer
=================
Quick exploration and validation of consolidated NFL tracking data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class NFLDataExplorer:
    """Explore and validate consolidated NFL data."""

    def __init__(self, data_dir: str = 'data/consolidated'):
        """Initialize explorer with path to consolidated data."""
        self.data_dir = Path(data_dir)
        self.datasets = {}

    def load_datasets(self, datasets: List[str] = None):
        """
        Load consolidated datasets.

        Args:
            datasets: List of dataset names to load (default: all)
        """
        if datasets is None:
            datasets = ['master_input', 'master_output', 'supplementary',
                       'play_level', 'trajectories', 'player_analysis']

        print("Loading datasets...")
        for name in datasets:
            file_path = self.data_dir / f"{name}.parquet"
            if file_path.exists():
                print(f"  Loading {name}...")
                self.datasets[name] = pd.read_parquet(file_path)
                print(f"    Shape: {self.datasets[name].shape}")
            else:
                print(f"  Warning: {name}.parquet not found")

        print(f"\nLoaded {len(self.datasets)} datasets")

    def show_dataset_info(self):
        """Display information about loaded datasets."""
        print("\n" + "=" * 80)
        print("DATASET INFORMATION")
        print("=" * 80)

        for name, df in self.datasets.items():
            print(f"\n{name.upper()}")
            print("-" * 80)
            print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            print(f"\nColumns: {', '.join(df.columns[:10].tolist())}" +
                  ("..." if len(df.columns) > 10 else ""))
            print(f"\nFirst row sample:")
            print(df.head(1).to_string())

    def validate_data_quality(self):
        """Check data quality issues."""
        print("\n" + "=" * 80)
        print("DATA QUALITY CHECKS")
        print("=" * 80)

        for name, df in self.datasets.items():
            print(f"\n{name.upper()}")
            print("-" * 80)

            # Missing values
            missing = df.isnull().sum()
            missing_pct = 100 * missing / len(df)
            missing_summary = missing[missing > 0].sort_values(ascending=False)

            if len(missing_summary) > 0:
                print("Missing values:")
                for col, count in missing_summary.head(10).items():
                    print(f"  {col:30s}: {count:8,} ({missing_pct[col]:5.2f}%)")
            else:
                print("✓ No missing values found")

            # Duplicate rows
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                print(f"\n⚠ Duplicate rows: {duplicates:,}")
            else:
                print("\n✓ No duplicate rows")

    def explore_play_level_data(self):
        """Explore play-level dataset."""
        if 'play_level' not in self.datasets:
            print("Play-level dataset not loaded")
            return

        df = self.datasets['play_level']

        print("\n" + "=" * 80)
        print("PLAY-LEVEL ANALYSIS")
        print("=" * 80)

        # Pass result distribution
        if 'pass_result' in df.columns:
            print("\nPass Result Distribution:")
            print(df['pass_result'].value_counts().to_string())

            # Completion percentage
            completions = (df['pass_result'] == 'C').sum()
            total_passes = len(df[df['pass_result'].isin(['C', 'I'])])
            if total_passes > 0:
                comp_pct = 100 * completions / total_passes
                print(f"\nCompletion Percentage: {comp_pct:.1f}%")

        # Coverage type analysis
        if 'team_coverage_type' in df.columns:
            print("\nTop 10 Coverage Types:")
            print(df['team_coverage_type'].value_counts().head(10).to_string())

        # Formation analysis
        if 'offense_formation' in df.columns:
            print("\nTop 10 Offensive Formations:")
            print(df['offense_formation'].value_counts().head(10).to_string())

        # Numeric statistics
        numeric_cols = ['s_mean', 's_max', 'a_mean', 'a_max', 'pass_length',
                       'yards_gained', 'expected_points_added']
        numeric_cols = [col for col in numeric_cols if col in df.columns]

        if numeric_cols:
            print("\nNumeric Feature Statistics:")
            print(df[numeric_cols].describe().to_string())

    def explore_player_analysis(self):
        """Explore player analysis dataset."""
        if 'player_analysis' not in self.datasets:
            print("Player analysis dataset not loaded")
            return

        df = self.datasets['player_analysis']

        print("\n" + "=" * 80)
        print("PLAYER ANALYSIS")
        print("=" * 80)

        # Role distribution
        if 'player_role' in df.columns:
            print("\nPlayer Role Distribution:")
            print(df['player_role'].value_counts().to_string())

        # Position distribution
        if 'player_position' in df.columns:
            print("\nTop 10 Player Positions:")
            print(df['player_position'].value_counts().head(10).to_string())

        # Displacement statistics by role
        if 'total_displacement' in df.columns and 'player_role' in df.columns:
            print("\nAverage Displacement by Role:")
            disp_by_role = df.groupby('player_role')['total_displacement'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(2)
            print(disp_by_role.to_string())

        # Speed statistics by position
        if 's' in df.columns and 'player_position' in df.columns:
            print("\nAverage Speed by Position (Top 10):")
            speed_by_pos = df.groupby('player_position')['s'].mean().sort_values(ascending=False).head(10)
            print(speed_by_pos.to_string())

    def create_visualizations(self, output_dir: str = 'visualizations'):
        """Create exploratory visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)

        # 1. Pass result distribution
        if 'play_level' in self.datasets and 'pass_result' in self.datasets['play_level'].columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.datasets['play_level']['pass_result'].value_counts().plot(kind='bar', ax=ax)
            ax.set_title('Pass Result Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Pass Result')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'pass_result_distribution.png', dpi=150)
            print(f"  Saved: pass_result_distribution.png")
            plt.close()

        # 2. Player displacement by role
        if 'player_analysis' in self.datasets:
            df = self.datasets['player_analysis']
            if 'total_displacement' in df.columns and 'player_role' in df.columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                df.boxplot(column='total_displacement', by='player_role', ax=ax)
                ax.set_title('Player Displacement Distribution by Role', fontsize=14, fontweight='bold')
                ax.set_xlabel('Player Role')
                ax.set_ylabel('Total Displacement (yards)')
                plt.suptitle('')  # Remove default title
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(output_path / 'displacement_by_role.png', dpi=150)
                print(f"  Saved: displacement_by_role.png")
                plt.close()

        # 3. Speed distribution
        if 'master_input' in self.datasets and 's' in self.datasets['master_input'].columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.datasets['master_input']['s'].hist(bins=50, ax=ax, edgecolor='black')
            ax.set_title('Player Speed Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Speed (yards/second)')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(output_path / 'speed_distribution.png', dpi=150)
            print(f"  Saved: speed_distribution.png")
            plt.close()

        # 4. Field position heatmap
        if 'master_input' in self.datasets:
            df = self.datasets['master_input']
            if 'x' in df.columns and 'y' in df.columns:
                # Sample for performance
                sample_df = df.sample(min(50000, len(df)))

                fig, ax = plt.subplots(figsize=(14, 6))
                hb = ax.hexbin(sample_df['x'], sample_df['y'], gridsize=50, cmap='YlOrRd')
                ax.set_title('Player Position Heatmap (Pre-Pass)', fontsize=14, fontweight='bold')
                ax.set_xlabel('X Position (yards)')
                ax.set_ylabel('Y Position (yards)')
                ax.set_xlim(0, 120)
                ax.set_ylim(0, 53.3)
                plt.colorbar(hb, ax=ax, label='Frequency')
                plt.tight_layout()
                plt.savefig(output_path / 'position_heatmap.png', dpi=150)
                print(f"  Saved: position_heatmap.png")
                plt.close()

        print(f"\nAll visualizations saved to: {output_path.absolute()}")

    def generate_report(self, output_file: str = 'exploration_report.txt'):
        """Generate comprehensive exploration report."""
        report_path = self.data_dir / output_file

        print("\n" + "=" * 80)
        print("GENERATING EXPLORATION REPORT")
        print("=" * 80)

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NFL TRACKING DATA EXPLORATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Dataset summaries
            f.write("LOADED DATASETS\n")
            f.write("-" * 80 + "\n")
            for name, df in self.datasets.items():
                f.write(f"{name}:\n")
                f.write(f"  Rows: {len(df):,}\n")
                f.write(f"  Columns: {len(df.columns)}\n")
                f.write(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n\n")

            # Data quality
            f.write("\nDATA QUALITY SUMMARY\n")
            f.write("-" * 80 + "\n")
            for name, df in self.datasets.items():
                missing = df.isnull().sum().sum()
                duplicates = df.duplicated().sum()
                f.write(f"{name}:\n")
                f.write(f"  Missing values: {missing:,}\n")
                f.write(f"  Duplicate rows: {duplicates:,}\n\n")

            f.write("=" * 80 + "\n")

        print(f"Report saved to: {report_path.absolute()}")

    def run_full_exploration(self):
        """Run complete exploration pipeline."""
        self.load_datasets()
        self.show_dataset_info()
        self.validate_data_quality()
        self.explore_play_level_data()
        self.explore_player_analysis()
        self.create_visualizations()
        self.generate_report()

        print("\n" + "=" * 80)
        print("EXPLORATION COMPLETE!")
        print("=" * 80)
