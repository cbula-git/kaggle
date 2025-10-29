#!/usr/bin/env python3
"""
Coverage Zone Value Analysis
=============================
Analyze zone exploitation weighted by actual value (yards gained), not just completion %.

This corrects for the fact that shallow completions are "allowed" by defensive design
and provides a more accurate picture of which zones are truly vulnerable.

Metrics:
- Yards per target (efficiency)
- Yards per completion (explosiveness)
- Expected value of targeting each zone
- Success rate (completions gaining positive yards)
"""

import sys
sys.path.insert(0, '/Users/cbulacan/cbula-git/kaggle/src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)

def load_data_with_outcomes():
    """Load zone data with play outcomes."""
    print("Loading data with play outcomes...")
    zone_data = pd.read_parquet('data/consolidated/zone_vulnerability_timeseries.parquet')
    supp_data = pd.read_parquet('data/consolidated/supplementary.parquet')

    # Merge
    data = zone_data.merge(
        supp_data[['game_id', 'play_id', 'team_coverage_type', 'team_coverage_man_zone',
                   'pass_result', 'yards_gained', 'yards_to_go']],
        on=['game_id', 'play_id'],
        how='left'
    )

    print(f"Loaded {len(data):,} zone-frame records")
    print(f"Plays with yards data: {data['yards_gained'].notna().sum() / len(data) * 100:.1f}%")
    print()

    return data

def analyze_zone_value_by_coverage(data):
    """
    Calculate value-weighted metrics for each zone-coverage combination.

    Metrics:
    - Yards per target (expected value)
    - Yards per completion (explosiveness when it works)
    - Success rate (% gaining positive yards)
    - Explosive play rate (% gaining 15+ yards)
    """
    print("Analyzing zone value by coverage type...")

    # Filter to throw frame
    throw_data = data[data['phase'] == 'at_throw'].copy()

    # Focus on zone coverages
    zone_coverages = throw_data[throw_data['team_coverage_man_zone'] == 'ZONE_COVERAGE'].copy()

    coverage_types = ['COVER_2_ZONE', 'COVER_3_ZONE', 'COVER_4_ZONE', 'COVER_6_ZONE']

    results = []

    for coverage in coverage_types:
        cov_data = zone_coverages[zone_coverages['team_coverage_type'] == coverage]

        if len(cov_data) == 0:
            continue

        total_plays = cov_data[['game_id', 'play_id']].drop_duplicates().shape[0]

        for zone_id in cov_data['zone_id'].unique():
            zone_data_cov = cov_data[cov_data['zone_id'] == zone_id]

            # Get target zone data
            targets = zone_data_cov[zone_data_cov['is_target_zone'] == True]

            times_targeted = targets[['game_id', 'play_id']].drop_duplicates().shape[0]

            if times_targeted == 0:
                continue

            # Get play outcomes
            target_plays = targets[['game_id', 'play_id', 'pass_result', 'yards_gained', 'yards_to_go']].drop_duplicates()

            # Calculate value metrics
            completions = target_plays[target_plays['pass_result'] == 'C']

            # Yards per target (expected value)
            yards_per_target = target_plays['yards_gained'].mean()

            # Yards per completion (explosiveness)
            yards_per_completion = completions['yards_gained'].mean() if len(completions) > 0 else 0

            # Success rate (gaining positive yards)
            success_rate = (target_plays['yards_gained'] > 0).mean()

            # Explosive play rate (15+ yards)
            explosive_rate = (target_plays['yards_gained'] >= 15).mean()

            # Completion rate
            completion_rate = (target_plays['pass_result'] == 'C').mean()

            # Target rate
            target_rate = times_targeted / total_plays

            # Expected points contribution (simplified: yards per target * target rate)
            expected_value = yards_per_target * target_rate * 100  # Per 100 plays

            results.append({
                'coverage_type': coverage,
                'zone_id': zone_id,
                'zone_depth': zone_id.split('_')[0],
                'zone_lateral': '_'.join(zone_id.split('_')[1:]),
                'total_plays': total_plays,
                'times_targeted': times_targeted,
                'target_rate': target_rate,
                'completion_rate': completion_rate,
                'yards_per_target': yards_per_target,
                'yards_per_completion': yards_per_completion,
                'success_rate': success_rate,
                'explosive_rate': explosive_rate,
                'expected_value': expected_value
            })

    results_df = pd.DataFrame(results)
    return results_df

def create_value_heatmaps(value_df, save_dir='visualizations'):
    """Create heatmaps showing yards per target for each coverage."""
    print("Creating value-weighted heatmaps...")

    save_dir = Path(save_dir)

    coverage_types = value_df['coverage_type'].unique()

    # Yards per target heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    depth_order = ['deep', 'intermediate', 'shallow']
    lateral_order = ['far_left', 'left_hash', 'middle', 'right_hash', 'far_right']

    for idx, coverage in enumerate(sorted(coverage_types)):
        if idx >= 4:
            break

        ax = axes[idx]

        cov_data = value_df[(value_df['coverage_type'] == coverage) &
                             (value_df['times_targeted'] >= 5)]

        pivot = cov_data.pivot_table(
            index='zone_depth',
            columns='zone_lateral',
            values='yards_per_target',
            aggfunc='first'
        )

        pivot = pivot.reindex(index=depth_order, columns=lateral_order)

        # Plot
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                    center=5,  # 5 yards is roughly average
                    cbar_kws={'label': 'Yards per Target'},
                    ax=ax, vmin=-2, vmax=12)

        coverage_name = coverage.replace('_ZONE', '').replace('_', ' ').title()
        total_plays = cov_data['total_plays'].iloc[0]
        ax.set_title(f'{coverage_name} - Yards per Target\n({total_plays:,} plays, min 5 targets)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Lateral Position', fontweight='bold')
        ax.set_ylabel('Depth from LOS', fontweight='bold')

    plt.suptitle('Zone Value by Coverage Type (Yards per Target)\nGreen = High value for offense, Red = Low value',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = save_dir / 'coverage_zone_value_yards_per_target.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {save_path}")

    # Expected value heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    for idx, coverage in enumerate(sorted(coverage_types)):
        if idx >= 4:
            break

        ax = axes[idx]

        cov_data = value_df[(value_df['coverage_type'] == coverage) &
                             (value_df['times_targeted'] >= 3)]

        pivot = cov_data.pivot_table(
            index='zone_depth',
            columns='zone_lateral',
            values='expected_value',
            aggfunc='first'
        )

        pivot = pivot.reindex(index=depth_order, columns=lateral_order)

        # Plot
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                    center=pivot.mean().mean(),
                    cbar_kws={'label': 'Expected Yards per 100 Plays'},
                    ax=ax)

        coverage_name = coverage.replace('_ZONE', '').replace('_', ' ').title()
        ax.set_title(f'{coverage_name} - Expected Value\n(yards per 100 plays)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Lateral Position', fontweight='bold')
        ax.set_ylabel('Depth from LOS', fontweight='bold')

    plt.suptitle('Expected Value by Zone (Target Rate × Yards per Target × 100)\nHigher = More valuable to attack',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = save_dir / 'coverage_zone_expected_value.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved expected value heatmap to {save_path}")

    return fig

def generate_value_insights(value_df):
    """Generate insights based on value metrics."""
    print()
    print("=" * 80)
    print("VALUE-WEIGHTED COVERAGE ANALYSIS")
    print("=" * 80)
    print()

    coverage_types = sorted(value_df['coverage_type'].unique())

    for coverage in coverage_types:
        cov_data = value_df[(value_df['coverage_type'] == coverage) &
                             (value_df['times_targeted'] >= 5)]

        coverage_name = coverage.replace('_ZONE', '').replace('_', ' ').title()
        total_plays = cov_data['total_plays'].iloc[0]

        print(f"{coverage_name}")
        print("-" * 80)
        print(f"Sample size: {total_plays:,} plays")
        print()

        # Highest value zones (yards per target)
        print("Highest Value Zones (Yards per Target):")
        top_value = cov_data.nlargest(5, 'yards_per_target')[
            ['zone_id', 'target_rate', 'yards_per_target', 'completion_rate', 'explosive_rate']
        ]
        for idx, row in top_value.iterrows():
            print(f"  {row['zone_id']:25s} - {row['yards_per_target']:5.1f} yds/target, "
                  f"{row['target_rate']*100:5.1f}% target rate, {row['completion_rate']*100:5.1f}% comp, "
                  f"{row['explosive_rate']*100:4.1f}% explosive")
        print()

        # Highest expected value (accounting for how often targeted)
        print("Highest Expected Value (Target Rate × Yards):")
        top_expected = cov_data.nlargest(5, 'expected_value')[
            ['zone_id', 'expected_value', 'target_rate', 'yards_per_target']
        ]
        for idx, row in top_expected.iterrows():
            print(f"  {row['zone_id']:25s} - {row['expected_value']:5.1f} yards/100 plays "
                  f"({row['target_rate']*100:4.1f}% × {row['yards_per_target']:4.1f} yds)")
        print()

        # Compare shallow vs deep value
        shallow = cov_data[cov_data['zone_depth'] == 'shallow']
        deep = cov_data[cov_data['zone_depth'] == 'deep']

        if len(shallow) > 0 and len(deep) > 0:
            print(f"Shallow vs Deep Comparison:")
            print(f"  Shallow zones: {shallow['yards_per_target'].mean():4.1f} yds/target, "
                  f"{shallow['completion_rate'].mean()*100:5.1f}% comp")
            print(f"  Deep zones:    {deep['yards_per_target'].mean():4.1f} yds/target, "
                  f"{deep['completion_rate'].mean()*100:5.1f}% comp")
            print()

        print("=" * 80)
        print()

def main():
    """Run value-weighted analysis."""
    print("=" * 80)
    print("COVERAGE ZONE VALUE ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    data = load_data_with_outcomes()

    # Analyze value
    value_df = analyze_zone_value_by_coverage(data)

    # Save results
    output_dir = Path('data/consolidated')
    value_df.to_csv(output_dir / 'coverage_zone_value.csv', index=False)
    print(f"✓ Saved value metrics to {output_dir}/coverage_zone_value.csv")
    print()

    # Create visualizations
    create_value_heatmaps(value_df)

    # Generate insights
    generate_value_insights(value_df)

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

    return value_df

if __name__ == '__main__':
    value_df = main()
