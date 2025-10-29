#!/usr/bin/env python3
"""
Team Defensive Analysis
========================
Analyze team defensive capabilities using zone vulnerability metrics.

This script calculates aggregate defensive metrics by team to assess:
- Overall defensive effectiveness
- Zone-specific strengths/weaknesses
- Coverage scheme performance
- Target zone vulnerability when passes are thrown
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
plt.rcParams['figure.figsize'] = (16, 10)

def load_data():
    """Load zone vulnerability and supplementary data."""
    print("Loading data...")
    zone_data = pd.read_parquet('data/consolidated/zone_vulnerability_timeseries.parquet')
    supp_data = pd.read_parquet('data/consolidated/supplementary.parquet')

    # Merge to get team information
    data = zone_data.merge(
        supp_data[['game_id', 'play_id', 'defensive_team', 'team_coverage_man_zone',
                   'team_coverage_type', 'pass_result']],
        on=['game_id', 'play_id'],
        how='left'
    )

    print(f"Loaded {len(data):,} zone-frame records across {data['game_id'].nunique()} games")
    print(f"Teams: {sorted(data['defensive_team'].unique())}")
    print()

    return data

def calculate_team_defensive_metrics(data):
    """
    Calculate comprehensive team defensive metrics.

    Returns DataFrame with one row per team containing:
    - Average void score (lower = better defense)
    - Target zone void score (vulnerability when ball is thrown there)
    - Zone coverage density
    - Zone abandonment rate
    - Deep zone vulnerability
    - Completion rate against
    """
    print("Calculating team defensive metrics...")

    # Filter to throw frame only (most critical moment)
    throw_frames = data[data['phase'] == 'at_throw'].copy()

    metrics = []

    for team in sorted(data['defensive_team'].unique()):
        if pd.isna(team):
            continue

        team_data = throw_frames[throw_frames['defensive_team'] == team]

        # Overall metrics
        avg_void_score = team_data['zone_void_score'].mean()
        median_void_score = team_data['zone_void_score'].median()

        # Target zone vulnerability (when QB throws to a zone, how vulnerable was it?)
        target_zones = team_data[team_data['is_target_zone'] == True]
        target_zone_void = target_zones['zone_void_score'].mean()

        # Coverage density
        avg_coverage_density = team_data['coverage_density'].mean()

        # Zone abandonment rate (% of zones with 0 defenders)
        zone_abandonment_rate = (team_data['defender_count'] == 0).mean()

        # Deep zone vulnerability (deep zones are critical)
        deep_zones = team_data[team_data['zone_depth_category'] == 'deep']
        deep_void_score = deep_zones['zone_void_score'].mean()
        deep_abandonment_rate = (deep_zones['defender_count'] == 0).mean()

        # Completion rate against
        target_plays = target_zones[['game_id', 'play_id', 'pass_result']].drop_duplicates()
        completion_rate = (target_plays['pass_result'] == 'C').mean()
        total_plays = len(target_plays)

        # Defenders per zone
        avg_defenders_per_zone = team_data['defender_count'].mean()

        metrics.append({
            'team': team,
            'plays_faced': total_plays,
            'avg_void_score': avg_void_score,
            'median_void_score': median_void_score,
            'target_zone_void_score': target_zone_void,
            'avg_coverage_density': avg_coverage_density,
            'zone_abandonment_rate': zone_abandonment_rate,
            'deep_zone_void_score': deep_void_score,
            'deep_zone_abandonment_rate': deep_abandonment_rate,
            'completion_rate_against': completion_rate,
            'avg_defenders_per_zone': avg_defenders_per_zone
        })

    metrics_df = pd.DataFrame(metrics)

    # Calculate rankings (1 = best)
    metrics_df['void_score_rank'] = metrics_df['avg_void_score'].rank()
    metrics_df['target_void_rank'] = metrics_df['target_zone_void_score'].rank()
    metrics_df['completion_rank'] = metrics_df['completion_rate_against'].rank()
    metrics_df['deep_void_rank'] = metrics_df['deep_zone_void_score'].rank()

    # Overall defensive rating (composite score - lower is better)
    metrics_df['defensive_rating'] = (
        metrics_df['void_score_rank'] +
        metrics_df['target_void_rank'] * 2 +  # Weight target zones more
        metrics_df['completion_rank'] * 1.5 +
        metrics_df['deep_void_rank'] * 1.5
    ) / 6.0

    metrics_df = metrics_df.sort_values('defensive_rating')

    return metrics_df

def calculate_team_zone_strengths(data):
    """
    Calculate team performance by zone to identify strengths/weaknesses.

    Returns DataFrame with team-zone level metrics.
    """
    print("Calculating zone-level team metrics...")

    # Filter to throw frame
    throw_frames = data[data['phase'] == 'at_throw'].copy()

    # Group by team and zone
    zone_metrics = throw_frames.groupby(['defensive_team', 'zone_id']).agg({
        'zone_void_score': ['mean', 'median', 'std'],
        'defender_count': ['mean', 'sum'],
        'coverage_density': 'mean',
        'is_target_zone': 'sum',  # How many times this zone was targeted
        'game_id': 'count'  # Total zone-plays
    }).reset_index()

    # Flatten columns
    zone_metrics.columns = ['team', 'zone_id', 'avg_void_score', 'median_void_score',
                             'std_void_score', 'avg_defenders', 'total_defenders',
                             'avg_coverage_density', 'times_targeted', 'total_plays']

    # Calculate target rate
    zone_metrics['target_rate'] = zone_metrics['times_targeted'] / zone_metrics['total_plays']

    # Remove NaN teams
    zone_metrics = zone_metrics[zone_metrics['team'].notna()]

    return zone_metrics

def plot_team_rankings(metrics_df, save_path=None):
    """Plot team defensive rankings."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Team Defensive Performance Metrics (at Throw)', fontsize=16, fontweight='bold')

    # 1. Overall Defensive Rating
    ax = axes[0, 0]
    top_15 = metrics_df.head(15)
    colors = ['green' if i < 5 else 'orange' if i < 10 else 'red' for i in range(len(top_15))]
    ax.barh(range(len(top_15)), top_15['defensive_rating'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(top_15['team'])
    ax.set_xlabel('Defensive Rating (lower = better)', fontweight='bold')
    ax.set_title('Overall Defensive Rating (Top 15)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 2. Target Zone Void Score
    ax = axes[0, 1]
    sorted_by_target = metrics_df.sort_values('target_zone_void_score').head(15)
    ax.barh(range(len(sorted_by_target)), sorted_by_target['target_zone_void_score'],
            color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(sorted_by_target)))
    ax.set_yticklabels(sorted_by_target['team'])
    ax.set_xlabel('Avg Void Score at Target Zone (lower = better)', fontweight='bold')
    ax.set_title('Target Zone Vulnerability (Top 15)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 3. Completion Rate Against
    ax = axes[1, 0]
    sorted_by_comp = metrics_df.sort_values('completion_rate_against').head(15)
    colors = ['green' if x < 0.6 else 'orange' if x < 0.65 else 'red'
              for x in sorted_by_comp['completion_rate_against']]
    ax.barh(range(len(sorted_by_comp)), sorted_by_comp['completion_rate_against'] * 100,
            color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_by_comp)))
    ax.set_yticklabels(sorted_by_comp['team'])
    ax.set_xlabel('Completion % Against', fontweight='bold')
    ax.set_title('Pass Defense Efficiency (Top 15)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=65, color='red', linestyle='--', alpha=0.5, label='League Avg')

    # 4. Deep Zone Defense
    ax = axes[1, 1]
    sorted_by_deep = metrics_df.sort_values('deep_zone_void_score').head(15)
    ax.barh(range(len(sorted_by_deep)), sorted_by_deep['deep_zone_void_score'],
            color='darkgreen', alpha=0.7)
    ax.set_yticks(range(len(sorted_by_deep)))
    ax.set_yticklabels(sorted_by_deep['team'])
    ax.set_xlabel('Deep Zone Void Score (lower = better)', fontweight='bold')
    ax.set_title('Deep Zone Defense (Top 15)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig

def plot_team_zone_heatmap(zone_metrics, team, save_path=None):
    """Plot heatmap of zone vulnerabilities for a specific team."""
    team_data = zone_metrics[zone_metrics['team'] == team].copy()

    # Parse zone_id into depth and lateral
    team_data['depth'] = team_data['zone_id'].str.split('_').str[0]
    team_data['lateral'] = team_data['zone_id'].str.split('_').str[1:].str.join('_')

    # Create pivot table
    pivot = team_data.pivot(index='depth', columns='lateral', values='avg_void_score')

    # Reorder for proper field visualization
    depth_order = ['deep', 'intermediate', 'shallow']
    lateral_order = ['far_left', 'left_hash', 'middle', 'right_hash', 'far_right']

    pivot = pivot.reindex(index=depth_order, columns=lateral_order)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', center=60,
                cbar_kws={'label': 'Avg Void Score (lower = better)'}, ax=ax)
    ax.set_title(f'{team} - Zone Vulnerability Heatmap (at Throw)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Lateral Position', fontweight='bold')
    ax.set_ylabel('Depth from LOS', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig

def main():
    """Run team defensive analysis."""
    print("=" * 70)
    print("TEAM DEFENSIVE ANALYSIS")
    print("=" * 70)
    print()

    # Load data
    data = load_data()

    # Calculate metrics
    team_metrics = calculate_team_defensive_metrics(data)
    zone_metrics = calculate_team_zone_strengths(data)

    # Display results
    print("=" * 70)
    print("TOP 10 DEFENSES (by composite rating)")
    print("=" * 70)
    print(team_metrics[['team', 'plays_faced', 'defensive_rating', 'avg_void_score',
                        'target_zone_void_score', 'completion_rate_against',
                        'deep_zone_void_score']].head(10).to_string(index=False))
    print()

    print("=" * 70)
    print("BOTTOM 10 DEFENSES (by composite rating)")
    print("=" * 70)
    print(team_metrics[['team', 'plays_faced', 'defensive_rating', 'avg_void_score',
                        'target_zone_void_score', 'completion_rate_against',
                        'deep_zone_void_score']].tail(10).to_string(index=False))
    print()

    # Save full results
    output_dir = Path('data/consolidated')
    team_metrics.to_csv(output_dir / 'team_defensive_metrics.csv', index=False)
    zone_metrics.to_csv(output_dir / 'team_zone_metrics.csv', index=False)
    print(f"✓ Saved metrics to {output_dir}/")
    print()

    # Create visualizations
    print("Creating visualizations...")
    viz_dir = Path('visualizations')
    viz_dir.mkdir(exist_ok=True)

    plot_team_rankings(team_metrics, save_path=viz_dir / 'team_defensive_rankings.png')

    # Plot heatmaps for top 3 and bottom 3 teams
    print("\nCreating zone heatmaps for select teams...")
    top_3_teams = team_metrics.head(3)['team'].tolist()
    bottom_3_teams = team_metrics.tail(3)['team'].tolist()

    for team in top_3_teams + bottom_3_teams:
        safe_name = team.replace(' ', '_')
        plot_team_zone_heatmap(zone_metrics, team,
                               save_path=viz_dir / f'zone_heatmap_{safe_name}.png')

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"Outputs:")
    print(f"  - Metrics: {output_dir}/team_defensive_metrics.csv")
    print(f"  - Zone Metrics: {output_dir}/team_zone_metrics.csv")
    print(f"  - Visualizations: {viz_dir}/")

    return team_metrics, zone_metrics

if __name__ == '__main__':
    team_metrics, zone_metrics = main()
