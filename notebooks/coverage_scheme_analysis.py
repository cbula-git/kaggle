#!/usr/bin/env python3
"""
Coverage Scheme Analysis
=========================
Analyze Man vs Zone coverage effectiveness by team.
"""

import sys
sys.path.insert(0, '/Users/cbulacan/cbula-git/kaggle/src')

import pandas as pd
import numpy as np

def main():
    # Load data
    print("Loading data...")
    zone_data = pd.read_parquet('data/consolidated/zone_vulnerability_timeseries.parquet')
    supp_data = pd.read_parquet('data/consolidated/supplementary.parquet')

    # Merge
    data = zone_data.merge(
        supp_data[['game_id', 'play_id', 'defensive_team', 'team_coverage_man_zone', 'pass_result']],
        on=['game_id', 'play_id'],
        how='left'
    )

    # Filter to throw frames
    throw_data = data[data['phase'] == 'at_throw']

    print('=' * 70)
    print('COVERAGE SCHEME EFFECTIVENESS BY TEAM')
    print('=' * 70)
    print()

    # Analyze by coverage type
    coverage_stats = []

    for team in sorted(data['defensive_team'].dropna().unique()):
        team_data = throw_data[throw_data['defensive_team'] == team]

        for coverage_type, coverage_name in [('MAN_COVERAGE', 'Man'), ('ZONE_COVERAGE', 'Zone')]:
            cov_data = team_data[team_data['team_coverage_man_zone'] == coverage_type]

            if len(cov_data) == 0:
                continue

            # Get target zones
            targets = cov_data[cov_data['is_target_zone'] == True]
            target_plays = targets[['game_id', 'play_id', 'pass_result']].drop_duplicates()

            if len(target_plays) == 0:
                continue

            coverage_stats.append({
                'team': team,
                'coverage': coverage_name,
                'plays': len(target_plays),
                'avg_void': cov_data['zone_void_score'].mean(),
                'target_void': targets['zone_void_score'].mean(),
                'comp_rate': (target_plays['pass_result'] == 'C').mean(),
            })

    stats_df = pd.DataFrame(coverage_stats)

    if len(stats_df) == 0:
        print("No coverage data found!")
        return

    print('Best MAN Coverage Defenses (by void score):')
    print()
    man_df = stats_df[stats_df['coverage'] == 'Man'].sort_values('avg_void').head(10)
    print('  Team  Plays  Void Score  Target Void  Comp %')
    print('  ' + '-' * 50)
    for idx, row in man_df.iterrows():
        print(f"  {row['team']:3s}   {row['plays']:4.0f}     {row['avg_void']:5.2f}      {row['target_void']:5.2f}     {row['comp_rate']*100:5.1f}%")

    print()
    print('Best ZONE Coverage Defenses (by void score):')
    print()
    zone_df = stats_df[stats_df['coverage'] == 'Zone'].sort_values('avg_void').head(10)
    print('  Team  Plays  Void Score  Target Void  Comp %')
    print('  ' + '-' * 50)
    for idx, row in zone_df.iterrows():
        print(f"  {row['team']:3s}   {row['plays']:4.0f}     {row['avg_void']:5.2f}      {row['target_void']:5.2f}     {row['comp_rate']*100:5.1f}%")

    print()
    print('=' * 70)
    print('COVERAGE SCHEME PREFERENCE')
    print('=' * 70)
    print()

    # Compare Man vs Zone for each team
    team_comparison = []
    for team in stats_df['team'].unique():
        team_stats = stats_df[stats_df['team'] == team]
        man_row = team_stats[team_stats['coverage'] == 'Man']
        zone_row = team_stats[team_stats['coverage'] == 'Zone']

        if len(man_row) > 0 and len(zone_row) > 0:
            man_void = man_row.iloc[0]['avg_void']
            zone_void = zone_row.iloc[0]['avg_void']
            diff = man_void - zone_void

            team_comparison.append({
                'team': team,
                'man_void': man_void,
                'zone_void': zone_void,
                'difference': diff,
                'better_at': 'Man' if diff < 0 else 'Zone'
            })

    comp_df = pd.DataFrame(team_comparison)

    print('Teams BETTER at Man Coverage (Man void < Zone void):')
    better_man = comp_df[comp_df['better_at'] == 'Man'].sort_values('difference').head(8)
    print('  Team  Man Void  Zone Void  Difference')
    print('  ' + '-' * 45)
    for idx, row in better_man.iterrows():
        print(f"  {row['team']:3s}    {row['man_void']:5.2f}     {row['zone_void']:5.2f}     {row['difference']:+5.2f}")

    print()
    print('Teams BETTER at Zone Coverage (Zone void < Man void):')
    better_zone = comp_df[comp_df['better_at'] == 'Zone'].sort_values('difference', ascending=False).head(8)
    print('  Team  Man Void  Zone Void  Difference')
    print('  ' + '-' * 45)
    for idx, row in better_zone.iterrows():
        print(f"  {row['team']:3s}    {row['man_void']:5.2f}     {row['zone_void']:5.2f}     {row['difference']:+5.2f}")

    print()
    print('=' * 70)

    # Save results
    stats_df.to_csv('data/consolidated/coverage_scheme_stats.csv', index=False)
    comp_df.to_csv('data/consolidated/team_coverage_preference.csv', index=False)
    print("✓ Saved results to data/consolidated/")

if __name__ == '__main__':
    main()
