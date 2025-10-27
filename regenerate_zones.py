#!/usr/bin/env python3
"""
Regenerate zone vulnerability timeseries dataset with bug fix.
"""
import sys
sys.path.insert(0, '/Users/cbulacan/cbula-git/kaggle/src')

from nfl_analysis import NFLDataConsolidator
import pandas as pd

# Load existing consolidated data
consolidator = NFLDataConsolidator(
    data_dir='data/raw/114239_nfl_competition_files_published_analytics_final',
    output_dir='data/consolidated'
)

# Load the master datasets that already exist
print('Loading existing consolidated datasets...')
consolidator.master_input = pd.read_parquet('data/consolidated/master_input.parquet')
consolidator.supplementary = pd.read_parquet('data/consolidated/supplementary.parquet')

print(f'Loaded {len(consolidator.master_input):,} input frames')
print(f'Loaded {len(consolidator.supplementary):,} plays')

# Regenerate zone vulnerability dataset with the fix
print('\nRegenerating zone vulnerability timeseries...')
zone_vulnerability = consolidator.create_zone_vulnerability_timeseries()

# Save the corrected dataset
print('\nSaving corrected dataset...')
consolidator.save_dataset(zone_vulnerability, 'zone_vulnerability_timeseries')

print('\n' + '=' * 70)
print('DATASET REGENERATION COMPLETE')
print('=' * 70)
print(f'Total zone records: {len(zone_vulnerability):,}')
print(f'Void score range: [{zone_vulnerability["zone_void_score"].min():.2f}, {zone_vulnerability["zone_void_score"].max():.2f}]')
print(f'Average void score: {zone_vulnerability["zone_void_score"].mean():.2f}')
print('\n✓ Zone vulnerability dataset regenerated successfully!')
