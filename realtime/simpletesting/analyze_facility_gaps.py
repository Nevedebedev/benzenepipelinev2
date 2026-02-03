#!/usr/bin/env python3
"""
Analyze gaps in facility data
- Which facilities have gaps
- When do gaps occur
- What values are in the gaps
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

SYNCED_DIR = Path('/Users/neevpratap/Desktop/madis_data_desktop_updated/synced')

print("="*80)
print("FACILITY DATA GAPS ANALYSIS")
print("="*80)

# Load all facility files
facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
facility_files = [f for f in facility_files if 'summary' not in f.name]

print(f"\nFound {len(facility_files)} facility files")

# Create expected timeline (hourly for all of 2019)
expected_timeline = pd.date_range('2019-01-01 00:00:00', '2019-12-31 23:00:00', freq='h')
print(f"Expected timestamps in 2019: {len(expected_timeline)} (hourly)")

# Analyze each facility
facility_data = {}

for f in facility_files:
    facility_name = f.stem.replace('_synced_training_data', '')
    df = pd.read_csv(f)
    df['timestamp'] = pd.to_datetime(df['t'])
    
    # Check coverage
    actual_timestamps = set(df['timestamp'])
    missing_count = len(expected_timeline) - len(actual_timestamps)
    coverage_pct = len(actual_timestamps) / len(expected_timeline) * 100
    
    # Check for zeros in Q_total
    zero_q = (df['Q_total'] == 0).sum()
    zero_q_pct = zero_q / len(df) * 100
    
    facility_data[facility_name] = {
        'total_records': len(df),
        'missing_timestamps': missing_count,
        'coverage_pct': coverage_pct,
        'zero_q_count': zero_q,
        'zero_q_pct': zero_q_pct,
        'q_min': df['Q_total'].min(),
        'q_max': df['Q_total'].max(),
        'q_mean': df['Q_total'].mean(),
        'timestamps': actual_timestamps,
        'df': df
    }

# Print summary
print("\n" + "="*80)
print("FACILITY COVERAGE SUMMARY")
print("="*80)
print(f"\n{'Facility':<30} | {'Records':<8} | {'Missing':<8} | {'Coverage':<10} | {'Zeros':<8}")
print("-" * 80)

for name, data in sorted(facility_data.items()):
    print(f"{name:<30} | {data['total_records']:<8} | {data['missing_timestamps']:<8} | "
          f"{data['coverage_pct']:>7.2f}%   | {data['zero_q_count']:<8}")

# Find common missing periods
print("\n" + "="*80)
print("TEMPORAL GAP ANALYSIS")
print("="*80)

# Count how many facilities are missing at each timestamp
all_timestamps = set(expected_timeline)
timestamp_coverage = {}

for ts in expected_timeline:
    facilities_present = sum(1 for data in facility_data.values() if ts in data['timestamps'])
    timestamp_coverage[ts] = {
        'present': facilities_present,
        'missing': len(facility_data) - facilities_present
    }

# Group by month and analyze
coverage_df = pd.DataFrame([
    {'timestamp': k, 'present': v['present'], 'missing': v['missing']}
    for k, v in timestamp_coverage.items()
])
coverage_df['month'] = coverage_df['timestamp'].dt.month

monthly_coverage = coverage_df.groupby('month').agg({
    'present': 'mean',
    'missing': 'mean'
}).round(2)

print("\nAverage number of facilities reporting per hour by month:")
print(f"{'Month':<6} | {'Avg Facilities Present':<25} | {'Avg Missing':<15}")
print("-" * 50)
for month, row in monthly_coverage.iterrows():
    print(f"{month:<6} | {row['present']:>22.2f}   | {row['missing']:>12.2f}")

# Find timestamps with fewest facilities
worst_coverage = coverage_df.nsmallest(10, 'present')
print("\n10 timestamps with WORST coverage (fewest facilities):")
print(f"{'Timestamp':<20} | {'Facilities Present':<20} | {'Facilities Missing':<20}")
print("-" * 65)
for _, row in worst_coverage.iterrows():
    print(f"{row['timestamp']}   | {row['present']:>18}   | {row['missing']:>18}")

# Check Q_total behavior
print("\n" + "="*80)
print("EMISSION RATE (Q_total) ANALYSIS")
print("="*80)

for name, data in sorted(facility_data.items())[:5]:  # Show first 5 facilities
    df = data['df']
    print(f"\n{name}:")
    print(f"  Records: {len(df)}")
    print(f"  Q_total range: [{data['q_min']:.6f}, {data['q_max']:.6f}]")
    print(f"  Q_total mean: {data['q_mean']:.6f}")
    print(f"  Zero Q_total: {data['zero_q_count']} ({data['zero_q_pct']:.2f}%)")
    
    # Show distribution
    q_vals = df['Q_total'].value_counts().sort_index()
    if len(q_vals) <= 10:
        print(f"  Q_total value distribution:")
        for val, count in q_vals.items():
            print(f"    {val:.6f}: {count} times")

# Check if gaps are filled with zeros or just missing
print("\n" + "="*80)
print("GAP FILLING STRATEGY")
print("="*80)

# For each month, check what happens during gaps
sample_facility = list(facility_data.keys())[0]
sample_df = facility_data[sample_facility]['df']

# Find a gap
all_ts = set(expected_timeline)
actual_ts = set(sample_df['timestamp'])
missing_ts = sorted(list(all_ts - actual_ts))[:5]

if missing_ts:
    print(f"\nSample missing timestamps from {sample_facility}:")
    for ts in missing_ts:
        print(f"  {ts} - NOT IN DATA (gap)")
else:
    print(f"\n{sample_facility} has complete coverage!")

# Check if there are timestamps with Q=0
zero_timestamps = sample_df[sample_df['Q_total'] == 0]['timestamp'].head()
if len(zero_timestamps) > 0:
    print(f"\nSample timestamps with Q_total=0 from {sample_facility}:")
    for ts in zero_timestamps:
        print(f"  {ts} - Q_total = 0 (present but zero)")

# Aggregate analysis - what happens when we merge?
print("\n" + "="*80)
print("AGGREGATION IMPACT")
print("="*80)

# Merge all facilities to see total Q
all_data = []
for name, data in facility_data.items():
    df = data['df'].copy()
    df['facility'] = name
    all_data.append(df[['timestamp', 'Q_total', 'facility']])

merged = pd.concat(all_data)
merged['month'] = merged['timestamp'].dt.month

# Group by timestamp to see total Q
hourly_total = merged.groupby('timestamp').agg({
    'Q_total': 'sum',
    'facility': 'count'
}).rename(columns={'facility': 'num_facilities'})

hourly_total['month'] = hourly_total.index.to_series().dt.month

print("\nMonthly statistics after aggregating all facilities:")
print(f"{'Month':<6} | {'Avg Facilities/Hour':<20} | {'Avg Q_total':<15} | {'Max Q_total':<15}")
print("-" * 70)

for month in range(1, 13):
    month_data = hourly_total[hourly_total['month'] == month]
    if len(month_data) > 0:
        avg_fac = month_data['num_facilities'].mean()
        avg_q = month_data['Q_total'].mean()
        max_q = month_data['Q_total'].max()
        print(f"{month:<6} | {avg_fac:>18.2f}   | {avg_q:>13.6f}   | {max_q:>13.6f}")

# Find hours with abnormally high Q_total
high_q = hourly_total.nlargest(10, 'Q_total')
print("\n10 hours with HIGHEST total Q_total:")
print(f"{'Timestamp':<20} | {'Num Facilities':<16} | {'Total Q':<15}")
print("-" * 60)
for ts, row in high_q.iterrows():
    print(f"{ts}   | {row['num_facilities']:>14}   | {row['Q_total']:>13.6f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Calculate correlation between facility count and total Q
correlation = hourly_total['num_facilities'].corr(hourly_total['Q_total'])
print(f"\nCorrelation between facility count and total Q: {correlation:.4f}")

if correlation > 0.5:
    print("✓ STRONG positive correlation: More facilities → Higher total Q")
    print("  This is EXPECTED and correct behavior")
elif correlation < 0:
    print("⚠️ NEGATIVE correlation: Fewer facilities → Higher total Q")
    print("  This suggests an AGGREGATION BUG")
else:
    print("~ WEAK correlation: Facility count doesn't strongly affect total Q")

print("\n" + "="*80)
