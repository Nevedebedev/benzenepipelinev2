#!/usr/bin/env python3
"""
Investigate why PINN overpredicts in certain months
Analyze meteorological conditions and emissions by month
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load all facility data
SYNCED_DIR = Path('/Users/neevpratap/Desktop/madis_data_desktop_updated/synced')
facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))

all_data = []
for f in facility_files:
    if 'summary' in f.name:
        continue
    df = pd.read_csv(f)
    df['timestamp'] = pd.to_datetime(df['t'])
    all_data.append(df)

merged = pd.concat(all_data, ignore_index=True)
merged['month'] = merged['timestamp'].dt.month
merged['hour'] = merged['timestamp'].dt.hour

print("="*80)
print("METEOROLOGICAL CONDITIONS BY MONTH")
print("="*80)

# Group by month and analyze
monthly_stats = merged.groupby('month').agg({
    'wind_u': ['mean', 'std', 'min', 'max'],
    'wind_v': ['mean', 'std', 'min', 'max'],
    'D': ['mean', 'std', 'min', 'max'],
    'Q_total': ['mean', 'std', 'min', 'max'],
    'source_diameter': 'mean'
}).round(4)

print("\n1. WIND U-COMPONENT (m/s) by month:")
print(monthly_stats['wind_u'])

print("\n2. WIND V-COMPONENT (m/s) by month:")
print(monthly_stats['wind_v'])

# Calculate wind speed magnitude
merged['wind_speed'] = np.sqrt(merged['wind_u']**2 + merged['wind_v']**2)
print("\n3. WIND SPEED MAGNITUDE (m/s) by month:")
print(merged.groupby('month')['wind_speed'].agg(['mean', 'std', 'min', 'max']).round(4))

print("\n4. DIFFUSION COEFFICIENT (D) by month:")
print(monthly_stats['D'])

print("\n5. EMISSION RATE (Q_total) by month:")
print(monthly_stats['Q_total'])

print("\n6. SOURCE DIAMETER by month:")
print(monthly_stats['source_diameter'])

# Count number of active facilities per month
print("\n7. NUMBER OF TIMESTAMPS PER MONTH:")
print(merged.groupby('month').size())

# Check for calm conditions (low wind = high concentrations)
print("\n" + "="*80)
print("CALM CONDITIONS ANALYSIS (Wind Speed < 1 m/s)")
print("="*80)

calm_by_month = merged[merged['wind_speed'] < 1.0].groupby('month').size()
total_by_month = merged.groupby('month').size()
calm_pct = (calm_by_month / total_by_month * 100).round(2)

print("\nPercent of timestamps with calm conditions (< 1 m/s):")
for month in range(1, 13):
    if month in calm_pct.index:
        print(f"  Month {month:2d}: {calm_pct[month]:5.2f}%")
    else:
        print(f"  Month {month:2d}: {0:5.2f}%")

# Analyze diffusion coefficient (related to atmospheric stability)
print("\n" + "="*80)
print("ATMOSPHERIC STABILITY ANALYSIS")
print("="*80)

print("\nDiffusion coefficient percentiles by month:")
for month in range(1, 13):
    month_data = merged[merged['month'] == month]['D']
    if len(month_data) > 0:
        p25, p50, p75 = np.percentile(month_data, [25, 50, 75])
        print(f"  Month {month:2d}: Q1={p25:.4f}, Median={p50:.4f}, Q3={p75:.4f}")

# Check emission rates
print("\n" + "="*80)
print("EMISSION RATE ANALYSIS")
print("="*80)

print("\nEmission rate (Q_total) percentiles by month:")
for month in range(1, 13):
    month_data = merged[merged['month'] == month]['Q_total']
    if len(month_data) > 0:
        p25, p50, p75 = np.percentile(month_data, [25, 50, 75])
        mean_q = month_data.mean()
        print(f"  Month {month:2d}: Mean={mean_q:.6f}, Median={p50:.6f}, Q3={p75:.6f}")

# Load PINN predictions to correlate
print("\n" + "="*80)
print("CORRELATION: PINN PREDICTION vs CONDITIONS")
print("="*80)

pinn_df = pd.read_csv('/Users/neevpratap/simpletesting/nn2trainingdata/total_concentrations.csv')
pinn_df['timestamp'] = pd.to_datetime(pinn_df['timestamp'])
pinn_df['month'] = pinn_df['timestamp'].dt.month

sensor_cols = [c for c in pinn_df.columns if c.startswith('sensor_')]
pinn_df['mean_concentration'] = pinn_df[sensor_cols].mean(axis=1)

# Aggregate facility data to hourly (same as PINN)
facility_hourly = merged.groupby('timestamp').agg({
    'wind_speed': 'mean',
    'D': 'mean',
    'Q_total': 'sum',  # Total emissions across all facilities
}).reset_index()

# Merge with PINN
combined = pd.merge(pinn_df[['timestamp', 'month', 'mean_concentration']], 
                   facility_hourly, on='timestamp', how='inner')

print("\nCorrelation with PINN concentration:")
print(f"  Wind Speed:  {combined['wind_speed'].corr(combined['mean_concentration']):.4f}")
print(f"  Diffusion:   {combined['D'].corr(combined['mean_concentration']):.4f}")
print(f"  Q_total:     {combined['Q_total'].corr(combined['mean_concentration']):.4f}")

# Monthly breakdown
print("\nMonthly average conditions and PINN predictions:")
print(f"{'Month':<6} | {'Wind (m/s)':<12} | {'Diffusion':<12} | {'Q_total':<12} | {'PINN (ppb)':<12}")
print("-" * 70)

for month in range(1, 13):
    month_comb = combined[combined['month'] == month]
    if len(month_comb) > 0:
        avg_wind = month_comb['wind_speed'].mean()
        avg_d = month_comb['D'].mean()
        avg_q = month_comb['Q_total'].mean()
        avg_pinn = month_comb['mean_concentration'].mean()
        print(f"{month:<6} | {avg_wind:>10.4f}   | {avg_d:>10.4f}   | {avg_q:>10.6f}   | {avg_pinn:>10.4f}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Identify problematic months
high_pinn_months = combined.groupby('month')['mean_concentration'].mean()
high_pinn_months = high_pinn_months[high_pinn_months > 20].index.tolist()

print(f"\nMonths with high PINN predictions (>20 ppb): {high_pinn_months}")

for month in high_pinn_months:
    month_data = combined[combined['month'] == month]
    print(f"\nMonth {month}:")
    print(f"  Mean wind speed: {month_data['wind_speed'].mean():.4f} m/s")
    print(f"  Mean diffusion:  {month_data['D'].mean():.4f}")
    print(f"  Mean Q_total:    {month_data['Q_total'].mean():.6f}")
    print(f"  PINN prediction: {month_data['mean_concentration'].mean():.4f} ppb")
    
    # Check for stagnation
    low_wind_pct = (month_data['wind_speed'] < 1).sum() / len(month_data) * 100
    print(f"  Low wind (<1 m/s): {low_wind_pct:.1f}% of time")

print("\n" + "="*80)
