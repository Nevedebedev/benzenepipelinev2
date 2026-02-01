#!/usr/bin/env python3
"""
Aggregate PINN Predictions Across All Sources (3-Hour Forecast)

Superimposes (sums) concentration predictions from all 20 facilities
for each sensor-timestamp combination, then shifts timestamps forward
by 3 hours to represent forecast time (T+3).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Output columns matching sensors_final_synced.csv format
SENSOR_IDS = [
    '482010026', '482010057', '482010069', '482010617', 
    '482010803', '482011015', '482011035', '482011039', '482016000'
]

def load_all_predictions(pinn_dir):
    """Load all facility prediction files"""
    pinn_path = Path(pinn_dir)
    prediction_files = sorted(pinn_path.glob('*_pinn_predictions.csv'))
    
    print(f"Loading {len(prediction_files)} prediction files...")
    
    all_predictions = []
    for pred_file in tqdm(prediction_files, desc="Loading files"):
        df = pd.read_csv(pred_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        all_predictions.append(df)
    
    # Concatenate all predictions
    combined = pd.concat(all_predictions, ignore_index=True)
    print(f"Total predictions loaded: {len(combined):,}")
    
    return combined


def superimpose_predictions(predictions_df):
    """
    Sum predictions across all facilities for each (timestamp, sensor) pair
    """
    print("\nSuperimposing predictions across all sources...")
    
    # Group by timestamp and sensor, sum the concentrations
    aggregated = predictions_df.groupby(['timestamp', 'sensor_id'], as_index=False).agg({
        'predicted_concentration': 'sum'
    })
    
    print(f"Unique timestamps: {aggregated['timestamp'].nunique():,}")
    print(f"Unique sensors: {aggregated['sensor_id'].nunique()}")
    
    return aggregated


def pivot_to_wide_format(aggregated_df):
    """
    Pivot from long format to wide format matching sensors_final_synced.csv
    Columns: t, sensor_482010026, sensor_482010057, ...
    
    IMPORTANT: Applies 3-hour forecast time-shift
    """
    print("\nPivoting to wide format...")
    
    # Pivot: rows=timestamp, columns=sensor_id, values=concentration
    wide_df = aggregated_df.pivot(
        index='timestamp',
        columns='sensor_id',
        values='predicted_concentration'
    )
    
    # Rename columns to match original format
    wide_df.columns = [f'sensor_{col}' for col in wide_df.columns]
    
    # Reset index and rename timestamp column to 't'
    wide_df = wide_df.reset_index()
    wide_df = wide_df.rename(columns={'timestamp': 't'})
    
    # ═══════════════════════════════════════════════════════════════
    # APPLY 3-HOUR FORECAST TIME-SHIFT
    # ═══════════════════════════════════════════════════════════════
    # PINN predictions made at time T are forecasts for T+3 hours
    # This aligns predictions with the actual future measurements
    print("  ⏰ Applying 3-hour forecast time-shift...")
    print(f"     Original time range: {wide_df['t'].min()} to {wide_df['t'].max()}")
    
    wide_df['t'] = wide_df['t'] + pd.Timedelta(hours=3)
    
    print(f"     Forecast time range: {wide_df['t'].min()} to {wide_df['t'].max()}")
    print("     (Predictions now represent 3-hour forecasts)")
    
    # Sort by timestamp
    wide_df = wide_df.sort_values('t')
    
    # Reorder columns to match sensors_final_synced.csv order
    column_order = ['t'] + [f'sensor_{sid}' for sid in SENSOR_IDS]
    # Only keep columns that exist
    existing_columns = [col for col in column_order if col in wide_df.columns]
    wide_df = wide_df[existing_columns]
    
    print(f"Output shape: {wide_df.shape}")
    print(f"Columns: {list(wide_df.columns)}")
    
    return wide_df


def main():
    pinn_dir = 'pinn_predictions'
    output_file = 'sensors_pinn_superimposed_forecast.csv'  # 3-hour forecast data
    
    # Load all predictions
    all_predictions = load_all_predictions(pinn_dir)
    
    # Superimpose (sum) across all sources
    aggregated = superimpose_predictions(all_predictions)
    
    # Pivot to wide format
    wide_format = pivot_to_wide_format(aggregated)
    
    # Save to CSV
    print(f"\nSaving to {output_file}...")
    wide_format.to_csv(output_file, index=False)
    
    # Display summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Date range: {wide_format['t'].min()} to {wide_format['t'].max()}")
    print(f"Total timestamps: {len(wide_format):,}")
    print(f"Total sensors: {len(wide_format.columns) - 1}")
    
    # Show sample statistics for each sensor
    print("\nSuperimposed Concentration Statistics (model scale):")
    for col in wide_format.columns:
        if col.startswith('sensor_'):
            stats = wide_format[col].describe()
            print(f"\n{col}:")
            print(f"  Count: {stats['count']:.0f}")
            print(f"  Mean:  {stats['mean']:.2f}")
            print(f"  Std:   {stats['std']:.2f}")
            print(f"  Min:   {stats['min']:.2f}")
            print(f"  Max:   {stats['max']:.2f}")
    
    print("\n" + "="*60)
    print(f"Output saved to: {output_file}")
    print("="*60)


if __name__ == '__main__':
    main()
