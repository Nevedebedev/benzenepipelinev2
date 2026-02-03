#!/usr/bin/env python3
"""
Run Pipeline on All of 2019

This script runs the full pipeline (PINN + NN2) on all timestamps in 2019
using the facility synced training data files.

Process:
1. Load facility synced training data files
2. For each timestamp in 2019:
   - Extract facility parameters from synced data
   - Convert to format expected by ConcentrationPredictor
   - Run pipeline to generate full domain predictions
   - Save results to continuous time series CSVs
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append("/Users/neevpratap/simpletesting")

from concentration_predictor import ConcentrationPredictor
from config import FACILITIES

# Paths
SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"

def load_facility_data():
    """Load all facility synced training data files"""
    print("Loading facility data...")
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    facility_data_dict = {}
    for f in facility_files:
        facility_name = f.stem.replace('_synced_training_data', '')
        df = pd.read_csv(f)
        df['timestamp'] = pd.to_datetime(df['t'])
        facility_data_dict[facility_name] = df
        print(f"  Loaded {facility_name}: {len(df)} rows")
    
    return facility_data_dict

def get_facility_params_for_timestamp(facility_data_dict, timestamp):
    """
    Extract facility parameters for a given timestamp
    
    Args:
        facility_data_dict: Dict of {facility_name: DataFrame}
        timestamp: Timestamp to extract data for
        
    Returns:
        dict: {facility_name: {source_x_cartesian, source_y_cartesian, ...}}
    """
    facility_params = {}
    
    for facility_name, df in facility_data_dict.items():
        # Find row matching this timestamp
        facility_row = df[df['timestamp'] == timestamp]
        
        if len(facility_row) == 0:
            continue
        
        row = facility_row.iloc[0]
        
        # Extract parameters in format expected by ConcentrationPredictor
        params = {
            'source_x_cartesian': row['source_x_cartesian'],
            'source_y_cartesian': row['source_y_cartesian'],
            'source_diameter': row['source_diameter'],
            'Q': row['Q_total'],  # Use Q_total from synced data
            'wind_u': row['wind_u'],
            'wind_v': row['wind_v'],
            'D': row['D'],
        }
        
        facility_params[facility_name] = params
    
    return facility_params

def get_available_timestamps(facility_data_dict, year=2019):
    """
    Get all timestamps available in facility data for a given year
    
    Returns:
        sorted list of timestamps
    """
    all_timestamps = set()
    
    for facility_name, df in facility_data_dict.items():
        # Filter to year
        year_mask = (df['timestamp'] >= f'{year}-01-01') & (df['timestamp'] < f'{year+1}-01-01')
        facility_timestamps = df[year_mask]['timestamp'].unique()
        all_timestamps.update(facility_timestamps)
    
    return sorted(all_timestamps)

def main():
    print("="*80)
    print("PIPELINE RUN - FULL YEAR 2019")
    print("="*80)
    print()
    
    # Load facility data
    facility_data_dict = load_facility_data()
    
    if len(facility_data_dict) == 0:
        print("ERROR: No facility data files found!")
        return
    
    # Get available timestamps for 2019
    print("\nFinding available timestamps in 2019...")
    timestamps = get_available_timestamps(facility_data_dict, year=2019)
    print(f"  Found {len(timestamps)} unique timestamps")
    
    if len(timestamps) == 0:
        print("ERROR: No timestamps found for 2019!")
        return
    
    # Initialize predictor
    print("\nInitializing ConcentrationPredictor...")
    predictor = ConcentrationPredictor(grid_resolution=100)
    print("  ✓ Predictor initialized\n")
    
    # Process each timestamp
    print("="*80)
    print("PROCESSING TIMESTAMPS")
    print("="*80)
    print()
    
    successful = 0
    failed = 0
    
    for timestamp in tqdm(timestamps, desc="Processing timestamps"):
        try:
            # Get facility parameters for this timestamp
            facility_params = get_facility_params_for_timestamp(facility_data_dict, timestamp)
            
            if len(facility_params) == 0:
                # Skip if no facility data available
                continue
            
            # Current time is 3 hours before forecast time
            # Predictions made at time t (using met data from t) are forecasts for t+3
            current_time = timestamp
            forecast_time = timestamp + pd.Timedelta(hours=3)
            
            # Run pipeline
            pinn_field, nn2_field, _ = predictor.predict_full_domain(
                facility_params, current_time
            )
            
            successful += 1
            
            # Print progress every 100 timestamps
            if successful % 100 == 0:
                print(f"\n  Progress: {successful} successful, {failed} failed")
                print(f"  Latest: {timestamp.strftime('%Y-%m-%d %H:%M')}")
                print(f"  PINN range: {pinn_field.min():.4f} - {pinn_field.max():.4f} ppb")
                print(f"  NN2 range: {nn2_field.min():.4f} - {nn2_field.max():.4f} ppb")
        
        except Exception as e:
            failed += 1
            print(f"\n  ERROR at {timestamp}: {str(e)}")
            continue
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"\nSuccessful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(timestamps)}")
    print("\nResults saved to:")
    print(f"  - {predictor.output_dir / 'superimposed_concentrations_timeseries.csv'}")
    print(f"  - {predictor.output_dir / 'nn2_corrected_domain_timeseries.csv'}")
    print(f"  - {predictor.predictions_dir / 'latest_spatial_grid.csv'}")
    print("="*80)

if __name__ == '__main__':
    main()

