#!/usr/bin/env python3
"""
Analyze 2019 EDF Sensor Data to Find Peak Hazard Events

Identifies the highest benzene concentration events in 2019 from actual EDF sensor data,
extracts the corresponding weather/facility conditions, and prepares data for PINN visualization.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add paths
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

# Paths
SENSOR_DATA_PATHS = [
    "/Users/neevpratap/Downloads/sensors_final_synced.csv",
    str(Path(__file__).parent / 'simpletesting/nn2trainingdata/sensors_final_synced.csv')
]

SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
OUTPUT_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sensor IDs (9 sensors)
SENSOR_IDS = [
    'sensor_482010026', 'sensor_482010057', 'sensor_482010069', 'sensor_482010617',
    'sensor_482010803', 'sensor_482011015', 'sensor_482011035', 'sensor_482011039',
    'sensor_482016000'
]

def load_edf_sensor_data():
    """Load EDF sensor data from available path"""
    print("Loading EDF sensor data...")
    
    for path in SENSOR_DATA_PATHS:
        if Path(path).exists():
            print(f"  Found: {path}")
            df = pd.read_csv(path)
            
            # Handle timestamp column
            if 't' in df.columns:
                df = df.rename(columns={'t': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter to 2019
            year_mask = (df['timestamp'] >= '2019-01-01') & (df['timestamp'] < '2020-01-01')
            df = df[year_mask].reset_index(drop=True)
            
            print(f"  Loaded {len(df)} timestamps from 2019")
            return df
    
    raise FileNotFoundError("Could not find EDF sensor data file")

def calculate_hazard_metrics(df):
    """Calculate hazard metrics from EDF sensor readings"""
    print("\nCalculating hazard metrics...")
    
    # Get sensor columns
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    
    # Calculate metrics per timestamp
    results = []
    
    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        sensor_values = row[sensor_cols].values
        
        # Replace NaN with 0
        sensor_values = np.nan_to_num(sensor_values, nan=0.0)
        
        # Calculate metrics
        max_concentration = float(np.max(sensor_values))
        mean_concentration = float(np.mean(sensor_values))
        total_exposure = float(np.sum(sensor_values))
        
        # Find peak sensor
        peak_idx = int(np.argmax(sensor_values))
        peak_sensor_id = sensor_cols[peak_idx]
        
        # Store all sensor values as dict
        sensor_dict = {col: float(sensor_values[i]) for i, col in enumerate(sensor_cols)}
        
        results.append({
            'timestamp': timestamp,
            'max_concentration_edf': max_concentration,
            'mean_concentration_edf': mean_concentration,
            'total_exposure_edf': total_exposure,
            'peak_sensor_id': peak_sensor_id,
            **sensor_dict
        })
    
    results_df = pd.DataFrame(results)
    
    # Rank by max concentration
    results_df = results_df.sort_values('max_concentration_edf', ascending=False)
    results_df['hazard_rank'] = range(1, len(results_df) + 1)
    
    print(f"  Calculated metrics for {len(results_df)} timestamps")
    print(f"  Max concentration found: {results_df['max_concentration_edf'].max():.2f} ppb")
    print(f"  Mean max concentration: {results_df['max_concentration_edf'].mean():.2f} ppb")
    
    return results_df

def identify_top_hazards(results_df, top_n=10):
    """Identify top N hazard events"""
    print(f"\nIdentifying top {top_n} hazard events...")
    
    top_hazards = results_df.head(top_n).copy()
    
    print("\nTop Hazard Events:")
    print("=" * 100)
    print(f"{'Rank':<6} {'Timestamp':<20} {'Max EDF (ppb)':<15} {'Mean EDF (ppb)':<15} {'Peak Sensor':<20}")
    print("-" * 100)
    
    for idx, row in top_hazards.iterrows():
        print(f"{row['hazard_rank']:<6} {str(row['timestamp']):<20} {row['max_concentration_edf']:<15.2f} "
              f"{row['mean_concentration_edf']:<15.2f} {row['peak_sensor_id']:<20}")
    
    return top_hazards

def extract_facility_conditions(hazard_timestamp):
    """Extract weather/facility conditions for a given hazard timestamp"""
    # Note: If EDF reading is at time T, we need met data from T-3 hours
    met_timestamp = hazard_timestamp - pd.Timedelta(hours=3)
    
    facility_params = {}
    
    # Load all facility files
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    for facility_file in facility_files:
        facility_name = facility_file.stem.replace('_synced_training_data', '')
        
        try:
            df_fac = pd.read_csv(facility_file)
            df_fac['t'] = pd.to_datetime(df_fac['t'])
            
            # Find row matching met_timestamp
            fac_row = df_fac[df_fac['t'] == met_timestamp]
            
            if not fac_row.empty:
                fac_row = fac_row.iloc[0]
                
                facility_params[facility_name] = {
                    'source_x_cartesian': float(fac_row['source_x_cartesian']),
                    'source_y_cartesian': float(fac_row['source_y_cartesian']),
                    'source_diameter': float(fac_row['source_diameter']),
                    'Q': float(fac_row['Q_total']),
                    'wind_u': float(fac_row['wind_u']),
                    'wind_v': float(fac_row['wind_v']),
                    'D': float(fac_row['D']),
                }
        except Exception as e:
            print(f"  Warning: Could not load {facility_name}: {e}")
            continue
    
    return facility_params

def main():
    print("=" * 100)
    print("ANALYZE 2019 EDF SENSOR DATA - HAZARD IDENTIFICATION")
    print("=" * 100)
    print()
    
    # Step 1: Load EDF sensor data
    sensor_df = load_edf_sensor_data()
    
    # Step 2: Calculate hazard metrics
    results_df = calculate_hazard_metrics(sensor_df)
    
    # Step 3: Identify top hazards
    top_hazards = identify_top_hazards(results_df, top_n=10)
    
    # Step 4: Extract facility conditions for each hazard
    print("\nExtracting facility conditions for top hazards...")
    hazards_with_conditions = []
    
    for idx, hazard_row in top_hazards.iterrows():
        timestamp = hazard_row['timestamp']
        print(f"\n  Processing hazard #{hazard_row['hazard_rank']}: {timestamp}")
        
        facility_params = extract_facility_conditions(timestamp)
        
        if facility_params:
            print(f"    Found conditions for {len(facility_params)} facilities")
            
            hazard_data = {
                'timestamp': timestamp,
                'hazard_rank': hazard_row['hazard_rank'],
                'max_concentration_edf': hazard_row['max_concentration_edf'],
                'mean_concentration_edf': hazard_row['mean_concentration_edf'],
                'peak_sensor_id': hazard_row['peak_sensor_id'],
                'facility_params': facility_params,
                **{col: hazard_row[col] for col in SENSOR_IDS if col in hazard_row}
            }
            hazards_with_conditions.append(hazard_data)
        else:
            print(f"    Warning: No facility conditions found for {timestamp}")
    
    # Step 5: Save results
    print("\nSaving results...")
    
    # Save top hazards summary
    top_hazards_summary = top_hazards[[
        'timestamp', 'hazard_rank', 'max_concentration_edf', 'mean_concentration_edf',
        'total_exposure_edf', 'peak_sensor_id'
    ] + [col for col in SENSOR_IDS if col in top_hazards.columns]]
    
    summary_path = OUTPUT_DIR / 'hazards_2019_summary.csv'
    top_hazards_summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")
    
    # Save detailed results
    results_path = OUTPUT_DIR / 'hazards_2019_detailed.csv'
    results_df.to_csv(results_path, index=False)
    print(f"  Saved: {results_path}")
    
    # Save hazards with conditions (as pickle for easy loading)
    import pickle
    hazards_path = OUTPUT_DIR / 'hazards_2019_with_conditions.pkl'
    with open(hazards_path, 'wb') as f:
        pickle.dump(hazards_with_conditions, f)
    print(f"  Saved: {hazards_path}")
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"\nTop {len(hazards_with_conditions)} hazards identified with facility conditions")
    print(f"Ready for PINN visualization")
    
    return hazards_with_conditions

if __name__ == "__main__":
    hazards = main()

