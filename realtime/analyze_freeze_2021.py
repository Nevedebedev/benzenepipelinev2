#!/usr/bin/env python3
"""
Analyze February 2021 Freeze Event (Feb 11-20)
Find worst hour per day based on EDF sensor data
Extract facility conditions for PINN predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

# Paths
SENSOR_DATA_PATH = Path('/Users/neevpratap/Desktop/madis_data_desktop_updated/results_2021/sensors_actual_wide_2021_full_feb.csv')
FACILITY_DATA_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/realtime_processing/houston_processed_2021/training_data_2021_feb')
OUTPUT_DIR = Path('data')
OUTPUT_DIR.mkdir(exist_ok=True)

# Sensor columns
SENSOR_COLS = [
    'sensor_482010026', 'sensor_482010057', 'sensor_482010069',
    'sensor_482010617', 'sensor_482010803', 'sensor_482011015',
    'sensor_482011035', 'sensor_482011039', 'sensor_482016000'
]

def load_sensor_data():
    """Load EDF sensor data for February 2021"""
    if not SENSOR_DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find sensor data file: {SENSOR_DATA_PATH}")
    
    print(f"Loading sensor data from: {SENSOR_DATA_PATH}")
    df = pd.read_csv(SENSOR_DATA_PATH)
    
    # Handle timestamp column (could be 't' or 'timestamp')
    if 't' in df.columns:
        df = df.rename(columns={'t': 'timestamp'})
    if 'timestamp' not in df.columns:
        # Try to find a datetime column
        datetime_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if datetime_cols:
            df = df.rename(columns={datetime_cols[0]: 'timestamp'})
        else:
            raise ValueError(f"No timestamp column found in {SENSOR_DATA_PATH}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def analyze_freeze_period():
    """Analyze Feb 11-20, 2021 freeze period"""
    print("="*80)
    print("FEBRUARY 2021 FREEZE EVENT ANALYSIS")
    print("="*80)
    print()
    
    # Load sensor data
    sensor_df = load_sensor_data()
    
    # Filter to freeze period: Feb 11-20, 2021
    freeze_start = pd.to_datetime('2021-02-11')
    freeze_end = pd.to_datetime('2021-02-21')  # Exclusive end
    
    freeze_df = sensor_df[
        (sensor_df['timestamp'] >= freeze_start) &
        (sensor_df['timestamp'] < freeze_end)
    ].copy()
    
    if len(freeze_df) == 0:
        print("ERROR: No data found for Feb 11-20, 2021")
        print(f"Available date range: {sensor_df['timestamp'].min()} to {sensor_df['timestamp'].max()}")
        return None
    
    print(f"Found {len(freeze_df)} timestamps in freeze period")
    print(f"Date range: {freeze_df['timestamp'].min()} to {freeze_df['timestamp'].max()}")
    print()
    
    # Calculate metrics for each timestamp
    freeze_df['max_concentration_edf'] = freeze_df[SENSOR_COLS].max(axis=1)
    freeze_df['mean_concentration_edf'] = freeze_df[SENSOR_COLS].mean(axis=1)
    freeze_df['total_exposure_edf'] = freeze_df[SENSOR_COLS].sum(axis=1)
    
    # Find peak sensor for each timestamp
    freeze_df['peak_sensor_id'] = freeze_df[SENSOR_COLS].idxmax(axis=1)
    freeze_df['peak_sensor_id'] = freeze_df['peak_sensor_id'].str.replace('sensor_', '')
    
    # Group by date and find worst hour per day
    freeze_df['date'] = freeze_df['timestamp'].dt.date
    
    worst_per_day = []
    
    for date, group in freeze_df.groupby('date'):
        # Find worst hour (highest max concentration)
        worst_idx = group['max_concentration_edf'].idxmax()
        worst_row = group.loc[worst_idx]
        
        worst_per_day.append({
            'date': date,
            'timestamp': worst_row['timestamp'],
            'max_concentration_edf': worst_row['max_concentration_edf'],
            'mean_concentration_edf': worst_row['mean_concentration_edf'],
            'total_exposure_edf': worst_row['total_exposure_edf'],
            'peak_sensor_id': worst_row['peak_sensor_id'],
            **{col: worst_row[col] for col in SENSOR_COLS}
        })
    
    worst_df = pd.DataFrame(worst_per_day)
    worst_df = worst_df.sort_values('max_concentration_edf', ascending=False)
    worst_df['rank'] = range(1, len(worst_df) + 1)
    
    print("="*80)
    print("WORST HOUR PER DAY (Feb 11-20, 2021)")
    print("="*80)
    print()
    print(f"{'Date':<12} {'Time':<8} {'EDF Peak (ppb)':<15} {'Peak Sensor':<20}")
    print("-"*80)
    
    for _, row in worst_df.iterrows():
        ts = pd.to_datetime(row['timestamp'])
        date_str = ts.strftime('%Y-%m-%d')
        time_str = ts.strftime('%H:%M')
        print(f"{date_str:<12} {time_str:<8} {row['max_concentration_edf']:<15.2f} {row['peak_sensor_id']:<20}")
    
    print()
    
    # Save summary
    summary_path = OUTPUT_DIR / 'freeze_2021_summary.csv'
    worst_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary to: {summary_path}")
    
    # Extract facility conditions for each worst hour
    print()
    print("="*80)
    print("EXTRACTING FACILITY CONDITIONS")
    print("="*80)
    print()
    
    freeze_events = []
    
    for idx, row in worst_df.iterrows():
        timestamp = pd.to_datetime(row['timestamp'])
        forecast_time = timestamp  # This is when we want to predict
        met_time = timestamp - pd.Timedelta(hours=3)  # Weather data from 3 hours earlier
        
        print(f"Processing: {timestamp.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"  Forecast time: {forecast_time.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"  Met data time: {met_time.strftime('%Y-%m-%d %H:%M')} UTC")
        
        # Load facility training data
        if not FACILITY_DATA_DIR.exists():
            print(f"    ✗ Facility data directory not found: {FACILITY_DATA_DIR}")
            continue
        
        facility_files = sorted(FACILITY_DATA_DIR.glob('*_training_data.csv'))
        facility_files = [f for f in facility_files if 'summary' not in f.name]
        
        if len(facility_files) == 0:
            print(f"    ✗ No facility data files found in {FACILITY_DATA_DIR}")
            continue
        
        facility_params = {}
        
        for fac_file in facility_files:
            fac_name = fac_file.stem.replace('_training_data', '')
            fac_df = pd.read_csv(fac_file)
            
            # Handle timestamp column
            if 't' in fac_df.columns:
                fac_df['timestamp'] = pd.to_datetime(fac_df['t'])
            elif 'timestamp' in fac_df.columns:
                fac_df['timestamp'] = pd.to_datetime(fac_df['timestamp'])
            else:
                print(f"    ⚠️  {fac_name}: No timestamp column found")
                continue
            
            # Find closest timestamp to met_time
            time_diffs = (fac_df['timestamp'] - met_time).abs()
            closest_idx = time_diffs.idxmin()
            closest_row = fac_df.loc[closest_idx]
            time_diff_hours = time_diffs.loc[closest_idx].total_seconds() / 3600
            
            if time_diff_hours > 1.0:
                print(f"    ⚠️  {fac_name}: No data within 1 hour (closest: {time_diff_hours:.1f}h away)")
                continue
            
            facility_params[fac_name] = {
                'wind_u': float(closest_row['wind_u']),
                'wind_v': float(closest_row['wind_v']),
                'D': float(closest_row['D']),
                'source_x_cartesian': float(closest_row['source_x_cartesian']),
                'source_y_cartesian': float(closest_row['source_y_cartesian']),
                'source_diameter': float(closest_row['source_diameter']),
                'Q': float(closest_row['Q_total']),
            }
        
        if len(facility_params) == 0:
            print(f"    ✗ No facility data found for {met_time}")
            continue
        
        print(f"    ✓ Found data for {len(facility_params)} facilities")
        
        # Get EDF sensor readings
        edf_readings = {}
        for col in SENSOR_COLS:
            sensor_id = col.replace('sensor_', '')
            edf_readings[sensor_id] = float(row[col]) if not pd.isna(row[col]) else 0.0
        
        freeze_events.append({
            'date': row['date'],
            'timestamp': timestamp,
            'forecast_time': forecast_time,
            'met_time': met_time,
            'hazard_rank': row['rank'],
            'max_concentration_edf': row['max_concentration_edf'],
            'mean_concentration_edf': row['mean_concentration_edf'],
            'total_exposure_edf': row['total_exposure_edf'],
            'peak_sensor_id': row['peak_sensor_id'],
            'edf_readings': edf_readings,
            'facility_params': facility_params,
        })
    
    # Save events with conditions
    events_path = OUTPUT_DIR / 'freeze_2021_with_conditions.pkl'
    with open(events_path, 'wb') as f:
        pickle.dump(freeze_events, f)
    print()
    print(f"✓ Saved {len(freeze_events)} events with conditions to: {events_path}")
    
    return freeze_events

if __name__ == '__main__':
    events = analyze_freeze_period()
    if events:
        print()
        print("="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"Total events: {len(events)}")
        print(f"Date range: {events[0]['date']} to {events[-1]['date']}")
        print()
        print("Next step: Run visualize_freeze_predictions.py to generate PINN predictions and visualizations")

