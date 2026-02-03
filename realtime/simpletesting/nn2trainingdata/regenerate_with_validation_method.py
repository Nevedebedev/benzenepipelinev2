#!/usr/bin/env python3
"""
Regenerate Training Data Using Validation Pipeline Method

This script:
1. Uses the SAME pipeline process as validation
2. PINN at sources → superimpose → convert to ppb
3. Processes all 2019 facility CSVs
4. Compares output with existing total_concentrations.csv

This will reveal if the training data generation had a bug.
"""

import sys
sys.path.append('/Users/neevpratap/simpletesting')

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from pinn import ParametricADEPINN
from tqdm import tqdm

# Paths
SYNCED_DIR = Path('/Users/neevpratap/Desktop/madis_data_desktop_updated/synced')
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
EXISTING_TRAINING_DATA = "/Users/neevpratap/simpletesting/nn2trainingdata/total_concentrations.csv"
OUTPUT_FILE = "/Users/neevpratap/simpletesting/nn2trainingdata/total_concentrations_RECOMPUTED.csv"

# Constants
UNIT_CONVERSION = 313210039.9
T_START = pd.to_datetime('2019-01-01 00:00:00')

# 9 sensor coordinates
SENSORS = {
    '482010026': (13972.62, 19915.57),
    '482010057': (3017.18, 12334.2),
    '482010069': (817.42, 9218.92),
    '482010617': (27049.57, 22045.66),
    '482010803': (8836.35, 15717.2),
    '482011015': (18413.8, 15068.96),
    '482011035': (1159.98, 12272.52),
    '482011039': (13661.93, 5193.24),
    '482016000': (1546.9, 6786.33),
}

def load_pinn():
    """Load PINN model"""
    print("Loading PINN...")
    pinn = ParametricADEPINN()
    checkpoint = torch.load(PINN_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    pinn.eval()
    print("  ✓ PINN loaded")
    return pinn

def compute_pinn_at_sensors(pinn, facility_data, timestamp):
    """
    PIPELINE PROCESS:
    1. Solve PINN at each source
    2. Superimpose at sensor locations  
    3. Convert to ppb
    """
    t_hours = (timestamp - T_START).total_seconds() / 3600.0
    
    sensor_concentrations = {sid: 0.0 for sid in SENSORS.keys()}
    
    for _, row in facility_data.iterrows():
        cx = row['source_x_cartesian']
        cy = row['source_y_cartesian']
        d = row['source_diameter']
        Q = row['Q_total']
        u = row['wind_u']
        v = row['wind_v']
        kappa = row['D']
        
        # Solve PINN at each sensor for this facility
        for sensor_id, (sx, sy) in SENSORS.items():
            with torch.no_grad():
                phi_raw = pinn(
                    torch.tensor([[sx]], dtype=torch.float32),
                    torch.tensor([[sy]], dtype=torch.float32),
                    torch.tensor([[t_hours]], dtype=torch.float32),
                    torch.tensor([[cx]], dtype=torch.float32),
                    torch.tensor([[cy]], dtype=torch.float32),
                    torch.tensor([[u]], dtype=torch.float32),
                    torch.tensor([[v]], dtype=torch.float32),
                    torch.tensor([[d]], dtype=torch.float32),
                    torch.tensor([[kappa]], dtype=torch.float32),
                    torch.tensor([[Q]], dtype=torch.float32)
                )
                
                # Convert to ppb and superimpose
                concentration_ppb = phi_raw.item() * UNIT_CONVERSION
                sensor_concentrations[sensor_id] += concentration_ppb
    
    return sensor_concentrations

def main():
    print("="*80)
    print("REGENERATING TRAINING DATA WITH VALIDATION PIPELINE METHOD")
    print("="*80)
    print()
    
    # Load PINN
    pinn = load_pinn()
    
    # Load all facility files
    print("\nLoading facility data...")
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_dfs = []
    for f in facility_files:
        if 'summary' in f.name:
            continue
        df = pd.read_csv(f)
        df['timestamp'] = pd.to_datetime(df['t'])
        facility_dfs.append(df)
    
    merged_facilities = pd.concat(facility_dfs, ignore_index=True)
    print(f"  Loaded {len(facility_files)} facilities")
    
    # Get unique timestamps
    unique_timestamps = sorted(merged_facilities['timestamp'].unique())
    print(f"  Found {len(unique_timestamps)} unique timestamps")
    
    # Process each timestamp
    print("\nProcessing with PINN pipeline...")
    results = []
    
    for timestamp in tqdm(unique_timestamps, desc="Computing"):
        facility_data = merged_facilities[merged_facilities['timestamp'] == timestamp]
        
        if len(facility_data) == 0:
            continue
        
        # PIPELINE: PINN → superimpose → ppb
        sensor_vals = compute_pinn_at_sensors(pinn, facility_data, timestamp)
        
        # Store result
        row = {'timestamp': timestamp}
        for sensor_id in sorted(SENSORS.keys()):
            row[f'sensor_{sensor_id}'] = sensor_vals[sensor_id]
        results.append(row)
    
    # Convert to DataFrame
    recomputed_df = pd.DataFrame(results)
    
    # Save
    recomputed_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✓ Saved recomputed data to {OUTPUT_FILE}")
    print(f"  Shape: {recomputed_df.shape}")
    
    # Load existing training data
    print("\nLoading EXISTING training data...")
    existing_df = pd.read_csv(EXISTING_TRAINING_DATA)
    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
    print(f"  Shape: {existing_df.shape}")
    
    # Compare statistics
    print("\n" + "="*80)
    print("COMPARISON: EXISTING vs RECOMPUTED")
    print("="*80)
    
    sensor_cols = [f'sensor_{sid}' for sid in sorted(SENSORS.keys())]
    
    print("\nPer-Sensor Means:")
    print(f"{'Sensor':<15} {'Existing':<12} {'Recomputed':<12} {'Ratio':<10}")
    print("-" * 50)
    
    for col in sensor_cols:
        existing_mean = existing_df[col].mean()
        recomputed_mean = recomputed_df[col].mean()
        ratio = existing_mean / recomputed_mean if recomputed_mean != 0 else np.inf
        
        print(f"{col:<15} {existing_mean:>10.4f}   {recomputed_mean:>10.4f}   {ratio:>8.2f}x")
    
    # Overall comparison
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    existing_overall = existing_df[sensor_cols].values.mean()
    recomputed_overall = recomputed_df[sensor_cols].values.mean()
    
    print(f"\nEXISTING training data mean:  {existing_overall:.4f} ppb")
    print(f"RECOMPUTED data mean:         {recomputed_overall:.4f} ppb")
    print(f"Ratio (existing/recomputed):  {existing_overall/recomputed_overall:.2f}x")
    
    if abs(existing_overall - recomputed_overall) < 0.01:
        print("\n✓ MATCH! Existing and recomputed data are identical.")
    elif existing_overall > recomputed_overall * 10:
        print("\n⚠️ MISMATCH! Existing data is 10x+ higher than recomputed.")
        print("   → Original training data generation had a bug!")
    else:
        print("\n⚠️ MISMATCH! Data differs but ratio is reasonable.")
        print("   → May be due to different facility data or timestamps")
    
    # Sample comparison
    print("\n" + "="*80)
    print("SAMPLE COMPARISON (First 5 timestamps)")
    print("="*80)
    
    # Merge on timestamp
    merged = pd.merge(
        existing_df.head(5),
        recomputed_df.head(5),
        on='timestamp',
        suffixes=('_existing', '_recomputed')
    )
    
    for idx, row in merged.iterrows():
        print(f"\nTimestamp: {row['timestamp']}")
        for sid in sorted(SENSORS.keys()):
            col_e = f'sensor_{sid}_existing'
            col_r = f'sensor_{sid}_recomputed'
            if col_e in row and col_r in row:
                diff = abs(row[col_e] - row[col_r])
                print(f"  {sid}: {row[col_e]:.6f} vs {row[col_r]:.6f} (diff: {diff:.6f})")
    
    print("\n" + "="*80)
    print("REGENERATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
