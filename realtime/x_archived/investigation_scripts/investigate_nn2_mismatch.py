#!/usr/bin/env python3
"""
Deep Investigation: NN2 Training vs Validation Data Comparison

This script will:
1. Load the ACTUAL training data used for NN2 (total_concentrations.csv)
2. Compute PINN predictions using the validation script method
3. Compare them side-by-side to find discrepancies
4. Check if NN2 predictions make sense given the training data
"""

import sys
sys.path.append('/Users/neevpratap/simpletesting')

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from pinn import ParametricADEPINN
import pickle
from nn2 import NN2_CorrectionNetwork

# Paths
TRAINING_PINN_DATA = "/Users/neevpratap/simpletesting/nn2trainingdata/total_concentrations.csv"
TRAINING_SENSOR_DATA = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
SYNCED_DIR = Path('/Users/neevpratap/Desktop/madis_data_desktop_updated/synced')
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
NN2_MODEL_PATH = "/Users/neevpratap/Desktop/nn2_updated/nn2_master_model_spatial-3.pth"
NN2_SCALERS_PATH = "/Users/neevpratap/Desktop/nn2_updated/nn2_master_scalers-2.pkl"

UNIT_CONVERSION = 313210039.9
T_START = pd.to_datetime('2019-01-01 00:00:00')

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

def load_pinn_model():
    """Load PINN"""
    pinn = ParametricADEPINN()
    checkpoint = torch.load(PINN_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    pinn.eval()
    return pinn

def compute_pinn_realtime(pinn, facility_data, timestamp):
    """Compute PINN the validation script way"""
    t_hours = (timestamp - T_START).total_seconds() / 3600.0
    
    sensor_pinn_ppb = {sid: 0.0 for sid in SENSORS.keys()}
    
    for _, row in facility_data.iterrows():
        cx = row['source_x_cartesian']
        cy = row['source_y_cartesian']
        d = row['source_diameter']
        Q = row['Q_total']
        u = row['wind_u']
        v = row['wind_v']
        kappa = row['D']
        
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
                
                concentration_ppb = phi_raw.item() * UNIT_CONVERSION
                sensor_pinn_ppb[sensor_id] += concentration_ppb
    
    return sensor_pinn_ppb

def main():
    print("="*80)
    print("DEEP INVESTIGATION: NN2 TRAINING DATA MISMATCH")
    print("="*80)
    print()
    
    # 1. Load training data (what NN2 was trained on)
    print("1. Loading TRAINING data (what NN2 was trained on)...")
    training_pinn_df = pd.read_csv(TRAINING_PINN_DATA)
    training_pinn_df['timestamp'] = pd.to_datetime(training_pinn_df['timestamp'])
    print(f"   Training PINN data: {len(training_pinn_df)} timestamps")
    
    training_sensor_df = pd.read_csv(TRAINING_SENSOR_DATA)
    if 't' in training_sensor_df.columns:
        training_sensor_df = training_sensor_df.rename(columns={'t': 'timestamp'})
    training_sensor_df['timestamp'] = pd.to_datetime(training_sensor_df['timestamp'])
    print(f"   Training sensor data: {len(training_sensor_df)} timestamps")
    
    # Filter to January only
    jan_mask = (training_pinn_df['timestamp'] >= '2019-01-01') & (training_pinn_df['timestamp'] < '2019-02-01')
    training_pinn_jan = training_pinn_df[jan_mask].reset_index(drop=True)
    print(f"   Training PINN (January only): {len(training_pinn_jan)} timestamps")
    
    jan_mask_sensor = (training_sensor_df['timestamp'] >= '2019-01-01') & (training_sensor_df['timestamp'] < '2019-02-01')
    training_sensor_jan = training_sensor_df[jan_mask_sensor].reset_index(drop=True)
    print(f"   Training sensors (January only): {len(training_sensor_jan)} timestamps")
    
    # 2. Load PINN model and facility data
    print("\n2. Loading PINN model and facility data...")
    pinn = load_pinn_model()
    
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_dfs = []
    for f in facility_files:
        if 'summary' in f.name:
            continue
        df = pd.read_csv(f)
        df['timestamp'] = pd.to_datetime(df['t'])
        facility_dfs.append(df)
    
    merged_facilities = pd.concat(facility_dfs, ignore_index=True)
    print(f"   Loaded {len(facility_files)} facilities")
    
    # 3. Compare on a sample of timestamps
    print("\n3. Comparing TRAINING data vs REAL-TIME computation...")
    print("   Taking first 10 timestamps from January...")
    
    sample_timestamps = training_pinn_jan['timestamp'].head(10).tolist()
    
    comparison = []
    for timestamp in sample_timestamps:
        # Get training data value
        training_row = training_pinn_jan[training_pinn_jan['timestamp'] == timestamp]
        if len(training_row) == 0:
            continue
        training_row = training_row.iloc[0]
        
        # Compute real-time value
        facility_data = merged_facilities[merged_facilities['timestamp'] == timestamp]
        if len(facility_data) == 0:
            continue
        
        realtime_pinn = compute_pinn_realtime(pinn, facility_data, timestamp)
        
        # Compare for each sensor
        for sensor_id in sorted(SENSORS.keys()):
            training_col = f'sensor_{sensor_id}'
            if training_col not in training_row:
                continue
            
            training_value = training_row[training_col]
            realtime_value = realtime_pinn[sensor_id]
            
            comparison.append({
                'timestamp': timestamp,
                'sensor_id': sensor_id,
                'training_pinn': training_value,
                'realtime_pinn': realtime_value,
                'diff': abs(training_value - realtime_value),
                'ratio': realtime_value / training_value if training_value != 0 else np.inf
            })
    
    comp_df = pd.DataFrame(comparison)
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON: Training PINN vs Real-time PINN")
    print("="*80)
    
    for sensor_id in sorted(SENSORS.keys()):
        sensor_data = comp_df[comp_df['sensor_id'] == sensor_id]
        if len(sensor_data) == 0:
            continue
        
        print(f"\nSensor {sensor_id}:")
        print(f"  Mean training PINN:  {sensor_data['training_pinn'].mean():.4f} ppb")
        print(f"  Mean realtime PINN:  {sensor_data['realtime_pinn'].mean():.4f} ppb")
        print(f"  Mean difference:     {sensor_data['diff'].mean():.4f} ppb")
        print(f"  Mean ratio:          {sensor_data['ratio'].mean():.4f}x")
        print(f"  Max difference:      {sensor_data['diff'].max():.4f} ppb")
    
    # Overall stats
    print("\n" + "="*80)
    print("OVERALL COMPARISON")
    print("="*80)
    print(f"Samples compared: {len(comp_df)}")
    print(f"Mean training PINN: {comp_df['training_pinn'].mean():.4f} ppb")
    print(f"Mean realtime PINN: {comp_df['realtime_pinn'].mean():.4f} ppb")
    print(f"Mean absolute diff: {comp_df['diff'].mean():.4f} ppb")
    print(f"Mean ratio: {comp_df['ratio'].mean():.4f}x")
    print(f"Max difference: {comp_df['diff'].max():.4f} ppb")
    
    # Check if they're essentially the same
    if comp_df['diff'].mean() < 0.01:
        print("\n✓ Training and real-time PINN predictions MATCH!")
    else:
        print("\n⚠️ Training and real-time PINN predictions DIFFER significantly!")
    
    # 4. Now test NN2 on the TRAINING data directly
    print("\n" + "="*80)
    print("4. Testing NN2 on ACTUAL TRAINING DATA")
    print("="*80)
    
    # Load NN2
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    # Merge training data
    merged_training = pd.merge(training_sensor_jan, training_pinn_jan, on='timestamp', suffixes=('_actual', '_pinn'))
    print(f"   Merged {len(merged_training)} timestamps")
    
    # Compute MAE on training PINN data
    sensor_ids_sorted = sorted(SENSORS.keys())
    
    pinn_maes = []
    actual_means = []
    pinn_means = []
    
    for sensor_id in sensor_ids_sorted:
        actual_col = f'sensor_{sensor_id}_actual'
        pinn_col = f'sensor_{sensor_id}_pinn'
        
        if actual_col not in merged_training.columns or pinn_col not in merged_training.columns:
            continue
        
        actual = merged_training[actual_col].values
        pinn = merged_training[pinn_col].values
        
        # Remove NaN
        mask = ~np.isnan(actual) & ~np.isnan(pinn)
        actual_clean = actual[mask]
        pinn_clean = pinn[mask]
        
        if len(actual_clean) == 0:
            continue
        
        mae = np.mean(np.abs(actual_clean - pinn_clean))
        pinn_maes.append(mae)
        actual_means.append(np.mean(actual_clean))
        pinn_means.append(np.mean(pinn_clean))
        
        print(f"\nSensor {sensor_id} (on TRAINING data):")
        print(f"  Actual mean: {np.mean(actual_clean):.4f} ppb")
        print(f"  PINN mean:   {np.mean(pinn_clean):.4f} ppb")
        print(f"  PINN MAE:    {mae:.4f} ppb")
    
    print("\n" + "="*80)
    print("OVERALL TRAINING DATA MAE")
    print("="*80)
    print(f"Average PINN MAE on training data: {np.mean(pinn_maes):.4f} ppb")
    print(f"Average actual sensor mean: {np.mean(actual_means):.4f} ppb")
    print(f"Average PINN mean: {np.mean(pinn_means):.4f} ppb")
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
