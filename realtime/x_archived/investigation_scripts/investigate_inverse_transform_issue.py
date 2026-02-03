#!/usr/bin/env python3
"""
Investigate Inverse Transform Issue

NN2 performs well in scaled space (35% improvement) but degrades in original space (-168%).
This script investigates why the inverse transform is causing degradation.
"""

import sys
from pathlib import Path
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

import torch
import pandas as pd
import numpy as np
from nn2 import NN2_CorrectionNetwork
import pickle

# Paths
PROJECT_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime')
SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
TRAINING_PINN_PATH = SYNCED_DIR / 'total_concentrations.csv'
NN2_MODEL_PATH = PROJECT_DIR / "nn2_timefix/nn2_master_model_spatial-3.pth"
NN2_SCALERS_PATH = PROJECT_DIR / "nn2_timefix/nn2_master_scalers-2.pkl"

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

def load_models():
    """Load NN2 model and scalers"""
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    return nn2, scalers, sensor_coords_spatial

def test_inverse_transform_strategies():
    """Test different inverse transform strategies"""
    print("="*80)
    print("INVESTIGATING INVERSE TRANSFORM ISSUE")
    print("="*80)
    
    nn2, scalers, sensor_coords_spatial = load_models()
    
    # Load training data
    training_pinn_df = pd.read_csv(TRAINING_PINN_PATH)
    training_pinn_df['timestamp'] = pd.to_datetime(training_pinn_df['timestamp'])
    
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    merged = training_pinn_df.merge(sensor_df, on='timestamp', how='inner', suffixes=('_pinn', '_actual'))
    
    # Load meteorology
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    first_file = facility_files[0]
    meteo_df = pd.read_csv(first_file)
    if 't' in meteo_df.columns:
        meteo_df['timestamp'] = pd.to_datetime(meteo_df['t'])
    elif 'timestamp' in meteo_df.columns:
        meteo_df['timestamp'] = pd.to_datetime(meteo_df['timestamp'])
    
    sensor_ids_sorted = sorted(SENSORS.keys())
    results = []
    
    print("\nTesting different inverse transform strategies on 100 samples...")
    
    for idx, row in merged.head(100).iterrows():
        timestamp = row['timestamp']
        
        # Get data
        pinn_values = {}
        actual_values = {}
        for sensor_id in SENSORS.keys():
            col_pinn = f'sensor_{sensor_id}_pinn'
            col_actual = f'sensor_{sensor_id}_actual'
            pinn_values[sensor_id] = row[col_pinn] if col_pinn in row and not pd.isna(row[col_pinn]) else 0.0
            actual_values[sensor_id] = row[col_actual] if col_actual in row and not pd.isna(row[col_actual]) else 0.0
        
        # Get meteorology
        met_data_timestamp = timestamp - pd.Timedelta(hours=3)
        meteo_row = meteo_df[meteo_df['timestamp'] == met_data_timestamp]
        if len(meteo_row) == 0:
            continue
        real_u = meteo_row['wind_u'].iloc[0]
        real_v = meteo_row['wind_v'].iloc[0]
        real_D = meteo_row['D'].iloc[0]
        if pd.isna(real_u) or pd.isna(real_v) or pd.isna(real_D):
            continue
        
        # Prepare inputs
        pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
        current_sensors_array = np.array([actual_values.get(sid, 0.0) for sid in sensor_ids_sorted])
        
        pinn_nonzero_mask = pinn_array != 0.0
        sensors_nonzero_mask = current_sensors_array != 0.0
        
        # Scale
        p_s = np.zeros_like(pinn_array)
        if pinn_nonzero_mask.any():
            p_s[pinn_nonzero_mask] = scalers['pinn'].transform(
                pinn_array[pinn_nonzero_mask].reshape(-1, 1)
            ).flatten()
        p_s = p_s.reshape(1, -1)
        
        s_s = np.zeros_like(current_sensors_array)
        if sensors_nonzero_mask.any():
            s_s[sensors_nonzero_mask] = scalers['sensors'].transform(
                current_sensors_array[sensors_nonzero_mask].reshape(-1, 1)
            ).flatten()
        s_s = s_s.reshape(1, -1)
        
        w_s = scalers['wind'].transform(np.array([[real_u, real_v]]))
        d_s = scalers['diffusion'].transform(np.array([[real_D]]))
        c_s = scalers['coords'].transform(sensor_coords_spatial)
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        is_weekend = 1.0 if day_of_week >= 5 else 0.0
        temporal_vals = np.array([[
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            is_weekend,
            month / 12.0
        ]])
        
        p_tensor = torch.tensor(p_s, dtype=torch.float32)
        s_tensor = torch.tensor(s_s, dtype=torch.float32)
        c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)
        w_tensor = torch.tensor(w_s, dtype=torch.float32)
        d_tensor = torch.tensor(d_s, dtype=torch.float32)
        t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)
        
        # Run NN2
        with torch.no_grad():
            corrected_scaled, corrections = nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
        
        corrected_scaled_np = corrected_scaled.cpu().numpy().flatten()
        
        # Strategy 1: Only inverse transform non-zero sensor values (current approach)
        nn2_strategy1 = np.zeros_like(corrected_scaled_np)
        if sensors_nonzero_mask.any():
            nn2_strategy1[sensors_nonzero_mask] = scalers['sensors'].inverse_transform(
                corrected_scaled_np[sensors_nonzero_mask].reshape(-1, 1)
            ).flatten()
        
        # Strategy 2: Inverse transform ALL values (no zero-masking)
        nn2_strategy2 = scalers['sensors'].inverse_transform(corrected_scaled_np.reshape(-1, 1)).flatten()
        
        # Strategy 3: Use PINN mask for inverse transform (match PINN zeros)
        nn2_strategy3 = np.zeros_like(corrected_scaled_np)
        if pinn_nonzero_mask.any():
            nn2_strategy3[pinn_nonzero_mask] = scalers['sensors'].inverse_transform(
                corrected_scaled_np[pinn_nonzero_mask].reshape(-1, 1)
            ).flatten()
        
        # Strategy 4: Check if output is non-zero before inverse transform
        output_nonzero_mask = np.abs(corrected_scaled_np) > 1e-6
        nn2_strategy4 = np.zeros_like(corrected_scaled_np)
        if output_nonzero_mask.any():
            nn2_strategy4[output_nonzero_mask] = scalers['sensors'].inverse_transform(
                corrected_scaled_np[output_nonzero_mask].reshape(-1, 1)
            ).flatten()
        
        # Compare strategies
        for i, sensor_id in enumerate(sensor_ids_sorted):
            actual = actual_values.get(sensor_id, 0.0)
            pinn = pinn_values[sensor_id]
            
            results.append({
                'timestamp': timestamp,
                'sensor_id': sensor_id,
                'actual': actual,
                'pinn': pinn,
                'corrected_scaled': corrected_scaled_np[i],
                'target_scaled': s_s[0, i],
                'pinn_scaled': p_s[0, i],
                'nn2_strategy1': nn2_strategy1[i],
                'nn2_strategy2': nn2_strategy2[i],
                'nn2_strategy3': nn2_strategy3[i],
                'nn2_strategy4': nn2_strategy4[i],
                'sensors_nonzero': sensors_nonzero_mask[i],
                'pinn_nonzero': pinn_nonzero_mask[i],
                'output_nonzero': output_nonzero_mask[i],
            })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("INVERSE TRANSFORM STRATEGY COMPARISON")
    print("="*80)
    
    # Calculate MAE for each strategy
    pinn_mae = np.mean(np.abs(results_df['actual'] - results_df['pinn']))
    strategy1_mae = np.mean(np.abs(results_df['actual'] - results_df['nn2_strategy1']))
    strategy2_mae = np.mean(np.abs(results_df['actual'] - results_df['nn2_strategy2']))
    strategy3_mae = np.mean(np.abs(results_df['actual'] - results_df['nn2_strategy3']))
    strategy4_mae = np.mean(np.abs(results_df['actual'] - results_df['nn2_strategy4']))
    
    print(f"\nPINN MAE:                    {pinn_mae:.4f} ppb")
    print(f"NN2 Strategy 1 (sensors mask): {strategy1_mae:.4f} ppb")
    print(f"NN2 Strategy 2 (all values):   {strategy2_mae:.4f} ppb")
    print(f"NN2 Strategy 3 (pinn mask):     {strategy3_mae:.4f} ppb")
    print(f"NN2 Strategy 4 (output mask):   {strategy4_mae:.4f} ppb")
    
    # Scaled space comparison
    pinn_mae_scaled = np.mean(np.abs(results_df['pinn_scaled'] - results_df['target_scaled']))
    nn2_mae_scaled = np.mean(np.abs(results_df['corrected_scaled'] - results_df['target_scaled']))
    
    print(f"\nSCALED SPACE:")
    print(f"  PINN MAE (scaled): {pinn_mae_scaled:.6f}")
    print(f"  NN2 MAE (scaled):  {nn2_mae_scaled:.6f}")
    print(f"  Improvement:      {((pinn_mae_scaled - nn2_mae_scaled) / pinn_mae_scaled * 100):.1f}%")
    
    # Analyze inverse transform issues
    print("\n" + "="*80)
    print("INVERSE TRANSFORM ANALYSIS")
    print("="*80)
    
    # Check cases where scaled space is good but original is bad
    good_scaled = results_df['corrected_scaled'] - results_df['target_scaled']
    good_scaled_abs = np.abs(good_scaled)
    bad_original = results_df['nn2_strategy1'] - results_df['actual']
    bad_original_abs = np.abs(bad_original)
    
    # Cases where scaled error is small but original error is large
    scaled_good_mask = good_scaled_abs < 0.1
    original_bad_mask = bad_original_abs > 0.5
    
    problematic_cases = results_df[scaled_good_mask & original_bad_mask]
    
    print(f"\nCases where scaled space is good but original space is bad:")
    print(f"  Count: {len(problematic_cases)} / {len(results_df)}")
    
    if len(problematic_cases) > 0:
        print(f"\n  Sample problematic cases:")
        for _, case in problematic_cases.head(5).iterrows():
            print(f"    Sensor {case['sensor_id']}:")
            print(f"      Scaled: target={case['target_scaled']:.4f}, pred={case['corrected_scaled']:.4f}, error={abs(case['corrected_scaled'] - case['target_scaled']):.4f}")
            print(f"      Original: actual={case['actual']:.4f}, pred={case['nn2_strategy1']:.4f}, error={abs(case['nn2_strategy1'] - case['actual']):.4f}")
            print(f"      Inverse transform: scaled={case['corrected_scaled']:.4f} → original={case['nn2_strategy1']:.4f}")
    
    # Check scaler statistics
    print(f"\nScaler Statistics:")
    print(f"  Sensors mean: {scalers['sensors'].mean_[0]:.4f}")
    print(f"  Sensors std:  {scalers['sensors'].scale_[0]:.4f}")
    print(f"  Scaled range: [{results_df['corrected_scaled'].min():.4f}, {results_df['corrected_scaled'].max():.4f}]")
    print(f"  Original range (strategy 1): [{results_df['nn2_strategy1'].min():.4f}, {results_df['nn2_strategy1'].max():.4f}]")
    
    # Save results
    output_path = PROJECT_DIR / 'validation_results' / 'nn2_inverse_transform_investigation.csv'
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved detailed results to {output_path}")
    
    return results_df

if __name__ == '__main__':
    print("="*80)
    print("INVERSE TRANSFORM ISSUE INVESTIGATION")
    print("="*80)
    
    results_df = test_inverse_transform_strategies()
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)

