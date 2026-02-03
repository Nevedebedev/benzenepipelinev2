#!/usr/bin/env python3
"""
Investigate Zero-Handling Mismatch in NN2

Critical question: Does training transform zeros or not?
If training transforms zeros but validation doesn't (or vice versa), 
this would cause a major discrepancy.
"""

import sys
from pathlib import Path
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

import torch
import pandas as pd
import numpy as np
from pinn import ParametricADEPINN
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

def test_zero_handling_strategies():
    """Test different zero-handling strategies"""
    print("="*80)
    print("TESTING ZERO-HANDLING STRATEGIES")
    print("="*80)
    
    # Load data
    training_pinn_df = pd.read_csv(TRAINING_PINN_PATH)
    training_pinn_df['timestamp'] = pd.to_datetime(training_pinn_df['timestamp'])
    
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    year_mask = (sensor_df['timestamp'] >= '2019-01-01') & (sensor_df['timestamp'] < '2020-01-01')
    sensor_df = sensor_df[year_mask].reset_index(drop=True)
    
    # Merge
    merged = training_pinn_df.merge(sensor_df, on='timestamp', how='inner', suffixes=('_pinn', '_actual'))
    
    nn2, scalers, sensor_coords_spatial = load_models()
    
    sensor_ids_sorted = sorted(SENSORS.keys())
    
    results = []
    
    print("\nTesting on sample data...")
    for idx in range(min(300, len(merged))):
        row = merged.iloc[idx]
        timestamp = row['timestamp']
        
        # Get PINN and actual values
        pinn_values = {}
        actual_values = {}
        for sensor_id in SENSORS.keys():
            pinn_col = f'sensor_{sensor_id}_pinn'
            actual_col = f'sensor_{sensor_id}_actual'
            pinn_values[sensor_id] = row[pinn_col] if pinn_col in row and not pd.isna(row[pinn_col]) else 0.0
            actual_values[sensor_id] = row[actual_col] if actual_col in row and not pd.isna(row[actual_col]) else 0.0
        
        # Prepare arrays
        pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
        current_sensors = np.array([actual_values.get(sid, 0.0) for sid in sensor_ids_sorted])
        
        # Meteo (simplified)
        meteo_u, meteo_v, meteo_D = 3.0, 0.0, 10.0
        
        # Temporal
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
        
        # Strategy 1: Zero-masking (matches training code in nn2.py)
        pinn_nonzero_mask = pinn_array != 0.0
        sensors_nonzero_mask = current_sensors != 0.0
        
        p_s_mask = np.zeros_like(pinn_array)
        if pinn_nonzero_mask.any():
            p_s_mask[pinn_nonzero_mask] = scalers['pinn'].transform(
                pinn_array[pinn_nonzero_mask].reshape(-1, 1)
            ).flatten()
        p_s_mask = p_s_mask.reshape(1, -1)
        
        s_s_mask = np.zeros_like(current_sensors)
        if sensors_nonzero_mask.any():
            s_s_mask[sensors_nonzero_mask] = scalers['sensors'].transform(
                current_sensors[sensors_nonzero_mask].reshape(-1, 1)
            ).flatten()
        s_s_mask = s_s_mask.reshape(1, -1)
        
        # Strategy 2: Transform ALL values (like benzene_pipeline.py)
        p_s_all = scalers['pinn'].transform(pinn_array.reshape(-1, 1)).flatten().reshape(1, -1)
        s_s_all = scalers['sensors'].transform(current_sensors.reshape(-1, 1)).flatten().reshape(1, -1)
        
        # Common scaling
        w_s = scalers['wind'].transform(np.array([[meteo_u, meteo_v]]))
        d_s = scalers['diffusion'].transform(np.array([[meteo_D]]))
        c_s = scalers['coords'].transform(sensor_coords_spatial)
        
        # Run NN2 with both strategies
        c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)
        w_tensor = torch.tensor(w_s, dtype=torch.float32)
        d_tensor = torch.tensor(d_s, dtype=torch.float32)
        t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)
        
        # Strategy 1: Zero-masking
        p_mask_tensor = torch.tensor(p_s_mask, dtype=torch.float32)
        s_mask_tensor = torch.tensor(s_s_mask, dtype=torch.float32)
        with torch.no_grad():
            corrected_mask_scaled, _ = nn2(s_mask_tensor, p_mask_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
        
        # Strategy 2: All values
        p_all_tensor = torch.tensor(p_s_all, dtype=torch.float32)
        s_all_tensor = torch.tensor(s_s_all, dtype=torch.float32)
        with torch.no_grad():
            corrected_all_scaled, _ = nn2(s_all_tensor, p_all_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
        
        # Inverse transform
        corrected_mask_np = corrected_mask_scaled.cpu().numpy().flatten()
        corrected_all_np = corrected_all_scaled.cpu().numpy().flatten()
        
        # Inverse transform for each sensor
        for i, sensor_id in enumerate(sensor_ids_sorted):
            actual = actual_values.get(sensor_id, 0.0)
            pinn = pinn_values[sensor_id]
            
            # Strategy 1 inverse
            if sensors_nonzero_mask[i]:
                nn2_mask = scalers['sensors'].inverse_transform(
                    corrected_mask_np[i].reshape(-1, 1)
                )[0, 0]
            else:
                if np.abs(corrected_mask_np[i]) > 1e-6:
                    nn2_mask = scalers['sensors'].inverse_transform(
                        corrected_mask_np[i].reshape(-1, 1)
                    )[0, 0]
                else:
                    nn2_mask = 0.0
            
            # Strategy 2 inverse (transform all)
            nn2_all = scalers['sensors'].inverse_transform(
                corrected_all_np[i].reshape(-1, 1)
            )[0, 0]
            
            results.append({
                'timestamp': timestamp,
                'sensor_id': sensor_id,
                'actual': actual,
                'pinn': pinn,
                'nn2_mask': nn2_mask,
                'nn2_all': nn2_all,
                'pinn_scaled_mask': p_s_mask[0, i],
                'pinn_scaled_all': p_s_all[0, i],
                'sensor_scaled_mask': s_s_mask[0, i],
                'sensor_scaled_all': s_s_all[0, i],
                'corrected_scaled_mask': corrected_mask_np[i],
                'corrected_scaled_all': corrected_all_np[i],
                'pinn_is_zero': (pinn == 0.0),
                'sensor_is_zero': (actual == 0.0),
            })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("RESULTS: Zero-Masking vs Transform-All")
    print("="*80)
    
    pinn_mae = np.mean(np.abs(results_df['actual'] - results_df['pinn']))
    nn2_mask_mae = np.mean(np.abs(results_df['actual'] - results_df['nn2_mask']))
    nn2_all_mae = np.mean(np.abs(results_df['actual'] - results_df['nn2_all']))
    
    print(f"\nPINN MAE:              {pinn_mae:.4f} ppb")
    print(f"NN2 Zero-Masking:      {nn2_mask_mae:.4f} ppb")
    print(f"NN2 Transform-All:     {nn2_all_mae:.4f} ppb")
    
    print(f"\nDifference between strategies:")
    diff = np.abs(results_df['nn2_mask'] - results_df['nn2_all'])
    print(f"  Mean absolute difference: {diff.mean():.6f} ppb")
    print(f"  Max difference: {diff.max():.6f} ppb")
    print(f"  Cases where different: {(diff > 1e-6).sum()} / {len(diff)}")
    
    # Check cases with zeros
    print(f"\nCases with zero PINN predictions:")
    zero_pinn = results_df[results_df['pinn_is_zero']]
    if len(zero_pinn) > 0:
        print(f"  Count: {len(zero_pinn)}")
        print(f"  PINN scaled (mask): {zero_pinn['pinn_scaled_mask'].mean():.6f}")
        print(f"  PINN scaled (all):  {zero_pinn['pinn_scaled_all'].mean():.6f}")
    
    print(f"\nCases with zero sensor readings:")
    zero_sensor = results_df[results_df['sensor_is_zero']]
    if len(zero_sensor) > 0:
        print(f"  Count: {len(zero_sensor)}")
        print(f"  Sensor scaled (mask): {zero_sensor['sensor_scaled_mask'].mean():.6f}")
        print(f"  Sensor scaled (all):  {zero_sensor['sensor_scaled_all'].mean():.6f}")
    
    # Save results
    output_path = PROJECT_DIR / 'validation_results' / 'nn2_zero_handling_investigation.csv'
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved results to {output_path}")
    
    return results_df

def load_models():
    """Load NN2 model and scalers"""
    print("Loading models...")
    
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    print("  ✓ Models loaded")
    return nn2, scalers, sensor_coords_spatial

if __name__ == '__main__':
    print("="*80)
    print("NN2 ZERO-HANDLING MISMATCH INVESTIGATION")
    print("="*80)
    
    results_df = test_zero_handling_strategies()
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)

