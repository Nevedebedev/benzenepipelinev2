#!/usr/bin/env python3
"""
Investigate NN2 Training vs Validation Mismatch

Key question: In leave-one-out training, what was used as current_sensors
for the held-out sensor? This is critical because:
- Training: Held-out sensor's current_sensors = 0 or PINN (not actual reading)
- Validation: We're using actual readings for ALL sensors

This mismatch could explain why NN2 performs worse in validation.
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

def test_leave_one_out_scenario():
    """
    Test NN2 as it was trained: with held-out sensor's current_sensors = 0
    vs validation approach: with held-out sensor's current_sensors = actual reading
    """
    print("\n" + "="*80)
    print("TESTING LEAVE-ONE-OUT SCENARIO")
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
    
    print("\nTesting on sample timestamps...")
    for idx in range(min(200, len(merged))):
        row = merged.iloc[idx]
        timestamp = row['timestamp']
        
        # Get PINN predictions
        pinn_values = {}
        for sensor_id in SENSORS.keys():
            col = f'sensor_{sensor_id}_pinn'
            if col in row and not pd.isna(row[col]):
                pinn_values[sensor_id] = row[col]
            else:
                pinn_values[sensor_id] = 0.0
        
        # Get actual sensor readings
        actual_values = {}
        for sensor_id in SENSORS.keys():
            col = f'sensor_{sensor_id}_actual'
            if col in row and not pd.isna(row[col]):
                actual_values[sensor_id] = row[col]
            else:
                actual_values[sensor_id] = 0.0
        
        # Test each sensor as "held-out"
        for held_out_idx, held_out_sensor_id in enumerate(sensor_ids_sorted):
            # Strategy 1: Leave-one-out (as trained)
            # Held-out sensor's current_sensors = 0 (or PINN)
            current_sensors_loo = {}
            for sensor_id in sensor_ids_sorted:
                if sensor_id == held_out_sensor_id:
                    # Held-out: use 0 (as in training)
                    current_sensors_loo[sensor_id] = 0.0
                else:
                    # Training sensors: use actual reading
                    current_sensors_loo[sensor_id] = actual_values.get(sensor_id, 0.0)
            
            # Strategy 2: Validation (current approach)
            # All sensors use actual readings
            current_sensors_val = actual_values.copy()
            
            # Prepare arrays
            pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
            current_sensors_loo_array = np.array([current_sensors_loo[sid] for sid in sensor_ids_sorted])
            current_sensors_val_array = np.array([current_sensors_val.get(sid, 0.0) for sid in sensor_ids_sorted])
            
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
            
            # Scale
            pinn_nonzero_mask = pinn_array != 0.0
            loo_nonzero_mask = current_sensors_loo_array != 0.0
            val_nonzero_mask = current_sensors_val_array != 0.0
            
            p_s = np.zeros_like(pinn_array)
            if pinn_nonzero_mask.any():
                p_s[pinn_nonzero_mask] = scalers['pinn'].transform(
                    pinn_array[pinn_nonzero_mask].reshape(-1, 1)
                ).flatten()
            p_s = p_s.reshape(1, -1)
            
            s_loo = np.zeros_like(current_sensors_loo_array)
            if loo_nonzero_mask.any():
                s_loo[loo_nonzero_mask] = scalers['sensors'].transform(
                    current_sensors_loo_array[loo_nonzero_mask].reshape(-1, 1)
                ).flatten()
            s_loo = s_loo.reshape(1, -1)
            
            s_val = np.zeros_like(current_sensors_val_array)
            if val_nonzero_mask.any():
                s_val[val_nonzero_mask] = scalers['sensors'].transform(
                    current_sensors_val_array[val_nonzero_mask].reshape(-1, 1)
                ).flatten()
            s_val = s_val.reshape(1, -1)
            
            w_s = scalers['wind'].transform(np.array([[meteo_u, meteo_v]]))
            d_s = scalers['diffusion'].transform(np.array([[meteo_D]]))
            c_s = scalers['coords'].transform(sensor_coords_spatial)
            
            # Run NN2 with LOO approach
            p_tensor = torch.tensor(p_s, dtype=torch.float32)
            s_loo_tensor = torch.tensor(s_loo, dtype=torch.float32)
            c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)
            w_tensor = torch.tensor(w_s, dtype=torch.float32)
            d_tensor = torch.tensor(d_s, dtype=torch.float32)
            t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)
            
            with torch.no_grad():
                corrected_loo_scaled, _ = nn2(s_loo_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
            
            # Run NN2 with validation approach
            s_val_tensor = torch.tensor(s_val, dtype=torch.float32)
            with torch.no_grad():
                corrected_val_scaled, _ = nn2(s_val_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
            
            # Inverse transform
            corrected_loo_np = corrected_loo_scaled.cpu().numpy().flatten()
            corrected_val_np = corrected_val_scaled.cpu().numpy().flatten()
            
            # Inverse transform for held-out sensor
            held_out_actual = actual_values.get(held_out_sensor_id, 0.0)
            held_out_pinn = pinn_values[held_out_sensor_id]
            
            # LOO result
            if loo_nonzero_mask[held_out_idx]:
                nn2_loo = scalers['sensors'].inverse_transform(
                    corrected_loo_np[held_out_idx].reshape(-1, 1)
                )[0, 0]
            else:
                # If held-out sensor was 0, check if output is non-zero
                if np.abs(corrected_loo_np[held_out_idx]) > 1e-6:
                    nn2_loo = scalers['sensors'].inverse_transform(
                        corrected_loo_np[held_out_idx].reshape(-1, 1)
                    )[0, 0]
                else:
                    nn2_loo = 0.0
            
            # Validation result
            if val_nonzero_mask[held_out_idx]:
                nn2_val = scalers['sensors'].inverse_transform(
                    corrected_val_np[held_out_idx].reshape(-1, 1)
                )[0, 0]
            else:
                if np.abs(corrected_val_np[held_out_idx]) > 1e-6:
                    nn2_val = scalers['sensors'].inverse_transform(
                        corrected_val_np[held_out_idx].reshape(-1, 1)
                    )[0, 0]
                else:
                    nn2_val = 0.0
            
            results.append({
                'timestamp': timestamp,
                'held_out_sensor': held_out_sensor_id,
                'actual': held_out_actual,
                'pinn': held_out_pinn,
                'nn2_loo': nn2_loo,
                'nn2_val': nn2_val,
                'pinn_error': np.abs(held_out_actual - held_out_pinn),
                'nn2_loo_error': np.abs(held_out_actual - nn2_loo),
                'nn2_val_error': np.abs(held_out_actual - nn2_val),
            })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("RESULTS: Leave-One-Out vs Validation Approach")
    print("="*80)
    
    pinn_mae = results_df['pinn_error'].mean()
    nn2_loo_mae = results_df['nn2_loo_error'].mean()
    nn2_val_mae = results_df['nn2_val_error'].mean()
    
    print(f"\nPINN MAE:                    {pinn_mae:.4f} ppb")
    print(f"NN2 LOO approach (as trained): {nn2_loo_mae:.4f} ppb")
    print(f"NN2 Validation approach:      {nn2_val_mae:.4f} ppb")
    
    loo_improvement = ((pinn_mae - nn2_loo_mae) / pinn_mae) * 100 if pinn_mae > 0 else 0
    val_improvement = ((pinn_mae - nn2_val_mae) / pinn_mae) * 100 if pinn_mae > 0 else 0
    
    print(f"\nLOO Improvement: {loo_improvement:.1f}%")
    print(f"Validation Improvement: {val_improvement:.1f}%")
    
    # Per-sensor breakdown
    print("\n" + "="*80)
    print("PER-SENSOR BREAKDOWN")
    print("="*80)
    
    for sensor_id in sensor_ids_sorted:
        sensor_data = results_df[results_df['held_out_sensor'] == sensor_id]
        if len(sensor_data) == 0:
            continue
        
        sensor_pinn_mae = sensor_data['pinn_error'].mean()
        sensor_loo_mae = sensor_data['nn2_loo_error'].mean()
        sensor_val_mae = sensor_data['nn2_val_error'].mean()
        
        print(f"\n{sensor_id}:")
        print(f"  PINN MAE: {sensor_pinn_mae:.4f} ppb")
        print(f"  NN2 LOO:  {sensor_loo_mae:.4f} ppb ({((sensor_pinn_mae - sensor_loo_mae) / sensor_pinn_mae * 100) if sensor_pinn_mae > 0 else 0:.1f}% improvement)")
        print(f"  NN2 Val:  {sensor_val_mae:.4f} ppb ({((sensor_pinn_mae - sensor_val_mae) / sensor_pinn_mae * 0) if sensor_pinn_mae > 0 else 0:.1f}% improvement)")
    
    # Save results
    output_path = PROJECT_DIR / 'validation_results' / 'nn2_loo_vs_validation.csv'
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved results to {output_path}")
    
    return results_df

if __name__ == '__main__':
    print("="*80)
    print("NN2 TRAINING VS VALIDATION MISMATCH INVESTIGATION")
    print("="*80)
    
    results_df = test_leave_one_out_scenario()
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)

