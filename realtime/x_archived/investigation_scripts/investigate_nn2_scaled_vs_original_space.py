#!/usr/bin/env python3
"""
Investigate NN2 Performance in Scaled vs Original Space

CRITICAL DISCOVERY: Training code evaluates in SCALED SPACE, not original ppb space!
This script compares NN2 performance in both spaces to understand the discrepancy.
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
    print("Loading models...")
    
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    print("  ✓ Models loaded")
    print(f"\n  Scaler Statistics:")
    print(f"    Sensors: mean={scalers['sensors'].mean_[0]:.4f}, std={scalers['sensors'].scale_[0]:.4f}")
    print(f"    PINN:    mean={scalers['pinn'].mean_[0]:.4f}, std={scalers['pinn'].scale_[0]:.4f}")
    
    return nn2, scalers, sensor_coords_spatial

def load_training_meteorology():
    """Load meteorology from single file (matching training)"""
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    if len(facility_files) == 0:
        return None
    
    first_file = facility_files[0]
    df = pd.read_csv(first_file)
    
    if 't' in df.columns:
        df['timestamp'] = pd.to_datetime(df['t'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        return None
    
    return df

def run_nn2_in_scaled_space(nn2, scalers, sensor_coords_spatial, pinn_values, current_sensors, 
                            meteo_u, meteo_v, meteo_D, timestamp):
    """Run NN2 and return predictions in SCALED SPACE (matching training evaluation)"""
    sensor_ids_sorted = sorted(SENSORS.keys())
    pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
    current_sensors_array = np.array([current_sensors.get(sid, 0.0) for sid in sensor_ids_sorted])
    
    # Temporal features
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
    
    # Scale inputs (zero-masking, matching training)
    pinn_nonzero_mask = pinn_array != 0.0
    sensors_nonzero_mask = current_sensors_array != 0.0
    
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
    
    w_s = scalers['wind'].transform(np.array([[meteo_u, meteo_v]]))
    d_s = scalers['diffusion'].transform(np.array([[meteo_D]]))
    c_s = scalers['coords'].transform(sensor_coords_spatial)
    
    # Convert to tensors
    p_tensor = torch.tensor(p_s, dtype=torch.float32)
    s_tensor = torch.tensor(s_s, dtype=torch.float32)
    c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)
    w_tensor = torch.tensor(w_s, dtype=torch.float32)
    d_tensor = torch.tensor(d_s, dtype=torch.float32)
    t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)
    
    # Run NN2 (outputs are in SCALED SPACE)
    with torch.no_grad():
        corrected_scaled, corrections = nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
    
    corrected_scaled_np = corrected_scaled.cpu().numpy().flatten()
    
    # Return scaled predictions and also scaled targets for comparison
    nn2_scaled = {sid: corrected_scaled_np[i] for i, sid in enumerate(sensor_ids_sorted)}
    target_scaled = {sid: s_s[0, i] for i, sid in enumerate(sensor_ids_sorted)}
    pinn_scaled = {sid: p_s[0, i] for i, sid in enumerate(sensor_ids_sorted)}
    
    return nn2_scaled, target_scaled, pinn_scaled

def investigate_scaled_vs_original():
    """Compare NN2 performance in scaled space vs original ppb space"""
    print("="*80)
    print("INVESTIGATING NN2: SCALED SPACE vs ORIGINAL SPACE")
    print("="*80)
    print("\nCRITICAL: Training code evaluates in SCALED SPACE, not original ppb!")
    print("This may explain the discrepancy between training results and validation.\n")
    
    # Load models
    nn2, scalers, sensor_coords_spatial = load_models()
    
    # Load training data
    training_pinn_df = pd.read_csv(TRAINING_PINN_PATH)
    training_pinn_df['timestamp'] = pd.to_datetime(training_pinn_df['timestamp'])
    
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    merged = training_pinn_df.merge(sensor_df, on='timestamp', how='inner', suffixes=('_pinn', '_actual'))
    
    print(f"\nTesting on {len(merged)} samples from training data")
    
    # Load meteorology
    meteo_df = load_training_meteorology()
    if meteo_df is None:
        print("ERROR: Could not load meteorology!")
        return
    
    sensor_ids_sorted = sorted(SENSORS.keys())
    results = []
    
    print("\nProcessing samples...")
    for idx, row in merged.head(1000).iterrows():  # Test on 1000 samples
        timestamp = row['timestamp']
        
        # Get PINN values (original ppb)
        pinn_values = {}
        for sensor_id in SENSORS.keys():
            col = f'sensor_{sensor_id}_pinn'
            if col in row and not pd.isna(row[col]):
                pinn_values[sensor_id] = row[col]
            else:
                pinn_values[sensor_id] = 0.0
        
        # Get actual sensor readings (original ppb)
        actual_values = {}
        for sensor_id in SENSORS.keys():
            col = f'sensor_{sensor_id}_actual'
            if col in row and not pd.isna(row[col]):
                actual_values[sensor_id] = row[col]
            else:
                actual_values[sensor_id] = 0.0
        
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
        
        # Run NN2 in scaled space (matching training evaluation)
        nn2_scaled, target_scaled, pinn_scaled = run_nn2_in_scaled_space(
            nn2, scalers, sensor_coords_spatial, pinn_values, actual_values,
            real_u, real_v, real_D, timestamp
        )
        
        # Also compute in original space (current validation approach)
        # Scale inputs
        pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
        current_sensors_array = np.array([actual_values.get(sid, 0.0) for sid in sensor_ids_sorted])
        
        pinn_nonzero_mask = pinn_array != 0.0
        sensors_nonzero_mask = current_sensors_array != 0.0
        
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
        
        p_tensor = torch.tensor(p_s, dtype=torch.float32)
        s_tensor = torch.tensor(s_s, dtype=torch.float32)
        c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)
        w_tensor = torch.tensor(w_s, dtype=torch.float32)
        d_tensor = torch.tensor(d_s, dtype=torch.float32)
        
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
        t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)
        
        with torch.no_grad():
            corrected_scaled, corrections = nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
        
        corrected_scaled_np = corrected_scaled.cpu().numpy().flatten()
        
        # Inverse transform to original space
        nn2_original = np.zeros_like(corrected_scaled_np)
        if sensors_nonzero_mask.any():
            nn2_original[sensors_nonzero_mask] = scalers['sensors'].inverse_transform(
                corrected_scaled_np[sensors_nonzero_mask].reshape(-1, 1)
            ).flatten()
        
        # Compare for each sensor
        for i, sensor_id in enumerate(sensor_ids_sorted):
            actual_ppb = actual_values.get(sensor_id, 0.0)
            pinn_ppb = pinn_values[sensor_id]
            
            # Scaled space (training evaluation method)
            nn2_scaled_val = nn2_scaled[sensor_id]
            target_scaled_val = target_scaled[sensor_id]
            pinn_scaled_val = pinn_scaled[sensor_id]
            
            # Original space (current validation method)
            nn2_original_val = nn2_original[i]
            
            # Compute errors
            error_scaled = abs(nn2_scaled_val - target_scaled_val)
            error_original = abs(nn2_original_val - actual_ppb)
            pinn_error_scaled = abs(pinn_scaled_val - target_scaled_val)
            pinn_error_original = abs(pinn_ppb - actual_ppb)
            
            results.append({
                'timestamp': timestamp,
                'sensor_id': sensor_id,
                'actual_ppb': actual_ppb,
                'pinn_ppb': pinn_ppb,
                'nn2_original_ppb': nn2_original_val,
                'target_scaled': target_scaled_val,
                'pinn_scaled': pinn_scaled_val,
                'nn2_scaled': nn2_scaled_val,
                'pinn_error_scaled': pinn_error_scaled,
                'nn2_error_scaled': error_scaled,
                'pinn_error_original': pinn_error_original,
                'nn2_error_original': error_original,
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("ERROR: No results generated!")
        return
    
    # Analyze results
    print("\n" + "="*80)
    print("RESULTS: Scaled Space vs Original Space")
    print("="*80)
    
    print(f"\nTotal samples: {len(results_df)}")
    
    # Scaled space (training evaluation method)
    print("\n" + "-"*80)
    print("SCALED SPACE (Training Evaluation Method)")
    print("-"*80)
    pinn_mae_scaled = results_df['pinn_error_scaled'].mean()
    nn2_mae_scaled = results_df['nn2_error_scaled'].mean()
    improvement_scaled = ((pinn_mae_scaled - nn2_mae_scaled) / pinn_mae_scaled * 100) if pinn_mae_scaled > 0 else 0
    
    print(f"  PINN MAE (scaled): {pinn_mae_scaled:.6f}")
    print(f"  NN2 MAE (scaled):  {nn2_mae_scaled:.6f}")
    print(f"  Improvement:      {improvement_scaled:.1f}%")
    
    # Original space (current validation method)
    print("\n" + "-"*80)
    print("ORIGINAL SPACE (Current Validation Method)")
    print("-"*80)
    pinn_mae_original = results_df['pinn_error_original'].mean()
    nn2_mae_original = results_df['nn2_error_original'].mean()
    improvement_original = ((pinn_mae_original - nn2_mae_original) / pinn_mae_original * 100) if pinn_mae_original > 0 else 0
    
    print(f"  PINN MAE (ppb):   {pinn_mae_original:.4f} ppb")
    print(f"  NN2 MAE (ppb):    {nn2_mae_original:.4f} ppb")
    print(f"  Improvement:      {improvement_original:.1f}%")
    
    # Comparison
    print("\n" + "="*80)
    print("KEY FINDING")
    print("="*80)
    
    if improvement_scaled > 0 and improvement_original < 0:
        print("\n⚠️  CRITICAL DISCOVERY:")
        print(f"   → NN2 performs WELL in scaled space: {improvement_scaled:.1f}% improvement")
        print(f"   → NN2 performs POORLY in original space: {improvement_original:.1f}% improvement")
        print("\n   This suggests the inverse transform is causing the degradation!")
        print("   The model was trained and evaluated in scaled space.")
        print("   When inverse transformed to ppb, performance degrades.")
    elif improvement_scaled > improvement_original:
        print(f"\n   NN2 performs better in scaled space ({improvement_scaled:.1f}%)")
        print(f"   than in original space ({improvement_original:.1f}%)")
        print("   This indicates inverse transform issues.")
    else:
        print(f"\n   Performance is similar in both spaces.")
        print(f"   Scaled: {improvement_scaled:.1f}%, Original: {improvement_original:.1f}%")
    
    # Save results
    output_path = PROJECT_DIR / 'validation_results' / 'nn2_scaled_vs_original_space.csv'
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved detailed results to {output_path}")
    
    return results_df

if __name__ == '__main__':
    print("="*80)
    print("NN2 SCALED vs ORIGINAL SPACE INVESTIGATION")
    print("="*80)
    
    results_df = investigate_scaled_vs_original()
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)

