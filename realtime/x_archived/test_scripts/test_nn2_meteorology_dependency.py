#!/usr/bin/env python3
"""
Test if NN2 Actually Uses Meteorology

This script tests whether meteorology (wind, diffusion) actually affects NN2 predictions
by running NN2 with different meteorology values and comparing outputs.
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
    return nn2, scalers, sensor_coords_spatial

def load_training_meteorology():
    """Load meteorology from the exact file used during training (first alphabetically)"""
    print("Loading training meteorology (BASF_Pasadena - first file alphabetically)...")
    
    # Match training code: sorted(Path(source_dir).glob('*.csv'))[0]
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    if len(facility_files) == 0:
        print("  ERROR: No facility files found!")
        return None
    
    first_file = facility_files[0]
    print(f"  Using: {first_file.name}")
    
    df = pd.read_csv(first_file)
    if 't' in df.columns:
        df['timestamp'] = pd.to_datetime(df['t'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        print("  ERROR: No timestamp column found!")
        return None
    
    print(f"  ✓ Loaded {len(df)} rows")
    return df

def run_nn2_with_meteorology(nn2, scalers, sensor_coords_spatial, pinn_values, current_sensors, 
                             meteo_u, meteo_v, meteo_D, timestamp):
    """Run NN2 with specified meteorology values"""
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
    
    # Scale meteorology
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
    
    # Run NN2
    with torch.no_grad():
        corrected_scaled, corrections = nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
    
    corrected_scaled_np = corrected_scaled.cpu().numpy().flatten()
    
    # Inverse transform
    nn2_corrected = np.zeros_like(corrected_scaled_np)
    if sensors_nonzero_mask.any():
        nn2_corrected[sensors_nonzero_mask] = scalers['sensors'].inverse_transform(
            corrected_scaled_np[sensors_nonzero_mask].reshape(-1, 1)
        ).flatten()
    
    # Return as dict
    nn2_values = {sid: nn2_corrected[i] for i, sid in enumerate(sensor_ids_sorted)}
    return nn2_values

def test_meteorology_dependency():
    """Test if meteorology actually affects NN2 predictions"""
    print("="*80)
    print("TESTING NN2 METEOROLOGY DEPENDENCY")
    print("="*80)
    
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
    
    # Load training meteorology (exact file used during training)
    meteo_df = load_training_meteorology()
    if meteo_df is None:
        print("ERROR: Could not load meteorology!")
        return
    
    # Test on first 100 samples
    test_samples = merged.head(100)
    
    results = []
    
    print("\nRunning tests with different meteorology values...")
    print("  Test 1: Real meteorology (from training)")
    print("  Test 2: Zero meteorology (wind=0, v=0, D=0)")
    print("  Test 3: 2x wind, 0.5x diffusion")
    print("  Test 4: -1x wind, 2x diffusion")
    
    for idx, row in test_samples.iterrows():
        timestamp = row['timestamp']
        
        # Get PINN values
        pinn_values = {}
        for sensor_id in SENSORS.keys():
            col = f'sensor_{sensor_id}_pinn'
            if col in row and not pd.isna(row[col]):
                pinn_values[sensor_id] = row[col]
            else:
                pinn_values[sensor_id] = 0.0
        
        # Get actual sensor readings
        current_sensors = {}
        for sensor_id in SENSORS.keys():
            col = f'sensor_{sensor_id}_actual'
            if col in row and not pd.isna(row[col]):
                current_sensors[sensor_id] = row[col]
            else:
                current_sensors[sensor_id] = 0.0
        
        # Get real meteorology (3 hours before forecast timestamp)
        met_data_timestamp = timestamp - pd.Timedelta(hours=3)
        meteo_row = meteo_df[meteo_df['timestamp'] == met_data_timestamp]
        
        if len(meteo_row) == 0:
            continue
        
        real_u = meteo_row['wind_u'].iloc[0]
        real_v = meteo_row['wind_v'].iloc[0]
        real_D = meteo_row['D'].iloc[0]
        
        if pd.isna(real_u) or pd.isna(real_v) or pd.isna(real_D):
            continue
        
        # Test 1: Real meteorology
        nn2_real = run_nn2_with_meteorology(nn2, scalers, sensor_coords_spatial, 
                                            pinn_values, current_sensors,
                                            real_u, real_v, real_D, timestamp)
        
        # Test 2: Zero meteorology
        nn2_zero = run_nn2_with_meteorology(nn2, scalers, sensor_coords_spatial,
                                           pinn_values, current_sensors,
                                           0.0, 0.0, 0.0, timestamp)
        
        # Test 3: 2x wind, 0.5x diffusion
        nn2_2x = run_nn2_with_meteorology(nn2, scalers, sensor_coords_spatial,
                                          pinn_values, current_sensors,
                                          real_u * 2.0, real_v * 2.0, real_D * 0.5, timestamp)
        
        # Test 4: -1x wind, 2x diffusion
        nn2_neg = run_nn2_with_meteorology(nn2, scalers, sensor_coords_spatial,
                                          pinn_values, current_sensors,
                                          -real_u, -real_v, real_D * 2.0, timestamp)
        
        # Compare results
        for sensor_id in SENSORS.keys():
            diff_zero = abs(nn2_real[sensor_id] - nn2_zero[sensor_id])
            diff_2x = abs(nn2_real[sensor_id] - nn2_2x[sensor_id])
            diff_neg = abs(nn2_real[sensor_id] - nn2_neg[sensor_id])
            
            results.append({
                'timestamp': timestamp,
                'sensor_id': sensor_id,
                'pinn': pinn_values[sensor_id],
                'actual': current_sensors[sensor_id],
                'nn2_real': nn2_real[sensor_id],
                'nn2_zero': nn2_zero[sensor_id],
                'nn2_2x': nn2_2x[sensor_id],
                'nn2_neg': nn2_neg[sensor_id],
                'diff_zero': diff_zero,
                'diff_2x': diff_2x,
                'diff_neg': diff_neg,
                'meteo_u': real_u,
                'meteo_v': real_v,
                'meteo_D': real_D,
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("ERROR: No results generated!")
        return
    
    # Analyze results
    print("\n" + "="*80)
    print("RESULTS: Meteorology Dependency Analysis")
    print("="*80)
    
    print(f"\nTotal samples tested: {len(results_df)}")
    
    # Test 1: Zero vs Real
    print("\n" + "-"*80)
    print("TEST 1: Zero Meteorology vs Real Meteorology")
    print("-"*80)
    mean_diff_zero = results_df['diff_zero'].mean()
    max_diff_zero = results_df['diff_zero'].max()
    identical_count = (results_df['diff_zero'] < 1e-6).sum()
    
    print(f"  Mean absolute difference: {mean_diff_zero:.6f} ppb")
    print(f"  Max difference: {max_diff_zero:.6f} ppb")
    print(f"  Identical predictions: {identical_count} / {len(results_df)} ({100*identical_count/len(results_df):.1f}%)")
    
    if mean_diff_zero < 1e-6:
        print("  → CONCLUSION: Meteorology does NOT affect predictions (ignored by model)")
    else:
        print(f"  → CONCLUSION: Meteorology DOES affect predictions (mean diff: {mean_diff_zero:.4f} ppb)")
    
    # Test 2: 2x wind vs Real
    print("\n" + "-"*80)
    print("TEST 2: 2x Wind, 0.5x Diffusion vs Real Meteorology")
    print("-"*80)
    mean_diff_2x = results_df['diff_2x'].mean()
    max_diff_2x = results_df['diff_2x'].max()
    
    print(f"  Mean absolute difference: {mean_diff_2x:.6f} ppb")
    print(f"  Max difference: {max_diff_2x:.6f} ppb")
    
    # Test 3: Negative wind vs Real
    print("\n" + "-"*80)
    print("TEST 3: -1x Wind, 2x Diffusion vs Real Meteorology")
    print("-"*80)
    mean_diff_neg = results_df['diff_neg'].mean()
    max_diff_neg = results_df['diff_neg'].max()
    
    print(f"  Mean absolute difference: {mean_diff_neg:.6f} ppb")
    print(f"  Max difference: {max_diff_neg:.6f} ppb")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if mean_diff_zero < 1e-6:
        print("\n❌ METEOROLOGY DOES NOT AFFECT NN2 PREDICTIONS")
        print("   → Model ignores meteorology inputs")
        print("   → Meteorology mismatch is NOT the cause of poor performance")
    else:
        print("\n✅ METEOROLOGY DOES AFFECT NN2 PREDICTIONS")
        print(f"   → Mean difference with zero meteo: {mean_diff_zero:.4f} ppb")
        print(f"   → Mean difference with 2x wind: {mean_diff_2x:.4f} ppb")
        print(f"   → Mean difference with -1x wind: {mean_diff_neg:.4f} ppb")
        print("   → Using wrong meteorology WILL affect predictions")
        print("   → Must use exact training meteorology (single file)")
    
    # Save detailed results
    output_path = PROJECT_DIR / 'validation_results' / 'nn2_meteorology_dependency_test.csv'
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved detailed results to {output_path}")
    
    return results_df

if __name__ == '__main__':
    print("="*80)
    print("NN2 METEOROLOGY DEPENDENCY TEST")
    print("="*80)
    
    results_df = test_meteorology_dependency()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

