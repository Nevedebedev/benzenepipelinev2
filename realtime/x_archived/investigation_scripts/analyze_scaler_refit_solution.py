#!/usr/bin/env python3
"""
Analyze the best solution for the inverse transform issue.

Options:
1. Retrain scaler on NN2 output distribution (scalable, no model retraining)
2. Clip outputs before inverse transform (simple but loses information)
3. Evaluate in scaled space (accurate but not interpretable)
4. Add constraints to model (requires retraining)

This script analyzes option 1: Retraining the scaler on NN2 outputs.
"""

import sys
from pathlib import Path
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

import torch
import pandas as pd
import numpy as np
from nn2 import NN2_CorrectionNetwork
from sklearn.preprocessing import StandardScaler
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

def analyze_scaler_refit_solution():
    """Analyze retraining the scaler on NN2 output distribution"""
    print("="*80)
    print("ANALYZING SCALER REFIT SOLUTION")
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
    
    print("\nGenerating NN2 outputs on full training data...")
    all_nn2_outputs_scaled = []
    all_actual_values = []
    
    for idx, row in merged.iterrows():
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
        
        # Collect NN2 outputs and actuals
        for i, sensor_id in enumerate(sensor_ids_sorted):
            nn2_output_scaled = corrected_scaled_np[i]
            actual_ppb = actual_values.get(sensor_id, 0.0)
            
            # Only collect non-zero actuals (matching training scaler fit)
            if actual_ppb != 0.0:
                all_nn2_outputs_scaled.append(nn2_output_scaled)
                all_actual_values.append(actual_ppb)
    
    all_nn2_outputs_scaled = np.array(all_nn2_outputs_scaled)
    all_actual_values = np.array(all_actual_values)
    
    print(f"\nCollected {len(all_nn2_outputs_scaled)} non-zero samples")
    
    # Analyze distributions
    print("\n" + "="*80)
    print("DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print(f"\nOriginal Scaler (fit on actual sensor values):")
    print(f"  Mean: {scalers['sensors'].mean_[0]:.4f}")
    print(f"  Std:  {scalers['sensors'].scale_[0]:.4f}")
    print(f"  Range: [{scalers['sensors'].mean_[0] - 3*scalers['sensors'].scale_[0]:.4f}, "
          f"{scalers['sensors'].mean_[0] + 3*scalers['sensors'].scale_[0]:.4f}]")
    
    print(f"\nNN2 Output Distribution (scaled space):")
    print(f"  Min:  {all_nn2_outputs_scaled.min():.4f}")
    print(f"  Max:  {all_nn2_outputs_scaled.max():.4f}")
    print(f"  Mean: {all_nn2_outputs_scaled.mean():.4f}")
    print(f"  Std:  {all_nn2_outputs_scaled.std():.4f}")
    print(f"  5th percentile: {np.percentile(all_nn2_outputs_scaled, 5):.4f}")
    print(f"  95th percentile: {np.percentile(all_nn2_outputs_scaled, 95):.4f}")
    
    print(f"\nActual Values Distribution (ppb):")
    print(f"  Min:  {all_actual_values.min():.4f}")
    print(f"  Max:  {all_actual_values.max():.4f}")
    print(f"  Mean: {all_actual_values.mean():.4f}")
    print(f"  Std:  {all_actual_values.std():.4f}")
    
    # Option 1: Fit new scaler on NN2 outputs mapped to actuals
    print("\n" + "="*80)
    print("OPTION 1: Fit Scaler on NN2 Output → Actual Mapping")
    print("="*80)
    
    # Create a mapping: NN2 output (scaled) → Actual (ppb)
    # Fit scaler on actual values, but use it to inverse transform NN2 outputs
    new_scaler = StandardScaler()
    new_scaler.fit(all_actual_values.reshape(-1, 1))
    
    print(f"\nNew Scaler (fit on actual values):")
    print(f"  Mean: {new_scaler.mean_[0]:.4f}")
    print(f"  Std:  {new_scaler.scale_[0]:.4f}")
    
    # But this doesn't help - we need to map scaled NN2 outputs to ppb
    # The issue is that NN2 outputs are in scaled space, not original space
    
    # Option 2: Fit scaler on NN2 outputs directly (for inverse transform)
    print("\n" + "="*80)
    print("OPTION 2: Fit New Scaler on NN2 Outputs (Scaled Space)")
    print("="*80)
    
    # This doesn't make sense - we need to map scaled outputs to ppb
    
    # Option 3: Create mapping function from NN2 scaled outputs to actual ppb
    print("\n" + "="*80)
    print("OPTION 3: Direct Mapping (NN2 Scaled Output → Actual PPB)")
    print("="*80)
    
    # Fit a scaler that maps NN2 outputs (scaled) to actual values (ppb)
    # This is essentially learning the inverse transform
    nn2_to_actual_scaler = StandardScaler()
    nn2_to_actual_scaler.fit(all_nn2_outputs_scaled.reshape(-1, 1))
    
    # But we need the inverse: given NN2 output (scaled), predict actual (ppb)
    # This requires a different approach - we need to learn the mapping
    
    # Actually, the real solution is to fit a scaler on the actual distribution
    # of what NN2 outputs should map to (the actual values)
    # But use it differently - we need to understand the relationship
    
    # Let's test: what if we fit the scaler on actual values but use it
    # to inverse transform by first "unscaling" the NN2 output?
    
    # Actually, I think the real issue is simpler:
    # The scaler was fit on actual sensor values (0-10 ppb range)
    # But NN2 outputs in scaled space can be outside the range that maps back to 0-10 ppb
    # We need to fit a scaler that can handle the full range of NN2 outputs
    
    # Best solution: Fit scaler on the actual values that correspond to NN2 outputs
    # But we already have that - it's the actual_values
    # The problem is the inverse transform of out-of-range values
    
    # Let's test clipping approach
    print("\n" + "="*80)
    print("OPTION 4: Clip NN2 Outputs to Scaler Range Before Inverse Transform")
    print("="*80)
    
    # Calculate the range of the original scaler in scaled space
    # Original scaler maps: ppb → scaled
    # We need: scaled → ppb (inverse)
    # The range in scaled space that maps to valid ppb is:
    # For ppb in [0, max_ppb], what is the scaled range?
    
    # Test with actual values
    test_ppb = np.linspace(0, 20, 1000)  # Test range
    test_scaled = scalers['sensors'].transform(test_ppb.reshape(-1, 1)).flatten()
    
    print(f"\nOriginal scaler mapping (ppb → scaled):")
    print(f"  PPB [0, 20] maps to scaled: [{test_scaled.min():.4f}, {test_scaled.max():.4f}]")
    
    # Clip NN2 outputs to this range
    clipped_nn2_outputs = np.clip(all_nn2_outputs_scaled, test_scaled.min(), test_scaled.max())
    
    # Inverse transform clipped outputs
    clipped_ppb = scalers['sensors'].inverse_transform(clipped_nn2_outputs.reshape(-1, 1)).flatten()
    
    # Compare errors
    original_ppb = scalers['sensors'].inverse_transform(all_nn2_outputs_scaled.reshape(-1, 1)).flatten()
    
    error_original = np.abs(original_ppb - all_actual_values).mean()
    error_clipped = np.abs(clipped_ppb - all_actual_values).mean()
    error_pinn = np.abs(all_actual_values - all_actual_values).mean()  # Just for reference
    
    print(f"\nError comparison:")
    print(f"  Original inverse transform: {error_original:.4f} ppb")
    print(f"  Clipped inverse transform:  {error_clipped:.4f} ppb")
    print(f"  Improvement: {((error_original - error_clipped) / error_original * 100):.1f}%")
    
    # Option 5: Fit new scaler on wider range
    print("\n" + "="*80)
    print("OPTION 5: Fit New Scaler on Extended Range")
    print("="*80)
    
    # Fit scaler on actual values but with extended range to cover NN2 outputs
    # We need to map NN2 scaled outputs to actual ppb values
    # The issue is: NN2 outputs are in a different scaled space
    
    # Actually, the real solution is to understand:
    # NN2 outputs corrected_scaled = pinn_scaled + corrections
    # We want to inverse transform this to ppb
    # But the scaler was fit on actual sensor values, not on NN2 outputs
    
    # The correct approach: Fit a new scaler specifically for NN2 outputs
    # by learning the mapping from NN2 scaled outputs to actual ppb values
    
    # Create extended range for scaler
    extended_ppb_range = np.concatenate([
        all_actual_values,
        np.linspace(0, 30, 1000)  # Extended range
    ])
    extended_scaler = StandardScaler()
    extended_scaler.fit(extended_ppb_range.reshape(-1, 1))
    
    print(f"\nExtended Scaler:")
    print(f"  Mean: {extended_scaler.mean_[0]:.4f}")
    print(f"  Std:  {extended_scaler.scale_[0]:.4f}")
    
    # But this still doesn't solve the problem - we need to map scaled outputs to ppb
    
    # The REAL solution: We need a scaler that maps NN2 scaled outputs directly to ppb
    # This means learning the inverse relationship
    
    return {
        'nn2_outputs_scaled': all_nn2_outputs_scaled,
        'actual_values': all_actual_values,
        'original_scaler': scalers['sensors'],
        'clipped_error': error_clipped,
        'original_error': error_original,
    }

if __name__ == '__main__':
    print("="*80)
    print("ANALYZING BEST SOLUTION FOR INVERSE TRANSFORM ISSUE")
    print("="*80)
    
    results = analyze_scaler_refit_solution()
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("\nThe most accurate and scalable solution is:")
    print("1. Fit a new scaler on the actual distribution of values that NN2 should predict")
    print("2. Use this scaler to inverse transform NN2 outputs")
    print("3. OR: Clip NN2 outputs to valid range before inverse transform")
    print("4. OR: Evaluate in scaled space (most accurate to training)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

