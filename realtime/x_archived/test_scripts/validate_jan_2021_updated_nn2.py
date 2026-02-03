#!/usr/bin/env python3
"""
Validate January 2021 predictions using updated NN2 model
Calculates MAE for PINN and NN2-corrected predictions
"""

import sys
import os
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting')

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Import models
try:
    from pinn import ParametricADEPINN
    from nn2 import NN2_CorrectionNetwork
except ImportError:
    # Try alternative path
    sys.path.insert(0, '/Users/neevpratap/simpletesting')
    from pinn import ParametricADEPINN
    from nn2 import NN2_CorrectionNetwork

import pickle

# Paths
BASE_DIR = "/Users/neevpratap/simpletesting"
MADIS_DIR = "/Users/neevpratap/Desktop/madis_data_desktop_updated"
PROJECT_DIR = Path("/Users/neevpratap/Desktop/benzenepipelinev2/realtime")

# Model paths - using updated NN2
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
NN2_MODEL_PATH = PROJECT_DIR / "nn2_updated/nn2_master_model_spatial-3.pth"
NN2_SCALERS_PATH = PROJECT_DIR / "nn2_updated/nn2_master_scalers-2.pkl"

# Data paths
PINN_PREDICTIONS_PATH = BASE_DIR + "/sensors_pinn_benzene_ppb_2021_jan.csv"
GROUND_TRUTH_PATH = MADIS_DIR + "/results_2021/sensors_actual_wide_2021_full_jan.csv"
TRAINING_DATA_DIR = MADIS_DIR + "/training_data_2021_full_jan_REPAIRED"

# Constants
UNIT_CONVERSION = 313210039.9  # kg/m^2 to ppb

# 9 sensor coordinates (Cartesian)
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
    """Load PINN and NN2 models"""
    print("Loading models...")
    
    # Load PINN
    pinn = ParametricADEPINN()
    checkpoint = torch.load(PINN_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    
    # Override normalization ranges (benchmark values)
    pinn.x_min = torch.tensor(0.0)
    pinn.x_max = torch.tensor(30000.0)
    pinn.y_min = torch.tensor(0.0)
    pinn.y_max = torch.tensor(30000.0)
    pinn.t_min = torch.tensor(0.0)
    pinn.t_max = torch.tensor(8760.0)
    pinn.cx_min = torch.tensor(0.0)
    pinn.cx_max = torch.tensor(30000.0)
    pinn.cy_min = torch.tensor(0.0)
    pinn.cy_max = torch.tensor(30000.0)
    pinn.u_min = torch.tensor(-15.0)
    pinn.u_max = torch.tensor(15.0)
    pinn.v_min = torch.tensor(-15.0)
    pinn.v_max = torch.tensor(15.0)
    pinn.d_min = torch.tensor(0.0)
    pinn.d_max = torch.tensor(200.0)
    pinn.kappa_min = torch.tensor(0.0)
    pinn.kappa_max = torch.tensor(200.0)
    pinn.Q_min = torch.tensor(0.0)
    pinn.Q_max = torch.tensor(0.01)
    
    pinn.eval()
    
    # Load NN2
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    nn2_checkpoint = torch.load(str(NN2_MODEL_PATH), map_location='cpu', weights_only=False)
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    # Load scalers
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    # Sensor coords for NN2
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    print("  ✓ Models loaded")
    return pinn, nn2, scalers, sensor_coords_spatial

def apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, meteo_data, timestamp, current_sensor_readings=None, nn2_mapping_model=None):
    """
    Apply NN2 correction to PINN predictions
    
    Args:
        current_sensor_readings: Dict of sensor_id -> actual reading at timestamp
                                 If None, will use PINN predictions as fallback
    """
    # Prepare inputs
    sensor_ids_sorted = sorted(SENSORS.keys())
    pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
    
    # Use actual sensor readings from time t (when prediction is made) as current_sensors
    if current_sensor_readings is not None:
        current_sensors = np.array([current_sensor_readings.get(sid, 0.0) for sid in sensor_ids_sorted])
    else:
        # Fallback: use PINN (not ideal, but allows validation to run)
        current_sensors = pinn_array.copy()
    
    # Meteorology
    avg_u = meteo_data['wind_u'].mean()
    avg_v = meteo_data['wind_v'].mean()
    avg_D = meteo_data['D'].mean()
    
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
    
    # Scale inputs - handle zeros the same way as training (mask before scaling)
    pinn_nonzero_mask = pinn_array != 0.0
    sensors_nonzero_mask = current_sensors != 0.0
    
    # Scale PINN predictions (only non-zero values)
    p_s = np.zeros_like(pinn_array)
    if pinn_nonzero_mask.any():
        p_s[pinn_nonzero_mask] = scalers['pinn'].transform(
            pinn_array[pinn_nonzero_mask].reshape(-1, 1)
        ).flatten()
    p_s = p_s.reshape(1, -1)
    
    # Scale current sensors (only non-zero values)
    s_s = np.zeros_like(current_sensors)
    if sensors_nonzero_mask.any():
        s_s[sensors_nonzero_mask] = scalers['sensors'].transform(
            current_sensors[sensors_nonzero_mask].reshape(-1, 1)
        ).flatten()
    s_s = s_s.reshape(1, -1)
    
    w_s = scalers['wind'].transform(np.array([[avg_u, avg_v]]))
    d_s = scalers['diffusion'].transform(np.array([[avg_D]]))
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
        corrected_scaled, _ = nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
    
    # Inverse transform using Gradient Boosting mapping model (more accurate)
    corrected_scaled_np = corrected_scaled.cpu().numpy().flatten()
    
    if nn2_mapping_model is not None:
        # Use mapping model (fixes out-of-distribution issue)
        nn2_corrected = nn2_scaled_to_ppb(corrected_scaled_np, mapping_model=nn2_mapping_model)
    else:
        # Fallback to original scaler (if mapping model not available)
        nn2_output_nonzero_mask = np.abs(corrected_scaled_np) > 1e-6
        nn2_corrected = np.zeros_like(corrected_scaled_np)
        if nn2_output_nonzero_mask.any():
            nn2_corrected[nn2_output_nonzero_mask] = scalers['sensors'].inverse_transform(
                corrected_scaled_np[nn2_output_nonzero_mask].reshape(-1, 1)
            ).flatten()
    
    # Return as dict
    nn2_values = {sid: nn2_corrected[i] for i, sid in enumerate(sensor_ids_sorted)}
    
    return nn2_values

def main():
    print("="*80)
    print("JANUARY 2021 VALIDATION - UPDATED NN2")
    print("="*80)
    print()
    
    # Load models
    pinn, nn2, scalers, sensor_coords_spatial = load_models()
    
    # Load PINN predictions
    print("\nLoading PINN predictions...")
    df_pinn = pd.read_csv(PINN_PREDICTIONS_PATH)
    if 't' in df_pinn.columns:
        df_pinn = df_pinn.rename(columns={'t': 'timestamp'})
    df_pinn['timestamp'] = pd.to_datetime(df_pinn['timestamp'])
    print(f"  Loaded {len(df_pinn)} PINN predictions")
    
    # Load ground truth
    print("\nLoading ground truth...")
    df_gt = pd.read_csv(GROUND_TRUTH_PATH)
    if 't' in df_gt.columns:
        df_gt = df_gt.rename(columns={'t': 'timestamp'})
    df_gt['timestamp'] = pd.to_datetime(df_gt['timestamp'])
    print(f"  Loaded {len(df_gt)} ground truth timestamps")
    
    # Load facility data for meteo (needed for NN2)
    print("\nLoading facility data for meteo...")
    facility_files = sorted(Path(TRAINING_DATA_DIR).glob('*_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    facility_files_dict = {}
    for f in facility_files:
        facility_name = f.stem.replace('_training_data', '')
        df = pd.read_csv(f)
        if 't' in df.columns:
            df['timestamp'] = pd.to_datetime(df['t'])
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        facility_files_dict[facility_name] = df
    
    print(f"  Loaded {len(facility_files_dict)} facilities")
    
    # Merge PINN predictions with ground truth
    print("\nMerging datasets...")
    df_merged = df_pinn.merge(df_gt, on='timestamp', how='inner', suffixes=('_pinn', '_gt'))
    print(f"  {len(df_merged)} matching timestamps")
    
    # Process each timestamp
    print("\nProcessing predictions with NN2 correction...")
    results = []
    
    for idx, row in df_merged.iterrows():
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(df_merged)}")
        
        timestamp = row['timestamp']
        
        # Extract PINN predictions
        pinn_values = {}
        for sensor_id in SENSORS.keys():
            col = f'sensor_{sensor_id}_pinn'
            if col in row and not pd.isna(row[col]):
                pinn_values[sensor_id] = row[col]
            else:
                # Try without suffix
                col_alt = f'sensor_{sensor_id}'
                if col_alt in df_pinn.columns:
                    pinn_values[sensor_id] = row.get(col_alt, 0.0)
                else:
                    pinn_values[sensor_id] = 0.0
        
        # Get meteo data for NN2 (average across all facilities at this timestamp)
        meteo_data_list = []
        for facility_name, facility_df in facility_files_dict.items():
            facility_data = facility_df[facility_df['timestamp'] == timestamp]
            if len(facility_data) > 0:
                meteo_data_list.append(facility_data)
        
        if len(meteo_data_list) == 0:
            continue
        
        # Combine all facility meteo data for averaging
        combined_meteo = pd.concat(meteo_data_list, ignore_index=True)
        
        # Get actual sensor readings for NN2's "current_sensors" input
        current_sensor_readings = {}
        for sensor_id in SENSORS.keys():
            col = f'sensor_{sensor_id}_gt'
            if col in row and not pd.isna(row[col]):
                current_sensor_readings[sensor_id] = row[col]
            else:
                # Try without suffix
                col_alt = f'sensor_{sensor_id}'
                if col_alt in df_gt.columns:
                    current_sensor_readings[sensor_id] = row.get(col_alt, 0.0)
                else:
                    current_sensor_readings[sensor_id] = 0.0
        
        # Apply NN2 correction
        nn2_values = apply_nn2_correction(
            nn2, scalers, sensor_coords_spatial, 
            pinn_values, combined_meteo, timestamp, 
            current_sensor_readings
        )
        
        # Collect results
        for sensor_id in SENSORS.keys():
            col_gt = f'sensor_{sensor_id}_gt'
            if col_gt not in row:
                col_gt = f'sensor_{sensor_id}'
            
            if col_gt not in row or pd.isna(row[col_gt]):
                continue
            
            results.append({
                'timestamp': timestamp,
                'sensor_id': sensor_id,
                'actual': row[col_gt],
                'pinn': pinn_values[sensor_id],
                'nn2': nn2_values[sensor_id]
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute MAE
    print("\n" + "="*80)
    print("RESULTS - JANUARY 2021")
    print("="*80)
    
    if len(results_df) == 0:
        print("\n⚠ No valid results to analyze!")
        return
    
    # Per-sensor MAE
    print("\nPer-Sensor MAE:")
    print("-" * 80)
    for sensor_id in sorted(SENSORS.keys()):
        sensor_data = results_df[results_df['sensor_id'] == sensor_id]
        
        if len(sensor_data) == 0:
            continue
        
        pinn_mae = np.mean(np.abs(sensor_data['actual'] - sensor_data['pinn']))
        nn2_mae = np.mean(np.abs(sensor_data['actual'] - sensor_data['nn2']))
        improvement = ((pinn_mae - nn2_mae) / pinn_mae) * 100 if pinn_mae > 0 else 0.0
        
        print(f"\nSensor {sensor_id}:")
        print(f"  PINN MAE:   {pinn_mae:.4f} ppb")
        print(f"  NN2 MAE:    {nn2_mae:.4f} ppb")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Samples: {len(sensor_data)}")
    
    # Overall MAE
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    pinn_overall_mae = np.mean(np.abs(results_df['actual'] - results_df['pinn']))
    nn2_overall_mae = np.mean(np.abs(results_df['actual'] - results_df['nn2']))
    overall_improvement = ((pinn_overall_mae - nn2_overall_mae) / pinn_overall_mae) * 100 if pinn_overall_mae > 0 else 0.0
    
    print(f"\nPINN MAE:   {pinn_overall_mae:.4f} ppb")
    print(f"NN2 MAE:    {nn2_overall_mae:.4f} ppb")
    print(f"Improvement: {overall_improvement:.1f}%")
    print(f"Total samples: {len(results_df)}")
    
    # Save results
    output_path = PROJECT_DIR / "validation_results" / "jan_2021_updated_nn2.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

