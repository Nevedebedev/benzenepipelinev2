#!/usr/bin/env python3
"""
Quick test: Apply PPB NN2 model to Jan 2021 with clamping, compare to PINN
DO NOT SAVE RESULTS
"""

import sys
from pathlib import Path
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))
sys.path.append(str(Path(__file__).parent / 'drive-download-20260202T042428Z-3-001'))

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from pinn import ParametricADEPINN
from nn2_ppbscale import NN2_CorrectionNetwork
import pickle

# Paths
PROJECT_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime')
FACILITY_DATA_BASE = PROJECT_DIR / 'realtime_processing/houston_processed_2021'
MADIS_DIR = Path('/Users/neevpratap/Desktop/madis_data_desktop_updated')

PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
NN2_MODEL_PATH = PROJECT_DIR / "nn2_ppbscale/nn2_master_model_ppb.pth"
NN2_SCALERS_PATH = PROJECT_DIR / "nn2_ppbscale/nn2_master_scalers-2.pkl"

# Ground truth sensor data paths
SENSOR_DATA_PATH = MADIS_DIR / "results_2021/sensors_actual_wide_2021_full_jan.csv"

# Facility data directory
FACILITY_DATA_DIR = FACILITY_DATA_BASE / 'training_data_2021_january_complete'

# Constants
UNIT_CONVERSION = 313210039.9  # kg/m^2 to ppb
FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast

# 9 sensor coordinates (Cartesian) - matches training data
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
    """Load PINN and new PPB-scale NN2 models"""
    print("Loading models...")
    
    # Load PINN
    pinn = ParametricADEPINN()
    checkpoint = torch.load(PINN_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    
    # Override normalization ranges
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
    
    # Load NN2 - PPB MODEL
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    scaler_mean = nn2_checkpoint.get('scaler_mean', None)
    scaler_scale = nn2_checkpoint.get('scaler_scale', None)
    output_ppb = nn2_checkpoint.get('output_ppb', True)
    
    if scaler_mean is None or scaler_scale is None:
        with open(NN2_SCALERS_PATH, 'rb') as f:
            scalers_temp = pickle.load(f)
        scaler_mean = scalers_temp['sensors'].mean_[0] if hasattr(scalers_temp['sensors'], 'mean_') else None
        scaler_scale = scalers_temp['sensors'].scale_[0] if hasattr(scalers_temp['sensors'], 'scale_') else None
    
    nn2 = NN2_CorrectionNetwork(
        n_sensors=9,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        output_ppb=output_ppb
    )
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    print("  ✓ Models loaded")
    return pinn, nn2, scalers, sensor_coords_spatial

def predict_pinn_at_sensors(pinn, facility_files_dict, timestamp):
    """Predict PINN at sensor locations"""
    input_timestamp = timestamp - pd.Timedelta(hours=3)
    sensor_pinn_ppb = {sid: 0.0 for sid in SENSORS.keys()}
    
    for facility_name, facility_df in facility_files_dict.items():
        facility_data = facility_df[facility_df['t'] == input_timestamp]
        if len(facility_data) == 0:
            continue
        
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
                        torch.tensor([[FORECAST_T_HOURS]], dtype=torch.float32),
                        torch.tensor([[cx]], dtype=torch.float32),
                        torch.tensor([[cy]], dtype=torch.float32),
                        torch.tensor([[u]], dtype=torch.float32),
                        torch.tensor([[v]], dtype=torch.float32),
                        torch.tensor([[d]], dtype=torch.float32),
                        torch.tensor([[kappa]], dtype=torch.float32),
                        torch.tensor([[Q]], dtype=torch.float32),
                        normalize=True
                    )
                    concentration_ppb = phi_raw.item() * UNIT_CONVERSION
                    sensor_pinn_ppb[sensor_id] += concentration_ppb
    
    return sensor_pinn_ppb

def apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, meteo_data, timestamp, current_sensor_readings):
    """Apply NN2 correction - outputs in ppb"""
    sensor_ids_sorted = sorted(SENSORS.keys())
    pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
    current_sensors = np.array([current_sensor_readings.get(sid, 0.0) for sid in sensor_ids_sorted])
    
    avg_u = meteo_data['wind_u'].mean()
    avg_v = meteo_data['wind_v'].mean()
    avg_D = meteo_data['D'].mean()
    
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
    
    # Scale inputs
    pinn_nonzero_mask = pinn_array != 0.0
    sensors_nonzero_mask = current_sensors != 0.0
    
    p_s = np.zeros_like(pinn_array)
    if pinn_nonzero_mask.any():
        p_s[pinn_nonzero_mask] = scalers['pinn'].transform(
            pinn_array[pinn_nonzero_mask].reshape(-1, 1)
        ).flatten()
    p_s = p_s.reshape(1, -1)
    
    s_s = np.zeros_like(current_sensors)
    if sensors_nonzero_mask.any():
        s_s[sensors_nonzero_mask] = scalers['sensors'].transform(
            current_sensors[sensors_nonzero_mask].reshape(-1, 1)
        ).flatten()
    s_s = s_s.reshape(1, -1)
    
    w_s = scalers['wind'].transform(np.array([[avg_u, avg_v]]))
    d_s = scalers['diffusion'].transform(np.array([[avg_D]]))
    c_s = scalers['coords'].transform(sensor_coords_spatial)
    
    p_tensor = torch.tensor(p_s, dtype=torch.float32)
    s_tensor = torch.tensor(s_s, dtype=torch.float32)
    c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)
    w_tensor = torch.tensor(w_s, dtype=torch.float32)
    d_tensor = torch.tensor(d_s, dtype=torch.float32)
    t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)
    
    with torch.no_grad():
        corrected_ppb, _ = nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
    
    nn2_corrected_ppb = corrected_ppb.cpu().numpy().flatten()
    
    # CLAMP NEGATIVE VALUES TO 0
    nn2_corrected_ppb = np.maximum(nn2_corrected_ppb, 0.0)
    
    nn2_values = {sid: nn2_corrected_ppb[i] for i, sid in enumerate(sensor_ids_sorted)}
    return nn2_values

def main():
    print("="*80)
    print("TEST: PPB NN2 MODEL ON JANUARY 2021 (WITH CLAMPING)")
    print("="*80)
    print()
    
    pinn, nn2, scalers, sensor_coords_spatial = load_models()
    
    # Load Jan 2021 sensor data
    print("\nLoading January 2021 sensor data...")
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 'timestamp' not in sensor_df.columns:
        if 't' in sensor_df.columns:
            sensor_df['timestamp'] = pd.to_datetime(sensor_df['t'])
        else:
            print("  ERROR: No timestamp column found")
            return
    else:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    print(f"  Loaded {len(sensor_df)} sensor readings for January 2021")
    
    # Load facility data for 2021
    print("\nLoading facility data...")
    if not FACILITY_DATA_DIR.exists():
        print(f"  ERROR: 2021 facility data not found at {FACILITY_DATA_DIR}")
        return
    
    facility_files = sorted(FACILITY_DATA_DIR.glob('*_training_data.csv'))
    print(f"  Found {len(facility_files)} facility files")
    
    facility_files_dict = {}
    for f in facility_files:
        df = pd.read_csv(f)
        if 'timestamp' not in df.columns:
            if 't' in df.columns:
                df['timestamp'] = pd.to_datetime(df['t'])
            else:
                continue
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['t'] = df['timestamp']  # For compatibility
        facility_name = f.stem.replace('_training_data', '')
        facility_files_dict[facility_name] = df
    
    print("\nProcessing predictions...")
    results = []
    
    for idx, row in tqdm(sensor_df.iterrows(), total=len(sensor_df), desc="Processing"):
        timestamp = row['timestamp']
        met_data_timestamp = timestamp - pd.Timedelta(hours=3)
        
        # PINN predictions
        pinn_values = predict_pinn_at_sensors(pinn, facility_files_dict, timestamp)
        
        # Get meteo data
        meteo_data_list = []
        for facility_name, facility_df in facility_files_dict.items():
            facility_data = facility_df[facility_df['t'] == met_data_timestamp]
            if len(facility_data) > 0:
                meteo_data_list.append(facility_data)
        
        if len(meteo_data_list) == 0:
            continue
        
        combined_meteo = pd.concat(meteo_data_list, ignore_index=True)
        
        # Get current sensor readings
        current_sensor_readings = {}
        for sensor_id in SENSORS.keys():
            actual_col = f'sensor_{sensor_id}'
            if actual_col in row and not pd.isna(row[actual_col]):
                current_sensor_readings[sensor_id] = row[actual_col]
            else:
                current_sensor_readings[sensor_id] = 0.0
        
        # Apply NN2 (with clamping)
        nn2_values = apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, combined_meteo, timestamp, current_sensor_readings)
        
        # Collect results
        for sensor_id in SENSORS.keys():
            # Try different column name formats
            actual_col = f'sensor_{sensor_id}'
            if actual_col not in sensor_df.columns:
                actual_col = sensor_id
            if actual_col not in sensor_df.columns:
                continue
            
            actual_val = row[actual_col]
            if pd.isna(actual_val):
                continue
            
            results.append({
                'timestamp': timestamp,
                'sensor_id': sensor_id,
                'actual': actual_val,
                'pinn': pinn_values[sensor_id],
                'nn2_clamped': nn2_values[sensor_id]
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\nERROR: No results generated!")
        return
    
    # Calculate metrics
    print("\n" + "="*80)
    print("RESULTS - JANUARY 2021")
    print("="*80)
    
    pinn_mae = np.abs(results_df['actual'] - results_df['pinn']).mean()
    nn2_mae = np.abs(results_df['actual'] - results_df['nn2_clamped']).mean()
    improvement = ((pinn_mae - nn2_mae) / pinn_mae * 100) if pinn_mae > 0 else 0
    
    print(f"\nPINN MAE:        {pinn_mae:.4f} ppb")
    print(f"Hybrid (NN2, clamped) MAE: {nn2_mae:.4f} ppb")
    print(f"Improvement:     {improvement:.1f}%")
    print(f"Total samples:   {len(results_df)}")
    
    # Per sensor breakdown
    print("\n" + "-"*80)
    print("PER SENSOR BREAKDOWN")
    print("-"*80)
    
    for sensor_id in sorted(SENSORS.keys()):
        sensor_data = results_df[results_df['sensor_id'] == sensor_id]
        if len(sensor_data) == 0:
            continue
        
        s_pinn_mae = np.abs(sensor_data['actual'] - sensor_data['pinn']).mean()
        s_nn2_mae = np.abs(sensor_data['actual'] - sensor_data['nn2_clamped']).mean()
        s_improvement = ((s_pinn_mae - s_nn2_mae) / s_pinn_mae * 100) if s_pinn_mae > 0 else 0
        
        print(f"\nSensor {sensor_id}:")
        print(f"  PINN MAE:        {s_pinn_mae:.4f} ppb")
        print(f"  Hybrid MAE:      {s_nn2_mae:.4f} ppb")
        print(f"  Improvement:     {s_improvement:.1f}%")
        print(f"  Samples:         {len(sensor_data)}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

