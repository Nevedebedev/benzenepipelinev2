#!/usr/bin/env python3
"""
Test New PPB-Scale NN2 Model Against All of 2019

Tests the new NN2 model that outputs directly in ppb space:
1. Uses simulation time t=3.0 hours (not absolute calendar time)
2. Computes PINN directly at sensor locations
3. Superimposes across all facilities
4. Applies NN2 correction (outputs directly in ppb - no inverse transform needed!)
5. Compares with actual sensor readings
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
from nn2 import NN2_CorrectionNetwork
import pickle
from nn2_mapping_utils import load_nn2_mapping_model, nn2_scaled_to_ppb

# Paths
SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
NN2_MODEL_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_timefix/nn2_master_model_spatial-3.pth"
NN2_SCALERS_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_timefix/nn2_master_scalers-2.pkl"

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
    """Load PINN and NN2 models"""
    print("Loading models...")
    
    # Load PINN
    pinn = ParametricADEPINN()
    checkpoint = torch.load(PINN_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    
    # Override normalization ranges (matches training data generation)
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
    
    # Load NN2 (outputs in scaled space)
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    # Load scalers
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    # Sensor coords for NN2
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    # Load NN2 output mapping model (for converting scaled outputs to ppb)
    nn2_mapping_model = None
    try:
        print("  Loading NN2 output mapping model...")
        mapping_data = load_nn2_mapping_model()
        nn2_mapping_model = mapping_data['model']
        print(f"  ✓ Mapping model loaded (type: {mapping_data['type']})")
    except FileNotFoundError as e:
        print(f"  ⚠ Warning: {e}. Will use scaler inverse transform.")
    
    print("  ✓ Models loaded")
    return pinn, nn2, scalers, sensor_coords_spatial, nn2_mapping_model

def predict_pinn_at_sensors(pinn, facility_files_dict, timestamp):
    """
    Predict PINN at sensor locations using EXACT same method as training data generation.
    Uses simulation time t=3.0 hours, not absolute calendar time.
    """
    # Get met data from 3 hours before (for 3-hour forecast)
    input_timestamp = timestamp - pd.Timedelta(hours=3)
    
    sensor_pinn_ppb = {sid: 0.0 for sid in SENSORS.keys()}
    
    # For each facility, compute PINN at all sensors and superimpose
    for facility_name, facility_df in facility_files_dict.items():
        facility_data = facility_df[facility_df['t'] == input_timestamp]
        
        if len(facility_data) == 0:
            continue
        
        # For this facility, compute PINN at all sensors and superimpose
        for _, row in facility_data.iterrows():
            cx = row['source_x_cartesian']
            cy = row['source_y_cartesian']
            d = row['source_diameter']
            Q = row['Q_total']
            u = row['wind_u']
            v = row['wind_v']
            kappa = row['D']
            
            # Solve PINN at each sensor location for this facility
            for sensor_id, (sx, sy) in SENSORS.items():
                with torch.no_grad():
                    phi_raw = pinn(
                        torch.tensor([[sx]], dtype=torch.float32),
                        torch.tensor([[sy]], dtype=torch.float32),
                        torch.tensor([[FORECAST_T_HOURS]], dtype=torch.float32),  # Simulation time
                        torch.tensor([[cx]], dtype=torch.float32),
                        torch.tensor([[cy]], dtype=torch.float32),
                        torch.tensor([[u]], dtype=torch.float32),
                        torch.tensor([[v]], dtype=torch.float32),
                        torch.tensor([[d]], dtype=torch.float32),
                        torch.tensor([[kappa]], dtype=torch.float32),
                        torch.tensor([[Q]], dtype=torch.float32),
                        normalize=True
                    )
                    
                    # Convert to ppb and superimpose
                    concentration_ppb = phi_raw.item() * UNIT_CONVERSION
                    sensor_pinn_ppb[sensor_id] += concentration_ppb
    
    return sensor_pinn_ppb

def apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, meteo_data, timestamp, current_sensor_readings, nn2_mapping_model=None):
    """
    Apply NN2 correction to PINN predictions.
    Model outputs in scaled space, then converted to ppb using mapping model or scaler.
    """
    # Prepare inputs
    sensor_ids_sorted = sorted(SENSORS.keys())
    pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
    current_sensors = np.array([current_sensor_readings.get(sid, 0.0) for sid in sensor_ids_sorted])
    
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
    
    # Scale inputs - handle zeros the same way as training
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
    
    # Run NN2 - outputs in scaled space
    with torch.no_grad():
        corrected_scaled, _ = nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
    
    # Convert from scaled space to ppb
    corrected_scaled_np = corrected_scaled.cpu().numpy().flatten()
    
    if nn2_mapping_model is not None:
        # Use mapping model (more accurate)
        nn2_corrected_ppb = nn2_scaled_to_ppb(corrected_scaled_np, nn2_mapping_model)
    else:
        # Fallback to scaler inverse transform
        nn2_corrected_ppb = np.zeros_like(corrected_scaled_np)
        nn2_output_nonzero_mask = np.abs(corrected_scaled_np) > 1e-6
        if nn2_output_nonzero_mask.any():
            nn2_corrected_ppb[nn2_output_nonzero_mask] = scalers['sensors'].inverse_transform(
                corrected_scaled_np[nn2_output_nonzero_mask].reshape(-1, 1)
            ).flatten()
    
    # Return as dict
    nn2_values = {sid: nn2_corrected_ppb[i] for i, sid in enumerate(sensor_ids_sorted)}
    return nn2_values

def main():
    print("="*80)
    print("TESTING NN2 MODEL - FULL 2019 DATA")
    print("="*80)
    print()
    
    # Load models
    pinn, nn2, scalers, sensor_coords_spatial, nn2_mapping_model = load_models()
    
    # Load sensor data (ground truth)
    print("\nLoading sensor data...")
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['t'])
    elif 'timestamp' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    print(f"  Loaded {len(sensor_df)} sensor readings")
    
    # Load facility data
    print("\nLoading facility data...")
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    print(f"  Found {len(facility_files)} facility files")
    
    facility_files_dict = {}
    for f in facility_files:
        df = pd.read_csv(f)
        df['t'] = pd.to_datetime(df['t'])
        facility_name = f.stem.replace('_synced_training_data', '')
        facility_files_dict[facility_name] = df
    
    # Process predictions
    print("\nProcessing predictions...")
    print("  Using EXACT same method as training data generation:")
    print("    - Simulation time t=3.0 hours (not absolute calendar time)")
    print("    - Direct PINN computation at sensor locations")
    print("    - Superimpose across all facilities")
    print("    - Apply NN2 correction (outputs in scaled space, converted to ppb)")
    print()
    
    results = []
    
    for idx, row in tqdm(sensor_df.iterrows(), total=len(sensor_df), desc="Processing"):
        timestamp = row['timestamp']
        
        # Get met data from 3 hours before (for 3-hour forecast)
        met_data_timestamp = timestamp - pd.Timedelta(hours=3)
        
        # Step 1-3: PINN at sources -> superimpose -> convert to ppb
        pinn_values = predict_pinn_at_sensors(pinn, facility_files_dict, timestamp)
        
        # Get meteo data for NN2
        meteo_data_list = []
        for facility_name, facility_df in facility_files_dict.items():
            facility_data = facility_df[facility_df['t'] == met_data_timestamp]
            if len(facility_data) > 0:
                meteo_data_list.append(facility_data)
        
        if len(meteo_data_list) == 0:
            continue
        
        combined_meteo = pd.concat(meteo_data_list, ignore_index=True)
        
        # Get actual sensor readings from time t+3 (forecast timestamp) for NN2's "current_sensors" input
        current_sensor_readings = {}
        for sensor_id in SENSORS.keys():
            actual_col = f'sensor_{sensor_id}'
            if actual_col in row and not pd.isna(row[actual_col]):
                current_sensor_readings[sensor_id] = row[actual_col]
            else:
                current_sensor_readings[sensor_id] = 0.0
        
        # Step 4: Apply NN2 correction (outputs in scaled space, converted to ppb)
        nn2_values = apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, combined_meteo, timestamp, current_sensor_readings, nn2_mapping_model)
        
        # Collect results
        for sensor_id in SENSORS.keys():
            actual_col = f'sensor_{sensor_id}'
            if actual_col not in row or pd.isna(row[actual_col]):
                continue
            
            results.append({
                'timestamp': timestamp,
                'sensor_id': sensor_id,
                'actual': row[actual_col],
                'pinn': pinn_values[sensor_id],
                'nn2': nn2_values[sensor_id]
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\nERROR: No results generated!")
        return
    
    # Calculate metrics
    print("\n" + "="*80)
    print("RESULTS - PER SENSOR")
    print("="*80)
    
    sensor_metrics = {}
    for sensor_id in SENSORS.keys():
        sensor_data = results_df[results_df['sensor_id'] == sensor_id]
        if len(sensor_data) == 0:
            continue
        
        pinn_mae = np.abs(sensor_data['actual'] - sensor_data['pinn']).mean()
        nn2_mae = np.abs(sensor_data['actual'] - sensor_data['nn2']).mean()
        improvement = ((pinn_mae - nn2_mae) / pinn_mae * 100) if pinn_mae > 0 else 0
        
        sensor_metrics[sensor_id] = {
            'pinn_mae': pinn_mae,
            'nn2_mae': nn2_mae,
            'improvement': improvement,
            'samples': len(sensor_data)
        }
        
        print(f"\nSensor {sensor_id}:")
        print(f"  PINN MAE:   {pinn_mae:.4f} ppb")
        print(f"  NN2 MAE:    {nn2_mae:.4f} ppb")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Samples: {len(sensor_data)}")
    
    # Overall metrics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    overall_pinn_mae = np.abs(results_df['actual'] - results_df['pinn']).mean()
    overall_nn2_mae = np.abs(results_df['actual'] - results_df['nn2']).mean()
    overall_improvement = ((overall_pinn_mae - overall_nn2_mae) / overall_pinn_mae * 100) if overall_pinn_mae > 0 else 0
    
    print(f"\nPINN MAE:   {overall_pinn_mae:.4f} ppb")
    print(f"Hybrid (NN2) MAE:    {overall_nn2_mae:.4f} ppb")
    print(f"Improvement: {overall_improvement:.1f}%")
    print(f"Total samples: {len(results_df)}")
    
    # Monthly breakdown
    print("\n" + "="*80)
    print("MONTHLY BREAKDOWN")
    print("="*80)
    
    results_df['month'] = pd.to_datetime(results_df['timestamp']).dt.month
    monthly_metrics = {}
    
    for month in range(1, 13):
        month_data = results_df[results_df['month'] == month]
        if len(month_data) == 0:
            continue
        
        month_pinn_mae = np.abs(month_data['actual'] - month_data['pinn']).mean()
        month_nn2_mae = np.abs(month_data['actual'] - month_data['nn2']).mean()
        month_improvement = ((month_pinn_mae - month_nn2_mae) / month_pinn_mae * 100) if month_pinn_mae > 0 else 0
        
        monthly_metrics[month] = {
            'pinn_mae': month_pinn_mae,
            'nn2_mae': month_nn2_mae,
            'improvement': month_improvement,
            'samples': len(month_data)
        }
        
        month_name = pd.to_datetime(f'2019-{month}-01').strftime('%B')
        print(f"\n{month_name}:")
        print(f"  PINN MAE:   {month_pinn_mae:.4f} ppb")
        print(f"  Hybrid (NN2) MAE:    {month_nn2_mae:.4f} ppb")
        print(f"  Improvement: {month_improvement:.1f}%")
        print(f"  Samples: {len(month_data)}")
    
    # Save results
    output_dir = Path(__file__).parent / 'validation_results'
    output_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(output_dir / 'test_2019_ppb_model_detailed.csv', index=False)
    print(f"\n  Saved detailed results to {output_dir / 'test_2019_ppb_model_detailed.csv'}")
    
    # Save summary
    summary = {
        'overall': {
            'pinn_mae': overall_pinn_mae,
            'nn2_mae': overall_nn2_mae,
            'improvement': overall_improvement,
            'total_samples': len(results_df)
        },
        'per_sensor': sensor_metrics,
        'monthly': monthly_metrics
    }
    
    import json
    with open(output_dir / 'test_2019_ppb_model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to {output_dir / 'test_2019_ppb_model_summary.json'}")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nSummary:")
    print(f"  PINN MAE: {overall_pinn_mae:.4f} ppb")
    print(f"  Hybrid (NN2) MAE: {overall_nn2_mae:.4f} ppb")
    print(f"  Improvement: {overall_improvement:.1f}%")
    print(f"  Total samples: {len(results_df)}")

if __name__ == '__main__':
    main()

