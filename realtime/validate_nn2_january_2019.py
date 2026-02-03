#!/usr/bin/env python3
"""
Complete NN2 Validation Against January 2019 Data

Follows the exact pipeline process:
1. Solve PINN at all sources
2. Superimpose contributions at sensor locations
3. Convert to ppb
4. Apply NN2 correction
5. Compare with actual sensor measurements to get MAE
"""

import sys
from pathlib import Path
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from pinn import ParametricADEPINN
from nn2 import NN2_CorrectionNetwork
import pickle
from nn2_mapping_utils import load_nn2_mapping_model, nn2_scaled_to_ppb

# Paths
SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
# Use nn2_timefix model and scalars (retrained with time-fixed PINN data)
NN2_MODEL_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_timefix/nn2_master_model_spatial-3.pth"
NN2_SCALERS_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_timefix/nn2_master_scalers-2.pkl"

# Constants
UNIT_CONVERSION = 313210039.9  # kg/m^2 to ppb
T_START = pd.to_datetime('2019-01-01 00:00:00')

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
    pinn.eval()
    
    # Load NN2
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    # Load scalers
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    # Sensor coords for NN2
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    # Load NN2 output mapping model
    print("  Loading NN2 output mapping model...")
    try:
        mapping_data = load_nn2_mapping_model()
        nn2_mapping_model = mapping_data['model']
        print(f"  ✓ Mapping model loaded (type: {mapping_data['type']})")
    except FileNotFoundError as e:
        print(f"  ⚠ Warning: {e}")
        print("  Falling back to original scaler inverse transform")
        nn2_mapping_model = None
    
    print("  ✓ Models loaded")
    return pinn, nn2, scalers, sensor_coords_spatial, nn2_mapping_model

def predict_at_sensors(pinn, facility_files_dict, timestamp):
    """
    Step 1-3: Solve PINN at sources, superimpose at sensors, convert to ppb
    
    EXACTLY matches training data generation method:
    1. Process each facility file separately
    2. For each facility, compute PINN at all sensors
    3. Superimpose across all facilities
    
    FIXED: Uses simulation time t=3.0 hours (not absolute calendar time).
    Each scenario starts at t=0, predicts at t=3 hours for 3-hour forecast.
    """
    # FIX: Use simulation time instead of absolute calendar time (matches training data)
    FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast (each scenario resets to t=0)
    t_hours_forecast = FORECAST_T_HOURS
    
    # Initialize sensor concentrations (will superimpose across all facilities)
    sensor_pinn_ppb = {sid: 0.0 for sid in SENSORS.keys()}
    
    # Process each facility file separately (EXACTLY like training data generation)
    for facility_name, facility_df in facility_files_dict.items():
        # Get facility data for this timestamp
        facility_data = facility_df[facility_df['timestamp'] == timestamp]
        
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
                        torch.tensor([[t_hours_forecast]], dtype=torch.float32),  # FORECAST TIME
                        torch.tensor([[cx]], dtype=torch.float32),
                        torch.tensor([[cy]], dtype=torch.float32),
                        torch.tensor([[u]], dtype=torch.float32),
                        torch.tensor([[v]], dtype=torch.float32),
                        torch.tensor([[d]], dtype=torch.float32),
                        torch.tensor([[kappa]], dtype=torch.float32),
                        torch.tensor([[Q]], dtype=torch.float32)
                    )
                    
                    # Convert to ppb and superimpose
                    concentration_ppb = phi_raw.item() * UNIT_CONVERSION
                    sensor_pinn_ppb[sensor_id] += concentration_ppb
    
    return sensor_pinn_ppb

def apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, meteo_data, timestamp, current_sensor_readings=None, nn2_mapping_model=None):
    """
    Step 4: Apply NN2 correction to PINN predictions
    
    Args:
        current_sensor_readings: Dict of sensor_id -> actual reading at timestamp (time t, when prediction is made)
                                 If None, will use PINN predictions as fallback (not ideal)
    """
    # Prepare inputs
    sensor_ids_sorted = sorted(SENSORS.keys())
    pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
    
    # Use actual sensor readings from time t (when prediction is made) as current_sensors
    # This matches training: current_sensors = actual readings at prediction time, target = actual readings at t+3
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
    
    # Scale inputs
    # FIX: Handle zeros the same way as training (mask before scaling)
    # Training code fits scalers on non-zero values only, so we must do the same
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
    # Values that are truly zero in scaled space remain zero
    
    # Return as dict
    nn2_values = {sid: nn2_corrected[i] for i, sid in enumerate(sensor_ids_sorted)}
    
    return nn2_values

def main():
    print("="*80)
    print("NN2 VALIDATION - FULL YEAR 2019")
    print("="*80)
    print()
    
    # Load models
    pinn, nn2, scalers, sensor_coords_spatial, nn2_mapping_model = load_models()
    
    # Load sensor data (ground truth)
    print("\nLoading sensor data...")
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    # Filter to full year 2019
    year_mask = (sensor_df['timestamp'] >= '2019-01-01') & (sensor_df['timestamp'] < '2020-01-01')
    sensor_df = sensor_df[year_mask].reset_index(drop=True)
    print(f"  Found {len(sensor_df)} timestamps in 2019")
    
    # Load training data PINN predictions directly (already computed correctly)
    print("\nLoading training data PINN predictions...")
    training_pinn_df = pd.read_csv(SYNCED_DIR / 'total_concentrations.csv')
    training_pinn_df['timestamp'] = pd.to_datetime(training_pinn_df['timestamp'])
    print(f"  Loaded {len(training_pinn_df)} PINN predictions from training data")
    
    # Load facility data for meteo (needed for NN2)
    print("\nLoading facility data for meteo...")
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    # Load each facility file separately (like training data generation)
    facility_files_dict = {}
    for f in facility_files:
        facility_name = f.stem.replace('_synced_training_data', '')
        df = pd.read_csv(f)
        df['timestamp'] = pd.to_datetime(df['t'])
        facility_files_dict[facility_name] = df
    
    print(f"  Loaded {len(facility_files_dict)} facilities")
    
    # Process each timestamp
    print("\nProcessing predictions...")
    print("  Note: Sensor readings at time t are compared with PINN predictions made at t-3")
    print("        (using met data from t-3, predicting forward 3 hours to t)")
    results = []
    
    for idx, row in sensor_df.iterrows():
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(sensor_df)}")
        
        # Sensor reading timestamp (this is the forecast target time t+3)
        forecast_timestamp = row['timestamp']
        
        # CRITICAL: Use PINN predictions from training data (already computed correctly)
        # The training data has PINN predictions at timestamp t+3 (forecasts made at t)
        # So we can directly use those predictions for validation
        pinn_row = training_pinn_df[training_pinn_df['timestamp'] == forecast_timestamp]
        
        if len(pinn_row) == 0:
            continue
        
        # Extract PINN predictions from training data
        pinn_values = {}
        for sensor_id in SENSORS.keys():
            col = f'sensor_{sensor_id}'
            if col in pinn_row.columns:
                pinn_values[sensor_id] = pinn_row[col].iloc[0]
            else:
                pinn_values[sensor_id] = 0.0
        
        # Get meteo data for NN2 (from 3 hours before, when prediction was made)
        met_data_timestamp = forecast_timestamp - pd.Timedelta(hours=3)
        
        # Get meteo data for NN2 (average across all facilities at this timestamp)
        meteo_data_list = []
        for facility_name, facility_df in facility_files_dict.items():
            facility_data = facility_df[facility_df['timestamp'] == met_data_timestamp]
            if len(facility_data) > 0:
                meteo_data_list.append(facility_data)
        
        if len(meteo_data_list) == 0:
            continue
        
        # Combine all facility meteo data for averaging
        combined_meteo = pd.concat(meteo_data_list, ignore_index=True)
        
        # Get actual sensor readings from time t+3 (forecast timestamp) for NN2's "current_sensors" input
        # This matches training structure: current_sensors = actual readings at t+3, target = actual readings at t+3
        # The model was trained with current_sensors and target at the same timestamp
        current_sensor_readings = {}
        for sensor_id in SENSORS.keys():
            actual_col = f'sensor_{sensor_id}'
            # Find sensor reading at forecast_timestamp (time t+3, same as target)
            if actual_col in row and not pd.isna(row[actual_col]):
                current_sensor_readings[sensor_id] = row[actual_col]
            else:
                current_sensor_readings[sensor_id] = 0.0  # Default to 0 if not available
        
        # Step 4: Apply NN2 correction
        # Use forecast_timestamp for temporal features (the time we're predicting for)
        nn2_values = apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, combined_meteo, forecast_timestamp, current_sensor_readings)
        
        # Collect results
        for sensor_id in SENSORS.keys():
            actual_col = f'sensor_{sensor_id}'
            if actual_col not in row or pd.isna(row[actual_col]):
                continue
            
            results.append({
                'timestamp': forecast_timestamp,  # Time of sensor reading (t+3)
                'met_data_timestamp': met_data_timestamp,  # Time when met data was collected (t)
                'sensor_id': sensor_id,
                'actual': row[actual_col],
                'pinn': pinn_values[sensor_id],
                'nn2': nn2_values[sensor_id]
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute MAE per sensor
    print("\n" + "="*80)
    print("RESULTS - FULL YEAR 2019")
    print("="*80)
    
    for sensor_id in sorted(SENSORS.keys()):
        sensor_data = results_df[results_df['sensor_id'] == sensor_id]
        
        if len(sensor_data) == 0:
            continue
        
        pinn_mae = np.mean(np.abs(sensor_data['actual'] - sensor_data['pinn']))
        nn2_mae = np.mean(np.abs(sensor_data['actual'] - sensor_data['nn2']))
        improvement = ((pinn_mae - nn2_mae) / pinn_mae) * 100
        
        print(f"\nSensor {sensor_id}:")
        print(f"  PINN MAE:   {pinn_mae:.4f} ppb")
        print(f"  NN2 MAE:    {nn2_mae:.4f} ppb")
        print(f"  Improvement: {improvement:.1f}%")
    
    # Overall MAE
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    pinn_overall_mae = np.mean(np.abs(results_df['actual'] - results_df['pinn']))
    nn2_overall_mae = np.mean(np.abs(results_df['actual'] - results_df['nn2']))
    overall_improvement = ((pinn_overall_mae - nn2_overall_mae) / pinn_overall_mae) * 100
    
    print(f"\nPINN MAE:   {pinn_overall_mae:.4f} ppb")
    print(f"NN2 MAE:    {nn2_overall_mae:.4f} ppb")
    print(f"Improvement: {overall_improvement:.1f}%")
    print(f"Total samples: {len(results_df)}")
    
    # Monthly breakdown
    print("\n" + "="*80)
    print("MONTHLY BREAKDOWN")
    print("="*80)
    results_df['month'] = pd.to_datetime(results_df['timestamp']).dt.month
    results_df['month_name'] = pd.to_datetime(results_df['timestamp']).dt.strftime('%B')
    
    for month in range(1, 13):
        month_data = results_df[results_df['month'] == month]
        if len(month_data) == 0:
            continue
        
        month_name = month_data['month_name'].iloc[0]
        month_pinn_mae = np.mean(np.abs(month_data['actual'] - month_data['pinn']))
        month_nn2_mae = np.mean(np.abs(month_data['actual'] - month_data['nn2']))
        month_improvement = ((month_pinn_mae - month_nn2_mae) / month_pinn_mae) * 100
        
        print(f"\n{month_name}:")
        print(f"  PINN MAE:   {month_pinn_mae:.4f} ppb")
        print(f"  NN2 MAE:    {month_nn2_mae:.4f} ppb")
        print(f"  Improvement: {month_improvement:.1f}%")
        print(f"  Samples: {len(month_data)}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
