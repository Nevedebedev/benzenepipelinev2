#!/usr/bin/env python3
"""
Test Fixed NN2 Model (No Data Leakage) Against All of 2019

Tests the fixed NN2 model that:
1. Does NOT receive current_sensors as input (data leakage fix)
2. Outputs directly in ppb space
3. Uses simulation time t=3.0 hours (not absolute calendar time)
4. Computes PINN directly at sensor locations
5. Superimposes across all facilities
6. Applies NN2 correction
"""

import sys
from pathlib import Path
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))
sys.path.append(str(Path(__file__).parent / 'drive-download-20260202T042428Z-3-001'))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from pinn import ParametricADEPINN
import pickle

# Import the fixed NN2 model definition
from nn2_model_only import NN2_CorrectionNetwork, InverseTransformLayer

# Compatibility class for 45-feature models (with current_sensors)
class NN2_CorrectionNetwork_45(nn.Module):
    """NN2 model with 45 input features (includes current_sensors)"""
    def __init__(self, n_sensors=9, scaler_mean=None, scaler_scale=None, output_ppb=True):
        super().__init__()
        self.n_sensors = n_sensors
        self.output_ppb = output_ppb
        
        # 45 input features: pinn(9) + coords(18) + current_sensors(9) + wind(2) + diffusion(1) + temporal(6)
        self.correction_network = nn.Sequential(
            nn.Linear(45, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_sensors)
        )
        
        if output_ppb and scaler_mean is not None and scaler_scale is not None:
            self.inverse_transform = InverseTransformLayer(scaler_mean, scaler_scale)
        else:
            self.inverse_transform = None
    
    def forward(self, pinn_predictions, sensor_coords, current_sensors, wind, diffusion, temporal):
        batch_size = pinn_predictions.shape[0]
        coords_flat = sensor_coords.reshape(batch_size, -1)
        features = torch.cat([
            pinn_predictions,      # [batch, 9]
            coords_flat,           # [batch, 18]
            current_sensors,       # [batch, 9] - required for 45-feature model
            wind,                  # [batch, 2]
            diffusion,             # [batch, 1]
            temporal               # [batch, 6]
        ], dim=-1)  # Total: 45 features
        
        corrections = self.correction_network(features)
        corrected_scaled = pinn_predictions + corrections
        
        if self.inverse_transform is not None:
            corrected_ppb = self.inverse_transform(corrected_scaled)
            return corrected_ppb, corrections
        else:
            return corrected_scaled, corrections

# Paths
SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
NN2_MODEL_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_updatedlossfunction/nn2_master_model_ppb.pth"
NN2_SCALERS_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_updatedlossfunction/nn2_master_scalers-2.pkl"

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
    for attr, val in [
        ('x_min', 0.0), ('x_max', 30000.0), ('y_min', 0.0), ('y_max', 30000.0),
        ('t_min', 0.0), ('t_max', 8760.0), ('cx_min', 0.0), ('cx_max', 30000.0),
        ('cy_min', 0.0), ('cy_max', 30000.0), ('u_min', -15.0), ('u_max', 15.0),
        ('v_min', -15.0), ('v_max', 15.0), ('d_min', 0.0), ('d_max', 200.0),
        ('kappa_min', 0.0), ('kappa_max', 200.0), ('Q_min', 0.0), ('Q_max', 0.01)
    ]:
        setattr(pinn, attr, torch.tensor(val))
    
    pinn.eval()
    
    # Load NN2 checkpoint
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Get scaler parameters from checkpoint
    scaler_mean = nn2_checkpoint.get('scaler_mean', None)
    scaler_scale = nn2_checkpoint.get('scaler_scale', None)
    output_ppb = nn2_checkpoint.get('output_ppb', True)
    
    # Check model architecture (45 features = with current_sensors, 36 = without)
    state_dict = nn2_checkpoint['model_state_dict']
    first_layer_shape = state_dict['correction_network.0.weight'].shape
    input_features = first_layer_shape[1]
    
    if input_features == 45:
        # Model was trained with current_sensors (old architecture)
        # We'll need to provide zeros for current_sensors during inference
        print(f"  ⚠️  Model has {input_features} input features (includes current_sensors)")
        print(f"  ⚠️  Will use zeros for current_sensors during inference")
        nn2 = NN2_CorrectionNetwork_45(
            n_sensors=9,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            output_ppb=output_ppb
        )
    else:
        # Model was trained without current_sensors (new architecture)
        nn2 = NN2_CorrectionNetwork(
            n_sensors=9,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            output_ppb=output_ppb
        )
    
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    # Load scalers
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    # Sensor coords for NN2
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    print("  ✓ Models loaded")
    print(f"  ✓ NN2 outputs in ppb: {output_ppb}")
    return pinn, nn2, scalers, sensor_coords_spatial

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

def apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, meteo_data, timestamp):
    """
    Apply NN2 correction to PINN predictions.
    Handles both 36-feature (no current_sensors) and 45-feature (with current_sensors) models.
    Model outputs directly in ppb space.
    """
    # Prepare inputs
    sensor_ids_sorted = sorted(SENSORS.keys())
    pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
    
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
    
    # Scale PINN predictions (only non-zero values)
    p_s = np.zeros_like(pinn_array)
    if pinn_nonzero_mask.any():
        p_s[pinn_nonzero_mask] = scalers['pinn'].transform(
            pinn_array[pinn_nonzero_mask].reshape(-1, 1)
        ).flatten()
    p_s = p_s.reshape(1, -1)
    
    # Scale wind, diffusion, coords
    w_s = scalers['wind'].transform(np.array([[avg_u, avg_v]]))
    d_s = scalers['diffusion'].transform(np.array([[avg_D]]))
    c_s = scalers['coords'].transform(sensor_coords_spatial)
    
    # Convert to tensors
    p_tensor = torch.tensor(p_s, dtype=torch.float32)
    c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)
    w_tensor = torch.tensor(w_s, dtype=torch.float32)
    d_tensor = torch.tensor(d_s, dtype=torch.float32)
    t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)
    
    # Check if model needs current_sensors (45-feature model)
    # We'll use zeros for current_sensors since we don't have actual values
    if isinstance(nn2, NN2_CorrectionNetwork_45):
        # Model expects current_sensors - provide zeros (scaled)
        current_sensors_scaled = np.zeros((1, 9))  # All zeros
        cs_tensor = torch.tensor(current_sensors_scaled, dtype=torch.float32)
        with torch.no_grad():
            corrected_ppb, _ = nn2(p_tensor, c_tensor, cs_tensor, w_tensor, d_tensor, t_tensor)
    else:
        # 36-feature model (no current_sensors)
        with torch.no_grad():
            corrected_ppb, _ = nn2(p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
    
    # Convert to numpy (already in ppb)
    corrected_ppb_np = corrected_ppb.cpu().numpy().flatten()
    
    # Return as dict
    nn2_values = {sid: corrected_ppb_np[i] for i, sid in enumerate(sensor_ids_sorted)}
    return nn2_values

def main():
    print("="*80)
    print("TESTING FIXED NN2 MODEL ON 2019 DATA")
    print("="*80)
    print(f"\nModel: {NN2_MODEL_PATH}")
    print(f"Scalers: {NN2_SCALERS_PATH}\n")
    
    # Load models
    pinn, nn2, scalers, sensor_coords_spatial = load_models()
    
    # Load sensor data
    print("\nLoading sensor data...")
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['t'])
    elif 'timestamp' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    # Load facility files
    print("Loading facility data...")
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    facility_files_dict = {}
    for f in facility_files:
        df = pd.read_csv(f)
        df['t'] = pd.to_datetime(df['t'])
        facility_name = f.stem.replace('_synced_training_data', '')
        facility_files_dict[facility_name] = df
    
    print(f"  ✓ Loaded {len(facility_files_dict)} facilities")
    
    # Process all timestamps
    print("\nProcessing predictions...")
    all_results = []
    
    for idx, row in tqdm(sensor_df.iterrows(), total=len(sensor_df), desc="Processing"):
        timestamp = row['timestamp']
        
        # Get PINN predictions
        pinn_values = predict_pinn_at_sensors(pinn, facility_files_dict, timestamp)
        
        # Get meteo data
        met_data_timestamp = timestamp - pd.Timedelta(hours=3)
        meteo_data_list = []
        for facility_name, facility_df in facility_files_dict.items():
            facility_data = facility_df[facility_df['t'] == met_data_timestamp]
            if len(facility_data) > 0:
                meteo_data_list.append(facility_data)
        
        if len(meteo_data_list) == 0:
            continue
        
        combined_meteo = pd.concat(meteo_data_list, ignore_index=True)
        
        # Apply NN2 correction (NO current_sensors!)
        nn2_values = apply_nn2_correction(
            nn2, scalers, sensor_coords_spatial, 
            pinn_values, combined_meteo, timestamp
        )
        
        # Get actual sensor readings
        for sensor_id in SENSORS.keys():
            actual_col = f'sensor_{sensor_id}'
            if actual_col in row and not pd.isna(row[actual_col]):
                actual_value = row[actual_col]
                
                all_results.append({
                    'timestamp': timestamp,
                    'sensor_id': sensor_id,
                    'pinn_ppb': pinn_values[sensor_id],
                    'nn2_ppb': nn2_values[sensor_id],
                    'actual_ppb': actual_value
                })
    
    # Calculate metrics
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Overall metrics
    pinn_mae = np.abs(results_df['pinn_ppb'] - results_df['actual_ppb']).mean()
    nn2_mae = np.abs(results_df['nn2_ppb'] - results_df['actual_ppb']).mean()
    improvement = ((pinn_mae - nn2_mae) / pinn_mae * 100) if pinn_mae > 0 else 0
    
    print(f"\nOverall Performance:")
    print(f"  PINN MAE: {pinn_mae:.4f} ppb")
    print(f"  NN2 MAE:  {nn2_mae:.4f} ppb")
    print(f"  Improvement: {improvement:.1f}%")
    
    # Per-sensor metrics
    print(f"\nPer-Sensor Performance:")
    for sensor_id in sorted(SENSORS.keys()):
        sensor_data = results_df[results_df['sensor_id'] == sensor_id]
        if len(sensor_data) > 0:
            sensor_pinn_mae = np.abs(sensor_data['pinn_ppb'] - sensor_data['actual_ppb']).mean()
            sensor_nn2_mae = np.abs(sensor_data['nn2_ppb'] - sensor_data['actual_ppb']).mean()
            sensor_improvement = ((sensor_pinn_mae - sensor_nn2_mae) / sensor_pinn_mae * 100) if sensor_pinn_mae > 0 else 0
            print(f"  {sensor_id}: PINN={sensor_pinn_mae:.4f}, NN2={sensor_nn2_mae:.4f}, Improvement={sensor_improvement:.1f}%")
    
    # Save results
    output_file = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/validation_results/test_2019_fixed_nn2.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()

