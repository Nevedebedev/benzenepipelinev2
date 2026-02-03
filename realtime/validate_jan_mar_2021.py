#!/usr/bin/env python3
"""
Validate Pipeline on January-March 2021

Runs the benzene dispersion pipeline on January-March 2021 weather data
and validates against EDF sensor readings, reporting MAE for both PINN-only
and Hybrid (PINN+NN2) predictions.

Uses the exact same method as training data generation:
1. Uses simulation time t=3.0 hours (not absolute calendar time)
2. Computes PINN directly at sensor locations
3. Superimposes across all facilities
4. Applies NN2 correction
"""

import sys
from pathlib import Path
# Add paths for PINN and NN2 imports
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from pinn import ParametricADEPINN
import pickle
import torch.nn as nn

# Import NN2 model from the fixed version
sys.path.append(str(Path(__file__).parent / 'drive-download-20260202T042428Z-3-001'))
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
PROJECT_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime')
FACILITY_DATA_BASE = PROJECT_DIR / 'realtime_processing/houston_processed_2021'
MADIS_DIR = Path('/Users/neevpratap/Desktop/madis_data_desktop_updated')

PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
# Updated to use Phase 1.5 balanced model
NN2_MODEL_PATH = PROJECT_DIR / "nn2_balanced_model/nn2_master_model_ppb_balanced.pth"
NN2_SCALERS_PATH = PROJECT_DIR / "nn2_balanced_model/nn2_master_scalers_balanced.pkl"

# Ground truth sensor data paths
SENSOR_DATA_PATHS = {
    'jan': MADIS_DIR / "results_2021/sensors_actual_wide_2021_full_jan.csv",
    'feb': MADIS_DIR / "results_2021/sensors_actual_wide_2021_full_feb.csv",
    'mar': MADIS_DIR / "results_2021/sensors_actual_wide_2021_full_march.csv",
}

# Facility data directories
FACILITY_DATA_DIRS = {
    'jan': FACILITY_DATA_BASE / 'training_data_2021_january_complete',
    'feb': FACILITY_DATA_BASE / 'training_data_2021_feb',
    'mar': FACILITY_DATA_BASE / 'training_data_2021_march',
}

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
    
    # Load NN2 checkpoint
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Get scaler parameters from checkpoint
    scaler_mean = nn2_checkpoint.get('scaler_mean', None)
    scaler_scale = nn2_checkpoint.get('scaler_scale', None)
    output_ppb = nn2_checkpoint.get('output_ppb', True)
    
    # Check model architecture from checkpoint
    architecture = nn2_checkpoint.get('architecture', None)
    state_dict = nn2_checkpoint['model_state_dict']
    first_layer_shape = state_dict['correction_network.0.weight'].shape
    input_features = first_layer_shape[1]
    first_layer_output = first_layer_shape[0]
    
    print(f"  ✓ NN2 model has {input_features} input features")
    
    if architecture == 'balanced':
        print(f"  ✓ Detected Phase 1.5 balanced model (36→64→16→9)")
        # Create balanced model class inline
        class BalancedNN2_CorrectionNetwork(nn.Module):
            """Balanced NN2 model: 36 → 64 → 16 → 9"""
            def __init__(self, n_sensors=9, scaler_mean=None, scaler_scale=None, output_ppb=True):
                super().__init__()
                self.n_sensors = n_sensors
                self.output_ppb = output_ppb
                self.correction_network = nn.Sequential(
                    nn.Linear(36, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 16),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(16, n_sensors)
                )
                if output_ppb and scaler_mean is not None and scaler_scale is not None:
                    self.inverse_transform = InverseTransformLayer(scaler_mean, scaler_scale)
                else:
                    self.inverse_transform = None
            
            def forward(self, pinn_predictions, sensor_coords, wind, diffusion, temporal):
                batch_size = pinn_predictions.shape[0]
                coords_flat = sensor_coords.reshape(batch_size, -1)
                features = torch.cat([
                    pinn_predictions, coords_flat, wind, diffusion, temporal
                ], dim=-1)
                corrections = self.correction_network(features)
                corrected_scaled = pinn_predictions + corrections
                if self.inverse_transform is not None:
                    corrected_ppb = self.inverse_transform(corrected_scaled)
                    return corrected_ppb, corrections
                else:
                    return corrected_scaled, corrections
        
        nn2 = BalancedNN2_CorrectionNetwork(
            n_sensors=9,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            output_ppb=output_ppb
        )
    elif input_features == 45:
        # Model was trained with current_sensors (old architecture)
        print(f"  ⚠️  Model has {input_features} input features (includes current_sensors)")
        print(f"  ⚠️  Will use zeros for current_sensors during inference")
        nn2 = NN2_CorrectionNetwork_45(
            n_sensors=9,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            output_ppb=output_ppb
        )
    else:
        # Model was trained without current_sensors (new architecture - standard)
        print(f"  ✓ Model uses standard architecture (36 features, no current_sensors)")
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

def load_facility_data(month_key):
    """Load facility weather data for a given month"""
    facility_dir = FACILITY_DATA_DIRS[month_key]
    
    if not facility_dir.exists():
        print(f"  Warning: {facility_dir} does not exist")
        return {}
    
    facility_files = sorted(facility_dir.glob('*_training_data.csv'))
    facility_files_dict = {}
    
    for f in facility_files:
        facility_name = f.stem.replace('_training_data', '')
        df = pd.read_csv(f)
        
        # Handle timestamp column
        if 't' in df.columns:
            df['timestamp'] = pd.to_datetime(df['t'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            print(f"  Warning: No timestamp column in {f.name}")
            continue
        
        facility_files_dict[facility_name] = df
    
    print(f"  Loaded {len(facility_files_dict)} facilities from {month_key}")
    return facility_files_dict

def predict_pinn_at_sensors(pinn, facility_files_dict, timestamp):
    """
    Predict PINN concentrations at all sensor locations
    EXACTLY matches training data generation method:
    1. Process each facility separately
    2. Use simulation time t=3.0 hours
    3. Superimpose across all facilities
    """
    # Initialize sensor concentrations
    sensor_pinn_ppb = {sid: 0.0 for sid in SENSORS.keys()}
    
    # Process each facility separately (EXACTLY like training data generation)
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

def apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, meteo_data, timestamp, current_sensor_readings=None):
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
    if isinstance(nn2, NN2_CorrectionNetwork_45):
        # Model expects current_sensors - provide zeros (scaled) since we don't have actual values
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
    print("PIPELINE VALIDATION - JANUARY-MARCH 2021")
    print("="*80)
    print()
    
    # Load models
    pinn, nn2, scalers, sensor_coords_spatial = load_models()
    
    # Load sensor data (ground truth) for all months
    print("\nLoading EDF sensor data...")
    all_sensor_data = []
    
    for month_key, sensor_path in SENSOR_DATA_PATHS.items():
        if not sensor_path.exists():
            print(f"  Warning: {sensor_path} does not exist")
            continue
        
        sensor_df = pd.read_csv(sensor_path)
        
        # Handle timestamp column
        if 't' in sensor_df.columns:
            sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
        sensor_df['month_key'] = month_key
        
        all_sensor_data.append(sensor_df)
        print(f"  Loaded {len(sensor_df)} timestamps from {month_key}")
    
    if len(all_sensor_data) == 0:
        print("  ERROR: No sensor data loaded!")
        return
    
    sensor_df = pd.concat(all_sensor_data, ignore_index=True)
    print(f"  Total: {len(sensor_df)} timestamps")
    
    # Load facility data for all months
    print("\nLoading facility weather data...")
    all_facility_data = {}
    
    for month_key in ['jan', 'feb', 'mar']:
        facility_data = load_facility_data(month_key)
        # Store with month prefix to avoid conflicts
        for facility_name, df in facility_data.items():
            all_facility_data[f"{month_key}_{facility_name}"] = df
    
    if len(all_facility_data) == 0:
        print("  ERROR: No facility data loaded!")
        return
    
    print(f"  Total: {len(all_facility_data)} facility-month combinations")
    
    # Process each timestamp
    print("\nProcessing predictions...")
    print("  Using EXACT same method as training data generation:")
    print("    - Simulation time t=3.0 hours (not absolute calendar time)")
    print("    - Direct PINN computation at sensor locations")
    print("    - Superimpose across all facilities")
    print("    - Apply NN2 correction")
    print()
    
    results = []
    
    for idx, row in tqdm(sensor_df.iterrows(), total=len(sensor_df), desc="Processing"):
        # Sensor reading timestamp (this is the forecast target time t+3)
        forecast_timestamp = row['timestamp']
        month_key = row['month_key']
        
        # CRITICAL: Get facility data from 3 hours BEFORE the sensor reading
        # Predictions made at time t (using met data from t) are forecasts for t+3
        met_data_timestamp = forecast_timestamp - pd.Timedelta(hours=3)
        
        # Get facility data for this month
        month_facility_data = {
            name.replace(f"{month_key}_", ""): df 
            for name, df in all_facility_data.items() 
            if name.startswith(f"{month_key}_")
        }
        
        if len(month_facility_data) == 0:
            continue
        
        # Step 1-3: PINN at sources -> superimpose -> convert to ppb
        pinn_values = predict_pinn_at_sensors(pinn, month_facility_data, met_data_timestamp)
        
        # Get meteo data for NN2
        meteo_data_list = []
        for facility_name, facility_df in month_facility_data.items():
            facility_data = facility_df[facility_df['timestamp'] == met_data_timestamp]
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
        
        # Step 4: Apply NN2 correction
        nn2_values = apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, combined_meteo, forecast_timestamp, current_sensor_readings)
        
        # Clip negative values (concentrations cannot be negative)
        for sid in nn2_values:
            nn2_values[sid] = max(0.0, nn2_values[sid])
        
        # Collect results
        for sensor_id in SENSORS.keys():
            actual_col = f'sensor_{sensor_id}'
            if actual_col not in row or pd.isna(row[actual_col]):
                continue
            
            results.append({
                'timestamp': forecast_timestamp,
                'month': month_key,
                'met_data_timestamp': met_data_timestamp,
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
    
    # Save detailed results
    output_path = PROJECT_DIR / 'validation_results' / 'validation_jan_mar_2021_detailed.csv'
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved detailed results to {output_path}")
    
    # Compute MAE per sensor
    print("\n" + "="*80)
    print("RESULTS - PER SENSOR")
    print("="*80)
    
    sensor_summary = []
    for sensor_id in sorted(SENSORS.keys()):
        sensor_data = results_df[results_df['sensor_id'] == sensor_id]
        
        if len(sensor_data) == 0:
            continue
        
        pinn_mae = np.mean(np.abs(sensor_data['actual'] - sensor_data['pinn']))
        nn2_mae = np.mean(np.abs(sensor_data['actual'] - sensor_data['nn2']))
        improvement = ((pinn_mae - nn2_mae) / pinn_mae) * 100 if pinn_mae > 0 else 0.0
        
        sensor_summary.append({
            'sensor_id': sensor_id,
            'pinn_mae': pinn_mae,
            'nn2_mae': nn2_mae,
            'improvement_pct': improvement,
            'samples': len(sensor_data)
        })
        
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
    print(f"Hybrid (NN2) MAE:    {nn2_overall_mae:.4f} ppb")
    print(f"Improvement: {overall_improvement:.1f}%")
    print(f"Total samples: {len(results_df)}")
    
    # Monthly breakdown
    print("\n" + "="*80)
    print("MONTHLY BREAKDOWN")
    print("="*80)
    
    month_names = {'jan': 'January', 'feb': 'February', 'mar': 'March'}
    monthly_summary = []
    
    for month_key in ['jan', 'feb', 'mar']:
        month_data = results_df[results_df['month'] == month_key]
        if len(month_data) == 0:
            continue
        
        month_name = month_names[month_key]
        month_pinn_mae = np.mean(np.abs(month_data['actual'] - month_data['pinn']))
        month_nn2_mae = np.mean(np.abs(month_data['actual'] - month_data['nn2']))
        month_improvement = ((month_pinn_mae - month_nn2_mae) / month_pinn_mae) * 100 if month_pinn_mae > 0 else 0.0
        
        monthly_summary.append({
            'month': month_name,
            'pinn_mae': month_pinn_mae,
            'nn2_mae': month_nn2_mae,
            'improvement_pct': month_improvement,
            'samples': len(month_data)
        })
        
        print(f"\n{month_name}:")
        print(f"  PINN MAE:   {month_pinn_mae:.4f} ppb")
        print(f"  Hybrid (NN2) MAE:    {month_nn2_mae:.4f} ppb")
        print(f"  Improvement: {month_improvement:.1f}%")
        print(f"  Samples: {len(month_data)}")
    
    # Save summary
    summary_df = pd.DataFrame([
        {
            'metric': 'Overall',
            'pinn_mae': pinn_overall_mae,
            'nn2_mae': nn2_overall_mae,
            'improvement_pct': overall_improvement,
            'samples': len(results_df)
        }
    ] + monthly_summary + sensor_summary)
    
    summary_path = PROJECT_DIR / 'validation_results' / 'validation_jan_mar_2021_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved summary to {summary_path}")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nSummary:")
    print(f"  PINN MAE: {pinn_overall_mae:.4f} ppb")
    print(f"  Hybrid (NN2) MAE: {nn2_overall_mae:.4f} ppb")
    print(f"  Improvement: {overall_improvement:.1f}%")
    print(f"  Total samples: {len(results_df)}")

if __name__ == '__main__':
    main()

