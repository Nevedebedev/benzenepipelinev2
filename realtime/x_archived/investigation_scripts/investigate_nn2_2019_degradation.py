#!/usr/bin/env python3
"""
Investigate NN2 Performance Degradation on 2019 Data

Compare training data characteristics vs validation predictions to identify
why NN2 performs worse on the same data it was trained on.
"""

import sys
from pathlib import Path
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pinn import ParametricADEPINN
from nn2 import NN2_CorrectionNetwork
import pickle
import matplotlib.pyplot as plt

# Paths
SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
NN2_MODEL_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_timefix/nn2_master_model_spatial-3.pth"
NN2_SCALERS_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_timefix/nn2_master_scalers-2.pkl"

UNIT_CONVERSION = 313210039.9
FORECAST_T_HOURS = 3.0

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
    """Load models and scalers"""
    print("Loading models...")
    
    pinn = ParametricADEPINN()
    checkpoint = torch.load(PINN_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    
    # Override normalization ranges
    for attr, val in [
        ('x_min', 0.0), ('x_max', 30000.0), ('y_min', 0.0), ('y_max', 30000.0),
        ('t_min', 0.0), ('t_max', 8760.0), ('cx_min', 0.0), ('cx_max', 30000.0),
        ('cy_min', 0.0), ('cy_max', 30000.0), ('u_min', -15.0), ('u_max', 15.0),
        ('v_min', -15.0), ('v_max', 15.0), ('d_min', 0.0), ('d_max', 200.0),
        ('kappa_min', 0.0), ('kappa_max', 200.0), ('Q_min', 0.0), ('Q_max', 0.01)
    ]:
        setattr(pinn, attr, torch.tensor(val))
    pinn.eval()
    
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    print("  ✓ Models loaded")
    return pinn, nn2, scalers, sensor_coords_spatial

def load_training_data():
    """Load the actual training data used for NN2"""
    print("\nLoading training data...")
    
    # Load total concentrations (PINN predictions at sensors)
    pinn_df = pd.read_csv(SYNCED_DIR / 'total_concentrations.csv')
    if 't' in pinn_df.columns:
        pinn_df['timestamp'] = pd.to_datetime(pinn_df['t'])
    elif 'timestamp' in pinn_df.columns:
        pinn_df['timestamp'] = pd.to_datetime(pinn_df['timestamp'])
    
    # Load sensor data (actual readings)
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['t'])
    elif 'timestamp' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    # Merge on timestamp
    merged = pinn_df.merge(sensor_df, on='timestamp', how='inner', suffixes=('_pinn', '_actual'))
    
    print(f"  Loaded {len(merged)} training samples")
    return merged

def analyze_training_data_distribution(merged_df, scalers):
    """Analyze distribution of training data features"""
    print("\n" + "="*80)
    print("TRAINING DATA DISTRIBUTION ANALYSIS")
    print("="*80)
    
    sensor_ids_sorted = sorted(SENSORS.keys())
    
    # Extract PINN predictions and actual sensor readings
    # Training data has columns like 'sensor_482010026_pinn' for PINN predictions
    # And 'sensor_482010026_actual' for actual sensor readings
    pinn_cols = [f'sensor_{sid}_pinn' for sid in sensor_ids_sorted]
    actual_cols = [f'sensor_{sid}_actual' for sid in sensor_ids_sorted]
    
    # Check which columns exist
    available_pinn = [col for col in pinn_cols if col in merged_df.columns]
    available_actual = [col for col in actual_cols if col in merged_df.columns]
    
    if not available_pinn:
        print("  ⚠ Warning: Could not find PINN columns")
        print(f"  Available columns: {merged_df.columns.tolist()[:15]}...")
        return None
    
    if not available_actual:
        print("  ⚠ Warning: Could not find actual sensor columns")
        print(f"  Available columns: {merged_df.columns.tolist()[:15]}...")
        return None
    
    # Get PINN and actual values
    pinn_values = merged_df[available_pinn].values.flatten()
    actual_values = merged_df[available_actual].values.flatten()
    
    # Remove zeros and NaNs
    pinn_nonzero = pinn_values[(pinn_values > 0) & ~np.isnan(pinn_values)]
    actual_nonzero = actual_values[(actual_values > 0) & ~np.isnan(actual_values)]
    
    print(f"\nPINN Predictions (ppb):")
    print(f"  Total samples: {len(pinn_values)}")
    print(f"  Non-zero samples: {len(pinn_nonzero)}")
    print(f"  Min: {pinn_nonzero.min():.4f} ppb")
    print(f"  Max: {pinn_nonzero.max():.4f} ppb")
    print(f"  Mean: {pinn_nonzero.mean():.4f} ppb")
    print(f"  Median: {np.median(pinn_nonzero):.4f} ppb")
    print(f"  Std: {pinn_nonzero.std():.4f} ppb")
    
    print(f"\nActual Sensor Readings (ppb):")
    print(f"  Total samples: {len(actual_values)}")
    print(f"  Non-zero samples: {len(actual_nonzero)}")
    print(f"  Min: {actual_nonzero.min():.4f} ppb")
    print(f"  Max: {actual_nonzero.max():.4f} ppb")
    print(f"  Mean: {actual_nonzero.mean():.4f} ppb")
    print(f"  Median: {np.median(actual_nonzero):.4f} ppb")
    print(f"  Std: {actual_nonzero.std():.4f} ppb")
    
    # Scale and check distribution
    print(f"\nScaled PINN Predictions (using training scaler):")
    pinn_scaled = scalers['pinn'].transform(pinn_nonzero.reshape(-1, 1)).flatten()
    print(f"  Min: {pinn_scaled.min():.4f}")
    print(f"  Max: {pinn_scaled.max():.4f}")
    print(f"  Mean: {pinn_scaled.mean():.4f}")
    print(f"  Std: {pinn_scaled.std():.4f}")
    
    print(f"\nScaled Actual Sensors (using training scaler):")
    actual_scaled = scalers['sensors'].transform(actual_nonzero.reshape(-1, 1)).flatten()
    print(f"  Min: {actual_scaled.min():.4f}")
    print(f"  Max: {actual_scaled.max():.4f}")
    print(f"  Mean: {actual_scaled.mean():.4f}")
    print(f"  Std: {actual_scaled.std():.4f}")
    
    return {
        'pinn_ppb': pinn_nonzero,
        'actual_ppb': actual_nonzero,
        'pinn_scaled': pinn_scaled,
        'actual_scaled': actual_scaled
    }

def predict_pinn_at_sensors(pinn, facility_files_dict, timestamp):
    """Predict PINN at sensor locations"""
    input_timestamp = timestamp - pd.Timedelta(hours=3)
    sensor_pinn_ppb = {sid: 0.0 for sid in SENSORS.keys()}
    
    for facility_name, facility_df in facility_files_dict.items():
        facility_data = facility_df[facility_df['t'] == input_timestamp]
        if len(facility_data) == 0:
            continue
        
        for _, row in facility_data.iterrows():
            for sensor_id, (sx, sy) in SENSORS.items():
                with torch.no_grad():
                    phi_raw = pinn(
                        torch.tensor([[sx]], dtype=torch.float32),
                        torch.tensor([[sy]], dtype=torch.float32),
                        torch.tensor([[FORECAST_T_HOURS]], dtype=torch.float32),
                        torch.tensor([[row['source_x_cartesian']]], dtype=torch.float32),
                        torch.tensor([[row['source_y_cartesian']]], dtype=torch.float32),
                        torch.tensor([[row['wind_u']]], dtype=torch.float32),
                        torch.tensor([[row['wind_v']]], dtype=torch.float32),
                        torch.tensor([[row['source_diameter']]], dtype=torch.float32),
                        torch.tensor([[row['D']]], dtype=torch.float32),
                        torch.tensor([[row['Q_total']]], dtype=torch.float32),
                        normalize=True
                    )
                    sensor_pinn_ppb[sensor_id] += phi_raw.item() * UNIT_CONVERSION
    
    return sensor_pinn_ppb

def analyze_validation_predictions(pinn, nn2, scalers, sensor_coords_spatial, facility_files_dict, sensor_df, sample_size=1000):
    """Analyze validation predictions and compare to training"""
    print("\n" + "="*80)
    print("VALIDATION PREDICTIONS ANALYSIS")
    print("="*80)
    
    print(f"\nProcessing {sample_size} validation samples...")
    
    all_pinn_ppb = []
    all_actual_ppb = []
    all_pinn_scaled = []
    all_actual_scaled = []
    all_nn2_scaled = []
    all_nn2_ppb = []
    
    sample_df = sensor_df.head(sample_size)
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing"):
        timestamp = row['timestamp']
        met_data_timestamp = timestamp - pd.Timedelta(hours=3)
        
        # Get PINN predictions
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
        
        # Get actual sensor readings
        current_sensor_readings = {}
        sensor_ids_sorted = sorted(SENSORS.keys())
        for sensor_id in sensor_ids_sorted:
            actual_col = f'sensor_{sensor_id}'
            if actual_col in row and not pd.isna(row[actual_col]):
                current_sensor_readings[sensor_id] = row[actual_col]
            else:
                current_sensor_readings[sensor_id] = 0.0
        
        # Prepare NN2 inputs
        pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
        current_sensors = np.array([current_sensor_readings.get(sid, 0.0) for sid in sensor_ids_sorted])
        
        avg_u = combined_meteo['wind_u'].mean()
        avg_v = combined_meteo['wind_v'].mean()
        avg_D = combined_meteo['D'].mean()
        
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
        
        # Run NN2
        with torch.no_grad():
            corrected_scaled, corrections = nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
        
        corrected_scaled_np = corrected_scaled.cpu().numpy().flatten()
        
        # Inverse transform
        nn2_ppb = np.zeros_like(corrected_scaled_np)
        nn2_output_nonzero_mask = np.abs(corrected_scaled_np) > 1e-6
        if nn2_output_nonzero_mask.any():
            nn2_ppb[nn2_output_nonzero_mask] = scalers['sensors'].inverse_transform(
                corrected_scaled_np[nn2_output_nonzero_mask].reshape(-1, 1)
            ).flatten()
        
        # Collect data (only non-zero for analysis)
        for i, sensor_id in enumerate(sensor_ids_sorted):
            if pinn_array[i] > 0 and current_sensors[i] > 0:
                all_pinn_ppb.append(pinn_array[i])
                all_actual_ppb.append(current_sensors[i])
                all_pinn_scaled.append(p_s[0, i])
                all_actual_scaled.append(s_s[0, i])
                all_nn2_scaled.append(corrected_scaled_np[i])
                all_nn2_ppb.append(nn2_ppb[i])
    
    all_pinn_ppb = np.array(all_pinn_ppb)
    all_actual_ppb = np.array(all_actual_ppb)
    all_pinn_scaled = np.array(all_pinn_scaled)
    all_actual_scaled = np.array(all_actual_scaled)
    all_nn2_scaled = np.array(all_nn2_scaled)
    all_nn2_ppb = np.array(all_nn2_ppb)
    
    print(f"\nValidation PINN Predictions (ppb):")
    print(f"  Samples: {len(all_pinn_ppb)}")
    print(f"  Min: {all_pinn_ppb.min():.4f} ppb")
    print(f"  Max: {all_pinn_ppb.max():.4f} ppb")
    print(f"  Mean: {all_pinn_ppb.mean():.4f} ppb")
    print(f"  Median: {np.median(all_pinn_ppb):.4f} ppb")
    print(f"  Std: {all_pinn_ppb.std():.4f} ppb")
    
    print(f"\nValidation Actual Sensor Readings (ppb):")
    print(f"  Samples: {len(all_actual_ppb)}")
    print(f"  Min: {all_actual_ppb.min():.4f} ppb")
    print(f"  Max: {all_actual_ppb.max():.4f} ppb")
    print(f"  Mean: {all_actual_ppb.mean():.4f} ppb")
    print(f"  Median: {np.median(all_actual_ppb):.4f} ppb")
    print(f"  Std: {all_actual_ppb.std():.4f} ppb")
    
    print(f"\nValidation Scaled PINN Predictions:")
    print(f"  Min: {all_pinn_scaled.min():.4f}")
    print(f"  Max: {all_pinn_scaled.max():.4f}")
    print(f"  Mean: {all_pinn_scaled.mean():.4f}")
    print(f"  Std: {all_pinn_scaled.std():.4f}")
    
    print(f"\nValidation Scaled Actual Sensors:")
    print(f"  Min: {all_actual_scaled.min():.4f}")
    print(f"  Max: {all_actual_scaled.max():.4f}")
    print(f"  Mean: {all_actual_scaled.mean():.4f}")
    print(f"  Std: {all_actual_scaled.std():.4f}")
    
    print(f"\nNN2 Outputs (scaled space):")
    print(f"  Min: {all_nn2_scaled.min():.4f}")
    print(f"  Max: {all_nn2_scaled.max():.4f}")
    print(f"  Mean: {all_nn2_scaled.mean():.4f}")
    print(f"  Std: {all_nn2_scaled.std():.4f}")
    print(f"  Negative values: {(all_nn2_scaled < 0).sum()} ({(all_nn2_scaled < 0).sum()/len(all_nn2_scaled)*100:.1f}%)")
    
    print(f"\nNN2 Outputs (ppb, after inverse transform):")
    print(f"  Min: {all_nn2_ppb.min():.4f} ppb")
    print(f"  Max: {all_nn2_ppb.max():.4f} ppb")
    print(f"  Mean: {all_nn2_ppb.mean():.4f} ppb")
    print(f"  Median: {np.median(all_nn2_ppb):.4f} ppb")
    print(f"  Std: {all_nn2_ppb.std():.4f} ppb")
    print(f"  Negative values: {(all_nn2_ppb < 0).sum()} ({(all_nn2_ppb < 0).sum()/len(all_nn2_ppb)*100:.1f}%)")
    
    # Check scaler range
    print(f"\nScaler Statistics (for inverse transform):")
    if hasattr(scalers['sensors'], 'mean_'):
        print(f"  Mean: {scalers['sensors'].mean_[0]:.4f}")
        print(f"  Scale: {scalers['sensors'].scale_[0]:.4f}")
        print(f"  Training range (approx): [{scalers['sensors'].mean_[0] - 3*scalers['sensors'].scale_[0]:.4f}, {scalers['sensors'].mean_[0] + 3*scalers['sensors'].scale_[0]:.4f}]")
        print(f"  NN2 output range: [{all_nn2_scaled.min():.4f}, {all_nn2_scaled.max():.4f}]")
        out_of_range = (all_nn2_scaled < scalers['sensors'].mean_[0] - 3*scalers['sensors'].scale_[0]) | (all_nn2_scaled > scalers['sensors'].mean_[0] + 3*scalers['sensors'].scale_[0])
        print(f"  Out-of-range values: {out_of_range.sum()} ({out_of_range.sum()/len(all_nn2_scaled)*100:.1f}%)")
    
    return {
        'pinn_ppb': all_pinn_ppb,
        'actual_ppb': all_actual_ppb,
        'pinn_scaled': all_pinn_scaled,
        'actual_scaled': all_actual_scaled,
        'nn2_scaled': all_nn2_scaled,
        'nn2_ppb': all_nn2_ppb
    }

def compare_distributions(train_data, val_data):
    """Compare training vs validation distributions"""
    print("\n" + "="*80)
    print("TRAINING vs VALIDATION DISTRIBUTION COMPARISON")
    print("="*80)
    
    print("\nPINN Predictions (ppb):")
    print(f"  Training - Mean: {train_data['pinn_ppb'].mean():.4f}, Std: {train_data['pinn_ppb'].std():.4f}")
    print(f"  Validation - Mean: {val_data['pinn_ppb'].mean():.4f}, Std: {val_data['pinn_ppb'].std():.4f}")
    print(f"  Difference - Mean: {val_data['pinn_ppb'].mean() - train_data['pinn_ppb'].mean():.4f}, Std: {val_data['pinn_ppb'].std() - train_data['pinn_ppb'].std():.4f}")
    
    print("\nActual Sensor Readings (ppb):")
    print(f"  Training - Mean: {train_data['actual_ppb'].mean():.4f}, Std: {train_data['actual_ppb'].std():.4f}")
    print(f"  Validation - Mean: {val_data['actual_ppb'].mean():.4f}, Std: {val_data['actual_ppb'].std():.4f}")
    print(f"  Difference - Mean: {val_data['actual_ppb'].mean() - train_data['actual_ppb'].mean():.4f}, Std: {val_data['actual_ppb'].std() - train_data['actual_ppb'].std():.4f}")
    
    print("\nScaled PINN Predictions:")
    print(f"  Training - Mean: {train_data['pinn_scaled'].mean():.4f}, Std: {train_data['pinn_scaled'].std():.4f}")
    print(f"  Validation - Mean: {val_data['pinn_scaled'].mean():.4f}, Std: {val_data['pinn_scaled'].std():.4f}")
    print(f"  Difference - Mean: {val_data['pinn_scaled'].mean() - train_data['pinn_scaled'].mean():.4f}, Std: {val_data['pinn_scaled'].std() - train_data['pinn_scaled'].std():.4f}")
    
    print("\nScaled Actual Sensors:")
    print(f"  Training - Mean: {train_data['actual_scaled'].mean():.4f}, Std: {train_data['actual_scaled'].std():.4f}")
    print(f"  Validation - Mean: {val_data['actual_scaled'].mean():.4f}, Std: {val_data['actual_scaled'].std():.4f}")
    print(f"  Difference - Mean: {val_data['actual_scaled'].mean() - train_data['actual_scaled'].mean():.4f}, Std: {val_data['actual_scaled'].std() - train_data['actual_scaled'].std():.4f}")

def main():
    print("="*80)
    print("NN2 PERFORMANCE DEGRADATION INVESTIGATION - 2019 DATA")
    print("="*80)
    
    # Load models
    pinn, nn2, scalers, sensor_coords_spatial = load_models()
    
    # Load training data
    train_df = load_training_data()
    train_dist = analyze_training_data_distribution(train_df, scalers)
    
    if train_dist is None:
        print("\n⚠ Could not analyze training data distribution")
        return
    
    # Load validation data
    print("\nLoading validation data...")
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['t'])
    elif 'timestamp' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    facility_files_dict = {}
    for f in facility_files:
        df = pd.read_csv(f)
        df['t'] = pd.to_datetime(df['t'])
        facility_name = f.stem.replace('_synced_training_data', '')
        facility_files_dict[facility_name] = df
    
    # Analyze validation predictions
    val_dist = analyze_validation_predictions(
        pinn, nn2, scalers, sensor_coords_spatial, 
        facility_files_dict, sensor_df, sample_size=2000
    )
    
    # Compare distributions
    compare_distributions(train_dist, val_dist)
    
    # Calculate errors
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    # Training errors - need to align by timestamp
    # For now, just report what we have
    print(f"\nTraining Data:")
    print(f"  PINN samples: {len(train_dist['pinn_ppb'])}")
    print(f"  Actual samples: {len(train_dist['actual_ppb'])}")
    print(f"  Note: Cannot compute exact MAE without timestamp alignment")
    
    # Validation errors
    val_pinn_mae = np.abs(val_dist['pinn_ppb'] - val_dist['actual_ppb']).mean()
    val_nn2_mae = np.abs(val_dist['nn2_ppb'] - val_dist['actual_ppb']).mean()
    
    print(f"\nValidation Data:")
    print(f"  PINN MAE: {val_pinn_mae:.4f} ppb")
    print(f"  NN2 MAE: {val_nn2_mae:.4f} ppb")
    print(f"  Improvement: {((val_pinn_mae - val_nn2_mae) / val_pinn_mae * 100):.1f}%")
    
    # Check if NN2 outputs are in valid range for inverse transform
    print("\n" + "="*80)
    print("INVERSE TRANSFORM VALIDITY CHECK")
    print("="*80)
    
    if hasattr(scalers['sensors'], 'mean_') and hasattr(scalers['sensors'], 'scale_'):
        scaler_mean = scalers['sensors'].mean_[0]
        scaler_scale = scalers['sensors'].scale_[0]
        
        # Expected range for inverse transform (roughly ±3 std)
        expected_min = scaler_mean - 3 * scaler_scale
        expected_max = scaler_mean + 3 * scaler_scale
        
        print(f"\nScaler Training Range (approx ±3σ):")
        print(f"  [{expected_min:.4f}, {expected_max:.4f}]")
        
        print(f"\nNN2 Scaled Output Range:")
        print(f"  [{val_dist['nn2_scaled'].min():.4f}, {val_dist['nn2_scaled'].max():.4f}]")
        
        out_of_range = (val_dist['nn2_scaled'] < expected_min) | (val_dist['nn2_scaled'] > expected_max)
        print(f"\nOut-of-Range Values:")
        print(f"  Count: {out_of_range.sum()} ({out_of_range.sum()/len(val_dist['nn2_scaled'])*100:.1f}%)")
        if out_of_range.any():
            print(f"  Min out-of-range: {val_dist['nn2_scaled'][out_of_range].min():.4f}")
            print(f"  Max out-of-range: {val_dist['nn2_scaled'][out_of_range].max():.4f}")
            print(f"  Mean out-of-range: {val_dist['nn2_scaled'][out_of_range].mean():.4f}")
    
    # Analyze corrections
    print("\n" + "="*80)
    print("CORRECTION ANALYSIS")
    print("="*80)
    
    # Corrections in scaled space (NN2 output - PINN input)
    corrections_scaled = val_dist['nn2_scaled'] - val_dist['pinn_scaled']
    
    print(f"\nCorrections (scaled space):")
    print(f"  Min: {corrections_scaled.min():.4f}")
    print(f"  Max: {corrections_scaled.max():.4f}")
    print(f"  Mean: {corrections_scaled.mean():.4f}")
    print(f"  Std: {corrections_scaled.std():.4f}")
    print(f"  Negative corrections: {(corrections_scaled < 0).sum()} ({(corrections_scaled < 0).sum()/len(corrections_scaled)*100:.1f}%)")
    
    # Corrections in ppb space
    corrections_ppb = val_dist['nn2_ppb'] - val_dist['pinn_ppb']
    
    print(f"\nCorrections (ppb space):")
    print(f"  Min: {corrections_ppb.min():.4f} ppb")
    print(f"  Max: {corrections_ppb.max():.4f} ppb")
    print(f"  Mean: {corrections_ppb.mean():.4f} ppb")
    print(f"  Std: {corrections_ppb.std():.4f} ppb")
    print(f"  Negative corrections: {(corrections_ppb < 0).sum()} ({(corrections_ppb < 0).sum()/len(corrections_ppb)*100:.1f}%)")
    
    # Analyze when corrections are too large (overcorrection)
    print(f"\nOvercorrection Analysis:")
    # Cases where correction is negative and larger than PINN (making result negative)
    overcorrection_mask = (corrections_ppb < 0) & (np.abs(corrections_ppb) > val_dist['pinn_ppb'])
    print(f"  Cases where correction > PINN (making result negative): {overcorrection_mask.sum()} ({overcorrection_mask.sum()/len(corrections_ppb)*100:.1f}%)")
    
    if overcorrection_mask.any():
        print(f"  Average PINN in overcorrections: {val_dist['pinn_ppb'][overcorrection_mask].mean():.4f} ppb")
        print(f"  Average correction in overcorrections: {corrections_ppb[overcorrection_mask].mean():.4f} ppb")
        print(f"  Average actual in overcorrections: {val_dist['actual_ppb'][overcorrection_mask].mean():.4f} ppb")
    
    # Check correlation between PINN and corrections
    print(f"\nCorrelation Analysis:")
    valid_mask = (val_dist['pinn_ppb'] > 0) & (val_dist['actual_ppb'] > 0)
    if valid_mask.sum() > 0:
        corr_pinn_correction = np.corrcoef(val_dist['pinn_ppb'][valid_mask], corrections_ppb[valid_mask])[0, 1]
        corr_actual_correction = np.corrcoef(val_dist['actual_ppb'][valid_mask], corrections_ppb[valid_mask])[0, 1]
        print(f"  PINN vs Correction correlation: {corr_pinn_correction:.4f}")
        print(f"  Actual vs Correction correlation: {corr_actual_correction:.4f}")
    
    # Analyze when NN2 is worse than PINN
    print(f"\nPerformance Analysis:")
    pinn_errors = np.abs(val_dist['pinn_ppb'] - val_dist['actual_ppb'])
    nn2_errors = np.abs(val_dist['nn2_ppb'] - val_dist['actual_ppb'])
    
    worse_mask = nn2_errors > pinn_errors
    print(f"  Cases where NN2 error > PINN error: {worse_mask.sum()} ({worse_mask.sum()/len(nn2_errors)*100:.1f}%)")
    
    if worse_mask.any():
        print(f"  Average PINN error when NN2 worse: {pinn_errors[worse_mask].mean():.4f} ppb")
        print(f"  Average NN2 error when NN2 worse: {nn2_errors[worse_mask].mean():.4f} ppb")
        print(f"  Average PINN value when NN2 worse: {val_dist['pinn_ppb'][worse_mask].mean():.4f} ppb")
        print(f"  Average actual value when NN2 worse: {val_dist['actual_ppb'][worse_mask].mean():.4f} ppb")
        print(f"  Average correction when NN2 worse: {corrections_ppb[worse_mask].mean():.4f} ppb")
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()

