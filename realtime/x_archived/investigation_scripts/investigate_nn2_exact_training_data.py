#!/usr/bin/env python3
"""
Investigate NN2 Using EXACT Training Data

Test NN2 on the exact same data it was trained on:
- Use training PINN predictions directly (from total_concentrations.csv)
- Use actual sensor readings at same timestamps
- Compare with leave-one-out results
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

def load_facility_data():
    """
    Load meteorology data from single file (matching training).
    
    ⚠️ TEMPORARY CHANGE - TO BE REVERTED LATER ⚠️
    Training used only the first file alphabetically (BASF_Pasadena).
    This function now loads only that single file to match training exactly.
    This may need to be changed back if we want to use all facilities individually.
    """
    print("Loading facility meteorology data (single file - matching training)...")
    
    # Match training code: sorted(Path(source_dir).glob('*.csv'))[0]
    # Training used only the first file alphabetically
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    if len(facility_files) == 0:
        print("  ERROR: No facility files found!")
        return None
    
    # Use only the first file (BASF_Pasadena) - matching training
    first_file = facility_files[0]
    print(f"  Using single file (matching training): {first_file.name}")
    
    try:
        df = pd.read_csv(first_file)
        
        # Handle timestamp column
        if 't' in df.columns:
            df['timestamp'] = pd.to_datetime(df['t'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            print(f"  Warning: No timestamp column in {first_file.name}")
            return None
        
        print(f"  ✓ Loaded {len(df)} rows from single file")
        return df  # Return single DataFrame, not dict
    except Exception as e:
        print(f"  Warning: Failed to load {first_file.name}: {e}")
        return None

def get_meteorology_for_timestamp(meteo_df, timestamp):
    """
    Get meteorology data from single file for a given timestamp.
    
    ⚠️ TEMPORARY CHANGE - TO BE REVERTED LATER ⚠️
    Changed from averaging across facilities to using single file (BASF_Pasadena).
    This matches training exactly. May need to revert if we want per-facility meteorology.
    
    Args:
        meteo_df: Single DataFrame with meteorology (not dict of multiple facilities)
        timestamp: Forecast timestamp (t+3)
    
    Returns:
        (wind_u, wind_v, D) or None if no data available
    """
    if meteo_df is None:
        return None
    
    # Get met data timestamp (3 hours before forecast timestamp)
    # Predictions made at time t (using met data from t) are forecasts for t+3
    met_data_timestamp = timestamp - pd.Timedelta(hours=3)
    
    meteo_row = meteo_df[meteo_df['timestamp'] == met_data_timestamp]
    
    if len(meteo_row) == 0:
        return None
    
    row = meteo_row.iloc[0]
    if 'wind_u' in row and 'wind_v' in row and 'D' in row:
        u = row['wind_u']
        v = row['wind_v']
        d = row['D']
        
        # Check for valid values
        if pd.isna(u) or pd.isna(v) or pd.isna(d):
            return None
        
        return u, v, d
    
    return None

def test_on_exact_training_data():
    """Test NN2 on exact training data"""
    print("="*80)
    print("TESTING NN2 ON EXACT TRAINING DATA")
    print("="*80)
    
    # Load training PINN predictions (EXACT as used in training)
    training_pinn_df = pd.read_csv(TRAINING_PINN_PATH)
    training_pinn_df['timestamp'] = pd.to_datetime(training_pinn_df['timestamp'])
    
    # Load sensor data
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    # Merge on timestamp (EXACT training data alignment)
    merged = training_pinn_df.merge(sensor_df, on='timestamp', how='inner', suffixes=('_pinn', '_actual'))
    
    print(f"\nTraining data:")
    print(f"  Training PINN timestamps: {len(training_pinn_df)}")
    print(f"  Sensor data timestamps: {len(sensor_df)}")
    print(f"  Common timestamps (training data): {len(merged)}")
    
    if len(merged) == 0:
        print("  ERROR: No common timestamps!")
        return
    
    nn2, scalers, sensor_coords_spatial = load_models()
    
    # Load facility data for meteorology (single file - matching training)
    meteo_df = load_facility_data()
    if meteo_df is None:
        print("  ERROR: Could not load meteorology data!")
        return
    
    sensor_ids_sorted = sorted(SENSORS.keys())
    
    results = []
    
    print("\nProcessing training data...")
    for idx, row in merged.iterrows():
        timestamp = row['timestamp']
        
        # Get EXACT training PINN predictions
        pinn_values = {}
        for sensor_id in SENSORS.keys():
            col = f'sensor_{sensor_id}_pinn'
            if col in row and not pd.isna(row[col]):
                pinn_values[sensor_id] = row[col]
            else:
                pinn_values[sensor_id] = 0.0
        
        # Get actual sensor readings (EXACT as used in training)
        actual_values = {}
        for sensor_id in SENSORS.keys():
            col = f'sensor_{sensor_id}_actual'
            if col in row and not pd.isna(row[col]):
                actual_values[sensor_id] = row[col]
            else:
                actual_values[sensor_id] = 0.0
        
        # Prepare arrays
        pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
        current_sensors = np.array([actual_values.get(sid, 0.0) for sid in sensor_ids_sorted])
        
        # Get real meteorology from single file (exact match with training)
        meteo_data = get_meteorology_for_timestamp(meteo_df, timestamp)
        if meteo_data is None:
            # Skip if no meteorology data available
            continue
        meteo_u, meteo_v, meteo_D = meteo_data
        
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
        
        # Scale (zero-masking, matching training)
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
        corrections_np = corrections.cpu().numpy().flatten()
        
        # Inverse transform
        for i, sensor_id in enumerate(sensor_ids_sorted):
            actual = actual_values.get(sensor_id, 0.0)
            pinn = pinn_values[sensor_id]
            
            if sensors_nonzero_mask[i]:
                nn2_corrected = scalers['sensors'].inverse_transform(
                    corrected_scaled_np[i].reshape(-1, 1)
                )[0, 0]
            else:
                if np.abs(corrected_scaled_np[i]) > 1e-6:
                    nn2_corrected = scalers['sensors'].inverse_transform(
                        corrected_scaled_np[i].reshape(-1, 1)
                    )[0, 0]
                else:
                    nn2_corrected = 0.0
            
            results.append({
                'timestamp': timestamp,
                'sensor_id': sensor_id,
                'actual': actual,
                'pinn': pinn,
                'nn2': nn2_corrected,
                'pinn_error': np.abs(actual - pinn),
                'nn2_error': np.abs(actual - nn2_corrected),
                'correction_scaled': corrections_np[i],
                'corrected_scaled': corrected_scaled_np[i],
            })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("RESULTS: Exact Training Data")
    print("="*80)
    
    pinn_mae = results_df['pinn_error'].mean()
    nn2_mae = results_df['nn2_error'].mean()
    improvement = ((pinn_mae - nn2_mae) / pinn_mae) * 100 if pinn_mae > 0 else 0
    
    print(f"\nPINN MAE:   {pinn_mae:.4f} ppb")
    print(f"NN2 MAE:    {nn2_mae:.4f} ppb")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Total samples: {len(results_df)}")
    
    # Per-sensor
    print("\n" + "="*80)
    print("PER-SENSOR RESULTS")
    print("="*80)
    
    for sensor_id in sensor_ids_sorted:
        sensor_data = results_df[results_df['sensor_id'] == sensor_id]
        if len(sensor_data) == 0:
            continue
        
        sensor_pinn_mae = sensor_data['pinn_error'].mean()
        sensor_nn2_mae = sensor_data['nn2_error'].mean()
        sensor_improvement = ((sensor_pinn_mae - sensor_nn2_mae) / sensor_pinn_mae) * 100 if sensor_pinn_mae > 0 else 0
        
        print(f"\n{sensor_id}:")
        print(f"  PINN MAE: {sensor_pinn_mae:.4f} ppb")
        print(f"  NN2 MAE:  {sensor_nn2_mae:.4f} ppb")
        print(f"  Improvement: {sensor_improvement:.1f}%")
        print(f"  Samples: {len(sensor_data)}")
    
    # Save results
    output_path = PROJECT_DIR / 'validation_results' / 'nn2_exact_training_data_test.csv'
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved results to {output_path}")
    
    return results_df

if __name__ == '__main__':
    print("="*80)
    print("NN2 EXACT TRAINING DATA TEST")
    print("="*80)
    
    results_df = test_on_exact_training_data()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

