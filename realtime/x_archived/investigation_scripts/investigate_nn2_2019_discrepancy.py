#!/usr/bin/env python3
"""
Investigate NN2 Discrepancy on 2019 Data

Why is NN2 performing worse than PINN alone on the data it was trained on?
This script compares training data structure vs validation usage.
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

# Paths
PROJECT_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime')
SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
NN2_MODEL_PATH = PROJECT_DIR / "nn2_timefix/nn2_master_model_spatial-3.pth"
NN2_SCALERS_PATH = PROJECT_DIR / "nn2_timefix/nn2_master_scalers-2.pkl"
TRAINING_PINN_PATH = SYNCED_DIR / 'total_concentrations.csv'

# Constants
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
    """Load PINN and NN2 models"""
    print("Loading models...")
    
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
    
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    print("  ✓ Models loaded")
    return pinn, nn2, scalers, sensor_coords_spatial

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

def investigate_inverse_transform():
    """Test different inverse transform strategies"""
    print("\n" + "="*80)
    print("INVESTIGATING INVERSE TRANSFORM")
    print("="*80)
    
    # Load training PINN data
    training_pinn_df = pd.read_csv(TRAINING_PINN_PATH)
    training_pinn_df['timestamp'] = pd.to_datetime(training_pinn_df['timestamp'])
    
    # Load sensor data
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    # Filter to 2019
    year_mask = (sensor_df['timestamp'] >= '2019-01-01') & (sensor_df['timestamp'] < '2020-01-01')
    sensor_df = sensor_df[year_mask].reset_index(drop=True)
    
    # Load models
    pinn, nn2, scalers, sensor_coords_spatial = load_models()
    
    # Load facility data for meteorology (single file - matching training)
    meteo_df = load_facility_data()
    if meteo_df is None:
        print("  ERROR: Could not load meteorology data!")
        return None
    
    # Test on a few samples
    print("\nTesting inverse transform strategies on sample data...")
    
    sample_results = []
    
    for idx in range(min(100, len(sensor_df))):
        row = sensor_df.iloc[idx]
        forecast_timestamp = row['timestamp']
        
        # Get training PINN prediction
        pinn_row = training_pinn_df[training_pinn_df['timestamp'] == forecast_timestamp]
        if len(pinn_row) == 0:
            continue
        
        # Extract PINN values
        pinn_values = {}
        for sensor_id in SENSORS.keys():
            col = f'sensor_{sensor_id}'
            if col in pinn_row.columns:
                pinn_values[sensor_id] = pinn_row[col].iloc[0]
            else:
                pinn_values[sensor_id] = 0.0
        
        # Get actual sensor readings
        current_sensor_readings = {}
        for sensor_id in SENSORS.keys():
            actual_col = f'sensor_{sensor_id}'
            if actual_col in row and not pd.isna(row[actual_col]):
                current_sensor_readings[sensor_id] = row[actual_col]
            else:
                current_sensor_readings[sensor_id] = 0.0
        
        # Prepare inputs
        sensor_ids_sorted = sorted(SENSORS.keys())
        pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
        current_sensors = np.array([current_sensor_readings.get(sid, 0.0) for sid in sensor_ids_sorted])
        
        # Get real meteorology from single file (matching training)
        meteo_data = get_meteorology_for_timestamp(meteo_df, forecast_timestamp)
        if meteo_data is None:
            # Skip if no meteorology data available
            continue
        meteo_u, meteo_v, meteo_D = meteo_data
        
        # Temporal features
        hour = forecast_timestamp.hour
        day_of_week = forecast_timestamp.weekday()
        month = forecast_timestamp.month
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
        
        # Strategy 1: Use sensors_nonzero_mask (current validation approach)
        nn2_strategy1 = np.zeros_like(corrected_scaled_np)
        if sensors_nonzero_mask.any():
            nn2_strategy1[sensors_nonzero_mask] = scalers['sensors'].inverse_transform(
                corrected_scaled_np[sensors_nonzero_mask].reshape(-1, 1)
            ).flatten()
        
        # Strategy 2: Use threshold on output
        nn2_output_nonzero_mask = np.abs(corrected_scaled_np) > 1e-6
        nn2_strategy2 = np.zeros_like(corrected_scaled_np)
        if nn2_output_nonzero_mask.any():
            nn2_strategy2[nn2_output_nonzero_mask] = scalers['sensors'].inverse_transform(
                corrected_scaled_np[nn2_output_nonzero_mask].reshape(-1, 1)
            ).flatten()
        
        # Strategy 3: Inverse transform ALL values (like benzene_pipeline.py)
        nn2_strategy3 = scalers['sensors'].inverse_transform(corrected_scaled_np.reshape(-1, 1)).flatten()
        
        # Compare strategies
        for i, sensor_id in enumerate(sensor_ids_sorted):
            actual = current_sensor_readings.get(sensor_id, 0.0)
            pinn_pred = pinn_values[sensor_id]
            
            sample_results.append({
                'timestamp': forecast_timestamp,
                'sensor_id': sensor_id,
                'actual': actual,
                'pinn': pinn_pred,
                'pinn_scaled': p_s[0, i],
                'current_sensor_scaled': s_s[0, i],
                'corrected_scaled': corrected_scaled_np[i],
                'correction_scaled': corrections_np[i],
                'nn2_strategy1': nn2_strategy1[i],
                'nn2_strategy2': nn2_strategy2[i],
                'nn2_strategy3': nn2_strategy3[i],
                'sensors_nonzero_mask': sensors_nonzero_mask[i],
                'output_nonzero_mask': nn2_output_nonzero_mask[i],
            })
    
    results_df = pd.DataFrame(sample_results)
    
    # Calculate MAE for each strategy
    print("\n" + "="*80)
    print("MAE COMPARISON - Different Inverse Transform Strategies")
    print("="*80)
    
    pinn_mae = np.mean(np.abs(results_df['actual'] - results_df['pinn']))
    strategy1_mae = np.mean(np.abs(results_df['actual'] - results_df['nn2_strategy1']))
    strategy2_mae = np.mean(np.abs(results_df['actual'] - results_df['nn2_strategy2']))
    strategy3_mae = np.mean(np.abs(results_df['actual'] - results_df['nn2_strategy3']))
    
    print(f"\nPINN MAE:                    {pinn_mae:.4f} ppb")
    print(f"NN2 Strategy 1 (sensors mask): {strategy1_mae:.4f} ppb")
    print(f"NN2 Strategy 2 (output mask):  {strategy2_mae:.4f} ppb")
    print(f"NN2 Strategy 3 (all values):   {strategy3_mae:.4f} ppb")
    
    # Save detailed results
    output_path = PROJECT_DIR / 'validation_results' / 'nn2_inverse_transform_investigation.csv'
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved detailed results to {output_path}")
    
    # Analyze differences
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    print(f"\nStrategy 1 vs Strategy 2 differences:")
    diff_12 = np.abs(results_df['nn2_strategy1'] - results_df['nn2_strategy2'])
    print(f"  Mean absolute difference: {diff_12.mean():.6f} ppb")
    print(f"  Max difference: {diff_12.max():.6f} ppb")
    print(f"  Cases where different: {(diff_12 > 1e-6).sum()} / {len(diff_12)}")
    
    print(f"\nStrategy 1 vs Strategy 3 differences:")
    diff_13 = np.abs(results_df['nn2_strategy1'] - results_df['nn2_strategy3'])
    print(f"  Mean absolute difference: {diff_13.mean():.6f} ppb")
    print(f"  Max difference: {diff_13.max():.6f} ppb")
    
    print(f"\nScaled space analysis:")
    print(f"  PINN scaled range: [{results_df['pinn_scaled'].min():.4f}, {results_df['pinn_scaled'].max():.4f}]")
    print(f"  Current sensor scaled range: [{results_df['current_sensor_scaled'].min():.4f}, {results_df['current_sensor_scaled'].max():.4f}]")
    print(f"  Corrected scaled range: [{results_df['corrected_scaled'].min():.4f}, {results_df['corrected_scaled'].max():.4f}]")
    print(f"  Correction scaled range: [{results_df['correction_scaled'].min():.4f}, {results_df['correction_scaled'].max():.4f}]")
    
    return results_df

def check_training_data_structure():
    """Check the exact structure of training data"""
    print("\n" + "="*80)
    print("CHECKING TRAINING DATA STRUCTURE")
    print("="*80)
    
    # Load training PINN data
    training_pinn_df = pd.read_csv(TRAINING_PINN_PATH)
    training_pinn_df['timestamp'] = pd.to_datetime(training_pinn_df['timestamp'])
    
    # Load sensor data
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    
    # Merge on timestamp
    merged = training_pinn_df.merge(sensor_df, on='timestamp', how='inner', suffixes=('_pinn', '_actual'))
    
    print(f"\nTraining data structure:")
    print(f"  Training PINN timestamps: {len(training_pinn_df)}")
    print(f"  Sensor data timestamps: {len(sensor_df)}")
    print(f"  Common timestamps: {len(merged)}")
    
    if len(merged) > 0:
        print(f"\nSample merged data:")
        print(merged[['timestamp'] + [f'sensor_{sid}_pinn' for sid in sorted(SENSORS.keys())[:3]] + 
                      [f'sensor_{sid}_actual' for sid in sorted(SENSORS.keys())[:3]]].head())
        
        # Check alignment
        print(f"\nChecking alignment...")
        for sensor_id in sorted(SENSORS.keys())[:3]:
            pinn_col = f'sensor_{sensor_id}_pinn'
            actual_col = f'sensor_{sensor_id}_actual'
            if pinn_col in merged.columns and actual_col in merged.columns:
                pinn_vals = merged[pinn_col].values
                actual_vals = merged[actual_col].values
                valid_mask = (~pd.isna(actual_vals)) & (actual_vals != 0)
                if valid_mask.sum() > 0:
                    mae = np.mean(np.abs(pinn_vals[valid_mask] - actual_vals[valid_mask]))
                    print(f"  {sensor_id}: PINN MAE = {mae:.4f} ppb (on {valid_mask.sum()} samples)")

if __name__ == '__main__':
    print("="*80)
    print("NN2 DISCREPANCY INVESTIGATION - 2019 DATA")
    print("="*80)
    
    check_training_data_structure()
    results_df = investigate_inverse_transform()
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)

