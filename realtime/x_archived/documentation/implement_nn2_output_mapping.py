#!/usr/bin/env python3
"""
Implement the most accurate and scalable solution for NN2 inverse transform.

Solution: Learn a direct mapping from NN2 scaled outputs to actual ppb values.
This is more accurate than using the inverse of the input scaler because:
1. NN2 outputs are in a different distribution than inputs
2. The mapping accounts for the actual relationship learned by the model
3. It handles out-of-range values gracefully

Method: Fit a simple linear regression or use quantile mapping to learn
the relationship between NN2 scaled outputs and actual ppb values.
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
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

def generate_training_mapping():
    """Generate mapping from NN2 scaled outputs to actual ppb values"""
    print("="*80)
    print("GENERATING NN2 OUTPUT → PPB MAPPING")
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
    nn2_outputs_scaled = []
    actual_ppb_values = []
    
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
        
        # Collect mapping data (only non-zero actuals)
        for i, sensor_id in enumerate(sensor_ids_sorted):
            nn2_output_scaled = corrected_scaled_np[i]
            actual_ppb = actual_values.get(sensor_id, 0.0)
            
            if actual_ppb != 0.0:  # Only collect non-zero actuals
                nn2_outputs_scaled.append(nn2_output_scaled)
                actual_ppb_values.append(actual_ppb)
    
    nn2_outputs_scaled = np.array(nn2_outputs_scaled)
    actual_ppb_values = np.array(actual_ppb_values)
    
    print(f"\nCollected {len(nn2_outputs_scaled)} mapping samples")
    
    # Fit mapping models
    print("\n" + "="*80)
    print("FITTING MAPPING MODELS")
    print("="*80)
    
    # Option 1: Linear Regression (simple, fast, scalable)
    print("\n1. Linear Regression:")
    linear_model = LinearRegression()
    linear_model.fit(nn2_outputs_scaled.reshape(-1, 1), actual_ppb_values)
    linear_pred = linear_model.predict(nn2_outputs_scaled.reshape(-1, 1))
    linear_mae = np.mean(np.abs(linear_pred - actual_ppb_values))
    linear_r2 = linear_model.score(nn2_outputs_scaled.reshape(-1, 1), actual_ppb_values)
    print(f"   MAE: {linear_mae:.4f} ppb")
    print(f"   R²:  {linear_r2:.4f}")
    print(f"   Equation: ppb = {linear_model.coef_[0]:.4f} * scaled + {linear_model.intercept_:.4f}")
    
    # Option 2: Gradient Boosting (more accurate, but slower)
    print("\n2. Gradient Boosting Regressor:")
    gbr_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gbr_model.fit(nn2_outputs_scaled.reshape(-1, 1), actual_ppb_values)
    gbr_pred = gbr_model.predict(nn2_outputs_scaled.reshape(-1, 1))
    gbr_mae = np.mean(np.abs(gbr_pred - actual_ppb_values))
    gbr_r2 = gbr_model.score(nn2_outputs_scaled.reshape(-1, 1), actual_ppb_values)
    print(f"   MAE: {gbr_mae:.4f} ppb")
    print(f"   R²:  {gbr_r2:.4f}")
    
    # Option 3: Use original scaler inverse (baseline)
    print("\n3. Original Scaler Inverse (baseline):")
    original_pred = scalers['sensors'].inverse_transform(nn2_outputs_scaled.reshape(-1, 1)).flatten()
    original_mae = np.mean(np.abs(original_pred - actual_ppb_values))
    print(f"   MAE: {original_mae:.4f} ppb")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\nLinear Regression:    {linear_mae:.4f} ppb (improvement: {((original_mae - linear_mae) / original_mae * 100):.1f}%)")
    print(f"Gradient Boosting:    {gbr_mae:.4f} ppb (improvement: {((original_mae - gbr_mae) / original_mae * 100):.1f}%)")
    print(f"Original Scaler:      {original_mae:.4f} ppb (baseline)")
    
    # Recommendation: Use Linear Regression (most scalable)
    if linear_mae < gbr_mae * 1.1:  # If linear is within 10% of GBR, use it
        print(f"\n✓ RECOMMENDATION: Use Linear Regression (scalable, accurate)")
        mapping_model = linear_model
        mapping_type = 'linear'
    else:
        print(f"\n✓ RECOMMENDATION: Use Gradient Boosting (more accurate)")
        mapping_model = gbr_model
        mapping_type = 'gbr'
    
    # Save mapping model
    output_path = PROJECT_DIR / 'nn2_timefix' / 'nn2_output_to_ppb_mapping.pkl'
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': mapping_model,
            'type': mapping_type,
            'nn2_outputs_scaled': nn2_outputs_scaled,
            'actual_ppb_values': actual_ppb_values,
            'mae': linear_mae if mapping_type == 'linear' else gbr_mae,
            'r2': linear_r2 if mapping_type == 'linear' else gbr_r2,
        }, f)
    
    print(f"\n✓ Saved mapping model to {output_path}")
    
    return mapping_model, mapping_type

if __name__ == '__main__':
    print("="*80)
    print("IMPLEMENTING NN2 OUTPUT → PPB MAPPING")
    print("="*80)
    
    mapping_model, mapping_type = generate_training_mapping()
    
    print("\n" + "="*80)
    print("MAPPING MODEL CREATED")
    print("="*80)
    print(f"\nType: {mapping_type}")
    print(f"Usage: ppb = mapping_model.predict(nn2_output_scaled.reshape(-1, 1))")
    print(f"\nThis mapping should be used instead of scalers['sensors'].inverse_transform()")

