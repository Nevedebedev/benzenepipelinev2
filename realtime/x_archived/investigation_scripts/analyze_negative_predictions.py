#!/usr/bin/env python3
"""
Analyze negative NN2 predictions before clamping
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
from tqdm import tqdm
from pinn import ParametricADEPINN
from nn2_ppbscale import NN2_CorrectionNetwork
import pickle

# Paths
SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
NN2_MODEL_PATH = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_ppbscale/nn2_master_model_ppb.pth')
NN2_SCALERS_PATH = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_ppbscale/nn2_master_scalers-2.pkl')

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
    
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    scaler_mean = nn2_checkpoint.get('scaler_mean', None)
    scaler_scale = nn2_checkpoint.get('scaler_scale', None)
    
    if scaler_mean is None or scaler_scale is None:
        with open(NN2_SCALERS_PATH, 'rb') as f:
            scalers_temp = pickle.load(f)
        scaler_mean = scalers_temp['sensors'].mean_[0] if hasattr(scalers_temp['sensors'], 'mean_') else None
        scaler_scale = scalers_temp['sensors'].scale_[0] if hasattr(scalers_temp['sensors'], 'scale_') else None
    
    nn2 = NN2_CorrectionNetwork(n_sensors=9, scaler_mean=scaler_mean, scaler_scale=scaler_scale, output_ppb=True)
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    return pinn, nn2, scalers, sensor_coords_spatial

def predict_pinn_at_sensors(pinn, facility_files_dict, timestamp):
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

def apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, meteo_data, timestamp, current_sensor_readings, return_unclamped=False):
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
        corrected_ppb, corrections = nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
    
    nn2_unclamped = corrected_ppb.cpu().numpy().flatten()
    nn2_clamped = np.maximum(nn2_unclamped, 0.0)
    
    if return_unclamped:
        return {sid: nn2_unclamped[i] for i, sid in enumerate(sensor_ids_sorted)}, \
               {sid: nn2_clamped[i] for i, sid in enumerate(sensor_ids_sorted)}
    else:
        return {sid: nn2_clamped[i] for i, sid in enumerate(sensor_ids_sorted)}

def main():
    print("="*80)
    print("ANALYZING NEGATIVE PREDICTIONS (BEFORE CLAMPING)")
    print("="*80)
    print()
    
    pinn, nn2, scalers, sensor_coords_spatial = load_models()
    
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
    
    print("Processing predictions (sampling first 1000 for speed)...")
    results = []
    
    sample_size = min(1000, len(sensor_df))
    for idx, row in tqdm(sensor_df.head(sample_size).iterrows(), total=sample_size, desc="Processing"):
        timestamp = row['timestamp']
        met_data_timestamp = timestamp - pd.Timedelta(hours=3)
        
        pinn_values = predict_pinn_at_sensors(pinn, facility_files_dict, timestamp)
        
        meteo_data_list = []
        for facility_name, facility_df in facility_files_dict.items():
            facility_data = facility_df[facility_df['t'] == met_data_timestamp]
            if len(facility_data) > 0:
                meteo_data_list.append(facility_data)
        
        if len(meteo_data_list) == 0:
            continue
        
        combined_meteo = pd.concat(meteo_data_list, ignore_index=True)
        
        current_sensor_readings = {}
        for sensor_id in SENSORS.keys():
            actual_col = f'sensor_{sensor_id}'
            if actual_col in row and not pd.isna(row[actual_col]):
                current_sensor_readings[sensor_id] = row[actual_col]
            else:
                current_sensor_readings[sensor_id] = 0.0
        
        nn2_unclamped, nn2_clamped = apply_nn2_correction(
            nn2, scalers, sensor_coords_spatial, pinn_values, combined_meteo, 
            timestamp, current_sensor_readings, return_unclamped=True
        )
        
        for sensor_id in SENSORS.keys():
            actual_col = f'sensor_{sensor_id}'
            if actual_col not in row or pd.isna(row[actual_col]):
                continue
            
            results.append({
                'timestamp': timestamp,
                'sensor_id': sensor_id,
                'actual': row[actual_col],
                'pinn': pinn_values[sensor_id],
                'nn2_unclamped': nn2_unclamped[sensor_id],
                'nn2_clamped': nn2_clamped[sensor_id],
                'was_negative': nn2_unclamped[sensor_id] < 0
            })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    negative_count = results_df['was_negative'].sum()
    total_count = len(results_df)
    
    print(f"\nTotal predictions analyzed: {total_count}")
    print(f"Negative predictions (before clamping): {negative_count} ({negative_count/total_count*100:.1f}%)")
    print()
    
    negative_df = results_df[results_df['was_negative']]
    
    if len(negative_df) > 0:
        print("Negative Prediction Analysis:")
        print(f"  Min negative value: {negative_df['nn2_unclamped'].min():.4f} ppb")
        print(f"  Max negative value: {negative_df['nn2_unclamped'].max():.4f} ppb")
        print(f"  Mean negative value: {negative_df['nn2_unclamped'].mean():.4f} ppb")
        print(f"  Median negative value: {negative_df['nn2_unclamped'].median():.4f} ppb")
        print()
        
        print("When predictions were negative:")
        print(f"  Average PINN: {negative_df['pinn'].mean():.4f} ppb")
        print(f"  Average actual: {negative_df['actual'].mean():.4f} ppb")
        print(f"  Median PINN: {negative_df['pinn'].median():.4f} ppb")
        print(f"  Median actual: {negative_df['actual'].median():.4f} ppb")
        print()
        
        # Check if these are overcorrections
        overcorrections = negative_df[(negative_df['pinn'] > 0) | (negative_df['actual'] > 0)]
        legitimate = negative_df[(negative_df['pinn'] == 0) & (negative_df['actual'] == 0)]
        
        print(f"Overcorrections (PINN>0 or actual>0): {len(overcorrections)} ({len(overcorrections)/len(negative_df)*100:.1f}% of negatives)")
        if len(overcorrections) > 0:
            print(f"  Average PINN: {overcorrections['pinn'].mean():.4f} ppb")
            print(f"  Average actual: {overcorrections['actual'].mean():.4f} ppb")
            print(f"  Max PINN: {overcorrections['pinn'].max():.4f} ppb")
            print(f"  Max actual: {overcorrections['actual'].max():.4f} ppb")
        print()
        
        print(f"Legitimate zeros (PINN=0 and actual=0): {len(legitimate)} ({len(legitimate)/len(negative_df)*100:.1f}% of negatives)")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

