"""
Investigate why Gradient Boosting mapping works on 2019 but not 2021
Compare distributions of NN2 scaled outputs, PINN predictions, and actual sensor readings
"""

import sys
from pathlib import Path
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from pinn import ParametricADEPINN
from nn2 import NN2_CorrectionNetwork
from nn2_mapping_utils import load_nn2_mapping_model, nn2_scaled_to_ppb
import matplotlib.pyplot as plt

# Paths
PROJECT_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime')
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
NN2_MODEL_PATH = PROJECT_DIR / 'nn2_timefix' / 'nn2_master_model_spatial-3.pth'
NN2_SCALERS_PATH = PROJECT_DIR / 'nn2_timefix' / 'nn2_master_scalers-2.pkl'

# Sensor locations (EXACT from training data generation)
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

UNIT_CONVERSION_FACTOR = 313210039.9

def load_models():
    """Load all models"""
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
    
    # Load NN2
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    # Load scalers
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    # Load mapping model
    mapping_data = load_nn2_mapping_model()
    nn2_mapping_model = mapping_data['model']
    
    print("  ✓ Models loaded")
    return pinn, nn2, scalers, nn2_mapping_model

def compute_pinn_predictions(pinn, facility_data, timestamp):
    """Compute PINN predictions at sensor locations for a given timestamp"""
    # Get met data from 3 hours before (for 3-hour forecast)
    input_timestamp = timestamp - pd.Timedelta(hours=3)
    
    sensor_predictions = {}
    for sensor_id, (sx, sy) in SENSORS.items():
        total_concentration = 0.0
        
        # Sum contributions from all facilities
        for facility_name, df in facility_data.items():
            # Find closest timestamp in facility data
            df['time_diff'] = (df['t'] - input_timestamp).abs()
            closest_row = df.loc[df['time_diff'].idxmin()]
            
            if closest_row['time_diff'] > pd.Timedelta(hours=1):
                continue  # Skip if too far from timestamp
            
            cx = closest_row['source_x_cartesian']
            cy = closest_row['source_y_cartesian']
            u = closest_row['wind_u']
            v = closest_row['wind_v']
            # Check column name (2019 uses 'source_diameter', 2021 might use 'source_height' or 'd')
            d = closest_row.get('source_diameter', closest_row.get('source_height', closest_row.get('d', 0.0)))
            kappa = closest_row.get('D', closest_row.get('kappa', 0.0))
            Q = closest_row.get('Q_total', closest_row.get('emission_rate', closest_row.get('Q', 0.0)))
            
            # Compute PINN at sensor location with simulation time t=3.0 hours
            t_hours_forecast = 3.0
            
            with torch.no_grad():
                phi_raw = pinn(
                    torch.tensor([[sx]], dtype=torch.float32),
                    torch.tensor([[sy]], dtype=torch.float32),
                    torch.tensor([[t_hours_forecast]], dtype=torch.float32),
                    torch.tensor([[cx]], dtype=torch.float32),
                    torch.tensor([[cy]], dtype=torch.float32),
                    torch.tensor([[u]], dtype=torch.float32),
                    torch.tensor([[v]], dtype=torch.float32),
                    torch.tensor([[d]], dtype=torch.float32),
                    torch.tensor([[kappa]], dtype=torch.float32),
                    torch.tensor([[Q]], dtype=torch.float32),
                    normalize=True
                )
                concentration_ppb = phi_raw.item() * UNIT_CONVERSION_FACTOR
                total_concentration += concentration_ppb
        
        sensor_predictions[sensor_id] = total_concentration
    
    return sensor_predictions

def apply_nn2_and_collect_stats(nn2, scalers, nn2_mapping_model, pinn_values, meteo_data, timestamp, current_sensor_readings):
    """Apply NN2 and collect statistics on scaled outputs"""
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
    
    # Scale inputs (handle zeros)
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
    c_s = scalers['coords'].transform(np.array([SENSORS[k] for k in sensor_ids_sorted]))
    
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
    
    corrected_scaled_np = corrected_scaled.cpu().numpy().flatten()
    
    # Collect statistics
    stats = {
        'nn2_scaled_outputs': corrected_scaled_np.copy(),
        'pinn_values': pinn_array.copy(),
        'current_sensors': current_sensors.copy(),
        'nn2_scaled_min': corrected_scaled_np.min(),
        'nn2_scaled_max': corrected_scaled_np.max(),
        'nn2_scaled_mean': corrected_scaled_np.mean(),
        'nn2_scaled_std': corrected_scaled_np.std(),
        'nn2_scaled_nonzero': corrected_scaled_np[np.abs(corrected_scaled_np) > 1e-6],
    }
    
    return stats

def analyze_year(year, pinn, nn2, scalers, nn2_mapping_model):
    """Analyze a specific year and collect statistics"""
    print(f"\n{'='*80}")
    print(f"ANALYZING {year}")
    print(f"{'='*80}")
    
    # Load data based on year
    if year == 2019:
        # Load 2019 data
        FACILITY_DATA_BASE = PROJECT_DIR / 'simpletesting' / 'nn2trainingdata'
        SENSOR_DATA_PATH = PROJECT_DIR / 'simpletesting' / 'nn2trainingdata' / 'sensors_final_synced.csv'
        
        # Load facility data
        facility_files = sorted(FACILITY_DATA_BASE.glob('*_synced_training_data.csv'))
        facility_files = [f for f in facility_files if 'summary' not in f.name]
        
        facility_data = {}
        for f in facility_files:
            df = pd.read_csv(f)
            df['t'] = pd.to_datetime(df['t'])
            facility_name = f.stem.replace('_synced_training_data', '')
            facility_data[facility_name] = df
        
        # Load sensor data
        sensor_df = pd.read_csv(SENSOR_DATA_PATH)
        # 2019 data uses 't' column, not 'timestamp'
        if 't' in sensor_df.columns:
            sensor_df['timestamp'] = pd.to_datetime(sensor_df['t'])
        elif 'timestamp' in sensor_df.columns:
            sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
        else:
            raise ValueError(f"Could not find timestamp column. Columns: {sensor_df.columns.tolist()}")
        
        # Sample timestamps (every 10th to speed up)
        timestamps = sensor_df['timestamp'].unique()[::10]
        
    elif year == 2021:
        # Load 2021 data
        FACILITY_DATA_BASE = PROJECT_DIR / 'realtime_processing' / 'houston_processed_2021'
        SENSOR_DATA_PATH = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/validation_results/validation_jan_mar_2021_detailed.csv')
        
        # Load facility data for Jan-Mar 2021
        facility_data = {}
        for month in ['jan', 'feb', 'mar']:
            month_dir = FACILITY_DATA_BASE / month
            if month_dir.exists():
                for facility_file in month_dir.glob('*.csv'):
                    facility_name = facility_file.stem
                    if facility_name not in facility_data:
                        df = pd.read_csv(facility_file)
                        df['t'] = pd.to_datetime(df['t'])
                        facility_data[facility_name] = df
                    else:
                        df = pd.read_csv(facility_file)
                        df['t'] = pd.to_datetime(df['t'])
                        facility_data[facility_name] = pd.concat([facility_data[facility_name], df])
        
        # Load sensor data from validation results
        sensor_df = pd.read_csv(SENSOR_DATA_PATH)
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
        timestamps = sensor_df['timestamp'].unique()[::10]
    
    # Collect statistics
    all_nn2_scaled = []
    all_pinn_values = []
    all_sensor_values = []
    
    print(f"Processing {len(timestamps)} timestamps...")
    for timestamp in tqdm(timestamps[:500]):  # Limit to 500 for speed
        # Get sensor readings
        sensor_rows = sensor_df[sensor_df['timestamp'] == timestamp]
        if len(sensor_rows) == 0:
            continue
        sensor_row = sensor_rows.iloc[0]
        current_sensor_readings = {sid: sensor_row.get(f'sensor_{sid}', 0.0) for sid in SENSORS.keys()}
        
        # Compute PINN predictions
        if year == 2021:
            # For 2021, use month-specific facility data
            month_key = timestamp.strftime('%b').lower()
            month_facility_data = {
                name.replace(f"{month_key}_", ""): df 
                for name, df in facility_data.items() 
                if name.startswith(f"{month_key}_")
            }
            pinn_values = compute_pinn_predictions(pinn, month_facility_data, timestamp)
        else:
            pinn_values = compute_pinn_predictions(pinn, facility_data, timestamp)
        
        # Get meteo data (for 2021, need to match month_key)
        if year == 2021:
            # Extract month key from timestamp
            month_key = timestamp.strftime('%b').lower()
            month_facility_data = {
                name.replace(f"{month_key}_", ""): df 
                for name, df in facility_data.items() 
                if name.startswith(f"{month_key}_")
            }
            if len(month_facility_data) == 0:
                continue
            
            meteo_data_list = []
            input_timestamp = timestamp - pd.Timedelta(hours=3)
            for facility_name, facility_df in month_facility_data.items():
                facility_data_filtered = facility_df[facility_df['t'] == input_timestamp]
                if len(facility_data_filtered) > 0:
                    meteo_data_list.append(facility_data_filtered)
            
            if len(meteo_data_list) == 0:
                continue
            
            combined_meteo = pd.concat(meteo_data_list, ignore_index=True)
            meteo_df = combined_meteo[['wind_u', 'wind_v', 'D']]
        else:
            # 2019 approach
            meteo_data = []
            for facility_name, df in facility_data.items():
                input_timestamp = timestamp - pd.Timedelta(hours=3)
                df['time_diff'] = (df['t'] - input_timestamp).abs()
                closest_row = df.loc[df['time_diff'].idxmin()]
                if closest_row['time_diff'] <= pd.Timedelta(hours=1):
                    meteo_data.append({
                        'wind_u': closest_row['wind_u'],
                        'wind_v': closest_row['wind_v'],
                        'D': closest_row['D']
                    })
            
            if not meteo_data:
                continue
            
            meteo_df = pd.DataFrame(meteo_data)
        
        # Apply NN2 and collect stats
        stats = apply_nn2_and_collect_stats(
            nn2, scalers, nn2_mapping_model, pinn_values, meteo_df, timestamp, current_sensor_readings
        )
        
        all_nn2_scaled.extend(stats['nn2_scaled_outputs'])
        all_pinn_values.extend(stats['pinn_values'])
        all_sensor_values.extend(stats['current_sensors'])
    
    # Convert to arrays
    all_nn2_scaled = np.array(all_nn2_scaled)
    all_pinn_values = np.array(all_pinn_values)
    all_sensor_values = np.array(all_sensor_values)
    
    # Filter non-zero
    nn2_nonzero = all_nn2_scaled[np.abs(all_nn2_scaled) > 1e-6]
    pinn_nonzero = all_pinn_values[all_pinn_values > 1e-6]
    sensor_nonzero = all_sensor_values[all_sensor_values > 1e-6]
    
    # Print statistics
    print(f"\nNN2 Scaled Outputs Statistics ({year}):")
    if len(all_nn2_scaled) > 0:
        print(f"  All values: min={all_nn2_scaled.min():.4f}, max={all_nn2_scaled.max():.4f}, mean={all_nn2_scaled.mean():.4f}, std={all_nn2_scaled.std():.4f}")
    else:
        print(f"  All values: No data collected")
    if len(nn2_nonzero) > 0:
        print(f"  Non-zero values: min={nn2_nonzero.min():.4f}, max={nn2_nonzero.max():.4f}, mean={nn2_nonzero.mean():.4f}, std={nn2_nonzero.std():.4f}")
        print(f"  Non-zero count: {len(nn2_nonzero)} / {len(all_nn2_scaled)} ({100*len(nn2_nonzero)/len(all_nn2_scaled):.1f}%)")
    else:
        print(f"  Non-zero values: None")
    
    print(f"\nPINN Predictions Statistics ({year}):")
    if len(all_pinn_values) > 0:
        print(f"  All values: min={all_pinn_values.min():.4f}, max={all_pinn_values.max():.4f}, mean={all_pinn_values.mean():.4f}, std={all_pinn_values.std():.4f}")
    else:
        print(f"  All values: No data collected")
    if len(pinn_nonzero) > 0:
        print(f"  Non-zero values: min={pinn_nonzero.min():.4f}, max={pinn_nonzero.max():.4f}, mean={pinn_nonzero.mean():.4f}, std={pinn_nonzero.std():.4f}")
    else:
        print(f"  Non-zero values: None")
    
    print(f"\nSensor Readings Statistics ({year}):")
    if len(all_sensor_values) > 0:
        print(f"  All values: min={all_sensor_values.min():.4f}, max={all_sensor_values.max():.4f}, mean={all_sensor_values.mean():.4f}, std={all_sensor_values.std():.4f}")
    else:
        print(f"  All values: No data collected")
    if len(sensor_nonzero) > 0:
        print(f"  Non-zero values: min={sensor_nonzero.min():.4f}, max={sensor_nonzero.max():.4f}, mean={sensor_nonzero.mean():.4f}, std={sensor_nonzero.std():.4f}")
    else:
        print(f"  Non-zero values: None")
    
    return {
        'nn2_scaled': all_nn2_scaled,
        'nn2_nonzero': nn2_nonzero,
        'pinn_values': all_pinn_values,
        'pinn_nonzero': pinn_nonzero,
        'sensor_values': all_sensor_values,
        'sensor_nonzero': sensor_nonzero,
    }

def main():
    print("="*80)
    print("INVESTIGATING 2019 vs 2021 DISTRIBUTION DIFFERENCES")
    print("="*80)
    
    # Load models
    pinn, nn2, scalers, nn2_mapping_model = load_models()
    
    # Analyze both years
    stats_2019 = analyze_year(2019, pinn, nn2, scalers, nn2_mapping_model)
    stats_2021 = analyze_year(2021, pinn, nn2, scalers, nn2_mapping_model)
    
    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON: 2019 vs 2021")
    print(f"{'='*80}")
    
    print("\nNN2 Scaled Outputs (Non-zero):")
    if len(stats_2019['nn2_nonzero']) > 0:
        print(f"  2019: min={stats_2019['nn2_nonzero'].min():.4f}, max={stats_2019['nn2_nonzero'].max():.4f}, mean={stats_2019['nn2_nonzero'].mean():.4f}, std={stats_2019['nn2_nonzero'].std():.4f}")
    else:
        print(f"  2019: No data")
    if len(stats_2021['nn2_nonzero']) > 0:
        print(f"  2021: min={stats_2021['nn2_nonzero'].min():.4f}, max={stats_2021['nn2_nonzero'].max():.4f}, mean={stats_2021['nn2_nonzero'].mean():.4f}, std={stats_2021['nn2_nonzero'].std():.4f}")
        if len(stats_2019['nn2_nonzero']) > 0:
            print(f"  Range difference: 2019 spans {stats_2019['nn2_nonzero'].max() - stats_2019['nn2_nonzero'].min():.4f}, 2021 spans {stats_2021['nn2_nonzero'].max() - stats_2021['nn2_nonzero'].min():.4f}")
    else:
        print(f"  2021: No data")
    
    print("\nPINN Predictions (Non-zero):")
    if len(stats_2019['pinn_nonzero']) > 0:
        print(f"  2019: min={stats_2019['pinn_nonzero'].min():.4f}, max={stats_2019['pinn_nonzero'].max():.4f}, mean={stats_2019['pinn_nonzero'].mean():.4f}, std={stats_2019['pinn_nonzero'].std():.4f}")
    else:
        print(f"  2019: No data")
    if len(stats_2021['pinn_nonzero']) > 0:
        print(f"  2021: min={stats_2021['pinn_nonzero'].min():.4f}, max={stats_2021['pinn_nonzero'].max():.4f}, mean={stats_2021['pinn_nonzero'].mean():.4f}, std={stats_2021['pinn_nonzero'].std():.4f}")
    else:
        print(f"  2021: No data")
    
    print("\nSensor Readings (Non-zero):")
    if len(stats_2019['sensor_nonzero']) > 0:
        print(f"  2019: min={stats_2019['sensor_nonzero'].min():.4f}, max={stats_2019['sensor_nonzero'].max():.4f}, mean={stats_2019['sensor_nonzero'].mean():.4f}, std={stats_2019['sensor_nonzero'].std():.4f}")
    else:
        print(f"  2019: No data")
    if len(stats_2021['sensor_nonzero']) > 0:
        print(f"  2021: min={stats_2021['sensor_nonzero'].min():.4f}, max={stats_2021['sensor_nonzero'].max():.4f}, mean={stats_2021['sensor_nonzero'].mean():.4f}, std={stats_2021['sensor_nonzero'].std():.4f}")
    else:
        print(f"  2021: No data")
    
    # Check mapping model training range
    print("\n" + "="*80)
    print("MAPPING MODEL TRAINING RANGE")
    print("="*80)
    # The mapping model was trained on 2019 training data
    # We need to check what range it was trained on
    print("Mapping model was trained on 2019 training data")
    print(f"2019 NN2 scaled outputs range: [{stats_2019['nn2_nonzero'].min():.4f}, {stats_2019['nn2_nonzero'].max():.4f}]")
    print(f"2021 NN2 scaled outputs range: [{stats_2021['nn2_nonzero'].min():.4f}, {stats_2021['nn2_nonzero'].max():.4f}]")
    
    # Check if 2021 values are outside 2019 range
    if len(stats_2019['nn2_nonzero']) > 0 and len(stats_2021['nn2_nonzero']) > 0:
        outside_range = (stats_2021['nn2_nonzero'] < stats_2019['nn2_nonzero'].min()) | (stats_2021['nn2_nonzero'] > stats_2019['nn2_nonzero'].max())
        pct_outside = 100 * outside_range.sum() / len(stats_2021['nn2_nonzero'])
        print(f"\n2021 values outside 2019 range: {outside_range.sum()} / {len(stats_2021['nn2_nonzero'])} ({pct_outside:.1f}%)")
        
        if pct_outside > 10:
            print("⚠ WARNING: Significant portion of 2021 values are outside 2019 training range!")
            print("   This could explain why the mapping model performs poorly on 2021 data.")
    else:
        print("\n⚠ Cannot compare ranges: missing data for one or both years")

if __name__ == '__main__':
    main()

