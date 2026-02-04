"""
Kalman Filter Validation for Jan-Mar 2021
Compute MAE for Kalman filter on 2021 data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import sys

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

from kalman_filter import BenzeneKalmanFilter
from pinn import ParametricADEPINN
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sensor IDs in order
SENSOR_IDS = [
    '482010026', '482010057', '482010069',
    '482010617', '482010803', '482011015',
    '482011035', '482011039', '482016000'
]

UNIT_CONVERSION_FACTOR = 313210039.9
FORECAST_T_HOURS = 3.0

# 2021 sensor data paths (from validate_jan_mar_2021.py)
MADIS_DIR = Path('/Users/neevpratap/Desktop/madis_data_desktop_updated')
SENSOR_DATA_PATHS = {
    'january': MADIS_DIR / "results_2021/sensors_actual_wide_2021_full_jan.csv",
    'february': MADIS_DIR / "results_2021/sensors_actual_wide_2021_full_feb.csv",
    'march': MADIS_DIR / "results_2021/sensors_actual_wide_2021_full_march.csv",
}

# Facility data paths
FACILITY_DATA_BASE = MADIS_DIR
FACILITY_DATA_PATHS = {
    'january': FACILITY_DATA_BASE / 'training_data_2021_full_jan_REPAIRED',
    'february': FACILITY_DATA_BASE / 'training_data_2021_feb_REPAIRED',
    'march': FACILITY_DATA_BASE / 'training_data_2021_march_REPAIRED',
}


def load_kalman_parameters() -> dict:
    """Load optimized Kalman parameters."""
    param_file = Path("realtime/data/kalman_parameters.json")
    
    if not param_file.exists():
        logger.warning("No tuned parameters found, using defaults")
        return {
            'process_noise': 1.0,
            'measurement_noise': 0.01,
            'decay_rate': 0.7,
            'pinn_weight': 0.3
        }
    
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    # Filter to only parameters
    params = {k: v for k, v in params.items() 
              if k in ['process_noise', 'measurement_noise', 'decay_rate', 'pinn_weight']}
    
    logger.info(f"Loaded parameters: {params}")
    return params


def load_pinn_model():
    """Load PINN model."""
    pinn_path = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
    
    pinn = ParametricADEPINN()
    checkpoint = torch.load(pinn_path, map_location='cpu', weights_only=False)
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
    return pinn


def load_facility_data(month_key):
    """Load facility data for a month."""
    facility_dir = FACILITY_DATA_PATHS[month_key]
    
    if not facility_dir.exists():
        logger.warning(f"Facility directory not found: {facility_dir}")
        return {}
    
    # Try different file patterns
    facility_files = sorted(facility_dir.glob('*_training_data.csv'))
    if len(facility_files) == 0:
        facility_files = sorted(facility_dir.glob('*.csv'))
        facility_files = [f for f in facility_files if 'summary' not in f.name and 'total' not in f.name.lower()]
    
    facility_data_dict = {}
    for facility_file in facility_files:
        facility_name = facility_file.stem.replace('_training_data', '').replace('_synced', '')
        try:
            df = pd.read_csv(facility_file)
            
            if 't' in df.columns:
                df = df.rename(columns={'t': 'timestamp'})
            elif 'timestamp' not in df.columns:
                logger.warning(f"  Warning: No timestamp column in {facility_file.name}")
                continue
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Check for required columns
            required_cols = ['source_x_cartesian', 'source_y_cartesian', 'source_diameter', 'Q_total', 'wind_u', 'wind_v', 'D']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"  Warning: Missing required columns in {facility_file.name}")
                continue
            
            facility_data_dict[facility_name] = df
        except Exception as e:
            logger.warning(f"  Error loading {facility_file.name}: {e}")
            continue
    
    logger.info(f"  Loaded {len(facility_data_dict)} facilities from {month_key}")
    return facility_data_dict


def predict_pinn_at_sensors(pinn, facility_data_dict, timestamp):
    """Predict PINN at sensors."""
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
    
    sensor_pinn_ppb = {sid: 0.0 for sid in SENSORS.keys()}
    
    for facility_name, facility_df in facility_data_dict.items():
        # Try exact match first
        facility_data = facility_df[facility_df['timestamp'] == timestamp]
        
        # If no exact match, try closest within 30 minutes
        if len(facility_data) == 0:
            time_diffs = (facility_df['timestamp'] - timestamp).abs()
            if len(time_diffs) > 0 and time_diffs.min() <= pd.Timedelta(minutes=30):
                closest_idx = time_diffs.idxmin()
                facility_data = facility_df.iloc[[closest_idx]]
        
        if len(facility_data) == 0:
            continue
        
        for _, row in facility_data.iterrows():
            cx = row['source_x_cartesian']
            cy = row['source_y_cartesian']
            d = row['source_diameter']
            Q = row['Q_total']
            u = row['wind_u']
            v = row['wind_v']
            kappa = row['D']
            
            for sensor_id, (sx, sy) in SENSORS.items():
                with torch.no_grad():
                    phi_raw = pinn(
                        torch.tensor([[sx]], dtype=torch.float32),
                        torch.tensor([[sy]], dtype=torch.float32),
                        torch.tensor([[FORECAST_T_HOURS]], dtype=torch.float32),
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
                    sensor_pinn_ppb[sensor_id] += concentration_ppb
    
    return np.array([sensor_pinn_ppb[sid] for sid in SENSOR_IDS])


def main():
    """Run validation on Jan-Mar 2021."""
    logger.info("Loading Kalman parameters...")
    kf_params = load_kalman_parameters()
    
    logger.info("Loading PINN model...")
    pinn = load_pinn_model()
    
    sensor_cols = [f'sensor_{sid}' for sid in SENSOR_IDS]
    
    all_errors_pinn = []
    all_errors_kalman = []
    
    # Process each month
    for month_key in ['january', 'february', 'march']:
        # Initialize Kalman filter fresh for each month
        kf = BenzeneKalmanFilter(**kf_params)
        logger.info(f"\nProcessing {month_key}...")
        
        # Load sensor data
        sensor_path = SENSOR_DATA_PATHS[month_key]
        if not sensor_path.exists():
            logger.warning(f"Sensor data not found: {sensor_path}")
            continue
        
        sensor_df = pd.read_csv(sensor_path)
        if 't' in sensor_df.columns:
            sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
        sensor_df = sensor_df.sort_values('timestamp').reset_index(drop=True)
        
        # Load facility data
        facility_data_dict = load_facility_data(month_key)
        if len(facility_data_dict) == 0:
            logger.warning(f"No facility data for {month_key}")
            continue
        
        logger.info(f"  Sensor data: {len(sensor_df)} timestamps")
        logger.info(f"  First timestamp: {sensor_df.iloc[0]['timestamp']}, Last: {sensor_df.iloc[-1]['timestamp']}")
        
        # Check facility data timestamps
        if len(facility_data_dict) > 0:
            sample_fac = list(facility_data_dict.values())[0]
            logger.info(f"  Facility data: {len(sample_fac)} timestamps")
            logger.info(f"  Facility first: {sample_fac['timestamp'].min()}, last: {sample_fac['timestamp'].max()}")
        
        # Process each timestamp pair
        valid_pairs = 0
        processed = 0
        pinn_failures = 0
        kalman_failures = 0
        no_valid_mask = 0
        
        for i in range(len(sensor_df) - 3):
            current_row = sensor_df.iloc[i]
            current_time = current_row['timestamp']
            
            # Look ahead 3 hours for the target
            target_row = sensor_df.iloc[i + 3]
            target_time = target_row['timestamp']
            
            # Verify it's 3 hours ahead
            target_diff = (target_time - current_time).total_seconds() / 3600
            if not (2.5 <= target_diff <= 3.5):
                continue
            
            processed += 1
            
            # Get sensor readings
            current_sensors_raw = current_row[sensor_cols].values.astype(float)
            future_actual_raw = target_row[sensor_cols].values.astype(float)
            
            # Find valid sensors (not NaN)
            valid_current_mask = ~np.isnan(current_sensors_raw)
            valid_future_mask = ~np.isnan(future_actual_raw)
            valid_sensors_mask = valid_current_mask & valid_future_mask
            
            # Need at least 5 valid sensors
            if valid_sensors_mask.sum() < 5:
                continue
            
            # Replace NaN with 0 for Kalman filter (it handles missing sensors)
            # But keep track of which are actually valid
            current_sensors = np.nan_to_num(current_sensors_raw, nan=0.0)
            future_actual = np.nan_to_num(future_actual_raw, nan=0.0)
            
            # Ensure all values are finite
            if np.any(~np.isfinite(current_sensors)) or np.any(~np.isfinite(future_actual)):
                continue
            
            # PINN prediction
            try:
                pinn_predictions = predict_pinn_at_sensors(pinn, facility_data_dict, current_time)
                
                # Check for invalid PINN predictions
                if np.any(~np.isfinite(pinn_predictions)):
                    pinn_failures += 1
                    if processed < 5:
                        logger.debug(f"PINN produced invalid values for {current_time}")
                    continue
                
                valid_pairs += 1
            except Exception as e:
                pinn_failures += 1
                if processed < 5:  # Debug first few
                    logger.debug(f"PINN failed for {current_time}: {e}")
                continue
            
            # Kalman forecast
            try:
                kalman_forecast, _ = kf.forecast(
                    current_sensors=current_sensors,
                    pinn_predictions=pinn_predictions,
                    hours_ahead=3,
                    return_uncertainty=False
                )
                
                # Check for invalid values
                if np.any(~np.isfinite(kalman_forecast)):
                    kalman_failures += 1
                    if processed < 5:
                        logger.debug(f"Kalman produced invalid values for {current_time}")
                    continue
            except Exception as e:
                kalman_failures += 1
                if processed < 5:
                    logger.debug(f"Kalman failed for {current_time}: {e}")
                continue
            
            # Calculate errors (only for valid sensors - use the mask we computed earlier)
            if valid_sensors_mask.sum() >= 5:
                errors_pinn = np.abs(future_actual[valid_sensors_mask] - pinn_predictions[valid_sensors_mask])
                errors_kalman = np.abs(future_actual[valid_sensors_mask] - kalman_forecast[valid_sensors_mask])
                
                all_errors_pinn.extend(errors_pinn)
                all_errors_kalman.extend(errors_kalman)
            else:
                no_valid_mask += 1
    
        # Log progress
        if processed % 100 == 0 and processed > 0:
            logger.info(f"  Progress: {processed} pairs processed, {valid_pairs} valid PINN, {len(all_errors_pinn)} errors collected")
    
    # Log summary for month
    logger.info(f"  Month summary: {processed} pairs, {valid_pairs} valid PINN, {pinn_failures} PINN failures, "
                f"{kalman_failures} Kalman failures, {no_valid_mask} insufficient valid sensors")
    
    # Calculate overall MAE
    if len(all_errors_pinn) > 0:
        # Filter out any NaN or inf values
        errors_pinn_clean = [e for e in all_errors_pinn if np.isfinite(e)]
        errors_kalman_clean = [e for e in all_errors_kalman if np.isfinite(e)]
        
        if len(errors_pinn_clean) > 0 and len(errors_kalman_clean) > 0:
            pinn_mae = np.mean(errors_pinn_clean)
            kalman_mae = np.mean(errors_kalman_clean)
            improvement = (pinn_mae - kalman_mae) / pinn_mae * 100 if pinn_mae > 0 else 0.0
            
            print("\n" + "="*70)
            print("KALMAN FILTER VALIDATION - JANUARY-MARCH 2021")
            print("="*70)
            print(f"\nTotal samples: {len(errors_pinn_clean):,}")
            print(f"PINN MAE:      {pinn_mae:.4f} ppb")
            print(f"Kalman MAE:    {kalman_mae:.4f} ppb")
            print(f"Improvement:   {improvement:+.1f}%")
            print("="*70)
        else:
            print(f"Error: {len(errors_pinn_clean)} valid PINN errors, {len(errors_kalman_clean)} valid Kalman errors")
    else:
        print("No valid samples found!")


if __name__ == "__main__":
    main()

