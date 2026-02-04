"""
Kalman Filter Parameter Tuning
Optimize Q, R, decay_rate, and pinn_weight using 2019 validation data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, Tuple
from tqdm import tqdm
import sys

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

from kalman_filter import BenzeneKalmanFilter
from concentration_predictor import ConcentrationPredictor
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

UNIT_CONVERSION_FACTOR = 313210039.9  # kg/m^2 to ppb
FORECAST_T_HOURS = 3.0  # Simulation time


def load_validation_data(year: int = 2019) -> pd.DataFrame:
    """
    Load EDF sensor data for validation.
    
    Returns:
        DataFrame with columns:
            - timestamp (or 't')
            - sensor_482010026, ..., sensor_482016000 (actual EDF readings)
    """
    # Try multiple possible locations
    sensor_files = [
        Path("realtime/simpletesting/nn2trainingdata/sensors_final_synced.csv"),
        Path("realtime/drive-download-20260202T042428Z-3-001/sensors_final_synced.csv"),
        Path("simpletesting/nn2trainingdata/sensors_final_synced.csv"),
        Path("/Users/neevpratap/Downloads/sensors_final_synced.csv"),
    ]
    
    sensor_file = None
    for f in sensor_files:
        if f.exists():
            sensor_file = f
            break
    
    if sensor_file is None:
        raise FileNotFoundError("Could not find sensors_final_synced.csv in any expected location")
    
    df = pd.read_csv(sensor_file)
    
    # Handle timestamp column name
    if 't' in df.columns:
        df = df.rename(columns={'t': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter by year
    df = df[df['timestamp'].dt.year == year].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} timestamps from {year}")
    return df


def load_facility_data(year: int = 2019):
    """
    Load facility data for PINN predictions.
    
    Returns:
        Dictionary mapping facility names to DataFrames
    """
    # Try to find facility data directory
    facility_dirs = [
        Path("realtime/simpletesting/nn2trainingdata"),
        Path("simpletesting/nn2trainingdata"),
    ]
    
    facility_dir = None
    for d in facility_dirs:
        if d.exists():
            facility_dir = d
            break
    
    if facility_dir is None:
        raise FileNotFoundError("Could not find facility data directory")
    
    facility_files = sorted(facility_dir.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    facility_data_dict = {}
    
    for facility_file in facility_files:
        facility_name = facility_file.stem.replace('_synced_training_data', '')
        df = pd.read_csv(facility_file)
        
        # Handle timestamp column
        if 't' in df.columns:
            df = df.rename(columns={'t': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by year
        df = df[df['timestamp'].dt.year == year].copy()
        
        if len(df) > 0:
            facility_data_dict[facility_name] = df
    
    logger.info(f"Loaded {len(facility_data_dict)} facilities")
    return facility_data_dict


def predict_pinn_at_sensors(pinn, facility_data_dict, timestamp):
    """
    Predict PINN concentrations at all sensor locations for a given timestamp.
    Matches training data generation method exactly.
    """
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
    
    # Initialize sensor concentrations
    sensor_pinn_ppb = {sid: 0.0 for sid in SENSORS.keys()}
    
    # Process each facility separately (EXACTLY like training data generation)
    for facility_name, facility_df in facility_data_dict.items():
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
                    
                    # Convert to ppb and superimpose
                    concentration_ppb = phi_raw.item() * UNIT_CONVERSION_FACTOR
                    sensor_pinn_ppb[sensor_id] += concentration_ppb
    
    # Return as array in sensor ID order
    return np.array([sensor_pinn_ppb[sid] for sid in SENSOR_IDS])


def load_pinn_model():
    """Load PINN model with correct normalization ranges."""
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


def evaluate_kalman_parameters(
    process_noise: float,
    measurement_noise: float,
    decay_rate: float,
    pinn_weight: float,
    validation_data: pd.DataFrame,
    facility_data_dict: dict,
    pinn
) -> Tuple[float, float, float]:
    """
    Evaluate Kalman filter with given parameters on validation data.
    
    Returns:
        mae: Mean Absolute Error [ppb]
        rmse: Root Mean Squared Error [ppb]
        detection_rate: Fraction of exceedances (>10 ppb) detected
    """
    # Initialize Kalman filter
    kf = BenzeneKalmanFilter(
        n_sensors=9,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        decay_rate=decay_rate,
        pinn_weight=pinn_weight
    )
    
    sensor_cols = [f'sensor_{sid}' for sid in SENSOR_IDS]
    
    errors = []
    detections = []
    
    # Process each timestamp (need T and T+3 pairs)
    for i in range(len(validation_data) - 1):
        current_row = validation_data.iloc[i]
        future_row = validation_data.iloc[i + 1]
        
        current_time = current_row['timestamp']
        future_time = future_row['timestamp']
        
        # Check if this is a valid 3-hour pair (allowing some tolerance)
        time_diff = (future_time - current_time).total_seconds() / 3600
        if not (2.5 <= time_diff <= 3.5):
            continue
        
        # Current sensor readings (time T)
        current_sensors = current_row[sensor_cols].values.astype(float)
        
        # Future actual readings (time T+3)
        future_actual = future_row[sensor_cols].values.astype(float)
        
        # Get PINN prediction for T+3 (using met data from T)
        try:
            pinn_predictions = predict_pinn_at_sensors(pinn, facility_data_dict, current_time)
        except Exception as e:
            logger.warning(f"PINN prediction failed for {current_time}: {e}")
            continue
        
        # Kalman forecast
        kalman_forecast, uncertainty = kf.forecast(
            current_sensors=current_sensors,
            pinn_predictions=pinn_predictions,
            hours_ahead=3,
            return_uncertainty=True
        )
        
        # Calculate errors (only for valid sensors)
        valid_mask = ~np.isnan(future_actual) & ~np.isnan(kalman_forecast)
        
        if valid_mask.any():
            error = np.abs(kalman_forecast[valid_mask] - future_actual[valid_mask])
            errors.extend(error)
            
            # Detection: did we predict exceedance correctly?
            exceedance_actual = future_actual[valid_mask] > 10.0
            lower, upper = kf.get_confidence_interval(kalman_forecast, uncertainty)
            exceedance_predicted = upper[valid_mask] > 10.0
            
            # True positive rate for exceedances
            if exceedance_actual.any():
                detected = exceedance_predicted[exceedance_actual].mean()
                detections.append(detected)
    
    # Calculate metrics
    mae = np.mean(errors) if errors else float('inf')
    rmse = np.sqrt(np.mean(np.array(errors)**2)) if errors else float('inf')
    detection_rate = np.mean(detections) if detections else 0.0
    
    return mae, rmse, detection_rate


def grid_search_parameters(
    validation_data: pd.DataFrame,
    facility_data_dict: dict,
    pinn
) -> Dict:
    """
    Grid search to find optimal Kalman filter parameters.
    
    Returns:
        best_params: Dictionary with optimal parameters
    """
    logger.info("Starting grid search for Kalman parameters...")
    
    # Define search grid
    param_grid = {
        'process_noise': [0.1, 0.5, 1.0, 2.0, 5.0],
        'measurement_noise': [0.001, 0.01, 0.05, 0.1],
        'decay_rate': [0.5, 0.6, 0.7, 0.8, 0.9],
        'pinn_weight': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    best_mae = float('inf')
    best_params = None
    results = []
    
    # Total combinations
    total = (len(param_grid['process_noise']) * 
             len(param_grid['measurement_noise']) * 
             len(param_grid['decay_rate']) * 
             len(param_grid['pinn_weight']))
    
    with tqdm(total=total, desc="Grid search") as pbar:
        for pn in param_grid['process_noise']:
            for mn in param_grid['measurement_noise']:
                for dr in param_grid['decay_rate']:
                    for pw in param_grid['pinn_weight']:
                        mae, rmse, det_rate = evaluate_kalman_parameters(
                            process_noise=pn,
                            measurement_noise=mn,
                            decay_rate=dr,
                            pinn_weight=pw,
                            validation_data=validation_data,
                            facility_data_dict=facility_data_dict,
                            pinn=pinn
                        )
                        
                        results.append({
                            'process_noise': pn,
                            'measurement_noise': mn,
                            'decay_rate': dr,
                            'pinn_weight': pw,
                            'mae': mae,
                            'rmse': rmse,
                            'detection_rate': det_rate
                        })
                        
                        if mae < best_mae:
                            best_mae = mae
                            best_params = {
                                'process_noise': pn,
                                'measurement_noise': mn,
                                'decay_rate': dr,
                                'pinn_weight': pw,
                                'mae': mae,
                                'rmse': rmse,
                                'detection_rate': det_rate
                            }
                            logger.info(f"New best MAE: {mae:.4f} ppb")
                        
                        pbar.update(1)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_dir = Path("realtime/data")
    output_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(output_dir / 'kalman_parameter_search.csv', index=False)
    
    logger.info(f"Grid search complete. Best MAE: {best_mae:.4f} ppb")
    logger.info(f"Best parameters: {best_params}")
    
    return best_params


def main():
    """Run parameter tuning."""
    # Load validation data
    logger.info("Loading validation data...")
    validation_data = load_validation_data(year=2019)
    
    # Load facility data
    logger.info("Loading facility data...")
    facility_data_dict = load_facility_data(year=2019)
    
    # Load PINN model
    logger.info("Loading PINN model...")
    pinn = load_pinn_model()
    
    # Run grid search
    best_params = grid_search_parameters(validation_data, facility_data_dict, pinn)
    
    # Save best parameters
    output_dir = Path("realtime/data")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "kalman_parameters.json"
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"Saved optimal parameters to {output_file}")


if __name__ == "__main__":
    main()

