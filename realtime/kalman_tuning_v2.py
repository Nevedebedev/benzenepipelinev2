"""
Kalman Filter Parameter Tuning V2 - Multi-Objective Optimization
Optimizes for detection rate (50%), MAE (20%), false alarms (20%), peak capture (10%).
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
    """Load EDF sensor data for validation."""
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
        raise FileNotFoundError("Could not find sensors_final_synced.csv")
    
    df = pd.read_csv(sensor_file)
    if 't' in df.columns:
        df = df.rename(columns={'t': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['timestamp'].dt.year == year].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def load_facility_data(year: int = 2019):
    """Load facility data for PINN predictions."""
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
        if 't' in df.columns:
            df = df.rename(columns={'t': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[df['timestamp'].dt.year == year].copy()
        if len(df) > 0:
            facility_data_dict[facility_name] = df
    
    return facility_data_dict


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


def predict_pinn_at_sensors(pinn, facility_data_dict, timestamp):
    """Predict PINN concentrations at all sensor locations."""
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
        facility_data = facility_df[facility_df['timestamp'] == timestamp]
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


def compute_metrics(
    process_noise: float,
    measurement_noise: float,
    decay_rate: float,
    pinn_weight: float,
    validation_data: pd.DataFrame,
    facility_data_dict: dict,
    pinn
) -> Tuple[float, Dict]:
    """
    Compute multi-objective metrics for Kalman filter.
    
    Returns:
        objective_score: Weighted objective score (lower is better)
        metrics: Dictionary with all metrics
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
    exceedance_actuals = []
    exceedance_predictions = []
    detections = []
    false_alarms = []
    normal_samples = 0
    
    # Process each timestamp pair
    for i in range(len(validation_data) - 1):
        current_row = validation_data.iloc[i]
        future_row = validation_data.iloc[i + 1]
        
        current_time = current_row['timestamp']
        future_time = future_row['timestamp']
        
        time_diff = (future_time - current_time).total_seconds() / 3600
        if not (2.5 <= time_diff <= 3.5):
            continue
        
        current_sensors = current_row[sensor_cols].values.astype(float)
        future_actual = future_row[sensor_cols].values.astype(float)
        
        try:
            pinn_predictions = predict_pinn_at_sensors(pinn, facility_data_dict, current_time)
        except Exception as e:
            continue
        
        kalman_forecast, uncertainty = kf.forecast(
            current_sensors=current_sensors,
            pinn_predictions=pinn_predictions,
            hours_ahead=3,
            return_uncertainty=True
        )
        
        lower, upper = kf.get_confidence_interval(kalman_forecast, uncertainty, confidence=0.95)
        
        # Calculate errors (only for valid sensors)
        valid_mask = ~np.isnan(future_actual) & ~np.isnan(kalman_forecast)
        
        if valid_mask.any():
            error = np.abs(kalman_forecast[valid_mask] - future_actual[valid_mask])
            errors.extend(error)
            
            # Exceedance analysis
            exceedance_actual = future_actual[valid_mask] > 10.0
            exceedance_predicted_upper = upper[valid_mask] > 10.0
            exceedance_predicted_point = kalman_forecast[valid_mask] > 10.0
            
            # Detection: upper bound > 10 when actual > 10
            if exceedance_actual.any():
                detected = exceedance_predicted_upper[exceedance_actual]
                detections.extend(detected)
                
                # Store actual and predicted for peak capture analysis
                exceedance_actuals.extend(future_actual[valid_mask][exceedance_actual])
                exceedance_predictions.extend(kalman_forecast[valid_mask][exceedance_actual])
            
            # False alarms: upper bound > 10 when actual <= 10
            normal_mask = ~exceedance_actual
            if normal_mask.any():
                normal_samples += normal_mask.sum()
                false_alarm = exceedance_predicted_upper[normal_mask]
                false_alarms.extend(false_alarm)
    
    # Calculate metrics
    mae = np.mean(errors) if errors else float('inf')
    rmse = np.sqrt(np.mean(np.array(errors)**2)) if errors else float('inf')
    
    # Detection rate
    detection_rate = np.mean(detections) if detections else 0.0
    
    # False alarm rate
    false_alarm_rate = np.mean(false_alarms) if false_alarms else 0.0
    
    # Peak capture: within 20% of actual magnitude for exceedances
    if exceedance_actuals and exceedance_predictions:
        exceedance_actuals_arr = np.array(exceedance_actuals)
        exceedance_predictions_arr = np.array(exceedance_predictions)
        peak_errors_pct = np.abs(exceedance_actuals_arr - exceedance_predictions_arr) / exceedance_actuals_arr * 100
        peak_capture = np.mean(peak_errors_pct <= 20.0)  # Fraction within 20%
    else:
        peak_capture = 1.0 if detection_rate > 0 else 0.0
    
    # Multi-objective score (lower is better)
    # Weights: MAE 20%, Detection 50%, False alarms 20%, Peak capture 10%
    objective_score = (
        0.2 * mae +                              # MAE in ppb
        0.5 * (1 - detection_rate) * 100 +      # CRITICAL: Missed events (0-100 scale)
        0.2 * false_alarm_rate * 10 +            # False alarms (0-10 scale)
        0.1 * (1 - peak_capture) * 50            # Peak capture (0-50 scale)
    )
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'detection_rate': detection_rate,
        'false_alarm_rate': false_alarm_rate,
        'peak_capture': peak_capture,
        'n_exceedances': len(detections),
        'n_false_alarms': sum(false_alarms),
        'objective_score': objective_score
    }
    
    return objective_score, metrics


def grid_search_multi_objective(
    validation_data: pd.DataFrame,
    facility_data_dict: dict,
    pinn
) -> Dict:
    """
    Grid search with multi-objective optimization.
    
    Returns:
        best_params: Dictionary with optimal parameters
    """
    logger.info("Starting multi-objective grid search...")
    
    # Expanded parameter grid
    param_grid = {
        'process_noise': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'measurement_noise': [0.001, 0.01, 0.05, 0.1, 0.5],
        'decay_rate': [0.1, 0.3, 0.5, 0.7, 0.9],
        'pinn_weight': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    best_score = float('inf')
    best_params = None
    results = []
    
    # Total combinations
    total = (len(param_grid['process_noise']) * 
             len(param_grid['measurement_noise']) * 
             len(param_grid['decay_rate']) * 
             len(param_grid['pinn_weight']))
    
    logger.info(f"Testing {total} parameter combinations...")
    
    with tqdm(total=total, desc="Multi-objective search") as pbar:
        for pn in param_grid['process_noise']:
            for mn in param_grid['measurement_noise']:
                for dr in param_grid['decay_rate']:
                    for pw in param_grid['pinn_weight']:
                        objective_score, metrics = compute_metrics(
                            process_noise=pn,
                            measurement_noise=mn,
                            decay_rate=dr,
                            pinn_weight=pw,
                            validation_data=validation_data,
                            facility_data_dict=facility_data_dict,
                            pinn=pinn
                        )
                        
                        result = {
                            'process_noise': pn,
                            'measurement_noise': mn,
                            'decay_rate': dr,
                            'pinn_weight': pw,
                            **metrics
                        }
                        results.append(result)
                        
                        if objective_score < best_score:
                            best_score = objective_score
                            best_params = result.copy()
                            logger.info(
                                f"New best score: {best_score:.4f} | "
                                f"Detection: {metrics['detection_rate']*100:.1f}% | "
                                f"MAE: {metrics['mae']:.4f} ppb | "
                                f"Params: pn={pn}, mn={mn}, dr={dr}, pw={pw}"
                            )
                        
                        pbar.update(1)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_dir = Path("realtime/data")
    output_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(output_dir / 'kalman_parameter_search_v2.csv', index=False)
    
    logger.info(f"\nGrid search complete. Best objective score: {best_score:.4f}")
    logger.info(f"Best parameters:")
    logger.info(f"  Detection rate: {best_params['detection_rate']*100:.1f}%")
    logger.info(f"  MAE: {best_params['mae']:.4f} ppb")
    logger.info(f"  False alarm rate: {best_params['false_alarm_rate']*100:.1f}%")
    logger.info(f"  Peak capture: {best_params['peak_capture']*100:.1f}%")
    
    return best_params


def main():
    """Run multi-objective parameter tuning."""
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
    best_params = grid_search_multi_objective(validation_data, facility_data_dict, pinn)
    
    # Save best parameters
    output_dir = Path("realtime/data")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / "kalman_parameters_v2.json"
    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"Saved optimal parameters to {output_file}")


if __name__ == "__main__":
    main()

