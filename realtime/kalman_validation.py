"""
Kalman Filter Validation
Test optimized filter on 2019 and 2021 data, including extreme events.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
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

UNIT_CONVERSION_FACTOR = 313210039.9  # kg/m^2 to ppb
FORECAST_T_HOURS = 3.0  # Simulation time


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
    
    logger.info(f"Loaded parameters: {params}")
    return params


def load_validation_data(year: int) -> pd.DataFrame:
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


def load_facility_data(year: int):
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


def validate_on_year(
    year: int,
    pinn,
    kf_params: dict,
    output_dir: Path
) -> pd.DataFrame:
    """
    Validate Kalman filter on a full year of data.
    
    Returns:
        DataFrame with columns:
            - timestamp
            - sensor_id
            - actual
            - pinn
            - kalman
            - kalman_uncertainty
            - kalman_lower_95
            - kalman_upper_95
    """
    logger.info(f"Validating on {year} data...")
    
    # Load data
    validation_data = load_validation_data(year)
    facility_data_dict = load_facility_data(year)
    
    # Initialize Kalman filter
    kf = BenzeneKalmanFilter(**kf_params)
    
    sensor_cols = [f'sensor_{sid}' for sid in SENSOR_IDS]
    
    results = []
    
    # Process each timestamp
    for i in range(len(validation_data) - 1):
        current_row = validation_data.iloc[i]
        future_row = validation_data.iloc[i + 1]
        
        current_time = current_row['timestamp']
        future_time = future_row['timestamp']
        
        # Check if 3-hour pair
        time_diff = (future_time - current_time).total_seconds() / 3600
        if not (2.5 <= time_diff <= 3.5):
            continue
        
        # Get data
        current_sensors = current_row[sensor_cols].values.astype(float)
        future_actual = future_row[sensor_cols].values.astype(float)
        
        # PINN prediction
        try:
            pinn_predictions = predict_pinn_at_sensors(pinn, facility_data_dict, current_time)
        except Exception as e:
            logger.warning(f"PINN failed for {current_time}: {e}")
            continue
        
        # Kalman forecast
        kalman_forecast, uncertainty = kf.forecast(
            current_sensors=current_sensors,
            pinn_predictions=pinn_predictions,
            hours_ahead=3,
            return_uncertainty=True
        )
        
        # Confidence intervals
        lower, upper = kf.get_confidence_interval(kalman_forecast, uncertainty, confidence=0.95)
        
        # Store results for each sensor
        for j, sensor_id in enumerate(SENSOR_IDS):
            results.append({
                'timestamp': future_time,
                'sensor_id': sensor_id,
                'actual': future_actual[j],
                'pinn': pinn_predictions[j],
                'kalman': kalman_forecast[j],
                'kalman_uncertainty': uncertainty[j],
                'kalman_lower_95': lower[j],
                'kalman_upper_95': upper[j]
            })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    output_file = output_dir / f"kalman_validation_{year}.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")
    
    return results_df


def analyze_results(results_df: pd.DataFrame, year: int, output_dir: Path):
    """Generate analysis and visualizations."""
    # Overall metrics
    valid_mask = ~results_df['actual'].isna() & ~results_df['kalman'].isna()
    valid_df = results_df[valid_mask]
    
    pinn_mae = (valid_df['actual'] - valid_df['pinn']).abs().mean()
    kalman_mae = (valid_df['actual'] - valid_df['kalman']).abs().mean()
    improvement = (pinn_mae - kalman_mae) / pinn_mae * 100
    
    # Extreme event analysis - multiple thresholds
    thresholds = [10.0, 20.0, 50.0]
    extreme_stats = {}
    
    for threshold in thresholds:
        extreme_mask = valid_df['actual'] > threshold
        extreme_df = valid_df[extreme_mask]
        
        if len(extreme_df) > 0:
            pinn_extreme_mae = (extreme_df['actual'] - extreme_df['pinn']).abs().mean()
            kalman_extreme_mae = (extreme_df['actual'] - extreme_df['kalman']).abs().mean()
            extreme_improvement = (pinn_extreme_mae - kalman_extreme_mae) / pinn_extreme_mae * 100
            
            # Detection rate - upper bound > threshold when actual > threshold
            upper_detected = (extreme_df['kalman_upper_95'] > threshold).sum()
            upper_detection_rate = (upper_detected / len(extreme_df)) * 100
            
            # Point forecast detection
            point_detected = (extreme_df['kalman'] > threshold).sum()
            point_detection_rate = (point_detected / len(extreme_df)) * 100
            
            # Peak capture: within 20% of actual magnitude
            peak_errors_pct = np.abs(extreme_df['actual'] - extreme_df['kalman']) / extreme_df['actual'] * 100
            peak_capture_rate = (peak_errors_pct <= 20.0).sum() / len(extreme_df) * 100
            mean_peak_error_pct = peak_errors_pct.mean()
            
            # False alarms: upper bound > threshold when actual <= threshold
            normal_mask = valid_df['actual'] <= threshold
            normal_df = valid_df[normal_mask]
            if len(normal_df) > 0:
                false_alarms = (normal_df['kalman_upper_95'] > threshold).sum()
                false_alarm_rate = (false_alarms / len(normal_df)) * 100
            else:
                false_alarm_rate = 0.0
            
            extreme_stats[threshold] = {
                'n_samples': len(extreme_df),
                'pinn_mae': float(pinn_extreme_mae),
                'kalman_mae': float(kalman_extreme_mae),
                'improvement_pct': float(extreme_improvement),
                'upper_detection_rate': float(upper_detection_rate),
                'point_detection_rate': float(point_detection_rate),
                'peak_capture_rate': float(peak_capture_rate),
                'mean_peak_error_pct': float(mean_peak_error_pct),
                'false_alarm_rate': float(false_alarm_rate)
            }
        else:
            extreme_stats[threshold] = {
                'n_samples': 0,
                'pinn_mae': 0.0,
                'kalman_mae': 0.0,
                'improvement_pct': 0.0,
                'upper_detection_rate': 0.0,
                'point_detection_rate': 0.0,
                'peak_capture_rate': 0.0,
                'mean_peak_error_pct': 0.0,
                'false_alarm_rate': 0.0
            }
    
    # Response time analysis (how fast filter responds to spikes)
    response_times = []
    for sensor_id in valid_df['sensor_id'].unique():
        sensor_df = valid_df[valid_df['sensor_id'] == sensor_id].sort_values('timestamp')
        if len(sensor_df) < 2:
            continue
        
        for i in range(1, len(sensor_df)):
            prev_actual = sensor_df.iloc[i-1]['actual']
            curr_actual = sensor_df.iloc[i]['actual']
            curr_pred_upper = sensor_df.iloc[i]['kalman_upper_95']
            prev_pred_upper = sensor_df.iloc[i-1]['kalman_upper_95']
            
            # If actual increased by >10 ppb
            if curr_actual - prev_actual > 10.0:
                # Check if prediction also increased significantly
                if curr_pred_upper - prev_pred_upper > 5.0:
                    response_times.append(1)  # Detected within 1 timestep
                else:
                    response_times.append(0)  # Missed
    
    response_rate = np.mean(response_times) * 100 if response_times else 0.0
    
    # Separate normal vs extreme conditions
    normal_mask = valid_df['actual'] <= 10.0
    normal_df = valid_df[normal_mask]
    extreme_df = valid_df[~normal_mask]
    
    if len(normal_df) > 0:
        normal_mae = (normal_df['actual'] - normal_df['kalman']).abs().mean()
    else:
        normal_mae = 0.0
    
    if len(extreme_df) > 0:
        extreme_mae = (extreme_df['actual'] - extreme_df['kalman']).abs().mean()
    else:
        extreme_mae = 0.0
    
    # Use 10 ppb threshold for main extreme stats
    extreme_10 = extreme_stats[10.0]
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"{year} VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"\nOverall Performance ({len(valid_df)} samples):")
    print(f"  PINN MAE:    {pinn_mae:.4f} ppb")
    print(f"  Kalman MAE:  {kalman_mae:.4f} ppb")
    print(f"  Improvement: {improvement:+.1f}%")
    
    print(f"\nNormal Conditions (≤ 10 ppb, {len(normal_df)} samples):")
    print(f"  Kalman MAE:         {normal_mae:.4f} ppb")
    
    print(f"\nExtreme Events (> 10 ppb, {extreme_10['n_samples']} samples):")
    print(f"  PINN MAE:           {extreme_10['pinn_mae']:.4f} ppb")
    print(f"  Kalman MAE:         {extreme_10['kalman_mae']:.4f} ppb")
    print(f"  Improvement:        {extreme_10['improvement_pct']:+.1f}%")
    print(f"  Detection rate (upper): {extreme_10['upper_detection_rate']:.1f}%")
    print(f"  Detection rate (point): {extreme_10['point_detection_rate']:.1f}%")
    print(f"  Peak capture rate:   {extreme_10['peak_capture_rate']:.1f}% (within 20%)")
    print(f"  Mean peak error:     {extreme_10['mean_peak_error_pct']:.1f}%")
    print(f"  False alarm rate:    {extreme_10['false_alarm_rate']:.1f}%")
    print(f"  Response rate:       {response_rate:.1f}% (detects spikes within 1 timestep)")
    
    # Additional thresholds
    for threshold in [20.0, 50.0]:
        stats = extreme_stats[threshold]
        if stats['n_samples'] > 0:
            print(f"\nVery High Events (> {threshold} ppb, {stats['n_samples']} samples):")
            print(f"  Detection rate:     {stats['upper_detection_rate']:.1f}%")
            print(f"  Peak capture:       {stats['peak_capture_rate']:.1f}%")
    
    # Per-sensor breakdown
    print(f"\nPer-Sensor Performance:")
    print(f"{'Sensor':<15} {'PINN MAE':<12} {'Kalman MAE':<12} {'Improvement':<15}")
    print(f"{'-'*60}")
    
    for sensor_id in results_df['sensor_id'].unique():
        sensor_df = results_df[results_df['sensor_id'] == sensor_id]
        sensor_valid = sensor_df[~sensor_df['actual'].isna() & ~sensor_df['kalman'].isna()]
        
        if len(sensor_valid) > 0:
            s_pinn_mae = (sensor_valid['actual'] - sensor_valid['pinn']).abs().mean()
            s_kalman_mae = (sensor_valid['actual'] - sensor_valid['kalman']).abs().mean()
            s_improvement = (s_pinn_mae - s_kalman_mae) / s_pinn_mae * 100
            
            print(f"{sensor_id:<15} {s_pinn_mae:<12.4f} {s_kalman_mae:<12.4f} {s_improvement:+.1f}%")
    
    # Save summary
    summary = {
        'year': year,
        'n_samples': len(valid_df),
        'pinn_mae': float(pinn_mae),
        'kalman_mae': float(kalman_mae),
        'improvement_pct': float(improvement),
        'normal_mae': float(normal_mae),
        'extreme_mae': float(extreme_mae),
        'response_rate_pct': float(response_rate),
        'extreme_events': extreme_stats
    }
    
    summary_file = output_dir / f"kalman_summary_{year}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create visualization
    create_validation_plots(results_df, year, output_dir)


def create_validation_plots(results_df: pd.DataFrame, year: int, output_dir: Path):
    """Create validation visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    valid_df = results_df[~results_df['actual'].isna() & ~results_df['kalman'].isna()].copy()
    
    # Plot 1: Scatter - PINN vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(valid_df['actual'], valid_df['pinn'], alpha=0.3, s=10)
    max_val = max(valid_df['actual'].max(), valid_df['pinn'].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    ax1.set_xlabel('Actual Concentration (ppb)')
    ax1.set_ylabel('PINN Prediction (ppb)')
    ax1.set_title(f'{year} PINN Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter - Kalman vs Actual
    ax2 = axes[0, 1]
    ax2.scatter(valid_df['actual'], valid_df['kalman'], alpha=0.3, s=10, color='green')
    ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    ax2.set_xlabel('Actual Concentration (ppb)')
    ax2.set_ylabel('Kalman Prediction (ppb)')
    ax2.set_title(f'{year} Kalman Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    ax3 = axes[1, 0]
    pinn_errors = (valid_df['actual'] - valid_df['pinn']).abs()
    kalman_errors = (valid_df['actual'] - valid_df['kalman']).abs()
    
    ax3.hist(pinn_errors, bins=50, alpha=0.5, label='PINN', color='blue')
    ax3.hist(kalman_errors, bins=50, alpha=0.5, label='Kalman', color='green')
    ax3.set_xlabel('Absolute Error (ppb)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'{year} Error Distribution')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Extreme events zoom
    ax4 = axes[1, 1]
    extreme_df = valid_df[valid_df['actual'] > 10.0]
    
    if len(extreme_df) > 0:
        ax4.scatter(extreme_df['actual'], extreme_df['pinn'], alpha=0.5, s=20, label='PINN', color='blue')
        ax4.scatter(extreme_df['actual'], extreme_df['kalman'], alpha=0.5, s=20, label='Kalman', color='green')
        
        # Show uncertainty bounds
        ax4.errorbar(extreme_df['actual'], extreme_df['kalman'], 
                    yerr=1.96*extreme_df['kalman_uncertainty'],
                    fmt='none', ecolor='green', alpha=0.3, label='95% CI')
        
        max_extreme = extreme_df['actual'].max()
        ax4.plot([0, max_extreme], [0, max_extreme], 'r--', label='Perfect')
        ax4.axhline(y=10, color='orange', linestyle='--', label='EPA threshold')
        ax4.set_xlabel('Actual Concentration (ppb)')
        ax4.set_ylabel('Prediction (ppb)')
        ax4.set_title(f'{year} Extreme Events (> 10 ppb)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No extreme events', ha='center', va='center')
        ax4.set_title(f'{year} Extreme Events')
    
    plt.tight_layout()
    plot_file = output_dir / f"kalman_validation_{year}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot to {plot_file}")


def main():
    """Run complete validation."""
    # Create output directory
    output_dir = Path("realtime/data/kalman_validation")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load parameters
    kf_params = load_kalman_parameters()
    # Filter to only parameters (exclude metrics)
    kf_params = {k: v for k, v in kf_params.items() 
                if k in ['process_noise', 'measurement_noise', 'decay_rate', 'pinn_weight']}
    
    # Load PINN model
    logger.info("Loading PINN model...")
    pinn = load_pinn_model()
    
    # Validate on each year
    for year in [2019, 2021]:
        try:
            results_df = validate_on_year(year, pinn, kf_params, output_dir)
            analyze_results(results_df, year, output_dir)
        except Exception as e:
            logger.error(f"Validation failed for {year}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

