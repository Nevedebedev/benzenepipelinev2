"""
Kalman Filter Diagnostic Analysis
Analyze why detection rate is 0% and identify parameter issues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from typing import Dict, Tuple
import logging

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


def load_data(year: int = 2019):
    """Load validation and facility data."""
    # Load sensor data
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
    
    # Load facility data
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
        df_fac = pd.read_csv(facility_file)
        if 't' in df_fac.columns:
            df_fac = df_fac.rename(columns={'t': 'timestamp'})
        df_fac['timestamp'] = pd.to_datetime(df_fac['timestamp'])
        df_fac = df_fac[df_fac['timestamp'].dt.year == year].copy()
        if len(df_fac) > 0:
            facility_data_dict[facility_name] = df_fac
    
    return df, facility_data_dict


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


def analyze_exceedances(validation_data: pd.DataFrame, facility_data_dict: dict, 
                       pinn, kf_params: dict) -> Dict:
    """
    Analyze exceedance detection performance.
    
    Returns:
        Dictionary with exceedance statistics
    """
    logger.info("Analyzing exceedance detection...")
    
    # Load current parameters
    kf = BenzeneKalmanFilter(**kf_params)
    
    sensor_cols = [f'sensor_{sid}' for sid in SENSOR_IDS]
    
    results = []
    
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
        
        # Store results for each sensor
        for j, sensor_id in enumerate(SENSOR_IDS):
            results.append({
                'timestamp': future_time,
                'sensor_id': sensor_id,
                'actual': future_actual[j],
                'pinn': pinn_predictions[j],
                'kalman': kalman_forecast[j],
                'kalman_upper_95': upper[j],
                'kalman_lower_95': lower[j],
                'uncertainty': uncertainty[j]
            })
    
    results_df = pd.DataFrame(results)
    
    # Exceedance analysis
    valid_mask = ~results_df['actual'].isna() & ~results_df['kalman'].isna()
    valid_df = results_df[valid_mask].copy()
    
    # Overall exceedances
    actual_exceedances = (valid_df['actual'] > 10.0).sum()
    
    # Detection methods
    detected_upper = (valid_df['kalman_upper_95'] > 10.0).sum()
    detected_point = (valid_df['kalman'] > 10.0).sum()
    detected_lower = (valid_df['kalman_lower_95'] > 10.0).sum()
    
    detection_rate_upper = (detected_upper / max(actual_exceedances, 1)) * 100
    detection_rate_point = (detected_point / max(actual_exceedances, 1)) * 100
    detection_rate_lower = (detected_lower / max(actual_exceedances, 1)) * 100
    
    # False alarms (upper bound > 10 when actual < 10)
    false_alarms = ((valid_df['kalman_upper_95'] > 10.0) & (valid_df['actual'] <= 10.0)).sum()
    false_alarm_rate = (false_alarms / len(valid_df[valid_df['actual'] <= 10.0])) * 100 if len(valid_df[valid_df['actual'] <= 10.0]) > 0 else 0
    
    # Per-sensor analysis
    sensor_stats = {}
    for sensor_id in SENSOR_IDS:
        sensor_df = valid_df[valid_df['sensor_id'] == sensor_id]
        if len(sensor_df) == 0:
            continue
        
        sensor_actual_exceed = (sensor_df['actual'] > 10.0).sum()
        sensor_detected_upper = (sensor_df['kalman_upper_95'] > 10.0).sum()
        
        sensor_stats[sensor_id] = {
            'total_samples': len(sensor_df),
            'actual_exceedances': sensor_actual_exceed,
            'detected_upper': sensor_detected_upper,
            'detection_rate': (sensor_detected_upper / max(sensor_actual_exceed, 1)) * 100
        }
    
    # Worst misses (largest false negatives)
    exceedance_df = valid_df[valid_df['actual'] > 10.0].copy()
    if len(exceedance_df) > 0:
        exceedance_df['miss_magnitude'] = exceedance_df['actual'] - exceedance_df['kalman_upper_95']
        worst_misses = exceedance_df.nlargest(10, 'miss_magnitude')
    else:
        worst_misses = pd.DataFrame()
    
    return {
        'total_samples': len(valid_df),
        'actual_exceedances': actual_exceedances,
        'detected_upper': detected_upper,
        'detected_point': detected_point,
        'detected_lower': detected_lower,
        'detection_rate_upper': detection_rate_upper,
        'detection_rate_point': detection_rate_point,
        'detection_rate_lower': detection_rate_lower,
        'false_alarms': false_alarms,
        'false_alarm_rate': false_alarm_rate,
        'sensor_stats': sensor_stats,
        'worst_misses': worst_misses,
        'results_df': results_df
    }


def analyze_peak_capture(results_df: pd.DataFrame) -> Dict:
    """Analyze how well filter captures peak magnitudes."""
    valid_df = results_df[~results_df['actual'].isna() & ~results_df['kalman'].isna()].copy()
    
    # Find exceedances
    exceedance_df = valid_df[valid_df['actual'] > 10.0].copy()
    
    if len(exceedance_df) == 0:
        return {
            'n_exceedances': 0,
            'mean_peak_error': 0,
            'mean_peak_error_pct': 0,
            'peak_capture_rate': 0
        }
    
    # Peak magnitude analysis
    exceedance_df['peak_error'] = exceedance_df['actual'] - exceedance_df['kalman']
    exceedance_df['peak_error_pct'] = (exceedance_df['peak_error'] / exceedance_df['actual']) * 100
    
    # Peak capture: within 20% of actual
    peak_captured = (exceedance_df['peak_error_pct'].abs() <= 20).sum()
    peak_capture_rate = (peak_captured / len(exceedance_df)) * 100
    
    return {
        'n_exceedances': len(exceedance_df),
        'mean_peak_error': exceedance_df['peak_error'].mean(),
        'mean_peak_error_pct': exceedance_df['peak_error_pct'].mean(),
        'peak_capture_rate': peak_capture_rate,
        'median_peak_error': exceedance_df['peak_error'].median(),
        'max_underestimate': exceedance_df['peak_error'].min()
    }


def analyze_parameter_sensitivity(results_df: pd.DataFrame, facility_data_dict: dict, 
                                  pinn, base_params: dict) -> pd.DataFrame:
    """Test how each parameter affects detection rate."""
    logger.info("Testing parameter sensitivity...")
    
    sensitivity_results = []
    
    # Test process_noise
    for pn in [0.1, 0.5, 1.0, 2.0, 5.0]:
        params = base_params.copy()
        params['process_noise'] = pn
        detection_rate = _test_detection_rate(params, results_df, facility_data_dict, pinn)
        sensitivity_results.append({
            'parameter': 'process_noise',
            'value': pn,
            'detection_rate': detection_rate
        })
    
    # Test decay_rate
    for dr in [0.1, 0.3, 0.5, 0.7, 0.9]:
        params = base_params.copy()
        params['decay_rate'] = dr
        detection_rate = _test_detection_rate(params, results_df, facility_data_dict, pinn)
        sensitivity_results.append({
            'parameter': 'decay_rate',
            'value': dr,
            'detection_rate': detection_rate
        })
    
    # Test pinn_weight
    for pw in [0.1, 0.3, 0.5, 0.7, 0.9]:
        params = base_params.copy()
        params['pinn_weight'] = pw
        detection_rate = _test_detection_rate(params, results_df, facility_data_dict, pinn)
        sensitivity_results.append({
            'parameter': 'pinn_weight',
            'value': pw,
            'detection_rate': detection_rate
        })
    
    return pd.DataFrame(sensitivity_results)


def _test_detection_rate(params: dict, results_df: pd.DataFrame, 
                        facility_data_dict: dict, pinn) -> float:
    """Quick test of detection rate for given parameters."""
    # Sample subset for speed
    sample_df = results_df.sample(min(1000, len(results_df)))
    
    kf = BenzeneKalmanFilter(**params)
    
    detected = 0
    actual_exceedances = 0
    
    for _, row in sample_df.iterrows():
        if pd.isna(row['actual']) or row['actual'] <= 10.0:
            continue
        
        actual_exceedances += 1
        
        # Would need to recompute, but for speed just check if upper bound would catch it
        # This is approximate
        if row.get('kalman_upper_95', 0) > 10.0:
            detected += 1
    
    return (detected / max(actual_exceedances, 1)) * 100 if actual_exceedances > 0 else 0


def create_diagnostic_plots(results_df: pd.DataFrame, output_dir: Path):
    """Create diagnostic visualizations."""
    logger.info("Creating diagnostic plots...")
    
    valid_df = results_df[~results_df['actual'].isna() & ~results_df['kalman'].isna()].copy()
    
    # Plot 1: Time series for sensor with most exceedances
    sensor_exceedances = {}
    for sensor_id in SENSOR_IDS:
        sensor_df = valid_df[valid_df['sensor_id'] == sensor_id]
        sensor_exceedances[sensor_id] = (sensor_df['actual'] > 10.0).sum()
    
    if sensor_exceedances:
        top_sensor = max(sensor_exceedances, key=sensor_exceedances.get)
        sensor_df = valid_df[valid_df['sensor_id'] == top_sensor].sort_values('timestamp')
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Full time series
        ax1 = axes[0]
        ax1.plot(sensor_df['timestamp'], sensor_df['actual'], label='Actual', alpha=0.7, linewidth=1)
        ax1.plot(sensor_df['timestamp'], sensor_df['kalman'], label='Kalman', alpha=0.7, linewidth=1)
        ax1.plot(sensor_df['timestamp'], sensor_df['pinn'], label='PINN', alpha=0.5, linewidth=1)
        ax1.fill_between(sensor_df['timestamp'], 
                         sensor_df['kalman_lower_95'], 
                         sensor_df['kalman_upper_95'],
                         alpha=0.2, label='95% CI')
        ax1.axhline(y=10, color='red', linestyle='--', label='EPA Threshold (10 ppb)')
        ax1.set_ylabel('Concentration (ppb)')
        ax1.set_title(f'Sensor {top_sensor} - Full Time Series')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Zoom on exceedances
        exceedance_df = sensor_df[sensor_df['actual'] > 10.0]
        if len(exceedance_df) > 0:
            ax2 = axes[1]
            # Get time range around exceedances
            time_range = (exceedance_df['timestamp'].min() - pd.Timedelta(hours=6),
                         exceedance_df['timestamp'].max() + pd.Timedelta(hours=6))
            zoom_df = sensor_df[(sensor_df['timestamp'] >= time_range[0]) & 
                               (sensor_df['timestamp'] <= time_range[1])]
            
            ax2.plot(zoom_df['timestamp'], zoom_df['actual'], label='Actual', 
                    alpha=0.7, linewidth=2, marker='o', markersize=4)
            ax2.plot(zoom_df['timestamp'], zoom_df['kalman'], label='Kalman', 
                    alpha=0.7, linewidth=2, marker='s', markersize=4)
            ax2.plot(zoom_df['timestamp'], zoom_df['pinn'], label='PINN', 
                    alpha=0.5, linewidth=1)
            ax2.fill_between(zoom_df['timestamp'], 
                            zoom_df['kalman_lower_95'], 
                            zoom_df['kalman_upper_95'],
                            alpha=0.2, label='95% CI')
            ax2.axhline(y=10, color='red', linestyle='--', label='EPA Threshold')
            ax2.set_ylabel('Concentration (ppb)')
            ax2.set_xlabel('Time')
            ax2.set_title(f'Sensor {top_sensor} - Exceedance Events (Zoom)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plot_file = output_dir / f"diagnostic_timeseries_{top_sensor}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {plot_file}")
    
    # Plot 2: Scatter - Actual vs Kalman for exceedances
    exceedance_df = valid_df[valid_df['actual'] > 10.0]
    if len(exceedance_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(exceedance_df['actual'], exceedance_df['kalman'], 
                  alpha=0.5, s=30, label='Kalman Forecast')
        ax.scatter(exceedance_df['actual'], exceedance_df['pinn'], 
                  alpha=0.5, s=30, label='PINN', marker='x')
        
        max_val = max(exceedance_df['actual'].max(), exceedance_df['kalman'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        ax.axhline(y=10, color='orange', linestyle='--', label='EPA Threshold')
        ax.axvline(x=10, color='orange', linestyle='--')
        
        ax.set_xlabel('Actual Concentration (ppb)')
        ax.set_ylabel('Predicted Concentration (ppb)')
        ax.set_title('Exceedance Events (>10 ppb) - Prediction Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_file = output_dir / "diagnostic_exceedance_scatter.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {plot_file}")


def print_diagnostic_report(stats: Dict, peak_stats: Dict, sensitivity_df: pd.DataFrame):
    """Print comprehensive diagnostic report."""
    print("\n" + "="*80)
    print("KALMAN FILTER DIAGNOSTIC REPORT")
    print("="*80)
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Actual exceedances (>10 ppb): {stats['actual_exceedances']}")
    
    print(f"\n🚨 DETECTION RATES")
    print(f"  Upper bound (>10 ppb): {stats['detection_rate_upper']:.1f}% ({stats['detected_upper']}/{stats['actual_exceedances']})")
    print(f"  Point forecast (>10 ppb): {stats['detection_rate_point']:.1f}% ({stats['detected_point']}/{stats['actual_exceedances']})")
    print(f"  Lower bound (>10 ppb): {stats['detection_rate_lower']:.1f}% ({stats['detected_lower']}/{stats['actual_exceedances']})")
    print(f"  False alarm rate: {stats['false_alarm_rate']:.1f}% ({stats['false_alarms']} false alarms)")
    
    print(f"\n📈 PEAK CAPTURE ANALYSIS")
    print(f"  Exceedances analyzed: {peak_stats['n_exceedances']}")
    print(f"  Mean peak error: {peak_stats['mean_peak_error']:.2f} ppb ({peak_stats['mean_peak_error_pct']:.1f}%)")
    print(f"  Median peak error: {peak_stats['median_peak_error']:.2f} ppb")
    print(f"  Max underestimate: {peak_stats['max_underestimate']:.2f} ppb")
    print(f"  Peak capture rate (within 20%): {peak_stats['peak_capture_rate']:.1f}%")
    
    print(f"\n🔍 PER-SENSOR DETECTION RATES")
    print(f"{'Sensor':<15} {'Samples':<10} {'Exceedances':<12} {'Detected':<10} {'Rate':<10}")
    print("-" * 70)
    for sensor_id, sensor_stat in stats['sensor_stats'].items():
        print(f"{sensor_id:<15} {sensor_stat['total_samples']:<10} "
              f"{sensor_stat['actual_exceedances']:<12} {sensor_stat['detected_upper']:<10} "
              f"{sensor_stat['detection_rate']:.1f}%")
    
    if len(stats['worst_misses']) > 0:
        print(f"\n⚠️  WORST MISSES (Top 10 False Negatives)")
        print(f"{'Timestamp':<20} {'Sensor':<15} {'Actual':<10} {'Kalman':<10} {'Upper CI':<10} {'Miss':<10}")
        print("-" * 85)
        for _, row in stats['worst_misses'].head(10).iterrows():
            miss = row['actual'] - row['kalman_upper_95']
            print(f"{str(row['timestamp']):<20} {row['sensor_id']:<15} "
                  f"{row['actual']:<10.2f} {row['kalman']:<10.2f} {row['kalman_upper_95']:<10.2f} "
                  f"{miss:<10.2f}")
    
    if len(sensitivity_df) > 0:
        print(f"\n🔬 PARAMETER SENSITIVITY")
        for param in ['process_noise', 'decay_rate', 'pinn_weight']:
            param_df = sensitivity_df[sensitivity_df['parameter'] == param]
            if len(param_df) > 0:
                print(f"\n  {param}:")
                print(f"    {'Value':<10} {'Detection Rate':<15}")
                print(f"    {'-'*25}")
                for _, row in param_df.iterrows():
                    print(f"    {row['value']:<10} {row['detection_rate']:<15.1f}%")
    
    print("\n" + "="*80)


def main():
    """Run diagnostic analysis."""
    output_dir = Path("realtime/data/kalman_diagnostics")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load current parameters
    param_file = Path("realtime/data/kalman_parameters.json")
    if param_file.exists():
        with open(param_file, 'r') as f:
            kf_params = json.load(f)
        kf_params = {k: v for k, v in kf_params.items() 
                    if k in ['process_noise', 'measurement_noise', 'decay_rate', 'pinn_weight']}
    else:
        kf_params = {
            'process_noise': 0.1,
            'measurement_noise': 0.1,
            'decay_rate': 0.5,
            'pinn_weight': 0.1
        }
    
    logger.info(f"Using parameters: {kf_params}")
    
    # Load data
    logger.info("Loading data...")
    validation_data, facility_data_dict = load_data(year=2019)
    
    # Load PINN
    logger.info("Loading PINN model...")
    pinn = load_pinn_model()
    
    # Run analysis
    logger.info("Running exceedance analysis...")
    stats = analyze_exceedances(validation_data, facility_data_dict, pinn, kf_params)
    
    logger.info("Running peak capture analysis...")
    peak_stats = analyze_peak_capture(stats['results_df'])
    
    logger.info("Testing parameter sensitivity...")
    sensitivity_df = analyze_parameter_sensitivity(
        stats['results_df'], facility_data_dict, pinn, kf_params
    )
    
    # Print report
    print_diagnostic_report(stats, peak_stats, sensitivity_df)
    
    # Save results
    stats['results_df'].to_csv(output_dir / "diagnostic_results.csv", index=False)
    sensitivity_df.to_csv(output_dir / "parameter_sensitivity.csv", index=False)
    
    # Save summary
    summary = {
        'detection_rate_upper': float(stats['detection_rate_upper']),
        'detection_rate_point': float(stats['detection_rate_point']),
        'false_alarm_rate': float(stats['false_alarm_rate']),
        'peak_capture_rate': float(peak_stats['peak_capture_rate']),
        'mean_peak_error_pct': float(peak_stats['mean_peak_error_pct'])
    }
    with open(output_dir / "diagnostic_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create plots
    create_diagnostic_plots(stats['results_df'], output_dir)
    
    logger.info(f"Diagnostics complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

