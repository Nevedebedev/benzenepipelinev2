"""
Extreme Event Testing for Kalman Filter
Tests on 2019 ITC fire and other high-concentration events.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

from kalman_filter import BenzeneKalmanFilter
from kalman_filter_adaptive import AdaptiveBenzeneKalmanFilter
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


def load_data(year: int = 2019):
    """Load sensor and facility data."""
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


def identify_extreme_events(validation_data: pd.DataFrame) -> List[Dict]:
    """
    Identify extreme events in the data.
    
    Returns:
        List of extreme event dictionaries with timestamps and details
    """
    sensor_cols = [f'sensor_{sid}' for sid in SENSOR_IDS]
    
    # Find high concentration periods (>50 ppb)
    max_concentrations = validation_data[sensor_cols].max(axis=1)
    high_conc_mask = max_concentrations > 50.0
    
    extreme_events = []
    
    # Group consecutive high concentration periods
    if high_conc_mask.any():
        in_event = False
        event_start = None
        
        for idx, row in validation_data.iterrows():
            max_conc = max_concentrations.iloc[idx]
            
            if max_conc > 50.0 and not in_event:
                in_event = True
                event_start = row['timestamp']
            elif max_conc <= 50.0 and in_event:
                in_event = False
                event_end = validation_data.iloc[idx-1]['timestamp']
                
                # Get event details
                event_data = validation_data[
                    (validation_data['timestamp'] >= event_start) &
                    (validation_data['timestamp'] <= event_end)
                ]
                
                max_conc_event = event_data[sensor_cols].max().max()
                max_sensor = event_data[sensor_cols].max().idxmax()
                
                extreme_events.append({
                    'start': event_start,
                    'end': event_end,
                    'duration_hours': len(event_data),
                    'max_concentration': max_conc_event,
                    'max_sensor': max_sensor,
                    'event_type': 'high_concentration'
                })
    
    # Find rapid increase events (>20 ppb increase in 1 hour)
    rapid_increases = []
    for i in range(1, len(validation_data)):
        prev_row = validation_data.iloc[i-1]
        curr_row = validation_data.iloc[i]
        
        time_diff = (curr_row['timestamp'] - prev_row['timestamp']).total_seconds() / 3600
        if time_diff > 1.5:  # Skip if >1.5 hours apart
            continue
        
        for sensor_col in sensor_cols:
            prev_conc = prev_row[sensor_col]
            curr_conc = curr_row[sensor_col]
            
            if not (pd.isna(prev_conc) or pd.isna(curr_conc)):
                increase = curr_conc - prev_conc
                if increase > 20.0:
                    rapid_increases.append({
                        'timestamp': curr_row['timestamp'],
                        'sensor': sensor_col,
                        'increase': increase,
                        'from': prev_conc,
                        'to': curr_conc,
                        'event_type': 'rapid_increase'
                    })
    
    # Add top rapid increases
    if rapid_increases:
        rapid_increases.sort(key=lambda x: x['increase'], reverse=True)
        extreme_events.extend(rapid_increases[:10])  # Top 10
    
    # ITC Fire period (March 17-20, 2019)
    itc_fire_start = pd.Timestamp('2019-03-17 00:00:00')
    itc_fire_end = pd.Timestamp('2019-03-20 23:59:59')
    
    itc_data = validation_data[
        (validation_data['timestamp'] >= itc_fire_start) &
        (validation_data['timestamp'] <= itc_fire_end)
    ]
    
    if len(itc_data) > 0:
        max_conc_itc = itc_data[sensor_cols].max().max()
        extreme_events.append({
            'start': itc_fire_start,
            'end': itc_fire_end,
            'duration_hours': len(itc_data),
            'max_concentration': max_conc_itc,
            'event_type': 'itc_fire',
            'description': '2019 ITC Deer Park Chemical Fire'
        })
    
    return extreme_events


def test_filter_on_event(
    event: Dict,
    validation_data: pd.DataFrame,
    facility_data_dict: dict,
    pinn,
    kf_params: dict,
    use_adaptive: bool = False
) -> Dict:
    """
    Test Kalman filter on a specific extreme event.
    
    Returns:
        Dictionary with test results
    """
    # Initialize filter
    if use_adaptive:
        kf = AdaptiveBenzeneKalmanFilter(**kf_params)
    else:
        kf = BenzeneKalmanFilter(**kf_params)
    
    sensor_cols = [f'sensor_{sid}' for sid in SENSOR_IDS]
    
    # Get event time range
    if 'start' in event:
        start_time = event['start']
        end_time = event['end']
    else:
        start_time = event['timestamp'] - pd.Timedelta(hours=6)
        end_time = event['timestamp'] + pd.Timedelta(hours=6)
    
    event_data = validation_data[
        (validation_data['timestamp'] >= start_time) &
        (validation_data['timestamp'] <= end_time)
    ].copy()
    
    if len(event_data) == 0:
        return {'error': 'No data for event'}
    
    results = []
    
    # Process event
    for i in range(len(event_data) - 1):
        current_row = event_data.iloc[i]
        future_row = event_data.iloc[i + 1]
        
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
        
        # Store results
        for j, sensor_id in enumerate(SENSOR_IDS):
            results.append({
                'timestamp': future_time,
                'sensor_id': sensor_id,
                'actual': future_actual[j],
                'kalman': kalman_forecast[j],
                'kalman_upper': upper[j],
                'pinn': pinn_predictions[j],
                'uncertainty': uncertainty[j]
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        return {'error': 'No valid predictions'}
    
    # Calculate metrics
    valid_mask = ~results_df['actual'].isna() & ~results_df['kalman'].isna()
    valid_df = results_df[valid_mask].copy()
    
    if len(valid_df) == 0:
        return {'error': 'No valid data'}
    
    # Exceedance detection
    actual_exceedances = (valid_df['actual'] > 10.0).sum()
    detected_upper = (valid_df['kalman_upper'] > 10.0).sum()
    detected_point = (valid_df['kalman'] > 10.0).sum()
    
    detection_rate_upper = (detected_upper / max(actual_exceedances, 1)) * 100
    detection_rate_point = (detected_point / max(actual_exceedances, 1)) * 100
    
    # Peak capture
    exceedance_df = valid_df[valid_df['actual'] > 10.0]
    if len(exceedance_df) > 0:
        peak_errors_pct = np.abs(exceedance_df['actual'] - exceedance_df['kalman']) / exceedance_df['actual'] * 100
        peak_capture_rate = np.mean(peak_errors_pct <= 20.0) * 100
        mean_peak_error_pct = peak_errors_pct.mean()
    else:
        peak_capture_rate = 0.0
        mean_peak_error_pct = 0.0
    
    # Response time (how fast filter responds to spikes)
    response_times = []
    for sensor_id in SENSOR_IDS:
        sensor_df = valid_df[valid_df['sensor_id'] == sensor_id].sort_values('timestamp')
        if len(sensor_df) < 2:
            continue
        
        for i in range(1, len(sensor_df)):
            prev_actual = sensor_df.iloc[i-1]['actual']
            curr_actual = sensor_df.iloc[i]['actual']
            curr_pred = sensor_df.iloc[i]['kalman_upper']
            
            # If actual increased by >10 ppb
            if curr_actual - prev_actual > 10.0:
                # Check if prediction also increased
                if curr_pred > prev_actual + 5.0:  # Some threshold
                    response_times.append(1)  # Detected within 1 timestep
                else:
                    response_times.append(0)  # Missed
    
    response_rate = np.mean(response_times) * 100 if response_times else 0.0
    
    # Overall MAE
    mae = np.mean(np.abs(valid_df['actual'] - valid_df['kalman']))
    
    return {
        'event_type': event.get('event_type', 'unknown'),
        'n_samples': len(valid_df),
        'actual_exceedances': int(actual_exceedances),
        'detection_rate_upper': float(detection_rate_upper),
        'detection_rate_point': float(detection_rate_point),
        'peak_capture_rate': float(peak_capture_rate),
        'mean_peak_error_pct': float(mean_peak_error_pct),
        'response_rate': float(response_rate),
        'mae': float(mae),
        'max_actual': float(valid_df['actual'].max()),
        'max_predicted': float(valid_df['kalman'].max()),
        'results_df': results_df
    }


def create_event_plots(event_results: Dict, output_dir: Path):
    """Create visualization plots for extreme events."""
    results_df = event_results['results_df']
    
    # Plot time series for sensor with highest concentration
    sensor_max = results_df.groupby('sensor_id')['actual'].max()
    top_sensor = sensor_max.idxmax()
    
    sensor_df = results_df[results_df['sensor_id'] == top_sensor].sort_values('timestamp')
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(sensor_df['timestamp'], sensor_df['actual'], 
           label='Actual', linewidth=2, marker='o', markersize=4)
    ax.plot(sensor_df['timestamp'], sensor_df['kalman'], 
           label='Kalman', linewidth=2, marker='s', markersize=4)
    ax.plot(sensor_df['timestamp'], sensor_df['pinn'], 
           label='PINN', linewidth=1, alpha=0.7)
    ax.fill_between(sensor_df['timestamp'], 
                   sensor_df['kalman'] - sensor_df['uncertainty'],
                   sensor_df['kalman'] + sensor_df['uncertainty'],
                   alpha=0.2, label='±1σ')
    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, label='EPA Threshold (10 ppb)')
    
    ax.set_ylabel('Concentration (ppb)', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_title(f"Extreme Event: {event_results['event_type']} - Sensor {top_sensor}\n"
                 f"Detection Rate: {event_results['detection_rate_upper']:.1f}% | "
                 f"Peak Capture: {event_results['peak_capture_rate']:.1f}%", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plot_file = output_dir / f"extreme_event_{event_results['event_type']}_{top_sensor}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot: {plot_file}")


def main():
    """Run extreme event testing."""
    output_dir = Path("realtime/data/kalman_extreme_events")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load parameters
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
    
    logger.info("Loading data...")
    validation_data, facility_data_dict = load_data(year=2019)
    
    logger.info("Loading PINN model...")
    pinn = load_pinn_model()
    
    logger.info("Identifying extreme events...")
    extreme_events = identify_extreme_events(validation_data)
    logger.info(f"Found {len(extreme_events)} extreme events")
    
    # Test on each event
    all_results = []
    
    for i, event in enumerate(extreme_events):
        logger.info(f"\nTesting event {i+1}/{len(extreme_events)}: {event.get('event_type', 'unknown')}")
        
        # Test standard filter
        result_standard = test_filter_on_event(
            event, validation_data, facility_data_dict, pinn, kf_params, use_adaptive=False
        )
        
        if 'error' not in result_standard:
            result_standard['filter_type'] = 'standard'
            all_results.append(result_standard)
            create_event_plots(result_standard, output_dir)
        
        # Test adaptive filter
        result_adaptive = test_filter_on_event(
            event, validation_data, facility_data_dict, pinn, kf_params, use_adaptive=True
        )
        
        if 'error' not in result_adaptive:
            result_adaptive['filter_type'] = 'adaptive'
            all_results.append(result_adaptive)
    
    # Save summary
    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'results_df'} 
        for r in all_results
    ])
    summary_df.to_csv(output_dir / "extreme_event_results.csv", index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("EXTREME EVENT TESTING SUMMARY")
    print("="*80)
    
    if len(summary_df) > 0:
        print(f"\n{'Event Type':<20} {'Filter':<12} {'Detection %':<12} {'Peak Capture %':<15} {'MAE':<10}")
        print("-" * 80)
        for _, row in summary_df.iterrows():
            print(f"{row['event_type']:<20} {row['filter_type']:<12} "
                  f"{row['detection_rate_upper']:<12.1f} {row['peak_capture_rate']:<15.1f} "
                  f"{row['mae']:<10.3f}")
        
        # Overall statistics
        print(f"\nOVERALL STATISTICS:")
        print(f"  Average detection rate: {summary_df['detection_rate_upper'].mean():.1f}%")
        print(f"  Average peak capture: {summary_df['peak_capture_rate'].mean():.1f}%")
        print(f"  Average MAE: {summary_df['mae'].mean():.3f} ppb")
    
    print("\n" + "="*80)
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()

