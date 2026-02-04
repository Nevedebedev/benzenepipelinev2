#!/usr/bin/env python3
"""
Analyze Top 100 Highest EDF Concentration Events
Find timestamps with highest actual concentrations and test PINN predictions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

import pandas as pd
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

from pinn import ParametricADEPINN

# Paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

SENSOR_DATA_PATH = Path('/Users/neevpratap/Downloads/sensors_final_synced.csv')
FACILITY_DATA_DIR = PROJECT_DIR / 'simpletesting' / 'nn2trainingdata'
PINN_MODEL_PATH = Path('/Users/neevpratap/Downloads/pinn_combined_final2.pth')

# Constants
UNIT_CONVERSION = 313210039.9  # kg/m^2 to ppb
FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast

# Sensor coordinates (Cartesian, meters)
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

# Sensor column names in CSV
SENSOR_COLS = [f'sensor_{sid}' for sid in SENSORS.keys()]

def load_pinn_model():
    """Load PINN model with exact normalization ranges from documentation"""
    print("Loading PINN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    pinn = ParametricADEPINN()
    checkpoint = torch.load(PINN_MODEL_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    
    # Override normalization ranges (matches training data generation)
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
    pinn.to(device)
    
    print("  ✓ PINN model loaded with normalization ranges")
    return pinn, device

def load_edf_data():
    """Load EDF sensor data and find top 100 highest concentration timestamps"""
    print("="*80)
    print("LOADING EDF SENSOR DATA")
    print("="*80)
    
    if not SENSOR_DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find sensor data file: {SENSOR_DATA_PATH}")
    
    print(f"Loading from: {SENSOR_DATA_PATH}")
    df = pd.read_csv(SENSOR_DATA_PATH)
    
    # Handle timestamp column
    if 't' in df.columns:
        df = df.rename(columns={'t': 'timestamp'})
    if 'timestamp' not in df.columns:
        raise ValueError(f"No timestamp column found in {SENSOR_DATA_PATH}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"  Loaded {len(df)} timestamps")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Calculate max concentration across all sensors for each timestamp
    # Replace NaN with 0 for calculation
    sensor_data = df[SENSOR_COLS].fillna(0.0)
    df['max_concentration_edf'] = sensor_data.max(axis=1)
    df['mean_concentration_edf'] = sensor_data.mean(axis=1)
    df['sum_concentration_edf'] = sensor_data.sum(axis=1)
    
    # Sort by max concentration and get top 100
    top100 = df.nlargest(100, 'max_concentration_edf').copy()
    top100 = top100.sort_values('max_concentration_edf', ascending=False)
    top100['rank'] = range(1, len(top100) + 1)
    
    print(f"\n  Top 100 concentration range: {top100['max_concentration_edf'].min():.2f} - {top100['max_concentration_edf'].max():.2f} ppb")
    print(f"  Mean of top 100: {top100['max_concentration_edf'].mean():.2f} ppb")
    
    return top100

def load_facility_data():
    """Load all facility training data files"""
    print("\n" + "="*80)
    print("LOADING FACILITY DATA")
    print("="*80)
    
    if not FACILITY_DATA_DIR.exists():
        raise FileNotFoundError(f"Could not find facility data directory: {FACILITY_DATA_DIR}")
    
    facility_files = sorted(FACILITY_DATA_DIR.glob('*_synced_training_data.csv'))
    
    if len(facility_files) == 0:
        raise FileNotFoundError(f"No facility data files found in {FACILITY_DATA_DIR}")
    
    facilities = {}
    for fac_file in facility_files:
        fac_name = fac_file.stem.replace('_synced_training_data', '')
        df = pd.read_csv(fac_file)
        
        # Handle timestamp column
        if 't' in df.columns:
            df['timestamp'] = pd.to_datetime(df['t'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            print(f"  ⚠️  {fac_name}: No timestamp column found, skipping")
            continue
        
        facilities[fac_name] = df
        print(f"  Loaded {fac_name}: {len(df)} timestamps")
    
    print(f"\n  ✓ Loaded {len(facilities)} facilities")
    return facilities

def get_facility_params_for_timestamp(facilities, met_timestamp):
    """
    Get facility parameters for a given met timestamp
    Returns dict of facility_name -> params dict
    """
    facility_params = {}
    
    for fac_name, df in facilities.items():
        # Find closest timestamp to met_timestamp
        time_diffs = (df['timestamp'] - met_timestamp).abs()
        closest_idx = time_diffs.idxmin()
        closest_row = df.loc[closest_idx]
        time_diff_hours = time_diffs.loc[closest_idx].total_seconds() / 3600
        
        if time_diff_hours > 1.0:
            # Skip if no data within 1 hour
            continue
        
        try:
            facility_params[fac_name] = {
                'wind_u': float(closest_row['wind_u']),
                'wind_v': float(closest_row['wind_v']),
                'D': float(closest_row['D']),
                'source_x_cartesian': float(closest_row['source_x_cartesian']),
                'source_y_cartesian': float(closest_row['source_y_cartesian']),
                'source_diameter': float(closest_row['source_diameter']),
                'Q': float(closest_row['Q_total']),
            }
        except (KeyError, ValueError) as e:
            # Skip if required columns missing or invalid
            continue
    
    return facility_params

def compute_pinn_at_sensors(pinn, device, facility_params):
    """
    Compute PINN predictions at sensor locations
    EXACTLY matches training data generation method:
    1. Process each facility separately
    2. Use simulation time t=3.0 hours
    3. Superimpose across all facilities
    """
    sensor_predictions = {}
    
    for sensor_id, (sensor_x, sensor_y) in SENSORS.items():
        total_ppb = 0.0
        
        # Process each facility separately
        for fac_name, params in facility_params.items():
            # Extract parameters
            cx = params['source_x_cartesian']
            cy = params['source_y_cartesian']
            d = params['source_diameter']
            Q = params['Q']
            u = params['wind_u']
            v = params['wind_v']
            kappa = params['D']
            
            # Handle NaN diffusion coefficient
            if np.isnan(kappa) or kappa <= 0:
                continue
            
            # Prepare tensors
            x_tensor = torch.tensor([[sensor_x]], dtype=torch.float32).to(device)
            y_tensor = torch.tensor([[sensor_y]], dtype=torch.float32).to(device)
            t_tensor = torch.tensor([[FORECAST_T_HOURS]], dtype=torch.float32).to(device)
            cx_tensor = torch.tensor([[cx]], dtype=torch.float32).to(device)
            cy_tensor = torch.tensor([[cy]], dtype=torch.float32).to(device)
            u_tensor = torch.tensor([[u]], dtype=torch.float32).to(device)
            v_tensor = torch.tensor([[v]], dtype=torch.float32).to(device)
            d_tensor = torch.tensor([[d]], dtype=torch.float32).to(device)
            kappa_tensor = torch.tensor([[kappa]], dtype=torch.float32).to(device)
            Q_tensor = torch.tensor([[Q]], dtype=torch.float32).to(device)
            
            # Compute PINN prediction
            with torch.no_grad():
                try:
                    concentration = pinn(
                        x_tensor, y_tensor, t_tensor,
                        cx_tensor, cy_tensor,
                        u_tensor, v_tensor,
                        d_tensor, kappa_tensor, Q_tensor,
                        normalize=True
                    )
                    
                    # Convert to ppb and superimpose
                    concentration_ppb = concentration.item() * UNIT_CONVERSION
                    
                    # Handle NaN/Inf
                    if np.isnan(concentration_ppb) or np.isinf(concentration_ppb):
                        continue
                    
                    total_ppb += concentration_ppb
                except Exception:
                    # Skip facilities that cause errors
                    continue
        
        sensor_predictions[sensor_id] = total_ppb
    
    return sensor_predictions

def main():
    """Main analysis function"""
    print("="*80)
    print("TOP 100 HIGHEST CONCENTRATION EVENTS - PINN ANALYSIS")
    print("="*80)
    print()
    
    # Step 1: Load EDF data and find top 100
    top100_df = load_edf_data()
    
    # Step 2: Load facility data
    facilities = load_facility_data()
    
    # Step 3: Load PINN model
    pinn, device = load_pinn_model()
    
    # Step 4: Process top 100 timestamps
    print("\n" + "="*80)
    print("PROCESSING TOP 100 TIMESTAMPS")
    print("="*80)
    print()
    
    results = []
    
    for idx, row in tqdm(top100_df.iterrows(), total=len(top100_df), desc="Processing timestamps"):
        timestamp = pd.to_datetime(row['timestamp'])
        met_timestamp = timestamp - pd.Timedelta(hours=3)  # Stagger: use met data from t-3
        
        # Get facility parameters for met timestamp
        facility_params = get_facility_params_for_timestamp(facilities, met_timestamp)
        
        if len(facility_params) == 0:
            print(f"  ⚠️  {timestamp}: No facility data found for met time {met_timestamp}")
            continue
        
        # Compute PINN predictions at sensors
        sensor_predictions = compute_pinn_at_sensors(pinn, device, facility_params)
        
        # Get EDF readings
        edf_readings = {}
        for sensor_id in SENSORS.keys():
            col_name = f'sensor_{sensor_id}'
            edf_val = row[col_name] if col_name in row else 0.0
            edf_val = float(edf_val) if not pd.isna(edf_val) else 0.0
            edf_readings[sensor_id] = edf_val
        
        # Calculate statistics
        edf_values = list(edf_readings.values())
        pinn_values = list(sensor_predictions.values())
        
        edf_max = max(edf_values) if edf_values else 0.0
        edf_mean = np.mean(edf_values) if edf_values else 0.0
        pinn_max = max(pinn_values) if pinn_values else 0.0
        pinn_mean = np.mean(pinn_values) if pinn_values else 0.0
        
        # Calculate ratios (avoid division by zero)
        ratio_max = edf_max / (pinn_max + 1e-6) if pinn_max > 0 else np.nan
        ratio_mean = edf_mean / (pinn_mean + 1e-6) if pinn_mean > 0 else np.nan
        
        # Store results
        result_row = {
            'rank': row['rank'],
            'timestamp': timestamp,
            'met_timestamp': met_timestamp,
            'edf_max_ppb': edf_max,
            'edf_mean_ppb': edf_mean,
            'pinn_max_ppb': pinn_max,
            'pinn_mean_ppb': pinn_mean,
            'ratio_max': ratio_max,
            'ratio_mean': ratio_mean,
            'num_facilities': len(facility_params),
        }
        
        # Add per-sensor comparisons
        for sensor_id in SENSORS.keys():
            result_row[f'edf_{sensor_id}'] = edf_readings.get(sensor_id, 0.0)
            result_row[f'pinn_{sensor_id}'] = sensor_predictions.get(sensor_id, 0.0)
            pinn_val = sensor_predictions.get(sensor_id, 0.0)
            if pinn_val > 0:
                result_row[f'ratio_{sensor_id}'] = edf_readings.get(sensor_id, 0.0) / pinn_val
            else:
                result_row[f'ratio_{sensor_id}'] = np.nan
        
        results.append(result_row)
    
    # Step 5: Generate results
    print("\n" + "="*80)
    print("GENERATING RESULTS")
    print("="*80)
    print()
    
    results_df = pd.DataFrame(results)
    
    # Save detailed CSV
    output_csv = DATA_DIR / 'top100_high_concentrations_pinn_analysis.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"✓ Saved detailed results to: {output_csv}")
    
    # Generate summary statistics
    summary_path = DATA_DIR / 'top100_high_concentrations_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TOP 100 HIGHEST CONCENTRATION EVENTS - PINN ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total events processed: {len(results_df)}\n")
        f.write(f"Date range: {results_df['timestamp'].min()} to {results_df['timestamp'].max()}\n\n")
        
        f.write("="*80 + "\n")
        f.write("EDF CONCENTRATION STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Max EDF: {results_df['edf_max_ppb'].max():.2f} ppb\n")
        f.write(f"Mean EDF (max): {results_df['edf_max_ppb'].mean():.2f} ppb\n")
        f.write(f"Median EDF (max): {results_df['edf_max_ppb'].median():.2f} ppb\n")
        f.write(f"Min EDF (max): {results_df['edf_max_ppb'].min():.2f} ppb\n\n")
        
        f.write("="*80 + "\n")
        f.write("PINN CONCENTRATION STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Max PINN: {results_df['pinn_max_ppb'].max():.2f} ppb\n")
        f.write(f"Mean PINN (max): {results_df['pinn_max_ppb'].mean():.2f} ppb\n")
        f.write(f"Median PINN (max): {results_df['pinn_max_ppb'].median():.2f} ppb\n")
        f.write(f"Min PINN (max): {results_df['pinn_max_ppb'].min():.2f} ppb\n\n")
        
        f.write("="*80 + "\n")
        f.write("PREDICTION RATIOS (EDF / PINN)\n")
        f.write("="*80 + "\n")
        valid_ratios = results_df['ratio_max'].dropna()
        if len(valid_ratios) > 0:
            f.write(f"Mean ratio (max): {valid_ratios.mean():.2f}x\n")
            f.write(f"Median ratio (max): {valid_ratios.median():.2f}x\n")
            f.write(f"Max ratio: {valid_ratios.max():.2f}x\n")
            f.write(f"Min ratio: {valid_ratios.min():.2f}x\n")
            f.write(f"Std ratio: {valid_ratios.std():.2f}x\n\n")
            
            # Count under/over predictions
            under_predict = (valid_ratios > 1.0).sum()
            over_predict = (valid_ratios < 1.0).sum()
            exact_predict = (valid_ratios == 1.0).sum()
            
            f.write(f"PINN under-predicts (ratio > 1.0): {under_predict} ({100*under_predict/len(valid_ratios):.1f}%)\n")
            f.write(f"PINN over-predicts (ratio < 1.0): {over_predict} ({100*over_predict/len(valid_ratios):.1f}%)\n")
            f.write(f"PINN exact (ratio = 1.0): {exact_predict} ({100*exact_predict/len(valid_ratios):.1f}%)\n\n")
        
        f.write("="*80 + "\n")
        f.write("CORRELATION ANALYSIS\n")
        f.write("="*80 + "\n")
        correlation = results_df['edf_max_ppb'].corr(results_df['pinn_max_ppb'])
        f.write(f"Correlation (EDF max vs PINN max): {correlation:.3f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("TOP 10 WORST CASES (Highest EDF)\n")
        f.write("="*80 + "\n\n")
        top10 = results_df.nlargest(10, 'edf_max_ppb')[['timestamp', 'edf_max_ppb', 'pinn_max_ppb', 'ratio_max']]
        for _, row in top10.iterrows():
            f.write(f"{row['timestamp']}: EDF={row['edf_max_ppb']:.2f} ppb, PINN={row['pinn_max_ppb']:.2f} ppb, Ratio={row['ratio_max']:.2f}x\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("TOP 10 WORST UNDER-PREDICTIONS (Highest Ratio)\n")
        f.write("="*80 + "\n\n")
        top10_under = results_df.nlargest(10, 'ratio_max')[['timestamp', 'edf_max_ppb', 'pinn_max_ppb', 'ratio_max']]
        for _, row in top10_under.iterrows():
            f.write(f"{row['timestamp']}: EDF={row['edf_max_ppb']:.2f} ppb, PINN={row['pinn_max_ppb']:.2f} ppb, Ratio={row['ratio_max']:.2f}x\n")
        f.write("\n")
    
    print(f"✓ Saved summary to: {summary_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Processed {len(results_df)} events")
    print(f"Results saved to: {output_csv}")
    print(f"Summary saved to: {summary_path}")

if __name__ == '__main__':
    main()

