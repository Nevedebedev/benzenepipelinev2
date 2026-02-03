#!/usr/bin/env python3
"""
Regenerate NN2 Training Data with CORRECT Tanh PINN

This script:
1. Loads the correct PINN model (with Tanh activations)
2. Processes all facility CSVs in the synced folder
3. For each timestamp, predicts concentration at each of the 9 sensors
4. Superimposes contributions from all facilities
5. Outputs total_concentrations.csv with columns matching NN2 training expectations
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

# Add simpletesting to path for correct PINN import
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting')
from pinn import ParametricADEPINN

# Configuration
# Facility CSV files are in the same directory as this script
SCRIPT_DIR = Path(__file__).parent
SYNCED_DIR = SCRIPT_DIR  # Facility CSVs are in the nn2trainingdata directory
OUTPUT_FILE = SCRIPT_DIR / 'total_superimposed_concentrations.csv'  # Match training code expectation
PINN_MODEL_PATH = Path('/Users/neevpratap/Downloads/pinn_combined_final2.pth')
UNIT_CONVERSION_FACTOR = 313210039.9  # kg/m^2 to ppb

# 9 sensor coordinates (Cartesian) - 2019 EDF Dataset
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

def load_pinn(model_path):
    """Load the correct PINN model with Tanh activations"""
    print(f"Loading PINN from {model_path}...")
    
    pinn = ParametricADEPINN()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Load weights but skip normalization ranges
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    
    # Override normalization ranges (benchmark values - matches concentration_predictor.py)
    pinn.x_min = torch.tensor(0.0)
    pinn.x_max = torch.tensor(30000.0)
    pinn.y_min = torch.tensor(0.0)
    pinn.y_max = torch.tensor(30000.0)
    pinn.t_min = torch.tensor(0.0)
    pinn.t_max = torch.tensor(8760.0)  # 1 year in hours
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
    
    # Verify architecture
    print("\\nPINN Architecture:")
    print(pinn.net)
    print()
    
    return pinn

def predict_pinn_at_sensors(pinn, facility_data):
    """
    Predict PINN concentrations at all sensor locations for a single facility
    
    FIXED: Uses simulation time t=3.0 hours (not absolute calendar time).
    Each scenario starts at t=0, predicts at t=3 hours for 3-hour forecast.
    This makes PINN truly steady-state - only wind/diffusion/emissions affect predictions.
    
    Args:
        pinn: ParametricADEPINN model
        facility_data: DataFrame with columns [t, source_x_cartesian, source_y_cartesian, 
                       source_diameter, Q_total, wind_u, wind_v, D]
    
    Returns:
        Dict mapping forecast_timestamp (t+3) -> sensor_id -> concentration (ppb)
        Note: Predictions made at time t are labeled as t+3 for NN2 training alignment
    """
    # FIX: Use simulation time instead of absolute calendar time
    # This removes the PINN time dependency bug that caused 12x variation
    # for identical conditions across different months
    FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast (each scenario resets to t=0)
    
    predictions = {}
    
    with torch.no_grad():
        for idx, row in facility_data.iterrows():
            input_timestamp = pd.to_datetime(row['t'])  # Time when met data was collected
            
            # Use simulation time t=3.0 hours (not absolute calendar time)
            # Each scenario is independent: starts at t=0, predicts at t=3 hours
            t_hours_forecast = FORECAST_T_HOURS
            
            # Facility parameters
            cx = row['source_x_cartesian']
            cy = row['source_y_cartesian']
            d = row['source_diameter']
            Q = row['Q_total']
            u = row['wind_u']
            v = row['wind_v']
            kappa = row['D']
            
            # Predict at each sensor
            sensor_concentrations = {}
            for sensor_id, (sx, sy) in SENSORS.items():
                # Run PINN with individual arguments (need 2D tensors)
                # CRITICAL: Add normalize=True to match pipeline exactly
                phi_raw = pinn(
                    torch.tensor([[sx]], dtype=torch.float32),
                    torch.tensor([[sy]], dtype=torch.float32),
                    torch.tensor([[t_hours_forecast]], dtype=torch.float32),  # FORECAST TIME
                    torch.tensor([[cx]], dtype=torch.float32),
                    torch.tensor([[cy]], dtype=torch.float32),
                    torch.tensor([[u]], dtype=torch.float32),
                    torch.tensor([[v]], dtype=torch.float32),
                    torch.tensor([[d]], dtype=torch.float32),
                    torch.tensor([[kappa]], dtype=torch.float32),
                    torch.tensor([[Q]], dtype=torch.float32),
                    normalize=True  # CRITICAL: Matches pipeline method
                )
                
                # Convert to ppb
                concentration_ppb = phi_raw.item() * UNIT_CONVERSION_FACTOR
                
                sensor_concentrations[sensor_id] = concentration_ppb
            
            # CRITICAL: Shift timestamp forward by 3 hours
            # Predictions made at time t (using met data from t) are labeled as t+3
            # This aligns PINN predictions with actual sensor readings at t+3 for NN2 training
            forecast_timestamp = input_timestamp + pd.Timedelta(hours=3)
            predictions[forecast_timestamp] = sensor_concentrations
    
    return predictions

def main():
    print("="*80)
    print("REGENERATING NN2 TRAINING DATA WITH CORRECT TANH PINN")
    print("="*80)
    print()
    
    # Load PINN
    pinn = load_pinn(PINN_MODEL_PATH)
    
    # Get all facility CSV files
    facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
    facility_files = [f for f in facility_files if 'summary' not in f.name]
    
    print(f"Found {len(facility_files)} facility files")
    print()
    
    # Storage for superimposed predictions
    # Structure: timestamp -> sensor_id -> total_concentration
    superimposed_predictions = {}
    
    # Process each facility
    for facility_file in tqdm(facility_files, desc="Processing facilities"):
        facility_name = facility_file.stem.replace('_synced_training_data', '')
        
        # Load facility data
        df = pd.read_csv(facility_file)
        
        # Skip if no data
        if len(df) == 0:
            print(f"  Skipping {facility_name}: no data")
            continue
        
        # Predict at sensors
        facility_predictions = predict_pinn_at_sensors(pinn, df)
        
        # Superimpose
        for timestamp, sensor_concs in facility_predictions.items():
            if timestamp not in superimposed_predictions:
                superimposed_predictions[timestamp] = {s: 0.0 for s in SENSORS.keys()}
            
            for sensor_id, conc in sensor_concs.items():
                superimposed_predictions[timestamp][sensor_id] += conc
    
    print()
    print(f"Processed {len(superimposed_predictions)} unique timestamps")
    print()
    
    # Convert to DataFrame
    print("Creating output DataFrame...")
    
    rows = []
    for timestamp in sorted(superimposed_predictions.keys()):
        row = {'t': timestamp}  # Use 't' column name to match existing format
        for sensor_id in SENSORS.keys():
            row[f'sensor_{sensor_id}'] = superimposed_predictions[timestamp][sensor_id]
        rows.append(row)
    
    output_df = pd.DataFrame(rows)
    
    # Save - overwrite existing file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✓ Saved to {OUTPUT_FILE}")
    print(f"  Shape: {output_df.shape}")
    print(f"  Columns: {list(output_df.columns)}")
    print()
    
    # Show statistics
    print("PREDICTION STATISTICS:")
    print("="*80)
    for sensor_id in SENSORS.keys():
        col = f'sensor_{sensor_id}'
        print(f"  {sensor_id}:")
        print(f"    Mean: {output_df[col].mean():.2f} ppb")
        print(f"    Median: {output_df[col].median():.2f} ppb")
        print(f"    Max: {output_df[col].max():.2f} ppb")
        print(f"    Min: {output_df[col].min():.2f} ppb")
    
    print()
    print("="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()
