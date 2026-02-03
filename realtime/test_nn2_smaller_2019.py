#!/usr/bin/env python3
"""
Test Simplified NN2 Model (Smaller Architecture) Against All of 2019

Tests the simplified NN2 model (36 → 256 → 128 → 64 → 9) on full 2019 data.
"""

import sys
from pathlib import Path
sys.path.append('/Users/neevpratap/simpletesting')
sys.path.append(str(Path(__file__).parent / 'simpletesting'))
sys.path.append(str(Path(__file__).parent / 'drive-download-20260202T042428Z-3-001'))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from pinn import ParametricADEPINN
import pickle

# Import the simplified NN2 model definition
from nn2_model_only import NN2_CorrectionNetwork, InverseTransformLayer

# Compatibility class for old architecture models
class NN2_CorrectionNetwork_Old(nn.Module):
    """NN2 model with old architecture (45 features, 512→512→256→128)"""
    def __init__(self, n_sensors=9, scaler_mean=None, scaler_scale=None, output_ppb=True):
        super().__init__()
        self.n_sensors = n_sensors
        self.output_ppb = output_ppb
        
        # Old architecture: 45 → 512 → 512 → 256 → 128 → 9
        self.correction_network = nn.Sequential(
            nn.Linear(45, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_sensors)
        )
        
        if output_ppb and scaler_mean is not None and scaler_scale is not None:
            self.inverse_transform = InverseTransformLayer(scaler_mean, scaler_scale)
        else:
            self.inverse_transform = None
    
    def forward(self, pinn_predictions, sensor_coords, current_sensors, wind, diffusion, temporal):
        batch_size = pinn_predictions.shape[0]
        coords_flat = sensor_coords.reshape(batch_size, -1)
        features = torch.cat([
            pinn_predictions,      # [batch, 9]
            coords_flat,           # [batch, 18]
            current_sensors,       # [batch, 9] - required for 45-feature model
            wind,                  # [batch, 2]
            diffusion,             # [batch, 1]
            temporal               # [batch, 6]
        ], dim=-1)  # Total: 45 features
        
        corrections = self.correction_network(features)
        corrected_scaled = pinn_predictions + corrections
        
        if self.inverse_transform is not None:
            corrected_ppb = self.inverse_transform(corrected_scaled)
            return corrected_ppb, corrections
        else:
            return corrected_scaled, corrections

# Paths
SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
NN2_MODEL_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_scaled/nn2_master_model_ppb-2.pth"
NN2_SCALERS_PATH = "/Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_scaled/nn2_master_scalers-2.pkl"

# Constants
UNIT_CONVERSION = 313210039.9  # kg/m^2 to ppb
FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast

# 9 sensor coordinates (Cartesian) - matches training data
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
    """Load PINN and NN2 models"""
    print("Loading models...")
    
    # Load PINN
    pinn = ParametricADEPINN()
    checkpoint = torch.load(PINN_MODEL_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.endswith('_min') and not k.endswith('_max')}
    pinn.load_state_dict(filtered_state_dict, strict=False)
    
    # Override normalization ranges (matches training data generation)
    for attr, val in [
        ('x_min', 0.0), ('x_max', 30000.0), ('y_min', 0.0), ('y_max', 30000.0),
        ('t_min', 0.0), ('t_max', 8760.0), ('cx_min', 0.0), ('cx_max', 30000.0),
        ('cy_min', 0.0), ('cy_max', 30000.0), ('u_min', -15.0), ('u_max', 15.0),
        ('v_min', -15.0), ('v_max', 15.0), ('d_min', 0.0), ('d_max', 200.0),
        ('kappa_min', 0.0), ('kappa_max', 200.0), ('Q_min', 0.0), ('Q_max', 0.01)
    ]:
        setattr(pinn, attr, torch.tensor(val))
    
    pinn.eval()
    
    # Check if master model exists
    if not Path(NN2_MODEL_PATH).exists():
        print(f"\n❌ ERROR: Master model not found at {NN2_MODEL_PATH}")
        print("   Please download nn2_master_model.pth from Colab to this location.")
        print("   The model should be saved in /content/models/leave_one_out/nn2_master_model.pth")
        print()
        print("   To download from Colab:")
        print("   1. In Colab, run: !cp /content/models/leave_one_out/nn2_master_model.pth /content/")
        print("   2. Download the file from Colab's file browser")
        print("   3. Place it in: /Users/neevpratap/Desktop/benzenepipelinev2/realtime/nn2_smaller/")
        return None, None, None, None
    
    # Load NN2 checkpoint
    nn2_checkpoint = torch.load(NN2_MODEL_PATH, map_location='cpu', weights_only=False)
    
    # Get scaler parameters from checkpoint
    scaler_mean = nn2_checkpoint.get('scaler_mean', None)
    scaler_scale = nn2_checkpoint.get('scaler_scale', None)
    output_ppb = nn2_checkpoint.get('output_ppb', True)
    
    # Check model architecture
    state_dict = nn2_checkpoint['model_state_dict']
    first_layer_shape = state_dict['correction_network.0.weight'].shape
    input_features = first_layer_shape[1]
    
    print(f"  ✓ NN2 model has {input_features} input features")
    
    # Check architecture based on layer sizes
    second_layer_shape = state_dict.get('correction_network.4.weight', None)
    if second_layer_shape is not None:
        second_layer_size = second_layer_shape.shape[0] if hasattr(second_layer_shape, 'shape') else second_layer_shape[0]
        if second_layer_size == 512:
            print(f"  ⚠️  Model uses OLD architecture (512→512→256→128), not simplified (256→128→64)")
            print(f"  ⚠️  Will load with compatibility layer (provides zeros for current_sensors)")
            nn2 = NN2_CorrectionNetwork_Old(
                n_sensors=9,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                output_ppb=output_ppb
            )
        elif second_layer_size == 128:
            print(f"  ✓ Model uses SIMPLIFIED architecture (256→128→64)")
            nn2 = NN2_CorrectionNetwork(
                n_sensors=9,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                output_ppb=output_ppb
            )
        else:
            print(f"  ⚠️  Unknown architecture, attempting to load...")
            nn2 = NN2_CorrectionNetwork(
                n_sensors=9,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                output_ppb=output_ppb
            )
    else:
        # Fallback: try simplified first
        if input_features == 36:
            nn2 = NN2_CorrectionNetwork(
                n_sensors=9,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                output_ppb=output_ppb
            )
        else:
            nn2 = NN2_CorrectionNetwork_Old(
                n_sensors=9,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                output_ppb=output_ppb
            )
    
    nn2.load_state_dict(nn2_checkpoint['model_state_dict'])
    nn2.eval()
    
    # Load scalers
    with open(NN2_SCALERS_PATH, 'rb') as f:
        scalers = pickle.load(f)
    
    # Sensor coords for NN2
    sensor_coords_spatial = np.array([SENSORS[k] for k in sorted(SENSORS.keys())])
    
    print("  ✓ Models loaded")
    print(f"  ✓ NN2 outputs in ppb: {output_ppb}")
    return pinn, nn2, scalers, sensor_coords_spatial

def predict_pinn_at_sensors(pinn, facility_files_dict, timestamp):
    """
    Predict PINN at sensor locations using EXACT same method as training data generation.
    Uses simulation time t=3.0 hours, not absolute calendar time.
    """
    # Get met data from 3 hours before (for 3-hour forecast)
    input_timestamp = timestamp - pd.Timedelta(hours=3)
    
    sensor_pinn_ppb = {sid: 0.0 for sid in SENSORS.keys()}
    
    # For each facility, compute PINN at all sensors and superimpose
    for facility_name, facility_df in facility_files_dict.items():
        facility_data = facility_df[facility_df['t'] == input_timestamp]
        
        if len(facility_data) == 0:
            continue
        
        # For this facility, compute PINN at all sensors and superimpose
        for _, row in facility_data.iterrows():
            # Handle different column name variations
            cx = row.get('source_x_cartesian', row.get('source_x', row.get('cx', 0)))
            cy = row.get('source_y_cartesian', row.get('source_y', row.get('cy', 0)))
            d = row.get('source_diameter', row.get('source_d', row.get('d', 0)))
            Q = row.get('Q_total', row.get('Q', 0))
            u = row.get('wind_u', row.get('u', 0))
            v = row.get('wind_v', row.get('v', 0))
            kappa = row.get('D', row.get('kappa', row.get('diffusion', 0)))
            
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
                    phi_ppb = phi_raw.item() * UNIT_CONVERSION
                    sensor_pinn_ppb[sensor_id] += phi_ppb
    
    return sensor_pinn_ppb

def apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, meteo_data, timestamp, current_sensors=None):
    """Apply NN2 correction to PINN predictions"""
    # Prepare features
    sensor_ids_sorted = sorted(SENSORS.keys())
    pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
    
    # Scale PINN predictions - scaler expects individual values, not arrays
    # Scale each sensor's PINN value individually
    pinn_scaled = np.array([
        scalers['pinn'].transform([[val]])[0, 0] if val > 0 else 0.0
        for val in pinn_array
    ])
    pinn_tensor = torch.tensor(pinn_scaled, dtype=torch.float32).unsqueeze(0)
    
    # Sensor coordinates
    coords_tensor = torch.tensor(sensor_coords_spatial, dtype=torch.float32).unsqueeze(0)
    
    # Wind (u, v)
    wind_tensor = torch.tensor([[meteo_data['wind_u'], meteo_data['wind_v']]], dtype=torch.float32)
    
    # Diffusion
    diffusion_tensor = torch.tensor([[meteo_data['D']]], dtype=torch.float32)
    
    # Temporal features
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    month = timestamp.month
    is_weekend = 1.0 if day_of_week >= 5 else 0.0
    
    temporal_features = np.array([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day_of_week / 7),
        np.cos(2 * np.pi * day_of_week / 7),
        is_weekend,
        month / 12.0
    ])
    
    # Scale temporal features - check if temporal scaler exists
    if 'temporal' in scalers:
        temporal_scaled = scalers['temporal'].transform(temporal_features.reshape(1, -1))[0]
    else:
        # If no temporal scaler, use wind scaler or don't scale
        # Temporal features are already normalized (sin/cos, 0-1 range)
        temporal_scaled = temporal_features
    temporal_tensor = torch.tensor(temporal_scaled, dtype=torch.float32).unsqueeze(0)
    
    # Apply NN2 - check if model needs current_sensors (old architecture)
    with torch.no_grad():
        # Check if model is old architecture (45 features) by checking forward signature
        import inspect
        sig = inspect.signature(nn2.forward)
        num_params = len(sig.parameters) - 1  # Exclude 'self'
        
        if num_params == 6:  # Old architecture: pinn, coords, current_sensors, wind, diffusion, temporal
            # Old architecture needs current_sensors
            if current_sensors is None:
                # Provide zeros for current_sensors (not available in deployment)
                current_sensors_scaled = torch.zeros(1, 9, dtype=torch.float32)
            else:
                current_sensors_array = np.array([current_sensors.get(sid, 0.0) for sid in sensor_ids_sorted])
                current_sensors_scaled = scalers['sensors'].transform(current_sensors_array.reshape(1, -1))[0]
                current_sensors_scaled = torch.tensor(current_sensors_scaled, dtype=torch.float32).unsqueeze(0)
            
            nn2_ppb, corrections = nn2(
                pinn_tensor,
                coords_tensor,
                current_sensors_scaled,
                wind_tensor,
                diffusion_tensor,
                temporal_tensor
            )
        else:  # New architecture (36 features, no current_sensors) - 5 params
            # New architecture (36 features, no current_sensors)
            nn2_ppb, corrections = nn2(
                pinn_tensor,
                coords_tensor,
                wind_tensor,
                diffusion_tensor,
                temporal_tensor
            )
    
    # Convert to dict
    nn2_dict = {sid: nn2_ppb[0, i].item() for i, sid in enumerate(sensor_ids_sorted)}
    
    return nn2_dict

def main():
    print("="*80)
    print("TESTING SIMPLIFIED NN2 MODEL ON FULL 2019 DATA")
    print("="*80)
    print()
    
    # Load models
    pinn, nn2, scalers, sensor_coords_spatial = load_models()
    if pinn is None:
        return
    
    # Load sensor data
    print("Loading sensor data...")
    sensor_df = pd.read_csv(SENSOR_DATA_PATH)
    if 't' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['t'])
    elif 'timestamp' in sensor_df.columns:
        sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
    print(f"  ✓ Loaded {len(sensor_df)} sensor readings")
    
    # Load facility data
    print("Loading facility data...")
    facility_files = sorted(SYNCED_DIR.glob('*.csv'))
    facility_files_dict = {}
    for f in facility_files:
        if 'total' in f.name.lower() or 'superimposed' in f.name.lower():
            continue
        df = pd.read_csv(f)
        # Handle timestamp column
        if 't' in df.columns:
            df['t'] = pd.to_datetime(df['t'])
        elif 'timestamp' in df.columns:
            df['t'] = pd.to_datetime(df['timestamp'])
        else:
            print(f"  ⚠️  Skipping {f.name} - no timestamp column")
            continue
        facility_files_dict[f.stem] = df
    print(f"  ✓ Loaded {len(facility_files_dict)} facilities")
    
    # Get common timestamps
    sensor_times = set(sensor_df['timestamp'])
    facility_times = set()
    for df in facility_files_dict.values():
        if 't' in df.columns:
            facility_times.update(df['t'])
    common_times = sorted(list(sensor_times & facility_times))
    print(f"  ✓ Found {len(common_times)} common timestamps")
    print()
    
    # Process each timestamp
    results = []
    print("Processing timestamps...")
    
    for timestamp in tqdm(common_times):
        # Get met data (3 hours before for forecast)
        input_timestamp = timestamp - pd.Timedelta(hours=3)
        meteo_data = None
        for df in facility_files_dict.values():
            row = df[df['t'] == input_timestamp]
            if len(row) > 0:
                meteo_data = row.iloc[0]
                break
        
        if meteo_data is None:
            continue
        
        # Predict PINN
        pinn_values = predict_pinn_at_sensors(pinn, facility_files_dict, timestamp)
        
        # Apply NN2 correction
        nn2_values = apply_nn2_correction(
            nn2, scalers, sensor_coords_spatial, pinn_values, meteo_data, timestamp
        )
        
        # Get actual sensor readings
        sensor_row = sensor_df[sensor_df['timestamp'] == timestamp]
        if len(sensor_row) == 0:
            continue
        
        sensor_row = sensor_row.iloc[0]
        
        # Store results
        for sensor_id in SENSORS.keys():
            actual = sensor_row.get(f'sensor_{sensor_id}', np.nan)
            if pd.notna(actual) and actual > 0:
                results.append({
                    'timestamp': timestamp,
                    'sensor_id': sensor_id,
                    'actual': actual,
                    'pinn': pinn_values[sensor_id],
                    'nn2': nn2_values[sensor_id]
                })
    
    # Calculate metrics
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("\n❌ No valid results!")
        return
    
    print(f"\n✓ Processed {len(results_df)} valid samples")
    print()
    
    # Overall metrics
    pinn_mae = np.abs(results_df['actual'] - results_df['pinn']).mean()
    nn2_mae = np.abs(results_df['actual'] - results_df['nn2']).mean()
    improvement = ((pinn_mae - nn2_mae) / pinn_mae * 100) if pinn_mae > 0 else 0
    
    print("="*80)
    print("RESULTS - FULL 2019 DATA")
    print("="*80)
    print(f"\nOverall Performance:")
    print(f"  PINN MAE: {pinn_mae:.6f} ppb")
    print(f"  NN2 MAE:  {nn2_mae:.6f} ppb")
    print(f"  Improvement: {improvement:.2f}%")
    print()
    
    # Per-sensor metrics
    print("Per-Sensor Performance:")
    print("-"*80)
    sensor_metrics = []
    for sensor_id in sorted(SENSORS.keys()):
        sensor_data = results_df[results_df['sensor_id'] == sensor_id]
        if len(sensor_data) == 0:
            continue
        
        sensor_pinn_mae = np.abs(sensor_data['actual'] - sensor_data['pinn']).mean()
        sensor_nn2_mae = np.abs(sensor_data['actual'] - sensor_data['nn2']).mean()
        sensor_improvement = ((sensor_pinn_mae - sensor_nn2_mae) / sensor_pinn_mae * 100) if sensor_pinn_mae > 0 else 0
        
        sensor_metrics.append({
            'sensor_id': sensor_id,
            'pinn_mae': sensor_pinn_mae,
            'nn2_mae': sensor_nn2_mae,
            'improvement': sensor_improvement,
            'n_samples': len(sensor_data)
        })
        
        print(f"  {sensor_id}:")
        print(f"    PINN MAE: {sensor_pinn_mae:.6f} ppb")
        print(f"    NN2 MAE:  {sensor_nn2_mae:.6f} ppb")
        print(f"    Improvement: {sensor_improvement:.2f}% ({len(sensor_data)} samples)")
        print()
    
    # Save results
    output_file = Path(__file__).parent / 'nn2_smaller_2019_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"✓ Results saved to: {output_file}")
    
    # Save summary
    summary_file = Path(__file__).parent / 'nn2_smaller_2019_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("SIMPLIFIED NN2 MODEL - 2019 VALIDATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Performance:\n")
        f.write(f"  PINN MAE: {pinn_mae:.6f} ppb\n")
        f.write(f"  NN2 MAE:  {nn2_mae:.6f} ppb\n")
        f.write(f"  Improvement: {improvement:.2f}%\n")
        f.write(f"  Total samples: {len(results_df)}\n\n")
        f.write("Per-Sensor Performance:\n")
        f.write("-"*80 + "\n")
        for m in sensor_metrics:
            f.write(f"{m['sensor_id']}: PINN={m['pinn_mae']:.6f}, NN2={m['nn2_mae']:.6f}, "
                   f"Imp={m['improvement']:.2f}%, n={m['n_samples']}\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    print()
    print("="*80)

if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    main()

