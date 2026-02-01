#!/usr/bin/env python3
"""
PINN Concentration Prediction Script

Uses the trained PINN model to predict pollutant concentrations at sensor locations
across all facility datasets.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Sensor coordinates (site_id, x_m, y_m)
SENSORS = {
    '482010026': (13972.62, 19915.57),
    '482010036': (14379.4, 16459.34),
    '482010057': (3017.18, 12334.2),
    '482010069': (817.42, 9218.92),
    '482010617': (27049.57, 22045.66),
    '482010803': (8836.35, 15717.2),
    '482010807': (24413.11, 16180.29),
    '482011015': (18413.8, 15068.96),
    '482011035': (1159.98, 12272.52),
    '482011039': (13661.93, 5193.24),
    '482011614': (15077.79, 9450.52),
    '482016000': (1546.9, 6786.33),
}


class PINNModel(nn.Module):
    """Physics-Informed Neural Network for atmospheric dispersion"""
    
    def __init__(self):
        super(PINNModel, self).__init__()
        # Architecture: 10 inputs -> 128 -> 128 -> 128 -> 128 -> 1 output
        self.net = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Normalization parameters (will be loaded from checkpoint, or defaults)
        # OVERRIDE: Hardcorded robust physical bounds for 2019-2021 domain
        self.register_buffer('x_min', torch.tensor(0.0))
        self.register_buffer('x_max', torch.tensor(30000.0))
        self.register_buffer('y_min', torch.tensor(0.0))
        self.register_buffer('y_max', torch.tensor(30000.0))
        self.register_buffer('t_min', torch.tensor(0.0))
        self.register_buffer('t_max', torch.tensor(8760.0))
        self.register_buffer('cx_min', torch.tensor(0.0))
        self.register_buffer('cx_max', torch.tensor(30000.0))
        self.register_buffer('cy_min', torch.tensor(0.0))
        self.register_buffer('cy_max', torch.tensor(30000.0))
        self.register_buffer('u_min', torch.tensor(-15.0))
        self.register_buffer('u_max', torch.tensor(15.0))
        self.register_buffer('v_min', torch.tensor(-15.0))
        self.register_buffer('v_max', torch.tensor(15.0))
        self.register_buffer('d_min', torch.tensor(0.0))
        self.register_buffer('d_max', torch.tensor(200.0))
        self.register_buffer('kappa_min', torch.tensor(0.0))
        self.register_buffer('kappa_max', torch.tensor(200.0))
        self.register_buffer('Q_min', torch.tensor(0.0))
        self.register_buffer('Q_max', torch.tensor(0.01))
    
    def normalize(self, x, y, t, cx, cy, u, v, d, kappa, Q):
        """Normalize inputs using stored min/max values (allowing extrapolation)"""
        # Add small epsilon to avoid potential division by zero if min==max
        epsilon = 1e-8
        
        x_norm = (x - self.x_min) / (self.x_max - self.x_min + epsilon)
        y_norm = (y - self.y_min) / (self.y_max - self.y_min + epsilon)
        t_norm = (t - self.t_min) / (self.t_max - self.t_min + epsilon)
        cx_norm = (cx - self.cx_min) / (self.cx_max - self.cx_min + epsilon)
        cy_norm = (cy - self.cy_min) / (self.cy_max - self.cy_min + epsilon)
        u_norm = (u - self.u_min) / (self.u_max - self.u_min + epsilon)
        v_norm = (v - self.v_min) / (self.v_max - self.v_min + epsilon)
        d_norm = (d - self.d_min) / (self.d_max - self.d_min + epsilon)
        kappa_norm = (kappa - self.kappa_min) / (self.kappa_max - self.kappa_min + epsilon)
        Q_norm = (Q - self.Q_min) / (self.Q_max - self.Q_min + epsilon)
        
        return torch.stack([x_norm, y_norm, t_norm, cx_norm, cy_norm, 
                           u_norm, v_norm, d_norm, kappa_norm, Q_norm], dim=1)
    
    def forward(self, x, y, t, cx, cy, u, v, d, kappa, Q):
        """Forward pass through the network"""
        inputs = self.normalize(x, y, t, cx, cy, u, v, d, kappa, Q)
        return self.net(inputs)


def load_model(model_path):
    """Load the trained PINN model"""
    print(f"Loading model from {model_path}...")
    
    # Create model
    model = PINNModel()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Filter out range buffers from checkpoint so they don't override our hardcoded ones
    # (Since checkpoint has default 0.0-1.0 values)
    state_dict = checkpoint['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('_min') and not k.endswith('_max')}
    
    # Load only network weights (strict=False allows missing buffer keys)
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    print("Model loaded successfully!")
    print(f"  Input features: 10 (x, y, t, cx, cy, u, v, d, kappa, Q)")
    print(f"  Hidden layers: 4 x 128 neurons")
    print(f"  Output: 1 (concentration)")
    
    return model


def predict_for_facility(model, facility_file, output_dir, device='cpu'):
    """
    Predict concentrations at all sensor locations for a single facility
    
    Args:
        model: Trained PINN model
        facility_file: Path to facility CSV file
        output_dir: Directory to save predictions
        device: torch device ('cpu' or 'cuda')
    """
    facility_name = facility_file.stem.replace('_synced_training_data', '')
    # print(f"\nProcessing {facility_name}...")
    
    # Load facility data
    df = pd.read_csv(facility_file)
    
    # Convert timestamp to datetime
    df['t'] = pd.to_datetime(df['t'])
    
    # Calculate time parameter (hours from 2019-01-01 00:00:00)
    t_start = pd.to_datetime('2019-01-01 00:00:00')
    df['t_hours'] = (df['t'] - t_start).dt.total_seconds() / 3600
    
    # Prepare results storage
    predictions = []
    
    # Process each timestamp
    # Optimization: Process in batches if loop is slow, but keeping simple loop for now
    for idx, row in df.iterrows():
    # for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {facility_name}"):
        # Extract facility parameters
        cx = row['source_x_cartesian']
        cy = row['source_y_cartesian']
        d = row['source_diameter']
        Q = row['Q_total']
        u = row['wind_u']
        v = row['wind_v']
        kappa = row['D']  # Use dispersion coefficient D as kappa
        t_hour = row['t_hours']
        timestamp = row['t']
        
        # Skip if any critical parameters are missing
        if pd.isna([cx, cy, d, Q, u, v, kappa, t_hour]).any():
            continue
        
        # Predict for all sensors
        for sensor_id, (sensor_x, sensor_y) in SENSORS.items():
            # Prepare inputs as tensors
            x_tensor = torch.tensor([sensor_x], dtype=torch.float32, device=device)
            y_tensor = torch.tensor([sensor_y], dtype=torch.float32, device=device)
            t_tensor = torch.tensor([t_hour], dtype=torch.float32, device=device)
            cx_tensor = torch.tensor([cx], dtype=torch.float32, device=device)
            cy_tensor = torch.tensor([cy], dtype=torch.float32, device=device)
            u_tensor = torch.tensor([u], dtype=torch.float32, device=device)
            v_tensor = torch.tensor([v], dtype=torch.float32, device=device)
            d_tensor = torch.tensor([d], dtype=torch.float32, device=device)
            kappa_tensor = torch.tensor([kappa], dtype=torch.float32, device=device)
            Q_tensor = torch.tensor([Q], dtype=torch.float32, device=device)
            
            # Predict concentration
            with torch.no_grad():
                raw_out = model(x_tensor, y_tensor, t_tensor, cx_tensor, cy_tensor,
                                u_tensor, v_tensor, d_tensor, kappa_tensor, Q_tensor)
            
            # ═══════════════════════════════════════════════════════════════
            # UNIT CONVERSION: kg/m3 (or dimensionless) -> ppb
            # ═══════════════════════════════════════════════════════════════
            # Factor: 3.13e8
            # Clip negative values to 0 (physics constraint)
            conc_ppb = max(raw_out.item() * 3.13e8, 0.0)
            
            # Store prediction
            predictions.append({
                'timestamp': timestamp,
                'facility': facility_name,
                'sensor_id': sensor_id,
                'sensor_x': sensor_x,
                'sensor_y': sensor_y,
                'source_x': cx,
                'source_y': cy,
                'source_diameter': d,
                'emission_rate': Q,
                'wind_u': u,
                'wind_v': v,
                'dispersion_coef': kappa,
                'predicted_concentration': conc_ppb
            })
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(predictions)
    output_file = output_dir / f"{facility_name}_pinn_predictions.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"  Saved {len(results_df)} predictions to {output_file}")
    print(f"  Average concentration: {results_df['predicted_concentration'].mean():.6f}")
    print(f"  Min/Max concentration: {results_df['predicted_concentration'].min():.6f} / {results_df['predicted_concentration'].max():.6f}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='PINN Concentration Predictions')
    parser.add_argument('--model', type=str, default='pinn_combined_final2.pth',
                       help='Path to trained PINN model')
    parser.add_argument('--data-dir', type=str, default='synced',
                       help='Directory containing facility CSV files')
    parser.add_argument('--output-dir', type=str, default='pinn_predictions',
                       help='Directory to save predictions')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process only one facility')
    parser.add_argument('--facility', type=str, default=None,
                       help='Specific facility to process (test mode)')
    parser.add_argument('--all', action='store_true',
                       help='Process all facilities')
    args = parser.parse_args()
    
    # Setup paths
    model_path = Path(args.model)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    device = 'cpu'
    model = model.to(device)
    
    # Get facility files
    facility_files = sorted(data_dir.glob('*_training_data.csv'))
    if not facility_files:
        facility_files = sorted(data_dir.glob('*_synced_training_data.csv'))
    
    if args.test or args.facility:
        # Test mode: process single facility
        if args.facility:
            facility_files = [f for f in facility_files if args.facility in f.stem]
            if not facility_files:
                print(f"Error: Facility '{args.facility}' not found")
                return
        facility_files = facility_files[:1]
        print(f"\nTest mode: Processing {len(facility_files)} facility")
    else:
        print(f"\nProcessing {len(facility_files)} facilities")
    
    # Process facilities
    all_predictions = []
    for facility_file in facility_files:
        try:
            predictions = predict_for_facility(model, facility_file, output_dir, device)
            all_predictions.append(predictions)
        except Exception as e:
            print(f"Error processing {facility_file.name}: {e}")
            continue
    
    # Summary
    if all_predictions:
        total_predictions = sum(len(df) for df in all_predictions)
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total facilities processed: {len(all_predictions)}")
        print(f"Total predictions: {total_predictions:,}")
        print(f"Predictions per sensor: {total_predictions // len(SENSORS):,}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
