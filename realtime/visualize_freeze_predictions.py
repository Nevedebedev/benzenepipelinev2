#!/usr/bin/env python3
"""
Visualize PINN Predictions for February 2021 Freeze Event
Generate full-domain visualizations comparing PINN vs EDF measurements
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

import pickle
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime
from tqdm import tqdm

from pinn import ParametricADEPINN

# Paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / 'data'
VIZ_DIR = DATA_DIR / 'visualizations' / 'freeze_2021'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

PINN_MODEL_PATH = Path('/Users/neevpratap/Downloads/pinn_combined_final2.pth')

# Constants
UNIT_CONVERSION = 313210039.9  # kg/m^2 to ppb
FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast

# Domain: 30km x 30km
DOMAIN_SIZE = 30000.0  # meters
GRID_RESOLUTION = 100  # 100x100 grid

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
    
    # Override normalization ranges (matches training data generation - EXACT from documentation)
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
    pinn.d_max = torch.tensor(200.0)  # Fixed: was 100.0, should be 200.0 per documentation
    pinn.kappa_min = torch.tensor(0.0)
    pinn.kappa_max = torch.tensor(200.0)
    pinn.Q_min = torch.tensor(0.0)
    pinn.Q_max = torch.tensor(0.01)
    
    pinn.eval()
    pinn.to(device)
    
    print("  ✓ PINN model loaded with normalization ranges")
    return pinn, device

def create_full_domain_grid():
    """Create full domain grid for visualization"""
    x = np.linspace(0, DOMAIN_SIZE, GRID_RESOLUTION)
    y = np.linspace(0, DOMAIN_SIZE, GRID_RESOLUTION)
    X, Y = np.meshgrid(x, y)
    return X, Y

def compute_pinn_full_domain(pinn, device, facility_params, X, Y):
    """
    Compute PINN predictions across full domain
    EXACTLY matches training data generation method:
    1. Process each facility separately
    2. Use simulation time t=3.0 hours
    3. Superimpose across all facilities
    4. Use coordinates AS-IS from facility data (no shifting/clipping)
    """
    batch_size = X.size
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # Initialize concentration field
    total_concentration = np.zeros(batch_size)
    
    # Process each facility separately (EXACTLY like training data generation)
    for fac_name, params in facility_params.items():
        # Extract parameters directly from facility data (AS-IS, no transformations)
        cx = params['source_x_cartesian']
        cy = params['source_y_cartesian']
        d = params['source_diameter']
        Q = params['Q']
        u = params['wind_u']
        v = params['wind_v']
        kappa = params['D']
        
        # Handle NaN diffusion coefficient
        if np.isnan(kappa) or kappa <= 0:
            # Skip facilities with invalid diffusion
            continue
        
        # Prepare inputs - use coordinates AS-IS
        x_tensor = torch.tensor(x_flat, dtype=torch.float32).unsqueeze(1).to(device)
        y_tensor = torch.tensor(y_flat, dtype=torch.float32).unsqueeze(1).to(device)
        t_tensor = torch.full((batch_size, 1), FORECAST_T_HOURS, dtype=torch.float32).to(device)
        cx_tensor = torch.full((batch_size, 1), cx, dtype=torch.float32).to(device)
        cy_tensor = torch.full((batch_size, 1), cy, dtype=torch.float32).to(device)
        u_tensor = torch.full((batch_size, 1), u, dtype=torch.float32).to(device)
        v_tensor = torch.full((batch_size, 1), v, dtype=torch.float32).to(device)
        d_tensor = torch.full((batch_size, 1), d, dtype=torch.float32).to(device)
        kappa_tensor = torch.full((batch_size, 1), kappa, dtype=torch.float32).to(device)
        Q_tensor = torch.full((batch_size, 1), Q, dtype=torch.float32).to(device)
        
        # Compute PINN prediction - EXACT signature from documentation
        with torch.no_grad():
            try:
                # PINN signature: x, y, t, cx, cy, u, v, d, kappa, Q, normalize
                concentration = pinn(
                    x_tensor, y_tensor, t_tensor,
                    cx_tensor, cy_tensor,
                    u_tensor, v_tensor,
                    d_tensor, kappa_tensor, Q_tensor,
                    normalize=True
                )
                
                # Convert to ppb
                concentration_ppb = concentration.cpu().numpy().flatten() * UNIT_CONVERSION
                
                # Handle NaN/Inf - skip this facility's contribution
                if np.any(np.isnan(concentration_ppb)) or np.any(np.isinf(concentration_ppb)):
                    # Replace NaN/Inf with 0 for this facility
                    concentration_ppb = np.nan_to_num(concentration_ppb, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Superimpose (add) to total
                total_concentration += concentration_ppb
            except Exception:
                # Skip facilities that cause errors
                continue
    
    # Reshape to grid
    concentration_2d = total_concentration.reshape(GRID_RESOLUTION, GRID_RESOLUTION)
    return concentration_2d

def compute_pinn_at_sensors(pinn, device, facility_params):
    """
    Compute PINN predictions at sensor locations
    EXACTLY matches training data generation method from validate_jan_mar_2021.py:
    1. Process each facility separately
    2. Use simulation time t=3.0 hours
    3. Superimpose across all facilities
    4. Use coordinates AS-IS from facility data (no shifting/clipping)
    """
    sensor_predictions = {}
    
    for sensor_id, (sensor_x, sensor_y) in SENSORS.items():
        total_ppb = 0.0
        
        # Process each facility separately (EXACTLY like training data generation)
        for fac_name, params in facility_params.items():
            # Extract parameters directly from facility data (AS-IS, no transformations)
            cx = params['source_x_cartesian']
            cy = params['source_y_cartesian']
            d = params['source_diameter']
            Q = params['Q']
            u = params['wind_u']
            v = params['wind_v']
            kappa = params['D']
            
            # Handle NaN diffusion coefficient
            if np.isnan(kappa) or kappa <= 0:
                # Skip facilities with invalid diffusion
                continue
            
            # Prepare tensors - use coordinates AS-IS
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
            
            # Compute PINN prediction - EXACT signature from documentation
            with torch.no_grad():
                try:
                    # PINN signature: x, y, t, cx, cy, u, v, d, kappa, Q, normalize
                    phi_raw = pinn(
                        x_tensor, y_tensor, t_tensor,
                        cx_tensor, cy_tensor,
                        u_tensor, v_tensor,
                        d_tensor, kappa_tensor, Q_tensor,
                        normalize=True
                    )
                    
                    # Convert to ppb and superimpose
                    concentration_ppb = phi_raw.item() * UNIT_CONVERSION
                    
                    # Handle NaN/Inf - skip this facility's contribution
                    if np.isnan(concentration_ppb) or np.isinf(concentration_ppb):
                        # Log warning for debugging
                        # print(f"    ⚠️  {fac_name}: PINN returned NaN/Inf (may be out-of-range coordinates)")
                        continue
                    
                    total_ppb += concentration_ppb
                except Exception as e:
                    # Skip facilities that cause errors
                    # print(f"    ⚠️  {fac_name}: Error - {e}")
                    continue
        
        sensor_predictions[sensor_id] = total_ppb
    
    return sensor_predictions

def create_heatmap(X, Y, concentration_2d, title, filename, vmin=None, vmax=None, 
                   sensor_predictions=None, edf_readings=None, facility_params=None):
    """Create heatmap visualization"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Replace NaN and inf with 0, clip negative values
    concentration_2d = np.nan_to_num(concentration_2d, nan=0.0, posinf=0.0, neginf=0.0)
    concentration_2d = np.maximum(concentration_2d, 0.0)
    
    # Create heatmap
    if vmin is None:
        vmin = 0.01
    if vmax is None:
        vmax = max(concentration_2d.max(), 0.1)
    
    # Use linear scale if all values are very small or zero
    max_val = concentration_2d.max()
    if max_val < 0.1 or vmax <= vmin:
        # Use linear scale for very small values
        im = ax.contourf(X, Y, concentration_2d, levels=20, cmap='YlOrRd', vmin=0, vmax=max(vmax, 0.1))
    else:
        # Use log scale for larger values
        im = ax.contourf(X, Y, concentration_2d, levels=50, cmap='YlOrRd', 
                         norm=LogNorm(vmin=vmin, vmax=vmax))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Benzene Concentration (ppb)', fontsize=12)
    
    # Add sensor locations
    for sensor_id, (sx, sy) in SENSORS.items():
        # Get values
        pinn_val = sensor_predictions.get(sensor_id, 0.0) if sensor_predictions else 0.0
        edf_val = edf_readings.get(sensor_id, 0.0) if edf_readings else 0.0
        
        # Plot sensor location
        ax.plot(sx, sy, 'ko', markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        # Add label
        label = f"{sensor_id}\nPINN: {pinn_val:.2f}\nEDF: {edf_val:.2f}"
        ax.annotate(label, (sx, sy), xytext=(10, 10), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add facility locations
    if facility_params:
        for fac_name, params in facility_params.items():
            fx = params['source_x_cartesian']
            fy = params['source_y_cartesian']
            ax.plot(fx, fy, 'r^', markersize=8, markeredgecolor='white', markeredgewidth=1)
            ax.annotate(fac_name[:10], (fx, fy), xytext=(5, 5), textcoords='offset points',
                       fontsize=6, color='red')
    
    # Add wind vectors (sample a few)
    if facility_params:
        # Use first facility's wind for visualization
        first_fac = list(facility_params.values())[0]
        u = first_fac['wind_u']
        v = first_fac['wind_v']
        
        # Sample wind vectors across domain
        for i in range(0, GRID_RESOLUTION, 20):
            for j in range(0, GRID_RESOLUTION, 20):
                x_pos = X[i, j]
                y_pos = Y[i, j]
                ax.arrow(x_pos, y_pos, u*500, v*500, head_width=200, head_length=200,
                        fc='blue', ec='blue', alpha=0.5, width=100)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filename.name}")

def create_comparison_plot(sensor_predictions, edf_readings, title, filename):
    """Create bar chart comparing PINN vs EDF at each sensor"""
    sensor_ids = sorted(SENSORS.keys())
    pinn_vals = [sensor_predictions.get(sid, 0.0) for sid in sensor_ids]
    edf_vals = [edf_readings.get(sid, 0.0) for sid in sensor_ids]
    
    x = np.arange(len(sensor_ids))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width/2, pinn_vals, width, label='PINN Prediction', color='orange', alpha=0.7)
    bars2 = ax.bar(x + width/2, edf_vals, width, label='EDF Actual', color='blue', alpha=0.7)
    
    ax.set_xlabel('Sensor ID', fontsize=12)
    ax.set_ylabel('Concentration (ppb)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sensor_ids, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Add value labels
    for i, (pinn_val, edf_val) in enumerate(zip(pinn_vals, edf_vals)):
        if pinn_val > 0:
            ax.text(i - width/2, pinn_val, f'{pinn_val:.2f}', ha='center', va='bottom', fontsize=8)
        if edf_val > 0:
            ax.text(i + width/2, edf_val, f'{edf_val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {filename.name}")

def main():
    """Main visualization function"""
    print("="*80)
    print("FEBRUARY 2021 FREEZE EVENT - PINN VISUALIZATION")
    print("="*80)
    print()
    
    # Load freeze events
    events_path = DATA_DIR / 'freeze_2021_with_conditions.pkl'
    if not events_path.exists():
        print(f"ERROR: {events_path} not found. Run analyze_freeze_2021.py first.")
        return
    
    with open(events_path, 'rb') as f:
        events = pickle.load(f)
    
    print(f"Loaded {len(events)} freeze events")
    print()
    
    # Load PINN model
    pinn, device = load_pinn_model()
    print()
    
    # Create full domain grid
    X, Y = create_full_domain_grid()
    print(f"Created {GRID_RESOLUTION}x{GRID_RESOLUTION} grid for full-domain visualization")
    print()
    
    # Process each event
    results = []
    
    for event in tqdm(events, desc="Processing events"):
        timestamp = event['timestamp']
        forecast_time = event['forecast_time']
        facility_params = event['facility_params']
        edf_readings = event['edf_readings']
        
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M')
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H:%M')
        
        print(f"\nProcessing: {date_str} {time_str} UTC")
        print(f"  EDF Peak: {event['max_concentration_edf']:.2f} ppb at {event['peak_sensor_id']}")
        
        # Compute PINN at sensors
        sensor_predictions = compute_pinn_at_sensors(pinn, device, facility_params)
        pinn_peak = max(sensor_predictions.values()) if sensor_predictions else 0.0
        pinn_peak_sensor = max(sensor_predictions, key=sensor_predictions.get) if sensor_predictions else None
        
        print(f"  PINN Peak: {pinn_peak:.2f} ppb at {pinn_peak_sensor}")
        
        # Compute PINN full domain
        print("  Computing full-domain PINN predictions...")
        pinn_2d = compute_pinn_full_domain(pinn, device, facility_params, X, Y)
        
        # Create visualizations
        print("  Generating visualizations...")
        
        # Full-domain heatmap
        pinn_path = VIZ_DIR / f"pinn_full_domain_{timestamp_str}.png"
        create_heatmap(
            X, Y, pinn_2d,
            title=f"PINN Benzene Forecast - Full Domain (30km × 30km)\n{date_str} {time_str} UTC",
            filename=pinn_path,
            vmin=0.01, vmax=max(pinn_2d.max(), 100),
            sensor_predictions=sensor_predictions,
            edf_readings=edf_readings,
            facility_params=facility_params
        )
        
        # Comparison plot
        comp_path = VIZ_DIR / f"comparison_{timestamp_str}.png"
        create_comparison_plot(
            sensor_predictions, edf_readings,
            title=f"PINN vs EDF Comparison\n{date_str} {time_str} UTC",
            filename=comp_path
        )
        
        # Store results
        results.append({
            'date': date_str,
            'timestamp': timestamp,
            'edf_peak_ppb': event['max_concentration_edf'],
            'edf_peak_sensor': event['peak_sensor_id'],
            'pinn_peak_ppb': pinn_peak,
            'pinn_peak_sensor': pinn_peak_sensor,
            'pinn_mean_ppb': np.mean(list(sensor_predictions.values())),
            'edf_mean_ppb': event['mean_concentration_edf'],
            **{f'pinn_{sid}': sensor_predictions.get(sid, 0.0) for sid in SENSORS.keys()},
            **{f'edf_{sid}': edf_readings.get(sid, 0.0) for sid in SENSORS.keys()},
        })
    
    # Save comparison CSV
    results_df = pd.DataFrame(results)
    comp_csv_path = DATA_DIR / 'freeze_2021_pinn_comparison.csv'
    results_df.to_csv(comp_csv_path, index=False)
    print()
    print(f"✓ Saved comparison to: {comp_csv_path}")
    
    # Generate summary report
    report_path = DATA_DIR / 'freeze_2021_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEBRUARY 2021 FREEZE EVENT - PINN PREDICTION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date Range: {events[0]['date']} to {events[-1]['date']}\n")
        f.write(f"Total Events: {len(events)}\n\n")
        f.write("="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"EDF Peak Range: {results_df['edf_peak_ppb'].min():.2f} - {results_df['edf_peak_ppb'].max():.2f} ppb\n")
        f.write(f"PINN Peak Range: {results_df['pinn_peak_ppb'].min():.2f} - {results_df['pinn_peak_ppb'].max():.2f} ppb\n")
        f.write(f"EDF Mean Range: {results_df['edf_mean_ppb'].min():.2f} - {results_df['edf_mean_ppb'].max():.2f} ppb\n")
        f.write(f"PINN Mean Range: {results_df['pinn_mean_ppb'].min():.2f} - {results_df['pinn_mean_ppb'].max():.2f} ppb\n\n")
        f.write(f"Average EDF Peak: {results_df['edf_peak_ppb'].mean():.2f} ppb\n")
        f.write(f"Average PINN Peak: {results_df['pinn_peak_ppb'].mean():.2f} ppb\n")
        f.write(f"Average Under-prediction: {(results_df['edf_peak_ppb'] / (results_df['pinn_peak_ppb'] + 1e-6)).mean():.1f}x\n\n")
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['date']} {row['timestamp'].strftime('%H:%M')} UTC:\n")
            f.write(f"  EDF: {row['edf_peak_ppb']:.2f} ppb at {row['edf_peak_sensor']}\n")
            f.write(f"  PINN: {row['pinn_peak_ppb']:.2f} ppb at {row['pinn_peak_sensor']}\n")
            f.write(f"  Ratio: {row['edf_peak_ppb'] / (row['pinn_peak_ppb'] + 1e-6):.1f}x\n\n")
    
    print(f"✓ Saved report to: {report_path}")
    print()
    print("="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"Visualizations saved to: {VIZ_DIR}")
    print(f"Total visualizations: {len(events) * 2}")

if __name__ == '__main__':
    main()

