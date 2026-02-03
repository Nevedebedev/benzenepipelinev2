"""
PINN Source Attribution Visualization
Shows which source contributes most to concentration at each location
Each source has its own color, marked with a star
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from pinn import ParametricADEPINN
import os

# Paths
BASE_DIR = "/Users/neevpratap/simpletesting"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_visualizations_q1_2021/source_attribution")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verified Source Metadata (from benzene_pipeline.py)
FACILITIES = [
    {'name': 'BASF_Pasadena', 'coords': (11473.88, 11613.95), 'diameter': 800.0},
    {'name': 'Chevron_Phillips_Chemical_Co', 'coords': (8781.44, 11769.53), 'diameter': 1885.0},
    {'name': 'Enterprise_Houston_Terminal', 'coords': (13701.44, 13525.43), 'diameter': 578.0},
    {'name': 'ExxonMobil_Baytown_Olefins_Plant', 'coords': (25003.9, 14792.35), 'diameter': 1050.0},
    {'name': 'ExxonMobil_Baytown_Refinery', 'coords': (24868.31, 13369.85), 'diameter': 3220.0},
    {'name': 'Goodyear_Baytown', 'coords': (21507.59, 2623.29), 'diameter': 532.0},
    {'name': 'Huntsman_International', 'coords': (500.72, 11147.19), 'diameter': 193.0},
    {'name': 'INEOS_PP_&_Gemini', 'coords': (17798.22, 11091.62), 'diameter': 530.0},
    {'name': 'INEOS_Phenol', 'coords': (10476.32, 12102.93), 'diameter': 2636.0},
    {'name': 'ITC_Deer_Park', 'coords': (17013.73, 12847.52), 'diameter': 1168.0},
    {'name': 'Invista', 'coords': (1895.36, 8991.21), 'diameter': 490.0},
    {'name': 'K-Solv_Channelview', 'coords': (16122.71, 16025.92), 'diameter': 143.0},
    {'name': 'LyondellBasell_Bayport_Polymers', 'coords': (21187.99, 500.65), 'diameter': 2130.0},
    {'name': 'LyondellBasell_Channelview_Complex', 'coords': (14970.18, 23127.32), 'diameter': 2515.0},
    {'name': 'LyondellBasell_Pasadena_Complex', 'coords': (3290.01, 9980.29), 'diameter': 1914.0},
    {'name': 'Oxy_Vinyls_Deer_Park', 'coords': (16016.17, 11925.12), 'diameter': 1008.0},
    {'name': 'Shell_Deer_Park_Refinery', 'coords': (13817.66, 10836.02), 'diameter': 2740.0},
    {'name': 'TPC_Group', 'coords': (1488.59, 8591.13), 'diameter': 1032.0},
    {'name': 'Total_Energies_Petrochemicals', 'coords': (17769.16, 11613.95), 'diameter': 667.0},
    {'name': 'Valero_Houston_Refinery', 'coords': (1517.65, 10991.6), 'diameter': 752.0},
]

print("="*70)
print("PINN SOURCE ATTRIBUTION MAPS")
print("="*70)

# Load PINN
print("\n[1/4] Loading PINN...")
pinn = ParametricADEPINN()
checkpoint = torch.load(os.path.join(BASE_DIR, "pinn_combined_final.pth 2"),  
                        map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']
filtered_state_dict = {k: v for k, v in state_dict.items() 
                       if not k.endswith('_min') and not k.endswith('_max')}
pinn.load_state_dict(filtered_state_dict, strict=False)

# Override with benchmark normalization ranges
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
print("  ✓ PINN loaded")

# Load meteorology
print("\n[2/4] Loading meteorology data...")
met_path = "/Users/neevpratap/Desktop/madis_data_desktop_updated/training_data_2021_full_jan_REPAIRED/BASF_Pasadena_training_data.csv"
met_df = pd.read_csv(met_path)
met_df['t'] = pd.to_datetime(met_df['t'])
met_df_valid = met_df.dropna(subset=['wind_u', 'wind_v', 'D'])

# Select 10 diverse timestamps
print("\n[3/4] Selecting timestamps...")
timestamp_indices = np.linspace(50, len(met_df_valid)-50, 10, dtype=int)
selected_timestamps = [met_df_valid['t'].iloc[idx] for idx in timestamp_indices]
print(f"  Selected {len(selected_timestamps)} timestamps")

# Create spatial grid
print("\n[4/4] Creating spatial grid...")
grid_resolution = 80  # Balance between detail and computation time
x_grid = np.linspace(0, 30000, grid_resolution)
y_grid = np.linspace(0, 30000, grid_resolution)
X, Y = np.meshgrid(x_grid, y_grid)

# Create color map for 20 sources
colors = plt.cm.tab20(np.linspace(0, 1, 20))

print("\n" + "="*70)
print("GENERATING SOURCE ATTRIBUTION MAPS")
print("="*70)

for viz_idx, representative_ts in enumerate(selected_timestamps, 1):
    print(f"\n[{viz_idx}/10] Processing {representative_ts.strftime('%Y-%m-%d %H:%M')}")
    
    # Get meteorology
    met_row = met_df[met_df['t'] == representative_ts].iloc[0]
    u_val = met_row['wind_u']
    v_val = met_row['wind_v']
    D_val = met_row['D']
    t_start = pd.to_datetime('2021-01-01 00:00:00')
    t_hours = (representative_ts - t_start).total_seconds() / 3600.0
    
    print(f"  Computing PINN for {len(FACILITIES)} sources...")
    
    # Store PINN predictions for each source
    source_fields = []
    Q_val = 0.001  # Standard emission rate
    
    for facility in FACILITIES:
        source_x, source_y = facility['coords']
        source_d = facility['diameter']
        
        # Compute PINN field for this source
        source_field = np.zeros_like(X)
        
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                x_pt = X[i, j]
                y_pt = Y[i, j]
                
                x_t = torch.tensor([[x_pt]], dtype=torch.float32)
                y_t = torch.tensor([[y_pt]], dtype=torch.float32)
                t_t = torch.tensor([[t_hours]], dtype=torch.float32)
                cx_t = torch.tensor([[source_x]], dtype=torch.float32)
                cy_t = torch.tensor([[source_y]], dtype=torch.float32)
                u_t = torch.tensor([[u_val]], dtype=torch.float32)
                v_t = torch.tensor([[v_val]], dtype=torch.float32)
                d_t = torch.tensor([[source_d]], dtype=torch.float32)
                kappa_t = torch.tensor([[D_val]], dtype=torch.float32)
                Q_t = torch.tensor([[Q_val]], dtype=torch.float32)
                
                with torch.no_grad():
                    phi = pinn(x_t, y_t, t_t, cx_t, cy_t, u_t, v_t, 
                              d_t, kappa_t, Q_t, normalize=True)
                
                source_field[i, j] = max(phi.item() * 3.13e8, 0.0)
        
        source_fields.append(source_field)
    
    # Determine dominant source at each location
    source_fields_array = np.array(source_fields)  # Shape: (20, grid_res, grid_res)
    dominant_source = np.argmax(source_fields_array, axis=0)  # Index of dominant source
    max_concentration = np.max(source_fields_array, axis=0)  # Maximum concentration
    
    print(f"  Creating visualization...")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Create custom colormap for sources
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, 20.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Only show areas with concentration > threshold
    threshold = 0.01  # ppb
    masked_dominant = np.ma.masked_where(max_concentration < threshold, dominant_source)
    
    # Plot dominant source map
    im = ax.pcolormesh(X, Y, masked_dominant, cmap=cmap, norm=norm, alpha=0.7, shading='auto')
    
    # Mark each source with a star
    for idx, facility in enumerate(FACILITIES):
        source_x, source_y = facility['coords']
        ax.scatter(source_x, source_y, c=[colors[idx]], s=400, marker='*', 
                  edgecolors='black', linewidth=2, zorder=10, 
                  label=facility['name'].replace('_', ' '))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=range(20), pad=0.02)
    cbar.ax.set_yticklabels([f.get('name', '').replace('_', ' ')[:20] for f in FACILITIES], 
                            fontsize=7)
    cbar.set_label('Dominant Source', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('X Coordinate (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontsize=13, fontweight='bold')
    ax.set_title(f'PINN Source Attribution Map\n{representative_ts.strftime("%Y-%m-%d %H:%M")} | ' +
                 f'Wind: u={u_val:.2f} m/s, v={v_val:.2f} m/s | D={D_val:.2f}',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add text showing max concentration
    total_conc = np.sum(source_fields_array, axis=0)
    ax.text(0.02, 0.98, f'Max Total: {total_conc.max():.2f} ppb', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"source_attribution_{viz_idx:02d}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: source_attribution_{viz_idx:02d}.png")
    print(f"  Max concentration: {total_conc.max():.2f} ppb")

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
print(f"Generated 10 source attribution maps in:")
print(f"  {OUTPUT_DIR}")
print("="*70)
