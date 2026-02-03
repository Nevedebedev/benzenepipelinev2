#!/usr/bin/env python3
"""
Simple diagnostic test showing PINN time dependency
Tests same conditions at different timestamps
"""

import torch
import pandas as pd
import numpy as np
import os
from pinn import ParametricADEPINN

# Load PINN model
print("Loading PINN model...")
pinn = ParametricADEPINN()
checkpoint = torch.load('/Users/neevpratap/simpletesting/pinn_combined_final.pth 2',  
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
print("✓ PINN loaded\n")

# Fixed conditions for all tests
print("=" * 80)
print("PINN TIME DEPENDENCY DIAGNOSTIC TEST")
print("=" * 80)
print("\n🔧 Fixed Conditions (IDENTICAL for all tests):")
print("  Sensor location: (13972.62, 19915.57)")
print("  Source location: (0, 0)")
print("  Wind: u=0.63, v=0.63 (speed = 0.89 m/s)")
print("  Diffusion (kappa): D=14.88")
print("  Emissions: Q=0.0008")
print("  Source diameter: 1300 m")
print("\n📅 Only variable: Timestamp (month of 2019)\n")

# Test parameters (FIXED)
sensor_x = 13972.62
sensor_y = 19915.57
source_x = 0.0
source_y = 0.0
wind_u = 0.63
wind_v = 0.63
D = 14.88
Q = 0.0008
source_diameter = 1300.0

# Test across different months
results = []
print(f"{'Month':<12} | {'t_hours':>8} | {'PINN Raw (φ)':>15} | {'Concentration (ppb)':>20}")
print("-" * 80)

for month in range(1, 13):
    # Create timestamp (middle of month at noon)
    timestamp = pd.Timestamp(f'2019-{month:02d}-15 12:00:00')
    t_start = pd.Timestamp('2019-01-01 00:00:00')
    t_hours = (timestamp - t_start).total_seconds() / 3600.0
    
    # Prepare tensors
    x_t = torch.tensor([[sensor_x]], dtype=torch.float32)
    y_t = torch.tensor([[sensor_y]], dtype=torch.float32)
    t_t = torch.tensor([[t_hours]], dtype=torch.float32)
    cx_t = torch.tensor([[source_x]], dtype=torch.float32)
    cy_t = torch.tensor([[source_y]], dtype=torch.float32)
    u_t = torch.tensor([[wind_u]], dtype=torch.float32)
    v_t = torch.tensor([[wind_v]], dtype=torch.float32)
    d_t = torch.tensor([[source_diameter]], dtype=torch.float32)
    kappa_t = torch.tensor([[D]], dtype=torch.float32)
    Q_t = torch.tensor([[Q]], dtype=torch.float32)
    
    # PINN prediction
    with torch.no_grad():
        phi = pinn(x_t, y_t, t_t, cx_t, cy_t, u_t, v_t, d_t, kappa_t, Q_t, normalize=True)
    
    # Convert to ppb
    concentration_ppb = phi.item() * 3.13e8
    
    month_name = timestamp.strftime('%B')
    print(f"{month_name:<12} | {t_hours:>8.1f} | {phi.item():>15.6e} | {concentration_ppb:>20.2f}")
    
    results.append({
        'month': month,
        'month_name': month_name,
        't_hours': t_hours,
        'phi': phi.item(),
        'concentration_ppb': concentration_ppb
    })

# Calculate statistics
df = pd.DataFrame(results)
min_conc = df['concentration_ppb'].min()
max_conc = df['concentration_ppb'].max()
ratio = max_conc / min_conc if min_conc > 0 else float('inf')

print("\n" + "=" * 80)
print("📊 STATISTICS")
print("=" * 80)
print(f"Minimum concentration: {min_conc:.2f} ppb ({df.loc[df['concentration_ppb'].idxmin(), 'month_name']})")
print(f"Maximum concentration: {max_conc:.2f} ppb ({df.loc[df['concentration_ppb'].idxmax(), 'month_name']})")
print(f"Max/Min ratio: {ratio:.2f}x")
print(f"\n⚠️  CRITICAL: This {ratio:.2f}x variation is PURELY from changing the timestamp!")
print("   All meteorological conditions and emissions were IDENTICAL!")
print("   For steady-state physics, this should be ~1.0x (no variation)")

# Compare specific months
jan_conc = df[df['month'] == 1]['concentration_ppb'].values[0]
oct_conc = df[df['month'] == 10]['concentration_ppb'].values[0]
jan_oct_ratio = oct_conc / jan_conc if jan_conc > 0 else float('inf')

print(f"\n🔍 January vs October comparison:")
print(f"  January (t_hours ~ {df[df['month'] == 1]['t_hours'].values[0]:.0f}): {jan_conc:.2f} ppb")
print(f"  October (t_hours ~ {df[df['month'] == 10]['t_hours'].values[0]:.0f}): {oct_conc:.2f} ppb")
print(f"  Ratio (Oct/Jan): {jan_oct_ratio:.2f}x")
print("\n  This explains why NN2 degrades!")
print("  - NN2 trained on October spikes (high t_hours)")
print("  - January has low t_hours → different PINN distribution")
print("  - NN2 scalers see January as out-of-distribution")

print("\n" + "=" * 80)
print("✅ RECOMMENDATION")
print("=" * 80)
print("Fix PINN by using FIXED t_hours (e.g., t=100) instead of actual timestamps")
print("This will make PINN predictions depend only on physics, not calendar date")
print("=" * 80 + "\n")
