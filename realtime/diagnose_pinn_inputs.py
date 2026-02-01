#!/usr/bin/env python3
"""
Diagnostic script to check PINN inputs and outputs
Compares real-time pipeline values with training data ranges
"""

import sys
sys.path.append('/Users/neevpratap/simpletesting')

import pandas as pd
import numpy as np
import torch  
from pinn import ParametricADEPINN
from datetime import datetime

# Load PINN
print("Loading PINN model...")
pinn = ParametricADEPINN()
cp = torch.load("/Users/neevpratap/simpletesting/pinn_combined_final.pth 2", 
                map_location='cpu', weights_only=False)
pinn.load_state_dict(cp['model_state_dict'])
pinn.eval()

# Test case: ExxonMobil Baytown Refinery at a sensor location
# Sensor 482010026 at (13972.62, 19915.57)
print("\n" + "="*70)
print("TEST 1: Single sensor location, typical training conditions")
print("="*70)

# Typical training values (from Jan 2021 data)
x = 13972.62
y = 19915.57
t = 100.0  # hours into year
source_x = 13263.0
source_y = -695.0 
source_d = 3220.0
Q_training = 0.003  # Peak weekday
wind_u = 0.56
wind_v = -2.62
D = 108.79

print(f"\nInputs (TRAINING-LIKE):")
print(f"  Sensor location: ({x:.2f}, {y:.2f})")
print(f"  Source: ({source_x:.2f}, {source_y:.2f}), diameter={source_d:.2f}")
print(f"  Q = {Q_training:.6f} kg/s (peak)")
print(f"  Wind: u={wind_u:.2f}, v={wind_v:.2f}")
print(f"  Diffusion D = {D:.2f}")

# Run PINN
x_t = torch.tensor([[x]], dtype=torch.float32)
y_t = torch.tensor([[y]], dtype=torch.float32)
t_t = torch.tensor([[t]], dtype=torch.float32)
cx_t = torch.tensor([[source_x]], dtype=torch.float32)
cy_t = torch.tensor([[source_y]], dtype=torch.float32)
u_t = torch.tensor([[wind_u]], dtype=torch.float32)
v_t = torch.tensor([[wind_v]], dtype=torch.float32)
d_t = torch.tensor([[source_d]], dtype=torch.float32)
kappa_t = torch.tensor([[D]], dtype=torch.float32)
Q_t = torch.tensor([[Q_training]], dtype=torch.float32)

with torch.no_grad():
    phi = pinn(x_t, y_t, t_t, cx_t, cy_t, u_t, v_t, d_t, kappa_t, Q_t, normalize=True)

concentration = phi.item() * 3.13e8
print(f"\nPINN output:")
print(f"  phi (raw) = {phi.item():.6e}")
print(f"  Concentration = {concentration:.4f} ppb")

# Now test real-time pipeline conditions
print("\n" + "="*70)
print("TEST 2: Real-time pipeline conditions")
print("="*70)

Q_realtime = 0.0008  # Off-peak
wind_u_rt = 5.02
wind_v_rt = -2.54
D_rt = 12.71
t_rt = 8760.0  # End of year (Feb 2026 is extrapolation)

print(f"\nInputs (REAL-TIME):")
print(f"  Sensor location: ({x:.2f}, {y:.2f})")
print(f"  Source: ({source_x:.2f}, {source_y:.2f}), diameter={source_d:.2f}")
print(f"  Q = {Q_realtime:.6f} kg/s (off-peak - 3.75x smaller!)")
print(f"  Wind: u={wind_u_rt:.2f}, v={wind_v_rt:.2f}")
print(f"  Diffusion D = {D_rt:.2f}")
print(f"  Time: t={t_rt:.1f} hours (Feb 2026 extrapolation)")

Q_t_rt = torch.tensor([[Q_realtime]], dtype=torch.float32)
u_t_rt = torch.tensor([[wind_u_rt]], dtype=torch.float32)
v_t_rt = torch.tensor([[wind_v_rt]], dtype=torch.float32)
kappa_t_rt = torch.tensor([[D_rt]], dtype=torch.float32)
t_t_rt = torch.tensor([[t_rt]], dtype=torch.float32)

with torch.no_grad():
    phi_rt = pinn(x_t, y_t, t_t_rt, cx_t, cy_t, u_t_rt, v_t_rt, d_t, kappa_t_rt, Q_t_rt, normalize=True)

concentration_rt = phi_rt.item() * 3.13e8
print(f"\nPINN output:")
print(f"  phi (raw) = {phi_rt.item():.6e}")
print(f"  Concentration = {concentration_rt:.4f} ppb")

print(f"\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"Q ratio (real-time / training): {Q_realtime/Q_training:.3f}x")
print(f"Concentration ratio: {concentration_rt/concentration:.3f}x")
print(f"\nExpected: With {Q_realtime/Q_training:.3f}x less emissions, concentration should be ~{Q_realtime/Q_training:.3f}x lower")
print(f"Actual: Concentration is {concentration_rt/concentration:.3f}x {'higher' if concentration_rt > concentration else 'lower'}")
