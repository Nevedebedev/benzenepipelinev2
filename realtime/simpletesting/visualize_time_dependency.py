#!/usr/bin/env python3
"""
Visualize PINN time dependency issue
Shows how PINN predictions vary with timestamp for identical conditions
"""

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append('/Users/neevpratap/simpletesting')
from pinn import ParametricADEPINN

def test_pinn_time_dependency():
    """Test PINN with identical conditions but different timestamps"""
    
    # Load PINN model
    pinn = ParametricADEPINN()
    checkpoint = torch.load('/Users/neevpratap/simpletesting/pinn_combined_final.pth 2', 
                          map_location='cpu', weights_only=False)
    pinn.load_state_dict(checkpoint)
    pinn.eval()
    
    # Fixed conditions (identical across all tests)
    x = 500.0
    y = 500.0
    source_x = 0.0
    source_y = 0.0
    wind_u = 0.63
    wind_v = 0.63
    D = 14.88
    Q = 0.0008
    source_diameter = 1300.0
    
    # Test across different months of 2019
    results = []
    
    # January to December (using 15th of each month at noon)
    for month in range(1, 13):
        timestamp = pd.Timestamp(f'2019-{month:02d}-15 12:00:00')
        t_hours = (timestamp - pd.Timestamp('2019-01-01')).total_seconds() / 3600.0
        
        # Normalize inputs
        x_norm = (x - source_x - pinn.x_min) / (pinn.x_max - pinn.x_min)
        y_norm = (y - source_y - pinn.y_min) / (pinn.y_max - pinn.y_min)
        t_norm = (t_hours - pinn.t_min) / (pinn.t_max - pinn.t_min)
        u_norm = (wind_u - pinn.u_min) / (pinn.u_max - pinn.u_min)
        v_norm = (wind_v - pinn.v_min) / (pinn.v_max - pinn.v_min)
        
        # Run PINN
        with torch.no_grad():
            inputs = torch.tensor([[x_norm, y_norm, t_norm, u_norm, v_norm]], dtype=torch.float32)
            phi = pinn(inputs).item()
        
        # Convert to ppb
        concentration_ppb = phi * 3.13e8 / (D * source_diameter)
        
        results.append({
            'month': month,
            'month_name': timestamp.strftime('%B'),
            't_hours': t_hours,
            'phi': phi,
            'concentration_ppb': concentration_ppb
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print results
    print("=" * 80)
    print("PINN TIME DEPENDENCY TEST")
    print("=" * 80)
    print("\n🔧 Fixed Conditions (identical for all tests):")
    print(f"  Location: ({x}, {y}) from source at ({source_x}, {source_y})")
    print(f"  Wind: u={wind_u}, v={wind_v} (speed = {np.sqrt(wind_u**2 + wind_v**2):.2f} m/s)")
    print(f"  Diffusion: D={D}")
    print(f"  Emissions: Q={Q}")
    print(f"  Source diameter: {source_diameter} m")
    print("\n📅 Only changing: Timestamp (month)\n")
    
    print(f"{'Month':<10} | {'t_hours':>8} | {'PINN Raw (φ)':>15} | {'Concentration (ppb)':>20}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['month_name']:<10} | {row['t_hours']:>8.1f} | {row['phi']:>15.6e} | {row['concentration_ppb']:>20.2f}")
    
    # Calculate statistics
    min_conc = df['concentration_ppb'].min()
    max_conc = df['concentration_ppb'].max()
    ratio = max_conc / min_conc if min_conc > 0 else float('inf')
    
    print("\n" + "=" * 80)
    print("📊 STATISTICS")
    print("=" * 80)
    print(f"Minimum concentration: {min_conc:.2f} ppb ({df.loc[df['concentration_ppb'].idxmin(), 'month_name']})")
    print(f"Maximum concentration: {max_conc:.2f} ppb ({df.loc[df['concentration_ppb'].idxmax(), 'month_name']})")
    print(f"Max/Min ratio: {ratio:.2f}x")
    print(f"\n⚠️  This {ratio:.2f}x variation is PURELY from timestamp!")
    print("   All meteorological conditions were IDENTICAL!")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Concentration vs Month
    ax1.plot(df['month'], df['concentration_ppb'], 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PINN Prediction (ppb)', fontsize=12, fontweight='bold')
    ax1.set_title('PINN Predictions Across Year - Identical Conditions', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['month'])
    ax1.set_xticklabels(df['month_name'].str[:3], rotation=45)
    
    # Add annotation for max and min
    max_idx = df['concentration_ppb'].idxmax()
    min_idx = df['concentration_ppb'].idxmin()
    ax1.annotate(f'MAX\n{max_conc:.1f} ppb', 
                xy=(df.loc[max_idx, 'month'], max_conc),
                xytext=(df.loc[max_idx, 'month'], max_conc + 20),
                ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.annotate(f'MIN\n{min_conc:.1f} ppb',
                xy=(df.loc[min_idx, 'month'], min_conc),
                xytext=(df.loc[min_idx, 'month'], min_conc + 10),
                ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Plot 2: Bar chart showing the issue
    colors = ['green' if i == min_idx else 'red' if i == max_idx else '#3498db' 
              for i in range(len(df))]
    ax2.bar(df['month'], df['concentration_ppb'], color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0.4, color='orange', linestyle='--', linewidth=2, label='Typical Sensor Reading (~0.4 ppb)')
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PINN Prediction (ppb)', fontsize=12, fontweight='bold')
    ax2.set_title('PINN vs Reality: Time Dependency Bug', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(df['month'])
    ax2.set_xticklabels(df['month_name'].str[:3], rotation=45)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = '/Users/neevpratap/simpletesting/pinn_time_dependency_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    
    plt.show()
    
    return df

if __name__ == "__main__":
    df = test_pinn_time_dependency()
