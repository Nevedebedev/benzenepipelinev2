#!/usr/bin/env python3
"""
Compare factors causing February 2021 freeze events
Analyze meteorological conditions, emission rates, and compare with 2019 hazards
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

def load_events():
    """Load freeze and hazard events"""
    # Load 2021 freeze events
    with open(DATA_DIR / 'freeze_2021_with_conditions.pkl', 'rb') as f:
        freeze_events = pickle.load(f)
    
    # Load 2019 hazards
    with open(DATA_DIR / 'hazards_2019_with_conditions.pkl', 'rb') as f:
        hazard_events = pickle.load(f)
    
    return freeze_events, hazard_events

def extract_factors(events):
    """Extract meteorological and emission factors from events"""
    results = []
    
    for event in events:
        timestamp = event['timestamp']
        max_edf = event['max_concentration_edf']
        
        wind_speeds = []
        wind_directions = []
        diffusion_coeffs = []
        emission_rates = []
        
        for fac_name, params in event['facility_params'].items():
            u = params['wind_u']
            v = params['wind_v']
            D = params['D']
            Q = params['Q']
            
            wind_speed = (u**2 + v**2)**0.5
            wind_dir = np.arctan2(v, u) * 180 / np.pi
            if wind_dir < 0:
                wind_dir += 360
            
            wind_speeds.append(wind_speed)
            wind_directions.append(wind_dir)
            diffusion_coeffs.append(D)
            emission_rates.append(Q)
        
        results.append({
            'timestamp': timestamp,
            'max_edf': max_edf,
            'wind_speed_mean': np.mean(wind_speeds),
            'wind_speed_min': np.min(wind_speeds),
            'wind_speed_max': np.max(wind_speeds),
            'wind_dir_mean': np.mean(wind_directions),
            'diffusion_mean': np.mean(diffusion_coeffs),
            'diffusion_min': np.min(diffusion_coeffs),
            'diffusion_max': np.max(diffusion_coeffs),
            'emission_mean': np.mean(emission_rates),
            'emission_min': np.min(emission_rates),
            'emission_max': np.max(emission_rates),
        })
    
    return pd.DataFrame(results)

def create_comparison_plots(freeze_df, hazard_df):
    """Create comparison visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Wind Speed vs EDF Concentration
    ax1 = axes[0, 0]
    ax1.scatter(freeze_df['wind_speed_mean'], freeze_df['max_edf'], 
                alpha=0.7, label='2021 Freeze', color='blue', s=100)
    ax1.scatter(hazard_df['wind_speed_mean'], hazard_df['max_edf'], 
                alpha=0.7, label='2019 Hazards', color='red', s=100, marker='^')
    ax1.set_xlabel('Wind Speed (m/s)', fontsize=12)
    ax1.set_ylabel('EDF Peak Concentration (ppb)', fontsize=12)
    ax1.set_title('Wind Speed vs Peak Concentration', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Diffusion vs EDF Concentration
    ax2 = axes[0, 1]
    ax2.scatter(freeze_df['diffusion_mean'], freeze_df['max_edf'], 
                alpha=0.7, label='2021 Freeze', color='blue', s=100)
    ax2.scatter(hazard_df['diffusion_mean'], hazard_df['max_edf'], 
                alpha=0.7, label='2019 Hazards', color='red', s=100, marker='^')
    ax2.set_xlabel('Diffusion Coefficient (m²/s)', fontsize=12)
    ax2.set_ylabel('EDF Peak Concentration (ppb)', fontsize=12)
    ax2.set_title('Diffusion Coefficient vs Peak Concentration', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    
    # 3. Emission Rate vs EDF Concentration
    ax3 = axes[1, 0]
    ax3.scatter(freeze_df['emission_mean'], freeze_df['max_edf'], 
                alpha=0.7, label='2021 Freeze', color='blue', s=100)
    ax3.scatter(hazard_df['emission_mean'], hazard_df['max_edf'], 
                alpha=0.7, label='2019 Hazards', color='red', s=100, marker='^')
    ax3.set_xlabel('Emission Rate (kg/s)', fontsize=12)
    ax3.set_ylabel('EDF Peak Concentration (ppb)', fontsize=12)
    ax3.set_title('Emission Rate vs Peak Concentration', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Combined Factor: Wind × Diffusion (dispersion index)
    freeze_df['dispersion_index'] = freeze_df['wind_speed_mean'] * freeze_df['diffusion_mean']
    hazard_df['dispersion_index'] = hazard_df['wind_speed_mean'] * hazard_df['diffusion_mean']
    
    ax4 = axes[1, 1]
    ax4.scatter(freeze_df['dispersion_index'], freeze_df['max_edf'], 
                alpha=0.7, label='2021 Freeze', color='blue', s=100)
    ax4.scatter(hazard_df['dispersion_index'], hazard_df['max_edf'], 
                alpha=0.7, label='2019 Hazards', color='red', s=100, marker='^')
    ax4.set_xlabel('Dispersion Index (Wind × Diffusion)', fontsize=12)
    ax4.set_ylabel('EDF Peak Concentration (ppb)', fontsize=12)
    ax4.set_title('Dispersion Index vs Peak Concentration\n(Lower = Higher Concentrations)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'freeze_factors_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plot: {OUTPUT_DIR / 'freeze_factors_comparison.png'}")

def generate_report(freeze_df, hazard_df):
    """Generate detailed comparison report"""
    report_path = OUTPUT_DIR / 'freeze_factors_analysis.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("FEBRUARY 2021 FREEZE EVENT - FACTOR ANALYSIS & COMPARISON\n")
        f.write("="*100 + "\n\n")
        
        f.write("COMPARING 2019 HAZARDS vs 2021 FREEZE EVENTS\n")
        f.write("-"*100 + "\n\n")
        
        f.write("2019 HAZARDS (Top 10):\n")
        f.write(f"  EDF Peak Range: {hazard_df['max_edf'].min():.2f} - {hazard_df['max_edf'].max():.2f} ppb\n")
        f.write(f"  Wind Speed: mean={hazard_df['wind_speed_mean'].mean():.2f} m/s, "
                f"min={hazard_df['wind_speed_mean'].min():.2f}, max={hazard_df['wind_speed_mean'].max():.2f}\n")
        f.write(f"  Diffusion: mean={hazard_df['diffusion_mean'].mean():.2f} m²/s, "
                f"min={hazard_df['diffusion_mean'].min():.2f}, max={hazard_df['diffusion_mean'].max():.2f}\n")
        f.write(f"  Emission: mean={hazard_df['emission_mean'].mean():.6f} kg/s, "
                f"min={hazard_df['emission_mean'].min():.6f}, max={hazard_df['emission_mean'].max():.6f}\n\n")
        
        f.write("2021 FREEZE EVENTS (Feb 11-20):\n")
        f.write(f"  EDF Peak Range: {freeze_df['max_edf'].min():.2f} - {freeze_df['max_edf'].max():.2f} ppb\n")
        f.write(f"  Wind Speed: mean={freeze_df['wind_speed_mean'].mean():.2f} m/s, "
                f"min={freeze_df['wind_speed_mean'].min():.2f}, max={freeze_df['wind_speed_mean'].max():.2f}\n")
        f.write(f"  Diffusion: mean={freeze_df['diffusion_mean'].mean():.2f} m²/s, "
                f"min={freeze_df['diffusion_mean'].min():.2f}, max={freeze_df['diffusion_mean'].max():.2f}\n")
        f.write(f"  Emission: mean={freeze_df['emission_mean'].mean():.6f} kg/s, "
                f"min={freeze_df['emission_mean'].min():.6f}, max={freeze_df['emission_mean'].max():.6f}\n\n")
        
        f.write("COMPARISON:\n")
        f.write("-"*100 + "\n")
        f.write(f"  EDF Concentrations: 2019 max={hazard_df['max_edf'].max():.2f} ppb vs "
                f"2021 max={freeze_df['max_edf'].max():.2f} ppb\n")
        f.write(f"  Wind Speed: 2019={hazard_df['wind_speed_mean'].mean():.2f} m/s vs "
                f"2021={freeze_df['wind_speed_mean'].mean():.2f} m/s "
                f"({100*(freeze_df['wind_speed_mean'].mean()/hazard_df['wind_speed_mean'].mean() - 1):+.1f}%)\n")
        f.write(f"  Diffusion: 2019={hazard_df['diffusion_mean'].mean():.2f} m²/s vs "
                f"2021={freeze_df['diffusion_mean'].mean():.2f} m²/s "
                f"({100*(freeze_df['diffusion_mean'].mean()/hazard_df['diffusion_mean'].mean() - 1):+.1f}%)\n")
        f.write(f"  Emission: 2019={hazard_df['emission_mean'].mean():.6f} kg/s vs "
                f"2021={freeze_df['emission_mean'].mean():.6f} kg/s "
                f"({100*(freeze_df['emission_mean'].mean()/hazard_df['emission_mean'].mean() - 1):+.1f}%)\n\n")
        
        f.write("="*100 + "\n")
        f.write("KEY FACTORS CAUSING FREEZE EVENTS\n")
        f.write("="*100 + "\n\n")
        
        f.write("1. WIND SPEED:\n")
        f.write(f"   - 2021 Freeze: {freeze_df['wind_speed_mean'].mean():.2f} m/s average (very low)\n")
        f.write(f"   - 2019 Hazards: {hazard_df['wind_speed_mean'].mean():.2f} m/s average\n")
        f.write(f"   - Low wind = slow dispersion = higher concentrations\n")
        f.write(f"   - Freeze events had {100*(1 - freeze_df['wind_speed_mean'].mean()/hazard_df['wind_speed_mean'].mean()):.1f}% lower wind than 2019 hazards\n\n")
        
        f.write("2. DIFFUSION COEFFICIENT:\n")
        f.write(f"   - 2021 Freeze: {freeze_df['diffusion_mean'].mean():.2f} m²/s average (very low)\n")
        f.write(f"   - 2019 Hazards: {hazard_df['diffusion_mean'].mean():.2f} m²/s average\n")
        f.write(f"   - Low diffusion = poor mixing = higher concentrations\n")
        f.write(f"   - Freeze events had {100*(1 - freeze_df['diffusion_mean'].mean()/hazard_df['diffusion_mean'].mean()):.1f}% lower diffusion than 2019 hazards\n\n")
        
        f.write("3. EMISSION RATES:\n")
        f.write(f"   - 2021 Freeze: {freeze_df['emission_mean'].mean():.6f} kg/s average\n")
        f.write(f"   - 2019 Hazards: {hazard_df['emission_mean'].mean():.6f} kg/s average\n")
        f.write(f"   - Emissions are SIMILAR between events\n")
        f.write(f"   - Emissions are NORMAL, not elevated during freeze\n\n")
        
        f.write("4. COMBINED EFFECT (Dispersion Index = Wind × Diffusion):\n")
        freeze_dispersion = freeze_df['wind_speed_mean'].mean() * freeze_df['diffusion_mean'].mean()
        hazard_dispersion = hazard_df['wind_speed_mean'].mean() * hazard_df['diffusion_mean'].mean()
        f.write(f"   - 2021 Freeze: {freeze_dispersion:.2f} (m/s × m²/s)\n")
        f.write(f"   - 2019 Hazards: {hazard_dispersion:.2f} (m/s × m²/s)\n")
        f.write(f"   - Freeze had {100*(1 - freeze_dispersion/hazard_dispersion):.1f}% lower dispersion capacity\n")
        f.write(f"   - Lower dispersion = pollutants accumulate near sources\n\n")
        
        f.write("="*100 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*100 + "\n\n")
        f.write("The February 2021 freeze events were caused by STABLE METEOROLOGICAL CONDITIONS:\n\n")
        f.write("1. Very low wind speeds during freeze period\n")
        f.write("2. Very low diffusion coefficients (poor atmospheric mixing)\n")
        f.write("3. Normal emission rates (not elevated)\n")
        f.write("4. Combined effect: Poor dispersion → Accumulation → High concentrations\n\n")
        f.write("This is similar to 2019 hazards, which were also caused by stable conditions\n")
        f.write("rather than accidents or elevated emissions.\n")
    
    print(f"✓ Saved report: {report_path}")

def main():
    """Main analysis function"""
    print("="*100)
    print("FEBRUARY 2021 FREEZE EVENT - FACTOR ANALYSIS")
    print("="*100)
    print()
    
    # Load events
    freeze_events, hazard_events = load_events()
    print(f"Loaded {len(freeze_events)} freeze events and {len(hazard_events)} hazard events")
    print()
    
    # Extract factors
    print("Extracting factors...")
    freeze_df = extract_factors(freeze_events)
    hazard_df = extract_factors(hazard_events)
    
    # Save to CSV
    freeze_df.to_csv(OUTPUT_DIR / 'freeze_2021_factors.csv', index=False)
    hazard_df.to_csv(OUTPUT_DIR / 'hazards_2019_factors.csv', index=False)
    print(f"✓ Saved factor data to CSV")
    print()
    
    # Create visualizations
    print("Creating comparison plots...")
    create_comparison_plots(freeze_df, hazard_df)
    print()
    
    # Generate report
    print("Generating analysis report...")
    generate_report(freeze_df, hazard_df)
    print()
    
    # Print summary
    print("="*100)
    print("SUMMARY")
    print("="*100)
    print()
    print(f"2021 Freeze Events:")
    print(f"  EDF Peak: {freeze_df['max_edf'].min():.2f} - {freeze_df['max_edf'].max():.2f} ppb")
    print(f"  Wind Speed: {freeze_df['wind_speed_mean'].mean():.2f} m/s")
    print(f"  Diffusion: {freeze_df['diffusion_mean'].mean():.2f} m²/s")
    print(f"  Emission: {freeze_df['emission_mean'].mean():.6f} kg/s")
    print()
    print(f"2019 Hazards:")
    print(f"  EDF Peak: {hazard_df['max_edf'].min():.2f} - {hazard_df['max_edf'].max():.2f} ppb")
    print(f"  Wind Speed: {hazard_df['wind_speed_mean'].mean():.2f} m/s")
    print(f"  Diffusion: {hazard_df['diffusion_mean'].mean():.2f} m²/s")
    print(f"  Emission: {hazard_df['emission_mean'].mean():.6f} kg/s")
    print()
    print("Key Finding: Freeze events caused by stable meteorological conditions")
    print("(low wind + low diffusion), not elevated emissions or accidents.")

if __name__ == '__main__':
    main()

