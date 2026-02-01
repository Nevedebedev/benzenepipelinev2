"""
Compare PINN outputs from pipeline vs benchmark

This will help identify if there's a systematic difference in how
the PINN is being called or if the outputs differ.
"""

import pandas as pd
import numpy as np
import torch
from benzene_pipeline import BenzenePipeline
import glob
import os

# Paths
DRIVE_DIR = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001"
BASE_DIR = "/Users/neevpratap/simpletesting"

# Sensor Coordinates (from drive data)
SENSORS = [
    {'id': 'sensor_482010026', 'x': 13970.0, 'y': 19920.0},
    {'id': 'sensor_482010057', 'x': 3020.0, 'y': 12330.0},
    {'id': 'sensor_482010069', 'x': 820.0, 'y': 9220.0},
    {'id': 'sensor_482010617', 'x': 27050.0, 'y': 22050.0},
    {'id': 'sensor_482010803', 'x': 8840.0, 'y': 15720.0},
    {'id': 'sensor_482011015', 'x': 18410.0, 'y': 15070.0},
    {'id': 'sensor_482011035', 'x': 1160.0, 'y': 12270.0},
    {'id': 'sensor_482011039', 'x': 13660.0, 'y': 5190.0},
    {'id': 'sensor_482016000', 'x': 1550.0, 'y': 6790.0}
]

def compare_pinn_outputs():
    """Compare pipeline PINN outputs to benchmark realtime concentrations"""
    
    # Initialize Pipeline
    print("Initializing Pipeline...")
    pinn_path = os.path.join(BASE_DIR, "pinn_combined_final.pth 2")
    nn2_path = os.path.join(BASE_DIR, "nn2_master_model_spatial.pth")
    pipeline = BenzenePipeline(pinn_path, nn2_path)
    
    # Load benchmark realtime concentrations
    print("Loading benchmark realtime concentrations...")
    benchmark_path = os.path.join(DRIVE_DIR, "total_superimposed_realtime_concentrations.csv")
    df_bench = pd.read_csv(benchmark_path)
    df_bench['t'] = pd.to_datetime(df_bench['t'])
    
    # Load facility data
    print("Loading facility data...")
    facility_files = glob.glob(os.path.join(DRIVE_DIR, "*_synced_training_data.csv"))
    facilities = {}
    for fpath in facility_files:
        fname = os.path.basename(fpath).replace("_synced_training_data.csv", "")
        df = pd.read_csv(fpath)
        df['t'] = pd.to_datetime(df['t'])
        facilities[fname] = df
    
    # Prepare sensor points
    sensor_points = np.array([[s['x'], s['y']] for s in SENSORS])
    
    # Test on a subset of timestamps
    test_timestamps = sorted(df_bench['t'].unique())[:50]
    
    comparisons = []
    
    for i, ts in enumerate(test_timestamps):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(test_timestamps)}: {ts}")
        
        # Get benchmark values for this timestamp
        bench_rows = df_bench[df_bench['t'] == ts]
        if bench_rows.empty:
            continue
        
        # Get meteorology from benchmark
        met_data = {
            'u': float(bench_rows.iloc[0]['wind_u']),
            'v': float(bench_rows.iloc[0]['wind_v']),
            'D': float(bench_rows.iloc[0]['D']),
            't': 3600.0
        }
        
        # Get emission rates from facility CSVs
        emissions = {}
        for fname, df in facilities.items():
            row_fac = df[df['t'] == ts]
            if not row_fac.empty:
                Q_val = float(row_fac.iloc[0]['Q_total'])
                emissions[fname] = Q_val
        
        # Run pipeline PINN
        try:
            raw_phi = pipeline.superimpose(met_data, sensor_points, emissions)
            pipeline_ppb = raw_phi * 313210039.9
            
            # Compare with benchmark for each sensor
            for j, sensor in enumerate(SENSORS):
                sensor_id_numeric = int(sensor['id'].replace('sensor_', ''))
                
                # Get benchmark value for this sensor
                bench_sensor = bench_rows[bench_rows['sensor_id'] == sensor_id_numeric]
                if not bench_sensor.empty:
                    bench_ppb = float(bench_sensor.iloc[0]['total_phi_ppb'])
                    bench_phi = float(bench_sensor.iloc[0]['total_phi_kg_m3'])
                    
                    comparisons.append({
                        'timestamp': ts,
                        'sensor_id': sensor['id'],
                        'pipeline_phi': raw_phi[j],
                        'benchmark_phi': bench_phi,
                        'pipeline_ppb': pipeline_ppb[j],
                        'benchmark_ppb': bench_ppb,
                        'phi_diff': raw_phi[j] - bench_phi,
                        'ppb_diff': pipeline_ppb[j] - bench_ppb,
                        'phi_ratio': raw_phi[j] / bench_phi if bench_phi != 0 else np.nan,
                        'ppb_ratio': pipeline_ppb[j] / bench_ppb if bench_ppb != 0 else np.nan
                    })
        except Exception as e:
            print(f"  Error at {ts}: {e}")
            continue
    
    # Create results dataframe
    df_comp = pd.DataFrame(comparisons)
    
    # Analysis
    print("\n" + "="*70)
    print("PINN OUTPUT COMPARISON: Pipeline vs Benchmark")
    print("="*70)
    
    # Remove NaN and inf values for statistics
    df_valid = df_comp.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(df_valid) > 0:
        print(f"\nValid comparisons: {len(df_valid)}")
        
        print("\n--- Raw Phi (kg/m³) ---")
        print(f"Pipeline mean: {df_valid['pipeline_phi'].mean():.2e}")
        print(f"Benchmark mean: {df_valid['benchmark_phi'].mean():.2e}")
        print(f"Mean absolute difference: {df_valid['phi_diff'].abs().mean():.2e}")
        print(f"Mean ratio (pipeline/benchmark): {df_valid['phi_ratio'].mean():.4f}")
        print(f"Median ratio: {df_valid['phi_ratio'].median():.4f}")
        
        print("\n--- PPB Values ---")
        print(f"Pipeline mean: {df_valid['pipeline_ppb'].mean():.4f}")
        print(f"Benchmark mean: {df_valid['benchmark_ppb'].mean():.4f}")
        print(f"Mean absolute difference: {df_valid['ppb_diff'].abs().mean():.4f}")
        print(f"Mean ratio (pipeline/benchmark): {df_valid['ppb_ratio'].mean():.4f}")
        print(f"Median ratio: {df_valid['ppb_ratio'].median():.4f}")
        
        # Check for systematic scaling
        print("\n--- Ratio Statistics ---")
        print(f"Phi ratio std dev: {df_valid['phi_ratio'].std():.4f}")
        print(f"PPB ratio std dev: {df_valid['ppb_ratio'].std():.4f}")
        
        # Save detailed comparison
        output_path = os.path.join(BASE_DIR, "pinn_output_comparison.csv")
        df_comp.to_csv(output_path, index=False)
        print(f"\nDetailed comparison saved to: {output_path}")
        
        # Sample of differences
        print("\n--- Sample Comparisons ---")
        print(df_valid[['timestamp', 'sensor_id', 'pipeline_ppb', 'benchmark_ppb', 'ppb_diff']].head(10))
    else:
        print("\nNo valid comparisons found!")

if __name__ == "__main__":
    compare_pinn_outputs()
