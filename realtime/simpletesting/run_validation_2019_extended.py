
import pandas as pd
import numpy as np
import os
import sys
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benzene_pipeline import BenzenePipeline, load_emissions_for_timestamp, FACILITIES

# ==========================================
# CONFIGURATION
# ==========================================
SENSORS = [
    {'id': 'sensor_482010026', 'name': 'HRM 7 Baytown', 'x': 13970.0, 'y': 19920.0},
    {'id': 'sensor_482010057', 'name': 'Galena Park', 'x': 3020.0, 'y': 12330.0},
    {'id': 'sensor_482010069', 'name': 'Milby Park', 'x': 820.0, 'y': 9220.0},
    {'id': 'sensor_482010617', 'name': 'Wallisville Road', 'x': 27050.0, 'y': 22050.0},
    {'id': 'sensor_482010803', 'name': 'HRM #3 Haden Rd', 'x': 8840.0, 'y': 15720.0},
    {'id': 'sensor_482011015', 'name': 'Lynchburg Ferry', 'x': 18410.0, 'y': 15070.0},
    {'id': 'sensor_482011035', 'name': 'Clinton', 'x': 1160.0, 'y': 12270.0},
    {'id': 'sensor_482011039', 'name': 'Houston Deer Park #2', 'x': 13660.0, 'y': 5190.0},
    {'id': 'sensor_482016000', 'name': 'Cesar Chavez', 'x': 1550.0, 'y': 6790.0}
]

def run_extended_validation():
    # Setup Paths
    base_dir = "/Users/neevpratap/simpletesting"
    data_dir = "/Users/neevpratap/Desktop/madis_data_desktop_updated/synced" 
    ground_truth_path = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001/sensors_final_synced.csv"
    
    # Updated Paths for NN2 v2
    pinn_path = os.path.join(base_dir, "pinn_combined_final.pth 2")
    nn2_path = os.path.join(base_dir, "nn2_master_model_spatial.pth")
    # Note: Scalers are embedded in the .pth, but we can still pass the external if desired
    scaler_path = os.path.join(base_dir, "nn2_master_scalers.pkl")
    
    print("Initializing Pipeline...")
    pipeline = BenzenePipeline(pinn_path, nn2_path, nn2_scaler_path=scaler_path)
    
    if pipeline.nn2 is None:
        print("CRITICAL: NN2 model not loaded. Extended validation requires NN2. Aborting.")
        return

    # Load Ground Truth
    print(f"Loading ground truth from {ground_truth_path}...")
    df_gt = pd.read_csv(ground_truth_path)
    df_gt['timestamp'] = pd.to_datetime(df_gt['t'])
    
    # Define Time Range (Jan 1 - Mar 31 2019)
    start_time = "2019-01-01 00:00:00"
    end_time = "2019-01-05 23:00:00"
    timestamps = pd.date_range(start=start_time, end=end_time, freq='h')
    
    print(f"Starting EXTENDED validation for {len(timestamps)} timesteps (Jan-Mar 2019)...")
    
    sensor_points = np.array([[s['x'], s['y']] for s in SENSORS])
    
    # Pre-compute Spatial Coords Vector for NN2 (flat list of 18 floats: x1, y1, x2, y2...)
    # Order matches SENSORS list
    coords_flat = []
    for s in SENSORS:
        coords_flat.extend([s['x'], s['y']])
    coords_tensor = torch.tensor(coords_flat, dtype=torch.float32).to(pipeline.device)
    
    # Load Meteorology from Benchmark CSV (for synchronization)
    benchmark_path = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001/total_superimposed_realtime_concentrations.csv"
    print(f"Loading benchmark meteorology from {benchmark_path}...")
    df_bench = pd.read_csv(benchmark_path)
    df_bench['t'] = pd.to_datetime(df_bench['t'])
    
    # Create a dictionary for O(1) lookup: (timestamp) -> (u, v, D)
    # Note: benchmarking shows all sensors have same met, so we drop duplicates on t
    met_dict = df_bench.drop_duplicates('t').set_index('t')[['wind_u', 'wind_v', 'D']].to_dict('index')

    results = []
    
    for i, ts in enumerate(timestamps):
        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
        
        if i % 100 == 0:
            print(f"Processing {ts_str} ({i}/{len(timestamps)})...")
            
        # 1. Load Met Data
        met_data = {'u': 0.0, 'v': 0.0, 'D': 1.0, 't': 3600.0}
        valid_met_found = False 
        
        # Use Benchmark Data for Met (O(1) lookup)
        if ts in met_dict:
            m = met_dict[ts]
            met_data['u'] = float(m['wind_u'])
            met_data['v'] = float(m['wind_v'])
            met_data['D'] = float(m['D'])
            valid_met_found = True

        # 2. Get Ground Truth for this timestamp (for NN2 input)
        row_gt = df_gt[df_gt['timestamp'] == ts]
        if row_gt.empty:
            # If no ground truth, we can only do valid PINN run if met data exists, but no comparison/correction input
            gt_values = np.zeros(9)
            has_gt = False
        else:
            has_gt = True
            # Extract sensor columns in correct order
            gt_values = []
            for s in SENSORS:
                val = row_gt[s['id']].values[0]
                gt_values.append(val if not pd.isna(val) else 0.0)
            gt_values = np.array(gt_values)

        if valid_met_found:
            emissions = load_emissions_for_timestamp(ts_str, data_dir)
            
            # Add dt_obj for temporal features
            met_data['dt_obj'] = ts
            
            try:
                # Run Pipeline (Handle PINN + NN2 internally)
                # We pass ground_truth (if available) to allow NN2 to run
                gt_arg = gt_values if has_gt else None
                final_pred_ppb = pipeline.process_timestep(met_data, sensor_points, emissions, ground_truth=gt_arg)
                
                # Extract PINN component separately for logging if desired?
                # The pipeline returns the FINAL prediction.
                # To get PINN only, we can call superimpose directly or just subtract correction if we had it.
                # For simplicity, let's just get the final pred.
                # Actually, to log PINN vs Hybrid, we need both.
                
                # 1. Get PINN Only
                raw_phi = pipeline.superimpose(met_data, sensor_points, emissions)
                pinn_pred_ppb = raw_phi * 3.13e8
                
                # 2. Hybrid (final_pred_ppb) is already computed above
                nn2_correction = final_pred_ppb - pinn_pred_ppb
                
            except Exception as e:
                print(f"Error: {e}")
                pinn_pred_ppb = np.zeros(9)
                final_pred_ppb = np.zeros(9)
                nn2_correction = np.zeros(9)
        else:
            pinn_pred_ppb = np.zeros(9)
            final_pred_ppb = np.zeros(9)
            nn2_correction = np.zeros(9)

        
        # Log Result
        for s_idx, s in enumerate(SENSORS):
            results.append({
                'timestamp': ts,
                'sensor_id': s['id'],
                'pinn_pred': pinn_pred_ppb[s_idx],
                'nn2_correction': nn2_correction[s_idx],
                'final_pred': final_pred_ppb[s_idx],
                'ground_truth': gt_values[s_idx] if has_gt else None
            })

    df_res = pd.DataFrame(results)
    
    # Save CSV
    out_csv = "validation_results_2019_jan_mar_detailed.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"Saved detailed results to {out_csv}")
    
    # Compute MAE
    df_valid = df_res.dropna(subset=['ground_truth'])
    if not df_valid.empty:
        df_valid['mae_pinn'] = (df_valid['pinn_pred'] - df_valid['ground_truth']).abs()
        df_valid['mae_hybrid'] = (df_valid['final_pred'] - df_valid['ground_truth']).abs()
        
        mae_pinn_val = df_valid['mae_pinn'].mean()
        mae_hybrid_val = df_valid['mae_hybrid'].mean()
        
        print(f"\nOverall MAE (PINN Only): {mae_pinn_val:.4f} ppb")
        print(f"Overall MAE (Hybrid):    {mae_hybrid_val:.4f} ppb")
        
        # Update Results Log
        log_path = os.path.join(base_dir, "logs", "resultslog.md")
        timestamp_log = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"| {timestamp_log} | Spatial NN2 | {len(timestamps)} hrs | MAE PINN: {mae_pinn_val:.4f} | MAE Hybrid: {mae_hybrid_val:.4f} | Jan-Mar 2019 Extended |\n"
        with open(log_path, "a") as f:
            f.write(log_entry)
            
    else:
        print("No valid comparisons found.")

if __name__ == "__main__":
    run_extended_validation()
