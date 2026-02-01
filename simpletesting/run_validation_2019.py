
import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benzene_pipeline import BenzenePipeline, load_emissions_for_timestamp

# ==========================================
# CONFIGURATION
# ==========================================
SENSORS = [
    {'id': 'sensor_482010026', 'name': 'HRM 7 Baytown', 'x': 12809.0, 'y': 2135.3},
    {'id': 'sensor_482010057', 'name': 'Galena Park', 'x': -8551.1, 'y': -1698.2},
    {'id': 'sensor_482010069', 'name': 'Milby Park', 'x': -10747.2, 'y': -4803.3},
    {'id': 'sensor_482010617', 'name': 'Wallisville Road', 'x': 15441.1, 'y': 7981.4},
    {'id': 'sensor_482010803', 'name': 'HRM #3 Haden Rd', 'x': -2741.7, 'y': 1673.7},
    {'id': 'sensor_482011015', 'name': 'Lynchburg Ferry', 'x': 6819.7, 'y': 1027.6},
    {'id': 'sensor_482011035', 'name': 'Clinton', 'x': -10405.2, 'y': -1759.7},
    {'id': 'sensor_482011039', 'name': 'Houston Deer Park #2', 'x': 2075.8, 'y': -8815.7},
    {'id': 'sensor_482016000', 'name': 'Cesar Chavez', 'x': -10019.0, 'y': -7227.9}
]

def run_validation():
    # Setup Paths
    base_dir = "/Users/neevpratap/simpletesting"
    # synced folder has the 2019 training data
    data_dir = "/Users/neevpratap/Desktop/madis_data_desktop_updated/synced" 
    ground_truth_path = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001/sensors_final_synced.csv"
    
    pinn_path = os.path.join(base_dir, "pinn_combined_final.pth 2")
    nn2_path = os.path.join(base_dir, "nn2_master_model_spatial.pth")

    # Initialize Pipeline
    print("Initializing Pipeline...")
    pipeline = BenzenePipeline(pinn_path, nn2_path)
    
    # Load Ground Truth
    print(f"Loading ground truth from {ground_truth_path}...")
    df_gt = pd.read_csv(ground_truth_path)
    df_gt['timestamp'] = pd.to_datetime(df_gt['t'])
    
    # Define Time Range (Jan 2019)
    start_time = "2019-01-01 00:00:00"
    end_time = "2019-01-31 23:00:00"
    timestamps = pd.date_range(start=start_time, end=end_time, freq='h')
    
    results = []
    
    print(f"Starting validation for {len(timestamps)} timesteps (Jan 2019)...")
    
    # Pre-compute sensor grid points
    sensor_points = np.array([[s['x'], s['y']] for s in SENSORS])
    
    for i, ts in enumerate(timestamps):
        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
        
        if i % 24 == 0:
            print(f"Processing {ts_str} ({i}/{len(timestamps)})...")
            
        # 1. Load Met Data from synced source file
        met_data = {'u': 0.0, 'v': 0.0, 'D': 1.0, 't': 3600.0}
        valid_met_found = False 
        
        # Use BASF file as met source (they are synced)
        t_source_file = os.path.join(data_dir, "BASF_Pasadena_synced_training_data.csv")
        try:
            if os.path.exists(t_source_file):
                df_met = pd.read_csv(t_source_file)
                row = df_met[df_met['t'] == ts_str]
                if not row.empty:
                    u_val = float(row['wind_u'].values[0])
                    v_val = float(row['wind_v'].values[0])
                    D_val = float(row['D'].values[0])
                    
                    if not pd.isna(u_val) and not pd.isna(v_val) and not pd.isna(D_val):
                        met_data['u'] = u_val
                        met_data['v'] = v_val
                        met_data['D'] = D_val
                        met_data['t'] = 3600.0 
                        valid_met_found = True
        except Exception as e:
            pass

        # 2. Load Emissions
        emissions = load_emissions_for_timestamp(ts_str, data_dir)
        
        # 3. Run Pipeline
        if valid_met_found:
            try:
                concs = pipeline.process_timestep(met_data, sensor_points, emissions)
            except Exception as e:
                print(f"Pipeline error at {ts_str}: {e}")
                concs = np.zeros(len(SENSORS))
        else:
            concs = np.zeros(len(SENSORS))
        
        # 4. Store Results
        row_dict = {'timestamp': ts} # keep as datetime for merging
        for sensor_idx, val in enumerate(concs):
            row_dict[SENSORS[sensor_idx]['id']] = val
            
        results.append(row_dict)

    # Convert results to DataFrame
    df_pred = pd.DataFrame(results)
    
    # Save Raw Predictions
    pred_file = "january_2019_validation_predictions.csv"
    df_pred.to_csv(pred_file, index=False)
    print(f"Saved predictions to {pred_file}")

    # ==========================================
    # VALIDATION (MAE Calculation)
    # ==========================================
    print("Calculating MAE...")
    
    # Merge Predictions with Ground Truth
    # df_gt has 'timestamp' (datetime) and sensor columns
    # df_pred has 'timestamp' (datetime) and sensor columns
    
    # Filter GT to relevant time range
    df_gt_jan = df_gt[(df_gt['timestamp'] >= pd.to_datetime(start_time)) & 
                      (df_gt['timestamp'] <= pd.to_datetime(end_time))]
                      
    if df_gt_jan.empty:
        print("Warning: No matching ground truth data found for Jan 2019 time range.")
        mae_val = "N/A (No GT)"
    else:
        merged = pd.merge(df_pred, df_gt_jan, on='timestamp', suffixes=('_pred', '_true'))
        
        errors = []
        for s in SENSORS:
            sid = s['id']
            col_pred = f"{sid}_pred"
            col_true = f"{sid}_true" # or just sid if no suffix? usually suffix applied to both if collision
            # Wait, if columns collide, suffixes applied.
            # df_gt has 'sensor_...'. df_pred has 'sensor_...'.
            # So yes, suffixes.
            
            if col_pred in merged.columns and col_true in merged.columns:
                # Calculate absolute error for this sensor
                # Filter out NaNs in ground truth
                valid_rows = merged[col_true].notna()
                diff = np.abs(merged.loc[valid_rows, col_pred] - merged.loc[valid_rows, col_true])
                errors.extend(diff.tolist())
        
        if errors:
            mae_val = np.mean(errors)
            print(f"Overall MAE: {mae_val:.4f} ppb")
        else:
            mae_val = "N/A (No Valid Pairs)"
            print("No valid prediction-truth pairs found.")

    # --- LOG RESULTS ---
    hours_processed = len(df_pred)
    num_cols = [c for c in df_pred.columns if c != 'timestamp']
    max_val = df_pred[num_cols].max().max() if not df_pred.empty else 0.0
    min_val = df_pred[num_cols].min().min() if not df_pred.empty else 0.0
    
    log_path = os.path.join(base_dir, "logs", "resultslog.md")
    timestamp_log = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Format MAE
    mae_str = f"{mae_val:.4f}" if isinstance(mae_val, float) else str(mae_val)
    
    log_entry = f"| {timestamp_log} | Spatial NN2 (Strict) | {hours_processed} | {max_val:.4f} | {min_val:.4f} | {mae_str} | Jan 2019 Validation |\n"
    
    try:
        with open(log_path, "a") as f:
            f.write(log_entry)
        print(f"Logged results to {log_path}")
    except Exception as e:
        print(f"Failed to write log: {e}")

if __name__ == "__main__":
    run_validation()
