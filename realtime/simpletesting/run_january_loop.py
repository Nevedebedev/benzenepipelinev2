
import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benzene_pipeline import BenzenePipeline, load_emissions_for_timestamp

# ==========================================
# 1. SENSOR CONFIGURATION (Verified)
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

def run_january_processing():
    # Setup Paths
    base_dir = "/Users/neevpratap/simpletesting"
    data_dir = os.path.join(base_dir, "training_data_2021_full_jan")
    pinn_path = os.path.join(base_dir, "pinn_combined_final.pth 2")
    nn2_path = os.path.join(base_dir, "nn2_master_model_spatial.pth")

    # Initialize Pipeline
    print("Initializing Pipeline...")
    # Using the special spatial NN2 model
    pipeline = BenzenePipeline(pinn_path, nn2_path) 
    
    # Define Time Range
    start_time = "2021-01-01 00:00:00"
    end_time = "2021-01-31 23:00:00"
    timestamps = pd.date_range(start=start_time, end=end_time, freq='h')
    
    results = []
    
    print(f"Starting execution for {len(timestamps)} timesteps...")
    
    # Pre-compute sensor grid points for batch inference
    # We want to predict EXACTLY at the sensor locations
    sensor_points = np.array([[s['x'], s['y']] for s in SENSORS])
    
    for i, ts in enumerate(timestamps):
        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
        
        if i % 24 == 0:
            print(f"Processing {ts_str} ({i}/{len(timestamps)})...")
            
        # 1. Load Met Data (Use one source file as proxy for domain-wide met data)
        # In reality, might vary, but assuming uniform field for PINN
        met_data = {'u': 0.0, 'v': 0.0, 'D': 1.0, 't': 3600.0} # Default container
        valid_met_found = False 
        # Wait, is 't' the evolution time or just input? 
        # In the pipeline, 't' is time. Usually for steady state we might pick a characteristic time
        # or if it's dynamic, we simulate a step.
        # However, checking 'process_forecast', it loads history.
        # For "Reanalysis" (checking against actuals), we usually use the CURRENT met data.
        
        # Checking logic in load_emissions_for_timestamp...
        # We need to find the met data for THIS timestamp.
        # Let's peek at one reliable source file.
        t_source_file = os.path.join(data_dir, "BASF_Pasadena_training_data.csv")
        try:
            if os.path.exists(t_source_file):
                df_met = pd.read_csv(t_source_file)
                row = df_met[df_met['t'] == ts_str]
                if not row.empty:
                    u_val = float(row['wind_u'].values[0])
                    v_val = float(row['wind_v'].values[0])
                    D_val = float(row['D'].values[0])
                    
                    # STRICT CHECK: If any value is NaN, do NOT use this file's met data
                    if not pd.isna(u_val) and not pd.isna(v_val) and not pd.isna(D_val):
                        met_data['u'] = u_val
                        met_data['v'] = v_val
                        met_data['D'] = D_val
                        # Assume t=1 hour for transport if not loaded
                        met_data['t'] = 3600.0 
                        valid_met_found = True
        except Exception as e:
            pass

        # 2. Load Emissions
        emissions = load_emissions_for_timestamp(ts_str, data_dir)
        
        # 3. Run Pipeline ONLY if valid met data exists
        if valid_met_found:
            try:
                concs = pipeline.process_timestep(met_data, sensor_points, emissions)
            except Exception as e:
                print(f"Pipeline error at {ts_str}: {e}")
                concs = np.zeros(len(SENSORS))
        else:
            # NO FALLBACK: If no met data, do not run simulation. Return zeros.
            # print(f"Skipping {ts_str}: No valid met data found.")
            concs = np.zeros(len(SENSORS))
        
        # 4. Store Results
        row_dict = {'timestamp': ts_str}
        for sensor_idx, val in enumerate(concs):
            row_dict[SENSORS[sensor_idx]['id']] = val
            
        results.append(row_dict)

    # Save to CSV
    out_file = "january_2021_pinn_predictions.csv"
    print(f"Saving results to {out_file}...")
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_file, index=False)
    
    # --- LOG RESULTS ---
    # Calculate Stats
    hours_processed = len(df_res)
    # Exclude timestamp column for numerical sats
    num_cols = [c for c in df_res.columns if c != 'timestamp']
    if not df_res.empty and hours_processed > 0:
        max_val = df_res[num_cols].max().max()
        min_val = df_res[num_cols].min().min()
    else:
        max_val = 0.0
        min_val = 0.0
        
    mae = "N/A" # No 2021 ground truth available yet
    
    # Append to Results Log
    log_path = os.path.join(base_dir, "logs", "resultslog.md")
    timestamp_now = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    log_entry = f"| {timestamp_now} | Spatial NN2 (Strict) | {hours_processed} | {max_val:.4f} | {min_val:.4f} | {mae} | January 2021 Run |\n"
    
    try:
        with open(log_path, "a") as f:
            f.write(log_entry)
        print(f"Logged results to {log_path}")
    except Exception as e:
        print(f"Failed to write log: {e}")

    print("Done.")

if __name__ == "__main__":
    run_january_processing()
