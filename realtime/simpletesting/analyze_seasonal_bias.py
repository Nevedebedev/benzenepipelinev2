
import torch
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

# Add paths to sys.path
sys.path.append('/Users/neevpratap/simpletesting')
from nn2 import NN2_CorrectionNetwork

def analyze_january_predictions():
    # Load model
    nn2 = NN2_CorrectionNetwork(n_sensors=9)
    checkpoint = torch.load('/Users/neevpratap/Desktop/nn2_updated/nn2_master_model_spatial-3.pth', map_location='cpu', weights_only=False)
    nn2.load_state_dict(checkpoint['model_state_dict'])
    nn2.eval()

    # Load scalers
    with open('/Users/neevpratap/Desktop/nn2_updated/nn2_master_scalers-2.pkl', 'rb') as f:
        scalers = pickle.load(f)

    # Load data
    df_pinn = pd.read_csv('/Users/neevpratap/simpletesting/nn2trainingdata/total_concentrations.csv')
    df_sensor = pd.read_csv('/Users/neevpratap/Downloads/sensors_final_synced.csv')
    
    # Merge and align
    if 't' in df_sensor.columns:
        df_sensor = df_sensor.rename(columns={'t': 'timestamp'})
    if 't' in df_pinn.columns:
        df_pinn = df_pinn.rename(columns={'t': 'timestamp'})
        
    df_sensor['timestamp'] = pd.to_datetime(df_sensor['timestamp'])
    df_pinn['timestamp'] = pd.to_datetime(df_pinn['timestamp'])
    
    df = pd.merge(df_sensor, df_pinn, on='timestamp', suffixes=('_act', '_pinn'))
    
    # Find a row in January
    jan_rows = df[df['timestamp'].dt.month == 1]
    if len(jan_rows) == 0:
        print("No January rows found in merged data!")
        return
    row = jan_rows.iloc[0]
    target_time = row['timestamp']
    
    # Find a row in October
    oct_rows = df[df['timestamp'].dt.month == 10]
    if len(oct_rows) == 0:
        print("No October rows found in merged data!")
        # Continue with dummy or return
    else:
        row_oct = oct_rows.iloc[0]
        target_time_oct = row_oct['timestamp']
    pinn_vals_oct = np.array([row_oct[s+'_pinn'] for s in sensor_ids])
    
    p_s_oct = scalers['pinn'].transform(pinn_vals_oct.reshape(-1, 1)).reshape(1, -1)
    s_s_oct = scalers['sensors'].transform(pinn_vals_oct.reshape(-1, 1)).reshape(1, -1)
    t_v_oct = np.array([[np.sin(2*np.pi*12/24), np.cos(2*np.pi*12/24), np.sin(2*np.pi*1/7), np.cos(2*np.pi*1/7), 0.0, 10.0/12.0]])
    
    with torch.no_grad():
        out_oct, _ = nn2(
            torch.tensor(s_s_oct, dtype=torch.float32),
            torch.tensor(p_s_oct, dtype=torch.float32),
            torch.tensor(c_s, dtype=torch.float32).unsqueeze(0),
            torch.tensor(w_s, dtype=torch.float32),
            torch.tensor(d_s, dtype=torch.float32),
            torch.tensor(t_v_oct, dtype=torch.float32)
        )
    corrected_oct = scalers['sensors'].inverse_transform(out_oct.numpy().reshape(-1, 1)).flatten()
    
    print(f"\nAnalysis for {target_time_oct}:")
    print(f"{'Sensor':<15} | {'PINN (ppb)':<10} | {'NN2 (ppb)':<10} | {'Act (ppb)':<10} | {'Ratio(N/P)':<10}")
    print("-" * 65)
    for i, s in enumerate(sensor_ids):
        p = row_oct[s+'_pinn']
        n = corrected_oct[i]
        a = row_oct[s+'_act']
        r = n / p if p != 0 else 0
        print(f"{s:<15} | {p:>10.4f} | {n:>10.4f} | {a:>10.4f} | {r:>10.2f}")

if __name__ == "__main__":
    analyze_january_predictions()
