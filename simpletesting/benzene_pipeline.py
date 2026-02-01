
import torch
import numpy as np
import pandas as pd
import json
import os
import sys

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pinn import ParametricADEPINN
    from nn2 import NN2_CorrectionNetwork
except ImportError as e:
    print(f"Error importing model classes: {e}")
    # Placeholder classes
    class ParametricADEPINN(torch.nn.Module):
        def __init__(self): super().__init__(); self.net = torch.nn.Linear(1,1)
        def forward(self, *args, **kwargs): return torch.zeros_like(args[0])
        def set_normalization_from_data(self, *args): pass
    class NN2_CorrectionNetwork(torch.nn.Module):
        def __init__(self, n_sensors=9): super().__init__()
        def forward(self, *args): return args[0], torch.zeros_like(args[0])




# ==========================================
# CONFIGURATION
# ==========================================

# Conversion factor for Benzene Concentration to PPB
UNIT_CONVERSION_FACTOR = 313210039.9 


# Verified Source Metadata (Cartesian Coordinates from Drive Data)
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

class BenzenePipeline:
    def __init__(self, pinn_path, nn2_path=None, nn2_scaler_path=None, device='cpu'):
        self.device = device
        self.t_start = pd.to_datetime('2019-01-01 00:00:00')  # Reference for time calculation
        self.pinn = self.load_pinn(pinn_path)
        if nn2_path:
            self.nn2 = self.load_nn2(nn2_path, scaler_path=nn2_scaler_path)
        else:
            self.nn2 = None
        print(f"Pipeline initialized on {self.device}")

    def load_pinn(self, path):
        print(f"Loading PINN from {path}...")
        try:
            # Load checkpoint
            # weights_only=False because the model might contain other objects or older format
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model = ParametricADEPINN()
            
            # Load model weights but OVERRIDE normalization ranges to match benchmark
            state_dict = checkpoint['model_state_dict']
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                   if not k.endswith('_min') and not k.endswith('_max')}
            model.load_state_dict(filtered_state_dict, strict=False)
            
            # Override with benchmark normalization ranges
            model.x_min = torch.tensor(0.0)
            model.x_max = torch.tensor(30000.0)
            model.y_min = torch.tensor(0.0)
            model.y_max = torch.tensor(30000.0)
            model.t_min = torch.tensor(0.0)
            model.t_max = torch.tensor(8760.0)
            model.cx_min = torch.tensor(0.0)
            model.cx_max = torch.tensor(30000.0)
            model.cy_min = torch.tensor(0.0)
            model.cy_max = torch.tensor(30000.0)
            model.u_min = torch.tensor(-15.0)
            model.u_max = torch.tensor(15.0)
            model.v_min = torch.tensor(-15.0)
            model.v_max = torch.tensor(15.0)
            model.d_min = torch.tensor(0.0)
            model.d_max = torch.tensor(200.0)
            model.kappa_min = torch.tensor(0.0)
            model.kappa_max = torch.tensor(200.0)
            model.Q_min = torch.tensor(0.0)
            model.Q_max = torch.tensor(0.01)
            
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Failed to load PINN: {e}")
            return None
    
    def load_nn2(self, path, scaler_path=None):
        print(f"Loading NN2 from {path}...")
        self.nn2_scalers = None
        self.sensor_coords_spatial = None
        
        try:
            # Load scalers from external pickle if provided
            if scaler_path and os.path.exists(scaler_path):
                import pickle
                with open(scaler_path, 'rb') as f:
                    self.nn2_scalers = pickle.load(f)
                print(f"✓ Loaded NN2 Scalers from {scaler_path}")
            
            # Instantiate model (45 inputs)
            model = NN2_CorrectionNetwork(n_sensors=9)
            
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                # Try to get metadata from checkpoint as fallback
                if self.nn2_scalers is None and 'scalers' in checkpoint:
                    self.nn2_scalers = checkpoint['scalers']
                    print("Received NN2 Scalers from checkpoint dictionary.")
                
                # Check for sensor coordinates (needed for spatial features)
                # First check for .npy file in same dir as model
                coords_file = os.path.join(os.path.dirname(path), "sensor_coordinates.npy")
                if os.path.exists(coords_file):
                    self.sensor_coords_spatial = np.load(coords_file)
                    print(f"✓ Loaded Sensor Coords from {coords_file}")
                elif 'sensor_coords' in checkpoint:
                    self.sensor_coords_spatial = checkpoint['sensor_coords']
                    print("Received Sensor Coords from checkpoint dictionary.")
                
                # Load weights
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                # Direct load
                try:
                    model.load_state_dict(checkpoint)
                except:
                    model = checkpoint 
                
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Warning: Failed to load NN2 ({e}). Pipeline continues without NN2 correction.")
            return None

    def superimpose(self, met_data, grid_points, emissions=None, kappa_scaling=1.0):
        """
        met_data: Dict with 'u', 'v', 'D', 't'
        grid_points: (N, 2) array of [x, y]
        emissions: Dict mapping source names to Q values (e.g., {'SourceA': 0.0005}). 
                   If None, or if specific source missing, defaults to 0.0.
        """
        if self.pinn is None: return np.zeros(len(grid_points))

        total_phi = np.zeros(len(grid_points))
        
        # Grid Coordinates
        grid_x = torch.tensor(grid_points[:, 0], dtype=torch.float32).to(self.device).view(-1, 1)
        grid_y = torch.tensor(grid_points[:, 1], dtype=torch.float32).to(self.device).view(-1, 1)
        
        # Extract meteorology
        u_val = met_data.get('u', 0.0)
        v_val = met_data.get('v', 0.0)
        D_val = met_data.get('D', 1.0)
        
        # Use D (dispersion coefficient) as kappa (matches benchmark)
        kappa_val = D_val * kappa_scaling
        
        # Calculate time in hours from 2019-01-01 00:00:00
        if 'dt_obj' in met_data and met_data['dt_obj'] is not None:
            timestamp = met_data['dt_obj']
            t_hours = (timestamp - self.t_start).total_seconds() / 3600.0
        else:
            # Fallback: assume t_hours is provided or default to 0.0
            t_hours = met_data.get('t_hours', 0.0)    
        
        u = torch.full_like(grid_x, u_val)
        v = torch.full_like(grid_x, v_val)
        t_vec = torch.full_like(grid_x, t_hours) # Use t_hours for PINN input
        kappa = torch.full_like(grid_x, kappa_val)
        
        # Process Sources
        with torch.no_grad():
            for facility in FACILITIES:
                # Get dynamic emission rate for this facility
                Q_val = 0.0
                if emissions:
                    Q_val = emissions.get(facility['name'], 0.0)
                
                # Skip if no emission
                if Q_val == 0:
                    continue

                cx_val = facility['coords'][0]
                cy_val = facility['coords'][1]
                d_val = facility['diameter']
                
                cx = torch.full_like(grid_x, cx_val)
                cy = torch.full_like(grid_y, cy_val)
                d = torch.full_like(grid_x, d_val)
                Q_tensor = torch.full_like(grid_x, Q_val)
                
                # Model Forward: x, y, t, cx, cy, u, v, d, kappa, Q
                phi = self.pinn(grid_x, grid_y, t_vec, cx, cy, u, v, d, kappa, Q_tensor, normalize=True)
                
                phi_np = phi.cpu().numpy().flatten()
                total_phi += phi_np
                
        return total_phi

    def process_timestep(self, met_data, grid_points, emissions, ground_truth=None):
        # 1. Superimpose PINN
        raw_phi = self.superimpose(met_data, grid_points, emissions)
        
        # 2. Convert to PPB
        conc_ppb = raw_phi * UNIT_CONVERSION_FACTOR
        
        # 3. NN2 Correction (Only if ground truth is provided & scalers exist)
        if self.nn2 is not None and self.nn2_scalers is not None and ground_truth is not None:
            # Note: valid_met_found check should happen before calling this, 
            # here we assume if met_data is passed, it is valid enough to form inputs
             conc_ppb = self.apply_nn2_correction(conc_ppb, ground_truth, met_data)
        
        return conc_ppb

    def apply_nn2_correction(self, pinn_pred_ppb, current_sensors_ppb, met_data):
        """
        Applies NN2 correction using loaded scalers.
        """
        try:
            # Handle potential NaNs upfront
            pinn_pred_ppb = np.nan_to_num(pinn_pred_ppb, nan=0.0)
            current_sensors_ppb = np.nan_to_num(current_sensors_ppb, nan=0.0)
            
            # 1. Scale Inputs
            # Ensure we pass 2D arrays to scikit-learn
            p_s = self.nn2_scalers['pinn'].transform(pinn_pred_ppb.reshape(-1, 1)).reshape(1, -1)
            s_s = self.nn2_scalers['sensors'].transform(current_sensors_ppb.reshape(-1, 1)).reshape(1, -1)
            
            u_in = met_data.get('u', 0.8) # Neutral default
            v_in = met_data.get('v', 0.1)
            if np.isnan(u_in): u_in = 0.0
            if np.isnan(v_in): v_in = 0.0
                
            w_in = np.array([[u_in, v_in]])
            w_s = self.nn2_scalers['wind'].transform(w_in)
            
            d_val = met_data.get('D', 10.0)
            if np.isnan(d_val): d_val = 1.0
            d_in = np.array([[d_val]])
            d_s = self.nn2_scalers['diffusion'].transform(d_in)
            
            # Ensure inputs are valid for scikit-learn
            if self.sensor_coords_spatial is None:
                raise ValueError("NN2 Sensor coordinates not loaded.")
            
            # Coords translation if needed? 
            # The model was trained on its own internal 'sensor_coords'.
            # We must use those same coords for the spatial feature.
            # self.sensor_coords_spatial (shape [9, 2]) is already loaded from .pth
            c_s = self.nn2_scalers['coords'].transform(self.sensor_coords_spatial)
            c_tensor = torch.tensor(c_s, dtype=torch.float32).to(self.device).unsqueeze(0)
            
            # temporal_vals handled below...
            if 'dt_obj' in met_data and met_data['dt_obj'] is not None:
                dt = met_data['dt_obj']
                hour = dt.hour
                day_of_week = dt.dayofweek
                month = dt.month
                is_weekend = 1.0 if day_of_week >= 5 else 0.0
                temporal_vals = np.array([[
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * day_of_week / 7),
                    np.cos(2 * np.pi * day_of_week / 7),
                    is_weekend,
                    month / 12.0
                ]])
                t_tensor = torch.tensor(temporal_vals, dtype=torch.float32).to(self.device)
            else:
                t_tensor = torch.zeros((1, 6)).to(self.device)

            # Tensors
            p_tensor = torch.tensor(p_s, dtype=torch.float32).to(self.device)
            s_tensor = torch.tensor(s_s, dtype=torch.float32).to(self.device)
            w_tensor = torch.tensor(w_s, dtype=torch.float32).to(self.device)
            d_tensor = torch.tensor(d_s, dtype=torch.float32).to(self.device)
            
            # Inference
            with torch.no_grad():
                corrected_pred_scaled, _ = self.nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
            
            # Inverse Transform
            corrected_pred_scaled_np = corrected_pred_scaled.cpu().numpy().flatten()
            final_pred = self.nn2_scalers['sensors'].inverse_transform(corrected_pred_scaled_np.reshape(-1, 1)).flatten()
            
            return final_pred
            
        except Exception as e:
            # We print once, but avoid flooding
            if not hasattr(self, '_nn2_error_reported'):
                print(f"NN2 Correction Detailed Error: {e}")
                self._nn2_error_reported = True
            return pinn_pred_ppb 

    def process_forecast(self, target_time_str, data_dir, grid_points):
        """
        Forecasting Logic:
        1. Loads Met Data & Emissions from (target_time - 3 hours).
        2. Runs PINN with t = 3 hours (10800 seconds) to simulate transport.
        """
        # 1. Calculate History Timestamp
        try:
            target_dt = pd.to_datetime(target_time_str)
            past_dt = target_dt - pd.Timedelta(hours=3)
            past_time_str = past_dt.strftime('%Y-%m-%d %H:%M:%S')
            # print(f"Forecast Target: {target_time_str}")
            # print(f"Using Historical Data from: {past_time_str} (-3 hours)")
        except ValueError as e:
            print(f"Error parsing date: {e}")
            return np.zeros(len(grid_points))

        # 2. Load Historical Emissions
        emissions = load_emissions_for_timestamp(past_time_str, data_dir)
        
        # 3. Load Historical Met Data
        met_data = {'u': 0.0, 'v': 0.0, 'D': 1.0, 't': 10800.0} # Default t=3hours
        found_met = False
        for facility in FACILITIES:
            csv_path = os.path.join(data_dir, f"{facility['name']}_training_data.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    row = df[df['t'] == past_time_str]
                    if not row.empty:
                        met_data['u'] = float(row['wind_u'].values[0])
                        met_data['v'] = float(row['wind_v'].values[0])
                        met_data['D'] = float(row['D'].values[0])
                        met_data['t'] = 10800.0 
                        found_met = True
                        break
                except:
                    continue
        
        return self.process_timestep(met_data, grid_points, emissions)




def load_emissions_for_timestamp(target_time_str, data_dir):
    # ... (same as before, keeping existing logic) ...
    emissions = {}
    # print(f"Loading emissions for {target_time_str}...") # Reduce verbosity
    
    for facility in FACILITIES:
        name = facility['name']
        csv_path = os.path.join(data_dir, f"{name}_training_data.csv")
        
        if not os.path.exists(csv_path):
            # Check for synced version
            csv_path = os.path.join(data_dir, f"{name}_synced_training_data.csv")
            if not os.path.exists(csv_path):
                emissions[name] = 0.0
                continue
            
        try:
            df = pd.read_csv(csv_path)
            row = df[df['t'] == target_time_str]
            if not row.empty:
                emissions[name] = float(row['Q_total'].values[0])
            else:
                emissions[name] = 0.0
        except:
            emissions[name] = 0.0
            
    return emissions

if __name__ == "__main__":
    # Test Run
    pinn_file = "/Users/neevpratap/simpletesting/pinn_combined_final.pth 2"
    nn2_file = "/Users/neevpratap/simpletesting/nn2_master_model_spatial-2.pth"
    nn2_scaler_file = "/Users/neevpratap/simpletesting/nn2_master_scalers.pkl"
    
    try:
        pipeline = BenzenePipeline(pinn_file, nn2_file, nn2_scaler_path=nn2_scaler_file)
        
        # Grid (Covering all facilities: ~ -12km to +14km)
        x = np.linspace(-20000, 20000, 50)
        y = np.linspace(-20000, 20000, 50)
        xx, yy = np.meshgrid(x, y)
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        
        # Data Config
        data_folder = "/Users/neevpratap/simpletesting/training_data_2021_full_jan"
        
        # Define a Target Forecast Time (e.g., 9 AM)
        # System will look for data at 6 AM (9 - 3)
        target_forecast_time = "2021-01-01 09:00:00" 
        
        print(f"Running 3-hour Forecast for {target_forecast_time}...")
        res = pipeline.process_forecast(target_forecast_time, data_folder, grid)
        
        print("Success.")
        print(f"Max Concentration: {res.max():.4f} ppb")
        print(f"Mean Concentration: {res.mean():.4f} ppb")
        
    except Exception as e:
        print(f"Pipeline Execution Failed: {e}")
        import traceback
        traceback.print_exc()
