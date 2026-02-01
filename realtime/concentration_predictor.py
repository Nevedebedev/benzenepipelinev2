#!/usr/bin/env python3
"""
Concentration Predictor for Real-Time Pipeline
Computes PINN+NN2 predictions across full Houston domain
Appends predictions to continuous time series CSVs
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import pickle

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append("/Users/neevpratap/simpletesting")

from config import FACILITIES, BASE_DIR, CONTINUOUS_DIR, PREDICTIONS_DIR, VISUALIZATIONS_DIR, CORRECTIONS_DIR
from pinn import ParametricADEPINN
from visualizer import PipelineVisualizer


class ConcentrationPredictor:
    """Predict benzene concentrations using PINN+NN2 across full domain"""
    
    def __init__(self, grid_resolution: int = 100):
        self.grid_resolution = grid_resolution
        self.device = 'cpu'
        
        # Output directories
        self.output_dir = CONTINUOUS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictions_dir = PREDICTIONS_DIR
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = PipelineVisualizer(VISUALIZATIONS_DIR, CORRECTIONS_DIR)
        
        # Model paths
        self.pinn_path = "/Users/neevpratap/simpletesting/pinn_combined_final.pth 2"
        self.nn2_path = "/Users/neevpratap/simpletesting/nn2_master_model_spatial.pth"
        self.nn2_scalers_path = "/Users/neevpratap/simpletesting/nn2_master_scalers.pkl"
        
        # Load models
        print("[Concentration Predictor] Loading models...")
        self.pinn = self._load_pinn()
        self.nn2, self.scalers, self.sensor_coords_spatial = self._load_nn2()
        print("  ✓ Models loaded\n")
        
        # Reference time for PINN
        self.t_start = pd.to_datetime('2021-01-01 00:00:00')
        
        # Create spatial grid
        self._create_grid()
    
    def _load_pinn(self):
        """Load PINN model"""
        pinn = ParametricADEPINN()
        checkpoint = torch.load(self.pinn_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                               if not k.endswith('_min') and not k.endswith('_max')}
        pinn.load_state_dict(filtered_state_dict, strict=False)
        
        # Override normalization ranges (benchmark values)
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
        return pinn
    
    def _load_nn2(self):
        """Load NN2 model with embedded scalers from checkpoint"""
        try:
            print("  Loading NN2 model with scalers...")
            
            # Load checkpoint
            checkpoint = torch.load(self.nn2_path, map_location='cpu', weights_only=False)
            
            # Load model
            from nn2 import NN2_CorrectionNetwork
            nn2 = NN2_CorrectionNetwork(n_sensors=9)
            nn2.load_state_dict(checkpoint['model_state_dict'])
            nn2.eval()
            
            # Load scalers from checkpoint
            scalers = checkpoint['scalers']
            sensor_coords = checkpoint['sensor_coords']
            
            print("  ✓ NN2 model loaded with scalers")
            
            return nn2, scalers, sensor_coords
            
        except Exception as e:
            print(f"  ✗ Failed to load NN2: {e}")
            print("  Falling back to simplified correction")
            return None, None, None
    
    def _create_grid(self):
        """Create spatial grid for domain predictions"""
        x_grid = np.linspace(0, 30000, self.grid_resolution)
        y_grid = np.linspace(0, 30000, self.grid_resolution)
        self.X, self.Y = np.meshgrid(x_grid, y_grid)
        
        # Flatten for easier processing
        self.x_flat = self.X.flatten()
        self.y_flat = self.Y.flatten()
        
        print(f"  Grid: {self.grid_resolution}x{self.grid_resolution} = {len(self.x_flat)} points")
    
    def predict_full_domain(
        self,
        facility_params: dict,
        current_time: datetime
    ) -> tuple:
        """
        Predict concentrations across full domain for t+3 hours
        
        Args:
            facility_params: dict from CSV generator with met data per facility
            current_time: Current timestamp
        
        Returns:
            (pinn_field, nn2_field, forecast_time)
        """
        forecast_time = current_time + timedelta(hours=3)
        
        print(f"[Concentration Predictor] Predicting for {forecast_time}")
        print(f"  Computing PINN for {len(facility_params)} facilities...")
        
        # Calculate time in hours since ref
        t_hours = (forecast_time - self.t_start).total_seconds() / 3600.0
        
        # Accumulate plumes from all facilities
        total_pinn_field = np.zeros(len(self.x_flat))
        
        for facility_name, params in facility_params.items():
            # Get facility info
            source_x_cart = params['source_x_cartesian']
            source_y_cart = params['source_y_cartesian'] 
            source_diameter = params['source_diameter']
            Q = params['Q']
            wind_u = params['wind_u']
            wind_v = params['wind_v']
            D = params['D']
            
            # Compute PINN for this facility across all grid points
            facility_field = self._compute_pinn_for_facility(
                self.x_flat, self.y_flat, t_hours,
                source_x_cart, source_y_cart, source_diameter,
                Q, wind_u, wind_v, D
            )
            
            # Accumulate (superposition)
            total_pinn_field += facility_field
        
        print(f"  PINN superposition: min={total_pinn_field.min():.4f}, max={total_pinn_field.max():.4f} ppb")
        
        # Apply NN2 correction across entire field
        print(f"  Applying NN2 correction...")
        nn2_field = self._apply_nn2_correction(
            self.x_flat, self.y_flat, total_pinn_field, facility_params, forecast_time
        )
        
        print(f"  NN2 corrected: min={nn2_field.min():.4f}, max={nn2_field.max():.4f} ppb")
        
        # Extract predictions at sensor locations for validation
        self._log_sensor_predictions(self.x_flat, self.y_flat, total_pinn_field, nn2_field, forecast_time)
        
        # Generate visualizations
        print(f"  Generating visualizations...")
        self.visualizer.generate_heatmaps(
            self.x_flat, self.y_flat,
            total_pinn_field, nn2_field,
            forecast_time, self.grid_resolution
        )
        
        # Save correction statistics
        self.visualizer.save_correction_stats(
            current_time, forecast_time,
            total_pinn_field, nn2_field
        )
        
        # Append to continuous CSVs
        self._append_domain_data(
            current_time, forecast_time,
            self.x_flat, self.y_flat,
            total_pinn_field, nn2_field
        )
        
        # Save latest snapshot
        self._save_latest_snapshot(
            forecast_time, 
            self.x_flat, self.y_flat,
            total_pinn_field, nn2_field
        )
        
        return total_pinn_field, nn2_field, forecast_time
    
    def _compute_pinn_for_facility(
        self, x, y, t,
        source_x, source_y, source_d,
        Q, wind_u, wind_v, D
    ):
        """
        Compute PINN predictions for single facility across grid
        
        Returns:
            Array of concentrations in ppb
        """
        n_points = len(x)
        concentrations = np.zeros(n_points)
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            batch_x = x[i:end_idx]
            batch_y = y[i:end_idx]
            batch_size_actual = len(batch_x)
            
            # Create tensors
            x_t = torch.tensor(batch_x.reshape(-1, 1), dtype=torch.float32)
            y_t = torch.tensor(batch_y.reshape(-1, 1), dtype=torch.float32)
            t_t = torch.full((batch_size_actual, 1), t, dtype=torch.float32)
            cx_t = torch.full((batch_size_actual, 1), source_x, dtype=torch.float32)
            cy_t = torch.full((batch_size_actual, 1), source_y, dtype=torch.float32)
            u_t = torch.full((batch_size_actual, 1), wind_u, dtype=torch.float32)
            v_t = torch.full((batch_size_actual, 1), wind_v, dtype=torch.float32)
            d_t = torch.full((batch_size_actual, 1), source_d, dtype=torch.float32)
            kappa_t = torch.full((batch_size_actual, 1), D, dtype=torch.float32)
            Q_t = torch.full((batch_size_actual, 1), Q, dtype=torch.float32)
            
            # Run PINN
            with torch.no_grad():
                phi = self.pinn(x_t, y_t, t_t, cx_t, cy_t, u_t, v_t, 
                               d_t, kappa_t, Q_t, normalize=True)
            
            # Convert to ppb
            concentrations[i:end_idx] = np.maximum(phi.numpy().flatten() * 3.13e8, 0.0)
        
        return concentrations
    
    def _apply_nn2_correction(self, x, y, pinn_pred, facility_params, forecast_time):
        """
        Apply NN2 correction efficiently using actual 9 sensor locations
        
        1. Compute PINN predictions at 9 sensor locations
        2. Apply NN2 correction to get corrected values at sensors
        3. Calculate correction field (NN2 - PINN) at sensors
        4. Interpolate correction field across entire domain using RBF
        5. Add interpolated corrections to PINN field
        """
        if self.nn2 is None or self.scalers is None:
            # Fall back to simplified correction
            print("    Using simplified correction (20% reduction)")
            return np.maximum(pinn_pred * 0.8, 0.0)
        
        # Define 9 sensor locations (Cartesian coordinates - CORRECT values from training)
        SENSOR_COORDS = np.array([
            [13972.62, 19915.57],  # 482010026
            [3017.18, 12334.2],    # 482010057
            [817.42, 9218.92],     # 482010069
            [8836.35, 15717.2],    # 482010803
            [18413.8, 15068.96],   # 482011015
            [1159.98, 12272.52],   # 482011035
            [13661.93, 5193.24],   # 482011039
            [15077.79, 9450.52],   # 482011614
            [1546.9, 6786.33],     # 482016000
        ])
        
        # Step 1: Interpolate PINN field to get values at sensor locations
        from scipy.interpolate import Rbf
        
        # Create RBF interpolator for PINN field
        pinn_rbf = Rbf(x, y, pinn_pred, function='multiquadric', smooth=0.1)
        sensor_pinn = pinn_rbf(SENSOR_COORDS[:, 0], SENSOR_COORDS[:, 1])
        
        # Step 2: Apply NN2 correction at sensor locations
        avg_u = np.mean([p['wind_u'] for p in facility_params.values()])
        avg_v = np.mean([p['wind_v'] for p in facility_params.values()])
        avg_D = np.mean([p['D'] for p in facility_params.values()])
        
        # Temporal features
        hour = forecast_time.hour
        day_of_week = forecast_time.weekday()
        month = forecast_time.month
        is_weekend = 1.0 if day_of_week >= 5 else 0.0
        
        temporal_vals = np.array([[
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            is_weekend,
            month / 12.0
        ]])
        
        # Use dummy current sensor values (we don't have actual sensor readings)
        current_sensors = sensor_pinn.copy()
        
        # Scale inputs
        p_s = self.scalers['pinn'].transform(sensor_pinn.reshape(-1, 1)).reshape(1, -1)
        s_s = self.scalers['sensors'].transform(current_sensors.reshape(-1, 1)).reshape(1, -1)
        
        w_in = np.array([[avg_u, avg_v]])
        w_s = self.scalers['wind'].transform(w_in)
        
        d_in = np.array([[avg_D]])
        d_s = self.scalers['diffusion'].transform(d_in)
        
        c_s = self.scalers['coords'].transform(self.sensor_coords_spatial)
        
        # Convert to tensors
        p_tensor = torch.tensor(p_s, dtype=torch.float32)
        s_tensor = torch.tensor(s_s, dtype=torch.float32)
        c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)
        w_tensor = torch.tensor(w_s, dtype=torch.float32)
        d_tensor = torch.tensor(d_s, dtype=torch.float32)
        t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)
        
        # Run NN2 to get corrected values at sensors
        with torch.no_grad():
            corrected_scaled, _ = self.nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
        
        # Inverse transform to get corrected concentrations
        corrected_scaled_np = corrected_scaled.cpu().numpy().flatten()
        sensor_corrected = self.scalers['sensors'].inverse_transform(
            corrected_scaled_np.reshape(-1, 1)
        ).flatten()
        
        # Step 3: Calculate correction field at sensors (NN2 - PINN)
        sensor_corrections = sensor_corrected - sensor_pinn
        
        print(f"    Sensor corrections range: {sensor_corrections.min():.2f} to {sensor_corrections.max():.2f} ppb")
        
        # Step 4: Interpolate correction field across domain using RBF
        correction_rbf = Rbf(
            SENSOR_COORDS[:, 0], 
            SENSOR_COORDS[:, 1], 
            sensor_corrections,
            function='multiquadric',
            smooth=0.1
        )
        
        # Evaluate at all grid points
        correction_field = correction_rbf(x, y)
        
        # NEW: Distance-based confidence weighting
        # Calculate distance from each grid point to nearest sensor
        distances_to_sensors = np.zeros(len(x))
        for i in range(len(x)):
            dists = np.sqrt((SENSOR_COORDS[:, 0] - x[i])**2 + (SENSOR_COORDS[:, 1] - y[i])**2)
            distances_to_sensors[i] = dists.min()
        
        # Define confidence decay: full trust within 2km, linear decay 2-5km, zero trust beyond 5km
        confidence = np.ones(len(x))
        confidence[distances_to_sensors > 2000] = 1.0 - (distances_to_sensors[distances_to_sensors > 2000] - 2000) / 3000
        confidence[distances_to_sensors > 5000] = 0.0
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Apply confidence-weighted correction
        # Near sensors: use full NN2 correction
        # Far from sensors: use PINN only
        weighted_correction = correction_field * confidence
        
        # Step 5: Apply weighted corrections to PINN field
        corrected_field = pinn_pred + weighted_correction
        
        # Clip to non-negative
        corrected_field = np.maximum(corrected_field, 0.0)
        
        # Log confidence statistics
        print(f"    Confidence weighting: {(confidence > 0.5).sum()}/{len(confidence)} points with >50% confidence")
        print(f"    Mean confidence: {confidence.mean():.2f}")
        
        return corrected_field
    
    def _log_sensor_predictions(
        self,
        x_grid, y_grid,
        pinn_field, nn2_field,
        forecast_time: datetime
    ):
        """
        Log predictions at the 9 sensor locations for validation
        """
        # Define sensor coordinates and IDs
        SENSORS = {
            '482010026': [13972.62, 19915.57],
            '482010057': [3017.18, 12334.2],
            '482010069': [817.42, 9218.92],
            '482010803': [8836.35, 15717.2],
            '482011015': [18413.8, 15068.96],
            '482011035': [1159.98, 12272.52],
            '482011039': [13661.93, 5193.24],
            '482011614': [15077.79, 9450.52],
            '482016000': [1546.9, 6786.33],
        }
        
        from scipy.interpolate import Rbf
        
        # Create RBF interpolators for PINN and NN2 fields
        pinn_rbf = Rbf(x_grid, y_grid, pinn_field, function='multiquadric', smooth=0.1)
        nn2_rbf = Rbf(x_grid, y_grid, nn2_field, function='multiquadric', smooth=0.1)
        
        print(f"\n  {'='*80}")
        print(f"  PREDICTIONS AT SENSOR LOCATIONS ({forecast_time.strftime('%Y-%m-%d %H:%M')} UTC)")
        print(f"  {'='*80}")
        print(f"  {'Sensor ID':<12} {'Location (x, y)':<25} {'PINN (ppb)':<12} {'NN2 (ppb)':<12} {'Correction'}")
        print(f"  {'-'*80}")
        
        pinn_values = []
        nn2_values = []
        
        for sensor_id, (sx, sy) in SENSORS.items():
            # Interpolate values at sensor location
            pinn_val = float(pinn_rbf(sx, sy))
            nn2_val = float(nn2_rbf(sx, sy))
            correction = nn2_val - pinn_val
            
            pinn_values.append(pinn_val)
            nn2_values.append(nn2_val)
            
            print(f"  {sensor_id:<12} ({sx:>7.1f}, {sy:>7.1f})   {pinn_val:>8.2f}     {nn2_val:>8.2f}     {correction:>+8.2f}")
        
        print(f"  {'-'*80}")
        print(f"  {'MEAN':<12} {'':<25} {np.mean(pinn_values):>8.2f}     {np.mean(nn2_values):>8.2f}     {np.mean(nn2_values)-np.mean(pinn_values):>+8.2f}")
        print(f"  {'MEDIAN':<12} {'':<25} {np.median(pinn_values):>8.2f}     {np.median(nn2_values):>8.2f}     {np.median(nn2_values)-np.median(pinn_values):>+8.2f}")
        print(f"  {'MAX':<12} {'':<25} {np.max(pinn_values):>8.2f}     {np.max(nn2_values):>8.2f}")
        print(f"  {'MIN':<12} {'':<25} {np.min(pinn_values):>8.2f}     {np.min(nn2_values):>8.2f}")
        print(f"  {'='*80}\n")
        
        # Also save to CSV for tracking over time
        from config import CONTINUOUS_DIR
        sensor_log_path = CONTINUOUS_DIR / "sensor_predictions_timeseries.csv"
        
        sensor_data = {
            'forecast_timestamp': forecast_time,
            'pinn_mean': np.mean(pinn_values),
            'pinn_median': np.median(pinn_values),
            'pinn_max': np.max(pinn_values),
            'pinn_min': np.min(pinn_values),
            'nn2_mean': np.mean(nn2_values),
            'nn2_median': np.median(nn2_values),
            'nn2_max': np.max(nn2_values),
            'nn2_min': np.min(nn2_values),
        }
        
        # Add individual sensor values
        for i, (sensor_id, (sx, sy)) in enumerate(SENSORS.items()):
            sensor_data[f'sensor_{sensor_id}_pinn'] = pinn_values[i]
            sensor_data[f'sensor_{sensor_id}_nn2'] = nn2_values[i]
        
        import pandas as pd
        df = pd.DataFrame([sensor_data])
        
        if sensor_log_path.exists():
            df.to_csv(sensor_log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(sensor_log_path, index=False)

    
    def _append_domain_data(
        self, current_time, forecast_time,
        x, y, pinn_field, nn2_field
    ):
        """
        Append full domain predictions to continuous time series CSVs
        """
        # Create DataFrame
        df = pd.DataFrame({
            'forecast_timestamp': forecast_time,
            'current_timestamp': current_time,
            'x': x,
            'y': y,
            'pinn_concentration': pinn_field,
            'nn2_concentration': nn2_field,
            'correction': nn2_field - pinn_field
        })
        
        # Append to superimposed concentrations (PINN only)
        superimposed_path = self.output_dir / "superimposed_concentrations_timeseries.csv"
        df_pinn = df[['forecast_timestamp', 'current_timestamp', 'x', 'y', 'pinn_concentration']]
        
        if superimposed_path.exists():
            df_pinn.to_csv(superimposed_path, mode='a', header=False, index=False)
        else:
            df_pinn.to_csv(superimposed_path, mode='w', header=True, index=False)
            print(f"    Created: {superimposed_path.name}")
        
        # Append to NN2-corrected domain
        nn2_corrected_path = self.output_dir / "nn2_corrected_domain_timeseries.csv"
        
        if nn2_corrected_path.exists():
            df.to_csv(nn2_corrected_path, mode='a', header=False, index=False)
        else:
            df.to_csv(nn2_corrected_path, mode='w', header=True, index=False)
            print(f"    Created: {nn2_corrected_path.name}")
        
        print(f"  ✓ Appended {len(df)} rows to continuous CSVs")
    
    def _save_latest_snapshot(
        self, forecast_time, x, y, pinn_field, nn2_field
    ):
        """
        Save latest forecast as standalone CSV (overwrite each time)
        """
        df = pd.DataFrame({
            'forecast_timestamp': forecast_time,
            'x': x,
            'y': y,
            'pinn_concentration': pinn_field,
            'nn2_concentration': nn2_field
        })
        
        snapshot_path = self.predictions_dir / "latest_spatial_grid.csv"
        df.to_csv(snapshot_path, index=False)
        
        print(f"  ✓ Saved latest snapshot: {snapshot_path.name}")


def test_predictor():
    """Test the concentration predictor"""
    from madis_fetcher import MADISFetcher
    from csv_generator import CSVGenerator
    
    print("\n" + "="*70)
    print("Testing Concentration Predictor")
    print("="*70)
    
    # Get mock data
    current_time = datetime(2021, 1, 31, 12, 0, 0)
    fetcher = MADISFetcher()
    weather_data = fetcher.fetch_latest(current_time)
    
    # Generate facility params
    generator = CSVGenerator()
    facility_params = generator.append_facility_data(weather_data, current_time)
    
    # Predict concentrations
    predictor = ConcentrationPredictor(grid_resolution=50)  # Small grid for testing
    pinn_field, nn2_field, forecast_time = predictor.predict_full_domain(
        facility_params, current_time
    )
    
    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)
    print(f"Forecast time: {forecast_time}")
    print(f"PINN field: {pinn_field.min():.4f} - {pinn_field.max():.4f} ppb")
    print(f"NN2 field: {nn2_field.min():.4f} - {nn2_field.max():.4f} ppb")
    print("="*70)


if __name__ == "__main__":
    test_predictor()
