#!/usr/bin/env python3
"""
Concentration Predictor for Real-Time Pipeline
Computes PINN+Kalman filter predictions across full Houston domain
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
from scipy.interpolate import Rbf

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append("/Users/neevpratap/simpletesting")
sys.path.append(str(Path(__file__).parent / 'simpletesting'))

from config import FACILITIES, BASE_DIR, CONTINUOUS_DIR, PREDICTIONS_DIR, VISUALIZATIONS_DIR, CORRECTIONS_DIR
from pinn import ParametricADEPINN
from visualizer import PipelineVisualizer


class ConcentrationPredictor:
    """Predict benzene concentrations using PINN+Kalman filter across full domain"""
    
    def __init__(self, grid_resolution: int = 100, use_kalman: bool = True):
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
        self.pinn_path = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
        
        # Load models
        print("[Concentration Predictor] Loading models...")
        self.pinn = self._load_pinn()
        print("  ✓ PINN model loaded")
        
        # Initialize Kalman filter
        self.use_kalman = use_kalman
        if use_kalman:
            import json
            param_file = Path("realtime/data/kalman_parameters.json")
            if param_file.exists():
                with open(param_file, 'r') as f:
                    kf_params = json.load(f)
                    # Filter to only parameters (exclude metrics)
                    kf_params = {k: v for k, v in kf_params.items() 
                                if k in ['process_noise', 'measurement_noise', 
                                        'decay_rate', 'pinn_weight']}
            else:
                # Default parameters
                kf_params = {
                    'process_noise': 1.0,
                    'measurement_noise': 0.01,
                    'decay_rate': 0.7,
                    'pinn_weight': 0.3
                }
                print(f"  ⚠ No tuned parameters found, using defaults: {kf_params}")
            
            from kalman_filter import BenzeneKalmanFilter
            self.kalman_filter = BenzeneKalmanFilter(**kf_params)
            print(f"  ✓ Kalman filter initialized with params: {kf_params}")
        
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
        current_time: datetime,
        current_sensor_readings: np.ndarray = None
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
        
        # FIXED: Use simulation time t=3.0 hours (not absolute calendar time)
        # Each scenario starts at t=0, predicts at t=3 hours for 3-hour forecast
        # This makes PINN truly steady-state - only wind/diffusion/emissions affect predictions
        FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast
        t_hours = FORECAST_T_HOURS
        
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
        
        # Apply Kalman filter correction across entire field
        if self.use_kalman and current_sensor_readings is not None:
            print(f"  Applying Kalman filter correction...")
            kalman_field = self._apply_kalman_correction(
                self.x_flat, self.y_flat, total_pinn_field, facility_params, forecast_time, current_sensor_readings
            )
            print(f"  Kalman corrected: min={kalman_field.min():.4f}, max={kalman_field.max():.4f} ppb")
        else:
            # No correction - use PINN only
            kalman_field = total_pinn_field.copy()
            print(f"  No correction applied (Kalman disabled or no sensor readings)")
        
        # For backward compatibility, use kalman_field as the corrected field
        corrected_field = kalman_field
        
        # Extract predictions at sensor locations for validation
        self._log_sensor_predictions(self.x_flat, self.y_flat, total_pinn_field, corrected_field, forecast_time)
        
        # Generate visualizations
        print(f"  Generating visualizations...")
        self.visualizer.generate_heatmaps(
            self.x_flat, self.y_flat,
            total_pinn_field, corrected_field,
            forecast_time, self.grid_resolution
        )
        
        # Save correction statistics
        self.visualizer.save_correction_stats(
            current_time, forecast_time,
            total_pinn_field, corrected_field
        )
        
        # Append to continuous CSVs
        self._append_domain_data(
            current_time, forecast_time,
            self.x_flat, self.y_flat,
            total_pinn_field, corrected_field
        )
        
        # Save latest snapshot
        self._save_latest_snapshot(
            forecast_time, 
            self.x_flat, self.y_flat,
            total_pinn_field, corrected_field
        )
        
        return total_pinn_field, corrected_field, forecast_time
    
    def _compute_pinn_for_facility(
        self, x, y, t,
        source_x, source_y, source_d,
        Q, wind_u, wind_v, D
    ):
        """
        Compute PINN predictions for single facility across grid
        
        FIXED: Uses simulation time t=3.0 hours (not absolute calendar time).
        Each scenario starts at t=0, predicts at t=3 hours for 3-hour forecast.
        This matches training data generation and removes PINN time dependency bug.
        
        Returns:
            Array of concentrations in ppb
        """
        # FIX: Use simulation time instead of absolute calendar time (matches training data)
        FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast (each scenario resets to t=0)
        t_simulation = FORECAST_T_HOURS
        
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
            t_t = torch.full((batch_size_actual, 1), t_simulation, dtype=torch.float32)  # Use simulation time
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
            
            # Convert to ppb (using same conversion factor as training data)
            UNIT_CONVERSION_FACTOR = 313210039.9  # kg/m^2 to ppb (matches training data)
            concentrations[i:end_idx] = np.maximum(phi.numpy().flatten() * UNIT_CONVERSION_FACTOR, 0.0)
        
        return concentrations
    
    def predict_pinn_at_sensors(self, facility_params):
        """
        Get PINN predictions at 9 sensor locations for given facility parameters.
        
        Uses existing _compute_pinn_for_facility logic.
        Processes each facility separately, then superimposes.
        
        Args:
            facility_params: Dictionary of facility parameters (same format as predict_full_domain)
        
        Returns:
            sensor_predictions: Array of PINN predictions at 9 sensors [ppb], shape (9,)
            Sensor order matches: ['482010026', '482010057', '482010069', '482010617', 
                                   '482010803', '482011015', '482011035', '482011039', '482016000']
        """
        # Define 9 sensor locations (Cartesian coordinates - EXACT values from training data generation)
        SENSOR_COORDS = np.array([
            [13972.62, 19915.57],  # 482010026
            [3017.18, 12334.2],    # 482010057
            [817.42, 9218.92],     # 482010069
            [27049.57, 22045.66],  # 482010617
            [8836.35, 15717.2],    # 482010803
            [18413.8, 15068.96],   # 482011015
            [1159.98, 12272.52],   # 482011035
            [13661.93, 5193.24],   # 482011039
            [1546.9, 6786.33],     # 482016000
        ])
        
        # Compute PINN at sensor locations for each facility and superimpose
        sensor_pinn = np.zeros(len(SENSOR_COORDS))
        FORECAST_T_HOURS = 3.0  # Simulation time (matches training data)
        UNIT_CONVERSION_FACTOR = 313210039.9  # kg/m^2 to ppb (matches training data)
        
        for facility_name, params in facility_params.items():
            # Use correct keys from facility_params structure
            source_x = params.get('source_x_cartesian', params.get('source_x', 0.0))
            source_y = params.get('source_y_cartesian', params.get('source_y', 0.0))
            source_d = params.get('source_diameter', 0.0)
            Q = params.get('Q_total', params.get('Q', 0.0))
            wind_u = params.get('wind_u', 0.0)
            wind_v = params.get('wind_v', 0.0)
            D = params.get('D', 0.0)
            
            # Compute PINN at each sensor location for this facility
            for i, (sx, sy) in enumerate(SENSOR_COORDS):
                with torch.no_grad():
                    phi_raw = self.pinn(
                        torch.tensor([[sx]], dtype=torch.float32),
                        torch.tensor([[sy]], dtype=torch.float32),
                        torch.tensor([[FORECAST_T_HOURS]], dtype=torch.float32),  # Simulation time
                        torch.tensor([[source_x]], dtype=torch.float32),
                        torch.tensor([[source_y]], dtype=torch.float32),
                        torch.tensor([[wind_u]], dtype=torch.float32),
                        torch.tensor([[wind_v]], dtype=torch.float32),
                        torch.tensor([[source_d]], dtype=torch.float32),
                        torch.tensor([[D]], dtype=torch.float32),
                        torch.tensor([[Q]], dtype=torch.float32),
                        normalize=True
                    )
                    
                    # Convert to ppb and superimpose
                    concentration_ppb = phi_raw.item() * UNIT_CONVERSION_FACTOR
                    sensor_pinn[i] += concentration_ppb
        
        return sensor_pinn
    
    def _apply_kalman_correction(self, x, y, pinn_pred, facility_params, forecast_time, current_sensor_readings):
        """
        Apply Kalman filter correction efficiently using actual 9 sensor locations
        
        1. Compute PINN predictions at 9 sensor locations
        2. Apply Kalman filter to get corrected values at sensors
        3. Calculate correction field (Kalman - PINN) at sensors
        4. Interpolate correction field across entire domain using RBF
        5. Add interpolated corrections to PINN field
        """
        if not self.use_kalman or self.kalman_filter is None:
            # Fall back to PINN only
            print("    Kalman filter not available, using PINN only")
            return pinn_pred
        
        # Define 9 sensor locations (Cartesian coordinates - EXACT values from training data generation)
        SENSOR_COORDS = np.array([
            [13972.62, 19915.57],  # 482010026
            [3017.18, 12334.2],    # 482010057
            [817.42, 9218.92],     # 482010069
            [27049.57, 22045.66],  # 482010617
            [8836.35, 15717.2],    # 482010803
            [18413.8, 15068.96],   # 482011015
            [1159.98, 12272.52],   # 482011035
            [13661.93, 5193.24],   # 482011039
            [1546.9, 6786.33],     # 482016000
        ])
        
        # Step 1: Get PINN predictions at sensor locations using helper method
        sensor_pinn = self.predict_pinn_at_sensors(facility_params)
        
        # Step 2: Apply Kalman filter at sensor locations
        # current_sensor_readings should be array shape (9,) in sensor ID order
        if current_sensor_readings is None or len(current_sensor_readings) != 9:
            print("    Warning: Invalid sensor readings, using PINN only")
            return pinn_pred
        
        # Kalman forecast
        kalman_forecast, uncertainty = self.kalman_filter.forecast(
            current_sensors=current_sensor_readings,
            pinn_predictions=sensor_pinn,
            hours_ahead=3,
            return_uncertainty=True
        )
        
        # Step 3: Calculate correction field at sensors (Kalman - PINN)
        sensor_corrections = kalman_forecast - sensor_pinn
        
        print(f"    Sensor corrections range: {sensor_corrections.min():.2f} to {sensor_corrections.max():.2f} ppb")
        print(f"    Uncertainty range: {uncertainty.min():.2f} to {uncertainty.max():.2f} ppb")
        
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
        
        # Distance-based confidence weighting
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
        pinn_field, corrected_field,
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
        
        # Create RBF interpolators for PINN and corrected fields
        pinn_rbf = Rbf(x_grid, y_grid, pinn_field, function='multiquadric', smooth=0.1)
        corrected_rbf = Rbf(x_grid, y_grid, corrected_field, function='multiquadric', smooth=0.1)
        
        print(f"\n  {'='*80}")
        print(f"  PREDICTIONS AT SENSOR LOCATIONS ({forecast_time.strftime('%Y-%m-%d %H:%M')} UTC)")
        print(f"  {'='*80}")
        print(f"  {'Sensor ID':<12} {'Location (x, y)':<25} {'PINN (ppb)':<12} {'Corrected (ppb)':<12} {'Correction'}")
        print(f"  {'-'*80}")
        
        pinn_values = []
        corrected_values = []
        
        for sensor_id, (sx, sy) in SENSORS.items():
            # Interpolate values at sensor location
            pinn_val = float(pinn_rbf(sx, sy))
            corrected_val = float(corrected_rbf(sx, sy))
            correction = corrected_val - pinn_val
            
            pinn_values.append(pinn_val)
            corrected_values.append(corrected_val)
            
            print(f"  {sensor_id:<12} ({sx:>7.1f}, {sy:>7.1f})   {pinn_val:>8.2f}     {corrected_val:>8.2f}     {correction:>+8.2f}")
        
        print(f"  {'-'*80}")
        print(f"  {'MEAN':<12} {'':<25} {np.mean(pinn_values):>8.2f}     {np.mean(corrected_values):>8.2f}     {np.mean(corrected_values)-np.mean(pinn_values):>+8.2f}")
        print(f"  {'MEDIAN':<12} {'':<25} {np.median(pinn_values):>8.2f}     {np.median(corrected_values):>8.2f}     {np.median(corrected_values)-np.median(pinn_values):>+8.2f}")
        print(f"  {'MAX':<12} {'':<25} {np.max(pinn_values):>8.2f}     {np.max(corrected_values):>8.2f}")
        print(f"  {'MIN':<12} {'':<25} {np.min(pinn_values):>8.2f}     {np.min(corrected_values):>8.2f}")
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
            'corrected_mean': np.mean(corrected_values),
            'corrected_median': np.median(corrected_values),
            'corrected_max': np.max(corrected_values),
            'corrected_min': np.min(corrected_values),
        }
        
        # Add individual sensor values
        for i, (sensor_id, (sx, sy)) in enumerate(SENSORS.items()):
            sensor_data[f'sensor_{sensor_id}_pinn'] = pinn_values[i]
            sensor_data[f'sensor_{sensor_id}_corrected'] = corrected_values[i]
        
        import pandas as pd
        df = pd.DataFrame([sensor_data])
        
        if sensor_log_path.exists():
            df.to_csv(sensor_log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(sensor_log_path, index=False)

    
    def _append_domain_data(
        self, current_time, forecast_time,
        x, y, pinn_field, corrected_field
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
            'corrected_concentration': corrected_field,
            'correction': corrected_field - pinn_field
        })
        
        # Append to superimposed concentrations (PINN only)
        superimposed_path = self.output_dir / "superimposed_concentrations_timeseries.csv"
        df_pinn = df[['forecast_timestamp', 'current_timestamp', 'x', 'y', 'pinn_concentration']]
        
        if superimposed_path.exists():
            df_pinn.to_csv(superimposed_path, mode='a', header=False, index=False)
        else:
            df_pinn.to_csv(superimposed_path, mode='w', header=True, index=False)
            print(f"    Created: {superimposed_path.name}")
        
        # Append to Kalman-corrected domain
        kalman_corrected_path = self.output_dir / "kalman_corrected_domain_timeseries.csv"
        
        if kalman_corrected_path.exists():
            df.to_csv(kalman_corrected_path, mode='a', header=False, index=False)
        else:
            df.to_csv(kalman_corrected_path, mode='w', header=True, index=False)
            print(f"    Created: {kalman_corrected_path.name}")
        
        print(f"  ✓ Appended {len(df)} rows to continuous CSVs")
    
    def _save_latest_snapshot(
        self, forecast_time, x, y, pinn_field, corrected_field
    ):
        """
        Save latest forecast as standalone CSV (overwrite each time)
        """
        df = pd.DataFrame({
            'forecast_timestamp': forecast_time,
            'x': x,
            'y': y,
            'pinn_concentration': pinn_field,
            'corrected_concentration': corrected_field
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
    pinn_field, corrected_field, forecast_time = predictor.predict_full_domain(
        facility_params, current_time
    )
    
    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)
    print(f"Forecast time: {forecast_time}")
    print(f"PINN field: {pinn_field.min():.4f} - {pinn_field.max():.4f} ppb")
    print(f"Corrected field: {corrected_field.min():.4f} - {corrected_field.max():.4f} ppb")
    print("="*70)


if __name__ == "__main__":
    test_predictor()
