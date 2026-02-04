"""
Benzene Kalman Filter
Combines PINN physics predictions with real-time sensor data
for improved 3-hour forecasts with uncertainty quantification.
"""

import numpy as np
from filterpy.kalman import KalmanFilter as KF
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BenzeneKalmanFilter:
    """
    Kalman filter for benzene concentration forecasting.
    
    State vector: [C_sensor_0, C_sensor_1, ..., C_sensor_8]
    
    Process model: Combines PINN physics with exponential decay
    Measurement model: Direct sensor observations
    """
    
    def __init__(
        self,
        n_sensors: int = 9,
        initial_state: Optional[np.ndarray] = None,
        process_noise: float = 1.0,
        measurement_noise: float = 0.01,
        decay_rate: float = 0.7,
        pinn_weight: float = 0.3
    ):
        """
        Initialize Kalman filter.
        
        Args:
            n_sensors: Number of monitoring sensors (default 9)
            initial_state: Initial concentration estimates [ppb]
            process_noise: Process noise variance (Q matrix diagonal)
            measurement_noise: Measurement noise variance (R matrix diagonal)
            decay_rate: Exponential decay coefficient (0-1)
            pinn_weight: Weight for PINN predictions in state transition (0-1)
        """
        self.n_sensors = n_sensors
        self.decay_rate = decay_rate
        self.pinn_weight = pinn_weight
        
        # Initialize Kalman filter
        self.kf = KF(dim_x=n_sensors, dim_z=n_sensors)
        
        # Initial state
        if initial_state is None:
            self.kf.x = np.zeros(n_sensors)  # Start at zero
        else:
            self.kf.x = initial_state.copy()
        
        # Initial state covariance (high uncertainty)
        self.kf.P = np.eye(n_sensors) * 10.0
        
        # Measurement matrix (identity - we measure concentrations directly)
        self.kf.H = np.eye(n_sensors)
        
        # Measurement noise covariance
        # R[i,i] = sensor uncertainty variance (assumed 0.1 ppb std dev)
        self.kf.R = np.eye(n_sensors) * measurement_noise
        
        # Process noise covariance
        # Q[i,i] = model uncertainty variance
        self.kf.Q = np.eye(n_sensors) * process_noise
        
        # State transition matrix (will be updated each step)
        # F = decay * I (concentrations decay over time)
        self.kf.F = np.eye(n_sensors) * decay_rate
        
        # Control input matrix (for PINN predictions)
        # B = (1 - decay) * pinn_weight * I
        self.kf.B = np.eye(n_sensors) * (1 - decay_rate) * pinn_weight
    
    def predict(
        self, 
        pinn_predictions: np.ndarray,
        time_delta: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step: Use PINN + physics to forecast future state.
        
        Args:
            pinn_predictions: PINN predicted concentrations at T+3 [ppb], shape (9,)
            time_delta: Time step in hours (default 3.0)
        
        Returns:
            predicted_state: Predicted concentrations [ppb], shape (9,)
            predicted_covariance: Uncertainty covariance matrix, shape (9, 9)
        """
        # Adjust state transition for time step
        # F = decay^(time_delta/3)  (scale decay by time step)
        decay_adjusted = self.decay_rate ** (time_delta / 3.0)
        self.kf.F = np.eye(self.n_sensors) * decay_adjusted
        self.kf.B = np.eye(self.n_sensors) * (1 - decay_adjusted) * self.pinn_weight
        
        # Control input = PINN predictions
        u = pinn_predictions
        
        # Kalman predict step
        self.kf.predict(u=u)
        
        return self.kf.x.copy(), self.kf.P.copy()
    
    def update(
        self, 
        sensor_measurements: np.ndarray,
        sensor_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step: Incorporate sensor measurements.
        
        Args:
            sensor_measurements: Current EDF sensor readings [ppb], shape (9,)
            sensor_mask: Boolean mask for valid sensors, shape (9,)
                        True = valid, False/None = missing/invalid
        
        Returns:
            updated_state: Corrected concentrations [ppb], shape (9,)
            updated_covariance: Updated uncertainty covariance, shape (9, 9)
        """
        # Handle missing sensors
        if sensor_mask is None:
            sensor_mask = ~np.isnan(sensor_measurements)
        
        valid_indices = np.where(sensor_mask)[0]
        
        if len(valid_indices) == 0:
            # No valid sensors - skip update, return prediction
            logger.warning("No valid sensor measurements - skipping update")
            return self.kf.x.copy(), self.kf.P.copy()
        
        # Extract valid measurements
        z = sensor_measurements[valid_indices]
        
        # Create measurement matrix for valid sensors only
        H_partial = self.kf.H[valid_indices, :]
        R_partial = self.kf.R[np.ix_(valid_indices, valid_indices)]
        
        # Kalman update (manually for partial measurements)
        # y = z - H*x (innovation)
        y = z - H_partial @ self.kf.x
        
        # S = H*P*H' + R (innovation covariance)
        S = H_partial @ self.kf.P @ H_partial.T + R_partial
        
        # K = P*H'*inv(S) (Kalman gain)
        K = self.kf.P @ H_partial.T @ np.linalg.inv(S)
        
        # x = x + K*y (state update)
        self.kf.x = self.kf.x + K @ y
        
        # P = (I - K*H)*P (covariance update)
        I_KH = np.eye(self.n_sensors) - K @ H_partial
        self.kf.P = I_KH @ self.kf.P
        
        return self.kf.x.copy(), self.kf.P.copy()
    
    def forecast(
        self,
        current_sensors: np.ndarray,
        pinn_predictions: np.ndarray,
        hours_ahead: int = 3,
        return_uncertainty: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Complete forecast cycle: predict + update.
        
        This is the main method for operational use.
        
        Args:
            current_sensors: EDF sensor readings at time T [ppb], shape (9,)
            pinn_predictions: PINN predictions for time T+3 [ppb], shape (9,)
            hours_ahead: Forecast horizon (default 3)
            return_uncertainty: Whether to return std deviations
        
        Returns:
            forecast: Predicted concentrations at T+hours_ahead [ppb], shape (9,)
            uncertainty: Standard deviations [ppb], shape (9,) (if return_uncertainty=True)
        """
        # Step 1: Predict using PINN
        predicted_state, predicted_cov = self.predict(pinn_predictions, time_delta=hours_ahead)
        
        # Step 2: Update using current sensor measurements
        forecast, final_cov = self.update(current_sensors)
        
        if return_uncertainty:
            # Extract std deviation from diagonal of covariance matrix
            uncertainty = np.sqrt(np.diag(final_cov))
            return forecast, uncertainty
        else:
            return forecast, None
    
    def get_confidence_interval(
        self, 
        forecast: np.ndarray, 
        uncertainty: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for forecasts.
        
        Args:
            forecast: Point forecasts [ppb], shape (9,)
            uncertainty: Standard deviations [ppb], shape (9,)
            confidence: Confidence level (default 0.95 for 95%)
        
        Returns:
            lower_bound: Lower confidence bound [ppb], shape (9,)
            upper_bound: Upper confidence bound [ppb], shape (9,)
        """
        # Z-score for confidence level
        from scipy.stats import norm
        z = norm.ppf((1 + confidence) / 2)
        
        lower_bound = forecast - z * uncertainty
        upper_bound = forecast + z * uncertainty
        
        # Concentrations cannot be negative
        lower_bound = np.maximum(lower_bound, 0.0)
        
        return lower_bound, upper_bound
    
    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset filter to initial conditions."""
        if initial_state is None:
            self.kf.x = np.zeros(self.n_sensors)
        else:
            self.kf.x = initial_state.copy()
        
        self.kf.P = np.eye(self.n_sensors) * 10.0

