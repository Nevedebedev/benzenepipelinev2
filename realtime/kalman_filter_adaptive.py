"""
Adaptive Kalman Filter for Benzene Forecasting
Changes parameters based on concentration level (normal/elevated/emergency).
"""

import numpy as np
from kalman_filter import BenzeneKalmanFilter
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdaptiveBenzeneKalmanFilter(BenzeneKalmanFilter):
    """
    Adaptive Kalman filter that adjusts parameters based on concentration level.
    
    Three modes:
    - Normal (< 5 ppb): Smooth operation, low process noise
    - Elevated (5-10 ppb): More responsive, moderate parameters
    - Emergency (> 10 ppb): Very responsive, trust PINN, slow decay
    """
    
    def __init__(
        self,
        n_sensors: int = 9,
        initial_state: Optional[np.ndarray] = None,
        base_process_noise: float = 0.1,
        base_measurement_noise: float = 0.01,
        base_decay_rate: float = 0.5,
        base_pinn_weight: float = 0.1,
        adaptive_mode: bool = True
    ):
        """
        Initialize adaptive Kalman filter.
        
        Args:
            n_sensors: Number of monitoring sensors
            initial_state: Initial concentration estimates
            base_process_noise: Base process noise (used in normal mode)
            base_measurement_noise: Base measurement noise
            base_decay_rate: Base decay rate (used in normal mode)
            base_pinn_weight: Base PINN weight (used in normal mode)
            adaptive_mode: Whether to use adaptive parameters (default True)
        """
        # Initialize with base parameters
        super().__init__(
            n_sensors=n_sensors,
            initial_state=initial_state,
            process_noise=base_process_noise,
            measurement_noise=base_measurement_noise,
            decay_rate=base_decay_rate,
            pinn_weight=base_pinn_weight
        )
        
        self.base_process_noise = base_process_noise
        self.base_measurement_noise = base_measurement_noise
        self.base_decay_rate = base_decay_rate
        self.base_pinn_weight = base_pinn_weight
        self.adaptive_mode = adaptive_mode
        
        # Store current mode for each sensor
        self.sensor_modes = np.zeros(n_sensors, dtype=int)  # 0=normal, 1=elevated, 2=emergency
    
    def get_adaptive_params(self, current_concentration: float) -> dict:
        """
        Get adaptive parameters based on concentration level.
        
        Args:
            current_concentration: Current concentration estimate [ppb]
        
        Returns:
            Dictionary with adaptive parameters
        """
        if not self.adaptive_mode:
            return {
                'process_noise': self.base_process_noise,
                'decay_rate': self.base_decay_rate,
                'pinn_weight': self.base_pinn_weight
            }
        
        if current_concentration < 5.0:
            # Normal conditions - smooth operation
            return {
                'process_noise': self.base_process_noise,
                'decay_rate': self.base_decay_rate,
                'pinn_weight': self.base_pinn_weight
            }
        elif current_concentration < 10.0:
            # Elevated - more responsive
            return {
                'process_noise': self.base_process_noise * 5.0,  # 5x more responsive
                'decay_rate': self.base_decay_rate * 0.6,        # Slower decay (0.3 if base=0.5)
                'pinn_weight': self.base_pinn_weight * 3.0       # Trust PINN more (0.3 if base=0.1)
            }
        else:
            # Emergency - very responsive, trust PINN, slow decay
            return {
                'process_noise': self.base_process_noise * 20.0,  # Very high uncertainty
                'decay_rate': self.base_decay_rate * 0.2,        # Very slow decay (0.1 if base=0.5)
                'pinn_weight': min(self.base_pinn_weight * 7.0, 0.9)  # Trust PINN (0.7 if base=0.1)
            }
    
    def predict(
        self, 
        pinn_predictions: np.ndarray,
        time_delta: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step with adaptive parameters.
        
        Uses current state to determine mode, then applies adaptive parameters.
        """
        if self.adaptive_mode:
            # Determine mode for each sensor based on current state
            for i in range(self.n_sensors):
                conc = self.kf.x[i]
                if conc < 5.0:
                    self.sensor_modes[i] = 0  # Normal
                elif conc < 10.0:
                    self.sensor_modes[i] = 1  # Elevated
                else:
                    self.sensor_modes[i] = 2  # Emergency
            
            # Get adaptive parameters for each sensor mode
            # For simplicity, use average concentration to determine global mode
            avg_concentration = np.mean(self.kf.x)
            adaptive_params = self.get_adaptive_params(avg_concentration)
            
            # Update filter parameters
            self.decay_rate = adaptive_params['decay_rate']
            self.pinn_weight = adaptive_params['pinn_weight']
            
            # Update process noise (can be per-sensor, but using average for now)
            # In full implementation, could use per-sensor process noise
            self.kf.Q = np.eye(self.n_sensors) * adaptive_params['process_noise']
        
        # Call parent predict with updated parameters
        return super().predict(pinn_predictions, time_delta)
    
    def forecast(
        self,
        current_sensors: np.ndarray,
        pinn_predictions: np.ndarray,
        hours_ahead: int = 3,
        return_uncertainty: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Complete forecast cycle with adaptive parameters.
        
        Before predict step, determines mode based on current sensors and PINN predictions.
        """
        if self.adaptive_mode:
            # Determine initial mode from current sensors and PINN predictions
            # Use maximum of current sensors and PINN to be conservative
            max_current = np.nanmax(current_sensors) if not np.all(np.isnan(current_sensors)) else 0.0
            max_pinn = np.nanmax(pinn_predictions) if not np.all(np.isnan(pinn_predictions)) else 0.0
            max_concentration = max(max_current, max_pinn)
            
            # Get adaptive parameters
            adaptive_params = self.get_adaptive_params(max_concentration)
            
            # Update filter parameters
            self.decay_rate = adaptive_params['decay_rate']
            self.pinn_weight = adaptive_params['pinn_weight']
            self.kf.Q = np.eye(self.n_sensors) * adaptive_params['process_noise']
            
            # Log mode if emergency
            if max_concentration >= 10.0:
                logger.warning(f"EMERGENCY MODE: Concentration {max_concentration:.2f} ppb detected. "
                             f"Using aggressive parameters: pn={adaptive_params['process_noise']:.2f}, "
                             f"dr={adaptive_params['decay_rate']:.2f}, pw={adaptive_params['pinn_weight']:.2f}")
        
        # Call parent forecast
        return super().forecast(current_sensors, pinn_predictions, hours_ahead, return_uncertainty)
    
    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset filter to initial conditions."""
        super().reset(initial_state)
        self.sensor_modes = np.zeros(self.n_sensors, dtype=int)

