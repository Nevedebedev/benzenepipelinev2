"""
NN2 Model Definition - Fixed (No Data Leakage)

This file contains ONLY the model architecture for NN2.
Copy this into your Colab notebook for training.

CRITICAL FIX: Removed current_sensors input to prevent data leakage.
The model now learns corrections based on:
- PINN predictions
- Sensor locations (spatial awareness)
- Wind conditions
- Diffusion coefficients
- Temporal features

NOT on actual sensor values (which aren't available at deployment time).
"""

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# INVERSE TRANSFORM LAYER (for direct ppb output)
# ═══════════════════════════════════════════════════════════════════

class InverseTransformLayer(nn.Module):
    """
    Layer that performs inverse StandardScaler transform: ppb = scaled * scale + mean
    This allows NN2 to output directly in ppb space.
    """
    def __init__(self, scaler_mean, scaler_scale):
        super().__init__()
        # Register as buffers so they're saved with the model
        self.register_buffer('mean', torch.tensor(scaler_mean, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scaler_scale, dtype=torch.float32))
    
    def forward(self, scaled_values):
        """
        Args:
            scaled_values: [batch, n_sensors] - values in scaled space
        Returns:
            ppb_values: [batch, n_sensors] - values in ppb space
        """
        # Inverse transform: ppb = scaled * scale + mean
        return scaled_values * self.scale + self.mean


# ═══════════════════════════════════════════════════════════════════
# NN2 CORRECTION NETWORK - FIXED (NO DATA LEAKAGE)
# ═══════════════════════════════════════════════════════════════════

class NN2_CorrectionNetwork(nn.Module):
    """
    NN2 Correction Network - Fixed Architecture
    
    FIXED: Removed current_sensors input to prevent data leakage.
    
    The model learns corrections to PINN predictions based on:
    - PINN predictions (9 sensors)
    - Sensor coordinates (spatial awareness, 18 features)
    - Wind conditions (2 features: u, v)
    - Diffusion coefficient (1 feature: D)
    - Temporal features (6 features: hour, day, etc.)
    
    Total input: 36 features (was 45 before fix)
    """
    
    def __init__(self, n_sensors=9, scaler_mean=None, scaler_scale=None, output_ppb=True):
        """
        Args:
            n_sensors: Number of sensors (default: 9)
            scaler_mean: Mean from StandardScaler (for inverse transform if output_ppb=True)
            scaler_scale: Scale from StandardScaler (for inverse transform if output_ppb=True)
            output_ppb: If True, output in ppb space; if False, output in scaled space (legacy)
        """
        super().__init__()
        self.n_sensors = n_sensors
        self.output_ppb = output_ppb

        # Input features (NO current_sensors - that was data leakage!):
        # - pinn_predictions (9)
        # - sensor_coords flattened (18)
        # - wind (2)
        # - diffusion (1)
        # - temporal (6)
        # Total: 36 features

        # SIMPLIFIED ARCHITECTURE: 36 → 256 → 128 → 64 → 9 (~100K params)
        # Previous: 36 → 512 → 512 → 256 → 128 → 9 (~452K params)
        # This reduces overfitting risk with limited training data (5,173 samples)
        self.correction_network = nn.Sequential(
            nn.Linear(36, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_sensors)
        )
        
        # Add inverse transform layer if outputting in ppb space
        if output_ppb and scaler_mean is not None and scaler_scale is not None:
            self.inverse_transform = InverseTransformLayer(scaler_mean, scaler_scale)
        else:
            self.inverse_transform = None

    def forward(self, pinn_predictions, sensor_coords, wind, diffusion, temporal):
        """
        Forward pass - computes corrections to PINN predictions.
        
        Args:
            pinn_predictions: [batch, n_sensors] - PINN predictions in scaled space
            sensor_coords: [batch, n_sensors, 2] - Sensor coordinates (x, y)
            wind: [batch, 2] - Wind components (u, v)
            diffusion: [batch, 1] - Diffusion coefficient (D)
            temporal: [batch, 6] - Temporal features (hour, day, etc.)
        
        Returns:
            corrected_predictions: [batch, n_sensors] - Corrected predictions
                - In ppb space if output_ppb=True
                - In scaled space if output_ppb=False (legacy)
            corrections: [batch, n_sensors] - Correction values in scaled space (for regularization)
        
        NOTE: Removed current_sensors input to prevent data leakage!
        The model must learn corrections without seeing actual sensor values.
        """
        batch_size = pinn_predictions.shape[0]

        # Flatten sensor coordinates: [batch, n_sensors, 2] -> [batch, n_sensors*2]
        coords_flat = sensor_coords.reshape(batch_size, -1)

        # Concatenate all features (NO current_sensors - that was data leakage!)
        features = torch.cat([
            pinn_predictions,      # [batch, 9]
            coords_flat,           # [batch, 18]
            wind,                  # [batch, 2]
            diffusion,             # [batch, 1]
            temporal               # [batch, 6]
        ], dim=-1)  # Total: 36 features

        # Compute corrections
        corrections = self.correction_network(features)
        
        # Apply corrections to PINN predictions
        corrected_scaled = pinn_predictions + corrections
        
        # Apply inverse transform if outputting in ppb space
        if self.inverse_transform is not None:
            corrected_ppb = self.inverse_transform(corrected_scaled)
            return corrected_ppb, corrections
        else:
            # Legacy: output in scaled space
            return corrected_scaled, corrections


# ═══════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════

"""
Example usage in training code:

# Initialize model
model = NN2_CorrectionNetwork(
    n_sensors=9,
    scaler_mean=scaler_mean,      # From StandardScaler.fit()
    scaler_scale=scaler_scale,    # From StandardScaler.fit()
    output_ppb=True               # Output in ppb space
)

# Forward pass (NO current_sensors!)
pred, corrections = model(
    pinn_predictions=pinn_preds,      # [batch, 9] - scaled
    sensor_coords=sensor_coords,      # [batch, 9, 2]
    wind=wind_conditions,             # [batch, 2]
    diffusion=diffusion_coeffs,       # [batch, 1]
    temporal=temporal_features         # [batch, 6]
)

# pred: [batch, 9] - corrected predictions in ppb
# corrections: [batch, 9] - correction values in scaled space
"""

