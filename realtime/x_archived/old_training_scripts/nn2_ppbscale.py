"""
NN2 Model with Direct PPB Output

This file contains the NN2_CorrectionNetwork model that outputs directly in ppb space,
eliminating the need for inverse transform and fixing distribution shift issues.

Copy this entire file into your Colab notebook.
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
# NN2 NETWORK - WITH SPATIAL AWARENESS + DIRECT PPB OUTPUT
# ═══════════════════════════════════════════════════════════════════

class NN2_CorrectionNetwork(nn.Module):
    def __init__(self, n_sensors=9, scaler_mean=None, scaler_scale=None, output_ppb=True):
        """
        Args:
            n_sensors: Number of sensors
            scaler_mean: Mean from StandardScaler (for inverse transform)
            scaler_scale: Scale from StandardScaler (for inverse transform)
            output_ppb: If True, output in ppb space; if False, output in scaled space (legacy)
        """
        super().__init__()
        self.n_sensors = n_sensors
        self.output_ppb = output_ppb

        # Input features:
        # - pinn_predictions (9)
        # - sensor_coords flattened (18)
        # - wind (2)
        # - diffusion (1)
        # - temporal (6)
        # Total: 36
        # NOTE: Removed current_sensors to prevent data leakage!

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
        Args:
            pinn_predictions: [batch, n_sensors] - in scaled space
            sensor_coords: [batch, n_sensors, 2]
            wind: [batch, 2]
            diffusion: [batch, 1]
            temporal: [batch, 6]
        Returns:
            corrected_predictions: [batch, n_sensors] - in ppb space if output_ppb=True, else scaled space
            corrections: [batch, n_sensors] - in scaled space (for regularization)
        
        NOTE: Removed current_sensors input to prevent data leakage!
        """
        batch_size = pinn_predictions.shape[0]

        # Flatten sensor coordinates
        coords_flat = sensor_coords.reshape(batch_size, -1)

        # Concatenate all features (NO current_sensors - that was data leakage!)
        features = torch.cat([
            pinn_predictions,      # [batch, 9]
            coords_flat,           # [batch, 18]
            wind,                  # [batch, 2]
            diffusion,             # [batch, 1]
            temporal               # [batch, 6]
        ], dim=-1)  # Total: 36

        corrections = self.correction_network(features)
        corrected_scaled = pinn_predictions + corrections
        
        # Apply inverse transform if outputting in ppb space
        if self.inverse_transform is not None:
            corrected_ppb = self.inverse_transform(corrected_scaled)
            return corrected_ppb, corrections
        else:
            # Legacy: output in scaled space
            return corrected_scaled, corrections

