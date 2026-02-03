"""
NN2 Training Script for Colab - Master Model Only (No Leave-One-Out)

This script trains ONLY the master model (no LOOCV).
Faster training since it skips the evaluation step.

CRITICAL FIX: Removed current_sensors input to prevent data leakage.
The model now learns corrections without seeing actual sensor values.

SIMPLIFIED ARCHITECTURE: 36 → 256 → 128 → 64 → 9 (~53K params)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import pickle  # ← CRITICAL: For saving scalers
from datetime import datetime
import os

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    'n_sensors': 9,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 50,
    'lambda_correction': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': '/content/models/master_only/',
    'data_dir': '/content/data/',
    'pinn_file': '/content/data/total_superimposed_concentrations.csv',
    'sensor_coords_file': '/content/data/sensor_coordinates.csv',
    'sensor_file': '/content/data/sensors_final.csv',
    'source_dir': '/content/data/data_nonzero/'
}

# Create save directory
Path(CONFIG['save_dir']).mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 1. INVERSE TRANSFORM LAYER (for direct ppb output)
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
# 2. NN2 NETWORK - FIXED (NO DATA LEAKAGE) - SIMPLIFIED ARCHITECTURE
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

        # Input features (NO current_sensors - that was data leakage!):
        # - pinn_predictions (9)
        # - sensor_coords flattened (18)
        # - wind (2)
        # - diffusion (1)
        # - temporal (6)
        # Total: 36 features

        # SIMPLIFIED ARCHITECTURE: 36 → 256 → 128 → 64 → 9 (~53K params)
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


# ═══════════════════════════════════════════════════════════════════
# 3. DATASET - WITH SCALER MANAGEMENT (FIXED - NO DATA LEAKAGE)
# ═══════════════════════════════════════════════════════════════════

class BenzeneDataset(Dataset):
    def __init__(self, data_path, source_dir, pinn_path, sensor_coords_path,
                 held_out_sensor_idx=None, scalers=None, fit_scalers=False):
        """
        Args:
            sensor_coords_path: Path to CSV with sensor coordinates
            held_out_sensor_idx: Index (0-8) of sensor to hold out (not used in master-only training)
            scalers: Pre-fitted scalers (if available)
            fit_scalers: Whether to fit scalers on this dataset
        """
        self.data_path = data_path
        self.source_dir = source_dir
        self.pinn_path = pinn_path
        self.sensor_coords_path = sensor_coords_path
        self.held_out_sensor_idx = held_out_sensor_idx
        
        # Load sensor data
        self.sensors_df = pd.read_csv(data_path)
        if 't' in self.sensors_df.columns:
            self.sensors_df['timestamp'] = pd.to_datetime(self.sensors_df['t'])
        elif 'timestamp' in self.sensors_df.columns:
            self.sensors_df['timestamp'] = pd.to_datetime(self.sensors_df['timestamp'])
        
        # Get sensor IDs
        sensor_cols = [col for col in self.sensors_df.columns if col.startswith('sensor_')]
        self.all_sensor_ids = sorted(sensor_cols)
        
        # Load sensor coordinates
        coords_df = pd.read_csv(sensor_coords_path)
        self.sensor_coords = {}
        for _, row in coords_df.iterrows():
            self.sensor_coords[row['sensor_id']] = (row['x'], row['y'])
        
        # Get sensor coordinates in order
        self.sensor_coords_array = np.array([
            self.sensor_coords[sid] for sid in self.all_sensor_ids
        ])
        
        # Load PINN predictions
        self.pinn_df = pd.read_csv(pinn_path)
        if 't' in self.pinn_df.columns:
            self.pinn_df['timestamp'] = pd.to_datetime(self.pinn_df['t'])
        elif 'timestamp' in self.pinn_df.columns:
            self.pinn_df['timestamp'] = pd.to_datetime(self.pinn_df['timestamp'])
        
        # Merge on timestamp
        self.merged_df = self.sensors_df.merge(
            self.pinn_df, 
            on='timestamp', 
            how='inner', 
            suffixes=('', '_pinn')
        )
        
        # Load source files for meteorological data
        source_files = sorted(Path(source_dir).glob('*_training_data.csv'))
        self.source_dfs = {}
        for f in source_files:
            name = f.stem.replace('_training_data', '')
            df = pd.read_csv(f, parse_dates=['t'])
            self.source_dfs[name] = df
        
        # Prepare actual sensor values
        self.actual_sensors = self.merged_df[self.all_sensor_ids].values.astype(float)
        self.valid_mask = ~np.isnan(self.actual_sensors)
        self.actual_sensors = np.nan_to_num(self.actual_sensors, nan=0.0)
        
        # Store original ppb values before scaling
        self.actual_sensors_ppb = self.actual_sensors.copy()
        
        # Prepare PINN predictions
        pinn_cols = [f'{sid}_pinn' if f'{sid}_pinn' in self.merged_df.columns else sid 
                     for sid in self.all_sensor_ids]
        self.pinn_predictions = self.merged_df[pinn_cols].values.astype(float)
        self.pinn_predictions = np.nan_to_num(self.pinn_predictions, nan=0.0)
        
        # Initialize scalers
        if scalers is None:
            self.scalers = {
                'pinn': StandardScaler(),
                'sensors': StandardScaler(),
                'wind': StandardScaler(),
                'diffusion': StandardScaler(),
                'coords': StandardScaler()
            }
        else:
            self.scalers = scalers
        
        # Fit or use pre-fitted scalers
        if fit_scalers:
            # Fit on non-zero values only
            valid_pinn = self.pinn_predictions[self.pinn_predictions != 0]
            valid_sensors = self.actual_sensors[self.actual_sensors != 0]
            
            if len(valid_pinn) > 0:
                self.scalers['pinn'].fit(valid_pinn.reshape(-1, 1))
            if len(valid_sensors) > 0:
                self.scalers['sensors'].fit(valid_sensors.reshape(-1, 1))
        
        # Scale PINN predictions
        self.pinn_predictions_scaled = np.zeros_like(self.pinn_predictions)
        for i in range(len(self.pinn_predictions)):
            mask = self.pinn_predictions[i] != 0
            if mask.any():
                self.pinn_predictions_scaled[i, mask] = self.scalers['pinn'].transform(
                    self.pinn_predictions[i, mask].reshape(-1, 1)
                ).flatten()
        
        # Scale actual sensors (for legacy compatibility, but NOT used as input!)
        # Iterate over sensors (columns), not samples (rows)
        for i in range(self.actual_sensors.shape[1]):  # For each sensor
            mask = self.actual_sensors[:, i] != 0  # Mask for this sensor across all samples
            if mask.any():
                self.actual_sensors[mask, i] = self.scalers['sensors'].transform(
                    self.actual_sensors[mask, i].reshape(-1, 1)
                ).flatten()
        
        # Prepare meteorological and temporal features
        self.wind = []
        self.diffusion = []
        self.temporal = []
        
        for idx, row in self.merged_df.iterrows():
            timestamp = row['timestamp']
            met_data_timestamp = timestamp - pd.Timedelta(hours=3)
            
            # Get meteo data from all sources
            all_u = []
            all_v = []
            all_D = []
            
            for name, df in self.source_dfs.items():
                source_data = df[df['t'] == met_data_timestamp]
                if len(source_data) > 0:
                    all_u.append(source_data['wind_u'].values[0])
                    all_v.append(source_data['wind_v'].values[0])
                    all_D.append(source_data['D'].values[0])
            
            if len(all_u) > 0:
                avg_u = np.mean(all_u)
                avg_v = np.mean(all_v)
                avg_D = np.mean(all_D)
            else:
                avg_u = avg_v = avg_D = 0.0
            
            self.wind.append([avg_u, avg_v])
            self.diffusion.append([avg_D])
            
            # Temporal features
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            month = timestamp.month
            is_weekend = 1.0 if day_of_week >= 5 else 0.0
            
            self.temporal.append([
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * day_of_week / 7),
                np.cos(2 * np.pi * day_of_week / 7),
                is_weekend,
                month / 12.0
            ])
        
        self.wind = np.array(self.wind)
        self.diffusion = np.array(self.diffusion)
        self.temporal = np.array(self.temporal)
        
        # Fit scalers for wind, diffusion, coords
        if fit_scalers:
            self.scalers['wind'].fit(self.wind)
            self.scalers['diffusion'].fit(self.diffusion)
            self.scalers['coords'].fit(self.sensor_coords_array)
        
        # Normalize coordinates
        self.sensor_coords_normalized = self.scalers['coords'].transform(self.sensor_coords_array)
        
        # Normalize wind and diffusion
        self.wind_normalized = self.scalers['wind'].transform(self.wind)
        self.diffusion_normalized = self.scalers['diffusion'].transform(self.diffusion)

    def __len__(self):
        return len(self.actual_sensors)

    def __getitem__(self, idx):
        coords_for_sample = torch.FloatTensor(self.sensor_coords_normalized)

        return {
            # REMOVED: 'current_sensors' - this was causing data leakage!
            # The model should learn corrections based on PINN + conditions, not actual sensor values
            'pinn_predictions': torch.FloatTensor(self.pinn_predictions_scaled[idx]),  # Scaled
            'sensor_coords': coords_for_sample,
            'wind': torch.FloatTensor(self.wind_normalized[idx]),
            'diffusion': torch.FloatTensor(self.diffusion_normalized[idx]),
            'temporal': torch.FloatTensor(self.temporal[idx]),
            'target': torch.FloatTensor(self.actual_sensors[idx]),  # Scaled (for legacy)
            'target_ppb': torch.FloatTensor(self.actual_sensors_ppb[idx]),  # PPB (for new training)
            'valid_mask': torch.BoolTensor(self.valid_mask[idx])
        }


# ═══════════════════════════════════════════════════════════════════
# 4. LOSS & TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def correction_loss(pred, target, corrections, valid_mask, lambda_correction=0.001, target_ppb=None, pinn_ppb=None):
    """
    Args:
        pred: Model predictions (in ppb if model outputs ppb, else scaled)
        target: Target values in scaled space (for legacy compatibility)
        corrections: Correction values in scaled space (for regularization)
        valid_mask: Mask for valid (non-zero) values
        lambda_correction: Regularization weight
        target_ppb: Target values in ppb space (for new training)
        pinn_ppb: PINN predictions in ppb space (for direction/size penalties)
    """
    # Use ppb target if provided (new training), else use scaled target (legacy)
    if target_ppb is not None:
        valid_pred = pred[valid_mask]
        valid_target = target_ppb[valid_mask]
    else:
        valid_pred = pred[valid_mask]
        valid_target = target[valid_mask]

    if valid_pred.numel() > 0:
        mse_loss = nn.functional.mse_loss(valid_pred, valid_target)
    else:
        mse_loss = torch.tensor(0.0, device=pred.device)

    correction_reg = lambda_correction * (corrections ** 2).mean()
    
    # NEW: Direction penalty (only if we have targets and PINN in ppb)
    # This penalizes corrections in the wrong direction during training
    if target_ppb is not None and pinn_ppb is not None:
        # Compute needed correction (error signal)
        error = target_ppb - pinn_ppb  # Needed correction
        
        # Compute corrections in ppb space: pred = pinn_ppb + corrections_ppb
        corrections_ppb = pred - pinn_ppb
        
        # Wrong direction: correction and error have opposite signs
        wrong_direction = (corrections_ppb * error < 0) & valid_mask
        if wrong_direction.any():
            direction_penalty = torch.relu(-corrections_ppb[wrong_direction] * error[wrong_direction]).mean()
        else:
            direction_penalty = torch.tensor(0.0, device=pred.device)
        
        # Size penalty: penalize corrections > 50% of PINN magnitude
        pinn_magnitude = torch.abs(pinn_ppb) + 1e-6
        correction_ratio = torch.abs(corrections_ppb) / pinn_magnitude
        size_penalty = torch.relu(correction_ratio[valid_mask] - 0.5).mean()
    else:
        direction_penalty = torch.tensor(0.0, device=pred.device)
        size_penalty = torch.tensor(0.0, device=pred.device)
    
    total = mse_loss + lambda_correction * correction_reg + 0.3 * direction_penalty + 0.1 * size_penalty

    return total, {
        'mse': mse_loss.item(),
        'correction_reg': correction_reg.item(),
        'direction_penalty': direction_penalty.item() if isinstance(direction_penalty, torch.Tensor) else direction_penalty,
        'size_penalty': size_penalty.item() if isinstance(size_penalty, torch.Tensor) else size_penalty,
        'n_valid': valid_pred.numel()
    }


def train_epoch(model, dataloader, optimizer, device, lambda_correction):
    model.train()
    total_loss = 0
    total_valid = 0

    for batch in dataloader:
        # REMOVED: current_sensors - this was data leakage!
        pinn_predictions = batch['pinn_predictions'].to(device)
        sensor_coords = batch['sensor_coords'].to(device)
        wind = batch['wind'].to(device)
        diffusion = batch['diffusion'].to(device)
        temporal = batch['temporal'].to(device)
        target = batch['target'].to(device)  # Scaled (legacy)
        target_ppb = batch.get('target_ppb', None)  # PPB (new)
        if target_ppb is not None:
            target_ppb = target_ppb.to(device)
        valid_mask = batch['valid_mask'].to(device)

        pred, corrections = model(pinn_predictions, sensor_coords,
                                  wind, diffusion, temporal)
        
        # Convert PINN to ppb for loss calculation (if model outputs ppb)
        pinn_ppb = None
        if model.output_ppb and model.inverse_transform is not None:
            pinn_ppb = model.inverse_transform(pinn_predictions)
        
        loss, loss_dict = correction_loss(pred, target, corrections, valid_mask, lambda_correction, target_ppb, pinn_ppb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * loss_dict['n_valid']
        total_valid += loss_dict['n_valid']

    avg_loss = total_loss / max(total_valid, 1)
    return avg_loss


def evaluate_sensor(model, dataloader, device, lambda_correction, sensor_idx):
    """Evaluate performance on a specific sensor"""
    model.eval()

    all_pinn_preds = []
    all_nn2_preds = []
    all_actual = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            # REMOVED: current_sensors - this was data leakage!
            pinn_predictions = batch['pinn_predictions'].to(device)
            sensor_coords = batch['sensor_coords'].to(device)
            wind = batch['wind'].to(device)
            diffusion = batch['diffusion'].to(device)
            temporal = batch['temporal'].to(device)
            target = batch['target'].to(device)
            target_ppb = batch.get('target_ppb', None)
            if target_ppb is not None:
                target_ppb = target_ppb.to(device)
            valid_mask = batch['valid_mask'].to(device)

            pred, corrections = model(pinn_predictions, sensor_coords,
                                     wind, diffusion, temporal)

            # If model outputs ppb, we need to compare with ppb targets
            # Also need to convert PINN predictions from scaled to ppb for comparison
            if target_ppb is not None and model.output_ppb:
                # Model outputs are in ppb, targets are in ppb
                # Need to convert PINN from scaled to ppb for fair comparison
                if model.inverse_transform is not None:
                    pinn_ppb = model.inverse_transform(pinn_predictions)
                else:
                    # Fallback: use target (scaled) - shouldn't happen
                    pinn_ppb = pinn_predictions
                
                all_pinn_preds.append(pinn_ppb[:, sensor_idx].cpu())
                all_nn2_preds.append(pred[:, sensor_idx].cpu())
                all_actual.append(target_ppb[:, sensor_idx].cpu())
            else:
                # Legacy: everything in scaled space
                all_pinn_preds.append(pinn_predictions[:, sensor_idx].cpu())
                all_nn2_preds.append(pred[:, sensor_idx].cpu())
                all_actual.append(target[:, sensor_idx].cpu())
            
            all_masks.append(valid_mask[:, sensor_idx].cpu())

    pinn_preds = torch.cat(all_pinn_preds, dim=0)
    nn2_preds = torch.cat(all_nn2_preds, dim=0)
    actual = torch.cat(all_actual, dim=0)
    masks = torch.cat(all_masks, dim=0)

    valid_pinn = pinn_preds[masks]
    valid_nn2 = nn2_preds[masks]
    valid_actual = actual[masks]

    if valid_pinn.numel() > 0:
        pinn_mae = torch.abs(valid_pinn - valid_actual).mean().item()
        pinn_rmse = torch.sqrt(((valid_pinn - valid_actual) ** 2).mean()).item()
        nn2_mae = torch.abs(valid_nn2 - valid_actual).mean().item()
        nn2_rmse = torch.sqrt(((valid_nn2 - valid_actual) ** 2).mean()).item()
        improvement = ((pinn_mae - nn2_mae) / pinn_mae * 100) if pinn_mae > 0 else 0
    else:
        pinn_mae = pinn_rmse = nn2_mae = nn2_rmse = improvement = 0

    return {
        'pinn_mae': pinn_mae,
        'pinn_rmse': pinn_rmse,
        'nn2_mae': nn2_mae,
        'nn2_rmse': nn2_rmse,
        'improvement': improvement
    }


# ═══════════════════════════════════════════════════════════════════
# 5. MASTER MODEL TRAINING (ALL SENSORS) - NO LEAVE-ONE-OUT
# ═══════════════════════════════════════════════════════════════════

def train_master_model():
    """Train master model on all sensors (no holdout, no LOOCV)"""
    
    print("="*80)
    print("NN2 TRAINING - MASTER MODEL ONLY (ALL SENSORS)")
    print("="*80)
    print(f"\nDevice: {CONFIG['device']}")
    print(f"Save directory: {CONFIG['save_dir']}\n")
    
    # First, fit scalers on full dataset
    print("Fitting scalers on full dataset...")
    full_dataset = BenzeneDataset(
        data_path=CONFIG['sensor_file'],
        source_dir=CONFIG['source_dir'],
        pinn_path=CONFIG['pinn_file'],
        sensor_coords_path=CONFIG['sensor_coords_file'],
        held_out_sensor_idx=None,  # No holdout
        scalers=None,
        fit_scalers=True
    )
    scalers = full_dataset.scalers
    print("✓ Scalers fitted\n")
    
    # Get scaler parameters for inverse transform
    scaler_mean = scalers['sensors'].mean_[0] if hasattr(scalers['sensors'], 'mean_') else None
    scaler_scale = scalers['sensors'].scale_[0] if hasattr(scalers['sensors'], 'scale_') else None
    
    # Split into train/val (80/20)
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    print(f"✓ Loaded {len(full_dataset)} samples")
    print(f"  Train samples: {n_train}")
    print(f"  Val samples: {n_val}\n")
    
    # Initialize model
    model = NN2_CorrectionNetwork(
        n_sensors=CONFIG['n_sensors'],
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        output_ppb=True  # Output directly in ppb
    ).to(CONFIG['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    print("Starting training...\n")
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(CONFIG['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, CONFIG['device'], CONFIG['lambda_correction'])
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pinn_predictions = batch['pinn_predictions'].to(CONFIG['device'])
                sensor_coords = batch['sensor_coords'].to(CONFIG['device'])
                wind = batch['wind'].to(CONFIG['device'])
                diffusion = batch['diffusion'].to(CONFIG['device'])
                temporal = batch['temporal'].to(CONFIG['device'])
                target = batch['target'].to(CONFIG['device'])
                target_ppb = batch.get('target_ppb', None)
                if target_ppb is not None:
                    target_ppb = target_ppb.to(CONFIG['device'])
                valid_mask = batch['valid_mask'].to(CONFIG['device'])

                pred, corrections = model(pinn_predictions, sensor_coords,
                                        wind, diffusion, temporal)
                
                # Convert PINN to ppb for loss calculation
                pinn_ppb_eval = None
                if model.output_ppb and model.inverse_transform is not None:
                    pinn_ppb_eval = model.inverse_transform(pinn_predictions)
                
                loss, _ = correction_loss(pred, target, corrections, valid_mask, CONFIG['lambda_correction'], target_ppb, pinn_ppb_eval)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save master model
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'scaler_mean': scaler_mean,
                'scaler_scale': scaler_scale,
                'output_ppb': True
            }
            torch.save(checkpoint, f"{CONFIG['save_dir']}/nn2_master_model_ppb.pth")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Evaluate on all sensors
    print("\nEvaluating master model on all sensors...")
    all_results = {}
    for sensor_idx in range(CONFIG['n_sensors']):
        sensor_id = full_dataset.all_sensor_ids[sensor_idx]
        eval_results = evaluate_sensor(model, val_loader, CONFIG['device'], CONFIG['lambda_correction'], sensor_idx)
        all_results[sensor_id] = eval_results
    
    # Print summary
    print("\n" + "="*80)
    print("MASTER MODEL RESULTS (All Sensors)")
    print("="*80)
    for sensor_id, results in all_results.items():
        print(f"{sensor_id}: PINN MAE={results['pinn_mae']:.4f} ppb, "
              f"NN2 MAE={results['nn2_mae']:.4f} ppb, "
              f"Improvement={results['improvement']:.1f}%")
    
    avg_improvement = np.mean([r['improvement'] for r in all_results.values()])
    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    print(f"Master model saved to: {CONFIG['save_dir']}/nn2_master_model_ppb.pth")
    print("="*80)
    
    # Save scalers
    scalers_file = f"{CONFIG['save_dir']}/nn2_master_scalers-2.pkl"
    with open(scalers_file, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to: {scalers_file}")
    
    return model, all_results, scalers


# ═══════════════════════════════════════════════════════════════════
# 6. RUN TRAINING
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Verify data files exist
    print("Verifying data files...")
    required_files = [
        CONFIG['sensor_file'],
        CONFIG['sensor_coords_file'],
        CONFIG['pinn_file']
    ]
    
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")
        print(f"  ✓ {f}")
    
    if not os.path.exists(CONFIG['source_dir']):
        raise FileNotFoundError(f"Source directory not found: {CONFIG['source_dir']}")
    print(f"  ✓ {CONFIG['source_dir']}")
    
    print("\n✓ All data files found!\n")
    
    # Train master model only (no LOOCV)
    master_model, master_results, scalers = train_master_model()
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFiles saved:")
    print(f"  • Master model: {CONFIG['save_dir']}/nn2_master_model_ppb.pth")
    print(f"  • Scalers: {CONFIG['save_dir']}/nn2_master_scalers-2.pkl")
    print("="*80)

