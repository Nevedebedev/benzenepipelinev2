import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
from datetime import datetime
import os

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    'n_sensors': 9,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 50,  # Reduced for faster iteration
    'lambda_correction': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': '/content/models/leave_one_out/',
    'data_dir': '/content/data/',
    'pinn_file': '/content/data/total_superimposed_concentrations.csv',
    'sensor_coords_file': '/content/data/sensor_coordinates.csv'  # NEW!
}

# ═══════════════════════════════════════════════════════════════════
# 1. NN2 NETWORK - NOW WITH SPATIAL AWARENESS
# ═══════════════════════════════════════════════════════════════════

class NN2_CorrectionNetwork(nn.Module):
    def __init__(self, n_sensors=9):
        super().__init__()
        self.n_sensors = n_sensors

        # Input features per sensor:
        # - pinn_prediction (1)
        # - current_sensor (1)
        # - sensor coordinates (2)
        # Plus global features:
        # - wind (2)
        # - diffusion (1)
        # - temporal (6)
        # Total: n_sensors * 4 + 9 = 9*4 + 9 = 45

        self.correction_network = nn.Sequential(
            nn.Linear(45, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_sensors)
        )

    def forward(self, current_sensors, pinn_predictions, sensor_coords, wind, diffusion, temporal):
        """
        Args:
            current_sensors: [batch, n_sensors]
            pinn_predictions: [batch, n_sensors]
            sensor_coords: [batch, n_sensors, 2] - NEW!
            wind: [batch, 2]
            diffusion: [batch, 1]
            temporal: [batch, 6]
        """
        # Flatten coords: [batch, 9, 2] -> [batch, 18]
        batch_size = current_sensors.shape[0]

        # Flatten sensor coordinates: [batch, n_sensors, 2] -> [batch, n_sensors*2]
        coords_flat = sensor_coords.reshape(batch_size, -1)

        # Concatenate all features
        features = torch.cat([
            pinn_predictions,      # [batch, 9]
            current_sensors,       # [batch, 9]
            coords_flat,           # [batch, 18]  <- NEW!
            wind,                  # [batch, 2]
            diffusion,             # [batch, 1]
            temporal               # [batch, 6]
        ], dim=-1)  # Total: 45

        corrections = self.correction_network(features)
        corrected_predictions = pinn_predictions + corrections
        return corrected_predictions, corrections


# ═══════════════════════════════════════════════════════════════════
# 2. DATASET - NOW LOADS SENSOR COORDINATES
# ═══════════════════════════════════════════════════════════════════

class BenzeneDataset(Dataset):
    def __init__(self, data_path, source_dir, pinn_path, sensor_coords_path,
                 held_out_sensor_idx=None, scalers=None, fit_scalers=False):
        """
        Args:
            sensor_coords_path: Path to CSV with sensor coordinates
            held_out_sensor_idx: Index (0-8) of sensor to hold out for testing
                                If None, use all sensors
        """
        self.held_out_sensor_idx = held_out_sensor_idx

        print(f"\n{'='*70}")
        if held_out_sensor_idx is not None:
            print(f"🔄 Loading Data - HOLDING OUT Sensor #{held_out_sensor_idx}")
        else:
            print(f"🔄 Loading Data - Using ALL Sensors")
        print(f"{'='*70}")

        # Load sensor coordinates
        print(f"\n📍 Loading sensor coordinates from {sensor_coords_path}...")
        coords_df = pd.read_csv(sensor_coords_path)
        # Ensure they're in order sensor_0, sensor_1, ..., sensor_8
        coords_df = coords_df.sort_values('sensor_id').reset_index(drop=True)
        self.sensor_coords = coords_df[['x', 'y']].values.astype(float)
        print(f"   Loaded coordinates for {len(self.sensor_coords)} sensors")
        print(f"   Coordinate range: X=[{self.sensor_coords[:, 0].min():.1f}, {self.sensor_coords[:, 0].max():.1f}], "
              f"Y=[{self.sensor_coords[:, 1].min():.1f}, {self.sensor_coords[:, 1].max():.1f}]")

        # Load sensor data
        self.sensors_df = pd.read_csv(data_path)
        if 'timestamp' not in self.sensors_df.columns:
            if 't' in self.sensors_df.columns:
                self.sensors_df = self.sensors_df.rename(columns={'t': 'timestamp'})
        self.sensors_df['timestamp'] = pd.to_datetime(self.sensors_df['timestamp'])

        print(f"\n✓ Sensor data: {len(self.sensors_df)} rows")

        # Load PINN predictions
        print(f"\n📊 Loading PINN predictions...")
        self.pinn_df = pd.read_csv(pinn_path)

        if 't' in self.pinn_df.columns:
            self.pinn_df = self.pinn_df.rename(columns={'t': 'timestamp'})
        self.pinn_df['timestamp'] = pd.to_datetime(self.pinn_df['timestamp'])

        # Load meteorology
        source_files = sorted(Path(source_dir).glob('*.csv'))
        first_file = source_files[0]
        df = pd.read_csv(first_file)
        if 'timestamp' not in df.columns:
            if 't' in df.columns:
                df = df.rename(columns={'t': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        self.meteo_df = df

        # Get all sensor IDs
        all_sensor_ids = [col for col in self.sensors_df.columns if col.startswith('sensor_')]
        self.all_sensor_ids = all_sensor_ids
        self.n_sensors = len(all_sensor_ids)

        # Determine which sensors to use for training
        if held_out_sensor_idx is not None:
            self.train_sensor_ids = [sid for i, sid in enumerate(all_sensor_ids) if i != held_out_sensor_idx]
            self.held_out_sensor_id = all_sensor_ids[held_out_sensor_idx]
            print(f"\n🎯 Training sensors: {len(self.train_sensor_ids)}")
            print(f"   {self.train_sensor_ids}")
            print(f"\n🔒 Held-out sensor: {self.held_out_sensor_id} (index {held_out_sensor_idx})")
            print(f"   Location: ({self.sensor_coords[held_out_sensor_idx, 0]:.1f}, "
                  f"{self.sensor_coords[held_out_sensor_idx, 1]:.1f})")
        else:
            self.train_sensor_ids = all_sensor_ids
            self.held_out_sensor_id = None

        # Initialize scalers
        if fit_scalers:
            self.scalers = {
                'sensors': StandardScaler(),
                'pinn': StandardScaler(),
                'wind': StandardScaler(),
                'diffusion': StandardScaler(),
                'coords': StandardScaler(),  # NEW!
            }
        else:
            self.scalers = scalers

        self._process_features(fit_scalers)

        print(f"\n{'='*70}")
        print(f"✓ Dataset ready: {len(self)} samples")
        print(f"{'='*70}\n")

    def _process_features(self, fit=False):
        print("\n🔄 Processing features...")

        # Find common timestamps
        sensor_times = set(self.sensors_df['timestamp'])
        pinn_times = set(self.pinn_df['timestamp'])
        meteo_times = set(self.meteo_df['timestamp'])

        common_times = sensor_times & pinn_times & meteo_times
        common_times = sorted(list(common_times))

        print(f"  Common timestamps: {len(common_times)}")

        # Filter to common timestamps and align
        self.sensors_df = self.sensors_df[self.sensors_df['timestamp'].isin(common_times)]
        self.sensors_df = self.sensors_df.sort_values('timestamp').reset_index(drop=True)

        self.pinn_df = self.pinn_df[self.pinn_df['timestamp'].isin(common_times)]
        self.pinn_df = self.pinn_df.sort_values('timestamp').reset_index(drop=True)

        self.meteo_df = self.meteo_df[self.meteo_df['timestamp'].isin(common_times)]
        self.meteo_df = self.meteo_df.sort_values('timestamp').reset_index(drop=True)

        n_samples = len(common_times)

        # ═══════════════════════════════════════════════════════════
        # ACTUAL SENSOR READINGS
        # ═══════════════════════════════════════════════════════════
        self.actual_sensors = self.sensors_df[self.all_sensor_ids].values.astype(float)
        self.valid_mask = ~np.isnan(self.actual_sensors)
        self.actual_sensors = np.nan_to_num(self.actual_sensors, nan=0.0)

        # ═══════════════════════════════════════════════════════════
        # PINN PREDICTIONS (at same timestamp)
        # ═══════════════════════════════════════════════════════════
        print(f"\n  🔮 Extracting PINN predictions from sensor columns...")
        self.pinn_predictions = self.pinn_df[self.all_sensor_ids].values.astype(float)

        # ═══════════════════════════════════════════════════════════
        # METEOROLOGY
        # ═══════════════════════════════════════════════════════════
        self.wind = self.meteo_df[['wind_u', 'wind_v']].values.astype(float)
        self.diffusion = self.meteo_df['D'].values.astype(float).reshape(-1, 1)

        # ═══════════════════════════════════════════════════════════
        # TEMPORAL FEATURES
        # ═══════════════════════════════════════════════════════════
        timestamps = self.sensors_df['timestamp']
        hour = timestamps.dt.hour
        day_of_week = timestamps.dt.dayofweek
        month = timestamps.dt.month
        is_weekend = (day_of_week >= 5).astype(float)

        self.temporal = np.column_stack([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            is_weekend,
            month / 12.0
        ])

        # ═══════════════════════════════════════════════════════════
        # NORMALIZATION
        # ═══════════════════════════════════════════════════════════
        if fit:
            print(f"\n  🔧 Fitting scalers...")

            # Fit coordinate scaler
            self.scalers['coords'].fit(self.sensor_coords)

            # If holding out a sensor, only fit on training sensors
            if self.held_out_sensor_idx is not None:
                # Create mask for training sensors only
                train_sensor_mask = np.ones(self.n_sensors, dtype=bool)
                train_sensor_mask[self.held_out_sensor_idx] = False

                # Fit on training sensors only
                train_actual = self.actual_sensors[:, train_sensor_mask]
                train_pinn = self.pinn_predictions[:, train_sensor_mask]

                valid_sensors = train_actual[train_actual != 0]
                if len(valid_sensors) > 0:
                    self.scalers['sensors'].fit(valid_sensors.reshape(-1, 1))

                valid_pinn = train_pinn[train_pinn != 0]
                if len(valid_pinn) > 0:
                    self.scalers['pinn'].fit(valid_pinn.reshape(-1, 1))
            else:
                # Use all sensors
                valid_sensors = self.actual_sensors[self.actual_sensors != 0]
                if len(valid_sensors) > 0:
                    self.scalers['sensors'].fit(valid_sensors.reshape(-1, 1))

                valid_pinn = self.pinn_predictions[self.pinn_predictions != 0]
                if len(valid_pinn) > 0:
                    self.scalers['pinn'].fit(valid_pinn.reshape(-1, 1))

            self.scalers['wind'].fit(self.wind)
            self.scalers['diffusion'].fit(self.diffusion)

        # Transform ALL sensors (including held-out for testing)
        for i in range(self.n_sensors):
            mask = self.actual_sensors[:, i] != 0
            if mask.any():
                self.actual_sensors[mask, i] = self.scalers['sensors'].transform(
                    self.actual_sensors[mask, i].reshape(-1, 1)
                ).flatten()

            mask = self.pinn_predictions[:, i] != 0
            if mask.any():
                self.pinn_predictions[mask, i] = self.scalers['pinn'].transform(
                    self.pinn_predictions[mask, i].reshape(-1, 1)
                ).flatten()

        self.wind = self.scalers['wind'].transform(self.wind)
        self.diffusion = self.scalers['diffusion'].transform(self.diffusion)

        # Transform coordinates
        self.sensor_coords_normalized = self.scalers['coords'].transform(self.sensor_coords)

    def __len__(self):
        return len(self.actual_sensors)

    def __getitem__(self, idx):
        # Repeat sensor coordinates for this sample
        # Shape: [n_sensors, 2]
        coords_for_sample = torch.FloatTensor(self.sensor_coords_normalized)

        return {
            'current_sensors': torch.FloatTensor(self.actual_sensors[idx]),
            'pinn_predictions': torch.FloatTensor(self.pinn_predictions[idx]),
            'sensor_coords': coords_for_sample,  # NEW!
            'wind': torch.FloatTensor(self.wind[idx]),
            'diffusion': torch.FloatTensor(self.diffusion[idx]),
            'temporal': torch.FloatTensor(self.temporal[idx]),
            'target': torch.FloatTensor(self.actual_sensors[idx]),
            'valid_mask': torch.BoolTensor(self.valid_mask[idx])
        }


# ═══════════════════════════════════════════════════════════════════
# 3. LOSS & TRAINING (updated for new model signature)
# ═══════════════════════════════════════════════════════════════════

def correction_loss(pred, target, corrections, valid_mask, lambda_correction=0.001):
    valid_pred = pred[valid_mask]
    valid_target = target[valid_mask]

    if valid_pred.numel() > 0:
        mse_loss = nn.functional.mse_loss(valid_pred, valid_target)
    else:
        mse_loss = torch.tensor(0.0, device=pred.device)

    correction_reg = lambda_correction * (corrections ** 2).mean()
    total = mse_loss + correction_reg

    return total, {
        'mse': mse_loss.item(),
        'correction_reg': correction_reg.item(),
        'n_valid': valid_pred.numel()
    }


def train_epoch(model, dataloader, optimizer, device, lambda_correction):
    model.train()
    total_loss = 0
    total_valid = 0

    for batch in dataloader:
        current_sensors = batch['current_sensors'].to(device)
        pinn_predictions = batch['pinn_predictions'].to(device)
        sensor_coords = batch['sensor_coords'].to(device)  # NEW!
        wind = batch['wind'].to(device)
        diffusion = batch['diffusion'].to(device)
        temporal = batch['temporal'].to(device)
        target = batch['target'].to(device)
        valid_mask = batch['valid_mask'].to(device)

        pred, corrections = model(current_sensors, pinn_predictions, sensor_coords,
                                  wind, diffusion, temporal)
        loss, loss_dict = correction_loss(pred, target, corrections, valid_mask, lambda_correction)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * loss_dict['n_valid']
        total_valid += loss_dict['n_valid']

    avg_loss = total_loss / max(total_valid, 1)
    return avg_loss


def evaluate_sensor(model, dataloader, device, lambda_correction, sensor_idx):
    """
    Evaluate performance on a SPECIFIC sensor
    """
    model.eval()

    all_pinn_preds = []
    all_nn2_preds = []
    all_actual = []
    all_masks = []

    with torch.no_grad():
        for batch in dataloader:
            current_sensors = batch['current_sensors'].to(device)
            pinn_predictions = batch['pinn_predictions'].to(device)
            sensor_coords = batch['sensor_coords'].to(device)  # NEW!
            wind = batch['wind'].to(device)
            diffusion = batch['diffusion'].to(device)
            temporal = batch['temporal'].to(device)
            target = batch['target'].to(device)
            valid_mask = batch['valid_mask'].to(device)

            pred, corrections = model(current_sensors, pinn_predictions, sensor_coords,
                                     wind, diffusion, temporal)

            all_pinn_preds.append(pinn_predictions[:, sensor_idx].cpu())
            all_nn2_preds.append(pred[:, sensor_idx].cpu())
            all_actual.append(target[:, sensor_idx].cpu())
            all_masks.append(valid_mask[:, sensor_idx].cpu())

    # Concatenate
    pinn_preds = torch.cat(all_pinn_preds, dim=0)
    nn2_preds = torch.cat(all_nn2_preds, dim=0)
    actual = torch.cat(all_actual, dim=0)
    masks = torch.cat(all_masks, dim=0)

    # Filter to valid
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
        'improvement': improvement,
        'n_valid': valid_pinn.numel()
    }


# ═══════════════════════════════════════════════════════════════════
# 4. LEAVE-ONE-SENSOR-OUT CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════

def leave_one_sensor_out_cv():
    print("\n" + "="*70)
    print("🔬 LEAVE-ONE-SENSOR-OUT CROSS-VALIDATION")
    print("   WITH SPATIAL COORDINATES")
    print("="*70)
    print("\nPurpose: Test if NN2 can generalize to unseen sensor locations")
    print("Method:  Train on 8 sensors, test on the 9th (held-out) sensor")
    print("Feature: Model now knows WHERE each sensor is located\n")

    Path(CONFIG['save_dir']).mkdir(exist_ok=True, parents=True)
    print(f"Saving models to: {CONFIG['save_dir']}")

    results_all_sensors = {}

    # Loop through each sensor as held-out
    for held_out_idx in range(9):

        print("\n" + "="*70)
        print(f"🎯 FOLD {held_out_idx + 1}/9: Holding out sensor #{held_out_idx}")
        print("="*70)

        # Load dataset with this sensor held out
        dataset = BenzeneDataset(
            data_path=f"{CONFIG['data_dir']}sensors_final.csv",
            source_dir=f"{CONFIG['data_dir']}data_nonzero/",
            pinn_path=CONFIG['pinn_file'],
            sensor_coords_path=CONFIG['sensor_coords_file'],  # NEW!
            held_out_sensor_idx=held_out_idx,
            fit_scalers=True
        )

        # Split remaining data into train/val
        n_total = len(dataset)
        n_train = int(0.85 * n_total)
        n_val = n_total - n_train

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        full_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)

        # Create model
        model = NN2_CorrectionNetwork(n_sensors=9).to(CONFIG['device'])
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # Training loop
        print(f"\n📚 Training on 8 sensors (excluding sensor #{held_out_idx})...")
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(CONFIG['epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, CONFIG['device'], CONFIG['lambda_correction'])

            # Evaluate on validation set
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    current_sensors = batch['current_sensors'].to(CONFIG['device'])
                    pinn_predictions = batch['pinn_predictions'].to(CONFIG['device'])
                    sensor_coords = batch['sensor_coords'].to(CONFIG['device'])  # NEW!
                    wind = batch['wind'].to(CONFIG['device'])
                    diffusion = batch['diffusion'].to(CONFIG['device'])
                    temporal = batch['temporal'].to(CONFIG['device'])
                    target = batch['target'].to(CONFIG['device'])
                    valid_mask = batch['valid_mask'].to(CONFIG['device'])

                    pred, corrections = model(current_sensors, pinn_predictions, sensor_coords,
                                            wind, diffusion, temporal)
                    loss, _ = correction_loss(pred, target, corrections, valid_mask, CONFIG['lambda_correction'])
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{CONFIG['epochs']}: Train={train_loss:.4f}, Val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

        # Save the best model for this fold
        if best_model_state is not None:
            model_save_path = Path(CONFIG['save_dir']) / f"model_fold_{held_out_idx}_spatial.pth"
            try:
                torch.save({
                    'model_state_dict': best_model_state,
                    'scalers': dataset.scalers,
                    'sensor_coords': dataset.sensor_coords,
                    'held_out_idx': held_out_idx
                }, model_save_path)
                if os.path.exists(model_save_path):
                    print(f"  ✓ Successfully saved model for fold {held_out_idx} to {model_save_path}")
                else:
                    print(f"  ❌ Failed to save model")
            except Exception as e:
                print(f"  ❌ Error saving model: {e}")
        else:
            print(f"  ⚠️ No best model state found, skipping save.")

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Evaluate on HELD-OUT sensor
        print(f"\n🔍 Testing on held-out sensor #{held_out_idx}...")
        held_out_results = evaluate_sensor(model, full_loader, CONFIG['device'], CONFIG['lambda_correction'], held_out_idx)

        # Also evaluate on training sensors for comparison
        train_sensor_results = []
        for train_idx in range(9):
            if train_idx != held_out_idx:
                train_results = evaluate_sensor(model, full_loader, CONFIG['device'], CONFIG['lambda_correction'], train_idx)
                train_sensor_results.append(train_results)

        # Average performance on training sensors
        avg_train_mae = np.mean([r['nn2_mae'] for r in train_sensor_results])
        avg_train_improvement = np.mean([r['improvement'] for r in train_sensor_results])

        # Store results
        results_all_sensors[held_out_idx] = {
            'held_out': held_out_results,
            'training_sensors_avg': {
                'nn2_mae': avg_train_mae,
                'improvement': avg_train_improvement
            }
        }

        # Print results for this fold
        print(f"\n{'='*70}")
        print(f"RESULTS FOR SENSOR #{held_out_idx}")
        print(f"{'='*70}")
        print(f"\n{'Metric':<30} {'Training Sensors':<20} {'Held-Out Sensor':<20}")
        print(f"{'─'*70}")
        print(f"{'NN2 MAE (ppb)':<30} {avg_train_mae:<20.4f} {held_out_results['nn2_mae']:<20.4f}")
        print(f"{'Improvement (%)':<30} {avg_train_improvement:<20.1f} {held_out_results['improvement']:<20.1f}")
        print(f"{'Valid samples':<30} {'-':<20} {held_out_results['n_valid']:<20d}")

        # Spatial generalization gap
        generalization_gap = held_out_results['nn2_mae'] - avg_train_mae
        generalization_pct = (generalization_gap / avg_train_mae * 100) if avg_train_mae > 0 else 0

        print(f"\n{'Spatial Generalization Gap:':<30}")
        print(f"  Absolute: {generalization_gap:+.4f} ppb")
        print(f"  Relative: {generalization_pct:+.1f}%")

        if generalization_gap < 0.05:
            print(f"  ✅ Excellent spatial generalization!")
        elif generalization_gap < 0.10:
            print(f"  ✓ Good spatial generalization")
        elif generalization_gap < 0.20:
            print(f"  ⚠️  Moderate spatial generalization")
        else:
            print(f"  ❌ Poor spatial generalization")

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY ACROSS ALL SENSORS
    # ═══════════════════════════════════════════════════════════

    print("\n" + "="*70)
    print("📊 FINAL SUMMARY: SPATIAL GENERALIZATION ANALYSIS")
    print("="*70)

    all_held_out_mae = [results_all_sensors[i]['held_out']['nn2_mae'] for i in range(9)]
    all_held_out_improvement = [results_all_sensors[i]['held_out']['improvement'] for i in range(9)]
    all_train_mae = [results_all_sensors[i]['training_sensors_avg']['nn2_mae'] for i in range(9)]

    print(f"\n{'Sensor #':<12} {'Train MAE':<15} {'Held-Out MAE':<15} {'Gap':<15} {'Improvement':<15}")
    print(f"{'─'*72}")
    for i in range(9):
        train_mae = all_train_mae[i]
        held_mae = all_held_out_mae[i]
        gap = held_mae - train_mae
        improvement = all_held_out_improvement[i]
        print(f"{i:<12} {train_mae:<15.4f} {held_mae:<15.4f} {gap:+15.4f} {improvement:<15.1f}%")

    print(f"{'─'*72}")
    print(f"{'AVERAGE':<12} {np.mean(all_train_mae):<15.4f} {np.mean(all_held_out_mae):<15.4f} {np.mean(all_held_out_mae) - np.mean(all_train_mae):+15.4f} {np.mean(all_held_out_improvement):<15.1f}%")
    print(f"{'STD DEV':<12} {np.std(all_train_mae):<15.4f} {np.std(all_held_out_mae):<15.4f}")

    avg_gap = np.mean(all_held_out_mae) - np.mean(all_train_mae)
    avg_gap_pct = (avg_gap / np.mean(all_train_mae) * 100)

    print(f"\n{'='*70}")
    print(f"🎯 KEY FINDINGS (with spatial coordinates):")
    print(f"{'='*70}")
    print(f"\nAverage performance on TRAINING sensors:  {np.mean(all_train_mae):.4f} ppb MAE")
    print(f"Average performance on HELD-OUT sensors:  {np.mean(all_held_out_mae):.4f} ppb MAE")
    print(f"\nSpatial generalization gap: {avg_gap:+.4f} ppb ({avg_gap_pct:+.1f}%)")

    if avg_gap < 0.05:
        print(f"\n✅ EXCELLENT! Model generalizes very well to new sensor locations!")
        print(f"   → Can confidently deploy to new locations within 10 km")
    elif avg_gap < 0.10:
        print(f"\n✓ GOOD! Model generalizes reasonably to new sensor locations")
        print(f"   → Should work at new locations with ~{avg_gap:.2f} ppb additional error")
    elif avg_gap < 0.20:
        print(f"\n⚠️  MODERATE! Model has some spatial overfitting")
        print(f"   → May work at new locations but validate first")
    else:
        print(f"\n❌ POOR! Model is spatially overfit to training sensors")
        print(f"   → Not recommended for new locations without retraining")

    # Save results
    with open(f"{CONFIG['save_dir']}/leave_one_out_results_spatial.json", 'w') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                      for kk, vv in v.items() if kk != 'held_out'}
                  for k, v in results_all_sensors.items()}, f, indent=2)

    print(f"\n📁 Results saved to: {CONFIG['save_dir']}/leave_one_out_results_spatial.json")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    leave_one_sensor_out_cv()

    # ═══════════════════════════════════════════════════════════════════
    # FINAL RUN: Train on entire dataset and save master model
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🚀 FINAL RUN: Training Master Model on Entire Dataset")
    print("   WITH SPATIAL COORDINATES")
    print("="*70)

    # Load the entire dataset (no held-out sensors)
    final_dataset = BenzeneDataset(
        data_path=f"{CONFIG['data_dir']}sensors_final.csv",
        source_dir=f"{CONFIG['data_dir']}data_nonzero/",
        pinn_path=CONFIG['pinn_file'],
        sensor_coords_path=CONFIG['sensor_coords_file'],  # NEW!
        held_out_sensor_idx=None,  # Use all sensors
        fit_scalers=True
    )

    final_loader = DataLoader(final_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # Create a new model for the final run
    master_model = NN2_CorrectionNetwork(n_sensors=9).to(CONFIG['device'])
    master_optimizer = optim.AdamW(master_model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
    master_scheduler = optim.lr_scheduler.ReduceLROnPlateau(master_optimizer, mode='min', factor=0.5, patience=5)

    print(f"\n📚 Training master model for {CONFIG['epochs']} epochs...")
    final_best_loss = float('inf')
    final_best_model_state = None

    for epoch in range(CONFIG['epochs']):
        train_loss = train_epoch(master_model, final_loader, master_optimizer, CONFIG['device'], CONFIG['lambda_correction'])

        if train_loss < final_best_loss:
            final_best_loss = train_loss
            final_best_model_state = master_model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{CONFIG['epochs']}: Train Loss={train_loss:.4f}")

    # Save the master model with all necessary info
    if final_best_model_state is not None:
        master_model_path = Path(CONFIG['save_dir']) / "nn2_master_model_spatial.pth"
        try:
            torch.save({
                'model_state_dict': final_best_model_state,
                'scalers': final_dataset.scalers,
                'sensor_coords': final_dataset.sensor_coords,
                'config': CONFIG
            }, master_model_path)
            if os.path.exists(master_model_path):
                print(f"\n✓ Successfully saved master model to: {master_model_path}")
            else:
                print(f"\n❌ Failed to save master model")
        except Exception as e:
            print(f"\n❌ Error saving master model: {e}")
    else:
        print("\n⚠️ Could not save master model")

    print(f"\n{'='*70}")
    print("✅ FINAL RUN COMPLETE!")
    print("   → nn2_master_model_spatial.pth ready for full domain reconstruction")
    print("   → Model now has spatial awareness of sensor locations!")
    print(f"{'='*70}")

    print(f"\nFinal check: Contents of {CONFIG['save_dir']}:")
    print(os.listdir(CONFIG['save_dir']))