# Complete Pipeline Documentation
## Benzene Dispersion Prediction Pipeline - Full Technical Specification

**Date Created**: 2024
**Purpose**: Complete documentation of every aspect of the benzene dispersion prediction pipeline, from training data generation to real-time prediction.

---

## Table of Contents

1. [Overview](#overview)
2. [Training Data Generation](#training-data-generation)
3. [PINN Model Details](#pinn-model-details)
4. [Real-Time Pipeline](#real-time-pipeline)
5. [Validation Process](#validation-process)
6. [Key Constants and Parameters](#key-constants-and-parameters)
7. [File Structure](#file-structure)
8. [Critical Fixes Applied](#critical-fixes-applied)
9. [ARCHIVED - NN2 Components](#archived---nn2-components)

---

## Overview

The pipeline consists of:
1. **PINN (Physics-Informed Neural Network)**: Solves the advection-diffusion equation for benzene dispersion
2. **Kalman Filter**: Combines PINN physics predictions with real-time sensor measurements for improved 3-hour forecasts with uncertainty quantification

**Pipeline Flow**:
```
Meteorological Data → PINN (at sources) → Superimpose → Convert to ppb → Kalman Filter (with sensor data) → Final Predictions
```

---

## Training Data Generation

### Script Location
`/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata/regenerate_training_data_correct_pinn.py`

### Process Overview

1. **Load PINN Model**
2. **Load Facility Data** (20 facilities)
3. **For each facility, for each timestamp:**
   - Compute PINN at all 9 sensor locations
   - Use simulation time t=3.0 hours
   - Convert to ppb
   - Shift timestamp forward by 3 hours
4. **Superimpose** contributions from all facilities
5. **Output**: `total_concentrations.csv`

### Detailed Step-by-Step

#### Step 1: Load PINN Model

```python
PINN_MODEL_PATH = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"

pinn = ParametricADEPINN()
checkpoint = torch.load(PINN_MODEL_PATH, map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']
filtered_state_dict = {k: v for k, v in state_dict.items() 
                       if not k.endswith('_min') and not k.endswith('_max')}
pinn.load_state_dict(filtered_state_dict, strict=False)

# Override normalization ranges (CRITICAL - must match these exact values)
pinn.x_min = torch.tensor(0.0)
pinn.x_max = torch.tensor(30000.0)
pinn.y_min = torch.tensor(0.0)
pinn.y_max = torch.tensor(30000.0)
pinn.t_min = torch.tensor(0.0)
pinn.t_max = torch.tensor(8760.0)  # 1 year in hours
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
```

#### Step 2: Load Facility Data

```python
SYNCED_DIR = Path('/Users/neevpratap/Desktop/benzenepipelinev2/realtime/simpletesting/nn2trainingdata')
facility_files = sorted(SYNCED_DIR.glob('*_synced_training_data.csv'))
facility_files = [f for f in facility_files if 'summary' not in f.name]

# Expected: 20 facility files
# Format: Each file has columns: [t, source_x_cartesian, source_y_cartesian, 
#          source_diameter, Q_total, wind_u, wind_v, D]
```

#### Step 3: Process Each Facility

**CRITICAL**: Process each facility file **separately**, then superimpose.

```python
# Storage for superimposed predictions
# Structure: timestamp -> sensor_id -> total_concentration
superimposed_predictions = {}

# Process each facility
for facility_file in facility_files:
    facility_name = facility_file.stem.replace('_synced_training_data', '')
    df = pd.read_csv(facility_file)
    
    # Predict at sensors for this facility
    facility_predictions = predict_pinn_at_sensors(pinn, df)
    
    # Superimpose
    for timestamp, sensor_concs in facility_predictions.items():
        if timestamp not in superimposed_predictions:
            superimposed_predictions[timestamp] = {s: 0.0 for s in SENSORS.keys()}
        
        for sensor_id, conc in sensor_concs.items():
            superimposed_predictions[timestamp][sensor_id] += conc
```

#### Step 4: PINN Prediction at Sensors (Per Facility)

**CRITICAL METHOD** - This is the exact method used:

```python
def predict_pinn_at_sensors(pinn, facility_data):
    """
    Predict PINN concentrations at all sensor locations for a single facility
    
    FIXED: Uses simulation time t=3.0 hours (not absolute calendar time).
    Each scenario starts at t=0, predicts at t=3 hours for 3-hour forecast.
    """
    FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast
    predictions = {}
    
    with torch.no_grad():
        for idx, row in facility_data.iterrows():
            input_timestamp = pd.to_datetime(row['t'])  # Time when met data was collected
            
            # Use simulation time t=3.0 hours (NOT absolute calendar time)
            t_hours_forecast = FORECAST_T_HOURS
            
            # Facility parameters
            cx = row['source_x_cartesian']
            cy = row['source_y_cartesian']
            d = row['source_diameter']
            Q = row['Q_total']
            u = row['wind_u']
            v = row['wind_v']
            kappa = row['D']
            
            # Predict at each sensor
            sensor_concentrations = {}
            for sensor_id, (sx, sy) in SENSORS.items():
                # Run PINN with individual arguments (need 2D tensors)
                phi_raw = pinn(
                    torch.tensor([[sx]], dtype=torch.float32),
                    torch.tensor([[sy]], dtype=torch.float32),
                    torch.tensor([[t_hours_forecast]], dtype=torch.float32),  # FORECAST TIME
                    torch.tensor([[cx]], dtype=torch.float32),
                    torch.tensor([[cy]], dtype=torch.float32),
                    torch.tensor([[u]], dtype=torch.float32),
                    torch.tensor([[v]], dtype=torch.float32),
                    torch.tensor([[d]], dtype=torch.float32),
                    torch.tensor([[kappa]], dtype=torch.float32),
                    torch.tensor([[Q]], dtype=torch.float32),
                    normalize=True  # CRITICAL: Must use normalize=True
                )
                
                # Convert to ppb
                concentration_ppb = phi_raw.item() * UNIT_CONVERSION_FACTOR
                sensor_concentrations[sensor_id] = concentration_ppb
            
            # CRITICAL: Shift timestamp forward by 3 hours
            # Predictions made at time t (using met data from t) are labeled as t+3
            # This aligns PINN predictions with actual sensor readings at t+3
            forecast_timestamp = input_timestamp + pd.Timedelta(hours=3)
            predictions[forecast_timestamp] = sensor_concentrations
    
    return predictions
```

#### Step 5: Output Format

```python
# Convert to DataFrame
rows = []
for timestamp in sorted(superimposed_predictions.keys()):
    row = {'timestamp': timestamp}
    for sensor_id in SENSORS.keys():
        row[f'sensor_{sensor_id}'] = superimposed_predictions[timestamp][sensor_id]
    rows.append(row)

output_df = pd.DataFrame(rows)
output_df.to_csv('total_concentrations.csv', index=False)
```

**Output File**: `total_concentrations.csv`
- Columns: `timestamp, sensor_482010026, sensor_482010057, sensor_482010069, sensor_482010617, sensor_482010803, sensor_482011015, sensor_482011035, sensor_482011039, sensor_482016000`
- Timestamps: Shifted forward by 3 hours (predictions made at t are labeled as t+3)

---

## PINN Model Details

### Model Architecture

**Class**: `ParametricADEPINN` (from `pinn.py`)

**Key Features**:
- Physics-informed neural network solving advection-diffusion equation
- Tanh activations
- Normalization layers for all inputs
- Temporal multiplier (soft constraint for initial conditions)

### Input Parameters

1. **x, y**: Observation location (Cartesian coordinates, meters)
   - Range: [0, 30000] meters
   - Domain: 30km × 30km

2. **t**: Time (hours)
   - **CRITICAL**: Use simulation time t=3.0 hours (NOT absolute calendar time)
   - Each scenario resets to t=0, predicts at t=3 hours
   - Normalization range: [0, 8760] hours (1 year)

3. **cx, cy**: Source location (Cartesian coordinates, meters)
   - Range: [0, 30000] meters

4. **u, v**: Wind velocity components (m/s)
   - Range: [-15, 15] m/s

5. **d**: Source diameter (meters)
   - Range: [0, 200] meters

6. **kappa (D)**: Diffusion coefficient
   - Range: [0, 200]

7. **Q**: Emission rate (kg/s)
   - Range: [0, 0.01] kg/s

### PINN Forward Pass

**Complete Forward Pass** (from `pinn.py`):

```python
def forward(self, x, y, t, cx, cy, u, v, d, kappa, Q, normalize=True):
    """
    Forward pass with optional normalization
    normalize=True: inputs are in physical units (RECOMMENDED)
    normalize=False: inputs are already normalized to [-1,1]
    """
    if normalize:
        # Normalize all inputs to [-1, 1] range
        x_n = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        y_n = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
        t_n = 2 * (t - self.t_min) / (self.t_max - self.t_min) - 1
        cx_n = 2 * (cx - self.cx_min) / (self.cx_max - self.cx_min) - 1
        cy_n = 2 * (cy - self.cy_min) / (self.cy_max - self.cy_min) - 1
        u_n = 2 * (u - self.u_min) / (self.u_max - self.u_min) - 1
        v_n = 2 * (v - self.v_min) / (self.v_max - self.v_min) - 1
        d_n = 2 * (d - self.d_min) / (self.d_max - self.d_min) - 1
        kappa_n = 2 * (kappa - self.kappa_min) / (self.kappa_max - self.kappa_min) - 1
        Q_n = 2 * (Q - self.Q_min) / (self.Q_max - self.Q_min) - 1
    else:
        # Assume inputs are already normalized
        x_n, y_n, t_n, cx_n, cy_n, u_n, v_n, d_n, kappa_n, Q_n = x, y, t, cx, cy, u, v, d, kappa, Q
    
    # Concatenate all normalized features
    features = torch.cat([x_n, y_n, t_n, cx_n, cy_n, u_n, v_n, d_n, kappa_n, Q_n], dim=-1)
    
    # Network forward pass (4 layers, 128 neurons each, Tanh activations)
    phi_raw = self.net(features)
    
    # Temporal multiplier (soft constraint for initial conditions)
    # This ensures phi=0 at t=0 (initial condition) without suppressing predictions at t=3
    t_norm_01 = (t_n + 1) / 2  # Convert from [-1,1] to [0,1]
    RAMP_STEEPNESS = 100.0  # Higher = steeper ramp (faster transition)
    temporal_multiplier = 1.0 - torch.exp(-RAMP_STEEPNESS * t_norm_01)
    
    # At t_norm_01 = 0 (t_min): multiplier = 0 (phi=0) - initial condition
    # At t_norm_01 = 0.05 (5% into time range): multiplier ≈ 0.99 (almost no effect)
    # For t_max=8760, this means full output by t≈438 hours
    # For t=3.0 hours: t_norm_01 ≈ 0.00034, multiplier ≈ 0.033 (small but non-zero)
    
    phi = temporal_multiplier * phi_raw
    
    return phi
```

**Network Architecture**:
```python
self.net = nn.Sequential(
    nn.Linear(10, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 1),
    nn.Softplus()  # Ensures non-negative output
)
```

### Output

- **phi**: Concentration in kg/m² (raw model output)
- **Convert to ppb**: `concentration_ppb = phi * UNIT_CONVERSION_FACTOR`
- **UNIT_CONVERSION_FACTOR**: `313210039.9` (kg/m² to ppb)

---

## ARCHIVED - NN2 Components

> **NOTE**: The NN2 correction network has been archived. All NN2-related content is preserved below for reference.

### NN2 Training Process

#### Training Data Structure

**Input Files**:
1. `total_concentrations.csv`: PINN predictions at sensor locations (timestamp shifted +3 hours)
2. `sensors_final_synced.csv`: Actual sensor readings
3. Facility meteo data: For wind, diffusion, temporal features

#### Data Alignment

**CRITICAL**: All data must be aligned at the same timestamps (t+3):
- PINN predictions: At timestamp t+3 (forecasts made at t, labeled as t+3)
- Sensor readings: At timestamp t+3 (target values)
- Current sensors: At timestamp t+3 (same as target - this is the training structure)

#### NN2 Inputs

1. **pinn_predictions**: PINN predictions at 9 sensors (scaled, non-zero values only)
2. **current_sensors**: Actual sensor readings at 9 sensors (scaled, non-zero values only)
   - **NOTE**: In training, current_sensors = target = actual readings at t+3
3. **sensor_coords**: Spatial coordinates of 9 sensors (normalized)
4. **wind**: Average wind (u, v) across all facilities (scaled)
5. **diffusion**: Average diffusion coefficient D across all facilities (scaled)
6. **temporal**: Temporal features (hour, day, month, weekend) (not scaled)

#### Scaling Process

**CRITICAL**: Scalers are fitted on **non-zero values only**:

```python
# Fit scalers on non-zero values only
# For each sensor, fit on non-zero values
for i in range(n_sensors):
    mask = actual_sensors[:, i] != 0
    if mask.any():
        valid_sensors = actual_sensors[mask, i]
        scalers['sensors'].fit(valid_sensors.reshape(-1, 1))
    
    mask = pinn_predictions[:, i] != 0
    if mask.any():
        valid_pinn = pinn_predictions[mask, i]
        scalers['pinn'].fit(valid_pinn.reshape(-1, 1))

# Transform (only non-zero values)
for i in range(n_sensors):
    mask = actual_sensors[:, i] != 0
    if mask.any():
        actual_sensors[mask, i] = scalers['sensors'].transform(
            actual_sensors[mask, i].reshape(-1, 1)
        ).flatten()
    
    mask = pinn_predictions[:, i] != 0
    if mask.any():
        pinn_predictions[mask, i] = scalers['pinn'].transform(
            pinn_predictions[mask, i].reshape(-1, 1)
        ).flatten()
```

**Scaler Types**:
- `sensors`: StandardScaler (fits on non-zero sensor readings)
- `pinn`: StandardScaler (fits on non-zero PINN predictions)
- `wind`: StandardScaler (fits on wind components)
- `diffusion`: StandardScaler (fits on diffusion coefficients)
- `coords`: StandardScaler (fits on sensor coordinates)

#### NN2 Model Architecture

```python
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
        # Total: 9*4 + 2 + 1 + 6 = 45 features
        
        self.correction_network = nn.Sequential(
            nn.Linear(45, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_sensors)
        )
    
    def forward(self, current_sensors, pinn_predictions, sensor_coords, wind, diffusion, temporal):
        # Flatten coords: [batch, 9, 2] -> [batch, 18]
        batch_size = current_sensors.shape[0]
        coords_flat = sensor_coords.reshape(batch_size, -1)
        
        # Concatenate all features
        features = torch.cat([
            pinn_predictions,      # [batch, 9]
            current_sensors,       # [batch, 9]
            coords_flat,           # [batch, 18]
            wind,                  # [batch, 2]
            diffusion,             # [batch, 1]
            temporal               # [batch, 6]
        ], dim=-1)  # Total: 45
        
        corrections = self.correction_network(features)
        corrected_predictions = pinn_predictions + corrections
        return corrected_predictions, corrections
```

#### Training Process

**Loss Function**:
```python
def correction_loss(pred, target, corrections, valid_mask, lambda_correction=0.001):
    valid_pred = pred[valid_mask]
    valid_target = target[valid_mask]
    
    mse_loss = nn.functional.mse_loss(valid_pred, valid_target)
    correction_reg = lambda_correction * (corrections ** 2).mean()
    
    total_loss = mse_loss + correction_reg
    return total_loss, {'mse': mse_loss, 'reg': correction_reg, 'n_valid': valid_pred.numel()}
```

**Training Configuration**:
- Batch size: 32
- Learning rate: 0.001
- Optimizer: AdamW
- Weight decay: 1e-5
- Epochs: 100
- Lambda correction: 0.001

**Optimizer**: AdamW with weight decay 1e-5
**Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
**Gradient Clipping**: max_norm=1.0

#### Model Outputs

**Saved Files**:
- Model: `nn2_master_model_spatial-3.pth`
- Scalers: `nn2_master_scalers-2.pkl`
- Results: `leave_one_out_results_spatial-3.json`

### NN2 Correction (At Sensor Locations)

**CRITICAL METHOD** - Must compute PINN at sensors directly (not interpolate):

```python
def _apply_nn2_correction(self, x, y, pinn_pred, facility_params, forecast_time):
    """
    Apply NN2 correction efficiently using actual 9 sensor locations
    
    1. Compute PINN at sensor locations directly (EXACT same method as training)
    2. Apply NN2 correction to get corrected values at sensors
    3. Calculate correction field (NN2 - PINN) at sensors
    4. Interpolate correction field across entire domain using RBF
    5. Add interpolated corrections to PINN field
    """
    
    # Define 9 sensor locations (EXACT values from training data generation)
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
    
    # Step 1: Compute PINN at sensor locations using EXACT same method as training data
    # Instead of interpolating, compute directly at sensor locations for each facility
    sensor_pinn = np.zeros(len(SENSOR_COORDS))
    FORECAST_T_HOURS = 3.0  # Simulation time (matches training data)
    UNIT_CONVERSION_FACTOR = 313210039.9  # kg/m^2 to ppb (matches training data)
    
    for facility_name, params in facility_params.items():
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
    
    # Step 2: Apply NN2 correction at sensor locations
    # ... (rest of NN2 correction process)
```

#### Scaling for NN2 (CRITICAL - Zero Handling)

```python
# Scale inputs - handle zeros the same way as training (mask before scaling)
pinn_nonzero_mask = sensor_pinn != 0.0
sensors_nonzero_mask = current_sensors != 0.0

# Scale PINN predictions (only non-zero values)
p_s = np.zeros_like(sensor_pinn)
if pinn_nonzero_mask.any():
    p_s[pinn_nonzero_mask] = self.scalers['pinn'].transform(
        sensor_pinn[pinn_nonzero_mask].reshape(-1, 1)
    ).flatten()
p_s = p_s.reshape(1, -1)

# Scale current sensors (only non-zero values)
s_s = np.zeros_like(current_sensors)
if sensors_nonzero_mask.any():
    s_s[sensors_nonzero_mask] = self.scalers['sensors'].transform(
        current_sensors[sensors_nonzero_mask].reshape(-1, 1)
    ).flatten()
s_s = s_s.reshape(1, -1)
```

#### Inverse Transform (CRITICAL - Zero Handling)

```python
# Inverse transform - handle zeros the same way as training
corrected_scaled_np = corrected_scaled.cpu().numpy().flatten()
sensor_corrected = np.zeros_like(corrected_scaled_np)
if sensors_nonzero_mask.any():
    sensor_corrected[sensors_nonzero_mask] = self.scalers['sensors'].inverse_transform(
        corrected_scaled_np[sensors_nonzero_mask].reshape(-1, 1)
    ).flatten()
# Values that are truly zero in scaled space remain zero
```

### Complete NN2 Correction Function (Real-Time)

```python
def apply_nn2_correction(nn2, scalers, sensor_coords_spatial, pinn_values, meteo_data, timestamp, current_sensor_readings):
    # Prepare inputs
    sensor_ids_sorted = sorted(SENSORS.keys())
    pinn_array = np.array([pinn_values[sid] for sid in sensor_ids_sorted])
    current_sensors = np.array([current_sensor_readings.get(sid, 0.0) for sid in sensor_ids_sorted])
    
    # Meteorology
    avg_u = meteo_data['wind_u'].mean()
    avg_v = meteo_data['wind_v'].mean()
    avg_D = meteo_data['D'].mean()
    
    # Temporal features
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    month = timestamp.month
    is_weekend = 1.0 if day_of_week >= 5 else 0.0
    
    temporal_vals = np.array([[
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day_of_week / 7),
        np.cos(2 * np.pi * day_of_week / 7),
        is_weekend,
        month / 12.0
    ]])
    
    # Scale inputs (handle zeros)
    pinn_nonzero_mask = pinn_array != 0.0
    sensors_nonzero_mask = current_sensors != 0.0
    
    p_s = np.zeros_like(pinn_array)
    if pinn_nonzero_mask.any():
        p_s[pinn_nonzero_mask] = scalers['pinn'].transform(
            pinn_array[pinn_nonzero_mask].reshape(-1, 1)
        ).flatten()
    p_s = p_s.reshape(1, -1)
    
    s_s = np.zeros_like(current_sensors)
    if sensors_nonzero_mask.any():
        s_s[sensors_nonzero_mask] = scalers['sensors'].transform(
            current_sensors[sensors_nonzero_mask].reshape(-1, 1)
        ).flatten()
    s_s = s_s.reshape(1, -1)
    
    w_s = scalers['wind'].transform(np.array([[avg_u, avg_v]]))
    d_s = scalers['diffusion'].transform(np.array([[avg_D]]))
    c_s = scalers['coords'].transform(sensor_coords_spatial)
    
    # Convert to tensors
    p_tensor = torch.tensor(p_s, dtype=torch.float32)
    s_tensor = torch.tensor(s_s, dtype=torch.float32)
    c_tensor = torch.tensor(c_s, dtype=torch.float32).unsqueeze(0)
    w_tensor = torch.tensor(w_s, dtype=torch.float32)
    d_tensor = torch.tensor(d_s, dtype=torch.float32)
    t_tensor = torch.tensor(temporal_vals, dtype=torch.float32)
    
    # Run NN2
    with torch.no_grad():
        corrected_scaled, _ = nn2(s_tensor, p_tensor, c_tensor, w_tensor, d_tensor, t_tensor)
    
    # Inverse transform (handle zeros)
    corrected_scaled_np = corrected_scaled.cpu().numpy().flatten()
    nn2_corrected = np.zeros_like(corrected_scaled_np)
    if sensors_nonzero_mask.any():
        nn2_corrected[sensors_nonzero_mask] = scalers['sensors'].inverse_transform(
            corrected_scaled_np[sensors_nonzero_mask].reshape(-1, 1)
        ).flatten()
    
    # Return as dict
    nn2_values = {sid: nn2_corrected[i] for i, sid in enumerate(sensor_ids_sorted)}
    return nn2_values
```

### RBF Interpolation Details

**Method**: Radial Basis Function (RBF) interpolation using `scipy.interpolate.Rbf`

```python
from scipy.interpolate import Rbf

# Interpolate correction field from sensors to full domain
correction_rbf = Rbf(
    SENSOR_COORDS[:, 0],      # x coordinates of sensors
    SENSOR_COORDS[:, 1],      # y coordinates of sensors
    sensor_corrections,        # Correction values at sensors
    function='multiquadric',   # RBF kernel function
    smooth=0.1                 # Smoothing parameter
)

# Evaluate at all grid points
correction_field = correction_rbf(x, y)
```

### Distance-Based Confidence Weighting

**Method**: Weight corrections based on distance from nearest sensor

```python
# Calculate distance from each grid point to nearest sensor
distances_to_sensors = np.zeros(len(x))
for i in range(len(x)):
    dists = np.sqrt((SENSOR_COORDS[:, 0] - x[i])**2 + (SENSOR_COORDS[:, 1] - y[i])**2)
    distances_to_sensors[i] = dists.min()

# Define confidence decay:
# - Full trust within 2km
# - Linear decay 2-5km
# - Zero trust beyond 5km
confidence = np.ones(len(x))
confidence[distances_to_sensors > 2000] = 1.0 - (distances_to_sensors[distances_to_sensors > 2000] - 2000) / 3000
confidence[distances_to_sensors > 5000] = 0.0
confidence = np.clip(confidence, 0.0, 1.0)

# Apply confidence-weighted correction
weighted_correction = correction_field * confidence
corrected_field = pinn_pred + weighted_correction
```

### NN2-Related Fixes

#### Fix 3: Zero-Value Handling

**Problem**: Scalers were fitted on non-zero values only, but validation was transforming all values (including zeros).

**Solution**: Mask zeros before scaling and inverse scaling.

**Code Change**:
```python
# Scale only non-zero values
pinn_nonzero_mask = pinn_array != 0.0
if pinn_nonzero_mask.any():
    p_s[pinn_nonzero_mask] = scalers['pinn'].transform(
        pinn_array[pinn_nonzero_mask].reshape(-1, 1)
    ).flatten()

# Inverse transform only non-zero values
if sensors_nonzero_mask.any():
    nn2_corrected[sensors_nonzero_mask] = scalers['sensors'].inverse_transform(
        corrected_scaled_np[sensors_nonzero_mask].reshape(-1, 1)
    ).flatten()
```

#### Fix 4: Sensor Location Computation

**Problem**: Pipeline was interpolating PINN field to sensor locations, but training data computed PINN directly at sensors.

**Solution**: Compute PINN directly at sensor locations for each facility, then superimpose (exactly like training data generation).

**Code Change**:
```python
# OLD (BUGGY): Interpolate
pinn_rbf = Rbf(x, y, pinn_pred, function='multiquadric', smooth=0.1)
sensor_pinn = pinn_rbf(SENSOR_COORDS[:, 0], SENSOR_COORDS[:, 1])

# NEW (FIXED): Compute directly
for facility_name, params in facility_params.items():
    for i, (sx, sy) in enumerate(SENSOR_COORDS):
        phi_raw = self.pinn(...)  # Direct computation
        sensor_pinn[i] += concentration_ppb  # Superimpose
```

#### Fix 5: Current Sensors Input

**Problem**: Validation was using sensor readings from time t (when prediction is made), but training used readings from t+3 (same as target).

**Solution**: Use sensor readings from t+3 (forecast timestamp) as current_sensors input.

**Code Change**:
```python
# Get actual sensor readings from time t+3 (forecast timestamp)
# This matches training structure: current_sensors = actual readings at t+3
current_sensor_readings = {}
for sensor_id in SENSORS.keys():
    actual_col = f'sensor_{sensor_id}'
    if actual_col in row and not pd.isna(row[actual_col]):
        current_sensor_readings[sensor_id] = row[actual_col]  # From forecast_timestamp
```

---

## Real-Time Pipeline

### Script Location
`/Users/neevpratap/Desktop/benzenepipelinev2/realtime/concentration_predictor.py`

### Class: ConcentrationPredictor

### Initialization

```python
class ConcentrationPredictor:
    def __init__(self, grid_resolution: int = 100, use_kalman: bool = True):
        self.grid_resolution = grid_resolution
        self.device = 'cpu'
        
        # Model paths
        self.pinn_path = "/Users/neevpratap/Downloads/pinn_combined_final2.pth"
        
        # Load models
        self.pinn = self._load_pinn()
        
        # Initialize Kalman filter
        self.use_kalman = use_kalman
        if use_kalman:
            param_file = Path("realtime/data/kalman_parameters.json")
            if param_file.exists():
                with open(param_file, 'r') as f:
                    kf_params = json.load(f)
                    kf_params = {k: v for k, v in kf_params.items() 
                                if k in ['process_noise', 'measurement_noise', 
                                        'decay_rate', 'pinn_weight']}
            else:
                kf_params = {
                    'process_noise': 1.0,
                    'measurement_noise': 0.01,
                    'decay_rate': 0.7,
                    'pinn_weight': 0.3
                }
            self.kalman_filter = BenzeneKalmanFilter(**kf_params)
        
        # Create spatial grid
        self._create_grid()
```

### PINN Computation (Per Facility)

**CRITICAL METHOD** - Must match training data generation exactly:

```python
def _compute_pinn_for_facility(
    self, x, y, t,
    source_x, source_y, source_d,
    Q, wind_u, wind_v, D
):
    """
    Compute PINN predictions for single facility across grid
    
    FIXED: Uses simulation time t=3.0 hours (not absolute calendar time).
    """
    # FIX: Use simulation time instead of absolute calendar time
    FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast
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
        UNIT_CONVERSION_FACTOR = 313210039.9  # kg/m^2 to ppb
        concentrations[i:end_idx] = np.maximum(phi.numpy().flatten() * UNIT_CONVERSION_FACTOR, 0.0)
    
    return concentrations
```

### Kalman Filter Correction

The Kalman filter combines PINN physics predictions with real-time sensor measurements to improve forecasts.

**Key Components**:
1. **State Vector**: 9 sensor concentrations [C_sensor_0, C_sensor_1, ..., C_sensor_8]
2. **Process Model**: Combines PINN predictions with exponential decay
3. **Measurement Model**: Direct sensor observations
4. **Uncertainty Quantification**: Provides confidence intervals for forecasts

**Process Flow**:
```python
def _apply_kalman_correction(self, pinn_pred, current_sensor_readings, facility_params, forecast_time):
    """
    Apply Kalman filter correction to PINN predictions.
    
    1. Get PINN predictions at sensor locations
    2. Run Kalman filter forecast (predict + update)
    3. Calculate corrections (Kalman - PINN) at sensors
    4. Interpolate corrections across domain using RBF
    5. Apply distance-based confidence weighting
    6. Add weighted corrections to PINN field
    """
    # Step 1: Get PINN at sensors
    sensor_pinn = self.predict_pinn_at_sensors(forecast_time, facility_params)
    
    # Step 2: Kalman forecast
    kalman_forecast, uncertainty = self.kalman_filter.forecast(
        current_sensors=current_sensor_readings,
        pinn_predictions=sensor_pinn,
        hours_ahead=3,
        return_uncertainty=True
    )
    
    # Step 3: Calculate corrections at sensors
    corrections_at_sensors = kalman_forecast - sensor_pinn
    
    # Step 4: Interpolate corrections to full domain
    correction_field = self._interpolate_corrections(corrections_at_sensors)
    
    # Step 5: Apply confidence weighting (distance-based)
    weighted_correction = correction_field * confidence_weights
    
    # Step 6: Apply to PINN field
    corrected_field = pinn_pred + weighted_correction
    
    return corrected_field
```

**Kalman Filter Parameters** (tuned via grid search):
- `process_noise`: Uncertainty in process model (default: 0.1)
- `measurement_noise`: Uncertainty in sensor measurements (default: 0.5)
- `decay_rate`: Exponential decay rate for concentrations (default: 0.7)
- `pinn_weight`: Influence of PINN predictions in state transition (default: 0.1)

**Real-Time Operation**:
- At time T: Get current EDF sensor readings
- At time T: Get PINN prediction for T+3
- Kalman filter combines both to produce corrected forecast for T+3
- Provides uncertainty bounds (95% confidence intervals)

---

## Validation Process

### Script Locations
- **Kalman Validation**: `/Users/neevpratap/Desktop/benzenepipelinev2/realtime/kalman_validation.py`
- **Kalman Diagnostics**: `/Users/neevpratap/Desktop/benzenepipelinev2/realtime/kalman_diagnostics.py`
- **Extreme Event Testing**: `/Users/neevpratap/Desktop/benzenepipelinev2/realtime/kalman_extreme_event_test.py`

### Process

1. **Load Models**: PINN and Kalman filter (with tuned parameters)
2. **Load Sensor Data**: Ground truth readings
3. **Load Facility Data**: Meteorological data
4. **For each timestamp pair (T, T+3):**
   - Get current sensor readings at time T
   - Get met data from T
   - Compute PINN predictions using met data from T, forecasting for T+3
   - Run Kalman filter forecast (combines current sensors + PINN)
   - Compare Kalman forecast with actual sensor readings at T+3

### Validation Metrics

- **Overall MAE**: Mean Absolute Error across all samples
- **Detection Rate**: Percentage of exceedances (>10 ppb) correctly detected
- **Peak Capture**: Accuracy of peak magnitude predictions (within 20%)
- **False Alarm Rate**: Percentage of false warnings
- **Response Rate**: How quickly filter responds to sudden spikes
- **Extreme Event Analysis**: Separate metrics for >10 ppb, >20 ppb, >50 ppb thresholds

### Key Validation Details

**Timestamp Alignment**:
- Sensor reading at: `forecast_timestamp` (t+3)
- Met data from: `met_data_timestamp = forecast_timestamp - 3 hours` (t)
- PINN prediction: Made at t, forecasting for t+3

**PINN Computation**:
- Use simulation time `t=3.0 hours`
- Process each facility separately
- Superimpose across all facilities

**Kalman Filter**:
- Uses current sensor readings at time T
- Combines with PINN predictions for T+3
- Produces corrected forecast with uncertainty bounds
- Parameters optimized via multi-objective grid search (prioritizes detection rate)

---

## Key Constants and Parameters

### Unit Conversion
```python
UNIT_CONVERSION_FACTOR = 313210039.9  # kg/m^2 to ppb
```

### Simulation Time
```python
FORECAST_T_HOURS = 3.0  # Simulation time for 3-hour forecast
# Each scenario resets to t=0, predicts at t=3 hours
```

### Sensor Coordinates (Cartesian, meters)
```python
SENSORS = {
    '482010026': (13972.62, 19915.57),
    '482010057': (3017.18, 12334.2),
    '482010069': (817.42, 9218.92),
    '482010617': (27049.57, 22045.66),
    '482010803': (8836.35, 15717.2),
    '482011015': (18413.8, 15068.96),
    '482011035': (1159.98, 12272.52),
    '482011039': (13661.93, 5193.24),
    '482016000': (1546.9, 6786.33),
}
```

### PINN Normalization Ranges
```python
x_min, x_max = 0.0, 30000.0
y_min, y_max = 0.0, 30000.0
t_min, t_max = 0.0, 8760.0
cx_min, cx_max = 0.0, 30000.0
cy_min, cy_max = 0.0, 30000.0
u_min, u_max = -15.0, 15.0
v_min, v_max = -15.0, 15.0
d_min, d_max = 0.0, 200.0
kappa_min, kappa_max = 0.0, 200.0
Q_min, Q_max = 0.0, 0.01
```

### Kalman Filter Parameters
```python
# Optimized parameters (from kalman_parameters.json)
process_noise = 0.1      # Process model uncertainty
measurement_noise = 0.5  # Sensor measurement uncertainty
decay_rate = 0.7         # Exponential decay rate
pinn_weight = 0.1        # PINN influence in state transition
```

**Parameter Tuning**:
- Optimized via multi-objective grid search (`kalman_tuning_v2.py`)
- Objective function weights: Detection rate (50%), MAE (20%), False alarms (20%), Peak capture (10%)
- Grid search tested 750 parameter combinations
- Best parameters saved to `realtime/data/kalman_parameters.json`

### Domain Size
- **Spatial**: 30km × 30km (0 to 30000 meters)
- **Grid Resolution**: 100×100 = 10,000 points (configurable)

---

## File Structure

### Training Data Files
```
realtime/simpletesting/nn2trainingdata/
├── total_concentrations.csv          # PINN predictions at sensors (output of training data generation)
├── *_synced_training_data.csv        # 20 facility files with met data
└── regenerate_training_data_correct_pinn.py  # Training data generation script
```

### Model Files
```
/Users/neevpratap/Downloads/
└── pinn_combined_final2.pth          # PINN model
```

### Sensor Data (Ground Truth)
```
/Users/neevpratap/Downloads/
└── sensors_final_synced.csv          # Actual sensor readings (ground truth)
```

**Source**: This file contains actual benzene sensor readings from 2019 EDF Dataset
**File Path**: `/Users/neevpratap/Downloads/sensors_final_synced.csv`

**File Details**:
- **Shape**: 5920 rows × 10 columns
- **Primary column**: `t` (timestamp) - hourly data
- **Sensor columns**: `sensor_482010026`, `sensor_482010057`, `sensor_482010069`, `sensor_482010617`, `sensor_482010803`, `sensor_482011015`, `sensor_482011035`, `sensor_482011039`, `sensor_482016000`
- **Values**: Benzene concentrations in ppb (parts per billion)
- **Timestamp range**: 2019-01-01 13:00:00 to 2019-12-31 22:00:00
- **Total timestamps**: 5920 (hourly, but not every hour - some gaps)
- **Unique timestamps**: 5920

**Data Characteristics**:
- Real measurements from 9 benzene monitoring sensors in Houston area
- Hourly resolution (but not continuous - some hours missing)
- Values typically range from 0.0 to ~200 ppb
- Some sensors have more missing data than others
- Missing values: Represented as NaN (converted to 0.0 in processing)

**Usage in Pipeline**:
1. **Validation**:
   - Used as ground truth to compare against PINN predictions
   - Sensor readings at time t+3 are compared with predictions made at time t (forecasting 3 hours ahead)

**Loading Code**:
```python
SENSOR_DATA_PATH = "/Users/neevpratap/Downloads/sensors_final_synced.csv"
sensor_df = pd.read_csv(SENSOR_DATA_PATH)
if 't' in sensor_df.columns:
    sensor_df = sensor_df.rename(columns={'t': 'timestamp'})
sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
```

### Pipeline Scripts
```
realtime/
├── concentration_predictor.py         # Main prediction class (with Kalman filter)
├── realtime_pipeline.py              # Real-time pipeline runner
├── kalman_filter.py                  # Kalman filter implementation
├── kalman_filter_adaptive.py         # Adaptive Kalman filter (concentration-dependent)
├── kalman_tuning_v2.py               # Multi-objective parameter optimization
├── kalman_validation.py              # Validation script (2019 & 2021)
├── kalman_diagnostics.py             # Diagnostic analysis tool
├── kalman_extreme_event_test.py      # Extreme event testing
└── test_pipeline_2019.py            # Full year test script
```

### Kalman Filter Data Files
```
realtime/data/
├── kalman_parameters.json            # Optimized parameters (from tuning)
├── kalman_parameter_search_v2.csv    # Full grid search results
├── kalman_validation/                # Validation results
│   ├── kalman_validation_2019.csv
│   ├── kalman_validation_2021.csv
│   ├── kalman_summary_2019.json
│   └── kalman_validation_2019.png
├── kalman_diagnostics/               # Diagnostic analysis results
│   ├── diagnostic_results.csv
│   ├── parameter_sensitivity.csv
│   └── diagnostic_summary.json
└── kalman_extreme_events/            # Extreme event test results
    └── extreme_event_results.csv
```

---

## Critical Fixes Applied

### Fix 1: PINN Time Dependency Bug

**Problem**: PINN was using absolute calendar time (hours since 2019-01-01), causing 12x variation for identical conditions across months.

**Solution**: Use simulation time `t=3.0 hours` instead of absolute calendar time. Each scenario resets to t=0, predicts at t=3 hours.

**Files Modified**:
- `regenerate_training_data_correct_pinn.py`
- `kalman_validation.py`
- `concentration_predictor.py`

**Code Change**:
```python
# OLD (BUGGY):
t_hours = (timestamp - t_start).total_seconds() / 3600.0
t_hours_forecast = t_hours + 3.0

# NEW (FIXED):
FORECAST_T_HOURS = 3.0  # Simulation time
t_hours_forecast = FORECAST_T_HOURS
```

### Fix 2: Timestamp Staggering

**Problem**: PINN predictions made at time t were not aligned with sensor readings at t+3.

**Solution**: Shift timestamps forward by 3 hours in training data generation.

**Code Change**:
```python
# Predictions made at time t (using met data from t) are labeled as t+3
forecast_timestamp = input_timestamp + pd.Timedelta(hours=3)
predictions[forecast_timestamp] = sensor_concentrations
```


---

## Complete Process Flow

### Training Data Generation Flow

```
1. Load PINN model
   └─> Set normalization ranges
   └─> Set to eval mode

2. Load facility files (20 facilities)
   └─> Each file: [t, source_x_cartesian, source_y_cartesian, source_diameter, Q_total, wind_u, wind_v, D]

3. For each facility file:
   a. Load facility data
   b. For each timestamp in facility data:
      - For each sensor location:
        * Run PINN with:
          - x, y = sensor coordinates
          - t = 3.0 hours (simulation time)
          - cx, cy = source location
          - u, v = wind components
          - d = source diameter
          - kappa = D (diffusion)
          - Q = emission rate
        * Convert output to ppb: phi * 313210039.9
        * Store concentration
      - Shift timestamp forward by 3 hours
      - Store predictions with shifted timestamp

4. Superimpose across all facilities
   └─> For each timestamp, sum concentrations from all facilities

5. Output total_concentrations.csv
   └─> Columns: timestamp, sensor_482010026, ..., sensor_482016000
```

### Real-Time Pipeline Flow

```
1. Load models (PINN + Kalman Filter)
   └─> Set normalization ranges for PINN
   └─> Load optimized Kalman parameters from kalman_parameters.json

2. Get facility parameters (from CSV generator)
   └─> For each facility: source location, emissions, wind, diffusion

3. Get current sensor readings (at time T)
   └─> Load from EDF sensor data or real-time API
   └─> Handle missing sensors (NaN values)

4. Compute PINN across full domain:
   a. Create spatial grid (100×100 = 10,000 points)
   b. For each facility:
      - Compute PINN at all grid points
      - Use simulation time t=3.0 hours
      - Convert to ppb
   c. Superimpose across all facilities

5. Apply Kalman Filter Correction:
   a. Get PINN predictions at sensor locations
   b. Run Kalman forecast:
      - Predict step: Use PINN + exponential decay model
      - Update step: Incorporate current sensor measurements
   c. Calculate corrections (Kalman - PINN) at sensors
   d. Interpolate corrections to full domain (RBF)
   e. Apply distance-based confidence weighting
   f. Add weighted corrections to PINN field

6. Output predictions
   └─> Save to CSV files (PINN and Kalman-corrected)
   └─> Generate visualizations
   └─> Include uncertainty bounds (95% confidence intervals)
```

---

## Important Notes

1. **Simulation Time**: Always use `t=3.0 hours` (simulation time), NOT absolute calendar time
2. **Timestamp Staggering**: Predictions made at time t are labeled as t+3
3. **Direct Computation**: Compute PINN directly at sensor locations, don't interpolate
4. **Superimposition**: Process each facility separately, then superimpose
5. **Kalman Filter**: Combines PINN physics with real-time sensor data for improved forecasts
6. **Sensor Data Alignment**: Current sensors at time T are used to correct forecasts for T+3
7. **Uncertainty Quantification**: Kalman filter provides 95% confidence intervals for all predictions
8. **Parameter Optimization**: Kalman parameters optimized via multi-objective search prioritizing detection rate
9. **Unit Conversion**: Always use `313210039.9` for kg/m² to ppb conversion
10. **Normalization Ranges**: Must match exactly the values specified in this document

---

## Verification Checklist

When implementing or modifying the pipeline, verify:

- [ ] PINN uses simulation time t=3.0 hours (not absolute calendar time)
- [ ] Timestamps are shifted forward by 3 hours in training data
- [ ] PINN is computed directly at sensor locations (not interpolated)
- [ ] Each facility is processed separately, then superimposed
- [ ] Unit conversion factor is 313210039.9
- [ ] PINN normalization ranges match exactly
- [ ] Sensor coordinates match exactly
- [ ] Kalman filter parameters loaded from kalman_parameters.json
- [ ] Current sensor readings (time T) are used for Kalman update
- [ ] PINN predictions (for T+3) are used for Kalman predict step
- [ ] Kalman corrections are interpolated to full domain with confidence weighting
- [ ] Uncertainty bounds (95% CI) are computed and included in output

---

## Additional Technical Details

### PINN Temporal Multiplier Explanation

The temporal multiplier is a "soft temporal constraint" that ensures the initial condition (phi=0 at t=0) while allowing predictions at t=3 hours:

```python
t_norm_01 = (t_n + 1) / 2  # Convert from [-1,1] to [0,1]
RAMP_STEEPNESS = 100.0
temporal_multiplier = 1.0 - torch.exp(-RAMP_STEEPNESS * t_norm_01)
```

**Behavior**:
- At t=0 (t_norm_01=0): multiplier = 0 → phi = 0 (initial condition satisfied)
- At t=3 hours (t_norm_01 ≈ 0.00034): multiplier ≈ 0.033 (small but allows prediction)
- At t=438 hours (t_norm_01 ≈ 0.05): multiplier ≈ 0.99 (almost full output)
- At t >> 438 hours: multiplier ≈ 1.0 (full output)

This ensures the model respects the initial condition while still making meaningful predictions at t=3 hours.

### Data File Formats

#### Facility Data Format
**File**: `*_synced_training_data.csv`
**Columns**:
- `t`: Timestamp (when met data was collected)
- `source_x_cartesian`: Source x coordinate (meters)
- `source_y_cartesian`: Source y coordinate (meters)
- `source_diameter`: Source diameter (meters)
- `Q_total`: Total emission rate (kg/s)
- `wind_u`: Wind u component (m/s)
- `wind_v`: Wind v component (m/s)
- `D`: Diffusion coefficient

#### Training Data Output Format
**File**: `total_concentrations.csv`
**Columns**:
- `timestamp`: Forecast timestamp (input_timestamp + 3 hours)
- `sensor_482010026`: PINN prediction at sensor 482010026 (ppb)
- `sensor_482010057`: PINN prediction at sensor 482010057 (ppb)
- ... (9 sensor columns total)

#### Sensor Data Format
**File**: `sensors_final_synced.csv`
**Columns**:
- `timestamp` or `t`: Timestamp
- `sensor_482010026`: Actual sensor reading (ppb)
- ... (9 sensor columns total)

### Error Handling

**Zero Division Protection**:
- All scaler operations check for non-zero values before fitting/transforming
- Division by zero is prevented by checking `mask.any()` before operations

**NaN Handling**:
- Sensor readings: `np.nan_to_num(..., nan=0.0)`
- PINN outputs: Clipped to non-negative with `np.maximum(..., 0.0)`

**Missing Data**:
- If facility data is missing for a timestamp, skip that timestamp
- If sensor reading is missing, use 0.0 as default

---

## Code Snippets Reference

### Complete Training Data Generation Function

```python
def predict_pinn_at_sensors(pinn, facility_data):
    FORECAST_T_HOURS = 3.0
    predictions = {}
    
    with torch.no_grad():
        for idx, row in facility_data.iterrows():
            input_timestamp = pd.to_datetime(row['t'])
            t_hours_forecast = FORECAST_T_HOURS
            
            cx = row['source_x_cartesian']
            cy = row['source_y_cartesian']
            d = row['source_diameter']
            Q = row['Q_total']
            u = row['wind_u']
            v = row['wind_v']
            kappa = row['D']
            
            sensor_concentrations = {}
            for sensor_id, (sx, sy) in SENSORS.items():
                phi_raw = pinn(
                    torch.tensor([[sx]], dtype=torch.float32),
                    torch.tensor([[sy]], dtype=torch.float32),
                    torch.tensor([[t_hours_forecast]], dtype=torch.float32),
                    torch.tensor([[cx]], dtype=torch.float32),
                    torch.tensor([[cy]], dtype=torch.float32),
                    torch.tensor([[u]], dtype=torch.float32),
                    torch.tensor([[v]], dtype=torch.float32),
                    torch.tensor([[d]], dtype=torch.float32),
                    torch.tensor([[kappa]], dtype=torch.float32),
                    torch.tensor([[Q]], dtype=torch.float32),
                    normalize=True
                )
                concentration_ppb = phi_raw.item() * UNIT_CONVERSION_FACTOR
                sensor_concentrations[sensor_id] = concentration_ppb
            
            forecast_timestamp = input_timestamp + pd.Timedelta(hours=3)
            predictions[forecast_timestamp] = sensor_concentrations
    
    return predictions
```

---

## Summary of Critical Points

1. **Simulation Time**: Always use `t=3.0 hours` (NOT absolute calendar time)
2. **Timestamp Staggering**: Predictions made at t are labeled as t+3
3. **Direct Computation**: Compute PINN directly at sensors (don't interpolate)
4. **Superimposition**: Process each facility separately, then superimpose
5. **Unit Conversion**: Always use `313210039.9` for kg/m² to ppb
6. **Normalization Ranges**: Must match exactly the values specified

---

## End of Documentation

This document contains every detail needed to reproduce the pipeline exactly as implemented. All constants, methods, file paths, and processes are documented above.

**Total Lines**: 1110+
**Last Updated**: 2024

