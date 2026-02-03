# Benzene Dispersion Pipeline - Overview

## System Architecture

This is a real-time benzene concentration prediction pipeline for the Houston area that combines:
1. **Meteorological data** (MADIS/NOAA)
2. **Physics-Informed Neural Network (PINN)** for benzene dispersion modeling
3. **Neural Network Correction (NN2)** that corrects PINN predictions using real sensor data

---

## Pipeline Flow

```
Meteorological Data (MADIS)
    ↓
[1] Fetch weather data (wind, temp, solar radiation)
    ↓
[2] Generate facility-specific CSVs (20 facilities)
    ↓
[3] Compute PINN predictions (superimpose all facilities)
    ↓
[4] Superimpose values at sensor locations
    ↓
[5] Convert to ppb (parts per billion)
    ↓
[6] Apply NN2 correction using real benzene sensor data
    ↓
[7] Interpolate corrections across full domain
    ↓
[8] Output: Corrected concentration field + visualizations
```

---

## Key Components

### 1. **MADIS Fetcher** (`madis_fetcher.py`)
- Fetches real-time meteorological data from NOAA MADIS/NWS API
- Monitors 4 weather stations: MGPT2, F1563, C4814, D1774
- Extracts: wind speed, wind direction, temperature, solar radiation
- Falls back to mock data if API unavailable

### 2. **CSV Generator** (`csv_generator.py`)
- Processes 20 industrial facilities in Houston area
- For each facility:
  - Maps to nearest weather station
  - Calculates wind components (u, v)
  - Computes atmospheric stability class
  - Calculates diffusion coefficient (D)
  - Determines time-varying emission rate (Q) based on schedule
- Appends data to facility-specific time series CSVs

### 3. **PINN Model** (`simpletesting/pinn.py`)
- **Physics-Informed Neural Network** for advection-diffusion equation
- Solves: `∂φ/∂t + u·∂φ/∂x + v·∂φ/∂y = κ(∂²φ/∂x² + ∂²φ/∂y²) + S(x,y)`
- Inputs: (x, y, t, source_x, source_y, u, v, d, κ, Q)
- Output: Concentration field φ
- Trained on FEniCS simulation data
- Model file: `pinn_combined_final2.pth`

### 4. **Concentration Predictor** (`concentration_predictor.py`)
- Orchestrates PINN + NN2 prediction
- **Step 1**: Compute PINN for each facility across 30km×30km domain
- **Step 2**: Superimpose all facility plumes (additive)
- **Step 3**: Convert to ppb using factor: `3.13e8`
- **Step 4**: Apply NN2 correction at 9 sensor locations
- **Step 5**: Interpolate corrections across full domain using RBF
- **Step 6**: Apply distance-based confidence weighting

### 5. **NN2 Correction Network** (`simpletesting/nn2.py`)
- Neural network that corrects PINN predictions using real sensor data
- **Inputs**:
  - PINN predictions at 9 sensors
  - Current sensor readings (ground truth)
  - Sensor coordinates (spatial awareness)
  - Wind (u, v)
  - Diffusion coefficient (D)
  - Temporal features (hour, day, month, weekend)
- **Output**: Corrected concentrations at sensors
- **Architecture**: 5-layer MLP with BatchNorm and Dropout
- **Training**: Leave-one-sensor-out cross-validation
- Model file: `nn2_updated/nn2_master_model_spatial-3.pth`
- Scalers: `nn2_updated/nn2_master_scalers-2.pkl`

### 6. **Real-Time Pipeline** (`realtime_pipeline.py`)
- Main orchestrator
- Runs every 15 minutes (configurable)
- Executes full pipeline: fetch → generate → predict → visualize
- Logs all operations to `logs/pipeline.log`
- Supports single-run or continuous mode

---

## Training Data Flow (Historical)

### NN2 Training Process

1. **2019 Empty CSVs**: Historical meteorological data from 2019 (no benzene measurements)
2. **PINN Predictions**: 
   - Load 2019 met data for each facility
   - Run PINN to predict concentrations at 9 sensor locations
   - Superimpose all facilities per timestamp
   - Output: `total_superimposed_concentrations.csv`
3. **Sensor Ground Truth**: Real benzene sensor readings from 2019
4. **NN2 Training**:
   - Input: PINN predictions + sensor coordinates + meteorology
   - Target: Actual sensor readings
   - Method: Leave-one-sensor-out cross-validation
   - Result: Model learns to correct PINN biases

### Training Data Files
- **PINN predictions**: `nn2trainingdata/total_superimposed_concentrations.csv`
- **Sensor data**: `sensors_final_synced.csv`
- **Sensor coordinates**: `sensor_coordinates.csv`
- **Facility data**: `realtime_processing/houston_processed_2021/training_data_2021_beststations/`

---

## Facilities (20 Total)

1. ExxonMobil Baytown Refinery
2. Shell Deer Park Refinery
3. Valero Houston Refinery
4. LyondellBasell Pasadena Complex
5. LyondellBasell Channelview Complex
6. ExxonMobil Baytown Olefins Plant
7. Chevron Phillips Chemical Co
8. TPC Group
9. INEOS Phenol
10. Total Energies Petrochemicals
11. BASF Pasadena
12. Huntsman International
13. Invista
14. Goodyear Baytown
15. LyondellBasell Bayport Polymers
16. INEOS PP & Gemini
17. K-Solv Channelview
18. Oxy Vinyls Deer Park
19. ITC Deer Park
20. Enterprise Houston Terminal

Each facility has:
- Location (lat/lon + Cartesian coordinates)
- Source diameter (m)
- Emission rate schedule (Q_schedule_kg_s) - varies by time of day
- Assigned weather station

---

## Sensor Locations (9 Total)

Located in Cartesian coordinates (30km×30km domain):
- 482010026: (13972.62, 19915.57)
- 482010057: (3017.18, 12334.2)
- 482010069: (817.42, 9218.92)
- 482010803: (8836.35, 15717.2)
- 482011015: (18413.8, 15068.96)
- 482011035: (1159.98, 12272.52)
- 482011039: (13661.93, 5193.24)
- 482011614: (15077.79, 9450.52)
- 482016000: (1546.9, 6786.33)

---

## Output Files

### Continuous Time Series
- `data/continuous/superimposed_concentrations_timeseries.csv` - PINN only
- `data/continuous/nn2_corrected_domain_timeseries.csv` - PINN + NN2 corrected
- `data/continuous/sensor_predictions_timeseries.csv` - Predictions at sensor locations
- `data/continuous/per_facility/*_timeseries.csv` - Per-facility meteorology

### Latest Snapshots
- `data/predictions/latest_spatial_grid.csv` - Latest forecast (overwrites)

### Visualizations
- `data/visualizations/pinn_*.png` - PINN concentration maps
- `data/visualizations/nn2_*.png` - NN2-corrected maps
- `data/visualizations/sensor_bounded/*.png` - Sensor-bounded views

### Corrections
- `data/corrections/correction_timeseries.csv` - Correction statistics

---

## Key Constants

- **Domain size**: 30km × 30km (0 to 30000 m)
- **Grid resolution**: 100×100 = 10,000 points (configurable)
- **Unit conversion**: `3.13e8` (concentration to ppb)
- **Reference time**: `2021-01-01 00:00:00` (for PINN time calculation)
- **Forecast horizon**: t+3 hours
- **Update interval**: 15 minutes

---

## NN2 Correction Method

1. **Interpolate PINN** to 9 sensor locations using RBF
2. **Scale inputs** using trained scalers
3. **Run NN2** to get corrected values at sensors
4. **Calculate corrections**: `correction = NN2 - PINN` at sensors
5. **Interpolate corrections** across domain using RBF
6. **Apply confidence weighting**:
   - Full trust within 2km of sensors
   - Linear decay 2-5km
   - Zero trust beyond 5km
7. **Add weighted corrections** to PINN field

---

## Model Files

- **PINN**: `/Users/neevpratap/Downloads/pinn_combined_final2.pth`
- **NN2 Model**: `/Users/neevpratap/Desktop/nn2_updated/nn2_master_model_spatial-3.pth`
- **NN2 Scalers**: `/Users/neevpratap/Desktop/nn2_updated/nn2_master_scalers-2.pkl`
- **Sensor Coords**: Embedded in NN2 checkpoint

---

## Usage

### Run Once
```bash
python realtime/realtime_pipeline.py --mode once
```

### Run Continuously (every 15 min)
```bash
python realtime/realtime_pipeline.py --mode continuous --interval 15
```

### Show Statistics
```bash
python realtime/realtime_pipeline.py --stats
```

---

## Logs

- **Pipeline log**: `realtime/logs/pipeline.log` or `realtime/pipeline.log`
- Contains: timestamps, forecast times, concentration ranges, errors

---

## Validation

- `test_nn2_validation_2019.py` - Validates NN2 against 2019 sensor data
- `validate_pipeline.py` - End-to-end pipeline validation
- `investigate_nn2_mismatch.py` - Debugging tool for NN2 discrepancies

---

## Notes

- PINN solves advection-diffusion equation with Gaussian source terms
- NN2 learns spatial and temporal biases in PINN predictions
- Training used 2019 data (empty CSVs → PINN → NN2 training)
- Real-time pipeline uses 2021+ data for predictions
- Distance-based confidence ensures corrections only near sensors
- All concentrations in ppb (parts per billion)

