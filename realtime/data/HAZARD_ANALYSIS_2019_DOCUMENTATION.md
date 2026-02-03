# 2019 Hazard Event Analysis - PINN Prediction Documentation

**Date**: February 3, 2026  
**Purpose**: Analyze peak benzene concentration events in 2019 EDF sensor data and evaluate PINN's ability to predict such hazards

---

## Executive Summary

This analysis identified the top 10 highest benzene concentration events in 2019 from actual EDF sensor measurements, extracted the corresponding meteorological and facility conditions, and ran PINN predictions to evaluate whether the physics-informed model can predict such hazard events.

### Key Findings

- **Highest Hazard Event**: 196.94 ppb at sensor 482011015 on 2019-03-30 09:00:00 UTC
- **Top 10 Hazards Range**: 48.03 - 196.94 ppb (EDF measurements)
- **PINN Predictions**: 0.15 - 0.59 ppb (significantly under-predicting)
- **Cause**: Meteorological conditions (stable atmosphere, low wind, poor dispersion) - NOT accidents
- **Critical Issue**: PINN fails to predict high concentrations despite having correct meteorological inputs
- **Spatial Pattern Analysis**: Visualizations show PINN's spatial distribution patterns

### The Core Question: Why Can't PINN Predict These Hazards?

**The Paradox:**
- Conditions are known: Low wind (1.66-2.08 m/s), low diffusion (1.61-27.25 m²/s), normal emissions
- Physics says: Poor dispersion → High concentrations
- PINN predicts: 0.3-0.6 ppb (VERY LOW)
- EDF measures: 48-197 ppb (VERY HIGH)

**Root Causes:**
1. PINN may not be trained on sufficient extreme stable-condition examples
2. Advection-diffusion physics may not capture all stable-boundary-layer phenomena
3. Actual emissions during stable conditions may be higher than scheduled Q values
4. Local effects and micro-scale dispersion not captured in 20-facility model
5. Numerical/extrapolation issues with very low diffusion coefficients

---

## Methodology

### 1. Data Sources

**EDF Sensor Data** (Ground Truth):
- Primary: `/Users/neevpratap/Downloads/sensors_final_synced.csv`
- Fallback: `realtime/simpletesting/nn2trainingdata/sensors_final_synced.csv`
- **9 sensors** monitoring benzene concentrations in ppb
- **5,920 timestamps** in 2019 (hourly data, some gaps)

**Facility Weather Data**:
- Source: `realtime/simpletesting/nn2trainingdata/*_synced_training_data.csv`
- **20 facilities** with wind (u, v), diffusion (D), and emission rates (Q)
- Timestamp alignment: EDF reading at time T uses met data from T-3 hours

### 2. Hazard Identification Process

1. **Load EDF sensor data** for 2019
2. **Calculate metrics per timestamp**:
   - Maximum concentration across all 9 sensors
   - Mean concentration across all sensors
   - Total exposure (sum across sensors)
   - Peak sensor identification
3. **Rank by maximum concentration** (EDF actual readings)
4. **Identify top 10 hazard events**

### 3. PINN Prediction Process

For each identified hazard event:

1. **Extract facility conditions** from T-3 hours (3-hour forecast alignment)
2. **Create full domain grid**: 100×100 = 10,000 points (30km × 30km)
3. **Compute PINN predictions**:
   - Simulation time: `t=3.0 hours` (not absolute calendar time)
   - Process each of 20 facilities separately
   - Superimpose concentrations across all facilities
   - Convert to ppb using `UNIT_CONVERSION_FACTOR = 313210039.9`
4. **Compute PINN at sensor locations** for direct comparison with EDF

### 4. Visualization Generation

For each hazard event, created:

- **Full domain heatmap** (30km × 30km) showing PINN predicted concentrations
- **Sensor overlay** showing:
  - Actual EDF sensor readings (ground truth)
  - PINN predictions at sensor locations
  - Facility source locations (yellow stars)
  - Sensor positions (white triangles)
- **Comparison plot** showing EDF vs PINN at each of 9 sensors

---

## Top 10 Hazard Events

| Rank | Timestamp | EDF Peak (ppb) | PINN Peak (ppb) | Error (ppb) | % Error | Peak Sensor |
|------|-----------|----------------|-----------------|-------------|---------|-------------|
| 1 | 2019-03-30 09:00:00 | 196.94 | 0.46 | -196.48 | -99.8% | 482011015 |
| 2 | 2019-03-21 04:00:00 | 190.68 | 0.59 | -190.09 | -99.7% | 482011039 |
| 3 | 2019-03-20 21:00:00 | 165.17 | 0.37 | -164.80 | -99.8% | 482011015 |
| 4 | 2019-03-29 22:00:00 | 129.83 | 0.55 | -129.28 | -99.6% | 482010026 |
| 5 | 2019-04-14 23:00:00 | 98.17 | 0.15 | -98.02 | -99.8% | 482011015 |
| 6 | 2019-03-20 19:00:00 | 84.80 | 0.41 | -84.39 | -99.5% | 482011015 |
| 7 | 2019-03-23 05:00:00 | 70.05 | 0.59 | -69.47 | -99.2% | 482010026 |
| 8 | 2019-03-29 21:00:00 | 53.73 | 0.47 | -53.26 | -99.1% | 482010026 |
| 9 | 2019-04-04 10:00:00 | 48.76 | 0.31 | -48.45 | -99.4% | 482011015 |
| 10 | 2019-03-21 05:00:00 | 48.03 | 0.55 | -47.48 | -98.8% | 482011039 |

### Key Observations

1. **Temporal Clustering**: Most hazards occurred in late March 2019 (7 of 10 events)
2. **Sensor Patterns**: 
   - Sensor 482011015: 5 events (highest frequency)
   - Sensor 482010026: 3 events
   - Sensor 482011039: 2 events
3. **Time of Day**: Hazards occurred at various times (04:00, 05:00, 09:00, 10:00, 19:00, 21:00, 22:00, 23:00)

### Weather Conditions Analysis

**Top 5 Hazard Events - Meteorological Conditions:**

| Rank | Timestamp | EDF Peak | Wind Speed | Diffusion (D) | Stability | Emission Rate (Q) |
|------|-----------|----------|------------|---------------|-----------|-------------------|
| 1 | 2019-03-30 09:00 | 196.94 ppb | 1.76 m/s | 27.25 m²/s | STABLE | 0.0005 kg/s |
| 2 | 2019-03-21 04:00 | 190.68 ppb | 2.08 m/s | **1.61 m²/s** | **VERY STABLE** | 0.0007 kg/s |
| 3 | 2019-03-20 21:00 | 165.17 ppb | 1.66 m/s | 23.53 m²/s | STABLE | 0.0007 kg/s |
| 4 | 2019-03-29 22:00 | 129.83 ppb | 4.95 m/s | 10.75 m²/s | NEUTRAL | 0.0007 kg/s |
| 5 | 2019-04-14 23:00 | 98.17 ppb | 4.46 m/s | 9.59 m²/s | NEUTRAL | 0.0003 kg/s |

**Key Findings:**

1. **Low Wind Speeds**: Top 3 hazards occurred during low wind conditions (1.66-2.08 m/s)
   - Low wind = poor dispersion = higher concentrations

2. **Low Diffusion Coefficients**: Especially Hazard #2 (1.61 m²/s) indicates very stable atmospheric conditions
   - Stable conditions = limited vertical mixing = pollutants stay near ground

3. **Normal Emission Rates**: All events show typical emission rates (0.0003-0.0007 kg/s)
   - No evidence of unusually high emissions that would indicate accidents

4. **Stable Atmospheric Conditions**: Top 3 hazards occurred during STABLE conditions
   - Stable conditions are common during:
     - Early morning (04:00-09:00) - temperature inversions
     - Evening/night (19:00-23:00) - cooling, reduced mixing

**Conclusion: These appear to be METEOROLOGICAL CONDITIONS, not accidents**

- **Evidence for meteorological cause**:
  - Normal emission rates (no spikes)
  - Consistent with stable atmospheric conditions
  - Low wind speeds and diffusion coefficients
  - Temporal clustering suggests weather pattern (late March 2019)
  
- **Limitations**:
  - Emission rates in data are scheduled/typical, not actual measured emissions
  - An accident might not show up if it's not reflected in the Q values
  - No independent accident/incident data available for verification

---

## PINN Prediction Analysis

### Quantitative Results

- **PINN Under-Prediction**: Consistent across all hazard events
  - PINN predictions: 0.15 - 0.59 ppb
  - EDF measurements: 48.03 - 196.94 ppb
  - **Error range**: -98.8% to -99.8%

### Interpretation

**Why PINN Under-Predicts Hazards Despite Known Meteorological Conditions:**

This is a critical question: If the conditions are known (low wind, low diffusion, normal emissions), why can't PINN predict the high concentrations?

**The Paradox:**
- **Known Conditions**: Low wind (1.66-2.08 m/s), low diffusion (1.61-27.25 m²/s), normal emissions (0.0005-0.0007 kg/s)
- **Expected Physics**: Poor dispersion → High concentrations
- **PINN Prediction**: 0.3-0.6 ppb (VERY LOW)
- **EDF Measurement**: 48-197 ppb (VERY HIGH)
- **Discrepancy**: PINN predicts ~400x LOWER than measured

**Root Causes:**

1. **PINN Training Data Distribution**:
   - Training data: Mean wind=3.05 m/s, Mean diffusion=30.87 m²/s
   - Hazard conditions: Wind=1.66-2.08 m/s, Diffusion=1.61-27.25 m²/s
   - **Issue**: PINN may not have sufficient training examples with very stable conditions
   - Very low diffusion (D < 2 m²/s) may be rare in training data
   - Model may not extrapolate well to extreme stable conditions

2. **Physics Model Limitations**:
   - Standard advection-diffusion equation: `∂φ/∂t + u·∇φ = D∇²φ + S`
   - **Missing physics during stable conditions**:
     * Temperature inversions (trapping pollutants in shallow layer)
     * Boundary layer collapse (mixing height → 0)
     * Non-linear accumulation (concentrations build up over time, not just space)
     * Buoyancy effects (cold air trapping)
   - PINN may not capture these complex stable-boundary-layer phenomena

3. **Numerical/Extrapolation Issues**:
   - Very low diffusion (D ≈ 1.0-2.0 m²/s) approaches numerical limits
   - Low wind speeds may cause numerical instabilities in PINN
   - Model may default to "safe" low predictions when conditions are extreme

4. **Emission Rate Uncertainty**:
   - Q values in data are scheduled/typical rates (0.0005-0.0007 kg/s)
   - **Actual emissions during stable conditions might be higher**:
     * Facilities may not reduce operations during poor dispersion
     * Fugitive emissions may be higher when atmospheric pressure is stable
     * Some sources may have higher emissions during certain times
   - If actual Q is 10-100x higher, that would explain the discrepancy

5. **Local Effects Not in Model**:
   - EDF sensors may capture:
     * Local sources not in 20-facility model
     * Street-level emissions (traffic, local industry)
     * Building wake effects
     * Micro-scale dispersion patterns
   - These could contribute significantly during stable conditions

6. **Model Architecture Limitations**:
   - PINN may have learned to predict "typical" concentrations
   - May not have learned the non-linear relationship: Low wind + Low diffusion → Very high concentrations
   - The model might be biased toward moderate predictions

### Spatial Pattern Analysis

While PINN significantly under-predicts absolute values, the visualizations allow evaluation of:

1. **Spatial Distribution**: Does PINN identify which regions have higher concentrations?
2. **Relative Patterns**: Are the relative concentrations across sensors consistent?
3. **Plume Direction**: Does PINN capture the wind-driven plume direction?

**Note**: Review the generated visualizations to assess spatial pattern accuracy.

---

## Files Generated

### Analysis Scripts

1. **`realtime/analyze_2019_hazards.py`**
   - Analyzes EDF sensor data
   - Identifies top hazard events
   - Extracts facility conditions
   - Generates summary statistics

2. **`realtime/visualize_hazard_predictions.py`**
   - Runs PINN predictions for hazard events
   - Generates full-domain visualizations
   - Creates comparison plots
   - Generates summary reports

### Output Files

1. **`realtime/data/hazards_2019_summary.csv`**
   - Top 10 hazards with EDF values and rankings
   - Columns: timestamp, hazard_rank, max_concentration_edf, mean_concentration_edf, peak_sensor_id, all sensor values

2. **`realtime/data/hazards_2019_detailed.csv`**
   - All 5,920 timestamps with calculated metrics
   - Full ranking of all events

3. **`realtime/data/hazards_2019_with_conditions.pkl`**
   - Top 10 hazards with extracted facility conditions
   - Used for PINN predictions

4. **`realtime/data/hazards_2019_report.txt`**
   - Text summary report with EDF vs PINN comparison

5. **`realtime/data/hazards_2019_pinn_comparison.csv`**
   - Detailed comparison: EDF peak, PINN peak, error, % error

6. **`realtime/data/visualizations/hazards_2019/`**
   - 10 full-domain visualization files:
     - `hazard_2019-03-20_1900_pinn_prediction.png`
     - `hazard_2019-03-20_2100_pinn_prediction.png`
     - `hazard_2019-03-21_0400_pinn_prediction.png`
     - `hazard_2019-03-21_0500_pinn_prediction.png`
     - `hazard_2019-03-23_0500_pinn_prediction.png`
     - `hazard_2019-03-29_2100_pinn_prediction.png`
     - `hazard_2019-03-29_2200_pinn_prediction.png`
     - `hazard_2019-03-30_0900_pinn_prediction.png`
     - `hazard_2019-04-04_1000_pinn_prediction.png`
     - `hazard_2019-04-14_2300_pinn_prediction.png`

---

## Technical Details

### PINN Model Configuration

- **Model Path**: `/Users/neevpratap/Downloads/pinn_combined_final2.pth`
- **Simulation Time**: `t=3.0 hours` (fixed, not absolute calendar time)
- **Domain**: 30km × 30km (0 to 30,000 meters)
- **Grid Resolution**: 100×100 = 10,000 points
- **Unit Conversion**: 313,210,039.9 (kg/m² to ppb)

### Normalization Ranges

```
x, y: 0.0 - 30,000.0 m
t: 0.0 - 8,760.0 hours
cx, cy: 0.0 - 30,000.0 m
u, v: -15.0 - 15.0 m/s
d: 0.0 - 200.0 m
kappa (D): 0.0 - 200.0 m²/s
Q: 0.0 - 0.01 kg/s
```

### Sensor Locations (Cartesian, meters)

| Sensor ID | X (m) | Y (m) |
|-----------|-------|-------|
| 482010026 | 13,972.62 | 19,915.57 |
| 482010057 | 3,017.18 | 12,334.20 |
| 482010069 | 817.42 | 9,218.92 |
| 482010617 | 27,049.57 | 22,045.66 |
| 482010803 | 8,836.35 | 15,717.20 |
| 482011015 | 18,413.80 | 15,068.96 |
| 482011035 | 1,159.98 | 12,272.52 |
| 482011039 | 13,661.93 | 5,193.24 |
| 482016000 | 1,546.90 | 6,786.33 |

### Facility Parameters

20 facilities included:
- ExxonMobil Baytown Refinery
- Shell Deer Park Refinery
- Valero Houston Refinery
- LyondellBasell Pasadena Complex
- LyondellBasell Channelview Complex
- ExxonMobil Baytown Olefins Plant
- Chevron Phillips Chemical Co
- TPC Group
- INEOS Phenol
- Total Energies Petrochemicals
- BASF Pasadena
- Huntsman International
- Invista
- Goodyear Baytown
- LyondellBasell Bayport Polymers
- INEOS PP & Gemini
- K-Solv Channelview
- Oxy Vinyls Deer Park
- ITC Deer Park
- Enterprise Houston Terminal

Each facility has:
- Source location (Cartesian coordinates)
- Source diameter
- Emission rate (Q_total, varies by time of day)
- Assigned weather station

---

## Visualization Details

Each visualization contains:

### Left Panel: Full Domain PINN Prediction
- **Heatmap**: PINN predicted concentrations across 30km × 30km domain
- **Color Scale**: Plasma colormap, 0 to max(prediction, 100) ppb
- **Sensors**: White triangles with black edges (9 sensors)
- **Facilities**: Yellow stars with black edges (20 sources)
- **Title**: Includes hazard rank, EDF peak value, peak sensor, and timestamp

### Right Panel: EDF vs PINN Comparison
- **Bar Chart**: Side-by-side comparison at all 9 sensor locations
- **Red Bars**: EDF actual readings (ground truth)
- **Blue Bars**: PINN predicted values
- **Value Labels**: Concentration values displayed on each bar
- **Title**: Hazard rank and timestamp

---

## Conclusions

### PINN Performance on Hazard Events

1. **Absolute Value Prediction**: PINN significantly under-predicts extreme events
   - Predictions are ~0.3-0.6 ppb vs EDF measurements of 48-197 ppb
   - This is consistent with previous validation showing PINN tends to predict lower values
   - **Critical Finding**: Even with known stable conditions (low wind, low diffusion), PINN predicts low values
   - This suggests PINN may not fully capture the concentration buildup during poor dispersion conditions

2. **The Core Problem**: PINN fails to predict hazards despite having correct meteorological inputs
   - **Known**: Wind speeds (1.66-2.08 m/s), Diffusion (1.61-27.25 m²/s), Emissions (0.0005-0.0007 kg/s)
   - **Expected Physics**: Poor dispersion → High concentrations
   - **PINN Output**: Very low concentrations (0.3-0.6 ppb)
   - **EDF Reality**: Very high concentrations (48-197 ppb)
   - **Conclusion**: PINN does not properly model the physics of stable atmospheric conditions

2. **Spatial Pattern Prediction**: Requires visual inspection
   - Review generated visualizations to assess if PINN captures:
     - Which sensors should have higher concentrations
     - Wind-driven plume directions
     - Relative spatial patterns

3. **Model Limitations**:
   - PINN may not capture all physical processes leading to extreme events
   - Local sources or phenomena not in the 20-facility model
   - Complex meteorological conditions during extreme events
   - **Stable conditions**: PINN may not adequately model concentration buildup during very stable atmospheric conditions (low diffusion, low wind)

### Cause Analysis: Accidents vs. Meteorological Conditions

**Evidence suggests these were METEOROLOGICAL CONDITIONS, not accidents:**

1. **Normal Emission Rates**: All hazard events show typical Q values (0.0003-0.0007 kg/s)
   - No spikes that would indicate emergency releases or accidents
   - Emission rates are consistent with scheduled operations

2. **Stable Atmospheric Conditions**: Top hazards occurred during:
   - Low wind speeds (1.66-2.08 m/s for top 3)
   - Low diffusion coefficients (1.61-27.25 m²/s)
   - STABLE atmospheric stability class
   - These conditions naturally lead to poor dispersion and higher concentrations

3. **Temporal Patterns**: 
   - Clustering in late March 2019 suggests a weather pattern
   - Times of day (04:00, 09:00, 21:00, 22:00) are typical for stable conditions
   - Early morning and evening are common times for temperature inversions

4. **Physical Explanation**:
   - Stable conditions = limited vertical mixing = pollutants accumulate near ground
   - Low wind = slow horizontal dispersion = higher local concentrations
   - This is a well-known phenomenon in air quality modeling

**Limitations of Analysis:**

- **Emission Rate Data**: The Q values in the dataset are scheduled/typical rates, not actual measured emissions
  - An accident might not be reflected if it's not in the scheduled Q values
  - Actual emissions during these events could have been higher

- **No Independent Verification**: No accident/incident reports were available to cross-reference
  - Cannot definitively rule out accidents without external data

- **Local Sources**: Some high concentrations might be from local sources not in the 20-facility model

**Recommendation**: These events are most likely due to **meteorological conditions** (stable atmosphere, poor dispersion) rather than accidents, but verification would require:
- Actual measured emission rates during these events
- Incident reports from facilities
- Independent air quality monitoring data

### Recommendations

1. **Review Visualizations**: Assess spatial pattern accuracy despite absolute value differences
   - Do PINN predictions show higher concentrations at the correct sensors?
   - Are spatial patterns (which sensors are high) correct even if magnitudes are wrong?

2. **Investigate Training Data Coverage**:
   - Check how many training examples have D < 2 m²/s
   - Check how many training examples have wind < 2 m/s
   - If rare, PINN may need more training data for stable conditions

3. **Model Improvements**:
   - **Retrain PINN with more stable-condition examples**
   - **Add physics constraints** for very low diffusion scenarios
   - **Include boundary layer height** as an input parameter
   - **Model accumulation over time** (not just steady-state)

4. **Emission Rate Investigation**:
   - Verify if actual emissions during these events were higher than scheduled Q
   - Check if facilities have different emission patterns during stable conditions
   - Consider using measured emissions if available

5. **Hybrid Approach**:
   - Use PINN for spatial patterns (which areas are affected)
   - Use statistical/empirical models for magnitude during extreme stable conditions
   - Combine physics-based (PINN) with data-driven (EDF patterns) approaches

6. **Stability-Aware Modeling**:
   - Develop separate models or corrections for different stability classes
   - Very stable conditions (D < 5 m²/s) may need special handling
   - Consider stability-dependent scaling factors

---

## Usage

### Re-running Analysis

```bash
# Step 1: Analyze EDF data and identify hazards
cd /Users/neevpratap/Desktop/benzenepipelinev2/realtime
python analyze_2019_hazards.py

# Step 2: Generate PINN predictions and visualizations
python visualize_hazard_predictions.py
```

### Viewing Results

1. **Summary Report**: `realtime/data/hazards_2019_report.txt`
2. **Detailed Comparison**: `realtime/data/hazards_2019_pinn_comparison.csv`
3. **Visualizations**: `realtime/data/visualizations/hazards_2019/*.png`

---

## References

- **EDF Sensor Data**: Actual benzene concentration measurements from 2019
- **PINN Model**: Physics-Informed Neural Network for advection-diffusion equation
- **Facility Data**: 20 industrial facilities in Houston area with emission rates and weather conditions
- **Training Data**: Generated using simulation time t=3.0 hours (not absolute calendar time)

---

**Document Version**: 1.0  
**Last Updated**: February 3, 2026  
**Analysis Date**: February 3, 2026

