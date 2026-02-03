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
from google.colab import drive # Import drive
import shutil # Import shutil for file operations

# ═══════════════════════
# 1. MOUNT GOOGLE DRIVE
# ═══════════════════════

print("Mounting Google Drive...")
drive.mount('/content/drive')
print("✓ Drive mounted successfully\n")

# ═══════════════════════
# 2. SET PATHS
# ═══════════════════════

import os

# Source path in Google Drive (now in My Drive root)
DRIVE_PATH = "/content/drive/MyDrive/nn2_data_2019"

# Verify path exists
if not os.path.exists(DRIVE_PATH):
    print("☢′ Path not found. Checking My Drive contents...")
    mydrive_contents = os.listdir("/content/drive/MyDrive")
    print(f"My Drive folders: {mydrive_contents}")
    raise FileNotFoundError(f"Could not find {DRIVE_PATH}")

# Show what's in the folder
print(f"✓ Found data folder: {DRIVE_PATH}")
contents = os.listdir(DRIVE_PATH)
print(f"  Files found: {len(contents)}")
for f in sorted(contents)[:10]:
    print(f"    • {f}")
if len(contents) > 10:
    print(f"    ... and {len(contents) - 10} more files")

# Destination path in Colab (local)
LOCAL_DATA_DIR = "/content/data"
LOCAL_SOURCE_DIR = f"{LOCAL_DATA_DIR}/data_nonzero"

# Create local directories
Path(LOCAL_DATA_DIR).mkdir(exist_ok=True, parents=True)
Path(LOCAL_SOURCE_DIR).mkdir(exist_ok=True, parents=True)

print(f"\nDestination: {LOCAL_DATA_DIR}\n")

# ═══════════════════════
# 3. COPY SENSOR DATA
# ═══════════════════════

print("Copying sensor data...")
sensor_file = f"{DRIVE_PATH}/sensors_final_synced.csv"
dest_sensor = f"{LOCAL_DATA_DIR}/sensors_final.csv"

shutil.copy2(sensor_file, dest_sensor)
print(f"✓ Copied: sensors_final_synced.csv → {dest_sensor}")

# Verify sensor file - first check what columns exist
df_sensors = pd.read_csv(dest_sensor)
print(f"  → Loaded {len(df_sensors)} rows, {len(df_sensors.columns)} columns")
print(f"  → Columns: {list(df_sensors.columns)}")

# Find the timestamp column (might be 't' or 'timestamp')
time_col = None
for col in ['timestamp', 't', 'time', 'datetime']:
    if col in df_sensors.columns:
        time_col = col
        break

if time_col:
    df_sensors[time_col] = pd.to_datetime(df_sensors[time_col])
    print(f"  → Time column: '{time_col}' ({df_sensors[time_col].min()} to {df_sensors[time_col].max()})")

    # Rename to 'timestamp' if needed
    if time_col != 'timestamp':
        df_sensors = df_sensors.rename(columns={time_col: 'timestamp'})
        df_sensors.to_csv(dest_sensor, index=False)
        print(f"  → Renamed '{time_col}' to 'timestamp'")
else:
    print(f"  ☢′ No timestamp column found! Available columns: {list(df_sensors.columns)}")
print()

# Copy sensor coordinates
print("Copying sensor coordinates...")
sensor_coords_file = f"{DRIVE_PATH}/sensor_coordinates.csv"
dest_coords = f"{LOCAL_DATA_DIR}/sensor_coordinates.csv"

try:
    shutil.copy2(sensor_coords_file, dest_coords)
    print(f"✓ Copied: sensor_coordinates.csv → {dest_coords}")

    # Verify sensor coordinates file
    df_coords = pd.read_csv(dest_coords)
    print(f"  → Loaded {len(df_coords)} sensor locations")
    print(f"  → Columns: {list(df_coords.columns)}")
    if 'x' in df_coords.columns and 'y' in df_coords.columns:
        print(f"  → X range: {df_coords['x'].min():.1f} to {df_coords['x'].max():.1f} m")
        print(f"  → Y range: {df_coords['y'].min():.1f} to {df_coords['y'].max():.1f} m")
        print(f"  → Domain: {(df_coords['x'].max()-df_coords['x'].min())/1000:.1f} km × {(df_coords['y'].max()-df_coords['y'].min())/1000:.1f} km")
except FileNotFoundError:
    print(f"  ✗ File not found: {sensor_coords_file}")
    print(f"  ☢′ Sensor coordinates are required for spatial learning in NN2!")
    raise
print()

# ═══════════════════════
# 4. COPY PINN SUPERIMPOSED CONCENTRATIONS
# ═══════════════════════

print("Copying PINN superimposed concentrations...")
pinn_file = f"{DRIVE_PATH}/total_superimposed_concentrations.csv"
dest_pinn = f"{LOCAL_DATA_DIR}/total_superimposed_concentrations.csv"

try:
    shutil.copy2(pinn_file, dest_pinn)
    print(f"✓ Copied: total_superimposed_concentrations.csv → {dest_pinn}")

    # Verify PINN file
    df_pinn = pd.read_csv(dest_pinn)
    print(f"  → Loaded {len(df_pinn)} rows, {len(df_pinn.columns)} columns")
    print(f"  → Columns: {list(df_pinn.columns)}")

    # Check for timestamp column
    pinn_time_col = None
    for col in ['timestamp', 't', 'time', 'datetime']:
        if col in df_pinn.columns:
            pinn_time_col = col
            break

    if pinn_time_col:
        df_pinn[pinn_time_col] = pd.to_datetime(df_pinn[pinn_time_col])
        print(f"  → Time column: '{pinn_time_col}' ({df_pinn[pinn_time_col].min()} to {df_pinn[pinn_time_col].max()})")

        # Check for sensor_id column
        if 'sensor_id' in df_pinn.columns:
            unique_sensors = df_pinn['sensor_id'].nunique()
            print(f"  → Unique sensors: {unique_sensors}")
            print(f"  → Sensor IDs: {sorted(df_pinn['sensor_id'].unique())}")

        # Check for concentration column
        conc_cols = [col for col in df_pinn.columns if 'phi' in col.lower() or 'concentration' in col.lower()]
        if conc_cols:
            print(f"  → Concentration columns: {conc_cols}")

            # Show sample statistics
            for conc_col in conc_cols:
                values = df_pinn[conc_col].dropna()
                if len(values) > 0:
                    print(f"     {conc_col}: min={values.min():.2e}, max={values.max():.2e}, mean={values.mean():.2e}")
    else:
        print(f"  ☢′ No timestamp column found! Available columns: {list(df_pinn.columns)}")

    print()

except FileNotFoundError:
    print(f"  ✗ File not found: {pinn_file}")
    print(f"  ☢′ PINN data is required for NN2 training!")
    print()

# ═══════════════════════
# 5. COPY ALL SOURCE FILES
# ═══════════════════════

print("Copying source emission files...")

# Try to find all .csv files matching the pattern
drive_files = os.listdir(DRIVE_PATH)
source_files = [f for f in drive_files if f.endswith('_synced_training_data.csv')]

n_sources = 0
source_info = {}

for filename in sorted(source_files):
    src = f"{DRIVE_PATH}/{filename}"

    # Clean up the destination filename (remove _synced)
    clean_name = filename.replace('_synced_training_data', '_training_data')
    dest = f"{LOCAL_SOURCE_DIR}/{clean_name}"

    try:
        shutil.copy2(src, dest)

        # Load and verify
        df = pd.read_csv(dest, parse_dates=['t'])
        source_name = clean_name.replace('_training_data.csv', '')
        source_info[source_name] = {
            'filename': clean_name,
            'rows': len(df),
            'columns': list(df.columns)
        }

        n_sources += 1
        print(f"  ✓ {n_sources:2d}. {source_name}")
        print(f"       → {len(df)} rows")

    except Exception as e:
        print(f"  ✗ Failed to copy {filename}: {e}")

print(f"\n✓ Copied {n_sources} source files\n")

# ═══════════════════════
# 6. VERIFY DATA STRUCTURE
# ═══════════════════════

print("="*70)
print("DATA STRUCTURE SUMMARY")
print("="*70)

print(f"\n📁 {LOCAL_DATA_DIR}/")
print(f"   ├── sensors_final.csv ({len(df_sensors)} rows)")
print(f"   ├── sensor_coordinates.csv ({len(df_coords) if 'df_coords' in locals() else '?'} sensors)")
print(f"   ├── total_superimposed_concentrations.csv ({len(df_pinn) if 'df_pinn' in locals() else '?'} rows)")
print(f"   └── data_nonzero/")

for i, (name, info) in enumerate(sorted(source_info.items()), 1):
    connector = "├──" if i < len(source_info) else "└──"
    print(f"       {connector} {info['filename']} ({info['rows']} rows)")

# ═══════════════════════
# 7. VERIFY COLUMN CONSISTENCY
# ═══════════════════════

print("\n" + "="*70)
print("COLUMN VERIFICATION")
print("="*70)

# Check sensor columns
sensor_cols = [col for col in df_sensors.columns if col.startswith('sensor_')]
print(f"\nSensor columns ({len(sensor_cols)}): {sensor_cols}")

# Verify sensor coordinates match sensor columns
if 'df_coords' in locals():
    coord_sensors = df_coords['sensor_id'].tolist()
    coord_sensor_nums = [int(s.replace('sensor_', '')) for s in coord_sensors]
    data_sensor_nums = [int(s.replace('sensor_', '')) for s in sensor_cols]

    print(f"\nSensor coordinate verification:")
    print(f"  Sensors in coordinates file: {coord_sensors}")
    print(f"  Sensors in data file: {sensor_cols}")

    if set(coord_sensor_nums) == set(data_sensor_nums):
        print(f"  ✓ All sensors have coordinates!")
    else:
        missing = set(data_sensor_nums) - set(coord_sensor_nums)
        extra = set(coord_sensor_nums) - set(data_sensor_nums)
        if missing:
            print(f"  ☢′ Missing coordinates for sensors: {missing}")
        if extra:
            print(f"  ☢′ Extra coordinates for sensors: {extra}")

# Check PINN data structure
if 'df_pinn' in locals():
    print(f"\nPINN data structure:")
    print(f"  Total rows: {len(df_pinn)}")
    if 'sensor_id' in df_pinn.columns:
        pinn_sensor_ids = sorted(df_pinn['sensor_id'].unique())
        print(f"  Sensor IDs in PINN: {pinn_sensor_ids}")

        # Compare with actual sensor columns
        sensor_nums = [int(s.replace('sensor_', '')) for s in sensor_cols]
        matching = set(pinn_sensor_ids) & set(sensor_nums)
        print(f"  Matching sensors: {len(matching)}/{len(sensor_cols)}")

        if len(matching) < len(sensor_cols):
            missing = set(sensor_nums) - set(pinn_sensor_ids)
            print(f"  ☢′ Sensors missing in PINN: {missing}")

# Check if all sources have required columns
required_source_cols = ['t', 'wind_u', 'wind_v', 'D', 'Q_total']

print(f"\nChecking source files for required columns: {required_source_cols}")
all_good = True

for name, info in sorted(source_info.items()):
    missing = [col for col in required_source_cols if col not in info['columns']]
    if missing:
        print(f"  ✗ {name}: Missing {missing}")
        all_good = False
    else:
        print(f"  ✓ {name}: All required columns present")

if all_good:
    print("\n✓ All source files have required columns!")
else:
    print("\n☢′ Some files are missing required columns!")

# ═══════════════════════
# 8. CREATE METADATA FILE
# ═══════════════════════

import json

metadata = {
    'n_sources': n_sources,
    'n_sensors': len(sensor_cols),
    'sensor_ids': sensor_cols,
    'has_coordinates': 'df_coords' in locals(),
    'coordinate_domain_km': {
        'width': float(f"{(df_coords['x'].max()-df_coords['x'].min())/1000:.2f}") if 'df_coords' in locals() else None,
        'height': float(f"{(df_coords['y'].max()-df_coords['y'].min())/1000:.2f}") if 'df_coords' in locals() else None
    },
    'source_names': sorted(list(source_info.keys())),
    'data_path': LOCAL_DATA_DIR,
    'source_dir': LOCAL_SOURCE_DIR,
    'sensor_rows': len(df_sensors),
    'pinn_rows': len(df_pinn) if 'df_pinn' in locals() else 0,
    'timestamp_range': {
        'sensors_start': str(df_sensors['timestamp'].min()),
        'sensors_end': str(df_sensors['timestamp'].max()),
        'pinn_start': str(df_pinn[pinn_time_col].min()) if 'df_pinn' in locals() and pinn_time_col else 'N/A',
        'pinn_end': str(df_pinn[pinn_time_col].max()) if 'df_pinn' in locals() and pinn_time_col else 'N/A'
    }
}

with open(f"{LOCAL_DATA_DIR}/metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*70)
print("✓ DATA DOWNLOAD COMPLETE!")
print("="*70)
print(f"\nYou can now train NN2 with:")
print(f"  • Sensor data (truth):       {dest_sensor}")
print(f"  • Sensor coordinates (x,y):  {dest_coords}")
print(f"  • PINN data (input):         {dest_pinn}")
print(f"  • Source data (meteo):       {LOCAL_SOURCE_DIR}/")
print(f"  • Metadata:                  {LOCAL_DATA_DIR}/metadata.json")
print(f"\nTotal: {n_sources} sources, {len(sensor_cols)} sensors")
if 'df_coords' in locals():
    print(f"Spatial domain: {(df_coords['x'].max()-df_coords['x'].min())/1000:.1f} km × {(df_coords['y'].max()-df_coords['y'].min())/1000:.1f} km")
if 'df_pinn' in locals():
    print(f"PINN predictions: {len(df_pinn)} rows")
print("="*70)

#Mounting Google Drive...
Mounted at /content/drive
✓ Drive mounted successfully

✓ Found data folder: /content/drive/MyDrive/nn2_data_2019
  Files found: 26
    • BASF_Pasadena_synced_training_data.csv
    • Chevron_Phillips_Chemical_Co_synced_training_data.csv
    • Enterprise_Houston_Terminal_synced_training_data.csv
    • ExxonMobil_Baytown_Olefins_Plant_synced_training_data.csv
    • ExxonMobil_Baytown_Refinery_synced_training_data.csv
    • Goodyear_Baytown_synced_training_data.csv
    • Huntsman_International_synced_training_data.csv
    • INEOS_PP_&_Gemini_synced_training_data.csv
    • INEOS_Phenol_synced_training_data.csv
    • ITC_Deer_Park_synced_training_data.csv
    ... and 16 more files

Destination: /content/data

Copying sensor data...
✓ Copied: sensors_final_synced.csv → /content/data/sensors_final.csv
  → Loaded 5920 rows, 10 columns
  → Columns: ['t', 'sensor_482010026', 'sensor_482010057', 'sensor_482010069', 'sensor_482010617', 'sensor_482010803', 'sensor_482011015', 'sensor_482011035', 'sensor_482011039', 'sensor_482016000']
  → Time column: 't' (2019-01-01 13:00:00 to 2019-12-31 22:00:00)
  → Renamed 't' to 'timestamp'

Copying sensor coordinates...
✓ Copied: sensor_coordinates.csv → /content/data/sensor_coordinates.csv
  → Loaded 9 sensor locations
  → Columns: ['sensor_id', 'x', 'y']
  → X range: -10747.2 to 15441.1 m
  → Y range: -8815.7 to 7981.4 m
  → Domain: 26.2 km × 16.8 km

Copying PINN superimposed concentrations...
✓ Copied: total_superimposed_concentrations.csv → /content/data/total_superimposed_concentrations.csv
  → Loaded 6001 rows, 10 columns
  → Columns: ['timestamp', 'sensor_482010026', 'sensor_482010057', 'sensor_482010069', 'sensor_482010617', 'sensor_482010803', 'sensor_482011015', 'sensor_482011035', 'sensor_482011039', 'sensor_482016000']
  → Time column: 'timestamp' (2019-01-01 16:00:00 to 2020-01-01 01:00:00)

Copying source emission files...
  ✓  1. BASF_Pasadena
       → 6001 rows
  ✓  2. Chevron_Phillips_Chemical_Co
       → 6001 rows
  ✓  3. Enterprise_Houston_Terminal
       → 6001 rows
  ✓  4. ExxonMobil_Baytown_Olefins_Plant
       → 6001 rows
  ✓  5. ExxonMobil_Baytown_Refinery
       → 6001 rows
  ✓  6. Goodyear_Baytown
       → 6001 rows
  ✓  7. Huntsman_International
       → 6001 rows
  ✓  8. INEOS_PP_&_Gemini
       → 6001 rows
  ✓  9. INEOS_Phenol
       → 6001 rows
  ✓ 10. ITC_Deer_Park
       → 6001 rows
  ✓ 11. Invista
       → 6001 rows
  ✓ 12. K-Solv_Channelview
       → 6001 rows
  ✓ 13. LyondellBasell_Bayport_Polymers
       → 6001 rows
  ✓ 14. LyondellBasell_Channelview_Complex
       → 6001 rows
  ✓ 15. LyondellBasell_Pasadena_Complex
       → 6001 rows
  ✓ 16. Oxy_Vinyls_Deer_Park
       → 6001 rows
  ✓ 17. Shell_Deer_Park_Refinery
       → 6001 rows
  ✓ 18. TPC_Group
       → 6001 rows
  ✓ 19. Total_Energies_Petrochemicals
       → 6001 rows
  ✓ 20. Valero_Houston_Refinery
       → 6001 rows

✓ Copied 20 source files

======================================================================
DATA STRUCTURE SUMMARY
======================================================================

📁 /content/data/
   ├── sensors_final.csv (5920 rows)
   ├── sensor_coordinates.csv (9 sensors)
   ├── total_superimposed_concentrations.csv (6001 rows)
   └── data_nonzero/
       ├── BASF_Pasadena_training_data.csv (6001 rows)
       ├── Chevron_Phillips_Chemical_Co_training_data.csv (6001 rows)
       ├── Enterprise_Houston_Terminal_training_data.csv (6001 rows)
       ├── ExxonMobil_Baytown_Olefins_Plant_training_data.csv (6001 rows)
       ├── ExxonMobil_Baytown_Refinery_training_data.csv (6001 rows)
       ├── Goodyear_Baytown_training_data.csv (6001 rows)
       ├── Huntsman_International_training_data.csv (6001 rows)
       ├── INEOS_PP_&_Gemini_training_data.csv (6001 rows)
       ├── INEOS_Phenol_training_data.csv (6001 rows)
       ├── ITC_Deer_Park_training_data.csv (6001 rows)
       ├── Invista_training_data.csv (6001 rows)
       ├── K-Solv_Channelview_training_data.csv (6001 rows)
       ├── LyondellBasell_Bayport_Polymers_training_data.csv (6001 rows)
       ├── LyondellBasell_Channelview_Complex_training_data.csv (6001 rows)
       ├── LyondellBasell_Pasadena_Complex_training_data.csv (6001 rows)
       ├── Oxy_Vinyls_Deer_Park_training_data.csv (6001 rows)
       ├── Shell_Deer_Park_Refinery_training_data.csv (6001 rows)
       ├── TPC_Group_training_data.csv (6001 rows)
       ├── Total_Energies_Petrochemicals_training_data.csv (6001 rows)
       └── Valero_Houston_Refinery_training_data.csv (6001 rows)

======================================================================
COLUMN VERIFICATION
======================================================================

Sensor columns (9): ['sensor_482010026', 'sensor_482010057', 'sensor_482010069', 'sensor_482010617', 'sensor_482010803', 'sensor_482011015', 'sensor_482011035', 'sensor_482011039', 'sensor_482016000']

Sensor coordinate verification:
  Sensors in coordinates file: ['sensor_482010026', 'sensor_482010057', 'sensor_482010069', 'sensor_482010617', 'sensor_482010803', 'sensor_482011015', 'sensor_482011035', 'sensor_482011039', 'sensor_482016000']
  Sensors in data file: ['sensor_482010026', 'sensor_482010057', 'sensor_482010069', 'sensor_482010617', 'sensor_482010803', 'sensor_482011015', 'sensor_482011035', 'sensor_482011039', 'sensor_482016000']
  ✓ All sensors have coordinates!

PINN data structure:
  Total rows: 6001

Checking source files for required columns: ['t', 'wind_u', 'wind_v', 'D', 'Q_total']
  ✓ BASF_Pasadena: All required columns present
  ✓ Chevron_Phillips_Chemical_Co: All required columns present
  ✓ Enterprise_Houston_Terminal: All required columns present
  ✓ ExxonMobil_Baytown_Olefins_Plant: All required columns present
  ✓ ExxonMobil_Baytown_Refinery: All required columns present
  ✓ Goodyear_Baytown: All required columns present
  ✓ Huntsman_International: All required columns present
  ✓ INEOS_PP_&_Gemini: All required columns present
  ✓ INEOS_Phenol: All required columns present
  ✓ ITC_Deer_Park: All required columns present
  ✓ Invista: All required columns present
  ✓ K-Solv_Channelview: All required columns present
  ✓ LyondellBasell_Bayport_Polymers: All required columns present
  ✓ LyondellBasell_Channelview_Complex: All required columns present
  ✓ LyondellBasell_Pasadena_Complex: All required columns present
  ✓ Oxy_Vinyls_Deer_Park: All required columns present
  ✓ Shell_Deer_Park_Refinery: All required columns present
  ✓ TPC_Group: All required columns present
  ✓ Total_Energies_Petrochemicals: All required columns present
  ✓ Valero_Houston_Refinery: All required columns present

✓ All source files have required columns!

======================================================================
✓ DATA DOWNLOAD COMPLETE!
======================================================================

You can now train NN2 with:
  • Sensor data (truth):       /content/data/sensors_final.csv
  • Sensor coordinates (x,y):  /content/data/sensor_coordinates.csv
  • PINN data (input):         /content/data/total_superimposed_concentrations.csv
  • Source data (meteo):       /content/data/data_nonzero/
  • Metadata:                  /content/data/metadata.json

Total: 20 sources, 9 sensors
Spatial domain: 26.2 km × 16.8 km
PINN predictions: 6001 rows
======================================================================


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
    'save_dir': '/content/models/leave_one_out/',
    'data_dir': '/content/data/',
    'pinn_file': '/content/data/total_superimposed_concentrations.csv',
    'sensor_coords_file': '/content/data/sensor_coordinates.csv'
}

# ═══════════════════════════════════════════════════════════════════
# 1. NN2 NETWORK - WITH SPATIAL AWARENESS
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

        self.correction_network = nn.Sequential(
            nn.Linear(36, 512),
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
# 2. DATASET - WITH SCALER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

class BenzeneDataset(Dataset):
    def __init__(self, data_path, source_dir, pinn_path, sensor_coords_path,
                 held_out_sensor_idx=None, scalers=None, fit_scalers=False):
        """
        Args:
            sensor_coords_path: Path to CSV with sensor coordinates
            held_out_sensor_idx: Index (0-8) of sensor to hold out
            scalers: Pre-fitted scalers (if available)
            fit_scalers: If True, fit new scalers on this data
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
        coords_df = coords_df.sort_values('sensor_id').reset_index(drop=True)
        self.sensor_coords = coords_df[['x', 'y']].values.astype(float)
        print(f"   Loaded coordinates for {len(self.sensor_coords)} sensors")

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

        # Get sensor IDs
        all_sensor_ids = [col for col in self.sensors_df.columns if col.startswith('sensor_')]
        self.all_sensor_ids = all_sensor_ids
        self.n_sensors = len(all_sensor_ids)

        if held_out_sensor_idx is not None:
            self.train_sensor_ids = [sid for i, sid in enumerate(all_sensor_ids) if i != held_out_sensor_idx]
            self.held_out_sensor_id = all_sensor_ids[held_out_sensor_idx]
            print(f"\n🎯 Training sensors: {len(self.train_sensor_ids)}")
            print(f"🔒 Held-out sensor: {self.held_out_sensor_id} (index {held_out_sensor_idx})")
        else:
            self.train_sensor_ids = all_sensor_ids
            self.held_out_sensor_id = None

        # Initialize or use provided scalers
        if fit_scalers:
            self.scalers = {
                'sensors': StandardScaler(),
                'pinn': StandardScaler(),
                'wind': StandardScaler(),
                'diffusion': StandardScaler(),
                'coords': StandardScaler(),
            }
        else:
            self.scalers = scalers if scalers is not None else {
                'sensors': StandardScaler(),
                'pinn': StandardScaler(),
                'wind': StandardScaler(),
                'diffusion': StandardScaler(),
                'coords': StandardScaler(),
            }

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
        common_times = sorted(list(sensor_times & pinn_times & meteo_times))
        print(f"  Common timestamps: {len(common_times)}")

        # Filter and align
        self.sensors_df = self.sensors_df[self.sensors_df['timestamp'].isin(common_times)].sort_values('timestamp').reset_index(drop=True)
        self.pinn_df = self.pinn_df[self.pinn_df['timestamp'].isin(common_times)].sort_values('timestamp').reset_index(drop=True)
        self.meteo_df = self.meteo_df[self.meteo_df['timestamp'].isin(common_times)].sort_values('timestamp').reset_index(drop=True)

        # Extract features
        self.actual_sensors = self.sensors_df[self.all_sensor_ids].values.astype(float)
        self.valid_mask = ~np.isnan(self.actual_sensors)
        self.actual_sensors = np.nan_to_num(self.actual_sensors, nan=0.0)
        
        # Store original ppb values BEFORE scaling (for ppb-space training)
        self.actual_sensors_ppb = self.actual_sensors.copy()

        self.pinn_predictions = self.pinn_df[self.all_sensor_ids].values.astype(float)

        self.wind = self.meteo_df[['wind_u', 'wind_v']].values.astype(float)
        self.diffusion = self.meteo_df['D'].values.astype(float).reshape(-1, 1)

        # Temporal features
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
        # FIT SCALERS
        # ═══════════════════════════════════════════════════════════
        if fit:
            print(f"\n  🔧 Fitting scalers...")

            # Coordinate scaler
            self.scalers['coords'].fit(self.sensor_coords)

            # Sensor and PINN scalers
            if self.held_out_sensor_idx is not None:
                train_sensor_mask = np.ones(self.n_sensors, dtype=bool)
                train_sensor_mask[self.held_out_sensor_idx] = False
                train_actual = self.actual_sensors[:, train_sensor_mask]
                train_pinn = self.pinn_predictions[:, train_sensor_mask]

                valid_sensors = train_actual[train_actual != 0]
                if len(valid_sensors) > 0:
                    self.scalers['sensors'].fit(valid_sensors.reshape(-1, 1))

                valid_pinn = train_pinn[train_pinn != 0]
                if len(valid_pinn) > 0:
                    self.scalers['pinn'].fit(valid_pinn.reshape(-1, 1))
            else:
                valid_sensors = self.actual_sensors[self.actual_sensors != 0]
                if len(valid_sensors) > 0:
                    self.scalers['sensors'].fit(valid_sensors.reshape(-1, 1))

                valid_pinn = self.pinn_predictions[self.pinn_predictions != 0]
                if len(valid_pinn) > 0:
                    self.scalers['pinn'].fit(valid_pinn.reshape(-1, 1))

            self.scalers['wind'].fit(self.wind)
            self.scalers['diffusion'].fit(self.diffusion)

            # ═══════════════════════════════════════════════════════════
            # PRINT SCALER STATISTICS (for verification)
            # ═══════════════════════════════════════════════════════════
            print(f"\n  📊 Scaler Statistics:")
            print(f"     Sensors:   mean={self.scalers['sensors'].mean_[0]:.4f}, std={self.scalers['sensors'].scale_[0]:.4f}")
            print(f"     PINN:      mean={self.scalers['pinn'].mean_[0]:.4f}, std={self.scalers['pinn'].scale_[0]:.4f}")
            print(f"     Wind U:    mean={self.scalers['wind'].mean_[0]:.4f}, std={self.scalers['wind'].scale_[0]:.4f}")
            print(f"     Wind V:    mean={self.scalers['wind'].mean_[1]:.4f}, std={self.scalers['wind'].scale_[1]:.4f}")
            print(f"     Diffusion: mean={self.scalers['diffusion'].mean_[0]:.4f}, std={self.scalers['diffusion'].scale_[0]:.4f}")
            print(f"     Coords X:  mean={self.scalers['coords'].mean_[0]:.4f}, std={self.scalers['coords'].scale_[0]:.4f}")
            print(f"     Coords Y:  mean={self.scalers['coords'].mean_[1]:.4f}, std={self.scalers['coords'].scale_[1]:.4f}")

        # Transform data
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
        self.sensor_coords_normalized = self.scalers['coords'].transform(self.sensor_coords)

    def __len__(self):
        return len(self.actual_sensors)

    def __getitem__(self, idx):
        coords_for_sample = torch.FloatTensor(self.sensor_coords_normalized)

        return {
            # REMOVED: 'current_sensors' - this was causing data leakage!
            # The model should learn corrections based on PINN + conditions, not actual sensor values
            'pinn_predictions': torch.FloatTensor(self.pinn_predictions[idx]),  # Scaled
            'sensor_coords': coords_for_sample,
            'wind': torch.FloatTensor(self.wind[idx]),
            'diffusion': torch.FloatTensor(self.diffusion[idx]),
            'temporal': torch.FloatTensor(self.temporal[idx]),
            'target': torch.FloatTensor(self.actual_sensors[idx]),  # Scaled (for legacy)
            'target_ppb': torch.FloatTensor(self.actual_sensors_ppb[idx]),  # PPB (for new training)
            'valid_mask': torch.BoolTensor(self.valid_mask[idx])
        }


# ═══════════════════════════════════════════════════════════════════
# 3. LOSS & TRAINING
# ═══════════════════════════════════════════════════════════════════

def correction_loss(pred, target, corrections, valid_mask, lambda_correction=0.001, target_ppb=None):
    """
    Args:
        pred: Model predictions (in ppb if model outputs ppb, else scaled)
        target: Target values in scaled space (for legacy compatibility)
        target_ppb: Target values in ppb space (for new training)
        corrections: Correction values in scaled space (for regularization)
        valid_mask: Mask for valid (non-zero) values
        lambda_correction: Regularization weight
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
        loss, loss_dict = correction_loss(pred, target, corrections, valid_mask, lambda_correction, target_ppb)

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
        'improvement': improvement,
        'n_valid': valid_pinn.numel()
    }


# ═══════════════════════════════════════════════════════════════════
# 4. LEAVE-ONE-SENSOR-OUT CV (WITH SCALER SAVING)
# ═══════════════════════════════════════════════════════════════════

def leave_one_sensor_out_cv():
    print("\n" + "="*70)
    print("🔬 LEAVE-ONE-SENSOR-OUT CROSS-VALIDATION")
    print("   WITH SPATIAL COORDINATES + SCALER SAVING")
    print("="*70)

    Path(CONFIG['save_dir']).mkdir(exist_ok=True, parents=True)
    results_all_sensors = {}

    for held_out_idx in range(9):
        print("\n" + "="*70)
        print(f"🎯 FOLD {held_out_idx + 1}/9: Holding out sensor #{held_out_idx}")
        print("="*70)

        # Load dataset
        dataset = BenzeneDataset(
            data_path=f"{CONFIG['data_dir']}sensors_final.csv",
            source_dir=f"{CONFIG['data_dir']}data_nonzero/",
            pinn_path=CONFIG['pinn_file'],
            sensor_coords_path=CONFIG['sensor_coords_file'],
            held_out_sensor_idx=held_out_idx,
            fit_scalers=True
        )

        # Split
        n_total = len(dataset)
        n_train = int(0.85 * n_total)
        n_val = n_total - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        full_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False)

        # Model - with inverse transform layer for direct ppb output
        scaler_mean = dataset.scalers['sensors'].mean_[0] if hasattr(dataset.scalers['sensors'], 'mean_') else None
        scaler_scale = dataset.scalers['sensors'].scale_[0] if hasattr(dataset.scalers['sensors'], 'scale_') else None
        
        model = NN2_CorrectionNetwork(
            n_sensors=9, 
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            output_ppb=True  # Output directly in ppb space
        ).to(CONFIG['device'])
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # Training
        print(f"\n📚 Training on 8 sensors (excluding sensor #{held_out_idx})...")
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(CONFIG['epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, CONFIG['device'], CONFIG['lambda_correction'])

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    # REMOVED: current_sensors - this was data leakage!
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
                    loss, _ = correction_loss(pred, target, corrections, valid_mask, CONFIG['lambda_correction'], target_ppb)
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{CONFIG['epochs']}: Train={train_loss:.4f}, Val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

        # ═══════════════════════════════════════════════════════════
        # SAVE MODEL + SCALERS (CRITICAL!)
        # ═══════════════════════════════════════════════════════════
        if best_model_state is not None:
            model_save_path = Path(CONFIG['save_dir']) / f"model_fold_{held_out_idx}_spatial.pth"
            scaler_save_path = Path(CONFIG['save_dir']) / f"scalers_fold_{held_out_idx}.pkl"

            try:
                # Save model checkpoint (including scaler stats for inverse transform)
                torch.save({
                    'model_state_dict': best_model_state,
                    'held_out_idx': held_out_idx,
                    'config': CONFIG,
                    'scaler_mean': dataset.scalers['sensors'].mean_[0] if hasattr(dataset.scalers['sensors'], 'mean_') else None,
                    'scaler_scale': dataset.scalers['sensors'].scale_[0] if hasattr(dataset.scalers['sensors'], 'scale_') else None,
                    'output_ppb': True  # Flag indicating model outputs in ppb space
                }, model_save_path)

                # Save scalers separately (CRITICAL!)
                with open(scaler_save_path, 'wb') as f:
                    pickle.dump(dataset.scalers, f)

                print(f"  ✓ Saved model to: {model_save_path}")
                print(f"  ✓ Saved scalers to: {scaler_save_path}")

            except Exception as e:
                print(f"  ❌ Error saving: {e}")

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Evaluate
        print(f"\n🔍 Testing on held-out sensor #{held_out_idx}...")
        held_out_results = evaluate_sensor(model, full_loader, CONFIG['device'], CONFIG['lambda_correction'], held_out_idx)

        train_sensor_results = []
        for train_idx in range(9):
            if train_idx != held_out_idx:
                train_results = evaluate_sensor(model, full_loader, CONFIG['device'], CONFIG['lambda_correction'], train_idx)
                train_sensor_results.append(train_results)

        avg_train_mae = np.mean([r['nn2_mae'] for r in train_sensor_results])
        avg_train_improvement = np.mean([r['improvement'] for r in train_sensor_results])

        results_all_sensors[held_out_idx] = {
            'held_out': held_out_results,
            'training_sensors_avg': {
                'nn2_mae': avg_train_mae,
                'improvement': avg_train_improvement
            }
        }

        # Print results
        print(f"\n{'='*70}")
        print(f"RESULTS FOR SENSOR #{held_out_idx}")
        print(f"{'='*70}")
        print(f"\n{'Metric':<30} {'Training Sensors':<20} {'Held-Out Sensor':<20}")
        print(f"{'─'*70}")
        print(f"{'NN2 MAE (ppb)':<30} {avg_train_mae:<20.4f} {held_out_results['nn2_mae']:<20.4f}")
        print(f"{'Improvement (%)':<30} {avg_train_improvement:<20.1f} {held_out_results['improvement']:<20.1f}")

        generalization_gap = held_out_results['nn2_mae'] - avg_train_mae
        generalization_pct = (generalization_gap / avg_train_mae * 100) if avg_train_mae > 0 else 0

        print(f"\n{'Spatial Generalization Gap:':<30}")
        print(f"  Absolute: {generalization_gap:+.4f} ppb")
        print(f"  Relative: {generalization_pct:+.1f}%")

        if generalization_gap < 0.05:
            print(f"  ✅ Excellent spatial generalization!")
        elif generalization_gap < 0.10:
            print(f"  ✓ Good spatial generalization")
        else:
            print(f"  ⚠️  Moderate spatial generalization")

    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)

    all_held_out_mae = [results_all_sensors[i]['held_out']['nn2_mae'] for i in range(9)]
    all_train_mae = [results_all_sensors[i]['training_sensors_avg']['nn2_mae'] for i in range(9)]

    avg_gap = np.mean(all_held_out_mae) - np.mean(all_train_mae)
    print(f"\nSpatial generalization gap: {avg_gap:+.4f} ppb")

    # Save results
    with open(f"{CONFIG['save_dir']}/leave_one_out_results_spatial.json", 'w') as f:
        json.dump(results_all_sensors, f, indent=2, default=float)

    print(f"\n📁 Results saved to: {CONFIG['save_dir']}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# 5. FINAL MASTER MODEL (WITH SCALER SAVING)
# ═══════════════════════════════════════════════════════════════════

def train_master_model():
    print("\n" + "="*70)
    print("🚀 FINAL RUN: Training Master Model")
    print("   WITH SCALER SAVING")
    print("="*70)

    # Load full dataset
    final_dataset = BenzeneDataset(
        data_path=f"{CONFIG['data_dir']}sensors_final.csv",
        source_dir=f"{CONFIG['data_dir']}data_nonzero/",
        pinn_path=CONFIG['pinn_file'],
        sensor_coords_path=CONFIG['sensor_coords_file'],
        held_out_sensor_idx=None,  # All sensors
        fit_scalers=True
    )

    final_loader = DataLoader(final_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # Model
    master_model = NN2_CorrectionNetwork(n_sensors=9).to(CONFIG['device'])
    master_optimizer = optim.AdamW(master_model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-5)

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

    # ═══════════════════════════════════════════════════════════
    # SAVE MASTER MODEL + SCALERS (CRITICAL!)
    # ═══════════════════════════════════════════════════════════
    if final_best_model_state is not None:
        master_model_path = Path(CONFIG['save_dir']) / "nn2_master_model_spatial.pth"
        master_scaler_path = Path(CONFIG['save_dir']) / "nn2_master_scalers.pkl"
        sensor_coords_path = Path(CONFIG['save_dir']) / "sensor_coordinates.npy"

        try:
            # Save model checkpoint
            torch.save({
                'model_state_dict': final_best_model_state,
                'config': CONFIG,
                'timestamp': datetime.now().isoformat()
            }, master_model_path)

            # Save scalers separately (CRITICAL FOR DEPLOYMENT!)
            with open(master_scaler_path, 'wb') as f:
                pickle.dump(final_dataset.scalers, f)

            # Save sensor coordinates
            np.save(sensor_coords_path, final_dataset.sensor_coords)

            print(f"\n{'='*70}")
            print(f"✅ SAVED MASTER MODEL FILES:")
            print(f"{'='*70}")
            print(f"  📦 Model:       {master_model_path}")
            print(f"  📊 Scalers:     {master_scaler_path}")
            print(f"  📍 Coordinates: {sensor_coords_path}")

            # Print scaler statistics for verification
            print(f"\n📊 Scaler Statistics (for deployment reference):")
            print(f"{'─'*70}")
            print(f"  Sensors:   mean={final_dataset.scalers['sensors'].mean_[0]:.4f}, std={final_dataset.scalers['sensors'].scale_[0]:.4f}")
            print(f"  PINN:      mean={final_dataset.scalers['pinn'].mean_[0]:.4f}, std={final_dataset.scalers['pinn'].scale_[0]:.4f}")
            print(f"  Wind U:    mean={final_dataset.scalers['wind'].mean_[0]:.4f}, std={final_dataset.scalers['wind'].scale_[0]:.4f}")
            print(f"  Wind V:    mean={final_dataset.scalers['wind'].mean_[1]:.4f}, std={final_dataset.scalers['wind'].scale_[1]:.4f}")
            print(f"  Diffusion: mean={final_dataset.scalers['diffusion'].mean_[0]:.4f}, std={final_dataset.scalers['diffusion'].scale_[0]:.4f}")
            print(f"  Coords X:  mean={final_dataset.scalers['coords'].mean_[0]:.4f}, std={final_dataset.scalers['coords'].scale_[0]:.4f}")
            print(f"  Coords Y:  mean={final_dataset.scalers['coords'].mean_[1]:.4f}, std={final_dataset.scalers['coords'].scale_[1]:.4f}")
            print(f"{'='*70}\n")

            # Test loading (verification)
            print(f"🔍 Verifying saved files...")
            with open(master_scaler_path, 'rb') as f:
                loaded_scalers = pickle.load(f)
            print(f"  ✓ Scalers loaded successfully")
            print(f"  ✓ Scaler keys: {list(loaded_scalers.keys())}")

            loaded_coords = np.load(sensor_coords_path)
            print(f"  ✓ Coordinates loaded: shape {loaded_coords.shape}")

        except Exception as e:
            print(f"\n❌ ERROR SAVING FILES:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️ Warning: No best model state to save!")

    print(f"\n{'='*70}")
    print(f"✅ MASTER MODEL TRAINING COMPLETE")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════
# 6. DEPLOYMENT HELPER: LOAD SCALERS
# ═══════════════════════════════════════════════════════════════════

def load_scalers_for_deployment(scaler_path):
    """
    Load scalers for deployment/inference

    Usage:
        scalers = load_scalers_for_deployment('models/nn2_master_scalers.pkl')

        # Normalize inputs
        wind_norm = scalers['wind'].transform([[wind_u, wind_v]])
        pinn_norm = scalers['pinn'].transform([[pinn_pred]])
        ...
    """
    print(f"\n{'='*70}")
    print(f"📦 LOADING SCALERS FOR DEPLOYMENT")
    print(f"{'='*70}")

    try:
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)

        print(f"✓ Loaded scalers from: {scaler_path}")
        print(f"  Available scalers: {list(scalers.keys())}")

        # Print statistics
        print(f"\n📊 Scaler Statistics:")
        print(f"{'─'*70}")
        print(f"  Sensors:   mean={scalers['sensors'].mean_[0]:.4f}, std={scalers['sensors'].scale_[0]:.4f}")
        print(f"  PINN:      mean={scalers['pinn'].mean_[0]:.4f}, std={scalers['pinn'].scale_[0]:.4f}")
        print(f"  Wind U:    mean={scalers['wind'].mean_[0]:.4f}, std={scalers['wind'].scale_[0]:.4f}")
        print(f"  Wind V:    mean={scalers['wind'].mean_[1]:.4f}, std={scalers['wind'].scale_[1]:.4f}")
        print(f"  Diffusion: mean={scalers['diffusion'].mean_[0]:.4f}, std={scalers['diffusion'].scale_[0]:.4f}")
        print(f"  Coords X:  mean={scalers['coords'].mean_[0]:.4f}, std={scalers['coords'].scale_[0]:.4f}")
        print(f"  Coords Y:  mean={scalers['coords'].mean_[1]:.4f}, std={scalers['coords'].scale_[1]:.4f}")
        print(f"{'='*70}\n")

        return scalers

    except FileNotFoundError:
        print(f"❌ ERROR: Scaler file not found: {scaler_path}")
        print(f"   Make sure you've run train_master_model() first!")
        return None
    except Exception as e:
        print(f"❌ ERROR loading scalers: {e}")
        import traceback
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════
# 7. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Removed sys.argv parsing to avoid '-f' issue
    # Explicitly call the 'all' mode to run both CV and master training.
    print("Running full pipeline (Leave-One-Sensor-Out CV and Master Model Training)...")
    leave_one_sensor_out_cv()
    train_master_model()

    #Running full pipeline (Leave-One-Sensor-Out CV and Master Model Training)...

======================================================================
🔬 LEAVE-ONE-SENSOR-OUT CROSS-VALIDATION
   WITH SPATIAL COORDINATES + SCALER SAVING
======================================================================

======================================================================
🎯 FOLD 1/9: Holding out sensor #0
======================================================================

======================================================================
🔄 Loading Data - HOLDING OUT Sensor #0
======================================================================

📍 Loading sensor coordinates from /content/data/sensor_coordinates.csv...
   Loaded coordinates for 9 sensors

✓ Sensor data: 5920 rows

📊 Loading PINN predictions...

🎯 Training sensors: 8
🔒 Held-out sensor: sensor_482010026 (index 0)

🔄 Processing features...
  Common timestamps: 5173

  🔧 Fitting scalers...

  📊 Scaler Statistics:
     Sensors:   mean=0.4563, std=2.7778
     PINN:      mean=0.3653, std=0.4558
     Wind U:    mean=-0.8118, std=1.5106
     Wind V:    mean=0.4695, std=1.7832
     Diffusion: mean=26.9593, std=32.5261
     Coords X:  mean=-590.9556, std=9746.5739
     Coords Y:  mean=-1276.3111, std=4904.3063

======================================================================
✓ Dataset ready: 5173 samples
======================================================================


📚 Training on 8 sensors (excluding sensor #0)...
  Epoch  10/50: Train=0.7919, Val=0.1738
  Epoch  20/50: Train=0.7081, Val=0.2539
  Epoch  30/50: Train=0.6372, Val=0.2128
  Epoch  40/50: Train=0.6444, Val=0.1385
  Epoch  50/50: Train=0.6070, Val=0.1908
  ✓ Saved model to: /content/models/leave_one_out/model_fold_0_spatial.pth
  ✓ Saved scalers to: /content/models/leave_one_out/scalers_fold_0.pkl

🔍 Testing on held-out sensor #0...

======================================================================
RESULTS FOR SENSOR #0
======================================================================

Metric                         Training Sensors     Held-Out Sensor     
──────────────────────────────────────────────────────────────────────
NN2 MAE (ppb)                  0.1552               0.1699              
Improvement (%)                73.9                 73.8                

Spatial Generalization Gap:   
  Absolute: +0.0146 ppb
  Relative: +9.4%
  ✅ Excellent spatial generalization!

======================================================================
🎯 FOLD 2/9: Holding out sensor #1
======================================================================

======================================================================
🔄 Loading Data - HOLDING OUT Sensor #1
======================================================================

📍 Loading sensor coordinates from /content/data/sensor_coordinates.csv...
   Loaded coordinates for 9 sensors

✓ Sensor data: 5920 rows

📊 Loading PINN predictions...

🎯 Training sensors: 8
🔒 Held-out sensor: sensor_482010057 (index 1)

🔄 Processing features...
  Common timestamps: 5173

  🔧 Fitting scalers...

  📊 Scaler Statistics:
     Sensors:   mean=0.4024, std=2.9919
     PINN:      mean=0.3952, std=0.4620
     Wind U:    mean=-0.8118, std=1.5106
     Wind V:    mean=0.4695, std=1.7832
     Diffusion: mean=26.9593, std=32.5261
     Coords X:  mean=-590.9556, std=9746.5739
     Coords Y:  mean=-1276.3111, std=4904.3063

======================================================================
✓ Dataset ready: 5173 samples
======================================================================


📚 Training on 8 sensors (excluding sensor #1)...
  Epoch  10/50: Train=0.7944, Val=0.2461
  Epoch  20/50: Train=0.6772, Val=0.1393
  Epoch  30/50: Train=0.6305, Val=0.0868
  Epoch  40/50: Train=0.6422, Val=0.1027
  Epoch  50/50: Train=0.6378, Val=0.0727
  ✓ Saved model to: /content/models/leave_one_out/model_fold_1_spatial.pth
  ✓ Saved scalers to: /content/models/leave_one_out/scalers_fold_1.pkl

🔍 Testing on held-out sensor #1...

======================================================================
RESULTS FOR SENSOR #1
======================================================================

Metric                         Training Sensors     Held-Out Sensor     
──────────────────────────────────────────────────────────────────────
NN2 MAE (ppb)                  0.1692               0.1771              
Improvement (%)                71.5                 74.8                

Spatial Generalization Gap:   
  Absolute: +0.0079 ppb
  Relative: +4.7%
  ✅ Excellent spatial generalization!

======================================================================
🎯 FOLD 3/9: Holding out sensor #2
======================================================================

======================================================================
🔄 Loading Data - HOLDING OUT Sensor #2
======================================================================

📍 Loading sensor coordinates from /content/data/sensor_coordinates.csv...
   Loaded coordinates for 9 sensors

✓ Sensor data: 5920 rows

📊 Loading PINN predictions...

🎯 Training sensors: 8
🔒 Held-out sensor: sensor_482010069 (index 2)

🔄 Processing features...
  Common timestamps: 5173

  🔧 Fitting scalers...

  📊 Scaler Statistics:
     Sensors:   mean=0.5086, std=3.1632
     PINN:      mean=0.3785, std=0.4388
     Wind U:    mean=-0.8118, std=1.5106
     Wind V:    mean=0.4695, std=1.7832
     Diffusion: mean=26.9593, std=32.5261
     Coords X:  mean=-590.9556, std=9746.5739
     Coords Y:  mean=-1276.3111, std=4904.3063

======================================================================
✓ Dataset ready: 5173 samples
======================================================================


📚 Training on 8 sensors (excluding sensor #2)...
  Epoch  10/50: Train=0.7655, Val=0.1471
  Epoch  20/50: Train=0.7090, Val=0.1188
  Epoch  30/50: Train=0.6773, Val=0.1046
  Epoch  40/50: Train=0.6507, Val=0.0876
  Epoch  50/50: Train=0.6359, Val=0.1466
  ✓ Saved model to: /content/models/leave_one_out/model_fold_2_spatial.pth
  ✓ Saved scalers to: /content/models/leave_one_out/scalers_fold_2.pkl

🔍 Testing on held-out sensor #2...

======================================================================
RESULTS FOR SENSOR #2
======================================================================

Metric                         Training Sensors     Held-Out Sensor     
──────────────────────────────────────────────────────────────────────
NN2 MAE (ppb)                  0.2289               0.2241              
Improvement (%)                62.5                 65.7                

Spatial Generalization Gap:   
  Absolute: -0.0048 ppb
  Relative: -2.1%
  ✅ Excellent spatial generalization!

======================================================================
🎯 FOLD 4/9: Holding out sensor #3
======================================================================

======================================================================
🔄 Loading Data - HOLDING OUT Sensor #3
======================================================================

📍 Loading sensor coordinates from /content/data/sensor_coordinates.csv...
   Loaded coordinates for 9 sensors

✓ Sensor data: 5920 rows

📊 Loading PINN predictions...

🎯 Training sensors: 8
🔒 Held-out sensor: sensor_482010617 (index 3)

🔄 Processing features...
  Common timestamps: 5173

  🔧 Fitting scalers...

  📊 Scaler Statistics:
     Sensors:   mean=0.5105, std=3.1589
     PINN:      mean=0.3570, std=0.4538
     Wind U:    mean=-0.8118, std=1.5106
     Wind V:    mean=0.4695, std=1.7832
     Diffusion: mean=26.9593, std=32.5261
     Coords X:  mean=-590.9556, std=9746.5739
     Coords Y:  mean=-1276.3111, std=4904.3063

======================================================================
✓ Dataset ready: 5173 samples
======================================================================


📚 Training on 8 sensors (excluding sensor #3)...
  Epoch  10/50: Train=0.6115, Val=0.1434
  Epoch  20/50: Train=0.5420, Val=0.1266
  Epoch  30/50: Train=0.5434, Val=0.1786
  Epoch  40/50: Train=0.5019, Val=0.1051
  Epoch  50/50: Train=0.4841, Val=0.1541
  ✓ Saved model to: /content/models/leave_one_out/model_fold_3_spatial.pth
  ✓ Saved scalers to: /content/models/leave_one_out/scalers_fold_3.pkl

🔍 Testing on held-out sensor #3...

======================================================================
RESULTS FOR SENSOR #3
======================================================================

Metric                         Training Sensors     Held-Out Sensor     
──────────────────────────────────────────────────────────────────────
NN2 MAE (ppb)                  0.1515               0.1779              
Improvement (%)                74.2                 73.3                

Spatial Generalization Gap:   
  Absolute: +0.0264 ppb
  Relative: +17.5%
  ✅ Excellent spatial generalization!

======================================================================
🎯 FOLD 5/9: Holding out sensor #4
======================================================================

======================================================================
🔄 Loading Data - HOLDING OUT Sensor #4
======================================================================

📍 Loading sensor coordinates from /content/data/sensor_coordinates.csv...
   Loaded coordinates for 9 sensors

✓ Sensor data: 5920 rows

📊 Loading PINN predictions...

🎯 Training sensors: 8
🔒 Held-out sensor: sensor_482010803 (index 4)

🔄 Processing features...
  Common timestamps: 5173

  🔧 Fitting scalers...

  📊 Scaler Statistics:
     Sensors:   mean=0.4747, std=3.1328
     PINN:      mean=0.3918, std=0.4665
     Wind U:    mean=-0.8118, std=1.5106
     Wind V:    mean=0.4695, std=1.7832
     Diffusion: mean=26.9593, std=32.5261
     Coords X:  mean=-590.9556, std=9746.5739
     Coords Y:  mean=-1276.3111, std=4904.3063

======================================================================
✓ Dataset ready: 5173 samples
======================================================================


📚 Training on 8 sensors (excluding sensor #4)...
  Epoch  10/50: Train=0.6852, Val=0.4282
  Epoch  20/50: Train=0.5751, Val=0.5629
  Epoch  30/50: Train=0.5325, Val=0.2385
  Epoch  40/50: Train=0.5210, Val=0.1988
  Epoch  50/50: Train=0.5147, Val=0.2282
  ✓ Saved model to: /content/models/leave_one_out/model_fold_4_spatial.pth
  ✓ Saved scalers to: /content/models/leave_one_out/scalers_fold_4.pkl

🔍 Testing on held-out sensor #4...

======================================================================
RESULTS FOR SENSOR #4
======================================================================

Metric                         Training Sensors     Held-Out Sensor     
──────────────────────────────────────────────────────────────────────
NN2 MAE (ppb)                  0.2660               0.1869              
Improvement (%)                57.0                 57.9                

Spatial Generalization Gap:   
  Absolute: -0.0791 ppb
  Relative: -29.7%
  ✅ Excellent spatial generalization!

======================================================================
🎯 FOLD 6/9: Holding out sensor #5
======================================================================

======================================================================
🔄 Loading Data - HOLDING OUT Sensor #5
======================================================================

📍 Loading sensor coordinates from /content/data/sensor_coordinates.csv...
   Loaded coordinates for 9 sensors

✓ Sensor data: 5920 rows

📊 Loading PINN predictions...

🎯 Training sensors: 8
🔒 Held-out sensor: sensor_482011015 (index 5)

🔄 Processing features...
  Common timestamps: 5173

  🔧 Fitting scalers...

  📊 Scaler Statistics:
     Sensors:   mean=0.4078, std=2.2816
     PINN:      mean=0.3766, std=0.4661
     Wind U:    mean=-0.8118, std=1.5106
     Wind V:    mean=0.4695, std=1.7832
     Diffusion: mean=26.9593, std=32.5261
     Coords X:  mean=-590.9556, std=9746.5739
     Coords Y:  mean=-1276.3111, std=4904.3063

======================================================================
✓ Dataset ready: 5173 samples
======================================================================


📚 Training on 8 sensors (excluding sensor #5)...
  Epoch  10/50: Train=1.2317, Val=0.2046
  Epoch  20/50: Train=1.1225, Val=0.1447
  Epoch  30/50: Train=1.1177, Val=0.1353
  Epoch  40/50: Train=1.0833, Val=0.1421
  Epoch  50/50: Train=1.1149, Val=0.1303
  ✓ Saved model to: /content/models/leave_one_out/model_fold_5_spatial.pth
  ✓ Saved scalers to: /content/models/leave_one_out/scalers_fold_5.pkl

🔍 Testing on held-out sensor #5...

======================================================================
RESULTS FOR SENSOR #5
======================================================================

Metric                         Training Sensors     Held-Out Sensor     
──────────────────────────────────────────────────────────────────────
NN2 MAE (ppb)                  0.1995               0.1722              
Improvement (%)                68.2                 70.2                

Spatial Generalization Gap:   
  Absolute: -0.0273 ppb
  Relative: -13.7%
  ✅ Excellent spatial generalization!

======================================================================
🎯 FOLD 7/9: Holding out sensor #6
======================================================================

======================================================================
🔄 Loading Data - HOLDING OUT Sensor #6
======================================================================

📍 Loading sensor coordinates from /content/data/sensor_coordinates.csv...
   Loaded coordinates for 9 sensors

✓ Sensor data: 5920 rows

📊 Loading PINN predictions...

🎯 Training sensors: 8
🔒 Held-out sensor: sensor_482011035 (index 6)

🔄 Processing features...
  Common timestamps: 5173

  🔧 Fitting scalers...

  📊 Scaler Statistics:
     Sensors:   mean=0.4908, std=3.1511
     PINN:      mean=0.3949, std=0.4610
     Wind U:    mean=-0.8118, std=1.5106
     Wind V:    mean=0.4695, std=1.7832
     Diffusion: mean=26.9593, std=32.5261
     Coords X:  mean=-590.9556, std=9746.5739
     Coords Y:  mean=-1276.3111, std=4904.3063

======================================================================
✓ Dataset ready: 5173 samples
======================================================================


📚 Training on 8 sensors (excluding sensor #6)...
  Epoch  10/50: Train=0.7809, Val=0.1177
  Epoch  20/50: Train=0.6870, Val=0.1019
  Epoch  30/50: Train=0.6620, Val=0.0962
  Epoch  40/50: Train=0.6424, Val=0.2039
  Epoch  50/50: Train=0.6355, Val=0.0887
  ✓ Saved model to: /content/models/leave_one_out/model_fold_6_spatial.pth
  ✓ Saved scalers to: /content/models/leave_one_out/scalers_fold_6.pkl

🔍 Testing on held-out sensor #6...

======================================================================
RESULTS FOR SENSOR #6
======================================================================

Metric                         Training Sensors     Held-Out Sensor     
──────────────────────────────────────────────────────────────────────
NN2 MAE (ppb)                  0.1575               0.1166              
Improvement (%)                73.9                 77.9                

Spatial Generalization Gap:   
  Absolute: -0.0408 ppb
  Relative: -25.9%
  ✅ Excellent spatial generalization!

======================================================================
🎯 FOLD 8/9: Holding out sensor #7
======================================================================

======================================================================
🔄 Loading Data - HOLDING OUT Sensor #7
======================================================================

📍 Loading sensor coordinates from /content/data/sensor_coordinates.csv...
   Loaded coordinates for 9 sensors

✓ Sensor data: 5920 rows

📊 Loading PINN predictions...

🎯 Training sensors: 8
🔒 Held-out sensor: sensor_482011039 (index 7)

🔄 Processing features...
  Common timestamps: 5173

  🔧 Fitting scalers...

  📊 Scaler Statistics:
     Sensors:   mean=0.4678, std=2.8522
     PINN:      mean=0.3938, std=0.4647
     Wind U:    mean=-0.8118, std=1.5106
     Wind V:    mean=0.4695, std=1.7832
     Diffusion: mean=26.9593, std=32.5261
     Coords X:  mean=-590.9556, std=9746.5739
     Coords Y:  mean=-1276.3111, std=4904.3063

======================================================================
✓ Dataset ready: 5173 samples
======================================================================


📚 Training on 8 sensors (excluding sensor #7)...
  Epoch  10/50: Train=0.9656, Val=0.1034
  Epoch  20/50: Train=0.8418, Val=0.1215
  Epoch  30/50: Train=0.8076, Val=0.2348
  Epoch  40/50: Train=0.8150, Val=0.1326
  Epoch  50/50: Train=0.8052, Val=0.1042
  ✓ Saved model to: /content/models/leave_one_out/model_fold_7_spatial.pth
  ✓ Saved scalers to: /content/models/leave_one_out/scalers_fold_7.pkl

🔍 Testing on held-out sensor #7...

======================================================================
RESULTS FOR SENSOR #7
======================================================================

Metric                         Training Sensors     Held-Out Sensor     
──────────────────────────────────────────────────────────────────────
NN2 MAE (ppb)                  0.1565               0.1254              
Improvement (%)                74.5                 75.6                

Spatial Generalization Gap:   
  Absolute: -0.0311 ppb
  Relative: -19.9%
  ✅ Excellent spatial generalization!

======================================================================
🎯 FOLD 9/9: Holding out sensor #8
======================================================================

======================================================================
🔄 Loading Data - HOLDING OUT Sensor #8
======================================================================

📍 Loading sensor coordinates from /content/data/sensor_coordinates.csv...
   Loaded coordinates for 9 sensors

✓ Sensor data: 5920 rows

📊 Loading PINN predictions...

🎯 Training sensors: 8
🔒 Held-out sensor: sensor_482016000 (index 8)

🔄 Processing features...
  Common timestamps: 5173

  🔧 Fitting scalers...

  📊 Scaler Statistics:
     Sensors:   mean=0.5038, std=3.1733
     PINN:      mean=0.3595, std=0.3312
     Wind U:    mean=-0.8118, std=1.5106
     Wind V:    mean=0.4695, std=1.7832
     Diffusion: mean=26.9593, std=32.5261
     Coords X:  mean=-590.9556, std=9746.5739
     Coords Y:  mean=-1276.3111, std=4904.3063

======================================================================
✓ Dataset ready: 5173 samples
======================================================================


📚 Training on 8 sensors (excluding sensor #8)...
  Epoch  10/50: Train=0.9887, Val=0.2823
  Epoch  20/50: Train=0.9319, Val=0.1882
  Epoch  30/50: Train=0.8419, Val=0.1790
  Epoch  40/50: Train=0.9103, Val=0.1452
  Epoch  50/50: Train=0.8022, Val=0.1341
  ✓ Saved model to: /content/models/leave_one_out/model_fold_8_spatial.pth
  ✓ Saved scalers to: /content/models/leave_one_out/scalers_fold_8.pkl

🔍 Testing on held-out sensor #8...

======================================================================
RESULTS FOR SENSOR #8
======================================================================

Metric                         Training Sensors     Held-Out Sensor     
──────────────────────────────────────────────────────────────────────
NN2 MAE (ppb)                  0.2092               0.3282              
Improvement (%)                71.8                 72.2                

Spatial Generalization Gap:   
  Absolute: +0.1190 ppb
  Relative: +56.9%
  ⚠️  Moderate spatial generalization

======================================================================
📊 FINAL SUMMARY
======================================================================

Spatial generalization gap: -0.0017 ppb

📁 Results saved to: /content/models/leave_one_out/
======================================================================


======================================================================
🚀 FINAL RUN: Training Master Model
   WITH SCALER SAVING
======================================================================

======================================================================
🔄 Loading Data - Using ALL Sensors
======================================================================

📍 Loading sensor coordinates from /content/data/sensor_coordinates.csv...
   Loaded coordinates for 9 sensors

✓ Sensor data: 5920 rows

📊 Loading PINN predictions...

🔄 Processing features...
  Common timestamps: 5173

  🔧 Fitting scalers...

  📊 Scaler Statistics:
     Sensors:   mean=0.4689, std=2.9754
     PINN:      mean=0.3792, std=0.4466
     Wind U:    mean=-0.8118, std=1.5106
     Wind V:    mean=0.4695, std=1.7832
     Diffusion: mean=26.9593, std=32.5261
     Coords X:  mean=-590.9556, std=9746.5739
     Coords Y:  mean=-1276.3111, std=4904.3063

======================================================================
✓ Dataset ready: 5173 samples
======================================================================


📚 Training master model for 50 epochs...
  Epoch  10/50: Train Loss=0.8025
  Epoch  20/50: Train Loss=0.7107
  Epoch  30/50: Train Loss=0.5824
  Epoch  40/50: Train Loss=0.5234
  Epoch  50/50: Train Loss=0.4520

======================================================================
✅ SAVED MASTER MODEL FILES:
======================================================================
  📦 Model:       /content/models/leave_one_out/nn2_master_model_spatial.pth
  📊 Scalers:     /content/models/leave_one_out/nn2_master_scalers.pkl
  📍 Coordinates: /content/models/leave_one_out/sensor_coordinates.npy

📊 Scaler Statistics (for deployment reference):
──────────────────────────────────────────────────────────────────────
  Sensors:   mean=0.4689, std=2.9754
  PINN:      mean=0.3792, std=0.4466
  Wind U:    mean=-0.8118, std=1.5106
  Wind V:    mean=0.4695, std=1.7832
  Diffusion: mean=26.9593, std=32.5261
  Coords X:  mean=-590.9556, std=9746.5739
  Coords Y:  mean=-1276.3111, std=4904.3063
======================================================================

🔍 Verifying saved files...
  ✓ Scalers loaded successfully
  ✓ Scaler keys: ['sensors', 'pinn', 'wind', 'diffusion', 'coords']
  ✓ Coordinates loaded: shape (9, 2)

======================================================================
✅ MASTER MODEL TRAINING COMPLETE
======================================================================
