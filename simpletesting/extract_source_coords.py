
import pandas as pd
import os
import glob

drive_dir = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001"
output = []

files = glob.glob(os.path.join(drive_dir, "*_synced_training_data.csv"))
for f in sorted(files):
    name = os.path.basename(f).replace("_synced_training_data.csv", "")
    df = pd.read_csv(f, nrows=1)
    if 'source_x_cartesian' in df.columns:
        x = df['source_x_cartesian'].iloc[0]
        y = df['source_y_cartesian'].iloc[0]
        d = df['source_diameter'].iloc[0]
        output.append({'name': name, 'coords': (float(x), float(y)), 'diameter': float(d)})

import pprint
pprint.pprint(output)
