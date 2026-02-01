
import os
import pandas as pd

# Directory containing the downloaded files
data_dir = "/Users/neevpratap/Downloads/drive-download-20260131T175758Z-3-001"

# List all CSV files
files = [f for f in os.listdir(data_dir) if f.endswith("_synced_training_data.csv")]

print("Processing files to extract source coordinates...")
results = []

for f in files:
    path = os.path.join(data_dir, f)
    try:
        # Read just the first row to get the metadata which is constant for the source
        df = pd.read_csv(path, nrows=1)
        
        # Extract facility name from filename (remove _synced_training_data.csv)
        facility_name = f.replace("_synced_training_data.csv", "")
        
        # Check for cartesian and lat/long columns
        if 'source_x_cartesian' in df.columns and 'source_y_cartesian' in df.columns:
            cx = df['source_x_cartesian'].values[0]
            cy = df['source_y_cartesian'].values[0]
            lat = df['source_x'].values[0]
            lon = df['source_y'].values[0]
            diam = df['source_diameter'].values[0] if 'source_diameter' in df.columns else 0
            
            results.append({
                "name": facility_name,
                "lat": float(lat),
                "lon": float(lon), 
                "x": float(cx),
                "y": float(cy),
                "diameter": float(diam)
            })
        else:
            print(f"WARNING: {f} missing cartesian columns")
            
    except Exception as e:
        print(f"Error processing {f}: {e}")

# Sort by name for consistency
results.sort(key=lambda x: x['name'])

with open('facilities_dump.txt', 'w') as f:
    f.write("FACILITIES = [\n")
    for r in results:
        # Format: Name, Lat, Lon, X, Y
        f.write(f"    {{'name': '{r['name']}', 'lat_lon': ({r['lat']}, {r['lon']}), 'coords': ({r['x']}, {r['y']}), 'diameter': {r['diameter']}}},\n")
    f.write("]\n")

print("Dumped to facilities_dump.txt")

# Print for User View
print(f"{'Facility Name':<40} | {'Lat':<10} | {'Lon':<10} | {'Cart X':<10} | {'Cart Y':<10}")
print("-" * 90)
for r in results:
    print(f"{r['name']:<40} | {r['lat']:<10.4f} | {r['lon']:<10.4f} | {r['x']:<10.1f} | {r['y']:<10.1f}")
