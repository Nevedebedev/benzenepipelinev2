
import os
import csv
import json

data_dir = "/Users/neevpratap/simpletesting/training_data_2021_march"
sources = []

files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
files.sort()

print(f"Found {len(files)} files.")

for f in files:
    path = os.path.join(data_dir, f)
    with open(path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        # Get first row
        try:
            row = next(reader)
            source_info = {
                "name": f.replace("_training_data.csv", ""),
                "dataset_file": f,
                "source_x": float(row['source_x']),
                "source_y": float(row['source_y']),
                "source_x_cartesian": float(row['source_x_cartesian']),
                "source_y_cartesian": float(row['source_y_cartesian']),
                "source_diameter": float(row['source_diameter']),
                # "Q_total": float(row['Q_total']) # Q might vary? Let's keep it static for metadata if it's source property
            }
            sources.append(source_info)
        except StopIteration:
            print(f"Empty file: {f}")

print(json.dumps(sources, indent=2))
