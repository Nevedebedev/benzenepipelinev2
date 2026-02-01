
import torch
import os
import sys

def inspect_model(path):
    print(f"Inspecting {path}...")
    try:
        # Load with map_location to cpu to avoid cuda errors if on mac
        data = torch.load(path, map_location=torch.device('cpu'))
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print("Keys:", data.keys())
            # optional: check for model structure if it's a checkpoint dict
            if 'model_state_dict' in data:
                print("Found model_state_dict")
        else:
            print("Object is not a dict (likely a full model or other object)")
            print(data)
    except Exception as e:
        print(f"Error loading {path}: {e}")

simple_dir = "/Users/neevpratap/simpletesting"
files = [f for f in os.listdir(simple_dir) if f.endswith('.pth') or f.endswith('.pth 2')]

for f in files:
    inspect_model(os.path.join(simple_dir, f))
