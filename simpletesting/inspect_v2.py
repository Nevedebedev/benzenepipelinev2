
import torch
import os

path = "/Users/neevpratap/simpletesting/nn2_master_model_spatial-2.pth"

try:
    print(f"Inspecting {path}...")
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        print("Checkpoint is a DICT. Keys:")
        print(checkpoint.keys())
        
        if 'sensor_coords' in checkpoint:
            print("FOUND SENSOR_COORDS in checkpoint!")
            print(checkpoint['sensor_coords'])
        else:
            print("NO SENSOR_COORDS found in checkpoint.")
    else:
        print("Checkpoint is not a dict.")
        
except Exception as e:
    print(f"Error: {e}")
