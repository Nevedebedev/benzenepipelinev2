
import torch
import os
import sys

path = "/Users/neevpratap/simpletesting/nn2_master_model_spatial.pth"

try:
    print(f"Inspecting {path}...")
    checkpoint = torch.load(path, map_location='cpu', weights_only=False) # Safe loading
    
    if isinstance(checkpoint, dict):
        print("Checkpoint is a DICT. Keys:")
        print(checkpoint.keys())
        
        if 'scalers' in checkpoint:
            print("FOUND SCALERS in checkpoint!")
            print(checkpoint['scalers'].keys())
        else:
            print("NO SCALERS found in checkpoint.")
            
        if 'model_state_dict' in checkpoint:
            sd = checkpoint['model_state_dict']
            first_layer = list(sd.keys())[0]
            print(f"First layer ({first_layer}) shape: {sd[first_layer].shape}")
    else:
        print("Checkpoint is direct model object or state dict (not the new dict format).")
        
except Exception as e:
    print(f"Error loading: {e}")
