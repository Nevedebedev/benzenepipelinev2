
import torch
import os

path = "/Users/neevpratap/simpletesting/pinn_combined_final.pth 2"

try:
    print(f"Inspecting {path}...")
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    # If it's a state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    buffers = ['x_min', 'x_max', 'y_min', 'y_max', 'cx_min', 'cx_max', 'cy_min', 'cy_max']
    for b in buffers:
        if b in state_dict:
            print(f"  {b}: {state_dict[b].item()}")
        else:
            print(f"  {b}: NOT FOUND")
            
except Exception as e:
    print(f"Error: {e}")
