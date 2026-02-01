
import pickle
import os

scaler_path = "/Users/neevpratap/simpletesting/nn2_master_scalers.pkl"

if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    print("Scalers keys:", list(scalers.keys()))
    for k, v in scalers.items():
        try:
            print(f"Scaler '{k}': mean={v.mean_}")
        except:
             print(f"Scaler '{k}': {type(v)}")
else:
    print("Scalers file not found.")
