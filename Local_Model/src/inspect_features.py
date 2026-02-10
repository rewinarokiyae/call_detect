import os
import numpy as np
import glob
import random

TRAIN_DIR = r"D:\Subject\honors\ai_project\call_detect\data\precomputed\train"
DEV_DIR = r"D:\Subject\honors\ai_project\call_detect\data\precomputed\dev"

def inspect_dir(name, path):
    print(f"\n--- Inspecting {name} ---")
    files = glob.glob(os.path.join(path, "*.npy"))
    if not files:
        print("No files found!")
        return
    
    print(f"Total files: {len(files)}")
    
    # Sample 10 files
    samples = random.sample(files, min(10, len(files)))
    
    all_means = []
    all_stds = []
    
    for f in samples:
        try:
            data = np.load(f)
            # data shape: (2346,)
            
            if np.all(data == 0):
                print(f"WARNING: {os.path.basename(f)} is ALL ZEROS")
            
            if np.isnan(data).any():
                print(f"WARNING: {os.path.basename(f)} contains NaNs")
                
            mean = data.mean()
            std = data.std()
            
            all_means.append(mean)
            all_stds.append(std)
            
            print(f"{os.path.basename(f)}: Shape={data.shape}, Mean={mean:.4f}, Std={std:.4f}, Min={data.min():.4f}, Max={data.max():.4f}")
            
        except Exception as e:
            print(f"Error loading {f}: {e}")

    avg_mean = np.mean(all_means)
    avg_std = np.mean(all_stds)
    print(f">> {name} Average Stats: Mean={avg_mean:.4f}, Std={avg_std:.4f}")

def main():
    inspect_dir("TRAIN", TRAIN_DIR)
    inspect_dir("DEV", DEV_DIR)

if __name__ == "__main__":
    main()
