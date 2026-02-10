import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class LocalDataset(Dataset):
    def __init__(self, feature_dir, proto_path, partition="train"):
        self.feature_dir = feature_dir
        self.partition = partition
        
        # Load Proto
        print(f"Loading protocol: {proto_path}")
        try:
            self.df = pd.read_csv(proto_path, sep='\s+', header=None, engine='python')
            if len(self.df.columns) >= 2:
                 self.df.rename(columns={1: 'utterance_id', 8: 'label'}, inplace=True)
        except Exception as e:
            print(f"Error loading proto: {e}")
            self.df = pd.DataFrame() # Empty
            
        self.file_list = []
        self.labels = []
        
        # Parse
        print("Parsing protocol...")
        valid_count = 0
        for idx, row in self.df.iterrows():
            uid = str(row['utterance_id'])
            # Label
            label_str = str(row.get('label', '')).lower()
            if 'bonafide' in label_str:
                lbl = 0.0
            else:
                lbl = 1.0 # Spoof
            
            # Check if feature exists (Strict for training)
            feat_path = os.path.join(self.feature_dir, f"{uid}.npy")
            if os.path.exists(feat_path):
                self.file_list.append(uid)
                self.labels.append(lbl)
                valid_count += 1
            
        print(f"Loaded {valid_count} ready samples (swapped from protocol) for {partition}.")
        
        # DEBUG STATS
        n_spoof = sum(self.labels)
        n_bonafide = len(self.labels) - n_spoof
        print(f"[{partition}] Bonafide: {n_bonafide}, Spoof: {n_spoof} (Ratio: {n_spoof/len(self.labels):.2f})")


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        uid = self.file_list[idx]
        label = self.labels[idx]
        
        feat_path = os.path.join(self.feature_dir, f"{uid}.npy")
        
        try:
            # Load
            features = np.load(feat_path)
            # features shape: (D,)
            return torch.from_numpy(features).float(), torch.tensor(label).float(), uid
            
        except Exception as e:
            # Return zeros if missing (robustness)
            # Assuming dim ~2346 based on extraction
            # Warn once?
            return torch.zeros(2346).float(), torch.tensor(label).float(), uid
