import os
import torch
import soundfile as sf
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ASVspoof5Dataset(Dataset):
    def __init__(self, config, partition="train", transform=None):
        self.config = config
        self.partition = partition
        self.transform = transform
        
        # Determine paths
        # Check if running on Vast or Local (Logic: check if Vast path exists)
        if os.path.exists(config['data']['root_vast']):
            self.root = config['data']['root_vast']
        else:
            self.root = config['data']['root_local']
            
        self.sample_rate = config['data']['sample_rate']
        self.max_len = config['data']['max_len']
        
        # Select Protocol and Audio Dir
        if partition == "train":
            proto_file = config['data']['train_proto']
            self.audio_dir = os.path.join(self.root, "flac_T")
        elif partition == "dev":
            proto_file = config['data']['dev_proto']
            self.audio_dir = os.path.join(self.root, "flac_D")
        elif partition == "eval":
            proto_file = config['data']['eval_proto']
            self.audio_dir = os.path.join(self.root, "flac_E_eval")
        else:
            raise ValueError(f"Unknown partition: {partition}")
            
        self.proto_path = os.path.join(self.root, proto_file)
        
        # Load Protocol
        # Assuming Format: [Speaker_ID, File_ID, ..., Label] or similar
        # We rely on File_ID (usually col 1) and Label (last col or specific string)
        self.file_list = []
        self.labels = []
        
        if os.path.exists(self.proto_path):
            with open(self.proto_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                    
                    filename = parts[1] # FileID is usually 2nd column
                    # ASVspoof5 labels might vary, checking for 'bonafide' vs 'spoof'
                    # Usually last column or key column
                    # For Track 1: bonafide vs all spoofs
                    
                    # Heuristic for label: search for 'bonafide'
                    label_str = parts[-1].lower() 
                    if 'bonafide' in line.lower():
                        label = 0
                    else:
                        label = 1
                        
                    self.file_list.append(filename)
                    self.labels.append(label)
        else:
            print(f"WARNING: Protocol not found at {self.proto_path}. Dataset empty.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        label = self.labels[idx]
        
        # Audio Path
        # Try .flac
        filepath = os.path.join(self.audio_dir, filename + ".flac")
        if not os.path.exists(filepath):
             # Try .wav just in case
             filepath = os.path.join(self.audio_dir, filename + ".wav")
        
        try:
            audio, sr = sf.read(filepath)
            # Resample if needed (simplistic check)
            if sr != self.sample_rate:
                # In production, use torchaudio.transforms.Resample
                # For now assume mostly 16k or handle offline
                pass 
                
        except Exception as e:
            # Return dummy if file fails (to prevent crash)
            print(f"Error loading {filepath}: {e}")
            audio = np.zeros(self.max_len)

        # Pad / Crop
        # Ensure audio is (1, L) tensor
        audio_len = len(audio)
        if audio_len < self.max_len:
            # Pad
            pad_len = self.max_len - audio_len
            audio = np.pad(audio, (0, pad_len), mode='wrap')
        elif audio_len > self.max_len:
            # Crop (random or center? Center for dev, random for train)
            if self.partition == 'train':
                start = np.random.randint(0, audio_len - self.max_len)
            else:
                start = (audio_len - self.max_len) // 2
            audio = audio[start:start+self.max_len]
            
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0) # (1, L)
        
        # Apply transforms (Augmentation)
        if self.transform:
            audio_tensor = self.transform(audio_tensor)
            
        return audio_tensor, torch.tensor(label, dtype=torch.float32), filename
