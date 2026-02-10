import os
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np

class ASVspoofDataset(Dataset):
    def __init__(self, base_dir, partition='LA', subset='train', max_len=64000):
        """
        Args:
            base_dir (str): Path to data directory (e.g., 'data')
            partition (str): 'LA' or 'PA'
            subset (str): 'train', 'dev', or 'eval'
            max_len (int): Max audio samples (approx 4 sec at 16kHz)
        """
        self.base_dir = base_dir
        self.partition = partition
        self.subset = subset
        self.max_len = max_len
        
        # Handle the discovered nested structure: data/LA/LA/...
        # We try both standard and nested path to be robust
        self.protocol_dir = os.path.join(base_dir, partition, f"ASVspoof2019_{partition}_cm_protocols")
        if not os.path.exists(self.protocol_dir):
            self.protocol_dir = os.path.join(base_dir, partition, partition, f"ASVspoof2019_{partition}_cm_protocols")
            self.audio_dir = os.path.join(base_dir, partition, partition, f"ASVspoof2019_{partition}_{subset}", "flac")
        else:
            self.audio_dir = os.path.join(base_dir, partition, f"ASVspoof2019_{partition}_{subset}", "flac")

        # Protocol file name construction
        protocol_filename = f"ASVspoof2019.{partition}.cm.{subset}.trn.txt"
        if subset != 'train':
            # dev and eval often have .trl or .txt depending on version, generic fix:
            protocol_filename = f"ASVspoof2019.{partition}.cm.{subset}.trl.txt" 
        
        self.protocol_path = os.path.join(self.protocol_dir, protocol_filename)
        
        # If specific file not found, try searching or fallback (eval often differs)
        if not os.path.exists(self.protocol_path):
             # Try simple .txt
             self.protocol_path = self.protocol_path.replace(".trl.txt", ".txt")

        if not os.path.exists(self.protocol_path):
            raise FileNotFoundError(f"Protocol file not found: {self.protocol_path}")

        self.data_entries = self._parse_protocol(self.protocol_path)
        print(f"Loaded {len(self.data_entries)} entries for {partition}-{subset}")

    def _parse_protocol(self, path):
        entries = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # LA format: SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID KEY
                # PA format might differ slightly but usually same columns for CM
                if len(parts) >= 4:
                    key = parts[-1] # 'bonafide' or 'spoof'
                    filename = parts[1]
                    label = 0 if key == 'bonafide' else 1
                    entries.append((filename, label))
        return entries

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        filename, label = self.data_entries[idx]
        file_path = os.path.join(self.audio_dir, filename + ".flac")
        
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as e:
            # Handle missing files gracefully during robust training or diagnosing
            print(f"Error loading {file_path}: {e}")
            waveform = torch.zeros(1, self.max_len)
            
        # Pad or Truncate
        if waveform.shape[1] < self.max_len:
            pad_len = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :self.max_len]
            
        return waveform, label

def collate_fn(batch):
    waveforms = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return waveforms, labels
