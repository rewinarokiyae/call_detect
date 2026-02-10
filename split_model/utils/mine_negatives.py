import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
from tqdm import tqdm
import json

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_robust import CachedASVDataset, multi_collate, parse_protocol, get_cache_path
from models.fusion import HybridDetectModel

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 32,
    "model_path": "checkpoints/model_best.pth",
    "output_file": "hard_negatives.json",
    "target_count": 50,
    "min_conf": 0.80 # Only select confident errors (hard negatives)
}

def mine():
    print("--- Hard Negative Mining ---")
    
    # 1. Load Data (Spoof Only from Dev)
    base_dir = "data/LA/LA"
    dev_audio = os.path.join(base_dir, "ASVspoof2019_LA_dev/flac")
    dev_proto = os.path.join(base_dir, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    
    print(f"Parsing {dev_proto}...")
    # Custom parse: Get only SPOOF files
    spoof_files = []
    with open(dev_proto, 'r') as f:
        for line in f:
            parts = line.strip().split()
            fname = parts[1]
            key = parts[-1]
            if key == "spoof":
                path = os.path.join(dev_audio, fname + ".flac")
                if os.path.exists(path):
                    spoof_files.append(path)
    
    print(f"Found {len(spoof_files)} spoof files in Dev set.")
    
    # Check cache coverage
    cached_spoofs = [f for f in spoof_files if os.path.exists(get_cache_path(f))]
    print(f"Using {len(cached_spoofs)} cached files (skipping uncached for speed).")
    
    if len(cached_spoofs) == 0:
        print("Error: No cached features found. Please run train_robust.py pre-computation first.")
        return

    # Dataset (Label 0=Spoof, but we don't need label for mining, we know they are spoof)
    # We pass label=0
    ds = CachedASVDataset(cached_spoofs, [0]*len(cached_spoofs), max_len=48000)
    dl = DataLoader(ds, batch_size=CONFIG['batch_size'], num_workers=4, collate_fn=multi_collate)
    
    # 2. Load Model
    model = HybridDetectModel().to(CONFIG['device'])
    state = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
    if 'model_state_dict' in state: state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    
    hard_negatives = []
    
    print("Scanning for False Negatives (Spoof -> Bonafide)...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dl)):
            raw, lfcc, cqcc, labels = batch
            if raw is None: continue
            
            raw, lfcc, cqcc = raw.to(CONFIG['device']), lfcc.to(CONFIG['device']), cqcc.to(CONFIG['device'])
            
            logits, weights = model(lfcc, cqcc, raw)
            probs = torch.softmax(logits, dim=1)
            
            # Check predictions
            # Class 0 = Spoof, 1=Clean, 2=Degraded
            # Bonafide Prob = P(1) + P(2)
            
            p_spoof = probs[:, 0]
            p_bonafide = probs[:, 1] + probs[:, 2]
            
            # Find Errors: True Label is Spoof (0), but Predicted Bonafide (> 0.5)
            # HARD Errors: Predicted Bonafide > CONFIG['min_conf']
            
            errors = (p_bonafide > CONFIG['min_conf'])
            error_indices = torch.nonzero(errors).squeeze(1)
            
            for idx in error_indices:
                idx = idx.item()
                global_idx = i * CONFIG['batch_size'] + idx
                if global_idx >= len(cached_spoofs): break
                
                filepath = cached_spoofs[global_idx]
                score = p_bonafide[idx].item()
                w = weights[idx].mean(dim=-1).squeeze().cpu().numpy() # [3]
                
                # We specifically want errors driven by LFCC?
                # User: "dominated by LFCC attention"
                # Let's prefer those, but collect any hard negative first.
                
                hard_negatives.append({
                    "path": filepath,
                    "p_bonafide": score,
                    "weights": w.tolist(), # [LFCC, CQCC, Raw]
                    "error_type": "False Negative"
                })
                
                if len(hard_negatives) >= CONFIG['target_count']:
                    break
            
            if len(hard_negatives) >= CONFIG['target_count']:
                break
                
    print(f"Mining complete. Found {len(hard_negatives)} hard negatives.")
    
    with open(CONFIG['output_file'], 'w') as f:
        json.dump(hard_negatives, f, indent=4)
        
    print(f"Saved to {CONFIG['output_file']}")

if __name__ == "__main__":
    mine()
