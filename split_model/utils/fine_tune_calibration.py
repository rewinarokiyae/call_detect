import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_robust import CachedASVDataset, multi_collate, get_cache_path
from models.fusion import HybridDetectModel

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 12, # Augmented batch size
    "lr": 1e-5,
    "epochs": 50, # Sufficient for convergence
    "input_model": "checkpoints/model_best.pth",
    "output_model": "checkpoints/model_calibrated.pth",
    "samples": [
        {"path": "data/live_recordings/ai_voice.wav", "label": 0},      
        {"path": "data/live_recordings/human voice.wav", "label": 1},   
        {"path": "data/live_recordings/my_voice.wav", "label": 2}       
    ]
}

def calibrate():
    print("--- User Calibration Phase (Robust) ---")
    
    # 1. Prepare Data
    files = []
    labels = []
    
    for item in CONFIG['samples']:
        if os.path.exists(item['path']):
            files.append(item['path'])
            labels.append(item['label'])
            print(f"Added Calibration Sample: {item['path']} -> Label {item['label']}")
        else:
            print(f"WARNING: File not found: {item['path']}")
            
    if not files:
        print("No files found for calibration.")
        return
    
    # Duplicate data x10 for stability (Batch Size 12)
    ds = CachedASVDataset(files*10, labels*10, max_len=64000) 
    dl = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=multi_collate)
    
    # 2. Load Model
    model = HybridDetectModel().to(CONFIG['device'])
    if not os.path.exists(CONFIG['input_model']):
        print("Input model best not found, trying robust.")
        CONFIG['input_model'] = "checkpoints/model_robust.pth"
        
    state = torch.load(CONFIG['input_model'], map_location=CONFIG['device'])
    if 'model_state_dict' in state: state = state['model_state_dict']
    model.load_state_dict(state)
    
    # Unfreeze Everything
    print("Unfreezing ALL layers for calibration...")
    for param in model.parameters(): param.requires_grad = True
    
    # Optimize
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    # Force BN to Eval
    print("Forcing BatchNorm to EVAL mode...")
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            
    print(f"Calibrating for {CONFIG['epochs']} epochs...")
    
    for ep in range(CONFIG['epochs']):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dl:
            raw, lfcc, cqcc, label_tens = batch
            if raw is None: continue
            
            raw, lfcc, cqcc, label_tens = raw.to(CONFIG['device']), lfcc.to(CONFIG['device']), cqcc.to(CONFIG['device']), label_tens.to(CONFIG['device'])
            
            optimizer.zero_grad()
            logits, weights = model(lfcc, cqcc, raw)
            
            loss = criterion(logits, label_tens)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label_tens).sum().item()
            total += label_tens.size(0)
            
            if ep % 10 == 0:
                 # Debug first batch of epoch
                 pass

        if (ep+1) % 5 == 0:
            print(f"Ep {ep+1}: Loss {total_loss:.4f} | Acc {100.*correct/total:.1f}%")

    # Save
    torch.save(model.state_dict(), CONFIG['output_model'])
    print(f"Saved calibrated model to {CONFIG['output_model']}")

if __name__ == "__main__":
    calibrate()
