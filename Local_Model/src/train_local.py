import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# Imports from same dir
from dataset_local import LocalDataset
from model_local import LocalMLP

# Config
DATA_ROOT = r"D:\Subject\honors\ai_project\call_detect\data"
FEATURE_DIR = os.path.join(DATA_ROOT, "precomputed")
PROTO_TRAIN = os.path.join(DATA_ROOT, "2026_data", "ASVspoof5.train.tsv")
PROTO_DEV = os.path.join(DATA_ROOT, "2026_data", "ASVspoof5.dev.track_1.tsv")
OUTPUT_DIR = "Local_Model/checkpoints"

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Datasets
    print("Init Datasets...")
    train_ds = LocalDataset(os.path.join(FEATURE_DIR, "train"), PROTO_TRAIN, "train")
    dev_ds = LocalDataset(os.path.join(FEATURE_DIR, "dev"), PROTO_DEV, "dev")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0) # 0 workers for win compatibility/speed with npy
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = LocalMLP().to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_eer = 1.0
    
    print("Start Training...")
    for epoch in range(1, args.epochs+1):
        # Train
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        
        for i, (feats, labels, uids) in enumerate(pbar):
            feats, labels = feats.to(device), labels.to(device)
            
            # Check for dummy features (zeros)
            if feats.sum() == 0 and feats.max() == 0:
                 continue # Skip missing files
                 
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            if args.dry_run: break
            
        scheduler.step()
        # Fix loss reporting for dry run / partial epochs
        actual_batches = i + 1
        train_loss = running_loss / actual_batches
        
        # --- SANITY CHECK: Eval on Train (first 1000 batches or less) ---
        model.eval()
        train_check_out = []
        train_check_tgt = []
        with torch.no_grad():
             for i, (feats, labels, uids) in enumerate(tqdm(train_loader, desc="Sanity Check [Train]", leave=False)):
                 if i > 50: break # Only check a bit
                 feats, labels = feats.to(device), labels.to(device)
                 logits = model(feats)
                 probs = torch.sigmoid(logits)
                 train_check_out.extend(probs.cpu().numpy())
                 train_check_tgt.extend(labels.cpu().numpy())
        
        try:
             sanity_eer = compute_eer(np.array(train_check_out), np.array(train_check_tgt))
             print(f"DEBUG: Sanity Check (Train Subset) EER: {sanity_eer*100:.2f}%")
        except:
             pass

        # Val
        outputs = []
        targets = []
        
        with torch.no_grad():
            for feats, labels, uids in tqdm(dev_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                feats, labels = feats.to(device), labels.to(device)
                if feats.sum() == 0: continue
                
                logits = model(feats)
                probs = torch.sigmoid(logits)
                
                outputs.extend(probs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
                
                if args.dry_run: break
                
        # Metric
        try:
            # DEBUG SCORES
            outputs = np.array(outputs)
            targets = np.array(targets)
            avg_bonafide = outputs[targets==0].mean() if (targets==0).sum() > 0 else 0
            avg_spoof = outputs[targets==1].mean() if (targets==1).sum() > 0 else 0
            print(f"\nDEBUG: Avg Score Bonafide: {avg_bonafide:.4f} | Avg Score Spoof: {avg_spoof:.4f}")
            
            val_eer = compute_eer(outputs, targets)
        except Exception as e:
            print(f"EER Error: {e}")
            val_eer = 1.0
            
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val EER={val_eer*100:.2f}%")
        
        if val_eer < best_eer:
            best_eer = val_eer
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"Saved Best Model (EER: {best_eer:.4f})")
            
        if args.dry_run: break

if __name__ == "__main__":
    main()
