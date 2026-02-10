import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse
import os
import sys

from dataset import ASVspoof5Dataset
from model import CountermeasureModel
from utils import seed_everything, get_logger, FocalLoss, compute_eer, save_checkpoint

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    all_scores = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for audio, labels, filenames in pbar:
        audio, labels = audio.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(audio)
            loss = criterion(logits, labels)
            
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item()
        
        # Collect for metrics (optional for train, but good for sanity)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
        
    avg_loss = running_loss / len(loader)
    return avg_loss

def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for audio, labels, filenames in pbar:
            audio, labels = audio.to(device), labels.to(device)
            
            logits = model(audio)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            
            # Scores for EER (Sigmoid for probability)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = running_loss / len(loader)
    try:
        eer = compute_eer(all_scores, all_labels)
    except Exception as e:
        print(f"EER Calc Failed: {e}")
        eer = 1.0 # Fail high
        
    return avg_loss, eer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--dry_run', action='store_true', help="Run 1 batch to check setup")
    args = parser.parse_args()
    
    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup
    seed_everything(config['project']['seed'])
    output_dir = config['project']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger("ASVspoof_Train", output_dir)
    logger.info(f"Loaded config from {args.config}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using Device: {device}")
    
    # Data
    logger.info("Initializing Datasets...")
    train_ds = ASVspoof5Dataset(config, 'train')
    dev_ds = ASVspoof5Dataset(config, 'dev')
    
    train_loader = DataLoader(train_ds, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'], pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'], pin_memory=True)
    
    # Model
    logger.info("Initializing Model...")
    model = CountermeasureModel(config).to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate']), 
        weight_decay=float(config['training']['weight_decay'])
    )
    
    if config['training']['loss'] == 'focal':
        criterion = FocalLoss(gamma=config['training']['focal_gamma']).to(device)
    else:
        criterion = nn.BCEWithLogitsLoss().to(device)
        
    scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] and device.type == 'cuda' else None
    
    # Training Loop
    best_eer = 1.0
    start_epoch = 1
    
    if args.dry_run:
        logger.info("DRY RUN MODE: Running 1 batch only.")
        config['training']['epochs'] = 1
    
    logger.info("Starting Training...")
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        if args.dry_run:
             # Just break after 1 step
             pass
             
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_eer = validate(model, dev_loader, criterion, device, epoch)
        
        logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val EER={val_eer*100:.2f}%")
        
        # Save Best
        if val_eer < best_eer:
            best_eer = val_eer
            logger.info(f"New Best EER! Saving checkpoint...")
            save_checkpoint(model, optimizer, epoch, val_eer, os.path.join(output_dir, "best_model.pth"))
            
        # Regular save
        save_checkpoint(model, optimizer, epoch, val_eer, os.path.join(output_dir, "last_model.pth"))
        
    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
