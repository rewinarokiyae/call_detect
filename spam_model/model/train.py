import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import sys
import yaml
import copy
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spam_model.model.classifier import ScamClassifier

# === Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

# === Dataset ===
class FeatureDataset(Dataset):
    def __init__(self, metadata_path, features_dir):
        self.df = pd.read_csv(metadata_path)
        self.features_dir = features_dir
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fid = str(row['file_id'])
        label = float(row['label'])
        
        feat_path = os.path.join(self.features_dir, f"{fid}.pt")
        if os.path.exists(feat_path):
            features = torch.load(feat_path, map_location='cpu')
        else:
            print(f"Warning: Feature file missing for {fid}, using zeros")
            features = torch.zeros(396) # fallback dimensions
            
        return features, torch.tensor([label], dtype=torch.float32)

def train_model():
    # Load Config
    config_path = "spam_model/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    data_dir = config['paths']['data_spam']
    splits_dir = config['paths']['splits_dir']
    features_dir = os.path.join(data_dir, "features")
    checkpoints_dir = config['paths']['checkpoints_dir']
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Check if splits exist
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    
    if not os.path.exists(train_csv):
        print("Splits not found. Please run prepare_data.py first.")
        return
        
    # Datasets & Loaders
    train_dataset = FeatureDataset(train_csv, features_dir)
    val_dataset = FeatureDataset(val_csv, features_dir)
    
    batch_size = config['models']['classifier']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    input_dim = config['models']['classifier']['input_dim']
    model = ScamClassifier(input_dim=input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training Params
    lr = config['models']['classifier']['learning_rate']
    epochs = config['models']['classifier']['epochs']
    patience = config['models']['classifier']['patience']
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0) # Address class imbalance
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    early_stop_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"Starting Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient Clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / len(val_dataset)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Checkpoint Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "best_model.pth"))
            early_stop_counter = 0
            # print("  > Saved Best Model")
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f"Early stop at epoch {epoch+1}")
            break
            
    # Save Last Model
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, "last_model.pth"))
    print("Training Complete.")
    
    # Reload Best Model for return
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    train_model()
