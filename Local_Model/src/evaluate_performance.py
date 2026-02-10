import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_local import LocalMLP
from dataset_local import LocalDataset

def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_threshold_idx]
    return eer, thresholds[eer_threshold_idx]

def evaluate(model_path, protocol_path, feature_dir, device='cuda'):
    print(f"Loading model from {model_path}...")
    
    # Init Model (Same architecture as training)
    # We need to know input dim. Inspecting saved model or assuming from config.
    # WvlM(2304) + HC(42) = ~2346. Let's try to load state dict and infer or use logic.
    # For now, we instantiate with expected dim.
    model = LocalMLP(input_dim=2346, hidden_dim=512).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Check if it's a dict with metadata or just state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[OK] Model loaded (Epoch {checkpoint.get('epoch', '?')})")
        else:
            model.load_state_dict(checkpoint)
            print("[OK] Model loaded (Direct State Dict)")
    except Exception as e:
        print(f"[gERROR] Failed to load model: {e}")
        return

    model.eval()
    
    print("Initializing Dev Dataset...")
    # LocalDataset(feature_dir, protocol_path, partition)
    # Correct order based on dataset_local.py definition:
    # def __init__(self, feature_dir, protocol_path, partition="train"):
    dataset = LocalDataset(feature_dir, protocol_path, partition="dev")
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    
    all_scores = []
    all_labels = []
    
    print("Running Inference on Dev Set...")
    with torch.no_grad():
        for batch_features, batch_labels, _ in loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features).squeeze()
            scores = torch.sigmoid(outputs)
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            
    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)
    
    # 1. Calculate EER and Optimal Threshold
    eer, optimal_threshold = calculate_eer(y_true, y_scores)
    print(f"\n[STATS] EER: {eer*100:.2f}%")
    print(f"[STATS] Optimal Threshold (at EER): {optimal_threshold:.4f}")
    
    # 2. Metrics at Default Threshold (0.5)
    print("\n--- Metrics at Threshold = 0.5 ---")
    y_pred_05 = (y_scores >= 0.5).astype(int)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred_05)*100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred_05)*100:.2f}% (Spoof)")
    print(f"Recall:    {recall_score(y_true, y_pred_05)*100:.2f}% (Spoof)")
    print(f"F1 Score:  {f1_score(y_true, y_pred_05)*100:.2f}%")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred_05)}")

    # 3. Metrics at Optimal Threshold
    print(f"\n--- Metrics at Optimal Threshold ({optimal_threshold:.4f}) ---")
    y_pred_opt = (y_scores >= optimal_threshold).astype(int)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred_opt)*100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred_opt)*100:.2f}% (Spoof)")
    print(f"Recall:    {recall_score(y_true, y_pred_opt)*100:.2f}% (Spoof)")
    print(f"F1 Score:  {f1_score(y_true, y_pred_opt)*100:.2f}%")
    
    return eer

if __name__ == "__main__":
    MODEL_PATH = "Local_Model/checkpoints/best_model.pth"
    PROTOCOL = "data/2026_data/ASVspoof5.dev.track_1.tsv"
    FEAT_DIR = "data/precomputed/dev"
    
    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        sys.exit(1)
        
    evaluate(MODEL_PATH, PROTOCOL, FEAT_DIR)
