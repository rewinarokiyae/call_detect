import torch
import pandas as pd
import os
import sys
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spam_model.model.classifier import ScamClassifier
from spam_model.model.train import FeatureDataset

def evaluate_model():
    config_path = "spam_model/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    splits_dir = config['paths']['splits_dir']
    features_dir = os.path.join(config['paths']['data_spam'], "features")
    checkpoints_dir = config['paths']['checkpoints_dir']
    eval_dir = config['paths']['evaluation_dir']
    os.makedirs(eval_dir, exist_ok=True)
    
    test_csv = os.path.join(splits_dir, "test.csv")
    if not os.path.exists(test_csv):
        print("Test split not found.")
        return
        
    test_dataset = FeatureDataset(test_csv, features_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Load Best Model
    model_path = os.path.join(checkpoints_dir, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoints_dir, "last_model.pth")
        
    if not os.path.exists(model_path):
        print("No model checkpoint found.")
        return
        
    input_dim = config['models']['classifier']['input_dim']
    model = ScamClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    y_true = []
    y_pred = []
    y_scores = []
    
    print("Evaluating on Test Set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            score = outputs.item()
            predicted = 1 if score > 0.5 else 0
            
            y_true.append(int(labels.item()))
            y_pred.append(predicted)
            y_scores.append(score)
            
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    report = classification_report(y_true, y_pred, target_names=['Bonafide', 'Scam'])
    
    # Save Report
    report_path = os.path.join(eval_dir, "report.txt")
    with open(report_path, 'w') as f:
        f.write("SCAM DETECTION MODEL EVALUATION REPORT\n")
        f.write("========================================\n\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"False Positive Rate (FPR): {fpr:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TN: {tn} | FP: {fp}\n")
        f.write(f"FN: {fn} | TP: {tp}\n\n")
        f.write("Detailed Report:\n")
        f.write(report)
        
    print(f"Evaluation Complete. Report saved to {report_path}")
    print(f"Accuracy: {acc:.4f} | FPR: {fpr:.4f}")

if __name__ == "__main__":
    evaluate_model()
