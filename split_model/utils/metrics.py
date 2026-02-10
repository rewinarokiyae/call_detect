import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(bonafide_scores, spoof_scores):
    """
    Returns the Equal Error Rate (EER) and the threshold.
    bonafide_scores: scores for genuine speech (target class, usually label 1 or 2 in our case if mapped)
    spoof_scores: scores for spoof speech (non-target class, usually label 0)
    
    Note: We assume higher scores = more likely bonafide.
    If the model outputs [SPOOF, BONAFIDE, ...], we might need to take score[BONAFIDE] - score[SPOOF] 
    or just score[BONAFIDE].
    """
    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 0.5, 0.0

    y_true = np.concatenate([np.ones(len(bonafide_scores)), np.zeros(len(spoof_scores))])
    y_scores = np.concatenate([bonafide_scores, spoof_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    
    # EER is where FPR = 1 - TPR (or FPR = FNR)
    fnr = 1 - tpr
    
    # Find index where abs(fpr - fnr) is minimized
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    
    eer = fpr[idx]
    threshold = thresholds[idx]
    
    return eer, threshold
