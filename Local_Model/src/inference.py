import os
import torch
import numpy as np
import librosa
import argparse
import sys
import logging

# Suppress Transformers Warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Add src to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_features import FeatureExtractor
from model_local import LocalMLP

SAMPLE_RATE = 16000
MAX_LEN = 64000

def predict(audio_path, model_path, device):
    print(f"Analyzing: {audio_path}")
    
    # 1. Load Audio
    try:
        wav_np, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return {"score": 0.0, "verdict": "ERROR", "error": str(e)}

    # Pad/Cut
    # Sliding Window / Segment-based Inference
    window_samples = MAX_LEN # 4 seconds
    stride = window_samples // 2
    
    scores = []
    
    # Pad to at least one window
    if len(wav_np) < window_samples:
        wav_np = np.pad(wav_np, (0, window_samples - len(wav_np)))
        
    num_windows = max(1, (len(wav_np) - window_samples) // stride + 1)
    
    # 3. Model Inference
    model = LocalMLP().to(device)
    
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}")
        return {"score": 0.0, "verdict": "ERROR", "error": "Model not found"}

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"score": 0.0, "verdict": "ERROR", "error": str(e)}
        
    model.eval()
    
    extractor = FeatureExtractor(device)
    
    with torch.no_grad():
        for i in range(num_windows):
            start = i * stride
            end = start + window_samples
            chunk = wav_np[start:end]
            
            if len(chunk) < window_samples:
                 chunk = np.pad(chunk, (0, window_samples - len(chunk)))

            chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).float().to(device)
            
            # WavLM
            wavlm_emb = extractor.compute_wavlm(chunk_tensor)
            
            # Handcrafted
            hc_emb = extractor.compute_hc(chunk)
            
            # Fuse
            final_emb = np.concatenate([wavlm_emb, hc_emb])
            final_tensor = torch.from_numpy(final_emb).float().unsqueeze(0).to(device)
            
            logit = model(final_tensor)
            seg_score = torch.sigmoid(logit).item()
            scores.append(seg_score)

    if not scores:
        score = 0.0
    else:
        # Statistical Analysis
        scores_np = np.array(scores)
        p50 = np.percentile(scores_np, 50)
        p75 = np.percentile(scores_np, 75)
        p90 = np.percentile(scores_np, 90)
        p_max = np.max(scores_np)
        
        print(f"Segment Stats: Max={p_max:.4f}, p90={p90:.4f}, p75={p75:.4f}, Median={p50:.4f}")
        print(f"All Scores: {[f'{s:.3f}' for s in scores]}")
        
        # Current Strategy: Max (Aggressive)
        # We will adjust this based on the comparison between call2 (Real AI) and WhatsApp (False AI)
        score = p_max
        
    # 4. Result
    print("-" * 30)
    print(f"Spoof Score: {score:.4f} (0=Human, 1=AI)")
    print(f"Median Score: {p50:.4f}")
    
    threshold = 0.5
    if score > threshold:
        verdict = "SPOOF"
        print("DETECTED: AI / SYNTHETIC AUDIO")
    else:
        verdict = "BONAFIDE"
        print("DETECTED: HUMAN / BONAFIDE AUDIO")
    print("-" * 30)
    
    return {
        "score": score,
        "verdict": verdict,
        "confidence": score if verdict == "SPOOF" else 1 - score
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default=r"D:\Subject\honors\ai_project\call_detect\Local_Model\checkpoints\best_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict(args.audio, args.model, device)
