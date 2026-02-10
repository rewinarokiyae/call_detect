import torch
import torch.nn.functional as F
import os
import sys
import pickle
import argparse
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np

# Add root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion import HybridDetectModel

# Reuse extraction logic
def extract_features(waveform, device):
    lfcc_trans = T.LFCC(sample_rate=16000, n_lfcc=60, speckwargs={"n_fft": 512, "hop_length": 160, "center": False}).to(device)
    lfcc = lfcc_trans(waveform)
    
    y = waveform.squeeze().cpu().numpy()
    target_len = int(0.2 * 16000) 
    if len(y) < target_len:
         y = np.pad(y, (0, target_len-len(y)))
         
    cqt = librosa.cqt(y, sr=16000, hop_length=160, n_bins=60, bins_per_octave=12)
    cqt_abs = np.abs(cqt)
    cqt_log = np.log1p(cqt_abs)
    cqcc_np = librosa.feature.mfcc(S=cqt_log, sr=16000, n_mfcc=40, dct_type=2)
    cqcc = torch.tensor(cqcc_np, dtype=torch.float32).unsqueeze(0).to(device) 
    
    min_t = min(lfcc.size(2), cqcc.size(2))
    lfcc = lfcc[:, :, :min_t]
    cqcc = cqcc[:, :, :min_t]
    
    return lfcc, cqcc

def preprocess_audio(file_path):
    try:
        waveform, sr = torchaudio.load(file_path, normalize=True)
        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def enroll(files, labels, output_path="checkpoints/enrolled_users.pkl"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Best Model
    ckpt = "checkpoints/model_best.pth"
    model = HybridDetectModel().to(device)
    state = torch.load(ckpt, map_location=device)
    if 'model_state_dict' in state: state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    
    enrolled_data = []
    
    print("--- Enrolling Users ---")
    
    with torch.no_grad():
        for fpath, label_name in zip(files, labels):
            if not os.path.exists(fpath):
                print(f"Skipping {fpath} (Not Found)")
                continue
                
            wave = preprocess_audio(fpath).to(device)
            # Process in chunks to handle long files?
            # Ideally we want the "Average Embedding" of the file.
            # Let's take 3-4 random segments and average them.
            
            emb_list = []
            
            # Extract multiple segments (1.5s)
            seg_len = int(1.5 * 16000)
            total_len = wave.size(1)
            
            starts = [0, total_len//2, total_len - seg_len]
            starts = [s for s in starts if s >= 0 and s+seg_len <= total_len]
            if not starts: starts = [0]
            
            for start in starts:
                if total_len < seg_len:
                    seg = torch.nn.functional.pad(wave, (0, seg_len - total_len))
                else:
                    seg = wave[:, start:start+seg_len]
                    
                lfcc, cqcc = extract_features(seg, device)
                raw = seg.unsqueeze(0)
                lfcc = lfcc.unsqueeze(0)
                cqcc = cqcc.unsqueeze(0)
                
                _, _, emb = model(lfcc, cqcc, raw, return_embedding=True)
                emb_list.append(emb)
                
            # Average
            if emb_list:
                avg_emb = torch.mean(torch.stack(emb_list), dim=0) # [1, 64]
                avg_emb = F.normalize(avg_emb, p=2, dim=1) # Normalize
                
                enrolled_data.append({
                    "name": label_name,
                    "embedding": avg_emb.cpu(),
                    "path": fpath
                })
                print(f"Enrolled: {label_name} (Vectors: {len(emb_list)})")
    
    with open(output_path, 'wb') as f:
        pickle.dump(enrolled_data, f)
        
    print(f"Saved {len(enrolled_data)} users to {output_path}")

if __name__ == "__main__":
    # Enroll User AND Known Spoof
    files = [
        "data/live_recordings/my_voice.wav", 
        "data/live_recordings/human voice.wav",
        "data/live_recordings/ai_voice.wav"
    ]
    labels = ["User (Owner)", "Reference Human", "Known Spoof"]
    
    enroll(files, labels)
