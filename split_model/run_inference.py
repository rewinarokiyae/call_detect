import torch
import torchaudio
import torchaudio.transforms as T
import argparse
import os
import sys
import numpy as np
import librosa
from models.fusion import HybridDetectModel
from models.prosody import extract_prosody_features
import torch.nn.functional as F
import pickle

# --- Configuration ---
SEGMENT_LEN = 1.5 
STEP_SIZE = 0.75 
SAMPLE_RATE = 16000
CONFIDENCE_THRESHOLD = 0.995 

# --- Artifact Detection ---
def detect_dsp_artifacts(waveform_np, sr=16000):
    indicators = []
    is_dsp = False
    
    vals = np.abs(waveform_np)
    if np.max(vals) == 0: return False, ["Silence"]
    
    rms = np.sqrt(np.mean(vals**2))
    peak = np.max(vals)
    dr_ratio = 20 * np.log10(peak / (rms + 1e-9))
    
    if dr_ratio < 14.0:
        indicators.append(f"High Compression (DR={dr_ratio:.1f}dB)")
        is_dsp = True
        
    return is_dsp, indicators

# --- Feature Extraction ---
def extract_features(waveform, sr=16000):
    # 1. LFCC
    lfcc_trans = T.LFCC(
        sample_rate=16000,
        n_lfcc=60,
        speckwargs={"n_fft": 512, "hop_length": 160, "center": False}
    ).to(waveform.device)
    lfcc = lfcc_trans(waveform)
    
    # 2. CQCC
    y = waveform.squeeze().cpu().numpy()
    target_len = int(0.2 * 16000) 
    if len(y) < target_len:
         y = np.pad(y, (0, target_len-len(y)))
         
    cqt = librosa.cqt(y, sr=16000, hop_length=160, n_bins=60, bins_per_octave=12)
    cqt_log = np.log1p(np.abs(cqt))
    cqcc_np = librosa.feature.mfcc(S=cqt_log, sr=16000, n_mfcc=40, dct_type=2)
    cqcc = torch.tensor(cqcc_np, dtype=torch.float32).unsqueeze(0).to(waveform.device) 
    
    # 3. Prosody
    prosody_vec = extract_prosody_features(y, sr).unsqueeze(0).to(waveform.device) # [1, 12]

    # Sync Time Dims for Spectral Features
    min_t = min(lfcc.size(2), cqcc.size(2))
    lfcc = lfcc[:, :, :min_t]
    cqcc = cqcc[:, :, :min_t]
    
    return lfcc, cqcc, prosody_vec

def load_model(checkpoint_path, device):
    model = HybridDetectModel().to(device)
    if os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in state: state = state['model_state_dict']
            
            # Partial Loading Logic
            model_dict = model.state_dict()
            # Filter mismatched keys (e.g. Gate resizing)
            pretrained_dict = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            print(f"Loaded {len(pretrained_dict)} layers from {checkpoint_path}")
            if len(pretrained_dict) < len(model_dict):
                print("WARNING: Partial load detected. Some layers initialized randomly (New Architecture).")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
    else:
        print(f"CRITICAL: Checkpoint {checkpoint_path} not found.")
        sys.exit(1)
    model.eval()
    return model

def preprocess_audio(file_path):
    try:
        try:
            waveform, sr = torchaudio.load(file_path, normalize=True)
        except Exception:
            import librosa
            y, sr = librosa.load(file_path, sr=None)
            waveform = torch.from_numpy(y).float().unsqueeze(0)

        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# --- Enrollment Logic ---
ENROLLED_USERS = []
def load_enrollment(path="checkpoints/enrolled_users.pkl"):
    global ENROLLED_USERS
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f: ENROLLED_USERS = pickle.load(f)
            print(f"Loaded {len(ENROLLED_USERS)} enrolled users.")
        except Exception as e: print(f"Error loading enrollment: {e}")

def check_enrollment_match(embedding):
    if not ENROLLED_USERS: return None, 0.0
    best_score = -1.0
    best_name = None
    emb_norm = F.normalize(embedding.cpu(), p=2, dim=1)
    for user in ENROLLED_USERS:
        u_emb = user['embedding'].view(1, -1) 
        sim = F.cosine_similarity(emb_norm, u_emb).item()
        if sim > best_score:
            best_score = sim
            best_name = user['name']
    return best_name, best_score

def predict_robust(model, waveform, device, temperature=1.5):
    """
    4-Branch Inference (Pure Model)
    """
    waveform = waveform.to(device) 
    total_samples = waveform.size(1)
    seg_samples = int(SEGMENT_LEN * SAMPLE_RATE)
    step_samples = int(STEP_SIZE * SAMPLE_RATE)
    
    segments_results = []
    
    if total_samples < seg_samples:
        waveform = torch.nn.functional.pad(waveform, (0, seg_samples - total_samples))
        total_samples = waveform.size(1)
        
    for start in range(0, total_samples - int(0.1*SAMPLE_RATE), step_samples):
        end = start + seg_samples
        if end > total_samples: break 
            
        seg_wave = waveform[:, start:end]
        
        # Artifact Check
        is_dsp, dsp_flags = detect_dsp_artifacts(seg_wave.squeeze().cpu().numpy())
        
        # Inference
        lfcc, cqcc, prosody = extract_features(seg_wave)
        in_raw = seg_wave.unsqueeze(0)
        in_lfcc = lfcc.unsqueeze(0)
        in_cqcc = cqcc.unsqueeze(0)
        in_prosody = prosody # Already [1, 12]
        
        with torch.no_grad():
            logits, weights, embedding, aux_out = model(in_lfcc, in_cqcc, in_raw, in_prosody, return_embedding=True)
            
            logits = logits / temperature
            probs = torch.softmax(logits, dim=1) 
            
            p_spoof_raw = probs[0][0].item()
            p_clean = probs[0][1].item()
            p_deg = probs[0][2].item()
            p_bonafide_raw = p_clean + p_deg
            
            w = weights[0].detach().cpu().numpy() # [4]
            match_name, match_score = check_enrollment_match(embedding)
            
            reasons = []
            
            # --- MARGIN-BASED DECISION ---
            diff = p_bonafide_raw - p_spoof_raw
            
            if diff > 0.15: 
                verdict = "BONAFIDE"
            elif diff < -0.1:
                verdict = "SPOOF"
            else:
                verdict = "SPOOF" # Conservative default
                reasons.append(f"Low Margin ({diff:.2f})")

            # Enrollment Override
            if match_name and match_score > 0.85:
                 verdict = "BONAFIDE"
                 reasons.append(f"Enrolled: {match_name}")
            
            if w[3] > 0.3: reasons.append(f"Prosody Focus ({w[3]:.2f})")
            
            # Check Aux Head if available (optional sanity check)
            aux_prob = torch.sigmoid(aux_out).item()
            if aux_prob > 0.8: reasons.append(f"Aux Flag ({aux_prob:.2f})")

            segments_results.append({
                'p_spoof': p_spoof_raw,
                'p_bonafide': p_bonafide_raw,
                'verdict': verdict,
                'w_lfcc': w[0],
                'w_cqcc': w[1],
                'w_raw': w[2],
                'w_prosody': w[3],
                'dsp_flags': dsp_flags,
                'start': start / SAMPLE_RATE,
                'enroll_score': match_score if match_name else 0.0
            })

    if not segments_results: return {"verdict": "ERROR", "confidence": 0.0, "reason": "No segments"}

    avg_spoof = np.mean([s['p_spoof'] for s in segments_results])
    
    final_verdict = "BONAFIDE"
    final_conf = 1.0 - avg_spoof
    
    if avg_spoof > 0.50:
        final_verdict = "SPOOF"
        final_conf = avg_spoof
        
    return {
        "verdict": final_verdict,
        "confidence": final_conf,
        "segments": segments_results,
        "reasons": reasons
    }

def run_inference(file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_enrollment()
    
    # Priority: Upgraded > Best > Latest
    ckpt = "checkpoints/model_upgraded.pth"
    if not os.path.exists(ckpt): ckpt = "checkpoints/model_best.pth"
    if not os.path.exists(ckpt): ckpt = "checkpoints/model_robust.pth"
    
    print(f"Loading checkpoint: {ckpt}")
    model = load_model(ckpt, device)
    
    waveform = preprocess_audio(file_path)
    if waveform is None: return

    res = predict_robust(model, waveform, device)
    
    print("\n" + "="*50)
    print(f"MODERN INFERENCE REPORT: {os.path.basename(file_path)}")
    print("="*50)
    print(f"Final Verdict: {res['verdict']}")
    print(f"Confidence:    {res['confidence']*100:.2f}%")
    if res['reasons']: print(f"Flags:         {res['reasons']}")
    print("-" * 50)
    print("Segment Analysis:")
    print(f"{'Time':<6} | {'Verdict':<8} | {'Sp%':<4} | {'Pro%':<4} | {'DSP'}")
    for s in res['segments']:
        t_str = f"{s['start']:.1f}s"
        v_str = s['verdict'][:8]
        sp_str = f"{s['p_spoof']*100:.0f}"
        pr_str = f"{s['w_prosody']*100:.0f}"
        dsp_str = "Yes" if s['dsp_flags'] else "-"
        print(f"{t_str:<6} | {v_str:<8} | {sp_str:<4} | {pr_str:<4} | {dsp_str}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, nargs='?', help="Path to audio file")
    args = parser.parse_args()
    if args.file: run_inference(args.file)
