import torch
import numpy as np
import librosa

def extract_prosody_features(waveform, sr=16000):
    """
    Extracts statistical prosody features from a raw waveform.
    Input:
        waveform: torch.Tensor [1, T] or numpy array [T]
        sr: sample rate (default 16000)
    Output:
        torch.Tensor [12] (Feature Vector)
    """
    if isinstance(waveform, torch.Tensor):
        y = waveform.squeeze().cpu().numpy()
    else:
        y = waveform
        
    if len(y) < sr * 0.1: # Too short
        return torch.zeros(12)

    # 1. Pitch (F0) using pYIN
    # Frame length ~30ms
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                 fmax=librosa.note_to_hz('C7'), sr=sr,
                                                 frame_length=2048, hop_length=512)
    
    # Filter only voiced frames for F0 stats
    f0_voiced = f0[voiced_flag]
    if len(f0_voiced) == 0:
        f0_mean, f0_std, f0_range, jitter = 0.0, 0.0, 0.0, 0.0
    else:
        f0_mean = np.mean(f0_voiced)
        f0_std = np.std(f0_voiced)
        f0_range = np.max(f0_voiced) - np.min(f0_voiced)
        
        # Approximate Jitter: Mean absolute difference between consecutive F0 periods
        # Jitter (local) = mean(|T_i - T_{i-1}|) / mean(T)
        # Here we use F0 directly: mean(|F0_i - F0_{i-1}|) / mean(F0)
        f0_diff = np.abs(np.diff(f0_voiced))
        jitter = np.mean(f0_diff) / (f0_mean + 1e-6)

    # 2. Energy (RMS) & Shimmer
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    
    energy_mean = np.mean(rms)
    energy_std = np.std(rms)
    
    # Shimmer approximation: Amplitude perturbation on voiced frames
    # Ideally should map rms frames to f0 frames
    # Simple approx: variation in RMS
    rms_diff = np.abs(np.diff(rms))
    shimmer = np.mean(rms_diff) / (energy_mean + 1e-6)

    # 3. Spectral features
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    flat_mean = np.mean(flatness)
    flat_std = np.std(flatness)
    
    # 4. Voicing / Rhythm
    # Fraction of frames that are voiced
    voicing_rate = np.sum(voiced_flag) / len(voiced_flag)
    
    # 5. Zero Crossing Rate (Roughness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # Feature Vector:
    # 0: F0 Mean
    # 1: F0 Std
    # 2: F0 Range
    # 3: Jitter
    # 4: Energy Mean
    # 5: Energy Std
    # 6: Shimmer
    # 7: Flatness Mean
    # 8: Flatness Std
    # 9: Voicing Rate
    # 10: ZCR Mean
    # 11: ZCR Std
    
    feats = np.array([
        f0_mean, f0_std, f0_range, jitter,
        energy_mean, energy_std, shimmer,
        flat_mean, flat_std, voicing_rate,
        zcr_mean, zcr_std
    ], dtype=np.float32)
    
    # Normalize/Sanitize (Log scale some features?)
    # For now, raw stats.
    
    return torch.from_numpy(feats)
