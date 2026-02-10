import librosa
import numpy as np
import sys
import argparse

def analyze(path):
    print(f"Analyzing {path}...")
    try:
        y, sr = librosa.load(path, sr=16000)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Energy
    rms = librosa.feature.rms(y=y)
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    max_rms = np.max(rms)
    
    # Pitch (Fundamental Frequency)
    # F0 range 50-500Hz covers most human speech
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0 = f0[~np.isnan(f0)]
    
    if len(f0) == 0:
        mean_f0 = 0
        std_f0 = 0
        voicing_rate = 0
    else:
        mean_f0 = np.mean(f0)
        std_f0 = np.std(f0)
        voicing_rate = len(f0) / len(voiced_flag)
    
    # Spectral Features
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    
    mean_rolloff = np.mean(rolloff)
    mean_bw = np.mean(bandwidth)

    print(f"RMS Energy: Mean={mean_rms:.4f}, Std={std_rms:.4f}, Max={max_rms:.4f}")
    print(f"Pitch: Mean={mean_f0:.2f} Hz, Std={std_f0:.2f} Hz, Voicing Rate={voicing_rate:.2%}")
    print(f"Spectral: Rolloff={mean_rolloff:.0f} Hz, Bandwidth={mean_bw:.0f} Hz")
    
    # Simple Heuristic?
    if mean_rms < 0.01:
        print("Verdict: LOW ENERGY (Silence/Noise)")
    elif voicing_rate < 0.1:
        print("Verdict: UNVOICED (Noise/Breathing/Music)")
    else:
        print("Verdict: SPEECH DETECTED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Audio file path")
    args = parser.parse_args()
    analyze(args.file)
