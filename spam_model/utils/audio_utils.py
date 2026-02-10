import soundfile as sf
import librosa
import torch
import numpy as np
import os

def load_audio_wav(file_path, target_sr=16000):
    """
    Load a WAV file, ensure it's mono, and resample to target_sr.
    Uses soundfile directly for speed and robustness without ffmpeg.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Load with soundfile
        audio, sr = sf.read(file_path)
        
        # Convert to float32 if not already
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Resample if necessary
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
        return torch.from_numpy(audio), target_sr
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio {file_path}: {e}")

def get_duration(file_path):
    """
    Get duration of a WAV file in seconds without full load.
    """
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return 0.0
