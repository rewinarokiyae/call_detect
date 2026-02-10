import os
import sys
import torch
import numpy as np

# Mock patching for torchaudio if needed (copied from diarization.py)
import types
import torchaudio
if not hasattr(torchaudio, 'backend'):
    torchaudio.backend = types.ModuleType("backend")
    sys.modules["torchaudio.backend"] = torchaudio.backend
if not hasattr(torchaudio.backend, "list_audio_backends"):
    torchaudio.backend.list_audio_backends = lambda: []

def test_speechbrain():
    print("Testing SpeechBrain Loading...")
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        print("✅ Model Loaded Successfully")
        
        # Test Inference
        dummy_wav = torch.randn(1, 16000) # 1 sec noise
        emb = model.encode_batch(dummy_wav)
        print(f"✅ Embedding Shape: {emb.shape}")
        
        if emb.sum() == 0:
            print("❌ Embedding is all zeros!")
        else:
            print("✅ Embedding content seems valid (non-zero).")
            
    except Exception as e:
        print(f"❌ Failed to load SpeechBrain: {e}")

if __name__ == "__main__":
    test_speechbrain()
