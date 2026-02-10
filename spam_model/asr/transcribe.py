import sys
import os
import torch
import whisper
import gc
import yaml

# Ensure we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_utils import load_audio_wav

class ASRTranscriber:
    def __init__(self, config_path="spam_model/config.yaml"):
        # Fix path resolution for config
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_full_path = os.path.join(base_dir, "config.yaml")
        
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_name = self.config['models']['asr']['name'].split("/")[-1] # e.g. whisper-base -> base
        if "openai/" in self.config['models']['asr']['name']:
             self.model_name = self.config['models']['asr']['name'].replace("openai/whisper-", "")

    def transcribe(self, audio_path):
        """
        Loads model, transcribes, and unloads model to free VRAM.
        Returns text.
        Uses soundfile via load_audio_wav to avoid ffmpeg dependency.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading ASR Model: {self.model_name} on {device}...")
        
        try:
            model = whisper.load_model(self.model_name, device=device)
        except Exception as e:
            print(f"Error loading whisper: {e}. Fallback to base")
            model = whisper.load_model("base", device=device)

        print(f"Transcribing {audio_path}...")
        
        # FIX: Load with soundfile (no ffmpeg)
        # load_audio_wav returns (tensor, sr). Whisper expects numpy array or tensor.
        audio_tensor, sr = load_audio_wav(audio_path, target_sr=16000)
        # Whisper expects float32 array, normalized -1 to 1. My util does that.
        # It handles flattening to mono too.
        audio_np = audio_tensor.numpy()
        
        result = model.transcribe(audio_np, fp16=(device=="cuda"))
        text = result['text']
        
        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("ASR Model unloaded.")
        
        return text.strip()
