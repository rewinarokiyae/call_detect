import os
import torch
from spam_model.utils.audio_utils import load_audio_wav

class AudioLoader:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def load_pair(self, agent_path, customer_path):
        """
        Load both agent and customer audio files.
        Returns a dictionary with tensors.
        """
        agent_audio, _ = load_audio_wav(agent_path, self.target_sr)
        customer_audio, _ = load_audio_wav(customer_path, self.target_sr)
        
        return {
            "agent": agent_audio,
            "customer": customer_audio
        }
