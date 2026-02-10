import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import gc
import yaml
import os

class EmbeddingExtractor:
    def __init__(self, config_path="spam_model/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.source = self.config['models']['embeddings']['name']
        
    def extract(self, audio_tensor):
        """
        Extracts 192-dim embedding from audio tensor.
        Audio tensor should be shape (1, T) or (T,).
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Embedding Model: {self.source} on {device}...")
        
        classifier = EncoderClassifier.from_hparams(source=self.source, run_opts={"device": device})
        
        # Ensure tensor is correct shape (1, T)
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        audio_tensor = audio_tensor.to(device)
        
        with torch.no_grad():
            embeddings = classifier.encode_batch(audio_tensor)
            # embeddings shape: (batch, 1, 192) -> (192,)
            emb_vector = embeddings[0, 0, :].cpu().numpy()
            
        # Cleanup
        del classifier
        gc.collect()
        torch.cuda.empty_cache()
        print("Embedding Model unloaded.")
        
        return emb_vector
