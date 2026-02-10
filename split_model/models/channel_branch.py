import torch
import torch.nn as nn
from .utils import compute_spectral_flatness, estimate_bandwidth

class ChannelBranch(nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        
        # We will take 3 handcrafted features + a small learned embedding
        # Features: [Flatness, Bandwidth, Energy_Variance]
        
        self.mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: [batch, samples]
        
        # 1. Extract Handcrafted Features (On CPU usually via Librosa, potentially slow but necessary)
        # For training efficiency, these might be pre-computed. 
        # For inference, we verify on the fly.
        
        with torch.no_grad():
            flatness = compute_spectral_flatness(x) # [batch, 1]
            bw = estimate_bandwidth(x, sr=self.sample_rate) # [batch, 1]
            
            # Simple Energy Variance (proxy for dynamic range)
            energy = torch.mean(x**2, dim=1, keepdim=True)
            energy_log = torch.log(energy + 1e-6)
            
        # Normalize roughly (Bandwidth 0-8000 -> 0-1)
        bw_norm = bw / 8000.0
        
        features = torch.cat([flatness, bw_norm, energy_log], dim=1) # [batch, 3]
        
        # 2. MLP
        embedding = self.mlp(features) # [batch, 64]
        
        return embedding
