import torch
import torch.nn as nn

class LocalMLP(nn.Module):
    def __init__(self, input_dim=2346, hidden_dim=512, dropout=0.5):
        super().__init__()
        
        self.net = nn.Sequential(
            # Input Normalization (Critical for HC features with large range)
            nn.LayerNorm(input_dim), 
            
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, 1) # Logit for BCE
        )
        
    def forward(self, x):
        return self.net(x).squeeze(1)
