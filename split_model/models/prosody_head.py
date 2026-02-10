import torch
import torch.nn as nn

class ProsodyBranch(nn.Module):
    def __init__(self, input_dim=12, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, embedding_dim),
            # No activation at end of embedding, fusion layer handles scaling
        )
        
    def forward(self, x):
        # x: [batch, 12]
        return self.net(x)
