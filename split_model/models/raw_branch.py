import torch
import torch.nn as nn

class RawBranch(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        
        # Lightweight Raw Waveform encoder (SincNet-style or simple 1D Conv)
        self.conv = nn.Sequential(
            # Large kernel for first layer (SincNet style intuition)
            nn.Conv1d(1, 16, kernel_size=128, stride=4, padding=64),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4),
            
            nn.Conv1d(16, 32, kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, kernel_size=32, stride=2, padding=16),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(16)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 16, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # x: [batch, samples]
        if x.dim() == 2:
            x = x.unsqueeze(1) # [batch, 1, samples]
            
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
