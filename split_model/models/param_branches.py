import torch
import torch.nn as nn

class FeatureBranch(nn.Module):
    def __init__(self, input_channels=1, output_dim=64):
        super().__init__()
        
        # Expecting input: [batch, 1, n_feats, time]
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 8)) # Pool to fixed time size
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        # x: [batch, n_feats, time] -> need to unsqueeze channel dim
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv(x) # [batch, 128, 1, 8]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LFCCBranch(FeatureBranch):
    def __init__(self):
        super().__init__()

class CQCCBranch(FeatureBranch):
    def __init__(self):
        super().__init__()
