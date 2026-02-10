import torch
import torch.nn as nn
import torchaudio

class SpectralBranch(nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=128
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64 mels
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32 mels
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 16)) # Force time dimension to fixed size for flattening
        )
        
        self.fc = nn.Linear(64 * 16, 64)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
            
        x = self.mel_spec(x) # [batch, n_mels, time]
        x = self.to_db(x)
        x = x.unsqueeze(1) # Add channel dim: [batch, 1, n_mels, time]
        
        x = self.encoder(x) # [batch, 64, 1, 16]
        x = x.view(x.size(0), -1)
        return self.fc(x)
