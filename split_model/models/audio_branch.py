import torch
import torch.nn as nn
import torchaudio

class AudioBranch(nn.Module):
    def __init__(self, sample_rate=16000, n_mfcc=40, hidden_dim=128):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 64}
        )
        # LFCC would be similar; for reconstruction we stick to MFCC as primary handcrafted feature
        
        # 1D CNN for local feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.bilstm = nn.LSTM(
            input_size=128, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Tanh()
        )
        
        self.fc = nn.Linear(hidden_dim * 2, 64)

    def forward(self, x):
        # x: [batch, 1, samples] or [batch, samples]
        if x.dim() == 3:
            x = x.squeeze(1) # [batch, samples]
            
        # FIX: Add epsilon noise to prevent Log(0) -> -Inf in MFCC on silent/padded parts
        x = x + (1e-6 * torch.randn_like(x))
        
        x = self.mfcc(x)  # [batch, n_mfcc, time]
        
        # CNN expects [batch, channels, time]
        x = self.cnn(x) # [batch, 128, reduced_time]
        
        # LSTM expects [batch, time, features]
        x = x.permute(0, 2, 1) 
        output, (hn, cn) = self.bilstm(x) # [batch, time, hidden*2]
        
        # Self-Attention Pooling
        attn_weights = torch.softmax(self.attention(output), dim=1)
        x = torch.sum(output * attn_weights, dim=1) # [batch, hidden*2]
        
        return self.fc(x)
