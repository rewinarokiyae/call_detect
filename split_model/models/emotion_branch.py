import torch
import torch.nn as nn
import torchaudio

class EmotionBranch(nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        # Emotion often relies on pitch/tone, wider frequency analysis can help?
        # Using a standard spectrogram with high n_fft
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=2048,
            hop_length=512
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Approx output dim calculation depends on exact input length
        # We will use AdaptiveAvgPool to handle variable lengths before LSTM
        self.pool = nn.AdaptiveAvgPool2d((16, 32)) # (Freq, Time)
        
        # Flattened dim = Channels(32) * Freq(16) * Time(32) = 16384
        # We reduce this complexity
        self.pre_lstm = nn.Sequential(
             nn.Linear(16384, 126), # Reserve 2 spots for Jitter/Shimmer
             nn.ReLU()
        )
        
        # LSTM Input: 126 (CNN) + 2 (J/S) = 128
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 64)

    def extract_micro_stability(self, x):
        """
        Approximate Jitter/Shimmer using zero-crossing variation and envelope variation.
        True Jitter requires Parselmouth (slow), we use rapid tensor approximations here.
        """
        # 1. Zero Crossing Rate Variability (Proxy for Jitter/Frequency stability)
        zcr = torchaudio.functional.compute_deltas(torch.abs(x))
        zcr_var = torch.std(zcr, dim=1, keepdim=True)
        
        # 2. Envelope Variability (Proxy for Shimmer/Amplitude stability)
        env = torch.abs(x)
        env_var = torch.std(env, dim=1, keepdim=True)
        
        return torch.cat([zcr_var, env_var], dim=1) # [batch, 2]

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
            
        # Extract Stability Features (Jitter/Shimmer proxies)
        stability_feats = self.extract_micro_stability(x) # [batch, 2]
            
        spec = self.spec(x)
        spec = self.to_db(spec)
        spec = spec.unsqueeze(1) # [batch, 1, freq, time]
        
        cnn_out = self.cnn(spec) 
        cnn_out = self.pool(cnn_out) # [batch, 32, 16, 32]
        
        cnn_flat = cnn_out.view(cnn_out.size(0), -1) # Flatten
        
        # Reduce CNN output
        cnn_reduced = self.pre_lstm(cnn_flat) # [batch, 126]
        
        # Fuse
        combined = torch.cat([cnn_reduced, stability_feats], dim=1) # [batch, 128]
        combined = combined.unsqueeze(1) # [batch, 1, 128] Fake seq len
        
        out, _ = self.lstm(combined)
        out = out[:, -1, :] # Take last step
        
        return self.fc(out)
