import torch
import torch.nn as nn

class ScamClassifier(nn.Module):
    def __init__(self, input_dim=392): # 192*2 + 4*2 = 392
        super(ScamClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)
