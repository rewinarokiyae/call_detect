import torch
import torch.nn as nn

from .param_branches import LFCCBranch, CQCCBranch
from .raw_branch import RawBranch
from .prosody_head import ProsodyBranch

class HybridDetectModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lfcc = LFCCBranch()
        self.cqcc = CQCCBranch()
        self.raw = RawBranch()
        self.prosody = ProsodyBranch()
        
        # 4 branches, each outputting 64 features
        feature_dim = 64 * 4
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, 4), # 4 weights
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 3) # [SPOOF, BONAFIDE_CLEAN, BONAFIDE_DEGRADED]
        )
        
        # Auxiliary Head (Binary: Is Spoof?)
        self.aux_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )
        
    def forward(self, lfcc, cqcc, raw, prosody_vec, return_embedding=False):
        # 1. Branch Encoders
        emb_l = self.lfcc(lfcc) # [batch, 64]
        emb_c = self.cqcc(cqcc) # [batch, 64]
        emb_r = self.raw(raw)   # [batch, 64]
        emb_p = self.prosody(prosody_vec) # [batch, 64]
        
        # 2. Attention Gate
        # Stack: [batch, 4, 64]
        stack = torch.stack([emb_l, emb_c, emb_r, emb_p], dim=1)
        
        # Concat for Gate: [batch, 256]
        concat = torch.cat([emb_l, emb_c, emb_r, emb_p], dim=1)
        weights = self.gate(concat) # [batch, 4]
        
        # Apply Weights
        w = weights.unsqueeze(-1)
        
        # Weighted Sum
        fused = torch.sum(stack * w, dim=1) # [batch, 64]
        
        # 3. Classifier
        logits = self.classifier(fused)
        aux_logits = self.aux_head(fused)
        
        if return_embedding:
            return logits, weights, fused, aux_logits
            
        return logits, weights
