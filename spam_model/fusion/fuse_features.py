import torch
import numpy as np

class FeatureFuser:
    def __init__(self):
        pass
        
    def fuse(self, agent_emb, customer_emb, agent_text_feat, customer_text_feat):
        """
        Concatenate all features into a single vector.
        Features must be numpy arrays or tensors.
        """
        # Ensure all are 1D numpy arrays
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy().flatten()
            return np.array(x).flatten()
            
        a_emb = to_numpy(agent_emb)
        c_emb = to_numpy(customer_emb)
        a_txt = to_numpy(agent_text_feat)
        c_txt = to_numpy(customer_text_feat)
        
        # Concatenate: [Agent Audio (192), Customer Audio (192), Agent Text (4), Customer Text (4)]
        fused = np.concatenate([a_emb, c_emb, a_txt, c_txt])
        
        return torch.from_numpy(fused).float()
