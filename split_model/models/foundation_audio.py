import torch
import torch.nn as nn
from transformers import WavLMModel

class FoundationAudioBranch(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base-plus", freeze_layers=False):
        super().__init__()
        # Load pre-trained WavLM
        # FORCE safetensors to bypass torch.load vulnerability check on torch < 2.6
        self.wavlm = WavLMModel.from_pretrained(model_name, use_safetensors=True)
        
        if freeze_layers:
            for param in self.wavlm.parameters():
                param.requires_grad = False
        else:
            # Enable Gradient Checkpointing to save VRAM on RTX 3050
            self.wavlm.gradient_checkpointing_enable() 
            for param in self.wavlm.parameters():
                param.requires_grad = True
        
        # WavLM base hidden size is 768
        self.fc = nn.Linear(768, 64)

    def forward(self, x):
        # x: [batch, samples] or [batch, 1, samples]
        if x.dim() == 3:
            x = x.squeeze(1)
            
        # WavLM expects raw audio values [batch, time]
        # REMOVED no_grad to allow fine-tuning
        outputs = self.wavlm(x)
        
        # Take last hidden state: [batch, time, 768]
        last_hidden = outputs.last_hidden_state
        
        # Mean pooling over time
        pooled = torch.mean(last_hidden, dim=1) # [batch, 768]
        
        return self.fc(pooled)
