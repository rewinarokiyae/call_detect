import os
import torch

# PATCH FIRST
print("Patching torch.load...")
_original_load = torch.load
def unsafe_load(*args, **kwargs):
    print(f"Intercepted torch.load with args: {args}, kwargs: {kwargs}")
    # Force unsafe
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = unsafe_load

from transformers import WavLMModel

def test_load():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    try:
        print("Loading model...")
        model = WavLMModel.from_pretrained("microsoft/wavlm-base").to(device)
        print("Success!")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
