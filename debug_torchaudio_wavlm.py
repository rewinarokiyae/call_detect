import torch
import torchaudio

def test_ta_wavlm():
    print("Testing Torchaudio WavLM...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    try:
        bundle = torchaudio.pipelines.WAVLM_BASE
        print(f"Loading bundle: {bundle}")
        model = bundle.get_model().to(device)
        print("Model loaded successfully!")
        
        # Test forward
        wav = torch.randn(1, 16000).to(device)
        with torch.no_grad():
            features, _ = model.extract_features(wav)
            # features is list of tensors
            print(f"Features type: {type(features)}")
            print(f"Num layers: {len(features)}")
            print(f"Shape: {features[-1].shape}")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ta_wavlm()
