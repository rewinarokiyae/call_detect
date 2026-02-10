import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import glob

# Config
DATA_ROOT = r"D:\Subject\honors\ai_project\call_detect\data\2026_data"
OUTPUT_DIR = r"D:\Subject\honors\ai_project\call_detect\data\precomputed"
SAMPLE_RATE = 16000
MAX_LEN = 64000 # 4s

PATHS = {
    "train": { "audio": os.path.join(DATA_ROOT, "flac_T"), "proto": os.path.join(DATA_ROOT, "ASVspoof5.train.tsv") },
    "dev":   { "audio": os.path.join(DATA_ROOT, "flac_D"), "proto": os.path.join(DATA_ROOT, "ASVspoof5.dev.track_1.tsv") }
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class FeatureExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # 1. WavLM (Torchaudio Native - Bypass Transformers Security)
        print("Loading Torchaudio WavLM Base...")
        bundle = torchaudio.pipelines.WAVLM_BASE
        self.wavlm = bundle.get_model().to(device)
        self.wavlm.eval()
        for p in self.wavlm.parameters():
            p.requires_grad = False
            
    def compute_wavlm(self, wav):
        # wav: (1, L) tensor
        with torch.no_grad():
            # Torchaudio extract_features returns a list of tensors (one per layer)
            features, _ = self.wavlm.extract_features(wav)
            # features: List[Tensor], each (1, T, 768)
            
            # Strategy: Mean + Std of Last Layer
            last_hidden = features[-1] # (1, T, 768)
            
            mean_pool = last_hidden.mean(dim=1).cpu().numpy().squeeze()
            std_pool = last_hidden.std(dim=1).cpu().numpy().squeeze()
            max_pool = last_hidden.max(dim=1).values.cpu().numpy().squeeze()
            
            return np.concatenate([mean_pool, std_pool, max_pool]) # 768*3 = 2304 dim

    def compute_hc(self, wav_numpy):
        # wav_numpy: (L,)
        # MFCC
        mfcc = librosa.feature.mfcc(y=wav_numpy, sr=SAMPLE_RATE, n_mfcc=20)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        
        # Spectral feats
        cent = librosa.feature.spectral_centroid(y=wav_numpy, sr=SAMPLE_RATE).mean()
        flat = librosa.feature.spectral_flatness(y=wav_numpy).mean()
        
        return np.concatenate([mfcc_mean, mfcc_std, [cent], [flat]])

def process_partition(name, paths, extractor, device):
    print(f"Processing {name}...")
    
    # Read Proto
    try:
        # Flexible separator
        df = pd.read_csv(paths['proto'], sep='\s+', header=None, engine='python')
        if len(df.columns) < 2: raise ValueError("Bad cols")
        # Col 1 is ID
        df.rename(columns={1: 'utterance_id', 8: 'label'}, inplace=True)
    except:
        print(f"Skipping {name} due to proto error")
        return

    # Output Dir
    part_dir = os.path.join(OUTPUT_DIR, name)
    ensure_dir(part_dir)
    
    # Audio Files (Scan once - Recursive)
    print("Scanning audio files...")
    audio_files = {}
    flac_files = glob.glob(os.path.join(paths['audio'], "**", "*.flac"), recursive=True)
    wav_files = glob.glob(os.path.join(paths['audio'], "**", "*.wav"), recursive=True)
    
    for f in flac_files + wav_files:
        audio_files[os.path.basename(f)] = f
             
    print(f"Found {len(audio_files)} files.")
    
    # Loop
    missing = 0
    skipped = 0
    processed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        uid = str(row['utterance_id'])
        out_path = os.path.join(part_dir, f"{uid}.npy")
        
        if os.path.exists(out_path):
            skipped += 1
            if skipped % 5000 == 0:
                 print(f"Skipped {skipped} existing...")
            continue 
            
        fname_flac = f"{uid}.flac"
        fname_wav = f"{uid}.wav"
        
        fpath = audio_files.get(fname_flac, audio_files.get(fname_wav))
        
        if not fpath:
            missing += 1
            if missing < 5: print(f"Missing: {uid}")
            continue
            
        # Load
        try:
            # Librosa for robustness
            wav_np, sr = librosa.load(fpath, sr=SAMPLE_RATE)
            
            # Pad/Cut
            if len(wav_np) < MAX_LEN:
                wav_np = np.pad(wav_np, (0, MAX_LEN - len(wav_np)))
            else:
                wav_np = wav_np[:MAX_LEN]
                
            wav = torch.from_numpy(wav_np).unsqueeze(0).float() # (1, L)
            
            # Extract
            # 1. WavLM (GPU)
            wav_cuda = wav.to(device)
            wavlm_emb = extractor.compute_wavlm(wav_cuda)
            
            # 2. HC (CPU)
            hc_emb = extractor.compute_hc(wav_np)
            
            # Fuse & Save
            final_emb = np.concatenate([wavlm_emb, hc_emb])
            np.save(out_path, final_emb.astype(np.float32))
            
            processed += 1
            
        except Exception as e:
            print(f"Error {uid}: {e}")
            
    print(f"Finished {name}. Processed: {processed}, Skipped: {skipped}, Missing: {missing}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    extractor = FeatureExtractor(device)
    
    # Process Train (Primary)
    process_partition('train', PATHS['train'], extractor, device)
    
    # Process Dev
    process_partition('dev', PATHS['dev'], extractor, device)

if __name__ == "__main__":
    main()
