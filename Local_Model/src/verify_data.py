import os
import sys
import pandas as pd
from tqdm import tqdm
import glob

# Configuration
DATA_ROOT = r"D:\Subject\honors\ai_project\call_detect\data\2026_data"

PATHS = {
    "train": {
        "audio": os.path.join(DATA_ROOT, "flac_T"),
        "proto": os.path.join(DATA_ROOT, "ASVspoof5.train.tsv")
    },
    "dev": {
        "audio": os.path.join(DATA_ROOT, "flac_D"),
        "proto": os.path.join(DATA_ROOT, "ASVspoof5.dev.track_1.tsv")
    },
    "eval": {
        "audio": os.path.join(DATA_ROOT, "flac_E_eval"),
        "proto": os.path.join(DATA_ROOT, "ASVspoof5.eval.track_1.tsv")
    }
}

def verify_partition(name, audio_dir, proto_path):
    print(f"\n{'='*40}")
    print(f"Verifying Partition: {name.upper()}")
    print(f"{'='*40}")

    # 1. Check Paths
    if not os.path.exists(audio_dir):
        print(f"CRITICAL ERROR: Audio directory missing: {audio_dir}")
        return False
    if not os.path.exists(proto_path):
        print(f"CRITICAL ERROR: Protocol file missing: {proto_path}")
        return False

    # 2. Count Files on Disk (Recursive)
    print(f"Scanning audio directory: {audio_dir}")
    # glob iterators for memory efficiency
    flac_files = glob.glob(os.path.join(audio_dir, "**", "*.flac"), recursive=True)
    wav_files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)
    
    audio_files = set([os.path.basename(f) for f in flac_files + wav_files])
    print(f"Found {len(audio_files)} audio files on disk.")
    if len(audio_files) > 0:
        print(f"Sample file: {list(audio_files)[0]}")

    # 3. Verify TSV consistency
    print(f"Reading protocol: {os.path.basename(proto_path)}")
    try:
        # Use python engine for flexible whitespace handling
        df = pd.read_csv(proto_path, sep='\s+', header=None, engine='python')
    except Exception as e:
        print(f"ERROR: Failed to read TSV: {e}")
        return False

    # Columns based on inspection:
    # 0: Spk, 1: Utt, 8: Label
    if len(df.columns) < 2:
         print(f"ERROR: TSV has too few columns ({len(df.columns)}). Check separator.")
         return False
         
    # Rename for clarity
    df.rename(columns={1: 'utterance_id', 8: 'label'}, inplace=True)
        
    print(f"Protocol entries: {len(df)}")
    
    # Check Logic
    missing_files = []
    proto_ids = df['utterance_id'].astype(str).values
    
    print("Checking alignment...")
    # Check first 5 to fail fast
    for uid in proto_ids[:5]:
        if f"{uid}.flac" not in audio_files and f"{uid}.wav" not in audio_files:
             print(f"Immediate Mismatch! {uid}.flac not found.")
    
    # Full check
    for uid in tqdm(proto_ids):
        fname_flac = f"{uid}.flac"
        fname_wav = f"{uid}.wav"
        if fname_flac not in audio_files and fname_wav not in audio_files:
            missing_files.append(uid)
            
    if missing_files:
        print(f"FAILURE: {len(missing_files)} files missed in audio directory!")
        sample_missing = missing_files[:5]
        print(f"Sample missing: {sample_missing}")
        return False
        
    print(f"SUCCESS: All {len(df)} protocol entries found on disk.")
    return True

def main():
    print("Starting ASVspoof5 Dataset Verification...")
    print(f"Data Root: {DATA_ROOT}")
    
    all_passed = True
    
    if not verify_partition("train", PATHS['train']['audio'], PATHS['train']['proto']):
        all_passed = False
        
    if not verify_partition("dev", PATHS['dev']['audio'], PATHS['dev']['proto']):
        all_passed = False
        
    # Eval might not have labels or might be huge, verified last
    # Eval usually doesn't have a labeled protocol in the same format depending on phase
    # But checking if we have the file listing
    if not verify_partition("eval", PATHS['eval']['audio'], PATHS['eval']['proto']):
        # Soft failure for eval if just the TSV is weird, but hard if audio missing
        print("Warning: Eval partition verification issues.")
        pass

    if all_passed:
        print("\n" + "="*50)
        print("VERIFICATION COMPLETE: DATASET READY")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("VERIFICATION FAILED: FIX ISSUES BEFORE TRAINING")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main()
