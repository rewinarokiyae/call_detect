import pandas as pd
import os
import sys
import yaml
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def prepare_splits(config_path="spam_model/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    metadata_path = config['paths']['metadata_file']
    splits_dir = config['paths']['splits_dir']
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    # Verify columns
    required_cols = ['file_id', 'agent_audio_path', 'customer_audio_path', 'label', 'transcript_agent', 'transcript_customer']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return

    # Check for minimal data
    if len(df) < 5:
        print("Warning: Dataset too small for proper splitting. Duplicating for testing...")
        df = pd.concat([df]*3, ignore_index=True)

    # Stratified Split if enough data, else random
    # Train: 70%, Temp: 30%
    stratify_col = df['label'] if len(df) > 20 else None
    
    try:
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=stratify_col, random_state=42)
    except ValueError:
        # Fallback if stratification fails (e.g. only 1 class in small sample)
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=None, random_state=42)
    
    # Val: 15% (original), Test: 15% (original) -> Split Temp 50/50
    # Update stratify for temp
    stratify_temp = temp_df['label'] if len(temp_df) > 10 else None
    try:
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=stratify_temp, random_state=42)
    except ValueError:
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=None, random_state=42)
    
    # Save splits
    os.makedirs(splits_dir, exist_ok=True)
    
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("Data Preparation Complete.")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Splits saved to {splits_dir}")

if __name__ == "__main__":
    prepare_splits()
