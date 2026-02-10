import torch
import pandas as pd
import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

# Add project root to path
# __file__ = spam_model/data_spam/precompute_features.py
# 1. spam_model/data_spam
# 2. spam_model
# 3. . (root call_detect)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spam_model.audio_loader.load_audio import AudioLoader
from spam_model.asr.transcribe import ASRTranscriber
from spam_model.audio_features.extract_embeddings import EmbeddingExtractor
from spam_model.text_features.extract_semantic_signals import TextFeatureExtractor
from spam_model.fusion.fuse_features import FeatureFuser

def precompute():
    config_path = "spam_model/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    metadata_path = config['paths']['metadata_file']
    data_dir = config['paths']['data_spam']
    features_dir = os.path.join(data_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    
    df = pd.read_csv(metadata_path)
    
    # Initialize Models Sequentially to avoid OOM
    # 1. ASR
    print("Initializing ASR...")
    asr = ASRTranscriber()
    
    transcripts = {}
    print("Step 1/4: Transcribing Audio...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fid = str(row['file_id'])
        
        # Check if already done (optional optimization, skip for now to ensure freshness)
        
        # Agent
        try:
            agent_txt = asr.transcribe(row['agent_audio_path'])
        except Exception as e:
            print(f"Error Agent {fid}: {e}")
            agent_txt = ""

        # Customer
        try:
            cust_txt = asr.transcribe(row['customer_audio_path'])
        except Exception as e:
            print(f"Error Customer {fid}: {e}")
            cust_txt = ""
            
        transcripts[fid] = (agent_txt, cust_txt)
        
    del asr
    torch.cuda.empty_cache()
    
    # 2. Audio Embeddings
    print("Step 2/4: Extracting Embeddings...")
    embedder = EmbeddingExtractor()
    loader = AudioLoader()
    
    embeddings = {}
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fid = str(row['file_id'])
        try:
            pair = loader.load_pair(row['agent_audio_path'], row['customer_audio_path'])
            agent_emb = embedder.extract(pair['agent'])
            cust_emb = embedder.extract(pair['customer'])
            embeddings[fid] = (agent_emb, cust_emb)
        except Exception as e:
            print(f"Error Embedding {fid}: {e}")
            # Zero vectors as fallback
            embeddings[fid] = (np.zeros(192), np.zeros(192))
            
    del embedder
    torch.cuda.empty_cache()
    
    # 3. Text Features & Fusion
    print("Step 3/4: Text Features & saving...")
    text_extractor = TextFeatureExtractor()
    fuser = FeatureFuser()
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fid = str(row['file_id'])
        agent_txt, cust_txt = transcripts.get(fid, ("", ""))
        agent_emb, cust_emb = embeddings.get(fid, (np.zeros(192), np.zeros(192)))
        
        # Extract Text Feats
        agent_tf = text_extractor.extract(agent_txt)
        cust_tf = text_extractor.extract(cust_txt)
        
        # Fuse
        fused = fuser.fuse(agent_emb, cust_emb, agent_tf, cust_tf)
        
        # Save
        save_path = os.path.join(features_dir, f"{fid}.pt")
        torch.save(fused, save_path)
        
    print(f"Done. Features saved to {features_dir}")

if __name__ == "__main__":
    precompute()
