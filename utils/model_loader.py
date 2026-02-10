import streamlit as st
import os
import sys
import torch
import shutil
import soundfile as sf
import re

# Adjust paths to include project root and split_model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "split_model"))

# Import Pipeline Components
from pipeline.diarization import Diarizer
from pipeline.role_id import RoleIdentifier
from pipeline.fusion import FusionClassifier

class UnifiedPipeline:
    def __init__(self):
        print("Initializing Unified Pipeline...")
        
        # 1. Diarizer (Pyannote / SpeechBrain)
        self.diarizer = Diarizer(n_clusters=2)
        
        # 2. Role Identifier (Wav2Vec2 + Lexicon)
        # Fix path to lexicon
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lexicon_path = os.path.join(base_dir, "split_model", "data", "customer_agent_vocabulary_10000_low_repeat.xlsx")
        
        self.role_id = RoleIdentifier(vocab_path=lexicon_path)
        self.fusion = FusionClassifier()
        
        print("Unified Pipeline Initialized.")

    def process(self, file_path, api_key=None):
        """
        Runs Diarization -> Role ID -> Splitting -> Local Spoof Check
        Returns a dict similar to what run_pipeline.py printed as JSON.
        """
        status_text = "Starting analysis..."
        
        # 1. Diarization
        print(f"Running Diarization on {file_path}")
        segments, y, sr = self.diarizer.process(file_path)
        
        # Save split audio
        # We need to save to record_split relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "record_split")
        
        segment_paths = self.diarizer.save_segments(segments, y, sr, output_dir=output_dir)
        
        # 2. Role Identification
        speaker_scores = {}
        speaker_texts = {}
        speaker_linguistic_scores = {}
        
        for spk_id, path in segment_paths.items():
            # Duration
            y_seg, sr_seg = sf.read(path)
            duration = len(y_seg) / sr_seg
            
            # Predict
            res = self.role_id.predict_role_robust(path, api_key=api_key)
            final_prob = self.fusion.fuse(res, duration)
            
            speaker_scores[spk_id] = final_prob
            speaker_texts[spk_id] = res['text']
            speaker_linguistic_scores[spk_id] = res.get('linguistic_score', 0.0)
            
        # Determine Agent
        if not speaker_scores:
            return {"error": "No speakers detected"}

        text_agent_id = max(speaker_scores, key=speaker_scores.get)
        
        # 2.5 Hybrid/Local Check (Optional but recommended for accuracy)
        # We can implement a simplified version here or call the external script if we want to keep it decoupled.
        # For speed, calling the script via subprocess is OK if the model loading IS THE SCRIPT.
        # WAIT: The Local Model also loads a model. 
        # For now, let's keep the Local Model check as a separate step in the UI (Step 2),
        # BUT we need to do the Role Confirmation part here.
        
        # To avoid circular imports or complex pathing to Local_Model from here, 
        # we can skip the "Deepfake Role Confirmation" for the fast path OR 
        # we rely on the user to run Step 2.
        
        # Current compromise: Use Text-based ID for speed.
        best_agent_id = text_agent_id 
        
        # Move files
        agent_dir = os.path.join(output_dir, "agent")
        cust_dir = os.path.join(output_dir, "customer")
        os.makedirs(agent_dir, exist_ok=True)
        os.makedirs(cust_dir, exist_ok=True)
        
        final_agent_path = ""
        final_cust_path = ""
        
        for spk_id, path in segment_paths.items():
            base = os.path.basename(path)
            if spk_id == best_agent_id:
                dest = os.path.join(agent_dir, base)
                shutil.move(path, dest)
                final_agent_path = dest
            else:
                dest = os.path.join(cust_dir, base)
                shutil.move(path, dest)
                final_cust_path = dest

        return {
            "status": "success",
            "agent_audio": final_agent_path,
            "customer_audio": final_cust_path,
            "spoof_score": 0.5 # Placeholder, Step 2 will refine this
        }

@st.cache_resource
def load_pipeline():
    """
    Singleton loader for the pipeline.
    """
    return UnifiedPipeline()
