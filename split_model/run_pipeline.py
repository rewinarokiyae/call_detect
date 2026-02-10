import os
import sys
import certifi
import logging

# Suppress Transformers Warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# FIX: Force usage of certifi's CA bundle to avoid SSL errors on Windows
# Must be set BEFORE other imports that might initialize SSL contexts
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import argparse

import os
import sys
import torch
import soundfile as sf
import shutil
import numpy as np
import io

# Force UTF-8 for Windows Console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')



# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- MONKEY PATCH TOP-LEVEL ---
import types
try:
    import torchaudio
    if not hasattr(torchaudio, 'backend'):
        torchaudio.backend = types.ModuleType("backend")
        sys.modules["torchaudio.backend"] = torchaudio.backend
    if not hasattr(torchaudio.backend, "list_audio_backends"):
        torchaudio.backend.list_audio_backends = lambda: []
except:
    pass
# ------------------------------

from pipeline.diarization import Diarizer
from pipeline.role_id import RoleIdentifier
from pipeline.fusion import FusionClassifier
from run_inference import load_model, preprocess_audio, HybridDetectModel

# Global Config
DATA_DIR = "data"
LEXICON_PATH = os.path.join("split_model", "data", "customer_agent_vocabulary_10000_low_repeat.xlsx")

def process_conversation(file_path, api_key=None):
    print("="*50)
    print("STARTING CALL INTELLIGENCE PIPELINE")
    print("="*50)

    # 1. Diarization (Advanced)
    print("\n[STAGE 1] Advanced Speaker Diarization")
    diarizer = Diarizer(n_clusters=2) 
    segments, y, sr = diarizer.process(file_path)
    
    # Save split audio immediately (PCM_16)
    segment_paths = diarizer.save_segments(segments, y, sr, output_dir="record_split")
    
    # 2. Role Identification & Fusion
    print("\n[STAGE 2 & 3] Robust Role Identification")
    role_id = RoleIdentifier(vocab_path=LEXICON_PATH)
    fusion = FusionClassifier()
    
    speaker_scores = {}
    speaker_texts = {}
    speaker_linguistic_scores = {}
    
    for spk_id, path in segment_paths.items():
        print(f"\nAnalyzing Speaker {spk_id}...")
        
        # Audio Duration
        y_seg, sr_seg = sf.read(path)
        duration = len(y_seg) / sr_seg
        
        # Predict
        res = role_id.predict_role_robust(path, api_key=api_key)
        
        # Fusion
        final_prob = fusion.fuse(res, duration)
        speaker_scores[spk_id] = final_prob
        speaker_texts[spk_id] = res['text']
        speaker_linguistic_scores[spk_id] = res.get('linguistic_score', 0.0)
        
        print(f"Text: {res['text'][:50]}...")
        print(f"Scores -> Lexicon: {res['agent_confidence_lexicon']:.2f}, Gemini: {res['agent_confidence_gemini']:.2f}")
        print(f">> FINAL AGENT PROBABILITY: {final_prob:.3f}")

    # Determine Agent (Text/Probability Based)
    if not speaker_scores:
        return {"error": "No speakers detected"}

    text_agent_id = max(speaker_scores, key=speaker_scores.get)
    print(f"\n>> CANDIDATE (Text/Lexicon): Speaker {text_agent_id}")
    
    # --- PROPOSED FIX: Hybrid Role Identification ---
    # Run Local_Model on ALL speakers. If one is clearly AI, they are the Agent.
    print("\n[STAGE 2.5] Analyzing Voice Spoofing for Role Confirmation...")
    
    import subprocess
    import re
    
    inf_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "Local_Model", "src", "inference.py")
    python_exe = sys.executable 
    
    max_spoof_score = -1.0
    spoof_agent_id = -1
    
    speaker_spoof_scores = {}

    for spk_id, path in segment_paths.items():
        try:
             cmd = [python_exe, inf_script, path]
             # Run inference (capture output to parse score)
             res = subprocess.run(cmd, capture_output=True, text=True)
             
             # Parse "Spoof Score: 0.1234"
             score_match = re.search(r"Spoof Score:\s+([0-9.]+)", res.stdout)
             median_match = re.search(r"Median Score:\s+([0-9.]+)", res.stdout)
             
             if score_match:
                 score = float(score_match.group(1))
                 median = float(median_match.group(1)) if median_match else 0.0
                 
                 print(f"Speaker {spk_id} Raw Spoof Score: {score:.4f}, Median: {median:.4f}")
                 
                 # --- FILTER FOR WHATSAPP/OGG ARTIFACTS ---
                 # Observations:
                 # - True AI (Sparse): Low Median (< 0.05), High Max (> 0.8) -> KEEP
                 # - True AI (Strong): High Median (> 0.2), Extreme Max (> 0.99) -> KEEP
                 # - Artifact (WhatsApp): Medium Median (> 0.2), High Max (< 0.99) -> REJECT
                 
                 is_artifact = False
                 if score > 0.5:
                     if median > 0.2 and score < 0.99:
                         is_artifact = True
                         print(f"[WARN] ARTIFACT DETECTED: Consistent 'mushy' AI score (Med {median:.2f}). Likely Compression/Noise.")
                         print(f"   Ignoring this speaker as potential False Positive.")
                         score = 0.0 
                 
                 speaker_spoof_scores[spk_id] = score
                 
                 if score > max_spoof_score:
                     max_spoof_score = score
                     spoof_agent_id = spk_id
             else:
                 print(f"Could not parse score for Speaker {spk_id}")
                 
        except Exception as e:
            print(f"Error checking spoof for speaker {spk_id}: {e}")

    # Decision Logic
    best_agent_id = text_agent_id
    
    # Threshold for override: 0.5 (Same as detection threshold)
    if max_spoof_score > 0.5:
        if spoof_agent_id != text_agent_id:
             print(f"[ALERT] OVERRIDE: Speaker {spoof_agent_id} has High AI Probability ({max_spoof_score:.4f}).")
             print(f"   Switching Agent Role from {text_agent_id} to {spoof_agent_id}")
             best_agent_id = spoof_agent_id
        else:
             print(f"[OK] Confirmation: Text and Audio Analysis agree on Speaker {best_agent_id}.")
    else:
        print(f"[INFO] No strong AI signal detected (Max Score: {max_spoof_score:.4f}). sticking with Text-based Agent: {text_agent_id}")

    print(f"\n>> FINAL IDENTIFIED AGENT: Speaker {best_agent_id}")
    
    # Move files to final folders
    agent_dir = os.path.join("record_split", "agent")
    cust_dir = os.path.join("record_split", "customer")
    os.makedirs(agent_dir, exist_ok=True)
    os.makedirs(cust_dir, exist_ok=True)
    
    final_agent_path = ""
    
    for spk_id, path in segment_paths.items():
        base = os.path.basename(path)
        if spk_id == best_agent_id:
            dest = os.path.join(agent_dir, base)
            shutil.move(path, dest)
            final_agent_path = dest
        else:
            dest = os.path.join(cust_dir, base)
            shutil.move(path, dest)
            
    print(f"Saved Agent audio to: {final_agent_path}")

    # 3. AI Detection (Switched to Local_Model ASVspoof5)
    verdict_data = {"verdict": "NOT_RUN", "confidence": 0.0, "reasons": []}
    
    if final_agent_path and os.path.exists(final_agent_path):
        print("\n[STAGE 5] AI Voice Detection on AGENT Audio (Local_Model)")
        
        import subprocess
        import re
        
        # Path to inference script
        inf_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "Local_Model", "src", "inference.py")
        
        python_exe = sys.executable 
        
        try:
            cmd = [python_exe, inf_script, final_agent_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            output = result.stdout
            print(output) # Show logs
            
            # Parse Score
            # Expected: "Spoof Score: 0.9848"
            score_match = re.search(r"Spoof Score:\s+([0-9.]+)", output)
            if score_match:
                score = float(score_match.group(1))
                
                # --- LINGUISTIC OVERRIDE ---
                ling_score = speaker_linguistic_scores.get(best_agent_id, 0.0)
                print(f"Linguistic Human Score: {ling_score:.4f}")
                
                if score > 0.5:
                    # AI Detected. Check if we should override based on disfluency.
                    if ling_score > 0.5: # Threshold: High human-like features (significant stuttering/fillers)
                        print(f"[WARN] LINGUISTIC OVERRIDE: Human Disfluencies Detected (Score {ling_score:.2f}).")
                        print(f"   The audio is clean/high-quality (High AI Score), but the TEXT is distinctly Human.")
                        print(f"   Reclassifying as BONAFIDE.")
                        score = 0.1 # Force low score
                        verdict = "BONAFIDE"
                        conf = 1.0 - score
                        reasons = [f"Linguistic Override (Disfluency {ling_score:.2f})", "Human Disfluencies Detected"]
                    else:
                        verdict = "SPOOF"
                        conf = score
                        reasons = ["High AI Probability"]
                else:
                    verdict = "BONAFIDE"
                    conf = 1.0 - score
                    reasons = ["Low AI Probability"]
                    
                verdict_data = {
                    "verdict": verdict,
                    "confidence": conf,
                    "reasons": reasons,
                    "score": score
                }
                print(f"Parsed Verdict: {verdict} ({conf:.2f})")
            else:
                print("Could not parse score from inference output.")
                
        except Exception as e:
            print(f"Error running local inference: {e}")

    final_cust_path = ""
    for spk_id, path in segment_paths.items():
        base = os.path.basename(path)
        if spk_id != best_agent_id:
            # We already moved it, reconstruct path
            final_cust_path = os.path.join(cust_dir, base)

    return {
        "status": "success",
        "agent_id": best_agent_id,
        "agent_audio": f"/recordings/agent/{os.path.basename(final_agent_path)}".replace("\\", "/"),
        "customer_audio": f"/recordings/customer/{os.path.basename(final_cust_path)}".replace("\\", "/") if final_cust_path else None,
        "transcript_snippet": speaker_texts.get(best_agent_id, "")[:200],
        "analysis": verdict_data
    }

def main(args):
    result = process_conversation(args.file, args.api_key)
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Input audio file")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API Key")
    args = parser.parse_args()
    
    if os.path.exists(args.file):
        main(args)
    else:
        print("File not found.")
