import argparse
import torch
import os
import sys
import json
import logging

# Suppress Transformers Warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Ensure checking path for importsh
# __file__ = spam_model/inference/predict.py
# dirname = spam_model/inference
# dirname = spam_model
# dirname = . (root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spam_model.audio_loader.load_audio import AudioLoader
from spam_model.asr.transcribe import ASRTranscriber
from spam_model.audio_features.extract_embeddings import EmbeddingExtractor
from spam_model.text_features.extract_semantic_signals import TextFeatureExtractor
from spam_model.fusion.fuse_features import FeatureFuser
from spam_model.model.classifier import ScamClassifier
from spam_model.reports.report_generator import ReportGenerator

def main(agent_audio, customer_audio, spoof_score=0.0, spoof_verdict="UNKNOWN"):
    print("="*50)
    print("SCAM DETECTION PIPELINE: INFERENCE START")
    print("="*50)
    
    # 1. Load Audio
    try:
        print(f"[1/5] Loading Audio...")
        loader = AudioLoader()
        audio_data = loader.load_pair(agent_audio, customer_audio)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # 2. ASR (Sequential to save VRAM)
    try:
        print(f"[2/5] Running ASR (Whisper)...")
        transcriber = ASRTranscriber()
        
        agent_msg = transcriber.transcribe(agent_audio)
        if not agent_msg: agent_msg = ""
        print(f"   > Agent: {agent_msg[:50]}...")
        
        cust_msg = transcriber.transcribe(customer_audio)
        if not cust_msg: cust_msg = ""
        print(f"   > Customer: {cust_msg[:50]}...")
        
        del transcriber # Explicit cleanup
    except Exception as e:
        print(f"Error in ASR: {e}")
        agent_msg = ""
        cust_msg = ""

    # 3. Audio Embeddings (Sequential)
    try:
        print(f"[3/5] Extracting Audio Embeddings...")
        extractor = EmbeddingExtractor()
        
        if audio_data['agent'].shape[0] < 1600: # < 0.1s
            print("Warning: Agent audio too short/silent. Using zero embedding.")
            agent_emb = np.zeros(192)
        else:
            agent_emb = extractor.extract(audio_data['agent'])

        if audio_data['customer'].shape[0] < 1600:
             print("Warning: Customer audio too short/silent. Using zero embedding.")
             cust_emb = np.zeros(192)
        else:
            cust_emb = extractor.extract(audio_data['customer'])
        
        del extractor # Explicit cleanup
    except Exception as e:
        print(f"Error in Embeddings: {e}")
        import numpy as np
        agent_emb = np.zeros(192)
        cust_emb = np.zeros(192)

    # 4. Text Features
    try:
        print(f"[4/5] Extracting Text Features...")
        text_extractor = TextFeatureExtractor()
        
        agent_txt_feat = text_extractor.extract(agent_msg)
        cust_txt_feat = text_extractor.extract(cust_msg)
    except Exception as e:
        print(f"Error in Text Features: {e}")
        agent_txt_feat = np.zeros(6)
        cust_txt_feat = np.zeros(6)
    
    # 5. Fusion & Classification
    try:
        print(f"[5/5] Fusion & Classification...")
        fuser = FeatureFuser()
        final_input = fuser.fuse(agent_emb, cust_emb, agent_txt_feat, cust_txt_feat)
        
        # Load Config for Threshold
        import yaml
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_full_path = os.path.join(base_dir, "config.yaml")
        with open(config_full_path, 'r') as f:
            config = yaml.safe_load(f)
        threshold = config['models']['classifier'].get('threshold', 0.5)

        # Load Model
        model = ScamClassifier(input_dim=len(final_input))
        ckpt_path = os.path.join(base_dir, "checkpoints", "best_model.pth")
        if not os.path.exists(ckpt_path):
             ckpt_path = os.path.join(base_dir, "checkpoints", "last_model.pth")
        
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path))
            model.eval()
        else:
            print("WARNING: Model checkpoint not found. Using variable initialization.")
            
        with torch.no_grad():
            score = model(final_input.unsqueeze(0))
            prob = score.item()
            
        label = "SCAM" if prob > threshold else "BONAFIDE"
        
        print("\n" + "="*50)
        print(f"FINAL RESULT: {label}")
        print(f"Probability: {prob:.4f} (Threshold: {threshold})")
        print("="*50)

        # 6. Report Generation
        try:
            print(f"[6/6] Generating Security Report...")
            
            # Collect Risk Indicators
            risk_indicators = []
            
            # Audio Indicators
            if audio_data['agent'].shape[0] < 1600:
                risk_indicators.append("Agent audio too short/silent")
            if audio_data['customer'].shape[0] < 1600:
                risk_indicators.append("Customer audio too short/silent")
                
            # Text Indicators
            urgency_keywords = ["immediate", "block", "suspend", "expire", "unauthorized", "verify now"]
            sensitive_keywords = ["otp", "pin", "cvv", "card number", "account number", "password"]
            
            agent_lower = agent_msg.lower() if agent_msg else ""
            if any(k in agent_lower for k in urgency_keywords):
                risk_indicators.append("Urgency indicators detected in Agent speech")
            if any(k in agent_lower for k in sensitive_keywords):
                found_keywords = [k for k in sensitive_keywords if k in agent_lower]
                risk_indicators.append(f"Agent requested sensitive information: {', '.join(found_keywords)}")
            
            # Add Spoof Indicators
            if spoof_score > 0.5:
                risk_indicators.append(f"Potential Voice Spoofing Detected (Score: {spoof_score:.2f})")

            reporter = ReportGenerator()
            # Generate a time-based ID for the report
            import time
            call_id = f"call_{int(time.time())}"
            
            # Create a combined metadata object for the report logic if needed
            # For now, just pass the indicators.
            
            # Determine Agent Type
            agent_type = "Unknown"
            if spoof_verdict == "SPOOF":
                agent_type = "AI Robot"
            elif spoof_verdict == "BONAFIDE":
                agent_type = "Human"

            report_data, report_text = reporter.generate_report(
                call_id=call_id,
                scam_prob=prob,
                agent_text=agent_msg,
                customer_text=cust_msg,
                risk_indicators=risk_indicators,
                agent_type=agent_type
            )
            
            print(report_text)
            print(f"Report saved to: {reporter.reports_dir}")

        except Exception as e:
            print(f"Error in Reporting: {e}")
    except Exception as e:
        print(f"Error in Classification: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="Path to agent audio wav")
    parser.add_argument("--customer", required=True, help="Path to customer audio wav")
    parser.add_argument("--spoof_score", type=float, default=0.0, help="Spoof/Deepfake Score from Local Model")
    parser.add_argument("--spoof_verdict", type=str, default="UNKNOWN", help="Spoof Verdict")
    args = parser.parse_args()
    
    main(args.agent, args.customer, args.spoof_score, args.spoof_verdict)
