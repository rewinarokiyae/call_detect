import pandas as pd
import torch
import re
from transformers import pipeline
import os
import logging

# Suppress Transformers Warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np

class RoleIdentifier:
    def __init__(self, vocab_path, model_name="facebook/wav2vec2-base-960h"):
        self.vocab_path = vocab_path
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Loading ASR model {model_name}...")
        try:
            self.asr = pipeline("automatic-speech-recognition", model=model_name, device=self.device)
        except:
            self.asr = pipeline("automatic-speech-recognition", model=model_name, device=-1)

        self.lexicon_data = self._load_lexicon()
        
        # Initialize Linguistic Verifier
        from .linguistic_verifier import LinguisticVerifier
        self.verifier = LinguisticVerifier()

    def _load_lexicon(self):
        print(f"Loading lexicon from {self.vocab_path}...")
        try:
            df = pd.read_excel(self.vocab_path)
            # Mandatory columns: role, content
            customer = set(df[df['role'] == 'customer']['content'].astype(str).str.lower())
            agent = set(df[df['role'] == 'agent']['content'].astype(str).str.lower())
            return {'customer': customer, 'agent': agent}
        except Exception as e:
            print(f"Error loading lexicon: {e}")
            return {'customer': set(), 'agent': set()}

    def transcribe(self, audio_path):
        try:
            # Handle empty/short files
            import soundfile as sf
            # Use soundfile to load WAV; robust and doesn't require ffmpeg for WAV
            y, sr = sf.read(audio_path)
            
            if len(y) < sr * 0.5: # < 0.5s
                return ""
                
            # Pass numpy array directly to ASR pipeline to avoid internal ffmpeg usage
            # transformers pipeline expects float32
            if y.dtype != np.float32:
                y = y.astype(np.float32)
                
            res = self.asr({"sampling_rate": sr, "raw": y})
            return res['text'].lower()
        except Exception as e:
            print(f"ASR Error: {e}")
            return ""

    def analyze_with_gemini(self, text, api_key):
        """
        Gemini Semantic Validation (Auxiliary).
        """
        if not api_key: return None
        import google.generativeai as genai
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"""
            Task: Semantic Validation for Call Center Speaker.
            Input Text: "{text}"
            
            Based on the tone and vocabulary (Institutional vs Emotional), is this an AGENT or CUSTOMER?
            Return JSON: {{ "role": "AGENT"|"CUSTOMER", "confidence": <float 0-1> }}
            """
            response = model.generate_content(prompt)
            import json
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass
        return None

    def extract_features(self, text):
        words = text.split()
        if not words:
            return {
                'agent_overlap': 0.0, 'cust_overlap': 0.0, 
                'pronoun_i': 0.0, 'pronoun_you_we': 0.0, 
                'modals': 0.0, 'institutional': 0.0,
                'word_count': 0.0
            }

        # 1. Lexicon Overlap (Strict)
        agent_matches = sum(1 for w in words if w in self.lexicon_data['agent'])
        cust_matches = sum(1 for w in words if w in self.lexicon_data['customer'])

        # 2. Pronouns
        # Customer: "I", "my", "me"
        # Agent: "You", "we", "us"
        pronoun_i = sum(1 for w in words if w in ['i', 'my', 'me', 'mine'])
        pronoun_you_we = sum(1 for w in words if w in ['you', 'your', 'we', 'our', 'us'])

        # 3. Modals & Institutional (Heuristic)
        # Agent: will, shall, can, may, policy, verify, ticket, system
        modals = sum(1 for w in words if w in ['will', 'shall', 'can', 'may'])
        institutional = sum(1 for w in words if w in ['policy', 'account', 'verify', 'system', 'ticket', 'procedure', 'check'])

        total = len(words)
        return {
            'agent_overlap': agent_matches / total,
            'cust_overlap': cust_matches / total,
            'pronoun_i': pronoun_i / total,
            'pronoun_you_we': pronoun_you_we / total,
            'modals': modals / total,
            'institutional': institutional / total,
            'word_count': total
        }

    def predict_role_robust(self, audio_path, api_key=None):
        text = self.transcribe(audio_path)
        feats = self.extract_features(text)
        
        # Linguistic Verification (Human Score)
        linguistic_score = self.verifier.analyze(text)
        
        # --- Scoring Logic ---
        # Agent Positive: Agent Overlap, We/You, Modals, Institutional
        # Customer Positive: Customer Overlap, I/My
        
        agent_score_raw = (
            feats['agent_overlap'] * 2.0 + 
            feats['pronoun_you_we'] * 1.0 + 
            feats['modals'] * 1.5 + 
            feats['institutional'] * 2.0
        )
        
        cust_score_raw = (
            feats['cust_overlap'] * 2.0 + 
            feats['pronoun_i'] * 1.5
        )
        
        # Softmax-like logic or simple difference
        # We need a confidence 0-1
        param_sum = agent_score_raw + cust_score_raw + 1e-6
        agent_conf = agent_score_raw / param_sum
        
        # --- Gemini Validation (Auxiliary) ---
        gemini_res = self.analyze_with_gemini(text, api_key)
        gemini_conf = 0.5
        if gemini_res:
            # Robust get: handle None values
            g_role = str(gemini_res.get('role') or '').lower()
            g_conf = float(gemini_res.get('confidence') or 0.5)
            
            if g_role == 'agent':
                gemini_conf = 0.5 + (g_conf / 2) # 0.5 to 1.0
            elif g_role == 'customer':
                gemini_conf = 0.5 - (g_conf / 2) # 0.5 to 0.0
        
        return {
            'text': text,
            'features': feats,
            'agent_confidence_lexicon': agent_conf,
            'agent_confidence_gemini': gemini_conf,
            'has_text': feats['word_count'] > 0,
            'linguistic_score': linguistic_score
        }
