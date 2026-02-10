import re
import yaml
import numpy as np

class TextFeatureExtractor:
    def __init__(self, config_path="spam_model/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sensitive_keywords = self.config['features']['text']['sensitive_keywords']
        self.urgency_keywords = self.config['features']['text']['urgency_keywords']

    def extract(self, text):
        """
        Extracts a feature vector from text.
        Vector: [has_sensitive, has_urgency, word_count, is_question]
        """
        text = text.lower()
        
        has_sensitive = any(k in text for k in self.sensitive_keywords)
        has_urgency = any(k in text for k in self.urgency_keywords)
        word_count = len(text.split())
        is_question = "?" in text
        
        # === GROQ SEMANTIC NORMALIZATION PLACEHOLDER ===
        # Purpose: Analyze text for semantic relationships (Family, Authority, etc.)
        # Interface:
        #   Input: text (str)
        #   Output: 
        #     - relationship_detected (bool)
        #     - relationship_type (str: "Family", "Friend", "Authority", "None")
        #     - confidence (float 0.0-1.0)
        
        relationship_detected = 0.0
        relationship_confidence = 0.0
        
        if self.config.get('groq', {}).get('enabled', False):
            # TODO: Implement Groq API Call
            # client = Groq(api_key=self.config['groq']['api_key'])
            # ...
            pass
        
        return np.array([
            1.0 if has_sensitive else 0.0,
            1.0 if has_urgency else 0.0,
            float(word_count),
            1.0 if is_question else 0.0,
            relationship_detected,
            relationship_confidence
        ], dtype=np.float32)
