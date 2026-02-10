import numpy as np

class FusionClassifier:
    def __init__(self):
        pass

    def fuse(self, role_result, audio_duration):
        """
        Multimodal Fusion.
        Inputs:
            role_result: Dict from RoleIdentifier.predict_role_robust
            audio_duration: float (seconds)
        
        Returns:
            final_agent_probability: float
        """
        # 1. Lexicon Score
        lex_score = role_result['agent_confidence_lexicon']
        
        # 2. Gemini Score (Auxiliary)
        gem_score = role_result['agent_confidence_gemini']
        
        # 3. Audio Heuristics (Optional, keep simple if not training a model)
        # We assume Diarization handles separation. Here we just identify role.
        
        # Weights
        # If text is very short, mistrust lexicon
        wc = role_result['features']['word_count']
        
        w_lex = 0.7
        w_gem = 0.3
        
        if wc < 5:
            # Low text confidence -> rely more on Gemini if available or neutral
            w_lex = 0.3
            w_gem = 0.7
        
        # Combine
        combined_score = w_lex * lex_score + w_gem * gem_score
        
        # Penalties
        # Very short duration (<10s) speakers are usually NOT the primary agent handling the call
        # (Though could be automated messages). 
        # But if it's <3s it's likely noise/interruption.
        if audio_duration < 3.0:
            combined_score *= 0.5 # Penalty
            
        return combined_score
