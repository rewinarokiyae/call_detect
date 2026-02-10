import re
import math

class LinguisticVerifier:
    def __init__(self):
        # Fillers: Common hesitation markers
        self.fillers = {
            'um', 'uh', 'er', 'ah', 'hmm', 'uhh', 'umm', 'err', 
            'like', 'you know', 'i mean', 'sort of', 'kind of'
        }
        
        # Disfluency Regex (Repeated words like "the the", "I I", "at a at")
        # Matches word, optional 1 word in between, same word
        self.repetition_regex = re.compile(r'\b(\w+)\s+(?:\w+\s+)?\1\b', re.IGNORECASE)
        
        # Hesitation matches (tokens that are just fillers)
        self.filler_regex = re.compile(r'\b(um+|uh+|er+|ah+|hmm+)\b', re.IGNORECASE)

    def analyze(self, text):
        """
        Analyzes text for human linguistic markers.
        Returns a score 0.0 (Perfect AI-like) to 1.0 (Highly Disfluent Human).
        """
        if not text:
            return 0.0
            
        text_lower = text.lower()
        words = text_lower.split()
        total_words = len(words)
        if total_words == 0:
            return 0.0

        # 1. Detect Repetitions (Stuttering)
        repetitions = len(self.repetition_regex.findall(text_lower))
        
        # 2. Detect Fillers
        # Count explicit filler words
        filler_count = 0
        for w in words:
            # Clean punctuation
            w_clean = re.sub(r'[^\w]', '', w)
            if w_clean in self.fillers:
                filler_count += 1
            elif self.filler_regex.match(w_clean):
                filler_count += 1
        
        # 3. Calculate Rates
        repetition_rate = repetitions / total_words
        filler_rate = filler_count / total_words
        
        # 4. Scoring (Heuristic)
        # Humans often have > 1-2% disfluency rate in spontaneous speech.
        # AI (TTS) effectively has 0%.
        
        # We want a high score if significant disfluency is found.
        # Thresholds: 
        # - 1 filler in 10 words (10%) -> Very Human
        # - 1 repetition in 50 words (2%) -> Likely Human
        
        human_score = 0.0
        
        # Repetition contribution (Strong indicator)
        if repetition_rate > 0.005: human_score += 0.4
        if repetition_rate > 0.02: human_score += 0.4  # Bonus for high repetition
        
        # Filler contribution
        if filler_rate > 0.01: human_score += 0.3
        if filler_rate > 0.05: human_score += 0.3
        
        # Cap at 1.0
        score = min(human_score, 1.0)
        
        # DEBUG LOGGING (Direct to file)
        try:
            with open("ling_debug.txt", "a", encoding="utf-8") as f:
                f.write(f"\n--- Analyze Call ---\n")
                f.write(f"Text: {text}\n")
                f.write(f"Repetitions: {repetitions} ({repetition_rate:.4f})\n")
                f.write(f"Fillers: {filler_count} ({filler_rate:.4f})\n")
                f.write(f"Score: {score:.4f}\n")
        except:
            pass
            
        return score

if __name__ == "__main__":
    # Test
    verifier = LinguisticVerifier()
    
    samples = [
        "Hello this is a perfect sentence from an AI.",
        "Um I uh think that we need to like check the account.",
        "I I didn't expect this.",
        "The the system is down."
    ]
    
    for s in samples:
        print(f"'{s}' -> Score: {verifier.analyze(s):.2f}")
