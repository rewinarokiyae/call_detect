import re

def detect_account(text):
    text = text.lower()
    print(f"Testing text: '{text}'")
    
    # Current Logic
    if "account" in text:
        match = re.search(r'\b\d{9,18}\b', text)
        if match:
            print(f"  [CURRENT] MATCH: {match.group(0)}")
        else:
            print(f"  [CURRENT] NO MATCH")
    else:
        print("  [CURRENT] 'account' keyword missing")

    # Proposed Logic
    # 1. Normalize: Remove spaces between digits
    # Look for sequences of digits that might be spaced out
    # "1 2 3" -> "123"
    
    # Broadened keywords
    keywords = ["account", "number", "id", "savings", "checking"]
    if any(k in text for k in keywords):
        # Allow spaces/dashes between digits
        # This regex matches a digit, followed by optional space/dash, repeat 9-18 times
        # We capture the whole group to see what matched
        
        # Better approach: Extract all digit sequences, join them, check length
        digits_only = re.sub(r'[^0-9]', '', text)
        print(f"  [DEBUG] Digits found: {digits_only}")
        
        if len(digits_only) >= 9 and len(digits_only) <= 18:
             print(f"  [PROPOSED] MATCH (Length {len(digits_only)})")
        else:
             print(f"  [PROPOSED] NO MATCH (Length {len(digits_only)})")

print("--- Scenario 1: Perfect Transcript ---")
detect_account("here is my account number 1234567890")

print("\n--- Scenario 2: Spaced Digits (ASR Common) ---")
detect_account("my account is 1 2 3 4 5 6 7 8 9 0")

print("\n--- Scenario 3: Broken Context ---")
detect_account("i have an account. wait. it is 1234567890")

print("\n--- Scenario 4: User Example (Reluctant then reveals) ---")
detect_account("i dont want to give account. okay it is 1 2 3 4 5 6 7 8 9")
