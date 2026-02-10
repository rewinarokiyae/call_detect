import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from spam_model.inference.predict import main as predict_main

def test_scenarios():
    print("========================================")
    print("EDGE CASE SCENARIO TESTING")
    print("========================================")
    
    # Define test cases (using available audio for now)
    scenarios = [
        ("Family Member Call", "record_split/agent/speaker_0.wav", "record_split/customer/speaker_1.wav"),
        ("Calm Legitimate Agent", "record_split/agent/speaker_0.wav", "record_split/customer/speaker_1.wav"),
        ("Aggressive Scam Call", "record_split/agent/speaker_0.wav", "record_split/customer/speaker_1.wav"),
        ("Customer Refusal", "record_split/agent/speaker_0.wav", "record_split/customer/speaker_1.wav"),
        ("Silence / Missing Audio", "missing_file.wav", "record_split/customer/speaker_1.wav") # Should handle gracefully
    ]
    
    for name, agent_path, customer_path in scenarios:
        print(f"\n>>> TESTING SCENARIO: {name}")
        try:
            # We call the main function directly or via subprocess
            # Here we just invoke the logic if possible or run command
            # predict_main(agent_path, customer_path) # This runs full pipeline
            
            # Use subprocess to isolate execution and test CLI interface
            cmd = f"venv_gpu\\Scripts\\python.exe spam_model\\inference\\predict.py --agent \"{agent_path}\" --customer \"{customer_path}\""
            print(f"Executing: {cmd}")
            os.system(cmd)
            
        except Exception as e:
            print(f"Scenario Failed: {e}")
            
    print("\n========================================")
    print("SCENARIO TESTING COMPLETE")
    print("========================================")

if __name__ == "__main__":
    test_scenarios()
