import os
import sys
import subprocess
import re
import argparse
import shutil

def log(msg):
    print(msg)
    with open("pipeline.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run_pipeline(input_audio):
    # Clear log
    with open("pipeline.log", "w", encoding="utf-8") as f:
        f.write("Pipeline Started\n")

    log("="*60)
    log("UNIFIED CALL DETECTION PIPELINE")
    log("="*60)
    log(f"Input File: {input_audio}")
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    split_script = os.path.join(base_dir, "split_model", "run_pipeline.py")
    spam_script = os.path.join(base_dir, "spam_model", "inference", "predict.py")
    python_exe = sys.executable

    # 1. STEP 0: Cleanup Old Utils
    log("\n>>> [STEP 0] Cleaning up previous run data...")
    record_split_base = os.path.join(base_dir, "record_split")
    agent_dir = os.path.join(record_split_base, "agent")
    cust_dir = os.path.join(record_split_base, "customer")
    
    try:
        if os.path.exists(agent_dir):
            shutil.rmtree(agent_dir)
            log(f"Cleared: {agent_dir}")
        if os.path.exists(cust_dir):
            shutil.rmtree(cust_dir)
            log(f"Cleared: {cust_dir}")
        
        # Re-create to avoid errors if split_model expects them (though it usually creates them)
        os.makedirs(agent_dir, exist_ok=True)
        os.makedirs(cust_dir, exist_ok=True)
        
    except Exception as e:
        log(f"Warning during cleanup: {e}")

    # Prepare environment with UTF-8 encoding
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # --- Step 1: Split & Local Spoof Detection ---
    log("\n>>> [STEP 1] Running Split & Local Spoof Detection...")
    
    agent_path = None
    customer_path = None
    spoof_score = 0.5 # Default
    
    try:
        cmd = [python_exe, split_script, "--file", input_audio]
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8', 
            errors='replace',
            env=env
        )
        
        # Stream output to show progress and capture data
        full_output = ""
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip()) # Mirror output to console
                full_output += line
                
                # Live Parsing
                if "Saved Agent audio to:" in line:
                    agent_path = line.split("Saved Agent audio to:")[-1].strip()
                    log(f"DEBUG: Found Agent Path: {agent_path}")
                
                if "Raw Spoof Score:" in line:
                     log(f"DEBUG: Found Spoof Score Line: {line.strip()}")

        log(f"DEBUG: Final Agent Path: {agent_path}")
        
        # Post-process check - Fallback if parsing failed
        if not agent_path or not os.path.exists(agent_path):
            log("DEBUG: Parsing failed or file missing. Attempting fallback to record_split/agent...")
            agent_dir = os.path.join(base_dir, "record_split", "agent")
            if os.path.exists(agent_dir):
                files = os.listdir(agent_dir)
                wavs = [os.path.join(agent_dir, f) for f in files if f.endswith('.wav')]
                if wavs:
                    # Pick newest
                    agent_path = max(wavs, key=os.path.getmtime)
                    log(f"DEBUG: Fallback Agent Path: {agent_path}")

        if not customer_path: # split_model doesn't print this explicitly usually
            log("DEBUG: Attempting fallback to record_split/customer...")
            cust_dir = os.path.join(base_dir, "record_split", "customer")
            if os.path.exists(cust_dir):
                files = os.listdir(cust_dir)
                wavs = [os.path.join(cust_dir, f) for f in files if f.endswith('.wav')]
                if wavs:
                    # Pick newest
                    customer_path = max(wavs, key=os.path.getmtime)
                    log(f"DEBUG: Fallback Customer Path: {customer_path}")
            
        # Parse Spoof Score from full output (Regex on full text is safer)
        spoof_matches = re.findall(r"Spoof Score:\s+([0-9.]+)", full_output)
        if spoof_matches:
            # Take the last one (most likely the final verification one)
            spoof_score = float(spoof_matches[-1]) 
            
        log(f"\n[PIPELINE] Detected Agent Path: {agent_path}")
        log(f"[PIPELINE] Detected Customer Path: {customer_path}")
        log(f"[PIPELINE] Detected Spoof Score: {spoof_score}")

    except Exception as e:
        log(f"Error in Step 1: {e}")
        return

    if not agent_path or not os.path.exists(agent_path):
        log("CRITICAL: Failed to locate Agent Audio. Aborting.")
        return
    if not customer_path:
        log("WARNING: Customer path not found. Using Agent path as fallback to prevent crash.")
        customer_path = agent_path
    elif not os.path.exists(customer_path):
         log("WARNING: Customer path invalid. Using Agent path as fallback.")
         customer_path = agent_path

    # 3. STEP 3: Spam Detection (Reporting)
    log("\n>>> [STEP 3] Running Spam Detection & Reporting...")
    try:
        cmd = [
            python_exe, spam_script,
            "--agent", agent_path,
            "--customer", customer_path,
            "--spoof_score", str(spoof_score),
            "--spoof_verdict", "SPOOF" if spoof_score > 0.5 else "BONAFIDE"
        ]
        
        log(f"Executing: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        # Stream Step 3 Output
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip())
                log(line.strip())

    except Exception as e:
        log(f"Error in Spam Detection: {e}")
        return

    # Extract JSON path from predict.py log (which prints "Report saved to: ...")
    # or just construct it since we know the pattern.
    # Pattern: reports/call_<timestamp>_report.json
    # Better to find it in stdout to be sure.
    
    json_path = None
    if 'result' in locals() and result.stdout:
        # predict.py prints "Report saved to: .../reports"
        # It doesn't print the full filename explicitly? 
        # Actually predict.py prints: "Report saved to: D:\...\reports"
        # And the filename is call_<id>_report.json
        # We need the full filename.
        # Let's verify predict.py output again.
        # It prints: "Report saved to: ..." (directory)
        # It DOES print the text report.
        # We need to find the latest file in reports dir? Or update predict.py to print filename.
        pass

    # Find the latest JSON in reports dir as fallback/primary method
    reports_dir = os.path.join(base_dir, "spam_model", "reports")
    json_path = None
    
    if os.path.exists(reports_dir):
        # Filter for files matching pattern: call_.*_report.json
        files = [os.path.join(reports_dir, f) for f in os.listdir(reports_dir) if f.endswith('_report.json')]
        if files:
            json_path = max(files, key=os.path.getmtime)
            
    if json_path:
        log(f"OUTPUT_JSON: {json_path}")
        print(f"OUTPUT_JSON: {json_path}") # Explicitly for UI to parse
    else:
        log("OUTPUT_JSON: NOT_FOUND")
        print("OUTPUT_JSON: NOT_FOUND")

    log("\n" + "="*60)
    log("PIPELINE COMPLETE")
    log("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input audio file path")
    args = parser.parse_args()
    
    if os.path.exists(args.file):
        run_pipeline(args.file)
    else:
        print("Input file not found.")
