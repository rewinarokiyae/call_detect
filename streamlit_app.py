import streamlit as st
import subprocess
import os
import sys
import time
import re
import json
import shutil

# Page config
st.set_page_config(
    page_title="Call Detect AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Super Style"
st.markdown('''
<style>
    /* Main Background and Text */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Card Style */
    .css-card {
        background-color: #262730;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border: 1px solid #363B47;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Metric Styles */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #A0A0A0;
    }
    
    /* Status Colors */
    .status-safe { color: #00CC96; }
    .status-danger { color: #EF553B; }
    .status-warning { color: #FFA15A; }
    
    /* Step Indicators */
    .step-header {
        background-color: #1F2937;
        padding: 10px 20px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 5px solid #3B82F6;
    }
    
    /* Custom divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, #EF553B, #00CC96);
        margin: 20px 0;
        border-radius: 2px;
    }
</style>
''', unsafe_allow_html=True)

# Application Title
st.title("🛡️ Call Detect AI")
st.markdown("### Real-time Scam & Deepfake Detection Pipeline")
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar & Setup
# ---------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
inputs_dir = os.path.join(base_dir, "data", "live_recordings")

# --- Performance Optimization: Load Models Early ---
from utils.model_loader import load_pipeline

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.header("Pipeline Control")

    # Load Model (Cached)
    with st.spinner("Initializing AI Models... (First Run Only)"):
        pipeline = load_pipeline()
    st.success("AI Models Loaded 🚀")
    
    # Selection Logic
    uploaded_file = st.file_uploader("Upload Audio (MP3/WAV)", type=["mp3", "wav"])
    
    selection_options = [""]
    if os.path.exists(inputs_dir):
        files = [f for f in os.listdir(inputs_dir) if f.lower().endswith(('.mp3', '.wav'))]
        selection_options += files
        
    selected_file = st.selectbox("Select Existing Sample", selection_options)
    
    # Handle File Selection Changes
    current_selection = None
    if uploaded_file:
        current_selection = f"upload_{uploaded_file.name}"
    elif selected_file:
        current_selection = selected_file
        
    # Initialize Session State
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'input_file' not in st.session_state:
        st.session_state.input_file = None
    if 'agent_audio' not in st.session_state:
        st.session_state.agent_audio = None
    if 'customer_audio' not in st.session_state:
        st.session_state.customer_audio = None
    if 'spoof_score' not in st.session_state:
        st.session_state.spoof_score = None
    if 'final_report' not in st.session_state:
        st.session_state.final_report = None
        
    def reset_pipeline():
        st.session_state.current_step = 0
        st.session_state.input_file = None
        st.session_state.agent_audio = None
        st.session_state.customer_audio = None
        st.session_state.spoof_score = None
        st.session_state.final_report = None

    if st.button("🔄 Reset Pipeline / New Call"):
        reset_pipeline()
        st.rerun()

    st.markdown("---")
    
    # Progress Indicator
    steps = ["Select Audio", "Split Model", "Local Model", "Spam Model"]
    progress = min(st.session_state.current_step, 3) / 3.0
    st.progress(progress)
    st.caption(f"Current Phase: {steps[min(st.session_state.current_step, 3)]}")

# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------

# STEP 0: Input Processing
if st.session_state.current_step == 0:
    st.info("👋 Select or Upload an audio file to begin the analysis pipeline.")
    
    file_path = None
    if uploaded_file:
        temp_dir = os.path.join(base_dir, "uploads")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    elif selected_file:
        file_path = os.path.join(inputs_dir, selected_file)
        
    if file_path and os.path.exists(file_path):
        st.session_state.input_file = file_path
        st.success(f"Selected: {os.path.basename(file_path)}")
        st.audio(file_path)
        
        if st.button("Start Pipeline ➡️"):
            st.session_state.current_step = 1
            st.rerun()

# STEP 1: Split Model (Diarization)
elif st.session_state.current_step >= 1:
    st.markdown("<div class='step-header'><h3>1️⃣ Step 1: Split Model (Diarization)</h3></div>", unsafe_allow_html=True)
    
    # Execution Logic for Step 1
    if st.session_state.agent_audio is None:
        
        col_run, col_check = st.columns([1,1])
        with col_run:
            start_split = st.button("▶️ Run Split Model (Fast Mode ⚡)")
        with col_check:
            force_check = st.button("🔍 Force Check for Audio Files")

        if start_split:
            with st.spinner("Processing Audio (Cached Models)..."):
                try:
                    # USE CACHED PIPELINE
                    result = pipeline.process(st.session_state.input_file)
                    
                    if result.get("status") == "success":
                        st.session_state.agent_audio = result.get("agent_audio")
                        st.session_state.customer_audio = result.get("customer_audio")
                        st.session_state.spoof_score = result.get("spoof_score", 0.5)
                        st.rerun()
                    else:
                        st.error(f"Pipeline Error: {result.get('error')}")

                except Exception as e:
                    st.error(f"Error in Split Model: {e}")
                    # Fallback to allow force check
        
        if force_check:
            # Manual check without running model
            found_agent = None
            found_costumer = None
            
            fallback_agent = os.path.join(base_dir, "record_split", "agent")
            if os.path.exists(fallback_agent):
                 waves = [os.path.join(fallback_agent, f) for f in os.listdir(fallback_agent) if f.endswith(".wav")]
                 if waves: found_agent = max(waves, key=os.path.getmtime)
            
            fallback_cust = os.path.join(base_dir, "record_split", "customer")
            if os.path.exists(fallback_cust):
                 waves = [os.path.join(fallback_cust, f) for f in os.listdir(fallback_cust) if f.endswith(".wav")]
                 if waves: found_costumer = max(waves, key=os.path.getmtime)
            
            if found_agent:
                 st.session_state.agent_audio = found_agent
                 st.session_state.customer_audio = found_costumer
                 st.session_state.spoof_score = 0.5 # Default if forced
                 st.success("Files found!")
                 st.rerun()
            else:
                 st.warning("No audio files found in record_split/agent folder yet.")


    # Display Output for Step 1
    if st.session_state.agent_audio:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🗣️ Agent Audio")
            if st.session_state.agent_audio and os.path.exists(st.session_state.agent_audio):
                st.audio(st.session_state.agent_audio)
            else:
                st.warning("Agent audio not found")
        with c2:
            st.markdown("#### 👤 Customer Audio")
            if st.session_state.customer_audio and os.path.exists(st.session_state.customer_audio):
                st.audio(st.session_state.customer_audio)
            else:
                st.warning("Customer audio not found")
        
        # SWAP BUTTON
        if st.button("🔁 Swap Speakers (If wrong)"):
             # Swap paths
             temp = st.session_state.agent_audio
             st.session_state.agent_audio = st.session_state.customer_audio
             st.session_state.customer_audio = temp
             st.rerun()

        st.success("✅ Splitting Complete")
        if st.session_state.current_step == 1:
            if st.button("Proceed to Local Model ➡️"):
                st.session_state.current_step = 2
                st.rerun()

    st.markdown("---")

# STEP 2: Local Model (Spoof Detection)
if st.session_state.current_step >= 2:
    st.markdown("<div class='step-header'><h3>2️⃣ Step 2: Local Model (Spoof Detection)</h3></div>", unsafe_allow_html=True)
    
    # We already captured the score in Step 1 (since the backend script runs them together).
    # UX-wise, we "Run" the analysis to reveal it.
    
    if st.session_state.current_step == 2:
        if st.button("▶️ Run Local Spoof Analysis"):
            with st.spinner("Analyzing Audio Features for Spoofing..."):
                # Real inference on agent audio if available
                if st.session_state.agent_audio and os.path.exists(st.session_state.agent_audio):
                    try:
                        # Call Inference Script via Subprocess (Lightweight compared to diarization)
                        # OR if we want to cache this too, we'd add it to model_loader
                        python_exe = os.path.join(base_dir, "venv_gpu", "Scripts", "python.exe")
                        inf_script = os.path.join(base_dir, "Local_Model", "src", "inference.py")
                        cmd = [python_exe, inf_script, st.session_state.agent_audio]
                        
                        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                        output = proc.stdout
                        match = re.search(r"Spoof Score:\s+([0-9.]+)", output)
                        if match:
                            st.session_state.spoof_score = float(match.group(1))
                    except Exception as e:
                        print(f"Local inference failed: {e}")
                
                time.sleep(1) 
                st.session_state.current_step = 3 
                st.rerun()
    
    # Display Result
    if st.session_state.current_step >= 3:
        score = st.session_state.spoof_score
        is_spoof = score > 0.5
        
        col_score, col_verdict = st.columns(2)
        with col_score:
            st.metric("Raw Spoof Score", f"{score:.4f}")
        with col_verdict:
            if is_spoof:
                st.error("🚨 PROBABLE SPOOF / AI VOICE")
            else:
                st.success("✅ GENUINE HUMAN VOICE")
        
        if st.session_state.current_step == 3:
            if st.button("Proceed to Spam Model ➡️"):
                st.session_state.current_step = 4
                st.rerun()

    st.markdown("---")

# STEP 3: Spam Model (Final Report)
if st.session_state.current_step >= 4:
    st.markdown("<div class='step-header'><h3>3️⃣ Step 3: Spam Model (Analysis & Reporting)</h3></div>", unsafe_allow_html=True)

    if st.session_state.final_report is None:
        if st.button("▶️ Run Spam Detection"):
            with st.spinner("Analyzing Content for Scam Patterns..."):
                # Run the SPAM model script specifically
                # "spam_model/inference/predict.py --agent ... --customer ... --spoof_score ..."
                
                python_exe = os.path.join(base_dir, "venv_gpu", "Scripts", "python.exe")
                spam_script = os.path.join(base_dir, "spam_model", "inference", "predict.py")
                
                agent = st.session_state.agent_audio
                cust = st.session_state.customer_audio or agent # Fallback
                score = st.session_state.spoof_score
                verdict = "SPOOF" if score > 0.5 else "BONAFIDE"
                
                cmd = [
                    python_exe, spam_script,
                    "--agent", agent,
                    "--customer", cust,
                    "--spoof_score", str(score),
                    "--spoof_verdict", verdict
                ]
                
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                
                logs = []
                json_path = None
                
                try:
                    process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                        text=True, encoding='utf-8', errors='replace', env=env
                    )
                    
                    st.code("Running Inference...", language="bash")
                    
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            logs.append(line.strip())
                            
                    # Find Resulting JSON
                    reports_dir = os.path.join(base_dir, "spam_model", "reports")
                    if os.path.exists(reports_dir):
                        files = [os.path.join(reports_dir, f) for f in os.listdir(reports_dir) if f.endswith('_report.json')]
                        if files:
                            # Use file modification time to find the one we just created
                            json_path = max(files, key=os.path.getmtime)
                            st.session_state.final_report = json_path
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"Error in Spam Model: {e}")

    # Display Final Report
    if st.session_state.final_report:
        if os.path.exists(st.session_state.final_report):
            with open(st.session_state.final_report, "r", encoding='utf-8') as f:
                data = json.load(f)
            
            # --- SUPER STYLE REPORT UI ---
            classification = data.get("caller_classification", "Unknown")
            agent_type = data.get("agent_type", "Unknown")
            risk_score = data.get("risk_score", 0)
            
            is_high_risk = classification in ["Likely Scam", "High Risk"] or risk_score > 70
            color = "status-danger" if is_high_risk else "status-safe"
            icon = "🚨" if is_high_risk else "✅"
            
            st.markdown(f'''
            <div class='css-card'>
                <h2 style='text-align: center;'>Final Verdict</h2>
                <div style='display: flex; justify-content: space-around; margin-top: 20px;'>
                    <div style='text-align: center;'>
                        <div class='metric-label'>Call Type</div>
                        <div class='metric-value {color}'>{icon} {classification}</div>
                    </div>
                    <div style='text-align: center;'>
                        <div class='metric-label'>Agent</div>
                        <div class='metric-value'>🤖 {agent_type}</div>
                    </div>
                    <div style='text-align: center;'>
                        <div class='metric-label'>Risk Score</div>
                        <div class='metric-value'>{risk_score}/100</div>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            with st.expander("📄 View Full Detailed Report", expanded=False):
                st.json(data)
                
            st.success("🎉 Pipeline Completed Successfully!")
