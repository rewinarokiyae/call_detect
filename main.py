import os
import sys
import subprocess

def main():
    """
    Entry point for the Call Detect AI project.
    This script launches the Streamlit web application.
    """
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the streamlit app
    app_path = os.path.join(base_dir, "streamlit_app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: Could not find '{app_path}'")
        sys.exit(1)
        
    print("Starting Call Detect AI Application (Web UI)...")
    
    # Run the streamlit app using the current Python interpreter
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    except Exception as e:
        print(f"\nAn error occurred while starting the application: {e}")

if __name__ == "__main__":
    main()
