import subprocess
import sys
import os

def main():
    os.chdir(os.path.join(os.path.dirname(__file__), '003_api'))

    backend = subprocess.Popen(
        ['uvicorn', 'backend:app', '--reload'],
    )

    frontend = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', 'frontend.py'],
    )

    print("Backend  running on http://localhost:8000")
    print("Frontend running on http://localhost:8501")
    print("\nPress Ctrl+C to stop both servers...\n")

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.terminate()
        frontend.terminate()

if __name__ == "__main__":
    main()