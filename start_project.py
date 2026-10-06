"""
start_project.py — Master One-Command Application Launcher
"""

import sys
import subprocess
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = ROOT_DIR / ".venv" / "Scripts" / "python.exe"

def main():
    print("==================================================")
    print("   TRACEAI — Image Provenance Laboratory Launcher ")
    print("==================================================")

    if not PYTHON_EXE.exists():
        print(f"Error: Python executable not found at {PYTHON_EXE}")
        sys.exit(1)

    print("\n[1/3] Starting FastAPI Backend Server (http://localhost:8000) ...")
    backend_cmd = [str(PYTHON_EXE), "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(ROOT_DIR))

    time.sleep(2)

    print("\n[2/3] Starting React Vite Frontend Server (http://localhost:5173) ...")
    frontend_dir = ROOT_DIR / "frontend"
    frontend_cmd = ["cmd", "/c", "npm", "run", "dev"]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(frontend_dir))

    time.sleep(3)

    print("\n[3/3] Opening Web Application in Browser ...")
    try:
        webbrowser.open("http://localhost:5173")
    except Exception:
        pass

    print("\n" + "=" * 50)
    print("  TRACEAI PLATFORM IS NOW ONLINE!")
    print("  Frontend UI:  http://localhost:5173")
    print("  Backend API:  http://localhost:8000")
    print("  API Docs:     http://localhost:8000/docs")
    print("  Press Ctrl+C in terminal to stop servers.")
    print("=" * 50 + "\n")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping TRACEAI servers...")
        backend_proc.terminate()
        frontend_proc.terminate()

if __name__ == "__main__":
    main()
