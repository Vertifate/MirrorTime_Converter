#!/usr/bin/env python3
"""
MirrorTime Converter Launch Script
Automatically starts backend and frontend servers
"""
#WDD [2026-01-19] [Minimalist launch script - English only, no emojis]

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("\nMirrorTime Converter v1.0.0")
    print("4DGS Data Preprocessing Pipeline")
    print("-" * 50)

def check_dependencies():
    """Check if dependencies are installed"""
    # Check Python dependencies
    backend_requirements = Path("modules/visualization-server/requirements.txt")
    if backend_requirements.exists():
        try:
            import fastapi
            import uvicorn
        except ImportError:
            print("Installing backend dependencies...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", 
                str(backend_requirements)
            ], check=True, capture_output=True)
    
    # Check Node.js dependencies
    frontend_path = Path("visualization-ui")
    if frontend_path.exists():
        node_modules = frontend_path / "node_modules"
        if not node_modules.exists():
            print("Installing frontend dependencies...")
            subprocess.run(
                ["npm", "install"], 
                cwd=str(frontend_path),
                check=True,
                capture_output=True
            )

def start_backend():
    """Start backend server"""
    backend_script = Path("modules/visualization-server/src/main.py")
    
    if not backend_script.exists():
        print(f"Error: Backend script not found at {backend_script}")
        sys.exit(1)
    
    # Start FastAPI server
    backend_process = subprocess.Popen(
        [sys.executable, str(backend_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    return backend_process

def start_frontend():
    """Start frontend development server"""
    frontend_path = Path("visualization-ui")
    
    if not frontend_path.exists():
        print(f"Error: Frontend directory not found at {frontend_path}")
        sys.exit(1)
    
    # Start Vite dev server
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    return frontend_process

def main():
    """Main function"""
    print_banner()
    
    # Process list
    processes = []
    
    def signal_handler(sig, frame):
        """Handle exit signal"""
        print("\n\nShutting down...")
        for process in processes:
            process.terminate()
        print("Stopped.")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Check dependencies
        check_dependencies()
        
        # Start backend
        backend_process = start_backend()
        processes.append(backend_process)
        time.sleep(2)
        
        # Start frontend
        frontend_process = start_frontend()
        processes.append(frontend_process)
        time.sleep(2)
        
        print("\n" + "=" * 50)
        print("MirrorTime Converter is running")
        print("Open: http://localhost:5173")
        print("Press Ctrl+C to exit")
        print("=" * 50 + "\n")
        
        # Monitor processes
        while True:
            if backend_process.poll() is not None:
                print("Error: Backend process exited")
                break
            if frontend_process.poll() is not None:
                print("Error: Frontend process exited")
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print(f"Error: {e}")
        for process in processes:
            process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()

