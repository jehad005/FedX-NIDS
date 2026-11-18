#!/usr/bin/env python3
"""
SecureFed-NIDS HQ Server Launcher
==================================
Starts both the API server and dashboard on localhost.

Usage:
    python start_hq.py

This will start:
    - API Server on http://localhost:8000
    - Dashboard on http://localhost:8001
"""

import sys
import subprocess
import time
import socket
import os
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def check_dependencies():
    """Check if required packages are installed (skip for venv)."""
    # Skip check - packages are in venv, not system Python
    return True


def ensure_directories():
    """Ensure required directories exist."""
    dirs = [
        ROOT / "server_hq" / "templates",
        ROOT / "server_hq" / "static",
        ROOT / "hq_store" / "models",
        ROOT / "models",
        ROOT / "edge_deployment"
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[✓] Directories ensured")


def is_port_in_use(port: int) -> bool:
    """Check if TCP port is currently bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def release_port(port: int):
    """Gracefully release port if occupied by a stale server process."""
    if not is_port_in_use(port):
        return
    print(f"[!] Port {port} is currently bound. Freeing port for fresh startup...")
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.8)
    except Exception:
        pass

    if is_port_in_use(port):
        try:
            out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip()
            for pid_str in out.split():
                if pid_str.isdigit():
                    os.kill(int(pid_str), signal.SIGKILL)
            time.sleep(0.8)
        except Exception:
            pass

    if is_port_in_use(port):
        print(f"[!] Warning: Port {port} could not be cleared automatically.")
    else:
        print(f"[✓] Port {port} released.")


def run_api_server():
    """Run the FastAPI server on port 8000."""
    print("\n" + "="*70)
    print("Starting SecureFed-NIDS API Server on http://localhost:8000")
    print("="*70)

    api_process = subprocess.Popen(
        [sys.executable, "-m", "server_hq.server_api"],
        cwd=str(ROOT)
    )

    return api_process


def run_dashboard():
    """Run the dashboard on port 8001."""
    print("\n" + "="*70)
    print("Starting SecureFed-NIDS Dashboard on http://localhost:8001")
    print("="*70)

    dashboard_process = subprocess.Popen(
        [sys.executable, "-m", "server_hq.dashboard"],
        cwd=str(ROOT)
    )

    return dashboard_process


def main():
    print("\n" + "="*70)
    print("SECUREFED-NIDS HQ SERVER LAUNCHER")
    print("="*70)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Ensure directories
    ensure_directories()

    # Clear lingering processes from prior runs to prevent [Errno 98] Address already in use
    release_port(8000)
    release_port(8001)

    # Start API server
    api_process = run_api_server()
    time.sleep(2)  # Give it time to start

    # Start dashboard
    dashboard_process = run_dashboard()

    print("\n" + "="*70)
    print("SERVERS STARTED SUCCESSFULLY!")
    print("="*70)
    print("\nAccess the following URLs:")
    print("  • API Server:   http://localhost:8000")
    print("  • Dashboard:    http://localhost:8001")
    print("  • API Docs:     http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop both servers")
    print("="*70 + "\n")

    try:
        # Wait for processes
        api_process.wait()
        dashboard_process.wait()
    except KeyboardInterrupt:
        print("\n\n[!] Shutting down servers...")
        api_process.terminate()
        dashboard_process.terminate()
        api_process.wait()
        dashboard_process.wait()
        print("[✓] Servers stopped successfully")


if __name__ == "__main__":
    main()