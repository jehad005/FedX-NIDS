#!/usr/bin/env python3
"""
SecureFed-NIDS Client Web Interface
====================================
Interactive dashboard for edge clients to:
- Download latest model from HQ
- Upload DP-weight updates
- Monitor connection status
- View model information

Usage:
    python client_interface.py --hq-url http://localhost:8000
"""

import argparse
import json
import time
import threading
import webbrowser
from pathlib import Path
from datetime import datetime

try:
    import requests
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
except ImportError as e:
    print(f"[!] Missing dependency: {e}")
    print("[!] Install: pip install requests fastapi uvicorn jinja2")
    sys.exit(1)

import sys

ROOT = Path(__file__).resolve().parent

# Client config
CLIENT_CONFIG = {
    "hq_url": "http://localhost:8000",
    "client_id": "edge_client_001",
    "client_name": "Default Client",
    "model_name": "SecureFed-Global-Ensemble",
    "download_dir": "edge_deployment",
    "config_file": "edge_deployment/client_config.json"
}


class ClientInterface:
    def __init__(self, hq_url: str, client_id: str, client_name: str):
        self.hq_url = hq_url.rstrip('/')
        self.client_id = client_id
        self.client_name = client_name

        self.app = FastAPI(title=f"SecureFed-NIDS Client: {client_id}", version="1.0.0")

        # Setup templates
        self.templates = Jinja2Templates(directory=str(ROOT / "client_hq" / "templates"))

        # Active WebSocket connections
        self.active_connections: list = []

        # Client state
        self.state = {
            "connected": False,
            "latest_version": None,
            "last_download": None,
            "last_upload": None,
            "status": "disconnected",
            "connection_errors": 0,
            "total_downloads": 0,
            "total_uploads": 0
        }

        # Setup routes
        self.setup_routes()

    def setup_routes(self):
        """Define all client routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def client_home(request: Request):
            """Main client dashboard page."""
            return self.templates.TemplateResponse(
                "client.html",
                {
                    "request": request,
                    "client_id": self.client_id,
                    "client_name": self.client_name,
                    "hq_url": self.hq_url,
                    "state": self.state,
                    "timestamp": datetime.now().isoformat()
                }
            )

        @self.app.get("/api/status")
        async def get_status():
            """Get current client status."""
            try:
                # Check HQ connectivity
                response = requests.get(f"{self.hq_url}/", timeout=2)
                if response.status_code == 200:
                    self.state["connected"] = True
                    self.state["status"] = "connected"
                    self.state["latest_version"] = response.json().get("latest_version")
                    self.state["connection_errors"] = 0
                else:
                    self.state["connected"] = False
                    self.state["status"] = "error"
                    self.state["connection_errors"] += 1
            except requests.exceptions.RequestException as e:
                self.state["connected"] = False
                self.state["status"] = "error"
                self.state["connection_errors"] += 1
                self.state["last_error"] = str(e)

            return JSONResponse({
                "status": "success",
                "client_id": self.client_id,
                "state": self.state,
                "timestamp": datetime.now().isoformat()
            })

        @self.app.get("/api/model/download")
        async def download_model():
            """Download latest model from HQ."""
            try:
                response = requests.get(
                    f"{self.hq_url}/api/v1/model/download/latest",
                    headers={"X-Client-ID": self.client_id},
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    self.state["latest_version"] = data["version"]
                    self.state["open_round"] = data.get("open_round", 1)
                    self.state["total_downloads"] += 1
                    self.state["last_download"] = datetime.now().isoformat()
                    self.state["status"] = "downloaded"

                    return JSONResponse({
                        "status": "success",
                        "message": f"Model v{data['version']} downloaded successfully",
                        "version": data["version"],
                        "model_name": data["model_name"],
                        "fl_method": data["fl_method"],
                        "weights_b64": data["weights_b64"][:100] + "...",  # Truncate for response
                        "size": len(data["weights_b64"]),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    return JSONResponse({
                        "status": "error",
                        "message": f"Failed to download: {response.status_code}"
                    }, status_code=response.status_code)

            except requests.exceptions.RequestException as e:
                return JSONResponse({
                    "status": "error",
                    "message": f"Connection failed: {str(e)}"
                }, status_code=500)

        @self.app.post("/api/model/upload")
        async def upload_update(update: dict):
            """Forward a clipped/noised .pt DP delta update to HQ as multipart/form-data."""
            import base64
            try:
                pt_bytes = None
                file_path = update.get("file_path")
                if file_path and Path(file_path).exists():
                    pt_bytes = Path(file_path).read_bytes()
                elif update.get("weights_b64"):
                    pt_bytes = base64.b64decode(update["weights_b64"])

                if not pt_bytes:
                    return JSONResponse({
                        "status": "error",
                        "message": "Provide a valid clipped/noised .pt delta file ('file_path' or 'weights_b64'). Placeholder JSON strings cannot be aggregated by HQ."
                    }, status_code=400)

                form_data = {
                    "client_id": str(update.get("client_id") or self.client_id),
                    "samples": int(update.get("samples", 0)),
                    "base_version": str(update.get("base_version") or self.state.get("latest_version") or "latest"),
                    "round": int(update.get("round") or self.state.get("open_round") or 1),
                }
                files = {"file": ("dp_delta.pt", pt_bytes, "application/octet-stream")}
                response = requests.post(
                    f"{self.hq_url}/api/v1/model/upload",
                    data=form_data,
                    files=files,
                    timeout=15,
                )

                if response.status_code == 200:
                    resp_json = response.json()
                    self.state["total_uploads"] += 1
                    self.state["last_upload"] = datetime.now().isoformat()
                    self.state["status"] = "uploaded"

                    return JSONResponse({
                        "status": "success",
                        "message": "Update uploaded successfully",
                        "update_id": resp_json.get("update_id", "N/A"),
                        "dp_epsilon": update.get("dp_epsilon", 0),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    return JSONResponse({
                        "status": "error",
                        "message": f"Upload failed ({response.status_code}): {response.text}"
                    }, status_code=response.status_code)

            except requests.exceptions.RequestException as e:
                return JSONResponse({
                    "status": "error",
                    "message": f"Connection failed: {str(e)}"
                }, status_code=500)

        @self.app.get("/api/model/config")
        async def get_config():
            """Get client configuration."""
            config_path = Path(CLIENT_CONFIG["config_file"])
            if config_path.exists():
                with open(config_path) as f:
                    return JSONResponse({
                        "status": "success",
                        "config": json.load(f),
                        "timestamp": datetime.now().isoformat()
                    })
            else:
                return JSONResponse({
                    "status": "success",
                    "config": CLIENT_CONFIG,
                    "timestamp": datetime.now().isoformat()
                })

        @self.app.get("/api/hq/versions")
        async def get_versions():
            """Get model versions from HQ."""
            try:
                response = requests.get(f"{self.hq_url}/api/v1/model/versions", timeout=5)
                if response.status_code == 200:
                    return JSONResponse({
                        "status": "success",
                        "versions": response.json().get("versions", []),
                        "latest": response.json().get("latest"),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    return JSONResponse({
                        "status": "error",
                        "message": f"Failed to fetch versions: {response.status_code}"
                    }, status_code=response.status_code)
            except requests.exceptions.RequestException as e:
                return JSONResponse({
                    "status": "error",
                    "message": f"Connection failed: {str(e)}"
                }, status_code=500)

        @self.app.websocket("/ws/client")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates."""
            await websocket.accept()
            self.active_connections.append(websocket)

            try:
                while True:
                    await websocket.receive_text()

                    # Check connection status
                    status = requests.get(f"{self.hq_url}/", timeout=2).json()
                    await websocket.send_json({
                        "type": "status_update",
                        "connected": status.get("latest_version") is not None,
                        "latest_version": status.get("latest_version"),
                        "timestamp": datetime.now().isoformat()
                    })

            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except requests.exceptions.RequestException:
                await websocket.send_json({
                    "type": "error",
                    "message": "Lost connection to HQ",
                    "timestamp": datetime.now().isoformat()
                })

    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the client interface server."""
        print("\n" + "="*70)
        print(f"SECUREFED-NIDS CLIENT INTERFACE")
        print("="*70)
        print(f"Client ID: {self.client_id}")
        print(f"Client Name: {self.client_name}")
        print(f"HQ URL: {self.hq_url}")
        print(f"Dashboard: http://{host}:{port}")
        print("="*70)

        uvicorn.run(self.app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="SecureFed-NIDS Client Interface")
    parser.add_argument("--hq-url", default="http://localhost:8000",
                        help="HQ Server URL")
    parser.add_argument("--client-id", default="edge_client_001",
                        help="Client ID")
    parser.add_argument("--client-name", default="Edge Client",
                        help="Client Name")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to run client interface on")
    parser.add_argument("--open-browser", action="store_true",
                        help="Open browser automatically")

    args = parser.parse_args()

    client = ClientInterface(
        hq_url=args.hq_url,
        client_id=args.client_id,
        client_name=args.client_name
    )

    # Open browser if requested
    if args.open_browser:
        time.sleep(1)
        webbrowser.open(f"http://localhost:{args.port}")

    client.run(port=args.port)


if __name__ == "__main__":
    main()