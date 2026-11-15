"""
SecureFed-NIDS HQ Dashboard
============================
Real-time monitoring dashboard for HQ server.
Includes:
- Model version management
- Client connection monitoring
- Performance analytics
- Update queue visualization
"""

from __future__ import annotations

import json
import sqlite3
import asyncio
import os
import sys
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
try:
    import numpy as np
except ImportError:  # Dashboard statistics do not require NumPy at runtime.
    np = None
from server_hq.visualization import (
    create_training_plots,
    create_version_detail_plots,
)

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from server_hq.registry import ModelRegistry
from server_hq.dashboard_page_handler import setup_dashboard_routes
from server_hq.auth import (
    SESSION_COOKIE_NAME,
    create_session_token,
    verify_session_token,
    verify_admin_credentials,
    _get_allowed_passwords,
)

ROOT = Path(__file__).resolve().parent.parent


def _clean_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(item) for item in obj]
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    # Convert to string as last resort
    return str(obj)


class _TrainingStoppedError(Exception):
    """Raised internally when a background training job is stopped by the user."""


class Dashboard:
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.app = FastAPI(title="SecureFed-NIDS HQ Dashboard", version="2.0.0")
        self.registry = registry if registry is not None else ModelRegistry()

        # Setup templates with default Jinja2
        self.templates = Jinja2Templates(directory=str(ROOT / "server_hq" / "templates"))

        # Active WebSocket connections & background training task registries
        self.active_connections: List[WebSocket] = []
        self._stop_events: Dict[str, threading.Event] = {}
        self._training_tasks: Dict[str, asyncio.Task] = {}

        # HQ training workspace
        self.training_uploads_dir = ROOT / "hq_store" / "training_uploads"
        self.training_logs_dir = ROOT / "hq_store" / "training_logs"
        self.training_uploads_dir.mkdir(parents=True, exist_ok=True)
        self.training_logs_dir.mkdir(parents=True, exist_ok=True)

        # Mount static files
        static_path = ROOT / "server_hq" / "static"
        static_path.mkdir(parents=True, exist_ok=True)
        self.app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

        # Plot storage directory (where training jobs save plots)
        plot_dir = self.training_logs_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        self.app.mount("/static/plots", StaticFiles(directory=str(plot_dir)), name="plots")

        # ── Plot‑serving endpoint ──
        # Serves individual plot files by version + filename so the Model Version
        # Details page can display them even after deployment (no absolute local
        # paths ever reach the browser).
        @self.app.get("/api/dashboard/plot/{version}/{filename}")
        async def serve_plot(version: str, filename: str):
            """
            Return a specific plot file for a published version.

            Plot files are stored in a flat directory with job_id prefix:
                hq_store/training_logs/plots/{job_id}_{filename}
            This endpoint searches for files matching the pattern to support
            both version-specific and flat directory layouts.
            """
            PLOTS_ROOT = self.training_logs_dir / "plots"

            # Try version-specific subdirectory first (for new versions)
            version_dir = PLOTS_ROOT / version
            full_path = os.path.normpath(os.path.join(version_dir, filename))

            # Security check for version-specific path
            base_real = os.path.realpath(str(PLOTS_ROOT))
            full_real = os.path.realpath(full_path)
            if full_real.startswith(base_real + os.sep) and os.path.isfile(full_path):
                return FileResponse(full_path)

            # Fall back to flat directory search (for existing versions)
            # Files are named like: {job_id}_{filename}
            if PLOTS_ROOT.exists():
                for existing_file in PLOTS_ROOT.iterdir():
                    if not existing_file.is_file():
                        continue
                    # existing layouts include flat files like:
                    #   {job_id}_{model}_{plot_type}.png
                    # so we match by exact name first, then suffix match.
                    if existing_file.name == filename or existing_file.name.endswith(filename):
                        return FileResponse(str(existing_file))

            raise HTTPException(status_code=404, detail="Plot file not found")

        # ── Mandatory Admin Access Control Middleware ──
        # All HQ control plane features are strictly gated behind admin authentication.
        @self.app.middleware("http")
        async def admin_auth_middleware(request: Request, call_next):
            path = request.url.path

            # Allow public assets and login/logout endpoints
            if (
                path in ("/login", "/logout", "/api/dashboard/auth/login", "/api/dashboard/auth/logout")
                or path.startswith("/static")
                or path.startswith("/api/v1")  # Client sync and model download endpoints
                or path.startswith("/docs")
                or path.startswith("/openapi.json")
                or path.startswith("/redoc")
            ):
                return await call_next(request)

            # Check cryptographically signed admin session cookie or X-Admin-Key header
            cookie_token = request.cookies.get(SESSION_COOKIE_NAME)
            header_key = request.headers.get("X-Admin-Key")

            is_authenticated = False
            if cookie_token and verify_session_token(cookie_token):
                is_authenticated = True
            elif header_key and (header_key.strip() in _get_allowed_passwords()):
                is_authenticated = True

            if not is_authenticated:
                accept = request.headers.get("accept", "")
                # If page navigation in browser, redirect immediately to login
                if "text/html" in accept or not path.startswith("/api/"):
                    return RedirectResponse(url=f"/login?next={path}", status_code=303)
                # If background API call
                return JSONResponse(
                    status_code=401,
                    content={
                        "status": "error",
                        "message": "Admin authorization required. Access Denied.",
                        "redirect": f"/login?next={path}"
                    }
                )

            return await call_next(request)

        self.setup_routes()
        # Register the dedicated dashboard pages (/models, /clients, /updates, /analytics).
        setup_dashboard_routes(self.app, self.registry)
        self.setup_events()

    @staticmethod
    def _verify_admin_header(x_admin_key: Optional[str]) -> str:
        """Validate optional X-Admin-Key consistently with server_api (strict if REQUIRE_DASHBOARD_ADMIN_KEY=1)."""
        expected = os.getenv("ADMIN_KEY", os.getenv("ADMIN_API_KEY", "securefed-admin-token-2026")).strip()
        if x_admin_key is not None and x_admin_key.strip() != expected:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid Admin Secret Key for HQ Dashboard API.")
        if x_admin_key is None and os.getenv("REQUIRE_DASHBOARD_ADMIN_KEY", "0") == "1":
            raise HTTPException(status_code=403, detail="Forbidden: Missing X-Admin-Key header.")
        return expected

    def setup_routes(self):
        """Define all dashboard routes."""

        @self.app.get("/login", response_class=HTMLResponse)
        async def login_page(request: Request, next: str = "/"):
            """Render the SOC Admin login page or redirect if already authenticated."""
            cookie_token = request.cookies.get(SESSION_COOKIE_NAME)
            if cookie_token and verify_session_token(cookie_token):
                dest = next if (next and next.startswith("/") and next != "/login") else "/"
                return RedirectResponse(url=dest, status_code=303)
            return self.templates.TemplateResponse(
                request=request,
                name="login.html",
                context={"request": request, "next": next}
            )

        @self.app.get("/logout")
        @self.app.post("/logout")
        async def logout_page():
            """Log out active admin session and redirect to /login."""
            resp = RedirectResponse(url="/login?logged_out=1", status_code=303)
            resp.delete_cookie(SESSION_COOKIE_NAME, path="/")
            return resp

        @self.app.post("/api/dashboard/auth/login")
        async def auth_login(request: Request):
            """Authenticate admin credentials and issue cryptographic session cookie."""
            try:
                body = await request.json()
            except Exception:
                body = {}
            username = (body.get("username") or "").strip()
            password = (body.get("password") or body.get("admin_key") or "").strip()
            next_url = (body.get("next") or "/").strip()
            if not next_url.startswith("/") or next_url == "/login":
                next_url = "/"

            if verify_admin_credentials(username, password):
                token = create_session_token(username)
                resp = JSONResponse({
                    "status": "success",
                    "message": "Authentication successful",
                    "token": token,
                    "redirect": next_url
                })
                resp.set_cookie(
                    key=SESSION_COOKIE_NAME,
                    value=token,
                    max_age=86400,
                    path="/",
                    httponly=True,
                    samesite="lax",
                )
                return resp

            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "message": "Invalid Admin credentials. Access Denied."
                }
            )

        @self.app.post("/api/dashboard/auth/logout")
        async def auth_logout():
            """Invalidate admin session."""
            resp = JSONResponse({"status": "success", "message": "Signed out successfully", "redirect": "/login?logged_out=1"})
            resp.delete_cookie(SESSION_COOKIE_NAME)
            return resp

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Render the summary dashboard; detailed views live on dedicated pages."""
            # Keep the home route intentionally lightweight.  Models, clients,
            # updates, and analytics each have their own page and API endpoint.
            template_context = {
                "request": request,
                "client_stats": _clean_for_json(self.get_client_stats()),
                "update_stats": _clean_for_json(self.get_update_stats()),
                "model_stats": _clean_for_json({
                    "total_versions": self.registry.count_versions(),
                    "latest_version": self.registry.latest_version(),
                }),
                "timestamp": datetime.now().isoformat(),
            }

            return self.templates.TemplateResponse(
                request=request,
                name="index.html",
                context=template_context,
            )

        @self.app.get("/api/dashboard/versions")
        async def get_versions(kind: Optional[str] = None):
            """API endpoint for model versions."""
            versions = self.registry.list_versions(kind=kind)
            return JSONResponse({
                "status": "success",
                "versions": versions,
                "latest": self.registry.latest_version(kind=kind),
                "last_pushed": self.registry.get_last_pushed_model(),
                "timestamp": datetime.now().isoformat()
            })

        @self.app.get("/api/dashboard/versions/{version}/package")
        async def get_version_package(version: str):
            """Full published record, on-disk meta.json, and the edge package contents."""
            record = self.registry.get_version(version)
            if record is None:
                raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")
            dest = self.registry.version_dir(record["version"])
            meta = {}
            meta_path = dest / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    meta = {"error": f"Could not read meta.json: {exc}"}
            # Rebase registered plot paths to the new plot serving endpoint.
            # A manifest keeps the Model Versions page self-contained even
            # after a training job is no longer visible in the job list.
            manifest = meta.get("plot_manifest", {}) if isinstance(meta, dict) else {}

            def _public_plot_url(path_value):
                if not path_value:
                    return None
                # Extract just the filename from the path
                filename = Path(str(path_value)).name
                # Return the new endpoint URL: /api/dashboard/plot/{version}/{filename}
                # We need to get the version from the record
                version = record.get("version", "unknown") if isinstance(record, dict) else "unknown"
                return f"/api/dashboard/plot/{version}/{filename}"

            def _publicize_manifest(value):
                if isinstance(value, dict):
                    return {key: _publicize_manifest(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [_publicize_manifest(item) for item in value]
                return _public_plot_url(value) if isinstance(value, str) and value.lower().endswith(".png") else value

            visualization = _publicize_manifest(manifest)

            # Every published version is linked to the exact job that created
            # it. New versions retain these values in meta.json; older versions
            # are resolved from training_jobs so their original result tables
            # and plot manifest remain viewable as well.
            training_result = {}
            try:
                conn = sqlite3.connect(self.registry.db_path)
                conn.row_factory = sqlite3.Row
                job_row = conn.execute(
                    """
                    SELECT job_id, dataset_name, dataset_format, target_column,
                           epochs, kfold, ensemble_method, voting_type, status,
                           progress, message, started_at, completed_at,
                           plot_paths, model_metrics, kfold_results,
                           statistical_results, ensemble_metrics
                    FROM training_jobs
                    WHERE published_version = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (record["version"],),
                ).fetchone()
                conn.close()
                if job_row:
                    training_result = dict(job_row)
                    for field in (
                        "plot_paths", "model_metrics", "kfold_results",
                        "statistical_results", "ensemble_metrics",
                    ):
                        raw_value = training_result.get(field)
                        try:
                            training_result[field] = json.loads(raw_value) if raw_value else {}
                        except (json.JSONDecodeError, TypeError):
                            training_result[field] = {}
                    if not visualization:
                        legacy_plots = training_result.get("plot_paths", {})
                        visualization = _publicize_manifest(
                            legacy_plots.get("version_detail", legacy_plots)
                        )
            except Exception:
                training_result = {}

            # The version's saved metadata is preferred, then the training job
            # provides a complete historical fallback for legacy versions.
            if isinstance(meta, dict):
                for key in ("candidate_test_metrics", "model_metrics", "ensemble_metrics", "training_config"):
                    if key in meta:
                        meta[key] = _clean_for_json(meta[key])
            if not meta.get("model_metrics") and training_result.get("model_metrics"):
                meta["model_metrics"] = _clean_for_json(training_result["model_metrics"])
            if not meta.get("candidate_test_metrics") and training_result.get("model_metrics"):
                meta["candidate_test_metrics"] = _clean_for_json(training_result["model_metrics"])
            if not meta.get("ensemble_metrics") and training_result.get("ensemble_metrics"):
                meta["ensemble_metrics"] = _clean_for_json(training_result["ensemble_metrics"])
            if training_result and not meta.get("training_config"):
                meta["training_config"] = {
                    "dataset_name": training_result.get("dataset_name"),
                    "dataset_format": training_result.get("dataset_format"),
                    "target_column": training_result.get("target_column"),
                    "epochs": training_result.get("epochs"),
                    "kfold": training_result.get("kfold"),
                    "ensemble_method": training_result.get("ensemble_method"),
                    "voting_type": training_result.get("voting_type"),
                    "training_job_id": training_result.get("job_id"),
                    "completed_at": training_result.get("completed_at"),
                }
            package_info = {"published": False}
            package_path = record.get("package_path")
            if package_path and Path(package_path).exists():
                try:
                    import torch
                    blob = torch.load(package_path, map_location="cpu", weights_only=False)
                    if isinstance(blob, dict):
                        package_info = {
                            "published": True,
                            "path": str(package_path),
                            "size_bytes": Path(package_path).stat().st_size,
                            "keys": sorted(blob.keys()),
                            "fields": {
                                key: value for key, value in blob.items()
                                if key not in {"model_state_dict", "state_dict"}
                            },
                            "state_dict_tensors": (
                                len(blob["model_state_dict"])
                                if isinstance(blob.get("model_state_dict"), dict) else None
                            ),
                        }
                    else:
                        package_info = {
                            "published": True,
                            "path": str(package_path),
                            "type": type(blob).__name__,
                        }
                except Exception as exc:
                    package_info = {"published": True, "error": f"{type(exc).__name__}: {exc}"}
            weights_path = record.get("weights_path")
            files = {}
            if dest.exists():
                for item in dest.iterdir():
                    if item.is_file():
                        files[item.name] = item.stat().st_size
            return JSONResponse({
                "status": "success",
                "record": record,
                "meta": meta,
                "visualization": visualization,
                "training_result": _clean_for_json(training_result),
                "package": package_info,
                "files": files,
                "weights_size_bytes": Path(weights_path).stat().st_size if weights_path and Path(weights_path).exists() else None,
            })

        @self.app.get("/api/dashboard/clients")
        async def get_clients():
            """API endpoint for client information."""
            clients = self.get_clients()
            return JSONResponse({
                "status": "success",
                "clients": clients,
                "total_clients": len(clients),
                "timestamp": datetime.now().isoformat()
            })

        @self.app.get("/api/dashboard/updates")
        @self.app.get("/api/v1/updates")
        async def get_updates(limit: int = 100):
            """API endpoint for pending and archived aggregated updates."""
            updates = self.registry.list_updates()
            pending_cnt = sum(1 for u in updates if u["status"] in ("received", "pending"))
            aggregated_cnt = sum(1 for u in updates if u["status"] in ("aggregated", "processed"))
            failed_cnt = sum(1 for u in updates if u["status"] == "failed")
            return JSONResponse({
                "status": "success",
                "updates": [
                    {
                        "id": u["id"],
                        "client_id": u["client_id"],
                        "base_version": u["base_version"],
                        "round": u["round"],
                        "samples": u["samples"],
                        "received_at": u["received_at"],
                        "status": u["status"],
                        "weights_path": u.get("weights_path"),
                        "aggregated_version": u.get("aggregated_version"),
                        "aggregation_method": u.get("aggregation_method"),
                        "aggregated_at": u.get("aggregated_at"),
                        "archive_path": u.get("archive_path"),
                        "dp_clipping_norm": u.get("dp_clipping_norm"),
                        "dp_noise_multiplier": u.get("dp_noise_multiplier"),
                        "local_steps": u.get("local_steps"),
                    }
                    for u in updates[:limit]
                ],
                "total_updates": len(updates),
                "pending_updates": pending_cnt,
                "aggregated_updates": aggregated_cnt,
                "failed_updates": failed_cnt,
                "latest_version": self.registry.latest_version(),
                "open_round": self.registry.get_open_round(),
                "aggregated_folder": str(self.registry.aggregated_updates_dir),
                "timestamp": datetime.now().isoformat()
            })

        @self.app.get("/api/v1/round")
        async def dashboard_open_round():
            """Return active base model version, open FL round, and DP upload policy."""
            from server_hq.central_aggregator import (
                MAX_CLIPPING_NORM,
                MIN_CLIPPING_NORM,
                MIN_NOISE_MULTIPLIER,
                SUPPORTED_FL_METHODS,
            )
            return JSONResponse({
                "status": "success",
                "base_version": self.registry.latest_version(),
                "open_round": self.registry.get_open_round(),
                "supported_methods": list(SUPPORTED_FL_METHODS),
                "dp_policy": {
                    "require_clipped_and_noised": True,
                    "min_clipping_norm": MIN_CLIPPING_NORM,
                    "max_clipping_norm": MAX_CLIPPING_NORM,
                    "min_noise_multiplier": MIN_NOISE_MULTIPLIER,
                },
            })

        @self.app.post("/api/v1/model/aggregate")
        async def run_aggregation_endpoint(request: Request, x_admin_key: Optional[str] = Header(None)):
            """Trigger federated aggregation of validated pending DP updates."""
            self._verify_admin_header(x_admin_key)
            from server_hq.central_aggregator import run_federated_round
            try:
                body = {}
                try:
                    body = await request.json()
                except Exception:
                    form = await request.form()
                    body = dict(form)
                method = body.get("method", "fedavg")
                base_version = body.get("base_version", "latest")
                notes = body.get("notes", "")
                min_updates = int(body.get("min_updates", 1))
                result = run_federated_round(
                    registry=self.registry,
                    base_version=base_version,
                    method=method,
                    notes=notes,
                    min_updates=min_updates,
                )
                return JSONResponse(result)
            except Exception as exc:
                return JSONResponse(status_code=400, content={"status": "error", "detail": str(exc), "message": str(exc)})

        @self.app.post("/api/v1/model/aggregate/compare")
        async def compare_methods_endpoint(request: Request, x_admin_key: Optional[str] = Header(None)):
            """Evaluate all 5 FL methods (FedAvg, FedProx, SCAFFOLD, FedNova, MOON) on the same pending deltas."""
            self._verify_admin_header(x_admin_key)
            from server_hq.central_aggregator import compare_aggregation_methods
            try:
                body = {}
                try:
                    body = await request.json()
                except Exception:
                    form = await request.form()
                    body = dict(form)
                methods = body.get("methods")
                if isinstance(methods, str):
                    methods = [m.strip() for m in methods.split(",") if m.strip()]
                result = compare_aggregation_methods(
                    registry=self.registry,
                    methods=methods,
                    base_version=str(body.get("base_version", "latest")),
                    min_updates=int(body.get("min_updates", 1)),
                )
                return JSONResponse(result)
            except Exception as exc:
                return JSONResponse(status_code=400, content={"status": "error", "detail": str(exc), "message": str(exc)})

        @self.app.get("/api/v1/model/file/{version}")
        async def dashboard_download_weights_file(version: str):
            record = self.registry.get_version(version)
            if record is None or not record.get("weights_path") or not Path(record["weights_path"]).exists():
                raise HTTPException(status_code=404, detail="Weights file not found.")
            return FileResponse(
                path=str(record["weights_path"]),
                filename=f"securefed_weights_v{record['version']}.pt",
                media_type="application/octet-stream",
            )

        @self.app.get("/api/v1/model/package/{version}")
        async def dashboard_download_package_file(version: str):
            record = self.registry.get_version(version)
            if record is None:
                raise HTTPException(status_code=404, detail="Version not found.")
            pkg = record.get("package_path")
            target = pkg if (pkg and Path(pkg).exists()) else record.get("weights_path")
            if not target or not Path(target).exists():
                raise HTTPException(status_code=404, detail="Package or weights file missing.")
            return FileResponse(
                path=str(target),
                filename=f"securefed_package_v{record['version']}.pt",
                media_type="application/octet-stream",
            )

        @self.app.post("/api/dashboard/models/{version}/push")
        async def push_model_to_edge(version: str, request: Request, x_admin_key: Optional[str] = Header(None)):
            """Push a published HQ model version directly to registered Client Edge node(s)."""
            admin_key = self._verify_admin_header(x_admin_key)
            import requests
            from config import API_CONFIG

            record = self.registry.get_version(version)
            if record is None:
                raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")

            body = {}
            try:
                body = await request.json()
            except Exception:
                body = {}

            requested_edge = (body.get("edge_url") or "").strip().rstrip("/")
            targets = []
            if requested_edge:
                targets = [requested_edge]
            else:
                for client in self.registry.list_clients():
                    disp = (client.get("display_name") or "").strip().rstrip("/")
                    if disp.startswith(("http://", "https://")) and disp not in targets:
                        targets.append(disp)
                    elif disp and not disp.startswith(("http://", "https://")):
                        if any(c.isdigit() for c in disp):
                            host_part = disp.split(":")[0]
                            port_part = disp.split(":")[1] if ":" in disp else "5000"
                            formatted = f"http://{host_part}:{port_part}"
                            if formatted not in targets:
                                targets.append(formatted)
                if not targets:
                    targets = ["http://127.0.0.1:5000"]

            hq_host = request.url.hostname or "127.0.0.1"
            api_port = int(API_CONFIG.get("port", 8000))
            hq_api_url = f"http://{hq_host}:{api_port}"

            results = []
            success_count = 0
            for edge_url in targets:
                try:
                    resp = requests.post(
                        f"{edge_url}/api/client/pull-model",
                        json={"admin_key": admin_key, "version": record["version"], "hq_url": hq_api_url, "source": "HQ Push"},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        success_count += 1
                        results.append({"edge_url": edge_url, "status": "success", "detail": resp.json().get("message", "Deployed")})
                    else:
                        results.append({"edge_url": edge_url, "status": "error", "detail": resp.text})
                except Exception as exc:
                    results.append({"edge_url": edge_url, "status": "unreachable", "detail": str(exc)})

            status = "success" if success_count > 0 else "error"
            push_info = None
            if success_count > 0:
                pushed_targets = [r["edge_url"] for r in results if r["status"] == "success"]
                push_info = self.registry.record_edge_push(record["version"], pushed_targets)

            return JSONResponse(
                status_code=200 if success_count > 0 else 502,
                content={
                    "status": status,
                    "version": record["version"],
                    "pushed_count": success_count,
                    "total_targets": len(targets),
                    "push_info": push_info,
                    "results": results,
                },
            )

        @self.app.delete("/api/dashboard/models/{version}")
        async def delete_model_version(version: str, x_admin_key: Optional[str] = Header(None)):
            """Delete a published model version from HQ registry and disk."""
            self._verify_admin_header(x_admin_key)
            try:
                res = self.registry.delete_version(version)
            except ValueError as val_err:
                raise HTTPException(status_code=400, detail=str(val_err)) from val_err
            if res is None:
                raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")
            return JSONResponse({
                "status": "success",
                "message": f"Deleted model version v{res['deleted_version']}.",
                **res,
            })

        @self.app.get("/api/dashboard/stats")
        async def get_dashboard_stats():
            """API endpoint for dashboard statistics."""
            return JSONResponse({
                "status": "success",
                "model_stats": {
                    "total_versions": self.registry.count_versions(),
                    "latest_version": self.registry.latest_version(),
                    "total_updates": self.count_updates(),
                    "active_clients": self.count_active_clients()
                },
                "client_stats": self.get_client_stats(),
                "update_stats": self.get_update_stats(),
                "timestamp": datetime.now().isoformat()
            })

        @self.app.post("/api/dashboard/training/upload")
        async def upload_training_dataset(
            dataset_name: str = Form(...),
            dataset_format: str = Form(...),
            target_column: str = Form(""),
            epochs: int = Form(3),
            kfold: int = Form(5),
            ensemble_method: str = Form("voting"),
            voting_type: str = Form("soft"),
            file: UploadFile = File(...)
        ):
            """Upload a dataset CSV/Parquet for HQ model training.
            Returns a job_id to track training progress."""
            job_id = str(uuid.uuid4())
            safe_name = "".join(c for c in dataset_name if c.isalnum() or c in "._- ").strip() or "dataset"
            ext = Path(file.filename).suffix.lower()
            if ext not in ('.csv', '.parquet', '.pq'):
                raise HTTPException(status_code=400, detail="Unsupported format; only CSV and Parquet are accepted.")
            dataset_path = self.training_uploads_dir / f"{job_id}_{safe_name}{ext}"
            contents = await file.read()
            dataset_path.write_bytes(contents)
            # Initialize job record
            self.ensure_database_tables()
            conn = sqlite3.connect(self.registry.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_jobs (job_id, dataset_name, dataset_path, dataset_format, target_column, epochs, kfold, ensemble_method, voting_type, status, log_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)
            ''', (job_id, dataset_name, str(dataset_path), dataset_format, target_column, epochs, kfold, ensemble_method, voting_type, f"{self.training_logs_dir}/{job_id}.log"))
            conn.commit()
            conn.close()
            # Start background training off the event loop with stop signal support
            self._stop_events[job_id] = threading.Event()
            self._training_tasks[job_id] = asyncio.create_task(self._run_training_job(
                job_id, str(dataset_path), dataset_format, target_column,
                epochs, kfold, ensemble_method, voting_type
            ))
            return JSONResponse({
                "status": "success",
                "job_id": job_id,
                "message": f"Dataset '{dataset_name}' uploaded; training started in background. Use /api/dashboard/training/status/{job_id} to monitor."
            })

        @self.app.get("/api/dashboard/training/status/{job_id}")
        async def training_status(job_id: str):
            """Get training job status and progress."""
            self.ensure_database_tables()
            conn = sqlite3.connect(self.registry.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT job_id, dataset_name, dataset_format, target_column, epochs, kfold, ensemble_method, voting_type, status, progress, message, started_at, completed_at, published_version, log_path, plot_paths, model_metrics, kfold_results, statistical_results, ensemble_metrics FROM training_jobs WHERE job_id = ?', (job_id,))
            row = cursor.fetchone()
            columns = [desc[0] for desc in cursor.description]
            conn.close()
            if not row:
                raise HTTPException(status_code=404, detail="Training job not found.")
            job = dict(zip(columns, row))
            # Append log tail if file exists
            log_path = job.get("log_path")
            if log_path and os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as lf:
                        lines = lf.readlines()[-3:]
                        job["log_tail"] = ''.join(lines)
                except Exception:
                    job["log_tail"] = ""
            else:
                job["log_tail"] = ""
            return JSONResponse({"status": "success", "job": job})

        @self.app.get("/api/dashboard/training/jobs")
        async def training_jobs():
            """List all training jobs."""
            conn = sqlite3.connect(self.registry.db_path)
            cursor = conn.cursor()
            self.ensure_database_tables()
            cursor.execute('SELECT job_id, dataset_name, dataset_format, target_column, epochs, kfold, ensemble_method, voting_type, status, progress, message, published_version, created_at FROM training_jobs ORDER BY created_at DESC')
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            jobs = [dict(zip(columns, row)) for row in rows]
            conn.close()
            return JSONResponse({"status": "success", "jobs": jobs})

        @self.app.post("/api/dashboard/training/stop/{job_id}")
        async def training_stop(job_id: str):
            """Request cooperative stop for a running/queued training job."""
            conn = sqlite3.connect(self.registry.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT job_id FROM training_jobs WHERE job_id = ?", (job_id,))
                if not cursor.fetchone():
                    raise HTTPException(status_code=404, detail="Training job not found.")
                stop_ev = self._stop_events.setdefault(job_id, threading.Event())
                stop_ev.set()
                cursor.execute(
                    "UPDATE training_jobs SET status = 'stopped', message = 'Stop requested by user.' WHERE job_id = ?",
                    (job_id,),
                )
                conn.commit()
            finally:
                conn.close()
            return JSONResponse({"status": "success", "job_id": job_id, "message": "Stop signal sent."})

        @self.app.post("/api/dashboard/training/resume/{job_id}")
        async def training_resume(job_id: str):
            """Resume a stopped/queued training job."""
            existing_task = self._training_tasks.get(job_id)
            if existing_task and not existing_task.done():
                return JSONResponse({"status": "success", "job_id": job_id, "message": "Training job already running."})

            conn = sqlite3.connect(self.registry.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT job_id FROM training_jobs WHERE job_id = ?", (job_id,))
                if not cursor.fetchone():
                    raise HTTPException(status_code=404, detail="Training job not found.")
                self._stop_events[job_id] = threading.Event()
                cursor.execute("UPDATE training_jobs SET status = 'queued', message = 'Resume requested.' WHERE job_id = ?", (job_id,))
                conn.commit()
            finally:
                conn.close()
            self._training_tasks[job_id] = asyncio.create_task(self._run_training_job(job_id))
            return JSONResponse({"status": "success", "job_id": job_id, "message": "Resume requested."})

        @self.app.post("/api/dashboard/training/cleanup/{job_id}")
        async def training_cleanup_job(job_id: str):
            """Delete a specific training job and all its artifacts (logs, uploads, plots)."""
            conn = sqlite3.connect(self.registry.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT job_id, log_path, dataset_path, plot_paths FROM training_jobs WHERE job_id = ?', (job_id,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                raise HTTPException(status_code=404, detail="Training job not found.")
            _, log_path, dataset_path, plot_paths_json = row
            deleted = []
            # Remove log file
            if log_path and os.path.exists(log_path):
                try:
                    os.remove(log_path)
                    deleted.append("log")
                except Exception:
                    pass
            # Remove uploaded dataset
            if dataset_path and os.path.exists(dataset_path):
                try:
                    os.remove(dataset_path)
                    deleted.append("dataset")
                except Exception:
                    pass
            # Remove plot files
            if plot_paths_json:
                try:
                    plot_files = json.loads(plot_paths_json)
                    for p in plot_files.values():
                        if p and os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                    deleted.append("plots")
                except Exception:
                    pass
            # Remove job record
            cursor.execute('DELETE FROM training_jobs WHERE job_id = ?', (job_id,))
            conn.commit()
            conn.close()
            return JSONResponse({"status": "success", "job_id": job_id, "deleted": deleted})

        @self.app.post("/api/dashboard/training/cleanup")
        async def training_cleanup():
            """Remove old completed/failed/stopped training artifacts."""
            keep_days = int(os.environ.get("TRAINING_KEEP_DAYS", "30"))
            cutoff_iso = (datetime.now() - timedelta(days=keep_days)).isoformat()
            conn = sqlite3.connect(self.registry.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT job_id, log_path, dataset_path, plot_paths FROM training_jobs "
                    "WHERE status IN ('completed','failed','stopped') AND completed_at IS NOT NULL AND completed_at < ?",
                    (cutoff_iso,),
                )
                old_jobs = cursor.fetchall()
                for old_id, log_path, dataset_path, plot_paths_json in old_jobs:
                    for p in (log_path, dataset_path):
                        if p and os.path.exists(p):
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                    if plot_paths_json:
                        try:
                            for p in (json.loads(plot_paths_json) or {}).values():
                                if isinstance(p, str) and os.path.exists(p):
                                    try:
                                        os.remove(p)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    cursor.execute("DELETE FROM training_jobs WHERE job_id = ?", (old_id,))
                conn.commit()
            finally:
                conn.close()
            return JSONResponse({"status": "success", "cleaned": len(old_jobs)})

        @self.app.websocket("/ws/dashboard")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            token = websocket.cookies.get(SESSION_COOKIE_NAME) or websocket.query_params.get("token")
            if not verify_session_token(token):
                header_key = websocket.headers.get("x-admin-key")
                if not (header_key and header_key.strip() in _get_allowed_passwords()):
                    await websocket.close(code=1008)
                    return

            await websocket.accept()
            self.active_connections.append(websocket)

            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()

                    # Send periodic updates
                    stats = {
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat(),
                        "active_clients": self.count_active_clients(),
                        "latest_version": self.registry.latest_version()
                    }
                    await websocket.send_json(stats)

            except WebSocketDisconnect:
                self.active_connections.remove(websocket)

        @self.app.get("/dashboard/version/{version_id}")
        async def version_detail(request: Request, version_id: str):
            """Detailed view for a specific model version."""
            try:
                version = self.registry.get_version(version_id)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            if not version:
                return JSONResponse({"error": "Version not found"}, status_code=404)

            return self.templates.TemplateResponse(
                request=request,
                name="version_detail.html",
                context={
                    "request": request,
                    "version": version,
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def setup_events(self):
        """Setup application startup/shutdown events."""

        @self.app.on_event("startup")
        async def startup_event():
            """Initialize dashboard on startup and restart any queued jobs."""
            print(f"SecureFed-NIDS HQ Dashboard starting up at {datetime.now()}")
            self.ensure_database_tables()
            # Restart any jobs that were left in 'queued' state
            # (they were created before a server restart and lost their asyncio tasks)
            try:
                conn = sqlite3.connect(self.registry.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT job_id FROM training_jobs WHERE status = 'queued'")
                queued = cursor.fetchall()
                conn.close()
                for (jid,) in queued:
                    print(f"Restarting queued job {jid}")
                    self._stop_events[jid] = threading.Event()
                    self._training_tasks[jid] = asyncio.create_task(self._run_training_job(jid))
            except Exception as e:
                print(f"Error restarting queued jobs: {e}")

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown."""
            print(f"SecureFed-NIDS HQ Dashboard shutting down at {datetime.now()}")
            for connection in self.active_connections:
                try:
                    await connection.close()
                except:
                    pass

    def _set_training_job(self, job_id: str, **fields) -> None:
        """Update a training_jobs row by field name (only allowed fields)."""
        allowed = {"status", "progress", "message", "started_at", "completed_at", "published_version", "plot_paths", "model_metrics", "kfold_results", "statistical_results", "ensemble_metrics"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        cols = ", ".join(f"{k} = ?" for k in updates)
        conn = sqlite3.connect(self.registry.db_path)
        conn.execute(f"UPDATE training_jobs SET {cols} WHERE job_id = ?",
                     (*updates.values(), job_id))
        conn.commit()
        conn.close()

    def _training_job_row(self, job_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.registry.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT job_id, dataset_name, dataset_path, dataset_format, target_column,'
            ' epochs, kfold, ensemble_method, voting_type, status, progress, message,'
            ' log_path, started_at, completed_at, created_at, published_version, plot_paths,'
            ' model_metrics, kfold_results, statistical_results, ensemble_metrics'
            ' FROM training_jobs WHERE job_id = ?',
            (job_id,))
        row = cursor.fetchone()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return dict(zip(columns, row)) if row else None

    def _is_job_stopped(self, job_id: str, stop_event: Optional[threading.Event] = None) -> bool:
        if stop_event is not None and stop_event.is_set():
            return True
        row = self._training_job_row(job_id)
        return bool(row and row.get("status") == "stopped")

    async def _run_training_job(self, job_id: str, dataset_path: Optional[str] = None,
                                dataset_format: str = "", target_column: str = "",
                                epochs: int = 3, kfold: int = 5,
                                ensemble_method: str = "voting",
                                voting_type: str = "soft") -> None:
        """Run the CPU-bound HQ training pipeline off the event loop in a worker thread."""
        stop_event = self._stop_events.setdefault(job_id, threading.Event())
        try:
            await asyncio.to_thread(
                self._execute_training_pipeline_sync,
                job_id, dataset_path, dataset_format, target_column,
                epochs, kfold, ensemble_method, voting_type, stop_event,
            )
        finally:
            self._stop_events.pop(job_id, None)
            self._training_tasks.pop(job_id, None)

    def _execute_training_pipeline_sync(
        self, job_id: str, dataset_path: Optional[str],
        dataset_format: str, target_column: str,
        epochs: int, kfold: int, ensemble_method: str, voting_type: str,
        stop_event: threading.Event,
    ) -> None:
        """Run the HQ model-training pipeline synchronously on a background thread.

        Trains the 6 candidate DL models on the uploaded dataset, selects the
        top-2 ensemble, saves the weights under hq_store/models, and publishes
        a new version to the registry. Progress and stop state are checked via
        SQLite and stop_event so stopping a job cancels further stages cleanly.
        """
        job = self._training_job_row(job_id)
        if job is None:
            return
        if not dataset_path:
            dataset_path = job.get("dataset_path")
        dataset_format = dataset_format or (job.get("dataset_format") or "")
        target_column = target_column or job.get("target_column") or ""
        epochs = epochs or int(job.get("epochs") or 3)
        kfold = kfold or int(job.get("kfold") or 5)
        ensemble_method = ensemble_method or (job.get("ensemble_method") or "voting")
        voting_type = voting_type or (job.get("voting_type") or "soft")

        log_path = self.training_logs_dir / f"{job_id}.log"
        log_handle = open(log_path, "w", encoding="utf-8")

        def log(message: str) -> None:
            log_handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")
            log_handle.flush()

        def ensure_not_stopped() -> None:
            if self._is_job_stopped(job_id, stop_event):
                raise _TrainingStoppedError("Training job stopped by user.")

        try:
            ensure_not_stopped()
            if not dataset_path or not Path(dataset_path).exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")

            self._set_training_job(job_id, status="running", progress=5,
                                   started_at=datetime.now().isoformat())
            log(f"Job started | dataset={Path(dataset_path).name} | format={dataset_format or 'auto'} "
                f"| target='{target_column or 'auto'}' | epochs={epochs} | kfold={kfold} "
                f"| ensemble={ensemble_method}/{voting_type}")

            # Make project modules importable regardless of the launch CWD.
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            import numpy as np
            import torch
            from main import load_and_preprocess_dataset
            from models.architectures import ModelFactory
            from models.training import train_single_model
            from models.ensemble import EnsembleTrainer

            X_train, y_train, X_val, y_val, X_test, y_test, feature_names = \
                load_and_preprocess_dataset(dataset_path, target_column or None)
            ensure_not_stopped()
            scaler_stats = getattr(load_and_preprocess_dataset, "last_scaler", None) or {}
            input_dim = X_train.shape[1]
            num_classes = int(len(np.unique(y_train)))
            log(f"Preprocessing done | {X_train.shape[0]} train / {X_val.shape[0]} val samples, "
                f"{input_dim} features, {num_classes} classes")
            self._set_training_job(job_id, progress=15)

            config = {"epochs": epochs, "batch_size": 128, "learning_rate": 0.001}
            model_names = ["mlp", "cnn", "lstm", "bilstm", "gru", "transformer"]
            metrics: Dict[str, Dict[str, float]] = {}
            models: Dict[str, Any] = {}
            histories: Dict[str, Dict[str, Any]] = {}
            cv_results: Dict[str, Dict[str, Any]] = {}

            # Stage 2: a single stratified K-fold run per candidate model. The
            # train+validation split is used here; the untouched test split is
            # reserved for the final ensemble evaluation.
            from models.evaluation import (
                run_kfold_cv, paired_t_test, mcnemar_test,
            )
            cv_device = "cuda" if torch.cuda.is_available() else "cpu"
            X_cv = np.vstack([X_train, X_val])
            y_cv = np.concatenate([y_train, y_val])
            for i, name in enumerate(model_names, start=1):
                ensure_not_stopped()
                log(f"{name.upper()}: running {kfold}-fold CV and final training ({i}/{len(model_names)}) ...")
                try:
                    cv_res = run_kfold_cv(
                        model_factory_fn=lambda model_name=name: ModelFactory.create_model(
                            model_name, input_dim, num_classes
                        ),
                        model_name=name,
                        X=X_cv,
                        y=y_cv,
                        k_folds=kfold,
                        epochs=epochs,
                        batch_size=config["batch_size"],
                        lr=config["learning_rate"],
                        device=cv_device,
                    )
                    cv_results[name] = cv_res
                    model, history = train_single_model(
                        name, input_dim, num_classes,
                        X_train, y_train, X_val, y_val, config
                    )
                    models[name] = model
                    histories[name] = history
                    # Use the same full metric set for top-k selection and for
                    # the dashboard result table.
                    metrics[name] = {
                        "accuracy": cv_res["mean_accuracy"],
                        "precision": cv_res["mean_precision"],
                        "recall": cv_res["mean_recall"],
                        "f1": cv_res["mean_f1"],
                        "mcc": cv_res["mean_mcc"],
                        "specificity": cv_res["mean_specificity"],
                        "auc": cv_res["mean_auc"],
                        "roc_auc": cv_res["mean_auc"],
                        "std_acc": cv_res["std_accuracy"],
                        "training_time_seconds": cv_res["mean_train_time"],
                        "inference_time_per_sample_ms": cv_res["mean_infer_time"] * 1000.0,
                    }
                    log(
                        f"{name.upper()} done | CV acc={cv_res['mean_accuracy']:.4f} "
                        f"±{cv_res['std_accuracy']:.4f} | train={cv_res['mean_train_time']:.2f}s "
                        f"| infer={cv_res['mean_infer_time'] * 1000:.3f}ms/sample"
                    )
                except Exception as exc:
                    log(f"{name.upper()} failed: {type(exc).__name__}: {exc}")
                ensure_not_stopped()
                self._set_training_job(job_id, progress=15 + int(55 * i / len(model_names)))

            ensure_not_stopped()
            if not models:
                raise RuntimeError("No candidate model could be trained — check the dataset contents.")

            log(f"Selecting top-2 ensemble ({ensemble_method}/{voting_type}) ...")
            self._set_training_job(job_id, progress=75)
            trainer = EnsembleTrainer(config={
                "top_k": 2,
                "selection_metric": "accuracy",
                "ensemble_method": ensemble_method,
                "voting_type": voting_type,
            })
            ensemble_model = trainer.train_ensemble(
                model_results=metrics, models=models,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                method=ensemble_method, voting_type=voting_type)
            ensemble_eval = trainer.evaluate_ensemble(X_test, y_test)
            accuracy = float(ensemble_eval["accuracy"])
            from sklearn.metrics import confusion_matrix
            from models.evaluation import calculate_metrics
            ensemble_eval["y_true"] = np.asarray(y_test).tolist()
            ensemble_eval["confusion_matrix"] = confusion_matrix(
                y_test, ensemble_eval["predictions"]
            ).tolist()

            # Evaluate every trained candidate on the identical untouched test
            # split.  This supplies scientifically comparable metric charts and
            # one confusion matrix per candidate; no raw traffic is persisted.
            individual_evaluations: Dict[str, Dict[str, Any]] = {}
            for candidate_name, candidate_model in models.items():
                candidate_model.eval()
                candidate_device = next(candidate_model.parameters()).device
                x_test_tensor = torch.as_tensor(
                    X_test, dtype=torch.float32, device=candidate_device
                )
                if candidate_device.type == "cuda":
                    torch.cuda.synchronize(candidate_device)
                infer_started = __import__("time").perf_counter()
                with torch.no_grad():
                    logits = candidate_model(x_test_tensor)
                    candidate_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    candidate_preds = candidate_probs.argmax(axis=1)
                if candidate_device.type == "cuda":
                    torch.cuda.synchronize(candidate_device)
                candidate_inference_seconds = float(__import__("time").perf_counter() - infer_started)
                candidate_metrics = calculate_metrics(y_test, candidate_preds, candidate_probs)
                candidate_metrics.update({
                    "confusion_matrix": confusion_matrix(y_test, candidate_preds).tolist(),
                    "inference_time_seconds": candidate_inference_seconds,
                    "inference_time_per_sample_ms": candidate_inference_seconds * 1000.0 / max(len(y_test), 1),
                })
                individual_evaluations[candidate_name] = candidate_metrics
                # Keep the dashboard result table aligned with test-set plots.
                metrics[candidate_name].update(candidate_metrics)
            log(
                f"Ensemble test metrics | accuracy={accuracy:.4f} | "
                f"precision={ensemble_eval['precision']:.4f} | recall={ensemble_eval['recall']:.4f} | "
                f"f1={ensemble_eval['f1']:.4f} | mcc={ensemble_eval['mcc']:.4f} | "
                f"specificity={ensemble_eval['specificity']:.4f} | auc={ensemble_eval['auc']:.4f}"
            )

            # Stage 3: paired t-test over fold accuracies and McNemar on the
            # untouched test set (best single model versus the ensemble).
            ranked = sorted(metrics.items(), key=lambda item: item[1].get("accuracy", 0.0), reverse=True)
            best_name = ranked[0][0]
            second_name = ranked[1][0] if len(ranked) > 1 else None
            statistical_results: Dict[str, Any] = {}
            if second_name and best_name in cv_results and second_name in cv_results:
                statistical_results["best_vs_second"] = {
                    "comparison": f"{best_name.upper()} vs {second_name.upper()}",
                    "paired_t_test": paired_t_test(
                        cv_results[best_name]["fold_accuracies"],
                        cv_results[second_name]["fold_accuracies"],
                    ),
                }
            best_model = models[best_name]
            best_device = next(best_model.parameters()).device
            best_model.eval()
            with torch.no_grad():
                best_preds = best_model(
                    torch.as_tensor(X_test, dtype=torch.float32, device=best_device)
                ).argmax(dim=1).detach().cpu().numpy()
            statistical_results["best_vs_ensemble"] = {
                "comparison": f"{best_name.upper()} vs ENSEMBLE",
                "mcnemar": mcnemar_test(y_test, best_preds, ensemble_eval["predictions"]),
            }
            log(
                f"Statistical tests complete | best={best_name.upper()} | "
                f"t-test/McNemar results stored"
            )
            self._set_training_job(job_id, progress=88)

            # Save only scalar/plot-ready ensemble fields in SQLite; raw arrays
            # remain available to the plot generator during this job.
            ensemble_metrics_for_db = {
                key: value for key, value in ensemble_eval.items()
                if key not in {"predictions", "probabilities", "y_true"}
            }

            try:
                plots_save_dir = self.training_logs_dir / "plots"
                plots_save_dir.mkdir(parents=True, exist_ok=True)
                # A single unified manifest is the source of truth for both
                # Model Training and Model Versions. Therefore the version
                # details show the exact same evaluations and plots generated
                # by this job—never an unrelated reference table.
                plot_files = create_training_plots(
                    ensemble_metrics=ensemble_eval,
                    model_histories=histories,
                    save_dir=plots_save_dir,
                    job_id=job_id,
                    kfold_results=cv_results,
                    statistical_results=statistical_results,
                )
                version_plot_manifest = create_version_detail_plots(
                    ensemble_metrics=ensemble_eval,
                    model_histories=histories,
                    individual_evaluations=individual_evaluations,
                    model_metrics=metrics,
                    save_dir=plots_save_dir,
                    job_id=job_id,
                )
                # Flatten all actual job plots into one manifest. It can be
                # rendered identically from either page.
                plot_files.update(version_plot_manifest)
                plot_paths_json = json.dumps(_clean_for_json(plot_files))
                log(
                    "Generated training/result plots: per-model accuracy/loss, "
                    "candidate and ensemble confusion matrices, metric charts, "
                    "K-fold and statistical-test visualizations."
                )
            except Exception as plot_exc:
                # One optional chart must never erase plots that were already
                # generated. Keep the successful top-level plots and record the
                # warning in the job log for diagnosis.
                plot_files = locals().get("plot_files", {}) or {}
                version_plot_manifest = locals().get("version_plot_manifest", {}) or {}
                plot_paths_json = json.dumps(_clean_for_json(plot_files))
                log(f"Plot generation warning: {type(plot_exc).__name__}: {plot_exc}")

            # Persist display-safe metadata with the immutable published version.
            # The manifest references only generated plot files; no raw samples,
            # feature rows, predictions, or traffic are copied into the registry.
            # If the optional detail generator failed, retain the successful
            # top-level training plots rather than publishing an empty manifest.
            ensure_not_stopped()
            try:
                self.registry.save_eval_holdout(
                    X_test=X_test,
                    y_test=y_test,
                    input_dim=input_dim,
                    feature_names=feature_names,
                    dataset_name=Path(dataset_path).name,
                )
            except Exception as holdout_exc:
                log(f"Warning: could not cache eval holdout: {holdout_exc}")

            if not version_plot_manifest:
                version_plot_manifest = plot_files
            published = self.registry.publish_ensemble(
                ensemble_model=ensemble_model,
                input_dim=input_dim,
                num_classes=num_classes,
                feature_names=feature_names,
                accuracy=accuracy,
                notes=f"dashboard training job {job_id[:8]} | ensemble={ensemble_method}",
                extra_metadata={
                    "training_job_id": job_id,
                    "scaler_mean": scaler_stats.get("mean"),
                    "scaler_scale": scaler_stats.get("scale"),
                    "training_config": {
                        "epochs": epochs,
                        "kfold": kfold,
                        "ensemble_method": ensemble_method,
                        "voting_type": voting_type,
                    },
                    "model_metrics": _clean_for_json(metrics),
                    "candidate_test_metrics": _clean_for_json(individual_evaluations),
                    "ensemble_metrics": _clean_for_json(ensemble_metrics_for_db),
                    "kfold_results": _clean_for_json(cv_results),
                    "statistical_results": _clean_for_json(statistical_results),
                    "plot_manifest": _clean_for_json(version_plot_manifest),
                },
            )
            log(f"Published to HQ registry: v{published['version']} → {published['weights_path']}")
            self._set_training_job(
                job_id, status="completed", progress=100,
                message=f"Published v{published['version']} | accuracy {accuracy * 100:.2f}% | "
                        f"{len(cv_results)} models with {kfold}-fold CV",
                published_version=published["version"],
                plot_paths=plot_paths_json,
                model_metrics=json.dumps(_clean_for_json(metrics)),
                kfold_results=json.dumps(_clean_for_json(cv_results)),
                statistical_results=json.dumps(_clean_for_json(statistical_results)),
                ensemble_metrics=json.dumps(_clean_for_json(ensemble_metrics_for_db)),
                completed_at=datetime.now().isoformat())
            log("Job completed successfully.")
        except _TrainingStoppedError:
            log("Job stopped cleanly by user request.")
            self._set_training_job(
                job_id, status="stopped",
                message="Training stopped by user.",
                completed_at=datetime.now().isoformat())
        except Exception as exc:
            log(f"Job failed: {type(exc).__name__}: {exc}")
            self._set_training_job(
                job_id, status="failed",
                message=f"{type(exc).__name__}: {exc}",
                completed_at=datetime.now().isoformat())
        finally:
            log_handle.close()

    def ensure_database_tables(self):
        """Ensure required database tables exist (migration-safe)."""
        conn = sqlite3.connect(self.registry.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dashboard_clients (
                client_id TEXT PRIMARY KEY,
                client_name TEXT,
                ip_address TEXT,
                last_seen TIMESTAMP,
                status TEXT DEFAULT 'active',
                model_version TEXT,
                download_count INTEGER DEFAULT 0,
                upload_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dashboard_updates (
                update_id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT,
                version_from TEXT,
                version_to TEXT,
                update_size INTEGER,
                dp_epsilon REAL,
                accuracy REAL,
                status TEXT DEFAULT 'pending',
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (client_id) REFERENCES dashboard_clients (client_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_jobs (
                job_id TEXT PRIMARY KEY,
                dataset_name TEXT,
                dataset_path TEXT,
                dataset_format TEXT,
                target_column TEXT,
                epochs INTEGER DEFAULT 3,
                kfold INTEGER DEFAULT 5,
                ensemble_method TEXT DEFAULT 'voting',
                voting_type TEXT DEFAULT 'soft',
                status TEXT DEFAULT 'queued',
                progress INTEGER DEFAULT 0,
                message TEXT,
                log_path TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                published_version TEXT,
                plot_paths TEXT,
                model_metrics TEXT,
                kfold_results TEXT,
                statistical_results TEXT,
                ensemble_metrics TEXT
            )
        ''')
        cursor.execute("PRAGMA table_info(training_jobs)")
        training_job_columns = {row[1] for row in cursor.fetchall()}
        migration_columns = {
            "published_version": "TEXT", "plot_paths": "TEXT",
            "model_metrics": "TEXT", "kfold_results": "TEXT",
            "statistical_results": "TEXT", "ensemble_metrics": "TEXT",
        }
        for column, sql_type in migration_columns.items():
            if column not in training_job_columns:
                cursor.execute(f"ALTER TABLE training_jobs ADD COLUMN {column} {sql_type}")
        conn.commit()
        conn.close()

    def get_clients(self) -> List[Dict[str, Any]]:
        """Get list of registered clients."""
        conn = sqlite3.connect(self.registry.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT
            client_id, client_name, ip_address, last_seen, status,
            model_version, download_count, upload_count, created_at
        FROM dashboard_clients
        ORDER BY last_seen DESC
        LIMIT 100
        ''')

        columns = [desc[0] for desc in cursor.description]
        clients = []

        for row in cursor.fetchall():
            client = dict(zip(columns, row))
            client["last_seen"] = client["last_seen"] or client["created_at"]
            clients.append(client)

        conn.close()
        return clients

    def get_recent_updates(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent model updates."""
        conn = sqlite3.connect(self.registry.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT
            update_id, client_id, version_from, version_to,
            update_size, dp_epsilon, accuracy, status,
            processed_at, created_at
        FROM dashboard_updates
        ORDER BY created_at DESC
        LIMIT ?
        ''', (limit,))

        columns = [desc[0] for desc in cursor.description]
        updates = []

        for row in cursor.fetchall():
            update = dict(zip(columns, row))
            updates.append(update)

        conn.close()
        return updates

    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        conn = sqlite3.connect(self.registry.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM dashboard_clients')
        total_clients = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM dashboard_clients WHERE status = 'active'")
        active_clients = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM dashboard_clients WHERE status = 'inactive'")
        inactive_clients = cursor.fetchone()[0]

        cursor.execute('SELECT SUM(download_count) FROM dashboard_clients')
        total_downloads = cursor.fetchone()[0] or 0

        cursor.execute('SELECT SUM(upload_count) FROM dashboard_clients')
        total_uploads = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_clients": total_clients,
            "active_clients": active_clients,
            "inactive_clients": inactive_clients,
            "total_downloads": total_downloads,
            "total_uploads": total_uploads,
            "download_ratio": total_downloads / max(total_clients, 1),
            "upload_ratio": total_uploads / max(total_clients, 1)
        }

    def get_update_stats(self) -> Dict[str, Any]:
        """Get update statistics unified across client_updates, dashboard_updates, and model_versions."""
        conn = sqlite3.connect(self.registry.db_path)
        try:
            cursor = conn.cursor()

            # Prefer live FL updates table (`client_updates`) as source of truth
            cursor.execute("SELECT COUNT(*) FROM client_updates")
            cu_total = cursor.fetchone()[0] or 0

            if cu_total > 0:
                total_updates = cu_total
                cursor.execute("SELECT COUNT(*) FROM client_updates WHERE status IN ('aggregated', 'processed')")
                processed_updates = cursor.fetchone()[0] or 0
                cursor.execute("SELECT COUNT(*) FROM client_updates WHERE status IN ('received', 'pending')")
                pending_updates = cursor.fetchone()[0] or 0
                cursor.execute("SELECT COUNT(*) FROM client_updates WHERE status = 'failed'")
                failed_updates = cursor.fetchone()[0] or 0
            else:
                cursor.execute("SELECT COUNT(*) FROM dashboard_updates")
                total_updates = cursor.fetchone()[0] or 0
                cursor.execute("SELECT COUNT(*) FROM dashboard_updates WHERE status IN ('processed', 'aggregated')")
                processed_updates = cursor.fetchone()[0] or 0
                cursor.execute("SELECT COUNT(*) FROM dashboard_updates WHERE status IN ('pending', 'received')")
                pending_updates = cursor.fetchone()[0] or 0
                cursor.execute("SELECT COUNT(*) FROM dashboard_updates WHERE status = 'failed'")
                failed_updates = cursor.fetchone()[0] or 0

            cursor.execute("SELECT AVG(accuracy) FROM dashboard_updates WHERE accuracy IS NOT NULL")
            avg_accuracy = cursor.fetchone()[0]
            if avg_accuracy is None:
                cursor.execute("SELECT AVG(accuracy) FROM model_versions WHERE accuracy IS NOT NULL")
                avg_accuracy = cursor.fetchone()[0] or 0.0

            cursor.execute("SELECT AVG(dp_epsilon) FROM dashboard_updates WHERE dp_epsilon IS NOT NULL")
            avg_epsilon = cursor.fetchone()[0] or 0.0

            return {
                "total_updates": total_updates,
                "processed_updates": processed_updates,
                "pending_updates": pending_updates,
                "failed_updates": failed_updates,
                "avg_accuracy": float(avg_accuracy or 0.0),
                "avg_epsilon": float(avg_epsilon or 0.0),
                "success_rate": processed_updates / max(total_updates, 1),
            }
        finally:
            conn.close()

    def count_updates(self) -> int:
        """Count total updates across primary FL registry (falling back to legacy dashboard_updates)."""
        conn = sqlite3.connect(self.registry.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM client_updates")
            cu_count = cursor.fetchone()[0] or 0
            if cu_count > 0:
                return int(cu_count)
            cursor.execute("SELECT COUNT(*) FROM dashboard_updates")
            return int(cursor.fetchone()[0] or 0)
        finally:
            conn.close()

    def count_active_clients(self) -> int:
        """Count active clients."""
        conn = sqlite3.connect(self.registry.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dashboard_clients WHERE status = 'active'")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            import psutil
            import platform
            return {
                "system": platform.system(),
                "python_version": platform.python_version(),
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "timestamp": datetime.now().isoformat()
            }
        except ImportError:
            return {
                "system": "Linux",
                "python_version": "Unknown",
                "cpu_percent": 0,
                "memory_percent": 0,
                "disk_usage": 0,
                "timestamp": datetime.now().isoformat()
            }


# Module-level ASGI app so `uvicorn server_hq.dashboard:app` works.
_app_instance = Dashboard()
app = _app_instance.app


def run_dashboard(host: str = "0.0.0.0", port: int = 8001):
    """Run the dashboard server."""
    print("\n" + "="*70)
    print("SECUREFED-NIDS HQ DASHBOARD")
    print("="*70)
    print(f"Dashboard URL: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    print(f"API Server: http://localhost:8000 (if running)")
    print("="*70)
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()