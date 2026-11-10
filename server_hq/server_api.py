"""SecureFed-NIDS HQ API — versioned global model serving and update intake.

Clients never send raw traffic. They download a numbered ensemble and
upload DP-noised weight files only.

Run:
    python -m server_hq.server_api
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import base64
import os
import pickle
import zlib
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import API_CONFIG
from server_hq.registry import ModelRegistry

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "securefed-admin-token-2026")

app = FastAPI(
    title="SecureFed-NIDS Central Server HQ",
    description="Versioned global ensemble serving. Clients download weights and upload updates only.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry()


def _public_version(row: dict) -> dict:
    return {
        "version": row["version"],
        "model_name": row["model_name"],
        "architecture": row["architecture"],
        "fl_method": row["fl_method"],
        "input_dim": row["input_dim"],
        "num_classes": row["num_classes"],
        "accuracy": row["accuracy"],
        "notes": row["notes"],
        "created_at": row["created_at"],
        "is_latest": bool(row["is_latest"]),
        "download_url": f"/api/v1/model/download/{row['version']}",
        "package_url": f"/api/v1/model/package/{row['version']}",
    }


@app.get("/")
def read_root():
    latest = registry.latest_version()
    return {
        "message": "SecureFed-NIDS Central Server is Running",
        "latest_version": latest,
        "endpoints": {
            "list_versions": "/api/v1/model/versions",
            "download_latest": "/api/v1/model/download/latest",
            "download_version": "/api/v1/model/download/{version}",
            "upload_update": "/api/v1/model/upload",
            "list_updates": "/api/v1/updates",
            "list_clients": "/api/v1/clients",
        },
    }


@app.get("/api/v1/model/versions")
def list_model_versions():
    versions = registry.list_versions()
    return {
        "status": "success",
        "latest": registry.latest_version(),
        "count": len(versions),
        "versions": [_public_version(v) for v in versions],
    }


@app.get("/api/v1/model/download")
@app.get("/api/v1/model/download/{version}")
def download_global_model(version: str = "latest", x_admin_key: Optional[str] = Header(None)):
    """Return compressed base64 weights for the requested (or latest) version. Requires Admin Auth if provided."""
    if x_admin_key is not None and x_admin_key.strip() != ADMIN_API_KEY.strip():
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Admin Secret Key for HQ Model API.")

    record = registry.get_version(version)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")

    weights_path = Path(record["weights_path"])
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Weights file missing on disk.")

    raw = weights_path.read_bytes()
    compressed = zlib.compress(raw)
    b64_encoded = base64.b64encode(compressed).decode("utf-8")

    meta_path = registry.version_dir(record["version"]) / "meta.json"
    metadata = {}
    if meta_path.exists():
        try:
            import json
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "status": "success",
        "version": record["version"],
        "model_name": record["model_name"],
        "architecture": record["architecture"],
        "fl_method": record["fl_method"],
        "input_dim": record["input_dim"],
        "num_classes": record["num_classes"],
        "feature_names": metadata.get("feature_names", []),
        "created_at": record["created_at"],
        "encoding": "zlib+base64",
        "weights_b64": b64_encoded,
    }


@app.get("/api/v1/model/metadata/{version}")
def get_model_metadata(version: str):
    """Return safe model schema/metadata for an edge client; never raw traffic."""
    record = registry.get_version(version)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")
    meta_path = registry.version_dir(record["version"]) / "meta.json"
    metadata = {}
    if meta_path.exists():
        try:
            import json
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not read model metadata: {exc}") from exc
    return {
        "status": "success",
        "version": record["version"],
        "model_name": record["model_name"],
        "architecture": record["architecture"],
        "input_dim": record["input_dim"],
        "num_classes": record["num_classes"],
        "feature_names": metadata.get("feature_names", []),
        "scaler_mean": metadata.get("scaler_mean"),
        "scaler_scale": metadata.get("scaler_scale"),
        "label_schema": {"benign": 0, "attack": 1},
    }


@app.get("/api/v1/model/package/{version}")
def download_package_file(version: str):
    """Stream the edge package .pt for a version (falls back to weights.pt if no separate package exists)."""
    record = registry.get_version(version)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")
    pkg = record.get("package_path")
    target = pkg if (pkg and Path(pkg).exists()) else record.get("weights_path")
    if not target or not Path(target).exists():
        raise HTTPException(status_code=404, detail="Package or weights file missing.")
    return FileResponse(
        path=str(target),
        filename=f"securefed_package_v{record['version']}.pt",
        media_type="application/octet-stream",
    )


@app.get("/api/v1/model/file/{version}")
def download_weights_file(version: str):
    """Stream the raw weights.pt for a version."""
    record = registry.get_version(version)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")
    weights_path = Path(record["weights_path"])
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Weights file missing on disk.")
    return FileResponse(
        path=str(weights_path),
        filename=f"securefed_weights_v{record['version']}.pt",
        media_type="application/octet-stream",
    )


@app.post("/api/v1/model/upload")
async def upload_client_update(
    client_id: str = Form(...),
    samples: int = Form(0),
    base_version: str = Form("latest"),
    round: int = Form(0),
    file: UploadFile = File(...),
):
    """Accept a compressed (or raw) client weight update. Never stores raw traffic."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty update file.")

        # Accept zlib-compressed pickle (legacy client) or raw .pt bytes.
        stored_bytes = content
        try:
            decompressed = zlib.decompress(content)
            pickle.loads(decompressed)  # validate it really is a pickle payload
            stored_bytes = decompressed
        except Exception:
            stored_bytes = content

        record = registry.save_update(
            client_id=client_id,
            weights_bytes=stored_bytes,
            samples=samples,
            base_version=base_version,
            round_idx=round,
        )
        return {
            "status": "success",
            "message": f"Update received from {client_id}.",
            "update_id": record["id"],
            "client_id": record["client_id"],
            "base_version": record["base_version"],
            "round": record["round"],
            "samples": record["samples"],
            "received_at": record["received_at"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/updates")
def list_updates(client_id: Optional[str] = None, version: Optional[str] = None):
    updates = registry.list_updates(client_id=client_id, version=version)
    return {
        "status": "success",
        "count": len(updates),
        "updates": [
            {
                "id": u["id"],
                "client_id": u["client_id"],
                "base_version": u["base_version"],
                "round": u["round"],
                "samples": u["samples"],
                "received_at": u["received_at"],
                "status": u["status"],
            }
            for u in updates
        ],
    }


@app.get("/api/v1/clients")
def list_clients():
    clients = registry.list_clients()
    return {"status": "success", "count": len(clients), "clients": clients}


@app.post("/api/v1/clients/register")
def register_client(client_id: str = Form(...), display_name: Optional[str] = Form(None)):
    record = registry.register_client(client_id, display_name)
    return {"status": "success", "client": record}


@app.post("/api/v1/model/aggregate")
def trigger_aggregation(
    base_version: str = Form("latest"),
    method: str = Form("fedavg"),
    notes: str = Form(""),
    min_updates: int = Form(1),
):
    """Aggregate pending client updates into a new global model version using FedAvg, Trimmed Mean, Median, or Krum."""
    from server_hq.central_aggregator import run_federated_round
    try:
        result = run_federated_round(
            registry=registry,
            base_version=base_version,
            method=method,
            notes=notes,
            min_updates=min_updates,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err)) from val_err
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Aggregation error: {exc}") from exc



def init_global_model(initial_weights: dict, version: str = "1.0.0", notes: str = "in-memory bootstrap"):
    """Back-compat helper. Prefer `python -m server_hq.publish` for persisted versions."""
    import torch
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "bootstrap.pt"
    torch.save(initial_weights, tmp)
    registry.publish(weights_src=tmp, version=version, metadata={"model_name": "SecureFed-Global-Ensemble"}, notes=notes)
    print(f"[HQ SERVER] Bootstrapped version {version} from in-memory weights.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_hq.server_api:app",
        host=API_CONFIG.get("host", "0.0.0.0"),
        port=API_CONFIG.get("port", 8000),
        reload=API_CONFIG.get("reload", False),
    )
