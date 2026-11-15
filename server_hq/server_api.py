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
import zlib
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import API_CONFIG
from server_hq.central_aggregator import (
    ALL_AGGREGATION_METHODS,
    MAX_CLIPPING_NORM,
    MIN_CLIPPING_NORM,
    MIN_NOISE_MULTIPLIER,
    SUPPORTED_FL_METHODS,
    UpdateValidationError,
    compare_aggregation_methods,
    extract_state_dict_from_artifact,
    run_federated_round,
    validate_client_update_payload,
)
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


def _extract_client_ip(request: Optional[Request]) -> Optional[str]:
    if request is None:
        return None
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


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
        "open_round": registry.get_open_round(),
        "endpoints": {
            "list_versions": "/api/v1/model/versions",
            "download_latest": "/api/v1/model/download/latest",
            "download_version": "/api/v1/model/download/{version}",
            "round_info": "/api/v1/round",
            "upload_update": "/api/v1/model/upload",
            "list_updates": "/api/v1/updates",
            "list_clients": "/api/v1/clients",
            "aggregate": "/api/v1/model/aggregate",
            "compare_methods": "/api/v1/model/aggregate/compare",
        },
    }


@app.get("/api/v1/round")
def get_open_round_info():
    """Return active base model version, open FL round, and DP upload policy."""
    return {
        "status": "success",
        "base_version": registry.latest_version(),
        "open_round": registry.get_open_round(),
        "supported_methods": list(SUPPORTED_FL_METHODS),
        "dp_policy": {
            "require_clipped_and_noised": True,
            "min_clipping_norm": MIN_CLIPPING_NORM,
            "max_clipping_norm": MAX_CLIPPING_NORM,
            "min_noise_multiplier": MIN_NOISE_MULTIPLIER,
        },
    }


@app.get("/api/v1/model/versions")
def list_model_versions():
    versions = registry.list_versions()
    return {
        "status": "success",
        "latest": registry.latest_version(),
        "open_round": registry.get_open_round(),
        "count": len(versions),
        "versions": [_public_version(v) for v in versions],
    }


@app.delete("/api/v1/models/{version}")
def delete_model_version_api(version: str, x_admin_key: Optional[str] = Header(None)):
    """Delete a published model version from HQ registry and disk."""
    if x_admin_key is not None and x_admin_key.strip() != ADMIN_API_KEY.strip():
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Admin Secret Key for HQ Model API.")
    try:
        res = registry.delete_version(version)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err)) from val_err
    if res is None:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")
    return {
        "status": "success",
        "message": f"Deleted model version v{res['deleted_version']}.",
        **res,
    }


@app.get("/api/v1/model/download")
@app.get("/api/v1/model/download/{version}")
def download_global_model(
    request: Request,
    version: str = "latest",
    client_id: Optional[str] = None,
    x_client_id: Optional[str] = Header(None),
    x_admin_key: Optional[str] = Header(None),
):
    """Return compressed base64 weights for the requested (or latest) version. Requires Admin Auth if provided."""
    if x_admin_key is not None and x_admin_key.strip() != ADMIN_API_KEY.strip():
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Admin Secret Key for HQ Model API.")

    record = registry.get_version(version)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")

    weights_path = Path(record["weights_path"])
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Weights file missing on disk.")

    eff_client_id = (client_id or x_client_id or "").strip()
    if eff_client_id:
        registry.record_client_download(eff_client_id, record["version"], _extract_client_ip(request))

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
        "open_round": registry.get_open_round(),
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
def download_package_file(
    version: str,
    request: Request,
    client_id: Optional[str] = None,
    x_client_id: Optional[str] = Header(None),
):
    """Stream the edge package .pt for a version (falls back to weights.pt if no separate package exists)."""
    record = registry.get_version(version)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")
    pkg = record.get("package_path")
    target = pkg if (pkg and Path(pkg).exists()) else record.get("weights_path")
    if not target or not Path(target).exists():
        raise HTTPException(status_code=404, detail="Package or weights file missing.")
    eff_client_id = (client_id or x_client_id or "").strip()
    if eff_client_id:
        registry.record_client_download(eff_client_id, record["version"], _extract_client_ip(request))
    return FileResponse(
        path=str(target),
        filename=f"securefed_package_v{record['version']}.pt",
        media_type="application/octet-stream",
    )


@app.get("/api/v1/model/file/{version}")
def download_weights_file(
    version: str,
    request: Request,
    client_id: Optional[str] = None,
    x_client_id: Optional[str] = Header(None),
):
    """Stream the raw weights.pt for a version."""
    record = registry.get_version(version)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found.")
    weights_path = Path(record["weights_path"])
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Weights file missing on disk.")
    eff_client_id = (client_id or x_client_id or "").strip()
    if eff_client_id:
        registry.record_client_download(eff_client_id, record["version"], _extract_client_ip(request))
    return FileResponse(
        path=str(weights_path),
        filename=f"securefed_weights_v{record['version']}.pt",
        media_type="application/octet-stream",
    )


@app.post("/api/v1/model/upload")
async def upload_client_update(
    request: Request,
    client_id: str = Form(...),
    samples: int = Form(0),
    base_version: str = Form("latest"),
    round: int = Form(0),
    file: UploadFile = File(...),
):
    """Validate and accept ONLY clipped/noised DP weight deltas (.pt) for the open FL round."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty update file.")

        stored_bytes = content
        # Support optional zlib compression of a .pt delta archive (never raw Python pickle objects)
        if not content[:2] == b"PK":
            try:
                decompressed = zlib.decompress(content)
                if decompressed[:2] == b"PK":
                    stored_bytes = decompressed
            except Exception:
                pass

        current_latest = registry.latest_version()
        if not current_latest:
            raise HTTPException(status_code=400, detail="No global model version exists on HQ yet.")
        latest_record = registry.get_version(current_latest)
        _, base_state = extract_state_dict_from_artifact(Path(latest_record["weights_path"]))
        open_round = registry.get_open_round()

        validated = validate_client_update_payload(
            raw_bytes=stored_bytes,
            filename=file.filename or "",
            client_id=client_id,
            samples=samples,
            base_version=base_version,
            round_idx=round,
            expected_base_version=current_latest,
            expected_round=open_round,
            base_state_dict=base_state,
        )

        record = registry.save_update(
            client_id=validated["client_id"],
            weights_bytes=stored_bytes,
            samples=validated["samples"],
            base_version=validated["base_version"],
            round_idx=validated["round"],
            clipping_norm=validated["clipping_norm"],
            noise_multiplier=validated["noise_multiplier"],
            local_steps=validated["local_steps"],
            ip_address=_extract_client_ip(request),
        )
        return {
            "status": "success",
            "message": f"Validated DP update received from {validated['client_id']} (round {validated['round']}).",
            "update_id": record["id"],
            "client_id": record["client_id"],
            "base_version": record["base_version"],
            "round": record["round"],
            "samples": record["samples"],
            "clipping_norm": validated["clipping_norm"],
            "noise_multiplier": validated["noise_multiplier"],
            "local_steps": validated["local_steps"],
            "received_at": record["received_at"],
        }
    except UpdateValidationError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err)) from val_err
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
        "latest_version": registry.latest_version(),
        "open_round": registry.get_open_round(),
        "aggregated_folder": str(registry.aggregated_updates_dir),
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
            for u in updates
        ],
    }


@app.get("/api/v1/clients")
def list_clients():
    clients = registry.list_clients()
    return {"status": "success", "count": len(clients), "clients": clients}


@app.post("/api/v1/clients/register")
def register_client(
    request: Request,
    client_id: str = Form(...),
    display_name: Optional[str] = Form(None),
    model_version: Optional[str] = Form(None),
):
    record = registry.register_client(
        client_id=client_id,
        display_name=display_name,
        ip_address=_extract_client_ip(request),
        model_version=model_version,
    )
    return {"status": "success", "client": record}


async def _parse_agg_request(request: Request) -> dict:
    try:
        data = await request.json()
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    try:
        form = await request.form()
        return dict(form)
    except Exception:
        return {}


@app.post("/api/v1/model/aggregate")
async def trigger_aggregation(request: Request):
    """Aggregate validated pending DP deltas using Auto-Select Best or a specified FL/Byzantine algorithm."""
    body = await _parse_agg_request(request)
    try:
        result = run_federated_round(
            registry=registry,
            base_version=str(body.get("base_version", "latest")),
            method=str(body.get("method", "auto_best")),
            notes=str(body.get("notes", "")),
            min_updates=int(body.get("min_updates", 1)),
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err)) from val_err
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Aggregation error: {exc}") from exc


@app.post("/api/v1/model/aggregate/compare")
async def compare_aggregation_methods_endpoint(request: Request):
    """Evaluate all 9 FL and Byzantine methods on the same pending deltas."""
    body = await _parse_agg_request(request)
    try:
        methods = body.get("methods")
        if isinstance(methods, str):
            methods = [m.strip() for m in methods.split(",") if m.strip()]
        return compare_aggregation_methods(
            registry=registry,
            methods=methods or list(ALL_AGGREGATION_METHODS),
            base_version=str(body.get("base_version", "latest")),
            min_updates=int(body.get("min_updates", 1)),
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err)) from val_err
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison error: {exc}") from exc



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
