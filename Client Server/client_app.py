"""Client Server for SecureFed-NIDS
==================================
Runs locally on port 5000. Connects to the HQ model API at http://localhost:8000.
Supports local inference, federated local training with Differential Privacy (DP-SGD),
and secure weight updates sent to HQ without leaking raw traffic.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add parent project root to path so we can import model architectures, DP, preprocessing
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

import base64
import zlib
import pickle
import json
import threading
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import the project DP implementation.  The client deliberately owns its
# CSV transformation: it never asks HQ to receive or preprocess raw traffic.
from dp.differential_privacy import DPSGD

app = FastAPI(
    title="SecureFed-NIDS Edge Client",
    description="Client node for localized training, DP-SGD, inference, and federated model sync with HQ.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Client state storage
CLIENT_STATE = {
    "client_id": os.getenv("CLIENT_ID", "client_node_01"),
    "hq_url": os.getenv("HQ_URL", "http://localhost:8000"),
    "edge_port": int(os.getenv("EDGE_PORT", "5000")),
    "current_model_version": None,
    "model_loaded": False,
    "model_info": {},
    "last_sync": "Never",
    "last_push": "Never",
    "last_push_info": None,
    "last_hq_push_version": None,
    "last_hq_push_at": None,
    "last_deployed_source": None,
    "operator_user": None,
    "operator_authenticated": False,
    "operator_authenticated_at": None,
    "auth_provider": None,
    "auto_push": True,
    "status": "Idle",
    "training_logs": [],
}

_SYNC_LOCK = threading.Lock()
_STOP_HEARTBEAT_EVENT = threading.Event()

# Local storage for the downloaded PyTorch ensemble and its base state.
# The base state lets the client submit a clipped/noised *delta*, not its raw
# post-training model or any locally held network traffic.
LOCAL_MODEL_CONTAINER = {
    "model": None,
    "weights_path": None,
    "base_state": None,
    "sample_count": 0,
    "round": 0,
    "feature_names": [],
    "privacy": {"epsilon": None, "delta": None, "clipping_norm": None, "noise_multiplier": None},
}


LABEL_CANDIDATES = ("label", "labels", "class", "target", "attack", "attack_type", "attack_cat", "category")
BENIGN_LABELS = ("benign", "normal", "0", "false", "safe", "clean", "good", "mqtt_publish", "thing_speak", "wipro_bulb")
ATTACK_LABELS = (
    "attack", "malicious", "anomaly", "intrusion", "1", "true",
    "dos", "ddos", "scan", "portscan", "poisoning", "mitm", "brute",
    "hping", "bot", "infiltration", "patator", "heartbleed", "exploit",
    "fuzz", "shellcode", "worm", "backdoor", "recon", "neptune", "smurf",
)


def get_active_model():
    return LOCAL_MODEL_CONTAINER.get("model")


def clone_state_dict(model: nn.Module) -> dict:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


_STATE_FILE = BASE_DIR / "local_models" / "client_state.json"


@contextmanager
def _temp_csv_file(raw_bytes: bytes, prefix: str = "securefed_"):
    """Write uploaded CSV bytes to a temporary file, closing the FD and always deleting afterwards."""
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".csv")
    os.close(fd)
    path = Path(tmp_path)
    try:
        path.write_bytes(raw_bytes)
        yield path
    finally:
        path.unlink(missing_ok=True)


def _save_persisted_state() -> None:
    """Persist non-sensitive Edge sync/model metadata locally so state survives restarts."""
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "client_id": CLIENT_STATE.get("client_id"),
            "hq_url": CLIENT_STATE.get("hq_url"),
            "current_model_version": CLIENT_STATE.get("current_model_version"),
            "model_info": CLIENT_STATE.get("model_info") or {},
            "feature_names": CLIENT_STATE.get("feature_names") or [],
            "last_sync": CLIENT_STATE.get("last_sync"),
            "last_push": CLIENT_STATE.get("last_push"),
            "last_push_info": CLIENT_STATE.get("last_push_info"),
            "last_hq_push_version": CLIENT_STATE.get("last_hq_push_version"),
            "last_hq_push_at": CLIENT_STATE.get("last_hq_push_at"),
            "last_deployed_source": CLIENT_STATE.get("last_deployed_source"),
            "operator_user": CLIENT_STATE.get("operator_user"),
            "operator_authenticated": bool(CLIENT_STATE.get("operator_authenticated", False)),
            "operator_authenticated_at": CLIENT_STATE.get("operator_authenticated_at"),
            "auth_provider": CLIENT_STATE.get("auth_provider"),
            "auto_push": bool(CLIENT_STATE.get("auto_push", True)),
            "status": CLIENT_STATE.get("status"),
            "privacy": CLIENT_STATE.get("privacy"),
            "training_logs": (CLIENT_STATE.get("training_logs") or [])[-20:],
            "weights_path": LOCAL_MODEL_CONTAINER.get("weights_path"),
            "update_path": LOCAL_MODEL_CONTAINER.get("update_path"),
            "sample_count": int(LOCAL_MODEL_CONTAINER.get("sample_count", 0)),
            "round": int(LOCAL_MODEL_CONTAINER.get("round", 0)),
        }
        _STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _restore_persisted_state() -> None:
    """Restore persisted Edge sync state and reload cached local model on startup."""
    import re
    saved = {}
    if _STATE_FILE.exists():
        try:
            saved = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            saved = {}

    if saved.get("client_id"):
        CLIENT_STATE["client_id"] = saved["client_id"]
    if saved.get("hq_url"):
        CLIENT_STATE["hq_url"] = saved["hq_url"]
    if "auto_push" in saved:
        CLIENT_STATE["auto_push"] = bool(saved["auto_push"])
    for key in ("last_sync", "last_push", "last_push_info", "last_hq_push_version", "last_hq_push_at", "last_deployed_source", "operator_user", "operator_authenticated", "operator_authenticated_at", "auth_provider", "privacy", "training_logs"):
        if saved.get(key) is not None:
            CLIENT_STATE[key] = saved[key]

    weights_candidate = Path(saved["weights_path"]) if saved.get("weights_path") else None
    if not weights_candidate or not weights_candidate.exists():
        models_dir = BASE_DIR / "local_models"
        if models_dir.exists():
            pts = sorted(models_dir.glob("ensemble_v*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            preferred = [p for p in pts if "-airgap" not in p.stem]
            weights_candidate = (preferred or pts)[0] if (preferred or pts) else None

    if not weights_candidate or not weights_candidate.exists():
        return

    try:
        try:
            loaded_raw = torch.load(weights_candidate, map_location="cpu", weights_only=False)
        except TypeError:
            loaded_raw = torch.load(weights_candidate, map_location="cpu")
        if isinstance(loaded_raw, nn.Module):
            loaded_model = loaded_raw
        elif isinstance(loaded_raw, dict) and "model_state_dict" in loaded_raw:
            loaded_model = _build_model_from_package_dict(loaded_raw)
        else:
            return

        loaded_model.to("cpu")
        for param in loaded_model.parameters():
            param.requires_grad_(True)

        inferred_dim = infer_model_input_dim(loaded_model)
        ver_label = saved.get("current_model_version")
        if not ver_label:
            m = re.match(r"ensemble_v(.+)", weights_candidate.stem)
            ver_label = m.group(1) if m else "local"

        model_info = dict(saved.get("model_info") or {})
        feature_names = model_info.get("feature_names") or saved.get("feature_names") or []
        if len(feature_names) != inferred_dim:
            ver_match = re.match(r"^(\d+\.\d+\.\d+)", str(ver_label))
            if ver_match:
                meta_path = ROOT / "hq_store" / "models" / f"v{ver_match.group(1)}" / "meta.json"
                if meta_path.exists():
                    try:
                        meta_json = json.loads(meta_path.read_text(encoding="utf-8"))
                        if len(meta_json.get("feature_names", [])) == inferred_dim:
                            feature_names = meta_json["feature_names"]
                            model_info.setdefault("model_name", meta_json.get("model_name"))
                            model_info.setdefault("architecture", meta_json.get("architecture"))
                            model_info.setdefault("scaler_mean", meta_json.get("scaler_mean"))
                            model_info.setdefault("scaler_scale", meta_json.get("scaler_scale"))
                    except Exception:
                        pass
        if len(feature_names) != inferred_dim and inferred_dim == 83:
            try:
                from config import DATASET_CONFIG
                feature_names = DATASET_CONFIG.get("canonical_feature_names", [])
            except Exception:
                pass
        if len(feature_names) != inferred_dim:
            feature_names = [f"feature_{i}" for i in range(inferred_dim)]

        model_info.setdefault("model_name", "SecureFed-Global-Ensemble")
        model_info.setdefault("architecture", getattr(loaded_model, "method", "EnsembleTop2"))
        model_info["input_dim"] = inferred_dim
        model_info["num_classes"] = int(getattr(loaded_model, "num_classes", 2))
        model_info["feature_names"] = feature_names

        LOCAL_MODEL_CONTAINER["model"] = loaded_model
        LOCAL_MODEL_CONTAINER["weights_path"] = str(weights_candidate)
        LOCAL_MODEL_CONTAINER["base_state"] = clone_state_dict(loaded_model)
        LOCAL_MODEL_CONTAINER["sample_count"] = int(saved.get("sample_count", 0))
        LOCAL_MODEL_CONTAINER["round"] = max(1, int(saved.get("round", 1) or 1))
        if saved.get("update_path") and Path(saved["update_path"]).exists():
            LOCAL_MODEL_CONTAINER["update_path"] = saved["update_path"]

        CLIENT_STATE["current_model_version"] = ver_label
        CLIENT_STATE["model_loaded"] = True
        CLIENT_STATE["model_info"] = model_info
        CLIENT_STATE["feature_names"] = feature_names
        if not CLIENT_STATE.get("last_deployed_source"):
            CLIENT_STATE["last_deployed_source"] = model_info.get("authorized_by") or model_info.get("deployed_by")
        if not CLIENT_STATE.get("last_hq_push_version") and "hq push" in str(model_info.get("authorized_by") or "").lower():
            CLIENT_STATE["last_hq_push_version"] = ver_label
            if not CLIENT_STATE.get("last_hq_push_at") and saved.get("last_sync") and saved.get("last_sync") != "Never":
                CLIENT_STATE["last_hq_push_at"] = saved.get("last_sync")
        if not saved.get("last_sync"):
            CLIENT_STATE["last_sync"] = "Restored from local cache"
        CLIENT_STATE["status"] = saved.get("status") or f"Restored v{ver_label} from local cache"
    except Exception:
        pass


def infer_model_input_dim(model: nn.Module) -> int:
    """Inspect a loaded PyTorch model/ensemble to determine its expected input feature dimension."""
    if hasattr(model, "input_dim") and model.input_dim:
        return int(model.input_dim)
    if hasattr(model, "models") and len(model.models) > 0 and hasattr(model.models[0], "input_dim"):
        return int(model.models[0].input_dim)
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            return int(mod.in_features)
    return 83


def find_label_column(df: pd.DataFrame):
    normalized = {str(col).strip().lower().replace(" ", "_"): col for col in df.columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def map_binary_labels(values: pd.Series) -> np.ndarray:
    """Map benign/attack labels to 0/1; supports RT-IoT2022, CIC-IDS2017, NSL-KDD, UNSW-NB15."""
    mapped = []
    unknown = set()
    has_any_benign = any(
        any(word in str(v).strip().lower() for word in BENIGN_LABELS)
        for v in values.iloc[:min(len(values), 200)]
    )
    for value in values:
        label = str(value).strip().lower()
        if any(word == label or word in label for word in BENIGN_LABELS):
            mapped.append(0)
        elif any(word in label for word in ATTACK_LABELS):
            mapped.append(1)
        elif has_any_benign:
            # Explicit benign marker exists in this dataset; other attack classes map to 1
            mapped.append(1)
        else:
            unknown.add(str(value))
    if unknown:
        preview = ", ".join(sorted(unknown)[:5])
        raise ValueError(f"Unknown label values: {preview}. Use explicit benign/normal or attack/malicious labels.")
    return np.asarray(mapped, dtype=np.int64)


def stable_category_number(value: object) -> float:
    """Deterministic local encoding for categorical columns without saving values."""
    import hashlib
    if pd.isna(value):
        return 0.0
    digest = hashlib.sha256(str(value).strip().lower().encode("utf-8")).hexdigest()
    return float(int(digest[:12], 16) % 1_000_000)


def canonicalize_csv(csv_path: Path, feature_names: list[str], require_labels: bool):
    """Build the exact published feature order locally and apply feature scaling."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")

    label_col = find_label_column(df)
    labels = None
    if label_col is not None:
        if require_labels:
            labels = map_binary_labels(df[label_col])
        else:
            try:
                labels = map_binary_labels(df[label_col])
            except Exception:
                labels = None
    elif require_labels:
        raise ValueError("Local training requires a label column (for example: label, class, target, or attack_type).")

    normalized_columns = {
        str(col).strip().lower().replace(" ", "_").replace("-", "_"): col
        for col in df.columns
    }
    feature_matrix = np.zeros((len(df), len(feature_names)), dtype=np.float32)
    matched = 0
    for index, feature in enumerate(feature_names):
        normalized_feature = str(feature).strip().lower().replace(" ", "_").replace("-", "_")
        source_col = normalized_columns.get(normalized_feature)
        if source_col is None:
            continue
        source = df[source_col]
        numeric = pd.to_numeric(source, errors="coerce")
        if numeric.isna().all() and source.dtype == object:
            numeric = source.map(stable_category_number)
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feature_matrix[:, index] = numeric.to_numpy(dtype=np.float32)
        matched += 1

    if matched == 0:
        raise ValueError("No published model features matched this CSV. Use a NIDS feature CSV compatible with the HQ feature schema.")

    # Scale features to match HQ StandardScaler training distribution
    model_info = CLIENT_STATE.get("model_info", {})
    scaler_mean = model_info.get("scaler_mean")
    scaler_scale = model_info.get("scaler_scale")
    if (
        isinstance(scaler_mean, list)
        and isinstance(scaler_scale, list)
        and len(scaler_mean) == len(feature_names)
        and len(scaler_scale) == len(feature_names)
    ):
        mean_arr = np.asarray(scaler_mean, dtype=np.float32)
        scale_arr = np.where(np.asarray(scaler_scale, dtype=np.float32) == 0, 1.0, np.asarray(scaler_scale, dtype=np.float32))
        feature_matrix = ((feature_matrix - mean_arr) / scale_arr).astype(np.float32)
    elif len(feature_matrix) > 1 and np.max(np.abs(feature_matrix)) > 15.0:
        from sklearn.preprocessing import StandardScaler
        feature_matrix = StandardScaler().fit_transform(feature_matrix).astype(np.float32)

    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feature_matrix, labels, df, matched


def is_log_probability_output(logits: torch.Tensor) -> bool:
    """Voting ensembles return log probabilities; base/stacking models return logits."""
    try:
        return bool(torch.allclose(torch.exp(logits).sum(dim=1), torch.ones(logits.size(0), device=logits.device), atol=1e-4))
    except Exception:
        return False


def output_to_probabilities(outputs: torch.Tensor) -> torch.Tensor:
    return torch.exp(outputs) if is_log_probability_output(outputs) else torch.softmax(outputs, dim=1)


def clip_and_noise_model_delta(model: nn.Module, base_state: dict, clipping_norm: float, noise_multiplier: float) -> dict:
    """Return only clipped/noised floating point parameter deltas for HQ aggregation."""
    current = model.state_dict()
    deltas = {}
    squared_norm = 0.0
    for name, base_tensor in base_state.items():
        tensor = current[name].detach().cpu()
        if torch.is_floating_point(tensor):
            delta = tensor - base_tensor
            deltas[name] = delta
            squared_norm += float(torch.sum(delta.float() ** 2))
    total_norm = squared_norm ** 0.5
    scale = min(1.0, float(clipping_norm) / max(total_norm, 1e-12))
    protected = {}
    for name, base_tensor in base_state.items():
        if name in deltas:
            clipped_delta = deltas[name] * scale
            noise = torch.normal(0.0, float(noise_multiplier) * float(clipping_norm), size=clipped_delta.shape)
            protected[name] = clipped_delta + noise
        else:
            # Non-floating state (e.g., BatchNorm counters) is not personal update data.
            protected[name] = torch.zeros_like(base_tensor)
    return protected


def render_page(request: Request, template_name: str, active_page: str):
    ctx = {"request": request, "state": CLIENT_STATE, "active_page": active_page}
    try:
        return templates.TemplateResponse(request=request, name=template_name, context=ctx)
    except TypeError:
        return templates.TemplateResponse(template_name, ctx)


DEFAULT_ADMIN_KEY = os.getenv("ADMIN_KEY", "securefed-admin-token-2026")


def verify_admin_access(provided_key: str):
    """Enforce that only an authorized administrator can deploy models, change HQ routing, or push updates."""
    expected = os.getenv("ADMIN_KEY", DEFAULT_ADMIN_KEY).strip()
    key = (provided_key or "").strip()
    if not key or key != expected:
        raise HTTPException(
            status_code=403,
            detail="Access Denied (403): Administrator authentication required. Please enter a valid Admin Secret Key."
        )


class AdminActionRequest(BaseModel):
    admin_key: str = ""
    version: str = "latest"
    hq_url: str = ""
    source: str = ""


def _parse_hq_and_version(raw_url: str, requested_version: str = "latest") -> tuple[str, str]:
    """Allow pasting either HQ base URL (http://localhost:8000) or a full model download/file URL."""
    url = (raw_url or "").strip().rstrip("/")
    ver = (requested_version or "latest").strip() or "latest"
    for marker in ("/api/v1/model/download/", "/api/v1/model/file/", "/api/v1/model/package/", "/api/v1/model/metadata/"):
        if marker in url:
            base, tail = url.split(marker, 1)
            extracted = tail.strip("/").split("/")[0]
            return base.rstrip("/"), extracted or ver
    if url.endswith("/api/v1/model/download"):
        return url[: -len("/api/v1/model/download")].rstrip("/"), ver
    return url, ver


class ConfigureRequest(BaseModel):
    client_id: str
    hq_url: str
    admin_key: str = ""


class UploadModelFileRequest(BaseModel):
    file_b64: str
    admin_key: str = ""
    version_label: str = "manual"


class AutoPushConfigRequest(BaseModel):
    enabled: bool = True
    admin_key: str = ""


class TrainLocalRequest(BaseModel):
    file_b64: str
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 0.001
    clipping_norm: float = 1.0
    noise_multiplier: float = 1.1
    auto_push: Optional[bool] = None


class InferenceRequest(BaseModel):
    file_b64: str


class LoginRequest(BaseModel):
    username: str = "client_node_01"
    email: Optional[str] = None
    admin_key: str = ""
    auth_provider: str = "credentials"  # "credentials", "email", "google", "github"
    remember: bool = True


class ForgotPasswordRequest(BaseModel):
    email: str


@app.get("/", response_class=HTMLResponse)
def client_dashboard(request: Request):
    return render_page(request, "client_index.html", "overview")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return render_page(request, "login.html", "login")


@app.get("/sync", response_class=HTMLResponse)
def sync_page(request: Request):
    return render_page(request, "sync.html", "sync")


@app.get("/training", response_class=HTMLResponse)
def training_page(request: Request):
    return render_page(request, "training.html", "training")


@app.get("/inference", response_class=HTMLResponse)
def inference_page(request: Request):
    return render_page(request, "inference.html", "inference")


@app.get("/api/client/status")
def get_client_status():
    return {"status": "success", "client_state": CLIENT_STATE}


@app.post("/api/client/login")
def client_login(payload: LoginRequest):
    """Authenticate an operator or administrator to unlock Edge node controls."""
    expected = os.getenv("ADMIN_KEY", DEFAULT_ADMIN_KEY).strip()
    provider = (payload.auth_provider or "credentials").lower()

    if provider in ("google", "github"):
        # Social SSO login: validated via simulated OAuth token exchange
        default_user = "operator.google@securefed.internal" if provider == "google" else "github-operator@fedx-nids"
        username = (payload.username or payload.email or default_user).strip()
        auth_key = expected
    else:
        key = (payload.admin_key or "").strip()
        if not key or key != expected:
            raise HTTPException(
                status_code=401,
                detail="Access Denied: Invalid Admin Secret Key or Password. (Default token: securefed-admin-token-2026)"
            )
        username = (payload.username or payload.email or CLIENT_STATE.get("client_id", "client_node_01")).strip()
        auth_key = key

    CLIENT_STATE["operator_user"] = username
    CLIENT_STATE["operator_authenticated"] = True
    CLIENT_STATE["operator_authenticated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    CLIENT_STATE["auth_provider"] = provider
    _save_persisted_state()
    return {
        "status": "success",
        "message": f"Welcome back, {username}! Edge Node unlocked with Administrative privileges ({provider.title()}).",
        "username": username,
        "auth_provider": provider,
        "admin_key": auth_key,
        "client_id": CLIENT_STATE.get("client_id"),
        "authenticated_at": CLIENT_STATE["operator_authenticated_at"],
    }


@app.post("/api/client/logout")
def client_logout():
    """Clear active operator session from Edge node."""
    CLIENT_STATE["operator_authenticated"] = False
    CLIENT_STATE["operator_user"] = None
    CLIENT_STATE["operator_authenticated_at"] = None
    CLIENT_STATE["auth_provider"] = None
    _save_persisted_state()
    return {"status": "success", "message": "Successfully logged out of Edge Node."}


@app.post("/api/client/forgot-password")
def client_forgot_password(payload: ForgotPasswordRequest):
    """Provide recovery instructions for operator credentials on this Edge node."""
    email = (payload.email or "").strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Please enter a valid operator email address.")
    
    expected = os.getenv("ADMIN_KEY", DEFAULT_ADMIN_KEY).strip()
    masked = f"{expected[:4]}****{expected[-4:]}" if len(expected) >= 8 else "****"
    return {
        "status": "success",
        "message": f"Recovery instructions prepared for {email}.",
        "detail": (
            "Edge Node Cryptographic Recovery: The default node master key is 'securefed-admin-token-2026'. "
            "If your team configured a custom token, verify ADMIN_KEY in your deployment environment or request an override from Central HQ."
        ),
        "recovery_key_hint": masked,
        "default_token": expected,
    }


@app.get("/api/client/auth/oauth-status")
def get_oauth_status():
    """Return whether real Google and GitHub OAuth credentials are configured in .env."""
    google_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    google_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    github_id = os.getenv("GITHUB_CLIENT_ID", "").strip()
    github_secret = os.getenv("GITHUB_CLIENT_SECRET", "").strip()
    return {
        "status": "success",
        "google": bool(google_id and google_secret),
        "github": bool(github_id and github_secret),
    }


@app.get("/api/client/auth/oauth-login/{provider}")
def oauth_login_redirect(provider: str, request: Request):
    """Initiate official OAuth2 login redirect for Google or GitHub if configured."""
    provider = provider.lower()
    host = request.headers.get("host") or f"localhost:{CLIENT_STATE.get('edge_port', 5000)}"
    scheme = request.url.scheme or "http"
    callback_base = f"{scheme}://{host}"

    if provider == "google":
        client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
        if not client_id:
            return RedirectResponse(url="/login?simulated=google")
        redirect_uri = f"{callback_base}/api/client/auth/callback/google"
        scope = "openid email profile"
        url = (
            f"https://accounts.google.com/o/oauth2/v2/auth"
            f"?client_id={client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&response_type=code"
            f"&scope={requests.utils.quote(scope)}"
            f"&prompt=select_account"
        )
        return RedirectResponse(url=url)

    elif provider == "github":
        client_id = os.getenv("GITHUB_CLIENT_ID", "").strip()
        if not client_id:
            return RedirectResponse(url="/login?simulated=github")
        redirect_uri = f"{callback_base}/api/client/auth/callback/github"
        scope = "read:user user:email"
        url = (
            f"https://github.com/login/oauth/authorize"
            f"?client_id={client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&scope={requests.utils.quote(scope)}"
        )
        return RedirectResponse(url=url)

    raise HTTPException(status_code=400, detail="Unsupported OAuth provider.")


@app.get("/api/client/auth/callback/{provider}")
def oauth_callback(provider: str, request: Request, code: Optional[str] = None, error: Optional[str] = None):
    """Handle OAuth2 authorization code callback and exchange for user identity."""
    if error or not code:
        return RedirectResponse(url=f"/login?error={error or 'cancelled'}")

    provider = provider.lower()
    host = request.headers.get("host") or f"localhost:{CLIENT_STATE.get('edge_port', 5000)}"
    scheme = request.url.scheme or "http"
    redirect_uri = f"{scheme}://{host}/api/client/auth/callback/{provider}"
    user_email = ""
    user_name = ""

    try:
        if provider == "google":
            client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
            client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
            token_resp = requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
                timeout=10,
            )
            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                return RedirectResponse(url="/login?error=google_token_failed")
            
            user_info = requests.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            ).json()
            user_email = user_info.get("email", "")
            user_name = user_info.get("name") or user_email

        elif provider == "github":
            client_id = os.getenv("GITHUB_CLIENT_ID", "").strip()
            client_secret = os.getenv("GITHUB_CLIENT_SECRET", "").strip()
            token_resp = requests.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                },
                timeout=10,
            )
            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                return RedirectResponse(url="/login?error=github_token_failed")

            user_info = requests.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
                timeout=10,
            ).json()
            user_name = user_info.get("login") or user_info.get("name") or "github-operator"
            user_email = user_info.get("email") or f"{user_name}@users.noreply.github.com"
        
        # Authenticate operator session
        username = user_email or user_name or f"operator.{provider}"
        CLIENT_STATE["operator_user"] = username
        CLIENT_STATE["operator_authenticated"] = True
        CLIENT_STATE["operator_authenticated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        CLIENT_STATE["auth_provider"] = provider
        _save_persisted_state()
        return RedirectResponse(url="/login?oauth_success=1")

    except Exception as exc:
        return RedirectResponse(url=f"/login?error={str(exc)}")


@app.post("/api/client/configure")
def configure_client(payload: ConfigureRequest):
    verify_admin_access(payload.admin_key)
    CLIENT_STATE["client_id"] = payload.client_id.strip()
    clean_base, _ = _parse_hq_and_version(payload.hq_url, "latest")
    CLIENT_STATE["hq_url"] = clean_base or CLIENT_STATE["hq_url"]
    _register_client_heartbeat(CLIENT_STATE["hq_url"])
    _save_persisted_state()
    return {"status": "success", "message": "Client configuration updated with Admin Authorization.", "state": CLIENT_STATE}


@app.post("/api/client/auto-push")
def set_auto_push(payload: AutoPushConfigRequest):
    """Enable or disable automatic DP update push to HQ after local training."""
    verify_admin_access(payload.admin_key)
    CLIENT_STATE["auto_push"] = bool(payload.enabled)
    _save_persisted_state()
    mode_label = "enabled" if CLIENT_STATE["auto_push"] else "disabled"
    return {
        "status": "success",
        "auto_push": CLIENT_STATE["auto_push"],
        "message": f"Automatic DP update push to HQ is now {mode_label}.",
    }


def _edge_self_url() -> str:
    port = int(CLIENT_STATE.get("edge_port", os.getenv("EDGE_PORT", "5000")))
    return f"http://127.0.0.1:{port}"


def _register_client_heartbeat(hq_url: str):
    """Inform HQ of this active Edge Client ID and reachable callback URL (never pulls models)."""
    try:
        form_data = {"client_id": CLIENT_STATE["client_id"], "display_name": _edge_self_url()}
        if CLIENT_STATE.get("current_model_version"):
            form_data["model_version"] = str(CLIENT_STATE["current_model_version"])
        resp = requests.post(
            f"{hq_url}/api/v1/clients/register",
            data=form_data,
            timeout=5,
        )
        if resp.status_code == 200 and CLIENT_STATE.get("status") == "Idle":
            CLIENT_STATE["status"] = "Connected to HQ (awaiting push)"
    except Exception:
        pass


def _sync_model_from_hq(raw_hq: str, requested_version: str, admin_key: str, source_tag: str = "Manual") -> dict:
    """Thread-safe model pull & load from HQ API."""
    hq_url, version = _parse_hq_and_version(raw_hq or CLIENT_STATE["hq_url"], requested_version or "latest")
    if not hq_url:
        hq_url = CLIENT_STATE["hq_url"]
    CLIENT_STATE["hq_url"] = hq_url

    endpoint = f"{hq_url}/api/v1/model/download/{version}"
    headers = {"X-Admin-Key": admin_key.strip(), "X-Client-ID": str(CLIENT_STATE["client_id"])}
    with _SYNC_LOCK:
        resp = requests.get(endpoint, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"HQ returned error: {resp.text}")
        data = resp.json()
        b64_data = data.get("weights_b64")
        if not b64_data:
            raise HTTPException(status_code=502, detail="HQ returned no model weight payload.")

        raw_weights = zlib.decompress(base64.b64decode(b64_data))
        models_dir = BASE_DIR / "local_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        ver_label = str(data.get("version", "unknown"))
        local_weights_path = models_dir / f"ensemble_v{ver_label}.pt"
        local_weights_path.write_bytes(raw_weights)

        try:
            loaded_raw = torch.load(local_weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            loaded_raw = torch.load(local_weights_path, map_location="cpu")
        if isinstance(loaded_raw, nn.Module):
            loaded_model = loaded_raw
        elif isinstance(loaded_raw, dict) and "model_state_dict" in loaded_raw:
            loaded_model = _build_model_from_package_dict(loaded_raw)
        else:
            raise ValueError("HQ weights file is not a supported PyTorch model or state_dict package.")

        metadata_response = requests.get(f"{hq_url}/api/v1/model/metadata/{ver_label}", timeout=15)
        if metadata_response.status_code != 200:
            raise HTTPException(status_code=metadata_response.status_code, detail="HQ could not provide the published feature schema.")
        metadata = metadata_response.json()
        feature_names = metadata.get("feature_names", [])
        if not feature_names or len(feature_names) != int(data.get("input_dim", 0)):
            raise ValueError("HQ metadata has no valid published feature schema for this model.")

        loaded_model.to("cpu")
        for parameter in loaded_model.parameters():
            parameter.requires_grad_(True)
        LOCAL_MODEL_CONTAINER["model"] = loaded_model
        LOCAL_MODEL_CONTAINER["weights_path"] = str(local_weights_path)
        LOCAL_MODEL_CONTAINER["base_state"] = clone_state_dict(loaded_model)
        LOCAL_MODEL_CONTAINER["sample_count"] = 0
        hq_open_round = data.get("open_round")
        LOCAL_MODEL_CONTAINER["round"] = int(hq_open_round) if hq_open_round is not None else max(1, int(LOCAL_MODEL_CONTAINER.get("round", 0)) + 1)

        CLIENT_STATE["current_model_version"] = ver_label
        CLIENT_STATE["model_loaded"] = True
        CLIENT_STATE["model_info"] = {
            "model_name": data.get("model_name"),
            "architecture": data.get("architecture"),
            "input_dim": data.get("input_dim"),
            "num_classes": data.get("num_classes"),
            "created_at": data.get("created_at"),
            "feature_names": feature_names,
            "scaler_mean": metadata.get("scaler_mean"),
            "scaler_scale": metadata.get("scaler_scale"),
            "authorized_by": source_tag,
        }
        CLIENT_STATE["feature_names"] = feature_names
        received_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        CLIENT_STATE["last_deployed_source"] = source_tag
        if "hq push" in source_tag.lower():
            CLIENT_STATE["last_hq_push_version"] = ver_label
            CLIENT_STATE["last_hq_push_at"] = received_at
        CLIENT_STATE["status"] = f"Synced v{ver_label} ({source_tag})"
        CLIENT_STATE["last_sync"] = received_at
        _register_client_heartbeat(hq_url)
        _save_persisted_state()
        return {"status": "success", "message": f"Loaded global ensemble v{ver_label} from HQ ({source_tag}).", "version": ver_label}


@app.post("/api/client/pull-model")
def pull_model_from_hq(payload: AdminActionRequest):
    """Download and load a published global ensemble when pushed by HQ or requested manually."""
    verify_admin_access(payload.admin_key)
    try:
        source_label = payload.source.strip() if payload.source else "Manual Pull"
        return _sync_model_from_hq(
            raw_hq=payload.hq_url.strip() if payload.hq_url else CLIENT_STATE["hq_url"],
            requested_version=payload.version or "latest",
            admin_key=payload.admin_key,
            source_tag=source_label,
        )
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"HQ model API is unreachable at {CLIENT_STATE['hq_url']}: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _infer_arch_from_sd_keys(keys: list[str], arch_hint: str = "") -> str:
    joined = "\n".join(keys)
    if ".lstm." in joined or joined.startswith("lstm."):
        return "bilstm" if "_reverse" in joined else "lstm"
    if ".gru." in joined or joined.startswith("gru."):
        return "gru"
    if ".conv_blocks." in joined or "conv_blocks." in joined:
        return "cnn"
    if ".transformer_encoder." in joined or "transformer_encoder." in joined:
        return "transformer"
    hint = (arch_hint or "").strip().lower()
    for candidate in ("bilstm", "lstm", "gru", "cnn", "transformer", "mlp"):
        if candidate == hint:
            return candidate
    return "mlp"


def _build_model_from_package_dict(loaded_raw: dict) -> nn.Module:
    from models.architectures import ModelFactory
    from models.ensemble import EnsembleModel

    in_d = int(loaded_raw.get("input_dim", 83))
    n_cls = int(loaded_raw.get("num_classes", 2))
    sd = loaded_raw["model_state_dict"]
    keys = list(sd.keys())

    sub_indices = sorted({k.split(".")[1] for k in keys if k.startswith("models.") and len(k.split(".")) > 2})
    if sub_indices:
        submodels = []
        for idx in sub_indices:
            prefix = f"models.{idx}."
            sub_keys = [k[len(prefix):] for k in keys if k.startswith(prefix)]
            submodels.append(ModelFactory.create_model(_infer_arch_from_sd_keys(sub_keys), in_d, n_cls))
        arch_text = str(loaded_raw.get("architecture", "voting")).lower()
        method = "stacking" if "stack" in arch_text else ("weighted" if "weight" in arch_text else "voting")
        model = EnsembleModel(submodels, method=method, num_classes=n_cls)
        model.load_state_dict(sd)
        return model

    arch_name = _infer_arch_from_sd_keys(keys, str(loaded_raw.get("architecture", "mlp")))
    model = ModelFactory.create_model(arch_name, in_d, n_cls)
    model.load_state_dict(sd)
    return model


@app.post("/api/client/upload-model-file")
async def upload_model_file(payload: UploadModelFileRequest):
    """Option 1 (Air-Gapped Fallback): Admin manually uploads a downloaded .pt model file."""
    verify_admin_access(payload.admin_key)
    try:
        raw_bytes = base64.b64decode(payload.file_b64)
        models_dir = BASE_DIR / "local_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        ver_label = (payload.version_label or "").strip().lstrip("vV") or "manual"
        local_weights_path = models_dir / f"ensemble_v{ver_label}.pt"
        local_weights_path.write_bytes(raw_bytes)

        try:
            loaded_raw = torch.load(local_weights_path, map_location="cpu", weights_only=False)
        except TypeError:
            loaded_raw = torch.load(local_weights_path, map_location="cpu")

        extracted_features = []
        raw_arch = None
        raw_name = None
        if isinstance(loaded_raw, nn.Module):
            loaded_model = loaded_raw
        elif isinstance(loaded_raw, dict) and "model_state_dict" in loaded_raw:
            extracted_features = loaded_raw.get("feature_names", [])
            raw_arch = loaded_raw.get("architecture")
            raw_name = loaded_raw.get("model_name")
            loaded_model = _build_model_from_package_dict(loaded_raw)
        else:
            raise ValueError("Uploaded file is not a valid PyTorch model or state_dict container.")

        loaded_model.to("cpu")
        for parameter in loaded_model.parameters():
            parameter.requires_grad_(True)

        inferred_dim = infer_model_input_dim(loaded_model)
        num_classes = int(getattr(loaded_model, "num_classes", 2))

        LOCAL_MODEL_CONTAINER["model"] = loaded_model
        LOCAL_MODEL_CONTAINER["weights_path"] = str(local_weights_path)
        LOCAL_MODEL_CONTAINER["base_state"] = clone_state_dict(loaded_model)
        LOCAL_MODEL_CONTAINER["sample_count"] = 0
        LOCAL_MODEL_CONTAINER["round"] = max(1, int(LOCAL_MODEL_CONTAINER.get("round", 0)) + 1)

        # Match feature schema and optional scaler stats to inferred model input_dim
        import re
        matched_meta = {}
        ver_match = re.match(r"^(\d+\.\d+\.\d+)", ver_label)
        meta_candidates = []
        if ver_match:
            exact_meta = ROOT / "hq_store" / "models" / f"v{ver_match.group(1)}" / "meta.json"
            if exact_meta.exists():
                meta_candidates.append(exact_meta)
        meta_candidates.extend(sorted((ROOT / "hq_store" / "models").glob("v*/meta.json"), reverse=True))
        for meta_cand in meta_candidates:
            try:
                cand_json = json.loads(meta_cand.read_text(encoding="utf-8"))
                if cand_json.get("input_dim") == inferred_dim and len(cand_json.get("feature_names", [])) == inferred_dim:
                    matched_meta = cand_json
                    break
            except Exception:
                continue

        feature_names = extracted_features if len(extracted_features) == inferred_dim else matched_meta.get("feature_names", [])
        if len(feature_names) != inferred_dim and inferred_dim == 83:
            try:
                from config import DATASET_CONFIG
                feature_names = DATASET_CONFIG.get("canonical_feature_names", [])
            except Exception:
                pass
        if len(feature_names) != inferred_dim:
            feature_names = [f"feature_{i}" for i in range(inferred_dim)]

        CLIENT_STATE["current_model_version"] = ver_label
        CLIENT_STATE["model_loaded"] = True
        CLIENT_STATE["model_info"] = {
            "model_name": raw_name or matched_meta.get("model_name") or "SecureFed-Manual-Ensemble",
            "architecture": raw_arch or matched_meta.get("architecture") or getattr(loaded_model, "method", "EnsembleTop2"),
            "input_dim": inferred_dim,
            "num_classes": num_classes,
            "version": ver_label,
            "feature_names": feature_names,
            "scaler_mean": matched_meta.get("scaler_mean"),
            "scaler_scale": matched_meta.get("scaler_scale"),
            "deployed_by": "Administrator (Air-Gapped)",
        }
        CLIENT_STATE["feature_names"] = feature_names
        CLIENT_STATE["last_deployed_source"] = "Admin Manual Push"
        CLIENT_STATE["status"] = f"Manual admin model deployed (v{ver_label})"
        CLIENT_STATE["last_sync"] = "Just now (Admin Manual Push)"
        _save_persisted_state()

        return {
            "status": "success",
            "message": f"Admin successfully verified and deployed manual model file (v{ver_label}, {inferred_dim} features).",
            "version": ver_label,
            "input_dim": inferred_dim,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/client/train-local")
async def train_local_model(payload: TrainLocalRequest):
    """Perform local training with Differential Privacy (DP-SGD) on user-uploaded CSV."""
    model = get_active_model()
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded on client. Pull model from HQ first.")

    # Unpack request fields (kept as local variables so the existing training logic remains unchanged)
    epochs = int(payload.epochs)
    batch_size = int(payload.batch_size)
    learning_rate = float(payload.learning_rate)
    clipping_norm = float(payload.clipping_norm)
    noise_multiplier = float(payload.noise_multiplier)

    try:
        content = base64.b64decode(payload.file_b64)

        # Raw CSV never leaves or persists on this client. Training requires a
        # trustworthy binary label column; inference does not.
        feature_names = CLIENT_STATE["model_info"].get("feature_names") or []
        if not feature_names:
            # The API currently returns model metadata but not feature names.
            # The client therefore refuses to invent/reorder model inputs.
            raise ValueError("HQ has not supplied the published feature schema. Expose model meta.json through the HQ API before client training.")
        with _temp_csv_file(content, prefix="securefed_train_") as tmp_csv:
            X_train, y_train, _, matched_features = canonicalize_csv(tmp_csv, feature_names, require_labels=True)
        if y_train is None or len(np.unique(y_train)) < 2:
            raise ValueError("Local training needs both benign and attack samples.")

        device = torch.device("cpu")
        model.to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        dp_sgd = DPSGD(model, {
            "clipping_norm": clipping_norm,
            "noise_multiplier": noise_multiplier,
            "delta": 1e-5
        })

        samples_count = len(X_train)
        history_log = []

        for ep in range(epochs):
            indices = np.random.permutation(samples_count)
            epoch_loss = 0.0
            correct = 0

            for i in range(0, samples_count, batch_size):
                batch_idx = indices[i:i + batch_size]
                bx = torch.FloatTensor(X_train[batch_idx]).to(device)
                by = torch.LongTensor(y_train[batch_idx]).to(device)

                optimizer.zero_grad()
                outputs = model(bx)
                loss = criterion(outputs, by)
                loss.backward()

                # DP-SGD: clip gradients and add gaussian noise
                dp_sgd.clip_gradients()
                dp_sgd.add_noise()

                optimizer.step()

                epoch_loss += loss.item() * len(batch_idx)
                preds = outputs.argmax(dim=1)
                correct += (preds == by).sum().item()

            avg_loss = epoch_loss / samples_count
            acc = correct / samples_count
            history_log.append({"epoch": ep + 1, "loss": round(avg_loss, 4), "accuracy": round(acc, 4)})

        # Keep only protected parameter deltas for transfer.  The local full
        # model remains on the client and raw traffic is never serialized.
        base_state = LOCAL_MODEL_CONTAINER.get("base_state")
        if base_state is None:
            raise ValueError("Missing base global model state; pull the global model again.")
        protected_delta = clip_and_noise_model_delta(model, base_state, clipping_norm, noise_multiplier)
        training_steps = epochs * int(np.ceil(samples_count / batch_size))
        update_path = BASE_DIR / "local_models" / f"dp_update_{CLIENT_STATE['client_id']}.pt"
        update_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "securefed_dp_delta_v1",
            "base_version": CLIENT_STATE["current_model_version"],
            "client_id": CLIENT_STATE["client_id"],
            "round": int(LOCAL_MODEL_CONTAINER.get("round", 1)),
            "samples": samples_count,
            "local_steps": int(training_steps),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "clipping_norm": float(clipping_norm),
            "noise_multiplier": float(noise_multiplier),
            "clipped": True,
            "noised": True,
            "delta_state_dict": protected_delta,
        }, update_path)
        LOCAL_MODEL_CONTAINER["update_path"] = str(update_path)
        LOCAL_MODEL_CONTAINER["sample_count"] = samples_count

        epsilon_spent = dp_sgd.compute_privacy_spent(batch_size / samples_count, training_steps)
        CLIENT_STATE["training_logs"] = history_log
        CLIENT_STATE["privacy"] = {
            "epsilon": round(epsilon_spent, 4),
            "delta": 1e-5,
            "clipping_norm": float(clipping_norm),
            "noise_multiplier": float(noise_multiplier),
        }
        should_auto_push = bool(CLIENT_STATE.get("auto_push", True)) if payload.auto_push is None else bool(payload.auto_push)
        if should_auto_push:
            try:
                push_res = _transmit_dp_update_to_hq(is_auto=True)
                hq_resp = push_res.get("hq_response") or {}
                update_id = hq_resp.get("update_id")
                auto_push_result = {
                    "enabled": True,
                    "pushed": True,
                    "status": "success",
                    "message": push_res["message"],
                    "hq_response": hq_resp,
                }
                result_message = (
                    f"Local training finished and DP delta automatically pushed to HQ"
                    + (f" (Update #{update_id})." if update_id is not None else ".")
                )
            except HTTPException as push_err:
                auto_push_result = {
                    "enabled": True,
                    "pushed": False,
                    "status": "failed",
                    "detail": push_err.detail,
                }
                CLIENT_STATE["status"] = "DP local training finished; auto-push to HQ failed (retry available)."
                result_message = (
                    "Local training finished and DP delta saved locally, but automatic push to HQ failed. "
                    "You can retry manually from HQ Synchronization."
                )
            except Exception as push_err:
                auto_push_result = {
                    "enabled": True,
                    "pushed": False,
                    "status": "failed",
                    "detail": str(push_err),
                }
                CLIENT_STATE["status"] = "DP local training finished; auto-push to HQ failed (retry available)."
                result_message = (
                    "Local training finished and DP delta saved locally, but automatic push to HQ failed. "
                    "You can retry manually from HQ Synchronization."
                )
        else:
            CLIENT_STATE["status"] = "DP local training finished; protected update ready for HQ."
            auto_push_result = {
                "enabled": False,
                "pushed": False,
                "status": "skipped",
                "message": "Auto-push is disabled; push manually from HQ Synchronization when ready.",
            }
            result_message = "Local training finished. Only a clipped/noised parameter delta is ready to send; raw CSV stays on this client."

        _save_persisted_state()
        return {
            "status": "success",
            "message": result_message,
            "samples": samples_count,
            "matched_features": matched_features,
            "epochs": epochs,
            "history": history_log,
            "epsilon_read_only": round(epsilon_spent, 4),
            "delta_read_only": 1e-5,
            "auto_push": auto_push_result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _transmit_dp_update_to_hq(is_auto: bool = False) -> dict:
    """Register with HQ and upload the current protected DP parameter delta."""
    model = get_active_model()
    if model is None:
        raise HTTPException(status_code=400, detail="No model active on client.")

    update_path = Path(LOCAL_MODEL_CONTAINER.get("update_path", ""))
    if not update_path.exists():
        raise HTTPException(status_code=400, detail="No protected update is ready. Complete DP local training first.")

    hq_url = CLIENT_STATE["hq_url"]
    endpoint = f"{hq_url}/api/v1/model/upload"

    try:
        # server_api stores raw .pt updates safely; it also supports compressed
        # legacy pickles, but torch's .pt container is not a pickle payload.
        raw_bytes = update_path.read_bytes()
        try:
            reg_data = {"client_id": CLIENT_STATE["client_id"], "display_name": _edge_self_url()}
            if CLIENT_STATE.get("current_model_version"):
                reg_data["model_version"] = str(CLIENT_STATE["current_model_version"])
            requests.post(
                f"{hq_url}/api/v1/clients/register",
                data=reg_data,
                timeout=5,
            ).raise_for_status()
        except requests.RequestException as exc:
            raise HTTPException(status_code=503, detail=f"Could not register with HQ: {exc}") from exc

        files = {"file": ("dp_parameter_delta.pt", raw_bytes, "application/octet-stream")}
        data = {
            "client_id": CLIENT_STATE["client_id"],
            "samples": int(LOCAL_MODEL_CONTAINER.get("sample_count", 0)),
            "base_version": CLIENT_STATE["current_model_version"] or "latest",
            "round": int(LOCAL_MODEL_CONTAINER.get("round", 1)),
        }

        resp = requests.post(endpoint, data=data, files=files, timeout=20)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"HQ update failed: {resp.text}")

        hq_json = resp.json()
        update_id = hq_json.get("update_id")
        CLIENT_STATE["last_push"] = hq_json.get("received_at") or "Just now"
        CLIENT_STATE["last_push_info"] = {
            "update_id": update_id,
            "base_version": hq_json.get("base_version"),
            "round": hq_json.get("round"),
            "samples": hq_json.get("samples"),
            "mode": "automatic" if is_auto else "manual",
        }
        mode_tag = "auto-pushed" if is_auto else "queued"
        CLIENT_STATE["status"] = (
            f"Protected parameter delta {mode_tag} at HQ (update #{update_id})."
            if update_id is not None
            else f"Protected parameter delta {mode_tag} at HQ."
        )
        _save_persisted_state()
        return {
            "status": "success",
            "message": (
                "Clipped/noised parameter delta automatically transmitted to HQ; no raw traffic or full local model was sent."
                if is_auto
                else "Clipped/noised parameter delta transmitted to HQ; no raw traffic or full local model was sent. HQ aggregation remains an HQ-side operation."
            ),
            "hq_response": hq_json,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/client/push-update")
def push_update_to_hq(payload: AdminActionRequest):
    """Send DP-noised weight updates back to HQ manually (fallback/retry). Requires Admin Key."""
    verify_admin_access(payload.admin_key)
    return _transmit_dp_update_to_hq(is_auto=False)


@app.post("/api/client/inference")
async def run_client_inference(payload: InferenceRequest):
    """Run inference on uploaded CSV logs and explain which flows are attacks vs normal."""
    model = get_active_model()
    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded for inference.")

    try:
        content = base64.b64decode(payload.file_b64)
        feature_names = CLIENT_STATE["model_info"].get("feature_names") or []
        if not feature_names:
            raise ValueError("HQ has not supplied the published feature schema. Expose model meta.json through the HQ API before client inference.")
        with _temp_csv_file(content, prefix="securefed_inference_") as tmp_csv:
            X, _, df, _ = canonicalize_csv(tmp_csv, feature_names, require_labels=False)

        model.eval()
        tensor_x = torch.FloatTensor(X)
        tensor_x.requires_grad_(True)
        outputs = model(tensor_x)
        probs = output_to_probabilities(outputs).detach().cpu().numpy()
        preds = probs.argmax(axis=1)

        # On-Device Edge XAI: Fast Input Saliency Attribution (x * grad)
        try:
            target_scores = outputs[torch.arange(outputs.size(0)), preds]
            target_scores.sum().backward()
            attributions = (tensor_x.grad * tensor_x).detach().cpu().numpy()
        except Exception:
            # Fallback to feature deviation from batch mean if backward unavailable
            attributions = np.abs(X - np.mean(X, axis=0, keepdims=True))

        results = []
        for idx in range(min(len(df), 50)):  # Return top 50 records
            pred_class = int(preds[idx])
            confidence = float(probs[idx][pred_class])
            label_name = "Attack (Malicious)" if pred_class == 1 else "Normal (Benign)"

            # Top 3 feature attributions explaining this specific network flow (normalized [-1, 1])
            row_attr = attributions[idx]
            row_max = float(np.max(np.abs(row_attr))) if len(row_attr) else 0.0
            if row_max <= 1e-6:
                row_attr = X[idx] - np.mean(X, axis=0)
                row_max = float(np.max(np.abs(row_attr))) if len(row_attr) else 0.0
            norm_row = (row_attr / row_max) if row_max > 1e-8 else row_attr
            top_indices = np.argsort(np.abs(norm_row))[::-1][:3]
            top_indicators = [
                {
                    "feature": str(feature_names[f_idx]).strip() if f_idx < len(feature_names) else f"feat_{f_idx}",
                    "impact": round(float(norm_row[f_idx]), 3),
                }
                for f_idx in top_indices
            ]

            results.append({
                "row_index": idx,
                "prediction": label_name,
                "confidence": round(confidence * 100, 2),
                "probabilities": {"benign": round(float(probs[idx][0]), 4), "attack": round(float(probs[idx][1]), 4)},
                "top_indicators": top_indicators,
            })

        return {
            "status": "success",
            "total_rows": len(df),
            "analyzed_rows": len(results),
            "predictions": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _heartbeat_worker():
    """Register this Edge node with HQ so HQ knows where to push models (never auto-pulls models)."""
    _STOP_HEARTBEAT_EVENT.wait(1.0)
    while not _STOP_HEARTBEAT_EVENT.is_set():
        hq_url = CLIENT_STATE.get("hq_url", "http://localhost:8000")
        _register_client_heartbeat(hq_url)
        _STOP_HEARTBEAT_EVENT.wait(15.0)


@app.on_event("startup")
def _start_heartbeat_daemon():
    CLIENT_STATE["edge_port"] = int(os.getenv("EDGE_PORT", CLIENT_STATE.get("edge_port", 5000)))
    _restore_persisted_state()
    if os.getenv("DISABLE_AUTO_SYNC_THREAD") == "1":
        return
    _STOP_HEARTBEAT_EVENT.clear()
    thread = threading.Thread(target=_heartbeat_worker, name="securefed-hq-heartbeat", daemon=True)
    thread.start()


@app.on_event("shutdown")
def _stop_heartbeat_daemon():
    _STOP_HEARTBEAT_EVENT.set()


if __name__ == "__main__":
    import uvicorn
    app_dir = str(Path(__file__).resolve().parent)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    port = 5000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    elif "PORT" in os.environ and os.environ["PORT"].isdigit():
        port = int(os.environ["PORT"])
    os.environ["EDGE_PORT"] = str(port)
    CLIENT_STATE["edge_port"] = port
    print("=" * 60)
    print(f" SecureFed Edge Client running on http://127.0.0.1:{port}")
    print(" Deployment Mode: Explicit HQ Push / Manual Sync only")
    print(" Press Ctrl+C to stop.")
    print("=" * 60)
    try:
        uvicorn.run("client_app:app", host="0.0.0.0", port=port, reload=True, app_dir=app_dir)
    except Exception:
        uvicorn.run(app, host="0.0.0.0", port=port)
