"""
SecureFed-NIDS Dashboard - Separate Page Routes
==============================================
Routes for dashboard sub-pages (Models, Clients, Updates, Training, Analytics)
"""

import json
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime
from typing import Any

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
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def setup_dashboard_routes(app, registry):
    """Setup separate page routes for the dashboard."""
    templates = Jinja2Templates(directory=str(ROOT / "server_hq" / "templates"))

    @app.get("/models", response_class=HTMLResponse)
    async def models_page(request: Request):
        versions = registry.list_versions()
        latest = registry.latest_version()

        # Get latest version info
        latest_info = {}
        if latest:
            if isinstance(latest, dict):
                latest_info = {
                    "version": latest.get("version", "N/A"),
                    "model_name": latest.get("model_name", "N/A"),
                    "accuracy": latest.get("accuracy"),
                }
            elif isinstance(latest, str):
                latest_info = {"version": latest}

        return templates.TemplateResponse(
            request,
            "models.html",
            {
                "request": request,
                "versions": _clean_for_json(versions),
                "latest": _clean_for_json(latest_info),
                "timestamp": datetime.now().isoformat()
            }
        )

    @app.get("/clients", response_class=HTMLResponse)
    async def clients_page(request: Request):
        # Get client data from registry
        try:
            import sqlite3
            conn = sqlite3.connect(registry.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT client_id, client_name, ip_address, last_seen, status,
                       download_count, upload_count
                FROM dashboard_clients
                ORDER BY last_seen DESC
                LIMIT 100
            ''')
            columns = [desc[0] for desc in cursor.description]
            clients = []
            for row in cursor.fetchall():
                client = dict(zip(columns, row))
                client["last_seen"] = client.get("last_seen") or client.get("created_at", "")
                clients.append(client)
            conn.close()
        except Exception as e:
            clients = []

        total = len(clients)

        return templates.TemplateResponse(
            request,
            "clients.html",
            {
                "request": request,
                "clients": _clean_for_json(clients),
                "total_clients": total,
                "timestamp": datetime.now().isoformat()
            }
        )

    @app.get("/updates", response_class=HTMLResponse)
    async def updates_page(request: Request):
        try:
            updates = registry.list_updates()
            total = len(updates)
            processed = sum(1 for u in updates if u.get("status") in ("aggregated", "processed"))
            pending = sum(1 for u in updates if u.get("status") in ("received", "pending"))
            failed = sum(1 for u in updates if u.get("status") == "failed")
            stats = {
                "total_updates": total,
                "processed_updates": processed,
                "pending_updates": pending,
                "failed_updates": failed,
            }
        except Exception:
            updates = []
            stats = {"total_updates": 0, "processed_updates": 0, "pending_updates": 0, "failed_updates": 0}

        return templates.TemplateResponse(
            request,
            "updates.html",
            {
                "request": request,
                "updates": _clean_for_json(updates[:50]),
                "total_updates": stats["total_updates"],
                "processed_updates": stats["processed_updates"],
                "pending_updates": stats["pending_updates"],
                "failed_updates": stats["failed_updates"],
                "latest_version": registry.latest_version(),
                "open_round": registry.get_open_round(),
                "aggregated_folder": str(registry.aggregated_updates_dir),
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.get("/training", response_class=HTMLResponse)
    async def training_page(request: Request):
        """Dataset upload and HQ model-training page."""
        return templates.TemplateResponse(
            request=request,
            name="training.html",
            context={
                "request": request,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.get("/analytics", response_class=HTMLResponse)
    async def analytics_page(request: Request):
        """Analytics overview page."""
        return templates.TemplateResponse(
            request=request,
            name="analytics.html",
            context={
                "request": request,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.get("/xai", response_class=HTMLResponse)
    async def xai_page(request: Request):
        """Explainable AI (SHAP) Dashboard."""
        return templates.TemplateResponse(
            request=request,
            name="xai_dashboard.html",
            context={
                "request": request,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @app.get("/api/xai/datasets")
    async def xai_datasets():
        """List available datasets in data/ and training_uploads/ for XAI analysis."""
        from config import DATA_DIR, PROJECT_ROOT
        names = set()
        data_dir = Path(DATA_DIR)
        if data_dir.exists():
            for f in data_dir.glob("*.csv"):
                names.add(f.stem)
        uploads_dir = PROJECT_ROOT / "hq_store" / "training_uploads"
        if uploads_dir.exists():
            for f in uploads_dir.glob("*.csv"):
                # Strip optional UUID prefix (uuid_filename.csv) for display readability
                parts = f.stem.split("_", 1)
                clean = parts[1] if len(parts) == 2 and len(parts[0]) == 36 else f.stem
                names.add(clean)
        return {"status": "success", "datasets": sorted(names)}

    @app.post("/api/explain")
    async def explain_model(request: Request):
        """Run real PyTorch model attribution (Input Saliency / Perturbation) for global XAI audit."""
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.preprocessing import StandardScaler
        from config import DATASET_CONFIG, PROJECT_ROOT

        try:
            body = await request.json()
            dataset_name = body.get("dataset", "RT_IOT2022")
            model_choice = body.get("model", "ensemble")

            # 1. Locate dataset CSV in data/ or hq_store/training_uploads/
            search_dirs = [PROJECT_ROOT / "data", PROJECT_ROOT / "hq_store" / "training_uploads"]
            csv_path = None
            for d in search_dirs:
                if not d.exists():
                    continue
                for p in sorted(d.glob("*.csv"), reverse=True):
                    if dataset_name.lower() in p.stem.lower():
                        csv_path = p
                        break
                if csv_path:
                    break
            if csv_path is None:
                for d in search_dirs:
                    if d.exists():
                        csvs = sorted(list(d.glob("*.csv")), reverse=True)
                        if csvs:
                            csv_path = csvs[0]
                            break

            if not csv_path or not csv_path.exists():
                raise FileNotFoundError("No dataset CSV found in data/ or hq_store/training_uploads/")

            # 2. Load latest published global ensemble model & its feature schema
            record = registry.get_version("latest")
            if record is None:
                raise ValueError("No published global model available in HQ registry.")
            meta_path = registry.version_dir(record["version"]) / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

            feature_names = meta.get("feature_names") or DATASET_CONFIG.get("canonical_feature_names", [])
            weights_path = Path(record["weights_path"])
            try:
                loaded_model = torch.load(weights_path, map_location="cpu", weights_only=False)
            except TypeError:
                loaded_model = torch.load(weights_path, map_location="cpu")

            # If a sub-architecture inside the ensemble is specifically requested, use it if available
            target_model = loaded_model
            if model_choice != "ensemble" and hasattr(loaded_model, "models"):
                for sub in loaded_model.models:
                    if model_choice.lower() in sub.__class__.__name__.lower():
                        target_model = sub
                        break

            # 3. Extract & standardize features matching the published model schema
            df = pd.read_csv(csv_path, nrows=128)
            norm_cols = {str(c).strip().lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
            X = np.zeros((len(df), len(feature_names)), dtype=np.float32)
            matched = 0
            for idx, feat in enumerate(feature_names):
                key = str(feat).strip().lower().replace(" ", "_").replace("-", "_")
                src_col = norm_cols.get(key)
                if src_col is None:
                    continue
                matched += 1
                col_s = pd.to_numeric(df[src_col], errors="coerce")
                if col_s.isna().all() and df[src_col].dtype == object:
                    col_s = pd.Series(pd.factorize(df[src_col].astype(str))[0], index=df.index)
                X[:, idx] = col_s.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            if len(X) > 1:
                X = StandardScaler().fit_transform(X).astype(np.float32)

            # 4. Run real forward pass + gradient input saliency attribution (x * d_logit/d_x)
            target_model.eval()
            tx = torch.tensor(X, dtype=torch.float32, requires_grad=True)
            outputs = target_model(tx)
            is_log_prob = torch.allclose(torch.exp(outputs).sum(dim=1), torch.ones(outputs.size(0)), atol=1e-4)
            probs = torch.exp(outputs) if is_log_prob else torch.softmax(outputs, dim=1)
            probs_np = probs.detach().cpu().numpy()
            preds = probs_np.argmax(axis=1)

            # Attribute towards the predicted / attack class logit
            attack_idx = 1 if outputs.size(1) > 1 else 0
            target_scores = outputs[:, attack_idx]
            target_scores.sum().backward()
            grad = tx.grad.detach().cpu().numpy() if tx.grad is not None else np.zeros_like(X)
            # Signed contribution relative to attack class: mean(x * grad_attack)
            attributions = (grad * X).mean(axis=0)

            # Scale values for crisp visualization while preserving relative sign & magnitude
            max_abs = float(np.max(np.abs(attributions))) if len(attributions) else 0.0
            if max_abs > 1e-8:
                norm_attr = attributions / max_abs
            else:
                # Feature-perturbation sensitivity fallback if gradient is flat
                base_prob = float(probs_np[:, attack_idx].mean())
                norm_attr = np.zeros(len(feature_names), dtype=np.float32)
                for i in range(min(len(feature_names), 40)):
                    X_perm = X.copy()
                    X_perm[:, i] = 0.0
                    with torch.no_grad():
                        out_p = target_model(torch.tensor(X_perm, dtype=torch.float32))
                        p_p = torch.exp(out_p) if is_log_prob else torch.softmax(out_p, dim=1)
                        norm_attr[i] = float(base_prob - p_p[:, attack_idx].mean().item())
                m_p = float(np.max(np.abs(norm_attr)))
                if m_p > 1e-8:
                    norm_attr = norm_attr / m_p

            features_list = [
                {"name": str(feature_names[i]).strip(), "value": round(float(norm_attr[i]), 4)}
                for i in range(len(feature_names))
            ]
            features_list.sort(key=lambda x: abs(x["value"]), reverse=True)
            top_24 = features_list[:24]

            attack_ratio = float((preds == 1).mean()) if len(preds) > 0 else 0.0
            majority_attack = attack_ratio >= 0.5
            pred_class = f"Attack ({attack_ratio*100:.0f}% flows)" if attack_ratio > 0 else "Normal (Benign)"
            avg_conf = float(probs_np[np.arange(len(preds)), preds].mean()) if len(preds) > 0 else 0.9

            return {
                "status": "success",
                "version": record["version"],
                "dataset": dataset_name,
                "model": f"v{record['version']} ({model_choice})",
                "matched_features": matched,
                "features": top_24,
                "top_feature": top_24[0]["name"] if top_24 else "N/A",
                "confidence": round(avg_conf, 4),
                "prediction": pred_class,
            }
        except Exception as exc:
            return JSONResponse(status_code=500, content={"status": "error", "message": str(exc), "features": []})
