"""SQLite registry for HQ model versions, clients, and weight updates.

Weights stay as .pt files on disk. This module only stores metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from config import DATABASE_CONFIG, PROJECT_ROOT

HQ_STORE_DIR = PROJECT_ROOT / "hq_store"
MODELS_DIR = HQ_STORE_DIR / "models"
UPDATES_DIR = HQ_STORE_DIR / "updates"
AGGREGATED_UPDATES_DIR = HQ_STORE_DIR / "aggregated_updates"
DB_PATH = Path(DATABASE_CONFIG.get("path", PROJECT_ROOT / "data" / "securefed_nids.db"))

VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_version(version: str) -> str:
    raw = version.strip()
    if raw.lower() in {"latest", "current"}:
        return "latest"
    match = VERSION_RE.match(raw)
    if not match:
        raise ValueError(f"Invalid version '{version}'. Use semver like 1.0.0 or v1.0.0.")
    major, minor, patch = (int(p) for p in match.groups())
    return f"{major}.{minor}.{patch}"


def bump_patch(version: str) -> str:
    major, minor, patch = (int(p) for p in normalize_version(version).split("."))
    return f"{major}.{minor}.{patch + 1}"


class ModelRegistry:
    def __init__(self, db_path: Optional[Path] = None, store_dir: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.store_dir = Path(store_dir) if store_dir else HQ_STORE_DIR
        self.models_dir = self.store_dir / "models"
        self.updates_dir = self.store_dir / "updates"
        self.aggregated_updates_dir = self.store_dir / "aggregated_updates"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.updates_dir.mkdir(parents=True, exist_ok=True)
        self.aggregated_updates_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL DEFAULT 'SecureFed-Global-Ensemble',
                    architecture TEXT,
                    fl_method TEXT,
                    input_dim INTEGER,
                    num_classes INTEGER,
                    accuracy REAL,
                    weights_path TEXT NOT NULL,
                    package_path TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    is_latest INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS clients (
                    client_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    registered_at TEXT NOT NULL,
                    last_seen TEXT,
                    last_version TEXT
                );

                CREATE TABLE IF NOT EXISTS client_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id TEXT NOT NULL,
                    base_version TEXT NOT NULL,
                    round INTEGER DEFAULT 0,
                    samples INTEGER DEFAULT 0,
                    weights_path TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'received',
                    aggregated_version TEXT,
                    aggregation_method TEXT,
                    aggregated_at TEXT,
                    archive_path TEXT,
                    dp_clipping_norm REAL,
                    dp_noise_multiplier REAL,
                    local_steps INTEGER DEFAULT 1,
                    FOREIGN KEY(client_id) REFERENCES clients(client_id)
                );

                CREATE TABLE IF NOT EXISTS fl_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

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
                );

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
                );
                """
            )
            existing_cols = {
                row["name"] for row in conn.execute("PRAGMA table_info(client_updates)").fetchall()
            }
            for col_name, col_type in (
                ("aggregated_version", "TEXT"),
                ("aggregation_method", "TEXT"),
                ("aggregated_at", "TEXT"),
                ("archive_path", "TEXT"),
                ("dp_clipping_norm", "REAL"),
                ("dp_noise_multiplier", "REAL"),
                ("local_steps", "INTEGER DEFAULT 1"),
            ):
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE client_updates ADD COLUMN {col_name} {col_type}")

            conn.execute(
                """
                INSERT OR IGNORE INTO fl_state (key, value, updated_at)
                VALUES ('open_round', '1', ?)
                """,
                (utc_now(),),
            )

            # Backfill legacy aggregated records if dashboard_updates has target version info
            try:
                conn.execute(
                    """
                    UPDATE client_updates
                    SET aggregated_version = (
                        SELECT d.version_to FROM dashboard_updates d
                        WHERE d.update_id = client_updates.id AND d.version_to IS NOT NULL
                    )
                    WHERE status = 'aggregated' AND aggregated_version IS NULL
                    """
                )
            except Exception:
                pass

    def get_open_round(self) -> int:
        """Return the currently open federated learning round number."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM fl_state WHERE key = 'open_round'"
            ).fetchone()
        try:
            return max(1, int(row["value"])) if row else 1
        except (TypeError, ValueError):
            return 1

    def set_open_round(self, round_num: int) -> int:
        val = max(1, int(round_num))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fl_state (key, value, updated_at)
                VALUES ('open_round', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (str(val), utc_now()),
            )
        return val

    def advance_open_round(self) -> int:
        """Increment and persist the open round after an aggregation completes."""
        return self.set_open_round(self.get_open_round() + 1)

    def record_edge_push(self, version: str, targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Persist the most recently pushed model version and Edge targets in fl_state."""
        now = utc_now()
        payload = {
            "version": normalize_version(version),
            "targets": list(targets or []),
            "pushed_at": now,
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fl_state (key, value, updated_at)
                VALUES ('last_pushed_model', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (json.dumps(payload), now),
            )
        return payload

    def get_last_pushed_model(self) -> Optional[Dict[str, Any]]:
        """Return metadata for the model version most recently pushed from HQ to Edge."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM fl_state WHERE key = 'last_pushed_model'"
            ).fetchone()
        if not row:
            return None
        try:
            data = json.loads(row["value"])
            return data if isinstance(data, dict) and data.get("version") else None
        except Exception:
            return None

    def fl_state_dir(self) -> Path:
        """Directory storing persistent HQ FL state (e.g. SCAFFOLD control variates)."""
        d = self.store_dir / "fl_state"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _holdout_schema_key(input_dim: int, feature_names: Optional[List[str]] = None) -> str:
        norm_feats = [str(f).strip().lower() for f in (feature_names or [])]
        if norm_feats:
            digest = hashlib.sha256("|".join(norm_feats).encode("utf-8")).hexdigest()[:12]
            return f"d{int(input_dim)}_{digest}"
        return f"d{int(input_dim)}_generic"

    def eval_holdout_path(self, input_dim: int, feature_names: Optional[List[str]] = None) -> Path:
        holdouts_dir = self.fl_state_dir() / "eval_holdouts"
        holdouts_dir.mkdir(parents=True, exist_ok=True)
        key = self._holdout_schema_key(input_dim, feature_names)
        return holdouts_dir / f"holdout_{key}.npz"

    def save_eval_holdout(
        self,
        X_test: Any,
        y_test: Any,
        input_dim: int,
        feature_names: Optional[List[str]] = None,
        dataset_name: str = "",
    ) -> Optional[Path]:
        """Persist HQ's server-side holdout evaluation split (never client traffic) for fast post-aggregation testing."""
        x_arr = np.asarray(X_test, dtype=np.float32)
        y_arr = np.asarray(y_test, dtype=np.int64).reshape(-1)
        if x_arr.ndim != 2 or len(x_arr) == 0 or len(x_arr) != len(y_arr):
            return None
        dest = self.eval_holdout_path(input_dim, feature_names)
        np.savez_compressed(dest, X_test=x_arr, y_test=y_arr)
        meta_dest = dest.with_suffix(".meta.json")
        meta_dest.write_text(
            json.dumps({
                "dataset_name": dataset_name or "hq_holdout",
                "input_dim": int(x_arr.shape[1]),
                "samples": int(len(x_arr)),
                "saved_at": utc_now(),
            }, indent=2),
            encoding="utf-8",
        )
        return dest

    def load_eval_holdout(
        self,
        input_dim: int,
        feature_names: Optional[List[str]] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """Load cached server-side holdout split matching model input_dim & schema."""
        candidates = [self.eval_holdout_path(input_dim, feature_names)]
        generic = self.eval_holdout_path(input_dim, None)
        if generic not in candidates:
            candidates.append(generic)
        holdouts_dir = self.fl_state_dir() / "eval_holdouts"
        if holdouts_dir.exists():
            for extra_path in sorted(holdouts_dir.glob(f"holdout_d{int(input_dim)}_*.npz")):
                if extra_path not in candidates:
                    candidates.append(extra_path)
        for path in candidates:
            if not path.exists():
                continue
            try:
                with np.load(path) as data:
                    x_arr = np.asarray(data["X_test"], dtype=np.float32)
                    y_arr = np.asarray(data["y_test"], dtype=np.int64).reshape(-1)
                if x_arr.ndim == 2 and x_arr.shape[1] == int(input_dim) and len(x_arr) == len(y_arr) and len(x_arr) > 0:
                    meta_path = path.with_suffix(".meta.json")
                    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                    return x_arr, y_arr, meta
            except Exception:
                continue
        return None

    def latest_version(self) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT version FROM model_versions WHERE is_latest = 1 ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row["version"] if row else None

    def next_version(self, requested: Optional[str] = None) -> str:
        if requested:
            version = normalize_version(requested)
            if self.get_version(version):
                raise ValueError(f"Version {version} already exists.")
            return version
        latest = self.latest_version()
        if not latest:
            return "1.0.0"
        return bump_patch(latest)

    def get_version(self, version: str) -> Optional[Dict[str, Any]]:
        if version.lower() in {"latest", "current"}:
            latest = self.latest_version()
            if not latest:
                return None
            version = latest
        else:
            version = normalize_version(version)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE version = ?", (version,)
            ).fetchone()
        return dict(row) if row else None

    def list_versions(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM model_versions ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def count_versions(self) -> int:
        """Return the number of published model versions."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM model_versions").fetchone()
        return int(row["count"] if row else 0)

    def version_dir(self, version: str) -> Path:
        return self.models_dir / f"v{normalize_version(version)}"

    def delete_version(self, version: str) -> Optional[Dict[str, Any]]:
        """Delete a published model version from SQLite registry and disk storage."""
        clean_ver = normalize_version(version)
        if clean_ver == "latest":
            raise ValueError("Specify a concrete version to delete (e.g. 1.0.5), not 'latest'.")

        record = self.get_version(clean_ver)
        if record is None:
            return None

        was_latest = bool(record.get("is_latest"))
        new_latest: Optional[str] = None

        last_pushed = self.get_last_pushed_model()
        with self._connect() as conn:
            conn.execute("DELETE FROM model_versions WHERE version = ?", (clean_ver,))
            if last_pushed and last_pushed.get("version") == clean_ver:
                conn.execute("DELETE FROM fl_state WHERE key = 'last_pushed_model'")
            if was_latest:
                nxt = conn.execute(
                    "SELECT id, version FROM model_versions ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if nxt:
                    conn.execute("UPDATE model_versions SET is_latest = 1 WHERE id = ?", (nxt["id"],))
                    new_latest = nxt["version"]
            else:
                new_latest = self.latest_version()

        target_dir = self.version_dir(clean_ver)
        files_removed = False
        try:
            models_root = self.models_dir.resolve()
            resolved_target = target_dir.resolve()
            if resolved_target != models_root and models_root in resolved_target.parents and resolved_target.exists():
                shutil.rmtree(resolved_target)
                files_removed = True
        except Exception:
            pass

        return {
            "deleted_version": clean_ver,
            "was_latest": was_latest,
            "new_latest": new_latest,
            "files_removed": files_removed,
        }

    def publish(
        self,
        weights_src: Path,
        package_src: Optional[Path] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        weights_src = Path(weights_src)
        if not weights_src.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_src}")

        new_version = self.next_version(version)
        dest_dir = self.version_dir(new_version)
        dest_dir.mkdir(parents=True, exist_ok=True)

        weights_dest = dest_dir / "weights.pt"
        weights_dest.write_bytes(weights_src.read_bytes())

        package_dest = None
        if package_src and Path(package_src).exists():
            package_dest = dest_dir / "package.pt"
            package_dest.write_bytes(Path(package_src).read_bytes())

        meta = dict(metadata or {})
        meta.update({
            "version": new_version,
            "created_at": utc_now(),
            "weights_path": str(weights_dest),
            "package_path": str(package_dest) if package_dest else None,
        })
        (dest_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        with self._connect() as conn:
            conn.execute("UPDATE model_versions SET is_latest = 0")
            conn.execute(
                """
                INSERT INTO model_versions (
                    version, model_name, architecture, fl_method, input_dim, num_classes,
                    accuracy, weights_path, package_path, notes, created_at, is_latest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    new_version,
                    meta.get("model_name", "SecureFed-Global-Ensemble"),
                    meta.get("architecture", "EnsembleTop2"),
                    meta.get("fl_method"),
                    meta.get("input_dim"),
                    meta.get("num_classes"),
                    meta.get("accuracy"),
                    str(weights_dest),
                    str(package_dest) if package_dest else None,
                    notes,
                    meta["created_at"],
                ),
            )
        record = self.get_version(new_version)
        if record is None:
            raise RuntimeError("Failed to persist published version.")
        return record

    def register_client(
        self,
        client_id: str,
        display_name: Optional[str] = None,
        ip_address: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = utc_now()
        clean_ip = (ip_address or "").strip() or None
        clean_ver = (model_version or "").strip() or None
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM clients WHERE client_id = ?", (client_id,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE clients SET display_name = COALESCE(?, display_name), last_seen = ?, last_version = COALESCE(?, last_version) WHERE client_id = ?",
                    (display_name, now, clean_ver, client_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO clients (client_id, display_name, registered_at, last_seen, last_version)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (client_id, display_name or client_id, now, now, clean_ver),
                )

            # Sync with dashboard_clients telemetry table
            conn.execute(
                """
                INSERT INTO dashboard_clients (client_id, client_name, ip_address, last_seen, status, model_version, created_at)
                VALUES (?, ?, ?, ?, 'active', ?, ?)
                ON CONFLICT(client_id) DO UPDATE SET
                    client_name = COALESCE(NULLIF(excluded.client_name, excluded.client_id), dashboard_clients.client_name),
                    ip_address = COALESCE(excluded.ip_address, dashboard_clients.ip_address),
                    model_version = COALESCE(excluded.model_version, dashboard_clients.model_version),
                    last_seen = excluded.last_seen,
                    status = 'active'
                """,
                (client_id, display_name or client_id, clean_ip, now, clean_ver, now),
            )

            row = conn.execute(
                "SELECT * FROM clients WHERE client_id = ?", (client_id,)
            ).fetchone()
        return dict(row)

    def record_client_download(
        self,
        client_id: Optional[str],
        model_version: str,
        ip_address: Optional[str] = None,
    ) -> None:
        clean_id = (client_id or "").strip()
        if not clean_id:
            return
        self.register_client(clean_id, ip_address=ip_address, model_version=model_version)
        now = utc_now()
        clean_ip = (ip_address or "").strip() or None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE dashboard_clients
                SET download_count = COALESCE(download_count, 0) + 1,
                    model_version = ?,
                    ip_address = COALESCE(?, ip_address),
                    last_seen = ?,
                    status = 'active'
                WHERE client_id = ?
                """,
                (model_version, clean_ip, now, clean_id),
            )

    def list_clients(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM clients ORDER BY last_seen DESC").fetchall()
        return [dict(r) for r in rows]

    def save_update(
        self,
        client_id: str,
        weights_bytes: bytes,
        samples: int = 0,
        base_version: str = "latest",
        round_idx: int = 0,
        clipping_norm: Optional[float] = None,
        noise_multiplier: Optional[float] = None,
        local_steps: int = 1,
        ip_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved = self.get_version(base_version)
        version_label = resolved["version"] if resolved else normalize_version(base_version) if base_version not in {"latest", "current"} else "unversioned"
        self.register_client(client_id, ip_address=ip_address, model_version=version_label)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        dest_dir = self.updates_dir / f"v{version_label}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{client_id}_{stamp}Z_r{round_idx}.pt"
        if dest_path.exists():
            micro = datetime.now(timezone.utc).strftime("%f")
            dest_path = dest_dir / f"{client_id}_{stamp}_{micro}Z_r{round_idx}.pt"
        dest_path.write_bytes(weights_bytes)

        now = utc_now()
        clean_ip = (ip_address or "").strip() or None
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO client_updates (
                    client_id, base_version, round, samples, weights_path, received_at, status,
                    dp_clipping_norm, dp_noise_multiplier, local_steps
                ) VALUES (?, ?, ?, ?, ?, ?, 'received', ?, ?, ?)
                """,
                (
                    client_id,
                    version_label,
                    int(round_idx),
                    int(samples),
                    str(dest_path),
                    now,
                    clipping_norm,
                    noise_multiplier,
                    max(1, int(local_steps or 1)),
                ),
            )
            upd_id = cur.lastrowid
            conn.execute(
                "UPDATE clients SET last_seen = ?, last_version = ? WHERE client_id = ?",
                (now, version_label, client_id),
            )

            # Sync with dashboard_updates & dashboard_clients telemetry tables
            conn.execute(
                """
                INSERT INTO dashboard_updates (update_id, client_id, version_from, update_size, status, created_at)
                VALUES (?, ?, ?, ?, 'pending', ?)
                ON CONFLICT(update_id) DO UPDATE SET
                    client_id = excluded.client_id,
                    version_from = excluded.version_from,
                    update_size = excluded.update_size,
                    status = 'pending'
                """,
                (upd_id, client_id, version_label, len(weights_bytes), now),
            )
            conn.execute(
                """
                UPDATE dashboard_clients
                SET upload_count = COALESCE(upload_count, 0) + 1,
                    model_version = ?,
                    ip_address = COALESCE(?, ip_address),
                    last_seen = ?,
                    status = 'active'
                WHERE client_id = ?
                """,
                (version_label, clean_ip, now, client_id),
            )

            row = conn.execute(
                "SELECT * FROM client_updates WHERE weights_path = ?", (str(dest_path),)
            ).fetchone()
        return dict(row)

    def publish_ensemble(self, ensemble_model, input_dim: int, num_classes: int,
                         feature_names: Optional[List[str]] = None, accuracy: float = 0.0,
                         notes: str = "", extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Save an in-memory ensemble model to disk and register a new version.

        ``extra_metadata`` is intentionally stored only in the version's
        ``meta.json`` file.  It lets the HQ dashboard keep reproducibility
        details, evaluation summaries, and visualization manifests alongside
        an immutable published model without placing tensors or raw traffic in
        SQLite.
        """
        import torch
        staging_dir = self.models_dir / "_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        weights_path = staging_dir / f"ensemble_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.pt"
        torch.save(ensemble_model, weights_path)
        metadata = {
            "model_name": "SecureFed-Global-Ensemble",
            "architecture": "EnsembleTop2",
            "input_dim": input_dim,
            "num_classes": num_classes,
            "feature_names": feature_names or [],
            "accuracy": accuracy,
        }
        metadata.update(extra_metadata or {})
        record = self.publish(
            weights_src=weights_path,
            metadata=metadata,
            notes=notes,
        )
        try:
            weights_path.unlink(missing_ok=True)
        except Exception:
            pass
        return record

    def list_updates(self, client_id: Optional[str] = None, version: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM client_updates WHERE 1=1"
        params: List[Any] = []
        if client_id:
            query += " AND client_id = ?"
            params.append(client_id)
        if version:
            query += " AND base_version = ?"
            params.append(normalize_version(version) if version not in {"latest", "current"} else (self.latest_version() or version))
        query += " ORDER BY id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def archive_aggregated_updates(
        self,
        update_ids: List[int],
        new_version: str,
        method: str,
        aggregated_at: Optional[str] = None,
        base_version: str = "",
        client_ids: Optional[List[str]] = None,
        total_samples: int = 0,
    ) -> Path:
        """Move merged client update .pt files into a dedicated aggregated_updates folder."""
        clean_new_ver = normalize_version(new_version)
        round_dir = self.aggregated_updates_dir / f"v{clean_new_ver}"
        round_dir.mkdir(parents=True, exist_ok=True)
        now = aggregated_at or utc_now()
        archived_files: List[Dict[str, Any]] = []

        with self._connect() as conn:
            for u_id in update_ids:
                row = conn.execute(
                    "SELECT id, client_id, weights_path, samples, round FROM client_updates WHERE id = ?",
                    (u_id,),
                ).fetchone()
                if not row:
                    continue
                src_path = Path(row["weights_path"])
                dest_path = round_dir / src_path.name
                if src_path.exists() and src_path.resolve() != dest_path.resolve():
                    if dest_path.exists():
                        dest_path = round_dir / f"u{u_id}_{src_path.name}"
                    shutil.move(str(src_path), str(dest_path))
                final_path = str(dest_path) if dest_path.exists() else str(src_path)
                conn.execute(
                    """
                    UPDATE client_updates
                    SET status = 'aggregated',
                        weights_path = ?,
                        archive_path = ?,
                        aggregated_version = ?,
                        aggregation_method = ?,
                        aggregated_at = ?
                    WHERE id = ?
                    """,
                    (final_path, final_path, clean_new_ver, method, now, u_id),
                )
                archived_files.append({
                    "update_id": int(u_id),
                    "client_id": row["client_id"],
                    "samples": int(row["samples"] or 0),
                    "round": int(row["round"] or 0),
                    "file": final_path,
                })

        manifest = {
            "aggregated_version": clean_new_ver,
            "base_version": base_version,
            "method": method,
            "aggregated_at": now,
            "total_samples": total_samples,
            "client_ids": client_ids or [],
            "update_ids": update_ids,
            "archived_files": archived_files,
        }
        (round_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return round_dir
