"""SQLite registry for HQ model versions, clients, and weight updates.

Weights stay as .pt files on disk. This module only stores metadata.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from config import DATABASE_CONFIG, PROJECT_ROOT

HQ_STORE_DIR = PROJECT_ROOT / "hq_store"
MODELS_DIR = HQ_STORE_DIR / "models"
UPDATES_DIR = HQ_STORE_DIR / "updates"
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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.updates_dir.mkdir(parents=True, exist_ok=True)
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
                    FOREIGN KEY(client_id) REFERENCES clients(client_id)
                );
                """
            )

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

    def register_client(self, client_id: str, display_name: Optional[str] = None) -> Dict[str, Any]:
        now = utc_now()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM clients WHERE client_id = ?", (client_id,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE clients SET display_name = COALESCE(?, display_name), last_seen = ? WHERE client_id = ?",
                    (display_name, now, client_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO clients (client_id, display_name, registered_at, last_seen)
                    VALUES (?, ?, ?, ?)
                    """,
                    (client_id, display_name or client_id, now, now),
                )

            # Sync with dashboard_clients table if present
            try:
                conn.execute(
                    """
                    INSERT INTO dashboard_clients (client_id, client_name, last_seen, status, created_at)
                    VALUES (?, ?, ?, 'active', ?)
                    ON CONFLICT(client_id) DO UPDATE SET last_seen = excluded.last_seen, status = 'active'
                    """,
                    (client_id, display_name or client_id, now, now),
                )
            except Exception:
                pass

            row = conn.execute(
                "SELECT * FROM clients WHERE client_id = ?", (client_id,)
            ).fetchone()
        return dict(row)

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
    ) -> Dict[str, Any]:
        self.register_client(client_id)
        resolved = self.get_version(base_version)
        version_label = resolved["version"] if resolved else normalize_version(base_version) if base_version not in {"latest", "current"} else "unversioned"

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest_dir = self.updates_dir / f"v{version_label}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{client_id}_{stamp}_r{round_idx}.pt"
        dest_path.write_bytes(weights_bytes)

        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO client_updates (
                    client_id, base_version, round, samples, weights_path, received_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'received')
                """,
                (client_id, version_label, round_idx, samples, str(dest_path), now),
            )
            conn.execute(
                "UPDATE clients SET last_seen = ?, last_version = ? WHERE client_id = ?",
                (now, version_label, client_id),
            )

            # Sync with dashboard_updates table if present
            try:
                conn.execute(
                    """
                    INSERT INTO dashboard_updates (client_id, version_from, update_size, status, created_at)
                    VALUES (?, ?, ?, 'pending', ?)
                    """,
                    (client_id, version_label, len(weights_bytes), now),
                )
                conn.execute(
                    "UPDATE dashboard_clients SET upload_count = upload_count + 1, last_seen = ? WHERE client_id = ?",
                    (now, client_id),
                )
            except Exception:
                pass

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
