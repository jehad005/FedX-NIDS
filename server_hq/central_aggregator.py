"""SecureFed-NIDS Central Aggregator
===================================
Robust Federated Aggregation Engine supporting:
- Federated Averaging (FedAvg with weighted sample counts)
- Coordinate-wise Trimmed Mean (Byzantine-robust against extreme gradient poisoning)
- Coordinate-wise Median (robust against high-magnitude adversarial noise)
- Krum & Multi-Krum (geometric distance-based Byzantine fault tolerance)

Handles both:
- DP-noised Parameter Deltas (ΔW = W_client - W_base)
- Full client weight dictionaries
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("securefed.aggregator")


def _flatten_tensors(tensors_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a dictionary of parameter tensors into a single 1D vector."""
    flat = []
    for k in sorted(tensors_dict.keys()):
        t = tensors_dict[k]
        if t.is_floating_point():
            flat.append(t.detach().cpu().flatten())
    return torch.cat(flat) if flat else torch.empty(0)


class CentralAggregator:
    def __init__(self, global_model: Optional[nn.Module] = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.global_model = global_model
        if self.global_model is not None:
            self.global_model.to(self.device)

    # -------------------------------------------------------------------------
    # Core Aggregation Algorithms
    # -------------------------------------------------------------------------

    def aggregate_fedavg(
        self,
        updates: List[Dict[str, torch.Tensor]],
        sample_counts: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Federated Averaging (FedAvg) weighted by client sample counts."""
        if not updates:
            raise ValueError("No updates provided for FedAvg aggregation.")

        n_clients = len(updates)
        if sample_counts is None or sum(sample_counts) == 0:
            weights = [1.0 / n_clients] * n_clients
        else:
            total_samples = float(sum(sample_counts))
            weights = [s / total_samples for s in sample_counts]

        aggregated: Dict[str, torch.Tensor] = {}
        first = updates[0]

        for key, tensor in first.items():
            if tensor.is_floating_point():
                acc = torch.zeros_like(tensor, dtype=tensor.dtype, device=self.device)
                for w, upd in zip(weights, updates):
                    if key in upd:
                        acc.add_(upd[key].to(self.device), alpha=w)
                aggregated[key] = acc.cpu()
            else:
                # Non-floating buffers (e.g. batchnorm num_batches_tracked)
                aggregated[key] = tensor.cpu().clone()

        return aggregated

    def aggregate_trimmed_mean(
        self,
        updates: List[Dict[str, torch.Tensor]],
        trim_ratio: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Coordinate-wise Trimmed Mean.

        Trims the top and bottom `trim_ratio` fraction of client values along each coordinate.
        Robust against Byzantine / model poisoning attacks.
        """
        if not updates:
            raise ValueError("No updates provided for Trimmed Mean aggregation.")

        n_clients = len(updates)
        k_trim = int(n_clients * trim_ratio)
        if k_trim * 2 >= n_clients:
            k_trim = max(0, (n_clients - 1) // 2)

        aggregated: Dict[str, torch.Tensor] = {}
        first = updates[0]

        for key, tensor in first.items():
            if tensor.is_floating_point():
                # Stack all clients along dim 0: shape [n_clients, *param_shape]
                stacked = torch.stack([upd[key].to(self.device) for upd in updates], dim=0)
                sorted_vals, _ = torch.sort(stacked, dim=0)
                if k_trim > 0:
                    trimmed = sorted_vals[k_trim : n_clients - k_trim]
                else:
                    trimmed = sorted_vals
                aggregated[key] = torch.mean(trimmed, dim=0).cpu()
            else:
                aggregated[key] = tensor.cpu().clone()

        return aggregated

    def aggregate_median(
        self,
        updates: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Coordinate-wise Median aggregation across all client updates."""
        if not updates:
            raise ValueError("No updates provided for Median aggregation.")

        aggregated: Dict[str, torch.Tensor] = {}
        first = updates[0]

        for key, tensor in first.items():
            if tensor.is_floating_point():
                stacked = torch.stack([upd[key].to(self.device) for upd in updates], dim=0)
                med = torch.median(stacked, dim=0).values
                aggregated[key] = med.cpu()
            else:
                aggregated[key] = tensor.cpu().clone()

        return aggregated

    def aggregate_krum(
        self,
        updates: List[Dict[str, torch.Tensor]],
        num_byzantine: int = 1,
        multi_krum: bool = False,
        top_m: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Krum / Multi-Krum Byzantine-resilient aggregation.

        Selects the client update closest to its m - f - 2 nearest neighbors in Euclidean distance.
        """
        n_clients = len(updates)
        if n_clients <= 2:
            return self.aggregate_fedavg(updates)

        f = max(0, min(num_byzantine, (n_clients - 3) // 2))
        closest_count = max(1, n_clients - f - 2)

        # Flatten updates to 1D vectors
        flat_vectors = [_flatten_tensors(upd) for upd in updates]

        # Compute pairwise squared Euclidean distances
        distances = torch.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                d = torch.sum((flat_vectors[i] - flat_vectors[j]) ** 2).item()
                distances[i, j] = d
                distances[j, i] = d

        # Compute Krum score for each client (sum of distances to closest neighbors)
        scores = []
        for i in range(n_clients):
            row = distances[i].clone()
            row[i] = float("inf")  # exclude self
            sorted_dist, _ = torch.sort(row)
            score = torch.sum(sorted_dist[:closest_count]).item()
            scores.append(score)

        if not multi_krum:
            best_idx = int(np.argmin(scores))
            return {k: v.cpu().clone() for k, v in updates[best_idx].items()}

        # Multi-Krum: average top-m candidates
        m = top_m or max(1, n_clients - f)
        m = min(m, n_clients)
        top_indices = np.argsort(scores)[:m]
        selected_updates = [updates[i] for i in top_indices]
        return self.aggregate_fedavg(selected_updates)

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        sample_counts: Optional[List[int]] = None,
        method: str = "fedavg",
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Dispatch aggregation to the requested method."""
        method_norm = method.lower().strip().replace("-", "_").replace(" ", "_")
        if method_norm in {"fedavg", "average", "avg"}:
            return self.aggregate_fedavg(updates, sample_counts=sample_counts)
        elif method_norm in {"trimmed_mean", "trimmedmean", "tm"}:
            trim_ratio = float(kwargs.get("trim_ratio", 0.1))
            return self.aggregate_trimmed_mean(updates, trim_ratio=trim_ratio)
        elif method_norm in {"median", "coord_median", "coordinate_median"}:
            return self.aggregate_median(updates)
        elif method_norm in {"krum"}:
            f = int(kwargs.get("num_byzantine", 1))
            return self.aggregate_krum(updates, num_byzantine=f, multi_krum=False)
        elif method_norm in {"multi_krum", "multikrum"}:
            f = int(kwargs.get("num_byzantine", 1))
            top_m = kwargs.get("top_m")
            return self.aggregate_krum(updates, num_byzantine=f, multi_krum=True, top_m=top_m)
        else:
            logger.warning(f"Unknown aggregation method '{method}', falling back to FedAvg.")
            return self.aggregate_fedavg(updates, sample_counts=sample_counts)

    # -------------------------------------------------------------------------
    # Apply to Model
    # -------------------------------------------------------------------------

    def apply_updates(self, aggregated_weights: Dict[str, torch.Tensor]) -> None:
        """Apply aggregated weights to the in-memory global model."""
        if self.global_model is None:
            raise ValueError("No global_model initialized in CentralAggregator.")
        self.global_model.load_state_dict(aggregated_weights)

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Return clone of current global model state dict."""
        if self.global_model is None:
            raise ValueError("No global_model initialized in CentralAggregator.")
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}


# =============================================================================
# High-Level Aggregation Pipeline Manager
# =============================================================================

def load_client_update_file(file_path: Path) -> Tuple[Dict[str, torch.Tensor], int, str, bool]:
    """Load a client update file.

    Returns:
        (param_dict, sample_count, client_id, is_delta)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Update file not found: {file_path}")

    try:
        payload = torch.load(file_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Could not load update file {file_path}: {exc}") from exc

    if isinstance(payload, dict) and payload.get("format") == "securefed_dp_delta_v1":
        delta_dict = payload["delta_state_dict"]
        sample_count = int(payload.get("samples", 1))
        client_id = str(payload.get("client_id", "unknown"))
        return delta_dict, sample_count, client_id, True

    if isinstance(payload, dict):
        if "state_dict" in payload:
            return payload["state_dict"], int(payload.get("samples", 1)), str(payload.get("client_id", "unknown")), False
        elif "delta_state_dict" in payload:
            return payload["delta_state_dict"], int(payload.get("samples", 1)), str(payload.get("client_id", "unknown")), True
        else:
            # Raw state dict
            return payload, 1, "client", False

    raise ValueError(f"Unrecognized update payload format in {file_path}")


def run_federated_round(
    registry,
    base_version: str = "latest",
    method: str = "fedavg",
    notes: str = "",
    min_updates: int = 1,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Collect pending updates, aggregate them with the base model, and publish a new version.

    Steps:
    1. Query all pending updates for `base_version` (or all received updates).
    2. Load base weights from HQ registry.
    3. Aggregate deltas/weights using the chosen robust algorithm.
    4. Construct new model state_dict: W_new = W_base + Delta_agg (or Avg(W)).
    5. Save new weights and register a new version (e.g. v1.0.5).
    6. Mark processed updates as 'aggregated'.
    """
    record = registry.get_version(base_version)
    if record is None:
        raise ValueError(f"Base model version '{base_version}' not found in registry.")

    base_ver = record["version"]
    base_weights_path = Path(record["weights_path"])
    if not base_weights_path.exists():
        raise FileNotFoundError(f"Base weights file missing on disk: {base_weights_path}")

    # 1. Fetch pending updates
    with registry._connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM client_updates
            WHERE (base_version = ? OR base_version = 'latest') AND status = 'received'
            ORDER BY id ASC
            """,
            (base_ver,),
        ).fetchall()

    if len(rows) < min_updates:
        raise ValueError(
            f"Insufficient pending updates for version {base_ver}. Found {len(rows)}, minimum required is {min_updates}."
        )

    # 2. Load base state
    base_raw = torch.load(base_weights_path, map_location="cpu", weights_only=False)
    if isinstance(base_raw, dict) and "model_state_dict" in base_raw:
        base_state = base_raw["model_state_dict"]
    elif hasattr(base_raw, "state_dict"):
        base_state = base_raw.state_dict()
    elif isinstance(base_raw, dict):
        base_state = base_raw
    else:
        raise ValueError("Could not extract state_dict from base weights file.")

    # 3. Load all client updates
    update_items: List[Dict[str, torch.Tensor]] = []
    sample_counts: List[int] = []
    client_ids: List[str] = []
    is_delta_list: List[bool] = []
    update_ids: List[int] = []

    for r in rows:
        upd_path = Path(r["weights_path"])
        if not upd_path.exists():
            continue
        try:
            param_dict, samples, c_id, is_delta = load_client_update_file(upd_path)
            # Ensure float tensors
            clean_dict = {
                k: v.clone().detach().to(torch.float32) if v.is_floating_point() else v.clone().detach()
                for k, v in param_dict.items()
            }
            update_items.append(clean_dict)
            sample_counts.append(samples)
            client_ids.append(c_id or r["client_id"])
            is_delta_list.append(is_delta)
            update_ids.append(r["id"])
        except Exception as exc:
            logger.warning(f"Skipping corrupt update {upd_path}: {exc}")

    if not update_items:
        raise ValueError("No valid client update files could be loaded.")

    # Normalize to deltas: if any update is full weights, compute delta = W_client - W_base
    delta_updates: List[Dict[str, torch.Tensor]] = []
    for upd, is_d in zip(update_items, is_delta_list):
        if is_d:
            delta_updates.append(upd)
        else:
            delta = {}
            for k in upd:
                if k in base_state and upd[k].is_floating_point():
                    delta[k] = upd[k] - base_state[k].to(upd[k].device)
                else:
                    delta[k] = upd[k]
            delta_updates.append(delta)

    # 4. Aggregate
    aggregator = CentralAggregator()
    aggregated_delta = aggregator.aggregate(
        updates=delta_updates,
        sample_counts=sample_counts,
        method=method,
        **kwargs,
    )

    # 5. Construct updated state_dict
    new_state_dict: Dict[str, torch.Tensor] = {}
    for k, v in base_state.items():
        if k in aggregated_delta and v.is_floating_point():
            new_state_dict[k] = v.to(torch.float32) + aggregated_delta[k].to(torch.float32)
        else:
            new_state_dict[k] = v.clone()

    # 6. Save weights to a staging location and publish
    next_ver = registry.next_version()
    staging_dir = registry.models_dir / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_weights = staging_dir / f"fed_round_{next_ver}.pt"

    # If base was full model or dict container, preserve format
    if isinstance(base_raw, dict) and "model_state_dict" in base_raw:
        new_package = copy.deepcopy(base_raw)
        new_package["model_state_dict"] = new_state_dict
        new_package["version"] = next_ver
        torch.save(new_package, staged_weights)
    elif hasattr(base_raw, "load_state_dict"):
        new_model = copy.deepcopy(base_raw)
        new_model.load_state_dict(new_state_dict)
        torch.save(new_model, staged_weights)
    else:
        torch.save(new_state_dict, staged_weights)

    # Preserve feature schema & scaler stats directly from base version's meta.json
    base_meta = {}
    try:
        base_meta_path = registry.version_dir(base_ver) / "meta.json"
        if base_meta_path.exists():
            base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
    except Exception:
        pass

    base_features = base_meta.get("feature_names") or record.get("feature_names") or []
    if not base_features:
        try:
            baseline_meta_path = registry.models_dir / "v1.0.4" / "meta.json"
            if baseline_meta_path.exists():
                base_features = json.loads(baseline_meta_path.read_text()).get("feature_names", [])
        except Exception:
            pass

    meta_info = {
        "model_name": base_meta.get("model_name", record.get("model_name", "SecureFed-Global-Ensemble")),
        "architecture": base_meta.get("architecture", record.get("architecture", "EnsembleTop2")),
        "fl_method": method,
        "input_dim": base_meta.get("input_dim", record.get("input_dim", len(base_features) or 83)),
        "num_classes": base_meta.get("num_classes", record.get("num_classes", 2)),
        "feature_names": base_features,
        "scaler_mean": base_meta.get("scaler_mean"),
        "scaler_scale": base_meta.get("scaler_scale"),
        "accuracy": record.get("accuracy", 0.0),
        "aggregated_from_version": base_ver,
        "num_clients_aggregated": len(update_items),
        "client_ids": client_ids,
        "total_samples": sum(sample_counts),
        "aggregation_method": method,
    }

    round_notes = notes or f"Federated Round {next_ver}: aggregated {len(update_items)} updates via {method.upper()}."
    published = registry.publish(
        weights_src=staged_weights,
        package_src=None,
        version=next_ver,
        metadata=meta_info,
        notes=round_notes,
    )

    # 7. Update status in database
    now = datetime.now(timezone.utc).isoformat()
    with registry._connect() as conn:
        for u_id in update_ids:
            conn.execute(
                "UPDATE client_updates SET status = 'aggregated' WHERE id = ?",
                (u_id,),
            )
        try:
            conn.execute(
                "UPDATE dashboard_updates SET status = 'processed', processed_at = ?, version_to = ? WHERE status = 'pending' AND version_from = ?",
                (now, next_ver, base_ver),
            )
        except Exception:
            pass

    return {
        "status": "success",
        "new_version": published["version"],
        "base_version": base_ver,
        "method": method,
        "num_updates_merged": len(update_items),
        "total_samples": sum(sample_counts),
        "client_ids": client_ids,
        "weights_path": published["weights_path"],
        "created_at": published["created_at"],
    }
