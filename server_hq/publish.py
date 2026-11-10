"""Publish a trained global ensemble as a numbered HQ model version.

Usage:
    python -m server_hq.publish
    python -m server_hq.publish --version 1.0.0 --notes "first public release"
    python -m server_hq.publish --weights models/securefed_global_ensemble.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server_hq.registry import ModelRegistry


def load_package_metadata(package_path: Path) -> dict:
    if not package_path.exists():
        return {}
    try:
        import torch
        pkg = torch.load(package_path, map_location="cpu", weights_only=False)
    except TypeError:
        import torch
        pkg = torch.load(package_path, map_location="cpu")
    if not isinstance(pkg, dict):
        return {}
    return {
        "model_name": pkg.get("model_name", "SecureFed-Global-Ensemble"),
        "architecture": pkg.get("architecture", "EnsembleTop2"),
        "fl_method": pkg.get("best_fl_method"),
        "input_dim": pkg.get("input_dim"),
        "num_classes": pkg.get("num_classes"),
        "accuracy": pkg.get("accuracy"),
    }


def publish_from_artifacts(
    weights: Path,
    package: Path,
    version: str | None = None,
    notes: str = "",
) -> dict:
    registry = ModelRegistry()
    metadata = load_package_metadata(package)
    record = registry.publish(
        weights_src=weights,
        package_src=package if package.exists() else None,
        version=version,
        metadata=metadata,
        notes=notes,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish SecureFed-Global-Ensemble to HQ registry")
    parser.add_argument("--weights", default=str(ROOT / "models" / "securefed_global_ensemble.pt"))
    parser.add_argument("--package", default=str(ROOT / "edge_deployment" / "securefed_edge_model.pt"))
    parser.add_argument("--version", default=None, help="Semver like 1.0.0. Auto-bumps patch if omitted.")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    weights = Path(args.weights)
    package = Path(args.package)
    if not weights.exists():
        print(f"[!] Weights not found: {weights}")
        print("    Run the training pipeline first (python main.py --mode all).")
        sys.exit(1)

    record = publish_from_artifacts(weights, package, version=args.version, notes=args.notes)
    print("[HQ] Published SecureFed-Global-Ensemble")
    print(json.dumps({k: record[k] for k in record if k != "id"}, indent=2, default=str))


if __name__ == "__main__":
    main()
