"""Smoke test for HQ versioned model serving (no live HTTP server required)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server_hq.registry import ModelRegistry


def run() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="securefed_hq_"))
    db = tmp / "hq.db"
    store = tmp / "hq_store"
    registry = ModelRegistry(db_path=db, store_dir=store)

    # Use plain bytes instead of torch tensors so test works without torch installed
    weights_a = tmp / "w1.pt"
    weights_a.write_bytes(b"weights-for-v1.0.0")
    pkg_a = tmp / "p1.pt"
    pkg_a.write_bytes(json.dumps({
        "model_name": "SecureFed-Global-Ensemble",
        "architecture": "EnsembleTop2",
        "best_fl_method": "fedavg",
        "input_dim": 83,
        "num_classes": 2,
        "accuracy": 0.997,
    }).encode())

    v1 = registry.publish(
        weights_src=weights_a,
        package_src=pkg_a,
        version="1.0.0",
        metadata={
            "model_name": "SecureFed-Global-Ensemble",
            "architecture": "EnsembleTop2",
            "fl_method": "fedavg",
            "input_dim": 83,
            "num_classes": 2,
            "accuracy": 0.997,
        },
        notes="initial release",
    )
    assert v1["version"] == "1.0.0"
    assert v1["is_latest"] == 1
    assert Path(v1["weights_path"]).exists()
    assert Path(v1["package_path"]).exists()
    meta = json.loads((store / "models" / "v1.0.0" / "meta.json").read_text())
    assert meta["version"] == "1.0.0"

    weights_b = tmp / "w2.pt"
    weights_b.write_bytes(b"weights-for-v1.0.1")
    v2 = registry.publish(
        weights_src=weights_b,
        metadata={"model_name": "SecureFed-Global-Ensemble", "fl_method": "moon", "accuracy": 0.998},
        notes="after moon round",
    )
    assert v2["version"] == "1.0.1"
    assert registry.latest_version() == "1.0.1"
    assert registry.get_version("latest")["version"] == "1.0.1"
    assert registry.get_version("v1.0.0")["is_latest"] == 0

    try:
        registry.publish(weights_src=weights_b, version="1.0.0")
        raise AssertionError("duplicate version should fail")
    except ValueError:
        pass

    update = registry.save_update(
        client_id="BUBT_Bank_01",
        weights_bytes=b"fake-client-update-weights",
        samples=1200,
        base_version="latest",
        round_idx=3,
    )
    assert update["client_id"] == "BUBT_Bank_01"
    assert update["base_version"] == "1.0.1"
    assert update["samples"] == 1200
    assert Path(update["weights_path"]).exists()

    clients = registry.list_clients()
    assert any(c["client_id"] == "BUBT_Bank_01" for c in clients)
    updates = registry.list_updates(client_id="BUBT_Bank_01")
    assert len(updates) == 1

    versions = registry.list_versions()
    assert [v["version"] for v in versions] == ["1.0.1", "1.0.0"]

    print("HQ registry smoke test passed.")
    print(f"  versions : {[v['version'] for v in versions]}")
    print(f"  latest   : {registry.latest_version()}")
    print(f"  clients  : {[c['client_id'] for c in clients]}")
    print(f"  update   : {update['weights_path']}")


if __name__ == "__main__":
    run()