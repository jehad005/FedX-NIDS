import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import torch
from models.architectures import MLP
from models.ensemble import EnsembleModel
from server_hq.registry import ModelRegistry
from server_hq.central_aggregator import (
    run_federated_round,
    validate_client_update_payload,
)

print("=" * 60)
print("TESTING END-TO-END FEDERATED UPDATE & AGGREGATION FLOW")
print("=" * 60)

with tempfile.TemporaryDirectory(prefix="securefed_e2e_") as tmp_root_str:
    tmp_root = Path(tmp_root_str)
    registry = ModelRegistry(
        db_path=tmp_root / "test_registry.db",
        store_dir=tmp_root / "hq_store",
    )

    # Bootstrap an isolated v1.0.0 global ensemble so tests never touch live DB/weights
    input_dim = 16
    num_classes = 2
    base_ensemble = EnsembleModel(
        [MLP(input_dim, num_classes), MLP(input_dim, num_classes)],
        method="voting",
        voting_type="soft",
        num_classes=num_classes,
    )
    fnames = [f"f_{i}" for i in range(input_dim)]
    boot = registry.publish_ensemble(
        ensemble_model=base_ensemble,
        input_dim=input_dim,
        num_classes=num_classes,
        feature_names=fnames,
        accuracy=0.92,
        notes="isolated E2E bootstrap",
    )
    X_hold = np.random.randn(120, input_dim).astype(np.float32)
    y_hold = (X_hold[:, 0] > 0).astype(np.int64)
    registry.save_eval_holdout(X_hold, y_hold, input_dim, fnames, dataset_name="e2e_holdout.csv")
    latest_ver = boot["version"]
    open_round = registry.get_open_round()
    print(f"Bootstrapped Version: v{latest_ver} | Open Round: {open_round}")
    record = registry.get_version(latest_ver)
    assert record is not None, "Latest version must exist"

    base_raw = torch.load(record["weights_path"], map_location="cpu", weights_only=False)
    base_state = base_raw.state_dict() if hasattr(base_raw, "state_dict") else base_raw

    def build_and_upload_delta(client_id: str, scale: float, samples: int, round_num: int):
        delta = {
            k: (torch.randn_like(v) * scale if v.is_floating_point() else torch.zeros_like(v))
            for k, v in base_state.items()
        }
        payload = {
            "format": "securefed_dp_delta_v1",
            "base_version": latest_ver,
            "client_id": client_id,
            "round": round_num,
            "samples": samples,
            "local_steps": 5,
            "clipping_norm": 1.0,
            "noise_multiplier": 1.1,
            "clipped": True,
            "noised": True,
            "delta_state_dict": delta,
        }
        pt_path = tmp_root / f"{client_id}_r{round_num}.pt"
        torch.save(payload, pt_path)
        raw_bytes = pt_path.read_bytes()
        validated = validate_client_update_payload(
            raw_bytes=raw_bytes,
            filename=pt_path.name,
            client_id=client_id,
            samples=samples,
            base_version=latest_ver,
            round_idx=round_num,
            expected_base_version=latest_ver,
            expected_round=registry.get_open_round(),
            base_state_dict=base_state,
        )
        return registry.save_update(
            client_id=validated["client_id"],
            weights_bytes=raw_bytes,
            samples=validated["samples"],
            base_version=validated["base_version"],
            round_idx=validated["round"],
            clipping_norm=validated["clipping_norm"],
            noise_multiplier=validated["noise_multiplier"],
            local_steps=validated["local_steps"],
        )

    print(f"\n1. Simulating Client 1 DP delta upload (Round {open_round})...")
    upd1 = build_and_upload_delta("client_edge_alpha", 0.01, 250, open_round)
    print(f"-> Client 1 update saved (id={upd1['id']})")

    print(f"\n2. Simulating Client 2 DP delta upload (Round {open_round})...")
    upd2 = build_and_upload_delta("client_edge_beta", 0.015, 300, open_round)
    print(f"-> Client 2 update saved (id={upd2['id']})")

    pending = registry.list_updates(version=latest_ver)
    print(f"\n3. Total updates recorded for base v{latest_ver}: {len(pending)}")
    assert len(pending) == 2

    print("\n4. Running Federated Aggregation (FedAvg)...")
    res = run_federated_round(
        registry=registry,
        base_version=latest_ver,
        method="fedavg",
        notes="E2E isolated multi-client aggregation",
    )

    new_version = res["new_version"]
    print(f"\nSUCCESS! Published new global model version: v{new_version}")
    print(f"Merged {res['num_updates_merged']} client updates with total {res['total_samples']} samples.")
    print(f"Next Open Round: {res['next_open_round']}")

    new_record = registry.get_version(new_version)
    assert new_record is not None
    assert new_record["is_latest"] == 1
    assert Path(new_record["weights_path"]).exists()
    assert registry.get_open_round() == open_round + 1
    ev = res.get("evaluation") or {}
    assert ev.get("evaluated") is True and ev.get("eval_samples") == 120
    assert res.get("accuracy") == ev["accuracy"] and res.get("f1") is not None and res.get("auc") is not None
    assert abs(float(new_record["accuracy"]) - float(ev["accuracy"])) < 1e-8
    meta = json.loads((registry.version_dir(new_version) / "meta.json").read_text(encoding="utf-8"))
    assert meta["post_aggregation_evaluation"]["evaluated"] is True
    print(f"Post-aggregation Eval Verified: Acc={res['accuracy']*100:.2f}% | F1={res['f1']:.4f} | AUC={res['auc']:.4f}")

    print("\n" + "=" * 60)
    print("ALL END-TO-END FLOW TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
