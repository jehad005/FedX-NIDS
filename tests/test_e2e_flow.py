import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from server_hq.registry import ModelRegistry
from server_hq.central_aggregator import run_federated_round

print("=" * 60)
print("TESTING END-TO-END FEDERATED UPDATE & AGGREGATION FLOW")
print("=" * 60)

registry = ModelRegistry()
latest_ver = registry.latest_version()
print(f"Current Latest Version: v{latest_ver}")
record = registry.get_version(latest_ver)
assert record is not None, "Latest version must exist"

# 1. Simulate Client 1 uploading a DP delta update
print("\n1. Simulating Client 1 (client_edge_alpha) delta upload...")
base_raw = torch.load(record["weights_path"], map_location="cpu", weights_only=False)
if hasattr(base_raw, "state_dict"):
    base_state = base_raw.state_dict()
elif isinstance(base_raw, dict) and "model_state_dict" in base_raw:
    base_state = base_raw["model_state_dict"]
else:
    base_state = base_raw

# Create small synthetic delta
delta1 = {k: torch.randn_like(v) * 0.01 if v.is_floating_point() else v.clone() for k, v in base_state.items()}
client1_payload = {
    "format": "securefed_dp_delta_v1",
    "base_version": latest_ver,
    "client_id": "client_edge_alpha",
    "samples": 250,
    "clipping_norm": 1.0,
    "noise_multiplier": 1.1,
    "delta_state_dict": delta1
}
import tempfile

temp_dir = Path(tempfile.mkdtemp(prefix="securefed_test_"))
p1 = temp_dir / "client1_delta.pt"
p2 = temp_dir / "client2_delta.pt"

torch.save(client1_payload, p1)
upd1 = registry.save_update(
    client_id="client_edge_alpha",
    weights_bytes=p1.read_bytes(),
    samples=250,
    base_version=latest_ver,
    round_idx=1
)
print(f"-> Client 1 update saved to DB (id={upd1['id']})")

# 2. Simulate Client 2 uploading a DP delta update
print("\n2. Simulating Client 2 (client_edge_beta) delta upload...")
delta2 = {k: torch.randn_like(v) * 0.015 if v.is_floating_point() else v.clone() for k, v in base_state.items()}
client2_payload = {
    "format": "securefed_dp_delta_v1",
    "base_version": latest_ver,
    "client_id": "client_edge_beta",
    "samples": 300,
    "clipping_norm": 1.0,
    "noise_multiplier": 1.1,
    "delta_state_dict": delta2
}
torch.save(client2_payload, p2)

upd2 = registry.save_update(
    client_id="client_edge_beta",
    weights_bytes=p2.read_bytes(),
    samples=300,
    base_version=latest_ver,
    round_idx=1
)
print(f"-> Client 2 update saved to DB (id={upd2['id']})")

# 3. Check pending updates in registry
pending = registry.list_updates(version=latest_ver)
print(f"\n3. Total updates recorded for base v{latest_ver}: {len(pending)}")

# 4. Trigger Federated Aggregation via FedAvg
print("\n4. Running Federated Aggregation (FedAvg)...")
res = run_federated_round(
    registry=registry,
    base_version=latest_ver,
    method="fedavg",
    notes="E2E test multi-client aggregation"
)

new_version = res["new_version"]
print(f"\nSUCCESS! Published new global model version: v{new_version}")
print(f"Merged {res['num_updates_merged']} client updates with total {res['total_samples']} samples.")
print(f"Client IDs: {res['client_ids']}")

# 5. Verify the new version in registry
new_record = registry.get_version(new_version)
assert new_record is not None
assert new_record["is_latest"] == 1
assert Path(new_record["weights_path"]).exists()
print(f"Verified new model on disk: {new_record['weights_path']}")
print(f"Latest Version now: v{registry.latest_version()}")

print("\n" + "=" * 60)
print("ALL END-TO-END FLOW TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
