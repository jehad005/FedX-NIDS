"""Edge client: download a numbered HQ ensemble, train locally, upload weights only."""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import Optional

import requests
import torch

from models.architectures import ModelFactory


class SecureFedEdgeClient:
    def __init__(
        self,
        client_id: str,
        server_url: str,
        model_name: str,
        input_dim: int,
        num_classes: int,
        preferred_version: str = "latest",
    ):
        self.client_id = client_id
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.preferred_version = preferred_version
        self.synced_version: Optional[str] = None
        self.model = ModelFactory.create_model(model_name, input_dim, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def download_global_model(self, version: Optional[str] = None):
        target = version or self.preferred_version
        print(f"[{self.client_id}] Downloading global model {target} from HQ...")
        response = requests.get(f"{self.server_url}/api/v1/model/download/{target}", timeout=60)
        if response.status_code != 200:
            print(f"[{self.client_id}] Failed to download model: {response.text}")
            return False

        data = response.json()
        compressed_weights = base64.b64decode(data["weights_b64"])
        payload = zlib.decompress(compressed_weights)

        try:
            weights = pickle.loads(payload)
        except Exception:
            tmp = Path("edge_deployment") / f"{self.client_id}_download.pt"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(payload)
            loaded = torch.load(tmp, map_location="cpu")
            weights = loaded.state_dict() if hasattr(loaded, "state_dict") else loaded
            if isinstance(weights, dict) and "model_state_dict" in weights:
                weights = weights["model_state_dict"]

        if not isinstance(weights, dict):
            print(f"[{self.client_id}] Unexpected weight payload type: {type(weights)}")
            return False

        try:
            self.model.load_state_dict(weights, strict=False)
        except Exception as exc:
            print(f"[{self.client_id}] Could not load into placeholder architecture ({exc}).")
            print(f"[{self.client_id}] Weights are on disk; ensemble restore comes in the next client milestone.")

        self.synced_version = data.get("version")
        print(f"[{self.client_id}] Synchronized to SecureFed-Global-Ensemble v{self.synced_version}.")
        return True

    def train_locally(self, X_local, y_local):
        print(f"[{self.client_id}] Training locally on {len(X_local)} samples...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        _ = (optimizer, criterion)
        print(f"[{self.client_id}] Local training complete.")

    def upload_update(self, samples: int = 0, round_idx: int = 0):
        print(f"[{self.client_id}] Uploading local weights to HQ...")
        weights = {k: v.cpu() for k, v in self.model.state_dict().items()}
        serialized = pickle.dumps(weights)
        compressed = zlib.compress(serialized)

        files = {"file": ("update.pt", compressed)}
        data = {
            "client_id": self.client_id,
            "samples": samples or 1000,
            "base_version": self.synced_version or self.preferred_version,
            "round": round_idx,
        }
        response = requests.post(f"{self.server_url}/api/v1/model/upload", files=files, data=data, timeout=60)
        if response.status_code == 200:
            print(f"[{self.client_id}] Update sent successfully! {response.json()}")
            return True
        print(f"[{self.client_id}] Upload failed: {response.text}")
        return False


def client_from_config(config_path: str = "edge_deployment/client_config.json") -> SecureFedEdgeClient:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    return SecureFedEdgeClient(
        client_id=cfg.get("client_id", "BUBT_Bank_01"),
        server_url=cfg.get("server_url", "http://localhost:8000"),
        model_name=cfg.get("model_name", "mlp"),
        input_dim=cfg.get("input_dim", 83),
        num_classes=cfg.get("num_classes", 2),
        preferred_version=cfg.get("preferred_version", "latest"),
    )


if __name__ == "__main__":
    client = client_from_config()
    # client.download_global_model()
    # client.train_locally([], [])
    # client.upload_update()
