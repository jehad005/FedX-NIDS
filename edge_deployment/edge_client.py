"""Edge client: download a numbered HQ ensemble, train locally, upload weights only."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
import torch.nn as nn

from dp.differential_privacy import DPSGD
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
        self.open_round: int = 1
        self.base_state: Optional[dict] = None
        self.clipping_norm: float = 1.0
        self.noise_multiplier: float = 1.1
        self.local_steps: int = 1
        self.last_samples: int = 0
        self.model = ModelFactory.create_model(model_name, input_dim, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def download_global_model(self, version: Optional[str] = None):
        import zlib
        target = version or self.preferred_version
        print(f"[{self.client_id}] Downloading global model {target} from HQ...")
        response = requests.get(
            f"{self.server_url}/api/v1/model/download/{target}",
            headers={"X-Client-ID": self.client_id},
            timeout=60,
        )
        if response.status_code != 200:
            print(f"[{self.client_id}] Failed to download model: {response.text}")
            return False

        data = response.json()
        compressed_weights = base64.b64decode(data["weights_b64"])
        payload = zlib.decompress(compressed_weights)

        tmp = Path("edge_deployment") / f"{self.client_id}_download.pt"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(payload)

        try:
            loaded = torch.load(tmp, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(tmp, map_location="cpu")

        if isinstance(loaded, nn.Module):
            self.model = loaded.to(self.device)
        else:
            weights = loaded.state_dict() if hasattr(loaded, "state_dict") else loaded
            if isinstance(weights, dict) and "model_state_dict" in weights:
                weights = weights["model_state_dict"]
            if not isinstance(weights, dict):
                print(f"[{self.client_id}] Unexpected weight payload type: {type(weights)}")
                return False
            self.model.load_state_dict(weights, strict=False)

        self.base_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.synced_version = data.get("version")
        self.open_round = int(data.get("open_round", 1) or 1)
        print(f"[{self.client_id}] Synchronized to SecureFed-Global-Ensemble v{self.synced_version} (open round {self.open_round}).")
        return True

    def train_locally(
        self,
        X_local,
        y_local,
        epochs: int = 1,
        batch_size: int = 64,
        lr: float = 0.001,
        clipping_norm: float = 1.0,
        noise_multiplier: float = 1.1,
    ):
        X_arr = np.asarray(X_local, dtype=np.float32)
        y_arr = np.asarray(y_local, dtype=np.int64)
        if len(X_arr) == 0:
            raise ValueError("Cannot train on empty local dataset.")

        self.clipping_norm = float(clipping_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.last_samples = len(X_arr)

        if self.base_state is None:
            self.base_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        print(f"[{self.client_id}] DP-SGD training locally on {len(X_arr)} samples...")
        dp_sgd = DPSGD(self.model, {
            "clipping_norm": self.clipping_norm,
            "noise_multiplier": self.noise_multiplier,
            "delta": 1e-5,
        })
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()

        steps = 0
        for _ in range(max(1, int(epochs))):
            perm = np.random.permutation(len(X_arr))
            for i in range(0, len(X_arr), max(1, int(batch_size))):
                idx = perm[i:i + batch_size]
                bx = torch.FloatTensor(X_arr[idx]).to(self.device)
                by = torch.LongTensor(y_arr[idx]).to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                dp_sgd.clip_gradients()
                dp_sgd.add_noise()
                optimizer.step()
                steps += 1
        self.local_steps = max(1, steps)
        print(f"[{self.client_id}] Local DP training complete ({self.local_steps} steps).")

    def upload_update(self, samples: int = 0, round_idx: Optional[int] = None):
        if self.base_state is None:
            raise ValueError("No base_state available; download or train a model first.")
        eff_samples = int(samples or self.last_samples or 0)
        if eff_samples < 1:
            raise ValueError("Sample count must be >= 1.")
        eff_round = int(round_idx if round_idx is not None else self.open_round)

        print(f"[{self.client_id}] Computing clipped/noised DP delta for round {eff_round}...")
        current = self.model.state_dict()
        raw_deltas = {}
        sq_norm = 0.0
        for name, base_t in self.base_state.items():
            cur_t = current[name].detach().cpu()
            if torch.is_floating_point(cur_t):
                delta = cur_t - base_t
                raw_deltas[name] = delta
                sq_norm += float(torch.sum(delta.float() ** 2))
        total_norm = sq_norm ** 0.5
        scale = min(1.0, self.clipping_norm / max(total_norm, 1e-12))

        protected_delta = {}
        for name, base_t in self.base_state.items():
            if name in raw_deltas:
                clipped = raw_deltas[name] * scale
                noise = torch.normal(0.0, self.noise_multiplier * self.clipping_norm, size=clipped.shape)
                protected_delta[name] = clipped + noise
            else:
                protected_delta[name] = torch.zeros_like(base_t)

        payload = {
            "format": "securefed_dp_delta_v1",
            "base_version": self.synced_version or self.preferred_version,
            "client_id": self.client_id,
            "round": eff_round,
            "samples": eff_samples,
            "local_steps": self.local_steps,
            "clipping_norm": self.clipping_norm,
            "noise_multiplier": self.noise_multiplier,
            "clipped": True,
            "noised": True,
            "delta_state_dict": protected_delta,
        }
        buf = io.BytesIO()
        torch.save(payload, buf)

        files = {"file": ("dp_update.pt", buf.getvalue(), "application/octet-stream")}
        data = {
            "client_id": self.client_id,
            "samples": eff_samples,
            "base_version": self.synced_version or self.preferred_version,
            "round": eff_round,
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
