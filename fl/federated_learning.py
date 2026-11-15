"""
Federated Learning Module for SecureFed-NIDS
=============================================
Implements FedAvg, FedProx, SCAFFOLD, FedNova, and MOON.
Stage 4 runs all five independently from the same global ensemble
checkpoint, then the caller picks the best by validation accuracy.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from config import FL_CONFIG


FL_METHODS = ["fedavg", "fedprox", "scaffold", "fednova", "moon"]


def _rebalance_small_clients(
    client_indices: List[np.ndarray],
    min_required: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Borrow samples from largest clients if any client fell below min_required."""
    bins = [idx.tolist() for idx in client_indices]
    while True:
        sizes = [len(b) for b in bins]
        smallest = int(np.argmin(sizes))
        largest = int(np.argmax(sizes))
        if sizes[smallest] >= min_required or sizes[largest] <= min_required:
            break
        need = min(min_required - sizes[smallest], sizes[largest] - min_required)
        if need <= 0:
            break
        donor = bins[largest]
        bins[smallest].extend(donor[-need:])
        bins[largest] = donor[:-need]
    out = []
    for b in bins:
        arr = np.asarray(b, dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def dirichlet_partition_indices(
    y: np.ndarray,
    num_clients: int = 5,
    alpha: float = 0.3,
    min_samples: int = 10,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Partition sample indices across clients using a Dirichlet label distribution Dir(alpha).

    Lower alpha values (e.g. 0.3) create stronger Non-IID class and sample skew
    across clients while keeping client sets strictly disjoint.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if alpha <= 0:
        raise ValueError(f"Dirichlet alpha must be > 0, got {alpha}")

    y_arr = np.asarray(y)
    n_total = len(y_arr)
    if n_total < num_clients:
        raise ValueError(f"Need at least {num_clients} samples for {num_clients} clients, got {n_total}")
    if num_clients == 1:
        return [np.arange(n_total, dtype=np.int64)]

    rng = np.random.default_rng(seed)
    classes = np.unique(y_arr)
    effective_min = min(max(int(min_samples), 1), max(1, n_total // (num_clients * 2)))

    best_indices: Optional[List[np.ndarray]] = None
    for _ in range(50):
        client_bins: List[List[np.ndarray]] = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_idx = np.where(y_arr == cls)[0].copy()
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.full(num_clients, float(alpha)))
            split_points = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
            for cid, shard in enumerate(np.split(cls_idx, split_points)):
                if len(shard) > 0:
                    client_bins[cid].append(shard)

        candidate = [
            np.concatenate(bins).astype(np.int64) if bins else np.array([], dtype=np.int64)
            for bins in client_bins
        ]
        best_indices = candidate
        if min(len(idx) for idx in candidate) >= effective_min:
            for idx in candidate:
                rng.shuffle(idx)
            return candidate

    assert best_indices is not None
    return _rebalance_small_clients(best_indices, effective_min, rng)


def iid_partition_indices(
    num_samples: int,
    num_clients: int = 5,
    seed: int = 42,
) -> List[np.ndarray]:
    """Partition sample indices evenly across clients (IID baseline)."""
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if num_samples < num_clients:
        raise ValueError(f"Need at least {num_clients} samples, got {num_samples}")
    perm = np.random.default_rng(seed).permutation(num_samples).astype(np.int64)
    return [shard.astype(np.int64) for shard in np.array_split(perm, num_clients)]


def create_dirichlet_non_iid_splits(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int = 5,
    alpha: float = 0.3,
    min_samples: int = 10,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return Non-IID (X_k, y_k) per client partitioned with Dir(alpha)."""
    indices = dirichlet_partition_indices(
        y=y,
        num_clients=num_clients,
        alpha=alpha,
        min_samples=min_samples,
        seed=seed,
    )
    return [(X[idx], y[idx]) for idx in indices]


def partition_clients_data(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int = 5,
    method: str = "dirichlet",
    alpha: float = 0.3,
    min_samples: int = 10,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split (X, y) across clients using 'dirichlet' (Non-IID) or 'iid'."""
    norm = str(method or "dirichlet").strip().lower()
    if norm in {"dirichlet", "non-iid", "non_iid", "noniid"}:
        return create_dirichlet_non_iid_splits(
            X=X,
            y=y,
            num_clients=num_clients,
            alpha=alpha,
            min_samples=min_samples,
            seed=seed,
        )
    if norm == "iid":
        indices = iid_partition_indices(len(y), num_clients=num_clients, seed=seed)
        return [(X[idx], y[idx]) for idx in indices]
    raise ValueError(f"Unsupported partition method '{method}'. Choose 'dirichlet' or 'iid'.")


def summarize_client_splits(
    clients_data: List[Tuple[np.ndarray, np.ndarray]],
) -> List[Dict[str, Any]]:
    """Compute per-client total samples and class distribution statistics."""
    summary: List[Dict[str, Any]] = []
    for cid, (x_c, y_c) in enumerate(clients_data, 1):
        y_arr = np.asarray(y_c)
        unique, counts = np.unique(y_arr, return_counts=True)
        total = int(len(y_arr))
        class_counts = {str(k): int(v) for k, v in zip(unique, counts)}
        class_ratios = {
            str(k): round(float(v) / max(total, 1), 4)
            for k, v in zip(unique, counts)
        }
        summary.append({
            "client_id": cid,
            "num_samples": total,
            "num_features": int(x_c.shape[1]) if np.asarray(x_c).ndim == 2 else 0,
            "class_counts": class_counts,
            "class_ratios": class_ratios,
        })
    return summary


def print_client_split_summary(
    clients_data: List[Tuple[np.ndarray, np.ndarray]],
    method: str = "dirichlet",
    alpha: Optional[float] = 0.3,
) -> None:
    """Print a compact summary table of per-client Non-IID/IID class skew."""
    stats = summarize_client_splits(clients_data)
    all_classes = sorted({cls for row in stats for cls in row["class_counts"].keys()})
    header_label = (
        f"Non-IID Dirichlet Distribution (alpha = {alpha})"
        if str(method).lower() != "iid"
        else "IID Balanced Partition"
    )
    print(f"Data Partition: {header_label}")
    for row in stats:
        cls_desc = " | ".join(
            f"Class {c}: {row['class_counts'].get(c, 0):>6d} ({row['class_ratios'].get(c, 0.0)*100:5.1f}%)"
            for c in all_classes
        )
        print(f"  Client {row['client_id']}: {row['num_samples']:>6d} samples | {cls_desc}")


def _clone_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _trainable_names(model: nn.Module) -> List[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def _zeros_like_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: torch.zeros_like(p.data) for name, p in model.named_parameters() if p.requires_grad}


def _extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """MOON representation: prefer get_features(), else pre-softmax logits."""
    if hasattr(model, "get_features"):
        try:
            feats = model.get_features(x)
            if isinstance(feats, torch.Tensor) and feats.dim() >= 2:
                return feats
        except Exception:
            pass
    logits = model(x)
    if logits.dim() > 2:
        logits = logits.view(logits.size(0), -1)
    return logits


class Client:
    def __init__(self, client_id: int, model: nn.Module,
                 data: Tuple[np.ndarray, np.ndarray], config: Dict[str, Any]):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = copy.deepcopy(model).to(self.device)
        self.data = data
        self.config = config

        X, y = data
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        self.loader = DataLoader(
            dataset,
            batch_size=config.get("batch_size", 128),
            shuffle=True,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.control_variate = _zeros_like_params(self.model)
        self.prev_model = copy.deepcopy(self.model).to(self.device)
        self.prev_model.eval()
        for p in self.prev_model.parameters():
            p.requires_grad_(False)

        # Enable gradients on ensemble members so FL actually updates weights.
        for p in self.model.parameters():
            p.requires_grad_(True)

    def train(self, global_state: Dict[str, torch.Tensor], method: str,
              global_control: Optional[Dict[str, torch.Tensor]] = None,
              epochs: int = 2) -> Dict[str, Any]:
        self.model.load_state_dict(global_state, strict=False)
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad_(True)

        lr = self.config.get("learning_rate", 0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        start_state = _clone_state(self.model.state_dict())
        local_steps = 0
        total_loss = 0.0
        method = method.lower()
        mu = self.config.get("fedprox_mu", 0.01)
        moon_mu = self.config.get("moon_mu", 0.01)
        temperature = self.config.get("moon_temperature", 0.5)

        global_params = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters() if p.requires_grad
        }
        # Keep a frozen global copy for MOON positive pair.
        frozen_global = copy.deepcopy(self.model).to(self.device)
        frozen_global.eval()
        for p in frozen_global.parameters():
            p.requires_grad_(False)

        if global_control is None:
            global_control = _zeros_like_params(self.model)

        for _ in range(epochs):
            for batch_x, batch_y in self.loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                if method == "fedprox":
                    prox = torch.tensor(0.0, device=self.device)
                    for name, param in self.model.named_parameters():
                        if not param.requires_grad:
                            continue
                        prox = prox + torch.norm(param - global_params[name]) ** 2
                    loss = loss + (mu / 2.0) * prox

                if method == "moon":
                    local_feat = _extract_features(self.model, batch_x)
                    with torch.no_grad():
                        pos_feat = _extract_features(frozen_global, batch_x)
                        neg_feat = _extract_features(self.prev_model, batch_x)
                    pos_sim = F.cosine_similarity(local_feat, pos_feat, dim=-1)
                    neg_sim = F.cosine_similarity(local_feat, neg_feat, dim=-1)
                    logits_con = torch.stack([pos_sim, neg_sim], dim=1) / temperature
                    labels = torch.zeros(local_feat.size(0), dtype=torch.long, device=self.device)
                    contrastive = F.cross_entropy(logits_con, labels)
                    loss = loss + moon_mu * contrastive

                loss.backward()

                if method == "scaffold":
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.grad is None or name not in self.control_variate:
                                continue
                            c = self.control_variate[name].to(self.device)
                            c_global = global_control[name].to(self.device)
                            param.grad.add_(c_global - c)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item())
                local_steps += 1

        # SCAFFOLD client control update: c_i <- c_i - c_global + (x - y_i) / (K * lr)
        if method == "scaffold" and local_steps > 0:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name not in self.control_variate:
                        continue
                    x = global_params[name]
                    y_i = param.data
                    c_global = global_control[name].to(self.device)
                    lr = self.config.get("learning_rate", 0.001)
                    # Corrected formula: (x - y_i) / (local_steps * lr)
                    correction = (x - y_i) / (local_steps * lr)
                    self.control_variate[name] = (
                        self.control_variate[name].to(self.device)
                        - c_global
                        + correction
                    ).detach().cpu()

        # Update previous local model for next MOON round.
        self.prev_model.load_state_dict(self.model.state_dict(), strict=False)

        avg_loss = total_loss / max(local_steps, 1)
        return {
            "state_dict": _clone_state(self.model.state_dict()),
            "num_samples": len(self.data[0]),
            "local_steps": local_steps,
            "train_loss": avg_loss,
            "control_variate": copy.deepcopy(self.control_variate) if method == "scaffold" else None,
            "start_state": start_state,
        }


class Server:
    def __init__(self, global_model: nn.Module, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = copy.deepcopy(global_model).to(self.device)
        for p in self.global_model.parameters():
            p.requires_grad_(True)
        self.config = config
        self.aggregation = config.get("aggregation", "fedavg").lower()
        self.global_control = _zeros_like_params(self.global_model)

    def aggregate(self, updates: List[Dict[str, Any]], method: str) -> Dict[str, torch.Tensor]:
        method = method.lower()
        if method == "scaffold":
            return self._scaffold_aggregate(updates)
        if method == "fednova":
            return self._fednova_aggregate(updates)
        return self._fedavg_aggregate(updates)

    def _fedavg_aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        total = sum(u["num_samples"] for u in updates) or 1
        aggregated = copy.deepcopy(self.global_model.state_dict())
        float_keys = [k for k, v in aggregated.items() if torch.is_floating_point(v)]
        for k in float_keys:
            aggregated[k] = torch.zeros_like(aggregated[k])
        for u in updates:
            w = u["num_samples"] / total
            sd = u["state_dict"]
            for k in float_keys:
                if k in sd:
                    aggregated[k] = aggregated[k] + sd[k].to(self.device) * w
            for k, v in aggregated.items():
                if k not in float_keys and k in sd:
                    aggregated[k] = sd[k].to(self.device)
        return aggregated

    def _scaffold_aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        averaged = self._fedavg_aggregate(updates)
        n = len(updates) or 1
        for name in self.global_control:
            acc = torch.zeros_like(self.global_control[name])
            for u in updates:
                cv = u.get("control_variate") or {}
                if name in cv:
                    acc = acc + cv[name]
            self.global_control[name] = acc / n
        return averaged

    def _fednova_aggregate(self, updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Normalize each client's update by its local step count, then weight by samples."""
        total = sum(u["num_samples"] for u in updates) or 1
        current = self.global_model.state_dict()
        aggregated = copy.deepcopy(current)
        float_keys = [k for k, v in aggregated.items() if torch.is_floating_point(v)]

        # tau_eff = sum_i (n_i / n) * tau_i
        tau_eff = 0.0
        for u in updates:
            tau_i = max(int(u.get("local_steps", 1)), 1)
            tau_eff += (u["num_samples"] / total) * tau_i
        tau_eff = max(tau_eff, 1.0)

        for k in float_keys:
            acc = torch.zeros_like(aggregated[k])
            for u in updates:
                tau_i = max(int(u.get("local_steps", 1)), 1)
                delta = u["state_dict"][k].to(self.device) - current[k].to(self.device)
                normed = delta / tau_i
                acc = acc + (u["num_samples"] / total) * normed
            aggregated[k] = current[k].to(self.device) + tau_eff * acc

        for k, v in aggregated.items():
            if k not in float_keys:
                # keep last client's non-float buffer (BatchNorm counters etc.)
                aggregated[k] = updates[-1]["state_dict"].get(k, v).to(self.device)
        return aggregated

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        self.global_model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X_val).to(self.device)
            logits = self.global_model(x_tensor)
            preds = logits.argmax(dim=1).cpu().numpy()
        return {
            "accuracy": float(accuracy_score(y_val, preds)),
            "precision": float(precision_score(y_val, preds, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_val, preds, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_val, preds, average="weighted", zero_division=0)),
        }


def run_federated_learning(model: nn.Module,
                           clients_data: List[Tuple[np.ndarray, np.ndarray]],
                           X_val: np.ndarray, y_val: np.ndarray,
                           config: Dict[str, Any] = None) -> Tuple[nn.Module, Dict]:
    """Run a single FL method. `config['aggregation']` selects the algorithm."""
    cfg = dict(FL_CONFIG)
    if config:
        cfg.update(config)

    method = cfg.get("aggregation", "fedavg").lower()
    rounds = int(cfg.get("rounds", 10))
    local_epochs = int(cfg.get("local_epochs", 5))

    server = Server(model, cfg)
    clients = [Client(i, model, data, cfg) for i, data in enumerate(clients_data)]

    history = {
        "method": method,
        "round": [],
        "round_acc": [],
        "round_f1": [],
        "train_loss": [],
    }

    print("\n" + "-" * 70)
    print(f"FL Method : {method.upper()} | Clients: {len(clients)} | "
          f"Rounds: {rounds} | Local epochs: {local_epochs}")
    print("-" * 70)

    for r in range(rounds):
        current_state = copy.deepcopy(server.global_model.state_dict())
        updates = []
        round_loss = 0.0
        for client in clients:
            u = client.train(
                global_state=current_state,
                method=method,
                global_control=server.global_control if method == "scaffold" else None,
                epochs=local_epochs,
            )
            updates.append(u)
            round_loss += u["train_loss"]

        new_state = server.aggregate(updates, method)
        server.global_model.load_state_dict(new_state, strict=False)

        metrics = server.evaluate(X_val, y_val)
        avg_loss = round_loss / max(len(clients), 1)
        history["round"].append(r + 1)
        history["round_acc"].append(metrics["accuracy"])
        history["round_f1"].append(metrics["f1"])
        history["train_loss"].append(avg_loss)
        print(f"  Round {r+1:2d}/{rounds} | Loss: {avg_loss:.4f} | "
              f"Val Acc: {metrics['accuracy']*100:.2f}% | Val F1: {metrics['f1']:.4f}")

    history["final_accuracy"] = history["round_acc"][-1] if history["round_acc"] else 0.0
    history["final_f1"] = history["round_f1"][-1] if history["round_f1"] else 0.0
    return server.global_model, history


def run_all_fl_methods(model: nn.Module,
                       clients_data: List[Tuple[np.ndarray, np.ndarray]],
                       X_val: np.ndarray, y_val: np.ndarray,
                       config: Dict[str, Any] = None,
                       methods: Optional[List[str]] = None) -> Tuple[nn.Module, str, Dict[str, Dict]]:
    """
    Train every FL algorithm from the same starting checkpoint.
    Returns (best_model, best_method_name, per_method_histories).
    """
    cfg = dict(FL_CONFIG)
    if config:
        cfg.update(config)
    methods = methods or FL_METHODS

    results: Dict[str, Dict[str, Any]] = {}
    best_name = methods[0]
    best_acc = -1.0
    best_model = None

    print("\n" + "=" * 75)
    print("STAGE 4: Federated Learning — running ALL methods from the same ensemble")
    print(f"Methods: {', '.join(m.upper() for m in methods)}")
    print("=" * 75)

    for method in methods:
        method_cfg = dict(cfg)
        method_cfg["aggregation"] = method
        trained, history = run_federated_learning(
            model=model,
            clients_data=clients_data,
            X_val=X_val,
            y_val=y_val,
            config=method_cfg,
        )
        acc = history.get("final_accuracy", 0.0)
        f1 = history.get("final_f1", 0.0)
        results[method] = {
            "accuracy": acc,
            "f1": f1,
            "history": history,
        }
        print(f"  -> {method.upper():8s} finished | Val Acc: {acc*100:.2f}% | Val F1: {f1:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_name = method
            best_model = trained

    print("\n" + "=" * 75)
    print("STAGE 4 SUMMARY — All Federated Algorithms")
    print("=" * 75)
    print(f"{'Method':<12} {'Val Acc':>10} {'Val F1':>10}")
    print("-" * 34)
    for method in methods:
        r = results[method]
        marker = "  <-- BEST" if method == best_name else ""
        print(f"{method.upper():<12} {r['accuracy']*100:9.2f}% {r['f1']:10.4f}{marker}")
    print("=" * 75)
    print(f"Selected global FL model: {best_name.upper()} "
          f"(Val Acc {best_acc*100:.2f}%)")
    print("=" * 75)

    return best_model, best_name, results
