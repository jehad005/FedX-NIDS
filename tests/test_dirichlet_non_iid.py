#!/usr/bin/env python3
"""
Tests for Dirichlet (alpha=0.3) Non-IID data partitioning across FL pipeline
and CSV splitter utility.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import FL_CONFIG
from fl.federated_learning import (
    create_dirichlet_non_iid_splits,
    dirichlet_partition_indices,
    partition_clients_data,
    print_client_split_summary,
    run_all_fl_methods,
    summarize_client_splits,
)
from scripts.split_noniid_csv import split_csv_dirichlet


def test_dirichlet_disjoint_and_skewed():
    rng = np.random.default_rng(7)
    n_samples = 2000
    n_features = 12
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 4, size=n_samples)

    indices = dirichlet_partition_indices(y=y, num_clients=5, alpha=0.3, min_samples=10, seed=42)
    assert len(indices) == 5
    concat = np.concatenate(indices)
    assert len(concat) == n_samples
    assert len(np.unique(concat)) == n_samples, "Client partitions must be disjoint"
    assert min(len(idx) for idx in indices) >= 10

    noniid_splits = create_dirichlet_non_iid_splits(X, y, num_clients=5, alpha=0.3, min_samples=10, seed=42)
    iid_splits = partition_clients_data(X, y, num_clients=5, method="iid", seed=42)

    noniid_stats = summarize_client_splits(noniid_splits)
    iid_stats = summarize_client_splits(iid_splits)

    noniid_c0_ratios = [s["class_ratios"].get("0", 0.0) for s in noniid_stats]
    iid_c0_ratios = [s["class_ratios"].get("0", 0.0) for s in iid_stats]
    assert np.std(noniid_c0_ratios) > np.std(iid_c0_ratios), (
        f"Expected Dirichlet alpha=0.3 to create stronger class skew than IID "
        f"(std {np.std(noniid_c0_ratios):.4f} vs {np.std(iid_c0_ratios):.4f})"
    )
    print_client_split_summary(noniid_splits, method="dirichlet", alpha=0.3)


def test_fl_config_defaults():
    assert FL_CONFIG.get("partition") == "dirichlet"
    assert abs(float(FL_CONFIG.get("dirichlet_alpha")) - 0.3) < 1e-9


def test_fl_pipeline_with_dirichlet_splits():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
    X_val = rng.normal(size=(80, 8)).astype(np.float32)
    y_val = (X_val[:, 0] + 0.5 * X_val[:, 1] > 0).astype(np.int64)

    clients_data = partition_clients_data(X, y, num_clients=5, method="dirichlet", alpha=0.3, min_samples=10, seed=42)

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 2)

        def get_features(self, x):
            return torch.relu(self.fc1(x))

        def forward(self, x):
            return self.fc2(self.get_features(x))

    best_model, best_name, results = run_all_fl_methods(
        model=TinyNet(),
        clients_data=clients_data,
        X_val=X_val,
        y_val=y_val,
        config={"rounds": 2, "local_epochs": 1, "batch_size": 32, "learning_rate": 0.01},
    )
    assert best_model is not None
    assert best_name in results
    assert len(results) == 5


def test_csv_splitter_utility():
    rng = np.random.default_rng(99)
    n = 600
    df = pd.DataFrame({
        "feat_1": rng.normal(size=n),
        "feat_2": rng.normal(size=n),
        "Label": np.where(rng.random(n) > 0.4, "BENIGN", "DDoS"),
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src_csv = tmp / "sample.csv"
        out_dir = tmp / "clients"
        df.to_csv(src_csv, index=False)

        summary = split_csv_dirichlet(
            data_path=src_csv,
            output_dir=out_dir,
            num_clients=5,
            alpha=0.3,
            min_samples=10,
            seed=42,
        )
        assert summary["dirichlet_alpha"] == 0.3
        assert summary["total_rows"] == n
        assert (out_dir / "split_summary.json").exists()
        total_out = 0
        for cid in range(1, 6):
            client_csv = out_dir / f"client_{cid}.csv"
            assert client_csv.exists()
            part_df = pd.read_csv(client_csv)
            assert list(part_df.columns) == ["feat_1", "feat_2", "Label"]
            total_out += len(part_df)
        assert total_out == n


if __name__ == "__main__":
    test_fl_config_defaults()
    test_dirichlet_disjoint_and_skewed()
    test_fl_pipeline_with_dirichlet_splits()
    test_csv_splitter_utility()
    print("\nALL DIRICHLET NON-IID TESTS PASSED!")
