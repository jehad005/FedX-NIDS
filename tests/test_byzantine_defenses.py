"""
Unit and Integration Tests for Byzantine-Robust Aggregation Defenses
====================================================================
Tests the defense mechanisms against model poisoning attacks:
- Coordinate-wise Trimmed Mean
- Coordinate-wise Median
- Krum (Geometric distance minimization)
- Multi-Krum (Top-m benign averaging)
"""

import math
import sys
from pathlib import Path
import pytest
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server_hq.central_aggregator import CentralAggregator


@pytest.fixture
def aggregator():
    return CentralAggregator(device="cpu")


def create_simulated_updates(num_honest=4, honest_target=1.0, noise_std=0.05, poison_val=-100.0):
    """
    Creates a simulated federated update set with honest clients centered at `honest_target`
    and 1 Byzantine malicious client sending an extreme poisoned weight update.
    """
    updates = []
    # Honest clients
    for i in range(num_honest):
        update = {
            "weight": torch.ones(20, 20) * (honest_target + (i - num_honest / 2) * 0.02) + torch.randn(20, 20) * noise_std,
            "bias": torch.ones(20) * (honest_target * 0.1) + torch.randn(20) * noise_std,
        }
        updates.append(update)

    # Malicious Byzantine client (sign-flip + extreme scale)
    poisoned_update = {
        "weight": torch.ones(20, 20) * poison_val,
        "bias": torch.ones(20) * poison_val,
    }
    updates.append(poisoned_update)
    return updates


def test_fedavg_fails_under_poisoning(aggregator):
    """Confirm that un-defended FedAvg is heavily vulnerable to poisoning."""
    updates = create_simulated_updates(num_honest=4, honest_target=1.0, poison_val=-100.0)
    fedavg_agg = aggregator.aggregate_fedavg(updates)
    mean_val = fedavg_agg["weight"].mean().item()

    # FedAvg average will be pulled negative by the poisoner (-19.2)
    assert mean_val < 0.0, f"FedAvg should be poisoned into negative range, got {mean_val}"


def test_trimmed_mean_defense(aggregator):
    """Verify Coordinate-wise Trimmed Mean cuts off extreme outlier updates."""
    updates = create_simulated_updates(num_honest=4, honest_target=1.0, poison_val=-100.0)
    tm_agg = aggregator.aggregate_trimmed_mean(updates, trim_ratio=0.2)
    mean_val = tm_agg["weight"].mean().item()

    # Must stay close to the honest target of 1.0
    assert 0.85 <= mean_val <= 1.15, f"Trimmed Mean failed to defend: {mean_val}"


def test_median_defense(aggregator):
    """Verify Coordinate-wise Median completely resists single adversarial outlier."""
    updates = create_simulated_updates(num_honest=4, honest_target=1.0, poison_val=-100.0)
    median_agg = aggregator.aggregate_median(updates)
    mean_val = median_agg["weight"].mean().item()

    # Must stay close to the honest target of 1.0
    assert 0.85 <= mean_val <= 1.15, f"Median failed to defend: {mean_val}"


def test_krum_defense(aggregator):
    """Verify Krum identifies and isolates the single Byzantine outlier using geometric distance."""
    updates = create_simulated_updates(num_honest=4, honest_target=1.0, poison_val=-100.0)
    krum_agg = aggregator.aggregate_krum(updates, num_byzantine=1, multi_krum=False)
    mean_val = krum_agg["weight"].mean().item()

    # Krum must select an honest client update
    assert 0.85 <= mean_val <= 1.15, f"Krum selected poisoned update: {mean_val}"


def test_multi_krum_defense(aggregator):
    """Verify Multi-Krum selects top-m honest clients and averages them safely."""
    updates = create_simulated_updates(num_honest=4, honest_target=1.0, poison_val=-100.0)
    mkrum_agg = aggregator.aggregate_krum(updates, num_byzantine=1, multi_krum=True, top_m=3)
    mean_val = mkrum_agg["weight"].mean().item()

    # Multi-Krum must average strictly from honest clients
    assert 0.85 <= mean_val <= 1.15, f"Multi-Krum failed to defend: {mean_val}"


def test_aggregate_with_metadata_dispatch(aggregator):
    """Verify dispatch method properly runs and returns metadata for Byzantine defenses."""
    updates = create_simulated_updates(num_honest=4, honest_target=1.0, poison_val=-100.0)

    for method in ["trimmed_mean", "median", "krum", "multi_krum"]:
        agg, meta = aggregator.aggregate_with_metadata(updates, method=method)
        assert "weight" in agg
        assert "algorithm" in meta
        mean_val = agg["weight"].mean().item()
        assert 0.80 <= mean_val <= 1.20, f"Method {method} failed defense: {mean_val}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
