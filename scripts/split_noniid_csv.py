#!/usr/bin/env python3
"""
Split a NIDS CSV dataset into Non-IID client CSV files using a Dirichlet
distribution Dir(alpha), for distributed testing across Client Edge servers.

Example:
    python scripts/split_noniid_csv.py --data data/RT_IOT2022.csv --clients 5 --alpha 0.3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATASET_CONFIG, FL_CONFIG
from fl.federated_learning import dirichlet_partition_indices


def detect_label_column(df: pd.DataFrame, target_column: Optional[str] = None) -> str:
    """Resolve the target label column matching main.py / DATASET_CONFIG conventions."""
    if target_column and target_column in df.columns:
        return target_column

    preferred = DATASET_CONFIG.get("target_column", "Attack_type")
    if preferred in df.columns:
        return preferred

    normalized_cols: Dict[str, str] = {
        str(c).lower().strip().replace(" ", "_").replace("-", "_"): c
        for c in df.columns
    }
    candidates = [
        "attack_type", "attack_cat", "label", "class", "target",
        "attack", "category", "classification", "is_attack",
        "malicious", "status",
    ]
    for cand in candidates:
        if cand in normalized_cols:
            return normalized_cols[cand]

    if len(df.columns) > 0:
        return df.columns[-1]
    raise ValueError("CSV contains no columns to partition on.")


def split_csv_dirichlet(
    data_path: Path,
    output_dir: Path,
    num_clients: int = 5,
    alpha: float = 0.3,
    target_column: Optional[str] = None,
    min_samples: int = 10,
    seed: int = 42,
    binary_labels: bool = False,
) -> Dict[str, object]:
    """Partition CSV rows across clients via Dir(alpha) and write per-client CSVs."""
    df = pd.read_csv(data_path, low_memory=False)
    label_col = detect_label_column(df, target_column)

    y_raw = df[label_col].astype(str).str.strip()
    if binary_labels:
        benign_keywords = {
            str(k).lower()
            for k in DATASET_CONFIG.get("benign_label_keywords", [])
        } | {"benign", "normal", "safe", "clean", "good", "0", "false"}
        y_for_split = np.array(
            [0 if val.lower() in benign_keywords else 1 for val in y_raw],
            dtype=np.int64,
        )
    else:
        _, y_for_split = np.unique(y_raw.values, return_inverse=True)

    client_indices = dirichlet_partition_indices(
        y=y_for_split,
        num_clients=num_clients,
        alpha=alpha,
        min_samples=min_samples,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    client_reports: List[Dict[str, object]] = []

    for cid, idx in enumerate(client_indices, 1):
        sub_df = df.iloc[idx].reset_index(drop=True)
        out_csv = output_dir / f"client_{cid}.csv"
        sub_df.to_csv(out_csv, index=False)

        counts = sub_df[label_col].astype(str).str.strip().value_counts().to_dict()
        total = int(len(sub_df))
        ratios = {k: round(float(v) / max(total, 1), 4) for k, v in counts.items()}
        client_reports.append({
            "client_id": cid,
            "file": str(out_csv),
            "num_samples": total,
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "class_ratios": ratios,
        })

    summary = {
        "source_dataset": str(data_path),
        "target_column": str(label_col),
        "partition_method": "dirichlet",
        "dirichlet_alpha": float(alpha),
        "num_clients": int(num_clients),
        "seed": int(seed),
        "total_rows": int(len(df)),
        "clients": client_reports,
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a NIDS CSV into Non-IID client CSVs using Dirichlet(alpha)."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset path")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "noniid_clients"),
        help="Directory for generated client_1.csv..client_K.csv files",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=int(FL_CONFIG.get("num_clients", 5)),
        help="Number of clients to partition across (default: 5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=float(FL_CONFIG.get("dirichlet_alpha", 0.3)),
        help="Dirichlet concentration parameter alpha (default: 0.3)",
    )
    parser.add_argument("--target-column", default=None, help="Explicit label column name")
    parser.add_argument("--seed", type=int, default=int(FL_CONFIG.get("partition_seed", 42)))
    parser.add_argument(
        "--binary-labels",
        action="store_true",
        help="Collapse raw attack labels to Benign(0)/Attack(1) before Dirichlet sampling",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    summary = split_csv_dirichlet(
        data_path=data_path,
        output_dir=Path(args.output_dir),
        num_clients=args.clients,
        alpha=args.alpha,
        target_column=args.target_column,
        min_samples=int(FL_CONFIG.get("min_client_samples", 10)),
        seed=args.seed,
        binary_labels=args.binary_labels,
    )

    print("=" * 75)
    print(f"Non-IID Dirichlet CSV Split Complete (alpha = {summary['dirichlet_alpha']})")
    print(f"Source Dataset : {os.path.basename(summary['source_dataset'])} ({summary['total_rows']} rows)")
    print(f"Label Column   : {summary['target_column']}")
    print(f"Output Folder  : {args.output_dir}")
    print("-" * 75)
    for c in summary["clients"]:
        top_classes = ", ".join(
            f"{k}: {v} ({c['class_ratios'][k]*100:.1f}%)"
            for k, v in list(c["class_counts"].items())[:4]
        )
        print(f"  Client {c['client_id']} -> {os.path.basename(c['file'])} | {c['num_samples']:>6d} rows | {top_classes}")
    print("=" * 75)


if __name__ == "__main__":
    main()
