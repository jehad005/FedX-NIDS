"""SecureFed-NIDS Configuration Module

Central configuration for all project components.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_CONFIG = {
    "name": "RT-IoT2022",
    "path": DATA_DIR / "RT_IOT2022.csv",
    "target_column": "Attack_type",
    "test_size": 0.2,
    "random_state": 42,
    "id_columns": ["no"],
    "categorical_columns": ["proto", "service"],
    "benign_classes": ["MQTT_Publish", "Thing_Speak", "Wipro_bulb"],
    "binary_label": True,
    # Canonical feature schema used by the published models (v1.0.3).
    # Any CSV is mapped to this exact ordering so the model input_dim stays 83.
    "canonical_feature_names": [
        "id.orig_p",
        "id.resp_p",
        "proto",
        "service",
        "flow_duration",
        "fwd_pkts_tot",
        "bwd_pkts_tot",
        "fwd_data_pkts_tot",
        "bwd_data_pkts_tot",
        "fwd_pkts_per_sec",
        "bwd_pkts_per_sec",
        "flow_pkts_per_sec",
        "down_up_ratio",
        "fwd_header_size_tot",
        "fwd_header_size_min",
        "fwd_header_size_max",
        "bwd_header_size_tot",
        "bwd_header_size_min",
        "bwd_header_size_max",
        "flow_FIN_flag_count",
        "flow_SYN_flag_count",
        "flow_RST_flag_count",
        "fwd_PSH_flag_count",
        "bwd_PSH_flag_count",
        "flow_ACK_flag_count",
        "fwd_URG_flag_count",
        "bwd_URG_flag_count",
        "flow_CWR_flag_count",
        "flow_ECE_flag_count",
        "fwd_pkts_payload.min",
        "fwd_pkts_payload.max",
        "fwd_pkts_payload.tot",
        "fwd_pkts_payload.avg",
        "fwd_pkts_payload.std",
        "bwd_pkts_payload.min",
        "bwd_pkts_payload.max",
        "bwd_pkts_payload.tot",
        "bwd_pkts_payload.avg",
        "bwd_pkts_payload.std",
        "flow_pkts_payload.min",
        "flow_pkts_payload.max",
        "flow_pkts_payload.tot",
        "flow_pkts_payload.avg",
        "flow_pkts_payload.std",
        "fwd_iat.min",
        "fwd_iat.max",
        "fwd_iat.tot",
        "fwd_iat.avg",
        "fwd_iat.std",
        "bwd_iat.min",
        "bwd_iat.max",
        "bwd_iat.tot",
        "bwd_iat.avg",
        "bwd_iat.std",
        "flow_iat.min",
        "flow_iat.max",
        "flow_iat.tot",
        "flow_iat.avg",
        "flow_iat.std",
        "payload_bytes_per_second",
        "fwd_subflow_pkts",
        "bwd_subflow_pkts",
        "fwd_subflow_bytes",
        "bwd_subflow_bytes",
        "fwd_bulk_bytes",
        "bwd_bulk_bytes",
        "fwd_bulk_packets",
        "bwd_bulk_packets",
        "fwd_bulk_rate",
        "bwd_bulk_rate",
        "active.min",
        "active.max",
        "active.tot",
        "active.avg",
        "active.std",
        "idle.min",
        "idle.max",
        "idle.tot",
        "idle.avg",
        "idle.std",
        "fwd_init_window_size",
        "bwd_init_window_size",
        "fwd_last_window_size",
    ],
    # Feature matching: fuzzy match if exact column name is not found.
    "feature_match_threshold": 0.8,
    # Label keywords used by auto-detection.
    "benign_label_keywords": [
        "MQTT_Publish",
        "Thing_Speak",
        "Wipro_bulb",
        "Normal",
        "normal",
        "Benign",
        "benign",
        "0",
        "false",
        "False",
    ],
    "attack_label_keywords": [
        "Attack",
        "attack",
        "Malicious",
        "malicious",
        "1",
        "true",
        "True",
        "DDoS",
        "DoS",
        "PortScan",
        "Bot",
        "Infiltration",
        "Web Attack",
    ],
}

PREPROCESSING_CONFIG = {
    "normalization": "standard",
    "encoding": "label",
    "handle_missing": "drop",
    "feature_selection": "variance_threshold",
    "variance_threshold": 0.01,
}

MODEL_CONFIG = {
    "mlp": {
        "hidden_dims": [256, 128, 64],
        "activation": "relu",
        "dropout": 0.3,
        "batch_norm": True,
    },
    "cnn": {
        "filters": [64, 128, 256],
        "kernel_sizes": [3, 3, 3],
        "pool_size": 2,
        "dropout": 0.3,
    },
    "lstm": {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": False,
    },
    "bilstm": {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": True,
    },
    "gru": {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": False,
    },
    "transformer": {
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dropout": 0.1,
    },
}

TRAINING_CONFIG = {
    "epochs": 50,
    "batch_size": 256,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
}

ENSEMBLE_CONFIG = {
    "top_k": 2,
    "selection_metric": "accuracy",
    "ensemble_method": "voting",
    "voting_type": "soft",
}

FL_CONFIG = {
    "num_clients": 5,
    "rounds": 20,
    "local_epochs": 5,
    "aggregation": "fedavg",
    "supported_methods": ["fedavg", "fedprox", "scaffold", "fednova", "moon"],
    "fedprox_mu": 0.01,
    "moon_temperature": 0.5,
    "moon_mu": 0.01,
}

DP_CONFIG = {
    "enabled": True,
    "epsilon": 1.0,
    "delta": 1e-5,
    "clipping_norm": 1.0,
    "noise_multiplier": 1.1,
    "max_grad_norm": 1.0,
    "min_clipping_norm": 0.1,
    "max_clipping_norm": 10.0,
    "min_noise_multiplier": 0.5,
    "privacy_levels": {
        "low": {"epsilon": 2.0, "noise_multiplier": 0.5},
        "medium": {"epsilon": 1.0, "noise_multiplier": 1.0},
        "high": {"epsilon": 0.5, "noise_multiplier": 2.0},
    },
}

XAI_CONFIG = {
    "shap": {
        "method": "kernel",
        "sample_size": 100,
        "background_samples": 50,
    },
    "lime": {
        "num_features": 10,
        "num_samples": 500,
    },
}

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "workers": 4,
    "cors_origins": ["*"],
    "jwt_secret": os.getenv("JWT_SECRET", "securefed-nids-secret-key"),
    "jwt_algorithm": "HS256",
    "jwt_expire_minutes": 60,
}

DASHBOARD_CONFIG = {
    "title": "SecureFed-NIDS SOC Dashboard",
    "theme": "dark",
    "refresh_interval": 5,
    "alerts_per_page": 10,
    "max_upload_size_mb": 100,
}

MONITORING_CONFIG = {
    "prometheus_port": 9090,
    "grafana_port": 3000,
    "metrics_interval": 30,
    "log_level": "INFO",
    "log_format": "json",
}

DATABASE_CONFIG = {
    "type": "sqlite",
    "path": DATA_DIR / "securefed_nids.db",
    "pool_size": 10,
    "max_overflow": 20,
}

HQ_CONFIG = {
    "store_dir": PROJECT_ROOT / "hq_store",
    "initial_version": "1.0.0",
    "model_name": "SecureFed-Global-Ensemble",
}

DOCKER_CONFIG = {
    "image_name": "securefed-nids",
    "version": "1.0.0",
    "python_version": "3.11",
    "cuda_version": "12.1",
    "registry": "ghcr.io",
}

CICD_CONFIG = {
    "test_framework": "pytest",
    "coverage_min": 0.80,
    "linting": ["ruff", "mypy"],
    "security_scan": ["bandit", "safety"],
}

EDGE_CONFIG = {
    "quantization": "dynamic",
    "pruning": False,
    "pruning_rate": 0.3,
    "target_latency_ms": 10,
}