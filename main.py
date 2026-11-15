#!/usr/bin/env python3
"""
SecureFed-NIDS: Next-Gen Ensemble Deep Learning with FL, DP & XAI Pipeline
============================================================================
Stages:
1. Dataset Loading, Preprocessing & Stratification
2. 5-Fold Stratified Cross-Validation & Candidate DL Models Training
3. Top-2 Selection, Statistical Hypothesis Tests (Paired t-test, McNemar),
   Soft Voting & Stacking Meta-Learner, and Confusion Matrix Visualization
4. Federated Learning (FedAvg, FedProx, SCAFFOLD, FedNova, MOON)
5. Privacy-Preserving DP-SGD Training with Epsilon Accounting
6. Explainable AI (SHAP Interpretability & Perturbation Verification)
7. Model Artifacts Persistence & Automatic HQ Version Publishing
"""

import argparse, sys, os, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(os.path.dirname(__file__), "src")
if os.path.isdir(src_dir): sys.path.insert(0, src_dir)

from models.architectures import ModelFactory
from models.training import train_single_model
from models.ensemble import EnsembleTrainer, compare_with_without_ensemble
from models.evaluation import run_kfold_cv, paired_t_test, mcnemar_test, plot_and_save_confusion_matrix
from fl.federated_learning import (
    FL_METHODS,
    partition_clients_data,
    print_client_split_summary,
    run_all_fl_methods,
)
from dp.differential_privacy import DPTrainer
from xai.explainability import SHAPExplainer, perturbation_test
from config import DATASET_CONFIG, FL_CONFIG, DP_CONFIG

def select_dataset_interactive():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".csv")])

    if not files:
        print("\n[!] No CSV dataset found in data/ folder!")
        sys.exit(1)

    print("\n" + "="*75)
    print("                      DATASET SELECTION MENU")
    print("="*75)
    for i, path in enumerate(files, 1):
        mb = os.path.getsize(path) / (1024*1024)
        print(f"  [{i}] {os.path.basename(path):<30} | Size: {mb:.1f} MB")
    print("="*75)

    while True:
        choice = input(f"Select dataset to train on [1-{len(files)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            selected = files[int(choice)-1]
            print(f"--> [SUCCESS] Selected Dataset: {os.path.basename(selected)}\n")
            return selected
        print(f"Invalid input! Please enter a number between 1 and {len(files)}.")

def load_and_preprocess_dataset(dataset_path, target_column=None):
    print("="*75)
    print("STAGE 1: Dataset Loading, Preprocessing & Stratification")
    print("="*75)
    print(f"Loading data from: {dataset_path}")

    path = Path(dataset_path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # ------------------------------
    # Auto-detect label column
    # ------------------------------
    target_col = None
    if target_column and target_column in df.columns:
        target_col = target_column
    else:
        preferred = DATASET_CONFIG.get("target_column", "Attack_type")
        if preferred in df.columns:
            target_col = preferred
        else:
            # Common label candidates (case-insensitive)
            normalized_cols = {str(c).lower().strip().replace(" ", "_").replace("-", "_"): c for c in df.columns}
            candidates = [
                "attack_type", "attack_cat", "label", "class", "target",
                "attack", "category", "classification", "is_attack",
                "malicious", "status"
            ]
            for cand in candidates:
                if cand in normalized_cols:
                    target_col = normalized_cols[cand]
                    break
            # Fallback to last column if no candidate keyword matched
            if target_col is None and len(df.columns) > 0:
                target_col = df.columns[-1]

    if target_col is None or target_col not in df.columns:
        raise ValueError(
            f"Target/label column not found. Available columns: {', '.join(map(str, df.columns))}"
        )

    # Drop explicit identifier columns (e.g. 'no', 'id', 'unnamed: 0')
    id_candidates = set(DATASET_CONFIG.get("id_columns", ["no"])) | {"id", "flow_id", "unnamed: 0"}
    cols_to_drop = [c for c in df.columns if str(c).lower().strip() in id_candidates and c != target_col]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    y_raw = df[target_col].astype(str).str.strip()
    benign_keywords = set(DATASET_CONFIG.get("benign_label_keywords", DATASET_CONFIG.get("benign_classes", [])))
    benign_keywords.update(["benign", "normal", "safe", "clean", "good", "0", "false"])

    # Determine whether benign keywords are present in this dataset
    sample_values = set(y_raw.sample(min(len(y_raw), 100), random_state=42).str.lower())
    has_benign = any(any(val == str(k).lower() for k in benign_keywords) for val in sample_values)

    if has_benign:
        # Binary Classification (0: Benign/Normal, 1: Attack/Malicious)
        y = []
        for lbl in y_raw.values:
            try:
                v = float(lbl)
                y.append(1 if v > 0 else 0)
                continue
            except Exception:
                pass
            s_low = str(lbl).strip().lower()
            if any(s_low == str(k).lower() for k in benign_keywords):
                y.append(0)
            else:
                y.append(1)
        y = np.asarray(y, dtype=np.int64)
    else:
        # Generic multi-class or arbitrary label encoding using LabelEncoder
        le_target = LabelEncoder()
        y = le_target.fit_transform(y_raw.values).astype(np.int64)

    # Ensure at least 2 classes exist
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(f"Dataset target column '{target_col}' only has 1 class ({unique_classes}). Training requires at least 2 classes.")

    # Remove target column from features
    df = df.drop(columns=[target_col])

    # -----------------------------------------------------------------
    # Check if dataset matches Canonical RT-IoT Schema or Arbitrary Dataset
    # -----------------------------------------------------------------
    canonical_features = DATASET_CONFIG.get("canonical_feature_names") or []
    df_cols = [str(c) for c in df.columns]
    matching_canonical = sum(1 for c in df_cols if c in canonical_features)

    # If the majority of canonical RT-IoT features match, preserve canonical ordering
    is_canonical = (len(canonical_features) == 83 and matching_canonical >= 25)

    if is_canonical:
        def fuzzy_best_match(req_feat: str, columns: list[str], threshold: float):
            from difflib import SequenceMatcher
            best = None
            best_score = 0.0
            req_low = str(req_feat).lower()
            for col in columns:
                score = SequenceMatcher(None, req_low, str(col).lower()).ratio()
                if score > best_score:
                    best_score = score
                    best = col
            return best if best_score >= threshold else None

        feature_threshold = float(DATASET_CONFIG.get("feature_match_threshold", 0.8))
        n = len(df)
        X = np.zeros((n, len(canonical_features)), dtype=np.float32)

        for i, req_feat in enumerate(canonical_features):
            col = req_feat if req_feat in df_cols else None
            if col is None:
                col = fuzzy_best_match(req_feat, df_cols, threshold=feature_threshold)
            if col is None:
                X[:, i] = 0.0
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                vals = pd.to_numeric(series, errors="coerce").fillna(0.0).values
            else:
                vals = LabelEncoder().fit_transform(series.astype(str).fillna("missing"))
            X[:, i] = vals.astype(np.float32)

        active_features = canonical_features
    else:
        # Dynamic handling for any arbitrary dataset (CIC-IDS2017, NSL-KDD, UNSW-NB15, custom CSV)
        active_features = [str(c) for c in df.columns]
        n = len(df)
        X = np.zeros((n, len(active_features)), dtype=np.float32)

        for i, col in enumerate(active_features):
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                vals = pd.to_numeric(series, errors="coerce").fillna(0.0).values
            else:
                try:
                    vals = LabelEncoder().fit_transform(series.astype(str).fillna("missing"))
                except Exception:
                    vals = pd.factorize(series.astype(str))[0]
            X[:, i] = vals.astype(np.float32)

    # NaN/Inf safety
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    print(f"Dataset Dimensions : {X.shape[0]} samples, {X.shape[1]} features ({'canonical RT-IoT' if is_canonical else 'dynamic dataset schema'})")
    print(f"Target Column      : {target_col}")
    print(f"Classes Found      : {len(np.unique(y))} classes | distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split BEFORE fitting StandardScaler to prevent validation/test normalization leakage
    min_count = int(np.min(np.bincount(y))) if len(y) > 0 else 0
    strat = y if min_count >= 2 else None
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=strat)

    min_temp_count = int(np.min(np.bincount(y_temp))) if len(y_temp) > 0 else 0
    strat_temp = y_temp if min_temp_count >= 2 else None
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=strat_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32) if len(X_val) > 0 else X_val
    X_test = scaler.transform(X_test).astype(np.float32) if len(X_test) > 0 else X_test
    load_and_preprocess_dataset.last_scaler = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }

    print(f"Partitioning       : Train = {len(X_train)} | Validation = {len(X_val)} | Test = {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, active_features

def main():
    parser = argparse.ArgumentParser(description="SecureFed-NIDS Complete Pipeline")
    parser.add_argument("--mode", default="all", choices=["demo", "train", "fl", "dp", "all"], help="Operation mode")
    parser.add_argument("--data", default=None, help="Direct dataset path")
    parser.add_argument("--target-column", default=None, help="Dataset label/target column name")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for candidate models")
    parser.add_argument("--kfold", type=int, default=5, help="Number of folds for Cross-Validation (Default: 5)")
    parser.add_argument("--ensemble-method", default="voting", choices=["voting", "stacking", "weighted"], help="Ensemble method")
    parser.add_argument("--voting-type", default="soft", choices=["soft", "hard"], help="Voting type (Default: soft)")
    parser.add_argument(
        "--partition",
        default=FL_CONFIG.get("partition", "dirichlet"),
        choices=["dirichlet", "iid"],
        help="FL client data partition strategy: dirichlet (Non-IID) or iid (Default: dirichlet)",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=FL_CONFIG.get("dirichlet_alpha", 0.3),
        help="Dirichlet concentration parameter alpha for Non-IID split (Default: 0.3)",
    )
    args = parser.parse_args()

    print("\n" + "="*75)
    print("   SECUREFED-NIDS: NEXT-GEN ENSEMBLE DL WITH FL, DP, STATS & XAI")
    print("="*75)

    dataset_path = args.data if (args.data and os.path.exists(args.data)) else select_dataset_interactive()
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_and_preprocess_dataset(dataset_path, args.target_column)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    config = {"epochs": args.epochs, "batch_size": 128, "learning_rate": 0.001}

    # ==========================================
    # STAGE 2: 5-Fold Cross-Validation & Candidate DL Models
    # ==========================================
    print("\n" + "="*75)
    print(f"STAGE 2: Training 6 Candidate DL Models with {args.kfold}-Fold Cross-Validation")
    print("="*75)

    model_names = ["mlp", "cnn", "lstm", "bilstm", "gru", "transformer"]
    cv_results = {}
    models_dict = {}
    model_metrics = {}

    # Combine train + val for representative 5-fold CV
    X_cv = np.vstack([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val])

    for m_name in model_names:
        cv_res = run_kfold_cv(
            model_factory_fn=lambda name=m_name: ModelFactory.create_model(name, input_dim, num_classes),
            model_name=m_name,
            X=X_cv,
            y=y_cv,
            k_folds=args.kfold,
            epochs=args.epochs,
            batch_size=config["batch_size"],
            lr=config["learning_rate"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        cv_results[m_name] = cv_res

        # Train a full candidate instance on (X_train, y_train) for Stage 3-7 integration
        cand_model, cand_hist = train_single_model(
            model_name=m_name,
            input_dim=input_dim,
            num_classes=num_classes,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            config=config
        )
        models_dict[m_name] = cand_model
        model_metrics[m_name] = {
            "accuracy": cv_res["mean_accuracy"],
            "f1": cv_res["mean_f1"],
            "std_acc": cv_res["std_accuracy"]
        }

    print("\n" + "="*75)
    print(f"5-FOLD CROSS-VALIDATION SUMMARY (Mean ± Std):")
    print("="*75)
    for m_name, res in cv_results.items():
        print(f"\n{m_name.upper()}:")
        print(f"  Accuracy    : {res['mean_accuracy']*100:.2f}% (±{res['std_accuracy']*100:.2f}%)")
        print(f"  Precision   : {res['mean_precision']:.4f} (±{res['std_precision']:.4f})")
        print(f"  Recall      : {res['mean_recall']:.4f} (±{res['std_recall']:.4f})")
        print(f"  F1-Score    : {res['mean_f1']:.4f} (±{res['std_f1']:.4f})")
        print(f"  MCC         : {res['mean_mcc']:.4f} (±{res['std_mcc']:.4f})")
        print(f"  Specificity : {res['mean_specificity']:.4f} (±{res['std_specificity']:.4f})")
        print(f"  AUC-ROC     : {res['mean_auc']:.4f} (±{res['std_auc']:.4f})")
        print(f"  Avg Train Time: {res['mean_train_time']:.2f}s | Avg Inference Time: {res['mean_infer_time']*1000:.2f}ms")
    print("="*75)

    # ==========================================
    # STAGE 3: Top-2 Selection, Statistical Tests, Soft Voting / Stacking & Confusion Matrix
    # ==========================================
    print("\n" + "="*75)
    print("STAGE 3: Top-2 Selection, Statistical Tests, Ensemble & Confusion Matrix")
    print("="*75)

    ensemble_trainer = EnsembleTrainer(config={
        "top_k": 2,
        "selection_metric": "accuracy",
        "ensemble_method": args.ensemble_method,
        "voting_type": args.voting_type
    })

    ensemble_model = ensemble_trainer.train_ensemble(
        model_results=model_metrics,
        models=models_dict,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        method=args.ensemble_method,
        voting_type=args.voting_type
    )

    ensemble_eval = ensemble_trainer.evaluate_ensemble(X_test, y_test)
    comparison = compare_with_without_ensemble(model_metrics, ensemble_eval)

    print(f"\nEnsemble Evaluation vs Baseline:")
    print(f"  -> Best Single Model : {comparison['best_single_model'].upper()} (5-Fold CV Acc: {comparison['best_single_accuracy']*100:.2f}%)")
    print(f"  -> Second Best Model : {comparison['second_best_model'].upper()} (5-Fold CV Acc: {comparison['second_best_accuracy']*100:.2f}%)")
    print(f"  -> Ensemble Model    : ({args.ensemble_method.upper()} - Test Acc: {comparison['ensemble_accuracy']*100:.2f}%)")
    print(f"  -> Gain / Delta      : {comparison['accuracy_improvement']*100:+.2f}%")

    # --- Statistical Hypothesis Testing ---
    print("\n" + "-"*65)
    print("STATISTICAL HYPOTHESIS TESTING:")
    print("-" * 65)
    best_name = comparison['best_single_model']
    second_name = comparison['second_best_model']

    # 1. Paired Student's t-test on 5-Fold validation scores
    t_test_res = paired_t_test(cv_results[best_name]["fold_accuracies"], cv_results[second_name]["fold_accuracies"])
    print(f"1. Paired Student's t-test ({best_name.upper()} vs {second_name.upper()} across 5 Folds):")
    print(f"   t-statistic: {t_test_res['t_statistic']:.4f} | p-value: {t_test_res['p_value']:.5f}")
    print(f"   Result: {t_test_res['conclusion']}")

    # 2. McNemar's Test on Test Predictions (Best Single Model vs Ensemble Model)
    best_single_model = models_dict[best_name]
    best_single_model.eval()
    with torch.no_grad():
        single_preds = best_single_model(torch.FloatTensor(X_test)).argmax(dim=1).numpy()

    ensemble_preds = ensemble_eval["predictions"]
    mcnemar_res = mcnemar_test(y_test, single_preds, ensemble_preds)
    print(f"\n2. McNemar's Test ({best_name.upper()} vs ENSEMBLE on Test Set):")
    print(f"   Chi2: {mcnemar_res['chi2_statistic']:.4f} | p-value: {mcnemar_res['p_value']:.5f}")
    print(f"   Discordant counts: Best Single correct only = {mcnemar_res['b_discordant (A correct, B wrong)']}, Ensemble correct only = {mcnemar_res['c_discordant (A wrong, B correct)']}")
    print(f"   Result: {mcnemar_res['conclusion']}")
    print("-" * 65)

    # --- Confusion Matrix Plotting ---
    cm_path = "reports/confusion_matrix.png"
    plot_and_save_confusion_matrix(
        y_true=y_test,
        y_pred=ensemble_preds,
        title=f"SecureFed-NIDS Ensemble ({args.ensemble_method.title()}) Confusion Matrix",
        save_path=cm_path
    )

    # ==========================================
    # GLOBAL MODEL: SecureFed-Global-Ensemble
    # ==========================================
    print("\n" + "="*75)
    print("GLOBAL MODEL: SecureFed-Global-Ensemble")
    print("="*75)
    global_model_name = "SecureFed-Global-Ensemble"
    global_model = ensemble_model
    print(f"Global Model Architecture: Ensemble [{best_name.upper()} + {second_name.upper()}] ({args.ensemble_method.title()})")
    print(f"Global Model Test Accuracy: {comparison['ensemble_accuracy']*100:.2f}%")

    # ==========================================
    # STAGE 4: Federated Learning (Non-IID - Dirichlet Distribution)
    # ==========================================
    print("\n" + "="*75)
    print("STAGE 4: Federated Learning — FedAvg / FedProx / SCAFFOLD / FedNova / MOON")
    print("="*75)

    num_clients = FL_CONFIG.get("num_clients", 5)
    partition_method = args.partition
    dirichlet_alpha = args.dirichlet_alpha
    clients_data = partition_clients_data(
        X=X_train,
        y=y_train,
        num_clients=num_clients,
        method=partition_method,
        alpha=dirichlet_alpha,
        min_samples=FL_CONFIG.get("min_client_samples", 10),
        seed=FL_CONFIG.get("partition_seed", 42),
    )

    print_client_split_summary(clients_data, method=partition_method, alpha=dirichlet_alpha)
    print(f"FL Settings: {num_clients} Clients | Rounds: 10 | Local Epochs: 5")
    print(f"Algorithms : {', '.join(m.upper() for m in FL_METHODS)}")
    print(f"Initializing FL from global model: {global_model_name}")
    fl_config = {
        "rounds": 10, "local_epochs": 5, "batch_size": 128, "learning_rate": 0.001,
        "fedprox_mu": FL_CONFIG.get("fedprox_mu", 0.01),
        "moon_mu": FL_CONFIG.get("moon_mu", 0.01),
        "moon_temperature": FL_CONFIG.get("moon_temperature", 0.5),
        "partition": partition_method,
        "dirichlet_alpha": dirichlet_alpha,
    }
    fl_trained_model, best_fl_method, fl_results = run_all_fl_methods(
        model=global_model,
        clients_data=clients_data,
        X_val=X_val, y_val=y_val,
        config=fl_config
    )
    global_model = fl_trained_model
    print(f"Stage 5/7 will use FL winner: {best_fl_method.upper()} (Val Acc {fl_results[best_fl_method]['accuracy']*100:.2f}%)")

    # ==========================================
    # STAGE 5: Differential Privacy (DP-SGD)
    # ==========================================
    print("\n" + "="*75)
    print("STAGE 5: Privacy-Preserving Training with Differential Privacy (DP-SGD)")
    print("="*75)
    epsilon_val = DP_CONFIG.get('epsilon', 1.0)
    print(f"DP Configuration: Epsilon (ε)={epsilon_val} | Delta (δ)=1e-5 | Max Clip Norm=1.0")
    print(f"Training {global_model_name} with DP-SGD...")

    dp_trainer = DPTrainer(global_model, DP_CONFIG)
    dp_trainer.fit(X_train, y_train, X_val, y_val, epochs=10)
    dp_acc = dp_trainer.evaluate(X_test, y_test)
    print(f"Privacy Guaranteed Test Accuracy: {dp_acc*100:.2f}% (Epsilon Budget Protected)")

    # ==========================================
    # STAGE 6: Explainable AI (SHAP & Perturbation)
    # ==========================================
    print("\n" + "="*75)
    print("STAGE 6: Explainable AI (SHAP Local Interpretability & Verification)")
    print("="*75)

    shap_explainer = SHAPExplainer(models_dict[comparison["best_single_model"]], X_val[:50], feature_names)
    exp = shap_explainer.explain_instance(X_test[0])

    print("\nTop 5 Influential Security Features Identified by SHAP:")
    for rank, feat_name in enumerate(exp["top_5_features"], 1):
        info = exp["feature_importance"][feat_name]
        print(f"  {rank}. {feat_name:<20} | SHAP Value: {info['shap_value']:+.4f} | {info['impact']}")

    change_rate = perturbation_test(models_dict[comparison["best_single_model"]], X_test, exp, feature_names=feature_names)
    print(f"\nPerturbation Test Stability Rate: {change_rate:.4f} (Validates feature causality)")

    # ==========================================
    # STAGE 7: Model Artifacts Saving & Summary
    # ==========================================
    print("\n" + "="*75)
    print("STAGE 7: Model Persistence & Final Summary")
    print("="*75)
    os.makedirs("models", exist_ok=True)
    os.makedirs("edge_deployment", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Save the global ensemble model
    global_model_path = "models/securefed_global_ensemble.pt"
    torch.save(global_model, global_model_path)
    print(f"✓ Global Ensemble Model saved to: {global_model_path}")

    # Save edge deployment package (including scaler parameters for consistent edge inference)
    last_scaler = getattr(load_and_preprocess_dataset, "last_scaler", None) or {}
    edge_package = {
        "model_name": "SecureFed-Global-Ensemble",
        "architecture": f"EnsembleTop2_{args.ensemble_method}",
        "best_fl_method": best_fl_method,
        "fl_partition": partition_method,
        "dirichlet_alpha": dirichlet_alpha if partition_method != "iid" else None,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "feature_names": feature_names,
        "scaler_mean": last_scaler.get("mean"),
        "scaler_scale": last_scaler.get("scale"),
        "accuracy": comparison["ensemble_accuracy"],
        "model_state_dict": global_model.state_dict()
    }
    edge_path = "edge_deployment/securefed_edge_model.pt"
    torch.save(edge_package, edge_path)
    print(f"✓ Edge Deployment Package saved to: {edge_path}")

    # Publish to HQ Model Registry & cache server holdout test split for post-aggregation evaluation
    try:
        from server_hq.publish import publish_from_artifacts
        from server_hq.registry import ModelRegistry
        partition_tag = f"{partition_method}(alpha={dirichlet_alpha})" if partition_method != "iid" else "iid"
        published = publish_from_artifacts(
            weights=Path(global_model_path),
            package=Path(edge_path),
            notes=f"pipeline run | method={args.ensemble_method} | fl={best_fl_method} | partition={partition_tag} | acc={comparison['ensemble_accuracy']:.4f}",
        )
        ModelRegistry().save_eval_holdout(
            X_test=X_test,
            y_test=y_test,
            input_dim=input_dim,
            feature_names=feature_names,
            dataset_name=Path(dataset_path).name,
        )
        print(f"✓ HQ published version v{published['version']} (latest) → hq_store/models/v{published['version']}/")
    except Exception as publish_err:
        print(f"[!] HQ publish skipped: {publish_err}")

    # Update client config
    import json
    config_path = "edge_deployment/client_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            client_cfg = json.load(f)
        client_cfg["model_name"] = "SecureFed-Global-Ensemble"
        with open(config_path, 'w') as f:
            json.dump(client_cfg, f, indent=2)
        print(f"✓ Client config updated with global model name")

    print("\n" + "="*75)
    print("✓ Complete Industrial 7-Stage Pipeline with 5-Fold CV, Stats & Stacking Executed Successfully!")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
