"""
Evaluation Module for SecureFed-NIDS
====================================
Comprehensive evaluation metrics and statistical tests for model assessment.
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset


def _as_numpy(value):
    """Convert tensors/array-like values to a detached NumPy array."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """Calculate the requested classification metrics.

    Precision, recall and F1 are weighted averages.  Specificity is the
    binary true-negative rate; for multiclass data it is macro one-vs-rest
    specificity.  ``y_prob`` may be either a NumPy array or a PyTorch tensor.
    """
    y_true = _as_numpy(y_true).reshape(-1)
    y_pred = _as_numpy(y_pred).reshape(-1)
    probabilities = _as_numpy(y_prob)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    elif len(labels) > 2:
        class_specificities = []
        total = cm.sum()
        for i in range(len(labels)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = total - tp - fn - fp
            class_specificities.append(tn / (tn + fp) if (tn + fp) else 0.0)
        metrics["specificity"] = float(np.mean(class_specificities))
    else:
        metrics["specificity"] = 0.0

    metrics["auc"] = 0.0
    if probabilities is not None:
        try:
            if probabilities.ndim == 1:
                metrics["auc"] = float(roc_auc_score(y_true, probabilities))
            elif probabilities.shape[1] == 2:
                metrics["auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
            elif probabilities.shape[1] > 2:
                metrics["auc"] = float(roc_auc_score(y_true, probabilities, multi_class="ovr"))
        except (ValueError, TypeError, IndexError):
            # AUC is undefined for a validation fold containing one class.
            metrics["auc"] = 0.0

    return metrics


def calculate_training_time(start_time: float, end_time: float) -> dict:
    """
    Calculate training and inference time.

    Parameters
    ----------
    start_time : float
        Start time (time.time() or similar)
    end_time : float
        End time

    Returns
    -------
    dict
        Training time and inference time in seconds
    """
    total_time = end_time - start_time

    # Estimate inference time (1 sample)
    inference_time_per_sample = total_time / max(1, 100)  # Assuming 100 samples for inference estimate

    return {
        "total_training_time_seconds": float(total_time),
        "estimated_inference_time_per_sample_seconds": float(inference_time_per_sample),
        "training_time_minutes": float(total_time / 60),
    }


def run_kfold_cv(
    model_factory_fn,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 0.001,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Run K-Fold Cross-Validation for a single model.

    Parameters
    ----------
    model_factory_fn : callable
        Function that returns a new model instance
    model_name : str
        Name of the model
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    k_folds : int
        Number of folds
    epochs : int
        Training epochs per fold
    batch_size : int
        Batch size
    lr : float
        Learning rate
    device : str
        Device to train on
    seed : int
        Random seed

    Returns
    -------
    dict
        Cross-validation results with mean and std for all metrics
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_metrics = {
        "accuracy": [], "precision": [], "recall": [], "f1": [],
        "mcc": [], "specificity": [], "auc": [],
        "train_time": [], "infer_time": []
    }
    fold_accuracies = []

    print(f"\n{'='*60}")
    print(f"Running {k_folds}-Fold CV for {model_name.upper()}")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{k_folds} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create model
        model = model_factory_fn()
        model.to(device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Data loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train
        start_time = time.time()
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                outputs = model(bx)
                loss = criterion(outputs, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

        train_time = time.time() - start_time

        # Validate
        model.eval()
        start_time = time.time()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                outputs = model(bx)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(by.cpu().numpy())

        infer_time = time.time() - start_time

        # Calculate metrics
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        probs = np.array(all_probs)

        metrics = calculate_metrics(labels, preds, probs)

        for key in fold_metrics:
            if key in metrics:
                fold_metrics[key].append(metrics[key])

        fold_metrics["train_time"].append(train_time)
        fold_metrics["infer_time"].append(infer_time / len(X_val))

        fold_accuracies.append(metrics["accuracy"])
        print(f"  Fold {fold}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, MCC={metrics['mcc']:.4f}")

    # Compute mean and std
    result = {
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_metrics["accuracy"])),
        "std_accuracy": float(np.std(fold_metrics["accuracy"])),
        "mean_precision": float(np.mean(fold_metrics["precision"])),
        "std_precision": float(np.std(fold_metrics["precision"])),
        "mean_recall": float(np.mean(fold_metrics["recall"])),
        "std_recall": float(np.std(fold_metrics["recall"])),
        "mean_f1": float(np.mean(fold_metrics["f1"])),
        "std_f1": float(np.std(fold_metrics["f1"])),
        "mean_mcc": float(np.mean(fold_metrics["mcc"])),
        "std_mcc": float(np.std(fold_metrics["mcc"])),
        "mean_specificity": float(np.mean(fold_metrics["specificity"])),
        "std_specificity": float(np.std(fold_metrics["specificity"])),
        "mean_auc": float(np.mean(fold_metrics["auc"])),
        "std_auc": float(np.std(fold_metrics["auc"])),
        "mean_train_time": float(np.mean(fold_metrics["train_time"])),
        "mean_infer_time": float(np.mean(fold_metrics["infer_time"])),
    }

    print(f"\n{model_name.upper()} {k_folds}-Fold CV Summary:")
    print(f"  Accuracy:    {result['mean_accuracy']*100:.2f}% (±{result['std_accuracy']*100:.2f}%)")
    print(f"  F1:          {result['mean_f1']:.4f} (±{result['std_f1']:.4f})")
    print(f"  MCC:         {result['mean_mcc']:.4f} (±{result['std_mcc']:.4f})")
    print(f"  Specificity: {result['mean_specificity']:.4f} (±{result['std_specificity']:.4f})")
    print(f"  AUC:         {result['mean_auc']:.4f} (±{result['std_auc']:.4f})")
    print(f"  Train Time:  {result['mean_train_time']:.2f}s | Infer Time: {result['mean_infer_time']*1000:.2f}ms")

    return result


def paired_t_test(fold_scores1: list, fold_scores2: list) -> dict:
    """
    Perform paired t-test between two sets of fold scores.

    Parameters
    ----------
    fold_scores1 : list
        Scores from model 1 across folds
    fold_scores2 : list
        Scores from model 2 across folds

    Returns
    -------
    dict
        T-test results
    """
    t_stat, p_val = stats.ttest_rel(fold_scores1, fold_scores2)
    mean_diff = np.mean(np.array(fold_scores1) - np.array(fold_scores2))

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "mean_difference": float(mean_diff),
        "conclusion": (
            f"Significant difference (p={p_val:.5f})" if p_val < 0.05
            else f"No significant difference (p={p_val:.5f})"
        )
    }


def mcnemar_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> dict:
    """
    Perform McNemar's test between two classifiers.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    pred1 : np.ndarray
        Predictions from classifier 1
    pred2 : np.ndarray
        Predictions from classifier 2

    Returns
    -------
    dict
        McNemar test results
    """
    # Confusion matrix for the difference
    both_correct = ((pred1 == y_true) & (pred2 == y_true)).sum()
    both_wrong = ((pred1 != y_true) & (pred2 != y_true)).sum()
    only1_correct = ((pred1 == y_true) & (pred2 != y_true)).sum()
    only2_correct = ((pred1 != y_true) & (pred2 == y_true)).sum()

    # McNemar's test
    b = only1_correct  # discordant pair 1
    c = only2_correct  # discordant pair 2

    if b + c > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
        p_value = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2 = 0
        p_value = 1.0

    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "b_discordant (A correct, B wrong)": int(b),
        "c_discordant (A wrong, B correct)": int(c),
        "conclusion": (
            "Significant difference between classifiers (p < 0.05)"
            if p_value < 0.05
            else "No significant difference between classifiers"
        )
    }


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    save_path: str = "confusion_matrix.png",
) -> np.ndarray:
    """
    Plot and save confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    classes : list, optional
        Class names
    normalize : bool
        Whether to normalize
    title : str
        Plot title
    save_path : str
        Path to save the plot

    Returns
    -------
    np.ndarray
        Confusion matrix
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    if classes is None:
        classes = ["Normal", "Attack"]

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
        fmt = '.2f'
    else:
        cm_norm = cm
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Confusion matrix saved to: {save_path}")
    return cm


def run_statistical_comparison(results_dict: dict) -> dict:
    """
    Run statistical tests comparing multiple models.

    Parameters
    ----------
    results_dict : dict
        Dictionary with model names as keys and metric dicts as values

    Returns
    -------
    dict
        Statistical test results
    """
    model_names = list(results_dict.keys())
    results = {}

    # Pairwise McNemar's tests
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_i = model_names[i]
            name_j = model_names[j]

            test_result = {
                "comparison": f"{name_i} vs {name_j}",
                "mcnemar": {
                    "p_value": None,
                    "interpretation": "Need prediction data"
                }
            }
            results[f"{name_i}_vs_{name_j}"] = test_result

    return results