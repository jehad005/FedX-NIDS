"""
Ensemble Module for SecureFed-NIDS
===================================
Selects top-k models and creates ensemble for improved intrusion detection.
Supports Soft Voting, Hard Voting, Weighted Voting, and Stacking Meta-Learner.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import ENSEMBLE_CONFIG
from models.evaluation import calculate_metrics


class EnsembleModel(nn.Module):
    """Ensemble of top-k models using Soft/Hard Voting, Weighted Average, or Stacking Meta-Learner."""

    def __init__(
        self,
        models: List[nn.Module],
        method: str = "voting",
        voting_type: str = "soft",
        weights: Optional[List[float]] = None,
        num_classes: int = 2
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method.lower()
        self.voting_type = voting_type.lower()
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.num_classes = num_classes

        # Stacking Meta-Learner (PyTorch nn.Sequential for end-to-end differentiability and serialization)
        meta_input_dim = len(models) * num_classes
        self.meta_learner = nn.Sequential(
            nn.Linear(meta_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        if self.method == "voting":
            return self._voting_forward(x)
        elif self.method == "stacking":
            return self._stacking_forward(x)
        elif self.method == "weighted":
            return self._weighted_forward(x)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def _voting_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Voting ensemble forward pass (Soft or Hard)."""
        all_probs = []
        grad_enabled = self.training or x.requires_grad

        with torch.set_grad_enabled(grad_enabled):
            for model in self.models:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)
            avg_probs = torch.mean(torch.stack(all_probs), dim=0)

        if self.training or self.voting_type == "soft":
            return torch.log(avg_probs + 1e-10)

        # Hard voting for inference evaluation; preserve gradients via STE when x.requires_grad is set
        votes = torch.stack([p.argmax(dim=1) for p in all_probs], dim=1)
        majority = torch.mode(votes, dim=1).values
        result = torch.zeros_like(avg_probs)
        result.scatter_(1, majority.unsqueeze(1), 1.0)
        if grad_enabled:
            result = result + (avg_probs - avg_probs.detach())
        return torch.log(result + 1e-10)

    def _weighted_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted average soft voting ensemble."""
        weighted_probs = []
        grad_enabled = self.training or x.requires_grad

        with torch.set_grad_enabled(grad_enabled):
            for model, weight in zip(self.models, self.weights):
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                weighted_probs.append(probs * weight)
            avg_probs = sum(weighted_probs)
            return torch.log(avg_probs + 1e-10)

    def _stacking_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stacking ensemble using PyTorch Meta-Learner."""
        all_probs = []
        grad_enabled = self.training or x.requires_grad

        with torch.set_grad_enabled(grad_enabled):
            for model in self.models:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)
            meta_features = torch.cat(all_probs, dim=1)
            return self.meta_learner(meta_features)

    def fit_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 5,
        batch_size: int = 128,
        lr: float = 0.005,
        device: str = "cpu"
    ):
        """Trains the stacking meta-learner on base model prediction outputs."""
        print(f"--> Training Stacking Meta-Learner on {len(X_train)} samples ({epochs} epochs)...")
        self.to(device)
        for m in self.models:
            m.eval()

        # Collect base predictions (frozen)
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.meta_learner.train()
        for ep in range(epochs):
            total_loss = 0.0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                with torch.no_grad():
                    base_probs = [torch.softmax(m(bx), dim=1) for m in self.models]
                    meta_in = torch.cat(base_probs, dim=1)

                optimizer.zero_grad()
                out = self.meta_learner(meta_in)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"    Meta-Learner Epoch {ep+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract concatenated features from all base models for representation learning."""
        feats = []
        for model in self.models:
            if hasattr(model, 'get_features'):
                try:
                    f = model.get_features(x)
                except Exception:
                    f = model(x)
            else:
                f = model(x)
            if f.dim() > 2:
                f = f.view(f.size(0), -1)
            feats.append(f)
        return torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]


class TopKSelector:
    """Select top-k best performing models."""

    def __init__(self, top_k: int = 2, metric: str = "accuracy"):
        self.top_k = top_k
        self.metric = metric

    def select(
        self,
        model_results: Dict[str, Dict[str, float]],
        X_val: np.ndarray,
        y_val: np.ndarray,
        models: Dict[str, nn.Module]
    ) -> Tuple[List[str], List[nn.Module]]:
        """Select top-k models based on validation performance."""
        scores = []

        for model_name, metrics in model_results.items():
            if self.metric in metrics:
                score = metrics[self.metric]
            else:
                model = models[model_name]
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_val)
                    logits = model(X_tensor)
                    preds = logits.argmax(dim=1).numpy()

                if self.metric == "accuracy":
                    score = accuracy_score(y_val, preds)
                elif self.metric == "precision":
                    score = precision_score(y_val, preds, average='weighted', zero_division=0)
                elif self.metric == "recall":
                    score = recall_score(y_val, preds, average='weighted', zero_division=0)
                elif self.metric == "f1":
                    score = f1_score(y_val, preds, average='weighted', zero_division=0)
                else:
                    score = metrics.get("accuracy", 0)

            scores.append((model_name, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        print(f"\nModel Rankings (by {self.metric}):")
        print("-" * 45)
        for i, (name, score) in enumerate(scores):
            print(f"  [{i+1}] {name.upper():<15}: {score*100:.2f}%" if score <= 1.0 else f"  [{i+1}] {name.upper():<15}: {score:.4f}")

        top_models = scores[:self.top_k]
        selected_names = [name for name, _ in top_models]
        selected_models = [models[name] for name in selected_names]

        print(f"\n--> Selected Top-{self.top_k}: {[n.upper() for n in selected_names]}")
        return selected_names, selected_models


class EnsembleTrainer:
    """Trainer and orchestrator for ensemble model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or ENSEMBLE_CONFIG
        self.selector = TopKSelector(
            top_k=self.config.get("top_k", 2),
            metric=self.config.get("selection_metric", "accuracy")
        )
        self.ensemble = None

    def train_ensemble(
        self,
        model_results: Dict[str, Dict[str, float]],
        models: Dict[str, nn.Module],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: Optional[str] = None,
        voting_type: Optional[str] = None
    ) -> EnsembleModel:
        """Create and train ensemble from top-k models."""
        selected_names, selected_models = self.selector.select(
            model_results, X_val, y_val, models
        )

        method = method or self.config.get("ensemble_method", "voting")
        voting_type = voting_type or self.config.get("voting_type", "soft")
        num_classes = len(np.unique(y_train))

        self.ensemble = EnsembleModel(
            models=selected_models,
            method=method,
            voting_type=voting_type,
            num_classes=num_classes
        )

        if method == "stacking":
            self.ensemble.fit_meta_learner(X_train, y_train, epochs=5)

        ensemble_metrics = self.evaluate_ensemble(X_val, y_val)

        print(f"\nEnsemble Evaluation ({method.upper()} - {voting_type.upper()}):")
        print(f"  -> Accuracy : {ensemble_metrics['accuracy']*100:.2f}%")
        print(f"  -> Precision: {ensemble_metrics['precision']:.4f}")
        print(f"  -> Recall   : {ensemble_metrics['recall']:.4f}")
        print(f"  -> F1-Score : {ensemble_metrics['f1']:.4f}")

        return self.ensemble

    def evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate the ensemble with metrics and measured inference latency."""
        if self.ensemble is None:
            raise ValueError("Ensemble not trained yet")

        self.ensemble.eval()
        device = next(self.ensemble.parameters()).device
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            logits = self.ensemble(X_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).detach().cpu().numpy()
            probabilities = probs.detach().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = float(time.perf_counter() - started)

        metrics = calculate_metrics(y, preds, probabilities)
        # Keep both spellings for compatibility with existing dashboard/report
        # code (``auc`` is used by evaluation.py; ``roc_auc`` by old plots).
        metrics.update({
            "auc": float(metrics.get("auc", 0.0)),
            "roc_auc": float(metrics.get("auc", 0.0)),
            "predictions": preds,
            "probabilities": probabilities,
            "inference_time_seconds": inference_seconds,
            "inference_time_per_sample_ms": inference_seconds * 1000.0 / max(len(y), 1),
        })
        return metrics


def compare_with_without_ensemble(
    model_results: Dict[str, Dict[str, float]],
    ensemble_metrics: Dict[str, float]
) -> Dict[str, Any]:
    """Compare performance with and without ensemble."""
    sorted_models = sorted(model_results.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True)
    best_single = sorted_models[0] if sorted_models else (None, {})
    second_best = sorted_models[1] if len(sorted_models) > 1 else (None, {})

    best_single_acc = best_single[1].get("accuracy", 0.0)
    ensemble_acc = ensemble_metrics.get("accuracy", 0.0)
    improvement = ensemble_acc - best_single_acc

    return {
        "best_single_model": best_single[0],
        "best_single_accuracy": best_single_acc,
        "second_best_model": second_best[0],
        "second_best_accuracy": second_best[1].get("accuracy", 0.0),
        "ensemble_accuracy": ensemble_acc,
        "ensemble_f1": ensemble_metrics.get("f1", 0.0),
        "accuracy_improvement": improvement
    }
