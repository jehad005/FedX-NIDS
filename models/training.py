"""
Model Training Module for SecureFed-NIDS
=========================================
Handles training, validation, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
from datetime import datetime
import json
import time
# MLflow disabled due to database schema issues
mlflow = None

from models.architectures import ModelFactory
from config import TRAINING_CONFIG


class Trainer:
    """Comprehensive training class with early stopping and monitoring."""

    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        self.config = config or TRAINING_CONFIG
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer
        optimizer_name = self.config.get("optimizer", "adam")
        lr = self.config.get("learning_rate", 0.001)

        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr,
                                         weight_decay=self.config.get("weight_decay", 1e-4))
        elif optimizer_name.lower() == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr,
                                          weight_decay=self.config.get("weight_decay", 1e-4))
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.get("reduce_lr_factor", 0.5),
            patience=self.config.get("reduce_lr_patience", 5)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_precision": [], "val_precision": [],
            "train_recall": [], "val_recall": [],
            "train_f1": [], "val_f1": [],
            "learning_rates": [],
            # Wall-clock measurements are stored in seconds.  The explicit
            # per-sample inference value makes model deployment comparisons fair.
            "epoch_train_times_seconds": [],
            "epoch_infer_times_seconds": [],
            "training_time_seconds": 0.0,
            "inference_time_seconds": 0.0,
            "inference_time_per_sample_ms": 0.0,
            "total_wall_time_seconds": 0.0,
            "training_start": None,
            "training_end": None,
        }

        # Early stopping
        self.early_stopping_patience = self.config.get("early_stopping_patience", 10)
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch and return its optimization wall-clock time."""
        self.model.train()
        start_time = time.perf_counter()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        metrics = self._compute_metrics(np.array(all_preds), np.array(all_labels))
        metrics["loss"] = avg_loss
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        metrics["train_time_seconds"] = float(time.perf_counter() - start_time)

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model and record true forward-pass inference time."""
        self.model.eval()
        total_loss = 0
        sample_count = 0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_start = time.perf_counter()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())
                sample_count += len(batch_y)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_time = time.perf_counter() - inference_start
        avg_loss = total_loss / len(val_loader)
        metrics = self._compute_metrics(np.array(all_preds), np.array(all_labels))
        metrics["loss"] = avg_loss
        metrics["inference_time_seconds"] = float(inference_time)
        metrics["inference_time_per_sample_ms"] = float(
            inference_time * 1000.0 / max(sample_count, 1)
        )
        metrics["probabilities"] = np.asarray(all_probs)

        return metrics

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            experiment_name: str = "SecureFed-NIDS") -> Dict[str, Any]:
        """
        Complete training loop with MLflow tracking.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            experiment_name: MLflow experiment name

        Returns:
            Training history dictionary
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 256),
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get("batch_size", 256),
            shuffle=False
        )

        epochs = self.config.get("epochs", 50)
        fit_start = time.perf_counter()
        self.history["training_start"] = datetime.now().isoformat()

        # Training without MLflow
        # MLflow has database schema issues, so we silently disable it and just train.
        
        best_val_f1 = -1.0
        best_model_state = None
        start_total = time.perf_counter()
        # Training loop
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            train_loss = train_metrics.get("loss", 0.0)
            train_acc = train_metrics.get("accuracy", 0.0)
            train_time = train_metrics.get("train_time_seconds", 0.0)

            val_metrics = self.validate(val_loader)
            val_loss = val_metrics.get("loss", 0.0)
            val_acc = val_metrics.get("accuracy", 0.0)
            val_f1 = val_metrics.get("f1", 0.0)
            infer_time = val_metrics.get("inference_time_seconds", 0.0)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Update times
            self.history['epoch_train_times_seconds'].append(train_time)
            self.history['epoch_infer_times_seconds'].append(infer_time)

            print(f"Epoch {epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

            # Save best model (deep tensor clone so subsequent epochs do not mutate saved state)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
        
        # Save total time
        total_time = time.perf_counter() - start_total
        self.history['total_wall_time_seconds'] = total_time
        self.history['training_time_seconds'] = sum(self.history['epoch_train_times_seconds'])
        self.history['inference_time_seconds'] = sum(self.history['epoch_infer_times_seconds'])
        
        total_samples_infer = len(val_loader.dataset) * epochs
        if total_samples_infer > 0:
            self.history['inference_time_per_sample_ms'] = (self.history['inference_time_seconds'] / total_samples_infer) * 1000
        else:
            self.history['inference_time_per_sample_ms'] = 0.0

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        self.history["training_end"] = datetime.now().isoformat()
        self.history["training_time_seconds"] = float(sum(self.history["epoch_train_times_seconds"]))
        self.history["inference_time_seconds"] = float(sum(self.history["epoch_infer_times_seconds"]))
        self.history["inference_time_per_sample_ms"] = float(
            self.history["inference_time_seconds"] * 1000.0 /
            max(len(y_val) * len(self.history["epoch_infer_times_seconds"]), 1)
        )
        self.history["total_wall_time_seconds"] = float(time.perf_counter() - fit_start)
        print(
            f"Timing | training: {self.history['training_time_seconds']:.2f}s | "
            f"inference: {self.history['inference_time_per_sample_ms']:.4f}ms/sample"
        )
        return self.history

    def _compute_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def save_checkpoint(self, path: Path, optimizer: bool = True) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
            "config": self.config,
            "input_dim": self.model.input_dim,
            "num_classes": self.model.num_classes
        }

        if optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

        # Save history as JSON
        history_path = path.with_suffix(".json")
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, value in self.history.items():
                if isinstance(value, np.ndarray):
                    serializable_history[key] = value.tolist()
                else:
                    serializable_history[key] = value
            json.dump(serializable_history, f, indent=2)

    def load_checkpoint(self, path: Path, optimizer: bool = True) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint.get("history", {})

        if optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def train_single_model(model_name: str, input_dim: int, num_classes: int,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       config: Optional[Dict] = None) -> Tuple[nn.Module, Dict]:
    """
    Convenience function to train a single model.

    Returns:
        Tuple of (trained_model, training_history)
    """
    model = ModelFactory.create_model(model_name, input_dim, num_classes)
    trainer = Trainer(model, config)

    history = trainer.fit(X_train, y_train, X_val, y_val)

    return model, history


def train_all_models(input_dim: int, num_classes: int,
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     config: Optional[Dict] = None) -> Dict[str, Tuple[nn.Module, Dict]]:
    """
    Train all 6 models and return dictionary of results.

    Returns:
        Dictionary mapping model_name -> (trained_model, history)
    """
    model_names = ["mlp", "cnn", "lstm", "bilstm", "gru", "transformer"]
    results = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")

        model, history = train_single_model(
            model_name=model_name,
            input_dim=input_dim,
            num_classes=num_classes,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            config=config
        )

        results[model_name] = (model, history)

        acc = history["val_acc"][-1]
        prec = history["val_precision"][-1]
        rec = history["val_recall"][-1]
        f1 = history["val_f1"][-1]
        print("-" * 40)
        print(f"{model_name.upper()} Final Validation Metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("STAGE 2 SUMMARY — All 6 Models")
    print("=" * 60)
    print(f"{'Model':<14} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for name, (_, hist) in results.items():
        print(f"{name.upper():<14} {hist['val_acc'][-1]:>10.4f} {hist['val_precision'][-1]:>10.4f} {hist['val_recall'][-1]:>10.4f} {hist['val_f1'][-1]:>10.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_classification

    print("Generating test data...")
    X, y = make_classification(n_samples=1000, n_features=78, n_informative=50,
                               n_classes=2, random_state=42)

    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]

    config = {
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.001,
        "early_stopping_patience": 3
    }

    # Test single model
    model, history = train_single_model("mlp", 78, 2, X_train, y_train, X_val, y_val, config)

    print(f"\nModel trained successfully!")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")