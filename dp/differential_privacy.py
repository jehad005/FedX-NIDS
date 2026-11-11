"""
Differential Privacy Module for SecureFed-NIDS
===============================================
Implements DP-SGD with gradient clipping and noise injection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple
import math

from config import DP_CONFIG


class DPSGD:
    """
    Differentially Private Stochastic Gradient Descent.

    Implements:
    - Gradient clipping
    - Gaussian noise injection
    - Privacy budget tracking
    """

    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or DP_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DP parameters
        self.epsilon = self.config.get("epsilon", 1.0)
        self.delta = self.config.get("delta", 1e-5)
        self.clipping_norm = self.config.get("clipping_norm", 1.0)
        self.noise_multiplier = self.config.get("noise_multiplier", 1.1)

        # Privacy accounting
        self.privacy_spent = 0.0
        self.steps = 0
        self.sampling_rate = 1.0  # Will be set based on batch size

        # Track gradient norms
        self.gradient_norms = []

    def clip_gradients(self) -> float:
        """
        Clip gradients per-layer and compute total gradient norm.

        Returns:
            Average gradient norm before clipping
        """
        total_norm = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = math.sqrt(total_norm)
        self.gradient_norms.append(total_norm)

        # Clip gradients
        clip_coef = self.clipping_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

        return total_norm

    def add_noise(self) -> None:
        """
        Add Gaussian noise to gradients for differential privacy.
        """
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.clipping_norm,
                    size=param.grad.shape,
                    device=param.grad.device,
                    dtype=param.grad.dtype,
                )
                param.grad.data.add_(noise)

    def compute_privacy_spent(self, sampling_rate: float, steps: int) -> float:
        """
        Compute privacy budget spent using RDP accounting approximation.

        Args:
            sampling_rate: Batch size / dataset size
            steps: Number of training steps

        Returns:
            Total epsilon spent
        """
        q = sampling_rate
        T = steps
        sigma = self.noise_multiplier

        # Improved RDP-based accounting for Gaussian mechanism
        # Using the standard formula: ε ≈ q * sqrt(2 * T * log(1/δ)) / σ
        import math
        if q <= 0 or sigma <= 0:
            return float('inf')

        # RDP accounting with alpha = 10
        alpha = 10.0
        rdp = (alpha * q ** 2 * T) / (2 * sigma ** 2)
        epsilon = rdp + math.log(1 / self.delta) / (alpha - 1)

        return min(epsilon, 1000.0)  # Cap to avoid overflow

    def get_privacy_budget(self) -> Dict[str, float]:
        """Get current privacy budget status."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "epsilon_spent": self.privacy_spent,
            "epsilon_remaining": max(0, self.epsilon - self.privacy_spent),
            "noise_multiplier": self.noise_multiplier,
            "clipping_norm": self.clipping_norm,
            "steps": self.steps
        }

    def is_budget_exceeded(self) -> bool:
        """Check if privacy budget has been exceeded."""
        return self.privacy_spent >= self.epsilon


class DPTrainer:
    """Trainer with differential privacy."""

    def __init__(self, model: nn.Module, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or DP_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # DP-SGD instance
        self.dp_sgd = DPSGD(model, config)

        # Optimizer (DP requires specific optimizer settings)
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.get("learning_rate", 0.01),
            momentum=0.9
        )

        self.criterion = nn.CrossEntropyLoss()

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "privacy_spent": [],
            "gradient_norms": []
        }

    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray,
                    batch_size: int = 256) -> Tuple[float, float]:
        """
        Train for one epoch with DP-SGD.

        Args:
            X_train, y_train: Training data
            batch_size: Batch size

        Returns:
            Tuple of (average_loss, accuracy)
        """
        if len(X_train) == 0:
            raise ValueError("Training set X_train cannot be empty.")
        batch_size = max(1, int(batch_size))

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        # Create batches
        indices = np.random.permutation(len(X_train))
        sampling_rate = min(1.0, batch_size / len(X_train))

        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = torch.FloatTensor(X_train[batch_indices]).to(self.device)
            batch_y = torch.LongTensor(y_train[batch_indices]).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()

            # DP steps: clip gradients
            grad_norm = self.dp_sgd.clip_gradients()

            # DP steps: add noise
            self.dp_sgd.add_noise()

            # Update weights
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

            # Update privacy accounting
            self.dp_sgd.steps += 1
            num_batches += 1

        # Compute privacy spent
        self.dp_sgd.privacy_spent = self.dp_sgd.compute_privacy_spent(
            sampling_rate=sampling_rate,
            steps=self.dp_sgd.steps
        )

        avg_loss = total_loss / max(1, num_batches)
        accuracy = correct / max(total, 1)

        return avg_loss, accuracy

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 50, batch_size: int = 256) -> Dict[str, Any]:
        """
        Complete training loop with DP.

        Returns:
            Training history dictionary
        """
        print("\n" + "="*70)
        print(f"Training with Differential Privacy")
        print(f"Privacy Budget: ε={self.dp_sgd.epsilon}, δ={self.dp_sgd.delta}")
        print(f"Noise Multiplier: {self.dp_sgd.noise_multiplier}")
        print(f"Clipping Norm: {self.dp_sgd.clipping_norm}")
        print("="*70)

        val_acc = self.evaluate(X_val, y_val) if len(X_val) > 0 and epochs <= 0 else 0.0

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)

            # Validate
            val_acc = self.evaluate(X_val, y_val)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["privacy_spent"].append(self.dp_sgd.privacy_spent)
            self.history["gradient_norms"].append(
                np.mean(self.dp_sgd.gradient_norms[-100:]) if self.dp_sgd.gradient_norms else 0
            )

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"ε spent: {self.dp_sgd.privacy_spent:.4f}")

            # Check privacy budget
            if self.dp_sgd.is_budget_exceeded():
                print(f"\nPrivacy budget exceeded at epoch {epoch+1}!")
                break

        print("\n" + "="*70)
        print(f"DP Training Complete!")
        print(f"Final ε spent: {self.dp_sgd.privacy_spent:.4f}")
        print(f"Final Val Accuracy: {val_acc:.4f}")
        print("="*70)

        return self.history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model accuracy."""
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)

            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(y_tensor).sum().item() / len(y)

        return accuracy


def compare_dp_vs_non_dp(model: nn.Module,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          privacy_levels: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Compare model performance with and without differential privacy.

    Args:
        model: Model architecture
        X_train, y_train: Training data
        X_val, y_val: Validation data
        privacy_levels: Dict of privacy level configurations

    Returns:
        Comparison results dictionary
    """
    privacy_levels = privacy_levels or DP_CONFIG.get("privacy_levels", {
        "none": {"epsilon": float('inf'), "noise_multiplier": 0},
        "low": {"epsilon": 2.0, "noise_multiplier": 0.5},
        "medium": {"epsilon": 1.0, "noise_multiplier": 1.0},
        "high": {"epsilon": 0.5, "noise_multiplier": 2.0}
    })

    results = {}

    for level_name, level_config in privacy_levels.items():
        print(f"\n{'='*60}")
        print(f"Training with Privacy Level: {level_name.upper()}")
        print(f"{'='*60}")

        if level_name == "none":
            # Train without DP
            from models.training import Trainer
            trainer = Trainer(model, config={"epochs": 10})
            history = trainer.fit(X_train, y_train, X_val, y_val,
                                 experiment_name=f"DP_Comparison_{level_name}")
            results[level_name] = {
                "accuracy": history["val_acc"][-1],
                "privacy_spent": 0.0
            }
        else:
            # Train with DP
            dp_config = {**DP_CONFIG, **level_config}
            trainer = DPTrainer(model, dp_config)
            history = trainer.fit(X_train, y_train, X_val, y_val, epochs=10)
            results[level_name] = {
                "accuracy": trainer.evaluate(X_val, y_val),
                "privacy_spent": trainer.dp_sgd.privacy_spent
            }

    return results


if __name__ == "__main__":
    # Test DP-SGD
    from models.architectures import ModelFactory
    from sklearn.datasets import make_classification

    print("Generating test data...")
    X, y = make_classification(n_samples=1000, n_features=78, n_informative=50,
                               n_classes=2, random_state=42)

    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]

    model = ModelFactory.create_model("mlp", 78, 2)

    dp_config = {
        "epsilon": 1.0,
        "delta": 1e-5,
        "clipping_norm": 1.0,
        "noise_multiplier": 1.1
    }

    trainer = DPTrainer(model, dp_config)
    history = trainer.fit(X_train, y_train, X_val, y_val, epochs=5)

    print("\nDP-SGD test completed!")