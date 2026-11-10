"""Explainable AI (XAI) Module for SecureFed-NIDS"""
import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def _to_scalar(val):
    if hasattr(val, "item"):
        return float(val.item())
    arr = np.asarray(val)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return float(np.mean(arr))

class SHAPExplainer:
    def __init__(self, model: nn.Module, X_train: np.ndarray, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or [f"Feature_{i+1}" for i in range(X_train.shape[1])]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.explainer = None

    def create_explainer(self, background_samples=50):
        if background_samples < len(self.X_train):
            idx = np.random.choice(len(self.X_train), background_samples, replace=False)
            bg = self.X_train[idx]
        else:
            bg = self.X_train

        def model_predict(x):
            self.model.eval()
            with torch.no_grad():
                xt = torch.FloatTensor(x).to(self.device)
                probs = torch.softmax(self.model(xt), dim=1).cpu().numpy()
            return probs

        self.explainer = shap.KernelExplainer(model_predict, bg, silent=True)
        return self.explainer

    def explain_instance(self, instance: np.ndarray, nsamples=100):
        if self.explainer is None:
            self.create_explainer()

        raw = self.explainer.shap_values(instance.reshape(1, -1), nsamples=nsamples, silent=True)
        if isinstance(raw, list):
            sv = raw[1][0] if len(raw) > 1 else raw[0][0]
        elif getattr(raw, "ndim", 0) == 3:
            sv = raw[0, :, 1] if raw.shape[2] > 1 else raw[0, :, 0]
        elif getattr(raw, "ndim", 0) == 2:
            sv = raw[0]
        else:
            sv = np.asarray(raw).reshape(-1)

        feat_imp = {}
        for name, val in zip(self.feature_names, sv):
            s_val = _to_scalar(val)
            feat_imp[name] = {
                "shap_value": s_val,
                "abs_importance": abs(s_val),
                "impact": "Positive (Attack indicator)" if s_val > 0 else "Negative (Benign indicator)"
            }

        sorted_f = sorted(feat_imp.items(), key=lambda x: x[1]["abs_importance"], reverse=True)
        return {
            "feature_importance": feat_imp,
            "top_5_features": [f[0] for f in sorted_f[:5]]
        }

def perturbation_test(model: nn.Module, X: np.ndarray, explanation: dict, top_k: int = 5) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        orig = model(torch.FloatTensor(X).to(device)).argmax(dim=1).cpu().numpy()

    top_feats = explanation.get("top_5_features", [])[:top_k]
    X_pert = X.copy()

    for f_name in top_feats:
        try:
            if f_name.startswith("Feature_"):
                idx = int(f_name.split("_")[1]) - 1
            else:
                idx = 0
            if idx < X.shape[1]:
                noise = np.random.normal(0, 0.5 * (np.std(X[:, idx]) + 1e-6), X.shape[0])
                X_pert[:, idx] += noise
        except Exception:
            continue

    with torch.no_grad():
        pert = model(torch.FloatTensor(X_pert).to(device)).argmax(dim=1).cpu().numpy()

    return float(np.mean(orig != pert))
