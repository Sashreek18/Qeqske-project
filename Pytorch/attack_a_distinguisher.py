"""
ATTACK A: Randomness Distinguishing  [PyTorch version]
=======================================================
Exact copy of sk-learn/attack_a_distinguisher.py with:
  - Random Forest / SVM: unchanged (sklearn, CPU only)
  - New: MLP (PyTorch) added as a third classifier for GPU acceleration

All feature extraction and dataset building code is identical.
"""

import math
import random as pyrandom
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nn_torch import get_device

DEVICE = get_device()


# ---------------------------------------------------------------------------
# Feature extraction (unchanged from sklearn version)
# ---------------------------------------------------------------------------

def extract_features(numbers, window_size=8):
    features = []
    for i in range(0, len(numbers) - window_size + 1, window_size):
        window = numbers[i:i + window_size]
        if len(window) < window_size:
            continue

        bits = []
        for n in window:
            bits.extend(int(b) for b in format(n, "016b"))

        feat = {
            "mean": np.mean(window),
            "std": np.std(window),
            "min": np.min(window),
            "max": np.max(window),
            "range": np.max(window) - np.min(window),
            "median": np.median(window),
            "skew": _skewness(window),
            "bit_ones_ratio": sum(bits) / len(bits),
            "autocorr_lag1": _autocorr(window, lag=1),
            "diff_mean": np.mean(np.diff(window)) if len(window) > 1 else 0,
            "diff_std": np.std(np.diff(window)) if len(window) > 1 else 0,
            "longest_run_bits": _longest_run(bits),
            "entropy": _shannon_entropy(window),
            "mod3_balance": _mod_balance(window, 3),
            "even_ratio": sum(1 for x in window if x % 2 == 0) / len(window),
        }
        features.append(list(feat.values()))

    return np.array(features), list(feat.keys())


def _skewness(window):
    n = len(window)
    mean = np.mean(window)
    std = np.std(window)
    if std == 0:
        return 0.0
    return (sum((x - mean) ** 3 for x in window) / n) / (std ** 3)


def _autocorr(window, lag=1):
    if len(window) <= lag:
        return 0.0
    x = np.array(window[:-lag])
    y = np.array(window[lag:])
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _longest_run(bits):
    max_run = run = 0
    for b in bits:
        if b == 1:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _shannon_entropy(window):
    from collections import Counter
    counts = Counter(window)
    n = len(window)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _mod_balance(window, m):
    from collections import Counter
    counts = Counter(x % m for x in window)
    expected = len(window) / m
    return sum(abs(counts.get(i, 0) - expected) for i in range(m)) / len(window)


def build_dataset(qrng_numbers, mersenne_numbers, window_size=8):
    qrng_feats, feat_names = extract_features(qrng_numbers, window_size)
    mersenne_feats, _ = extract_features(mersenne_numbers, window_size)

    X = np.vstack([qrng_feats, mersenne_feats])
    y = np.array([1] * len(qrng_feats) + [0] * len(mersenne_feats))

    return X, y, feat_names


# ---------------------------------------------------------------------------
# PyTorch MLP Classifier (tabular, GPU-accelerated)
# ---------------------------------------------------------------------------

class _TabularMLP(nn.Module):
    def __init__(self, n_features, hidden_sizes=(64, 32)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_torch_mlp(X_train, y_train, n_features,
                    hidden_sizes=(64, 32), lr=1e-3, epochs=50,
                    batch_size=256, seed=42):
    torch.manual_seed(seed)
    net = _TabularMLP(n_features, hidden_sizes).to(DEVICE)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()

    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    X_norm = ((X_train - mu) / sd).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(X_norm).to(DEVICE),
        torch.from_numpy(y_train.astype(np.float32)).to(DEVICE)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            optimiser.zero_grad()
            criterion(net(Xb), yb).backward()
            optimiser.step()

    return net, mu, sd


@torch.no_grad()
def predict_torch_mlp(net, mu, sd, X):
    net.eval()
    X_norm = ((X - mu) / (sd + 1e-8)).astype(np.float32)
    Xt = torch.from_numpy(X_norm).to(DEVICE)
    return net(Xt).cpu().numpy()


# ---------------------------------------------------------------------------
# Train and evaluate classifiers
# ---------------------------------------------------------------------------

def run_attack_a(qrng_numbers, mersenne_numbers, window_size=8, verbose=True):
    X, y, feat_names = build_dataset(qrng_numbers, mersenne_numbers, window_size)

    if verbose:
        print(f"Device          : {DEVICE}")
        print(f"Dataset built   : {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  QRNG samples  : {sum(y == 1)}")
        print(f"  MT samples    : {sum(y == 0)}")

    if len(set(y)) < 2 or X.shape[0] < 10:
        print("Not enough samples for a meaningful evaluation.")
        return None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # --- Random Forest (sklearn, unchanged) ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    rf_acc_mean = np.mean(rf_scores)
    rf_acc_std = np.std(rf_scores)
    rf_ci_margin = 1.96 * (rf_acc_std / np.sqrt(5))
    results["RandomForest"] = {"mean": rf_acc_mean, "std": rf_acc_std,
                                "ci": rf_ci_margin, "model": rf}

    # --- SVM (sklearn, unchanged) ---
    svm = SVC(kernel="rbf", probability=False, random_state=42, max_iter=2000)
    svm_scores = cross_val_score(svm, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    svm_acc_mean = np.mean(svm_scores)
    svm_acc_std = np.std(svm_scores)
    svm_ci_margin = 1.96 * (svm_acc_std / np.sqrt(5))
    results["SVM"] = {"mean": svm_acc_mean, "std": svm_acc_std,
                      "ci": svm_ci_margin, "model": svm}

    # --- PyTorch MLP (GPU/MPS accelerated) ---
    mlp_accs = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        net, mu, sd = train_torch_mlp(X[tr], y[tr], X.shape[1])
        prob = predict_torch_mlp(net, mu, sd, X[te])
        pred = (prob >= 0.5).astype(int)
        mlp_accs.append((pred == y[te]).mean())
    mlp_mean = np.mean(mlp_accs)
    mlp_std  = np.std(mlp_accs)
    mlp_ci   = 1.96 * (mlp_std / np.sqrt(5))
    results["MLP (PyTorch)"] = {"mean": mlp_mean, "std": mlp_std, "ci": mlp_ci}

    if verbose:
        print(f"\n{'=' * 60}")
        print("ATTACK A RESULTS (PyTorch version)")
        print(f"{'=' * 60}")
        for name, r in results.items():
            verdict = ("DISTINGUISHABLE (potential leak!)" if r["mean"] > 0.60
                       else "NOT reliably distinguishable (good for QRNG security)")
            print(f"  {name:20s} accuracy = {r['mean']:.2%} ± {r['std']:.2%}"
                  f"  (95% CI ±{r['ci']:.2%}) -> {verdict}")
        print(f"\n  Baseline (random guess) = 50.00%")
        print(f"  NOTE: {X.shape[0]} samples via 5-Fold CV. "
              f"MLP trained on {DEVICE}.")

        # Feature importance from Random Forest
        rf.fit(X, y)
        importances = rf.feature_importances_
        top_features = sorted(zip(feat_names, importances),
                               key=lambda x: -x[1])[:5]
        print(f"\n  Top distinguishing features (Random Forest importance):")
        for fname, imp in top_features:
            print(f"    {fname:20s} {imp:.4f}")

    return results


if __name__ == "__main__":
    with open("../qrng_combined.txt") as f:
        qrng_numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    with open("../mersenne_large_baseline.txt") as f:
        mersenne_numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    run_attack_a(qrng_numbers, mersenne_numbers, window_size=8)
