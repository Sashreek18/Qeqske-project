"""
System Comparison: Random Baseline vs. Standard ML-KEM vs. QEQSKE  [PyTorch version]
======================================================================================
Drop-in replacement for sk-learn/system_comparison.py.

Changes vs. sklearn version:
  - Logistic Regression: UNCHANGED (sklearn)
  - Random Forest: now supplemented by PyTorch MLP as an additional model.
    Best of the three is reported.

All timing collection (time_ml_kem, time_qeqske) is identical.
"""

import time
import os
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import attack_c_stats
from qrng_handler import QRNGSource
from qeqske_core import QEQSKERandom, key_gen as qeqske_keygen
from attack_c_leaky_vs_clean import key_gen_leaky as qeqske_keygen_leaky, _secret_properties
from nn_torch import get_device

from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768

DEVICE = get_device()


# ---------------------------------------------------------------------------
# Timing collection (unchanged from sklearn version)
# ---------------------------------------------------------------------------

def time_ml_kem(num_trials=300, param_set=ML_KEM_512):
    rows = []
    for trial in range(num_trials):
        t0 = time.perf_counter()
        ek, dk = param_set.keygen()
        elapsed = time.perf_counter() - t0
        weight = sum(bin(byte).count("1") for byte in dk)
        rows.append({"timing_us": elapsed * 1e6, "weight": weight})

    weights = [r["weight"] for r in rows]
    median_w = np.median(weights)
    for r in rows:
        r["label"] = int(r["weight"] > median_w)
    return rows


def time_qeqske(numbers, keygen_fn, num_trials=300, n=4, k=2):
    rows = []
    cursor = 0
    for trial in range(num_trials):
        batch = numbers[cursor:cursor + 60]
        if len(batch) < 30:
            break
        qrng = QRNGSource(numbers=list(batch))
        rng  = QEQSKERandom(qrng)
        t0   = time.perf_counter()
        keys = keygen_fn(rng, n, k)
        elapsed = time.perf_counter() - t0
        cursor += qrng.total_consumed
        weight, _ = _secret_properties(keys["private_key"])
        rows.append({"timing_us": elapsed * 1e6, "label": int(weight > 5)})
    return rows


# ---------------------------------------------------------------------------
# PyTorch MLP helper
# ---------------------------------------------------------------------------

def _torch_mlp_acc(X, y, rskf, hidden=(16,), lr=1e-3,
                   epochs=100, batch_size=64, seed=42):
    accs = []
    for tr, te in rskf.split(X, y):
        torch.manual_seed(seed)
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
        Xtr = ((X[tr] - mu) / sd).astype(np.float32)
        Xte = ((X[te] - mu) / sd).astype(np.float32)
        ytr = y[tr].astype(np.float32)

        layers, in_d = [], X.shape[1]
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.ReLU()]; in_d = h
        layers += [nn.Linear(in_d, 1), nn.Sigmoid()]
        net = nn.Sequential(*layers).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr).to(DEVICE),
                           torch.from_numpy(ytr).to(DEVICE)),
            batch_size=batch_size, shuffle=True)
        net.train()
        for _ in range(epochs):
            for Xb, yb in loader:
                opt.zero_grad()
                nn.BCELoss()(net(Xb).squeeze(-1), yb).backward()
                opt.step()
        with torch.no_grad():
            net.eval()
            prob = net(torch.from_numpy(Xte).to(DEVICE)).squeeze(-1).cpu().numpy()
        accs.append(((prob >= 0.5).astype(int) == y[te]).mean())
    return np.array(accs)


# ---------------------------------------------------------------------------
# Attack runner
# ---------------------------------------------------------------------------

def run_attack(rows, name):
    X = np.array([[r["timing_us"]] for r in rows])
    y = np.array([r["label"] for r in rows])

    if len(set(y)) < 2:
        return {"system": name, "baseline": None,
                "best_model": None, "best_acc": None}

    baseline = max(np.mean(y == 0), np.mean(y == 1))
    lr_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rf       = RandomForestClassifier(n_estimators=200, random_state=42)
    rskf     = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    ml_stats_lr = attack_c_stats.compute_ml_stats(lr_model, X, y, rskf)
    ml_stats_rf = attack_c_stats.compute_ml_stats(rf, X, y, rskf)
    acc_mlp     = _torch_mlp_acc(X, y, rskf)

    acc_lr_mean  = ml_stats_lr["accuracy"]["mean"]
    acc_rf_mean  = ml_stats_rf["accuracy"]["mean"]
    acc_mlp_mean = acc_mlp.mean()
    acc_mlp_std  = acc_mlp.std(ddof=1)

    best_acc = max(acc_lr_mean, acc_rf_mean, acc_mlp_mean)
    if best_acc == acc_lr_mean:
        best_name = "Logistic Regression"
        best_std  = ml_stats_lr["accuracy"]["sd"]
    elif best_acc == acc_rf_mean:
        best_name = "Random Forest"
        best_std  = ml_stats_rf["accuracy"]["sd"]
    else:
        best_name = f"MLP (PyTorch {DEVICE})"
        best_std  = acc_mlp_std

    print(f"  {name:<28}baseline={baseline*100:5.2f}%   "
          f"best={best_name} {best_acc*100:5.2f}% +/- {best_std*100:.2f}%   "
          f"margin={((best_acc - baseline) * 100):+.2f}pts")

    attack_c_stats.update_json_results(
        "../results/attack_c_full_stats_torch.json",
        f"system_comp_{name}_LR", ml_stats_lr)
    attack_c_stats.update_json_results(
        "../results/attack_c_full_stats_torch.json",
        f"system_comp_{name}_RF", ml_stats_rf)

    return {"system": name, "baseline": baseline,
            "best_model": best_name, "best_acc": best_acc,
            "best_std": best_std, "margin": best_acc - baseline}


if __name__ == "__main__":
    with open("../qrng_combined.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("#" * 70)
    print(f"SYSTEM COMPARISON  [PyTorch version]  Device: {DEVICE}")
    print("#" * 70)

    print("\nCollecting timing data...")
    print("  - ML-KEM-512, 300 trials...")
    mlkem_rows = time_ml_kem(num_trials=300)

    print("  - QEQSKE-clean, 300 trials...")
    clean_rows = time_qeqske(numbers, qeqske_keygen, num_trials=300)

    print("  - QEQSKE-leaky, 300 trials...")
    leaky_rows = time_qeqske(numbers, qeqske_keygen_leaky, num_trials=300)

    print("\n" + "=" * 70)
    print("Timing-based ML attack results")
    print("=" * 70)
    results = []
    results.append(run_attack(mlkem_rows, "Standard ML-KEM-512 (fixed params)"))
    results.append(run_attack(clean_rows, "QEQSKE-clean"))
    results.append(run_attack(leaky_rows, "QEQSKE-leaky (positive control)"))

    print("\nDone.")
