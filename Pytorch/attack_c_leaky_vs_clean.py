"""
ATTACK C — Positive Control: QEQSKE-clean vs. QEQSKE-leaky  [PyTorch version]
===============================================================================
Drop-in replacement for sk-learn/attack_c_leaky_vs_clean.py.

Changes vs sklearn version:
  - Logistic Regression, Random Forest, SVM: UNCHANGED (sklearn)
  - Neural Network (MLP): replaced by PyTorch MLP (GPU/MPS accelerated)

All data collection (collect_dataset) and the intentionally-leaky key_gen
variant (key_gen_leaky) are identical.
"""

import time
import numpy as np
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import attack_c_stats
from qrng_handler import QRNGSource
from qeqske_core import (QEQSKERandom, mat_mult_mod, mat_add_mod,
                          right_rotate, key_gen as key_gen_clean)
from nn_torch import get_device

DEVICE = get_device()


def _secret_properties(s):
    s_flat = [val for row in s for val in row]
    return sum(1 for v in s_flat if v != 0), sum(abs(v) for v in s_flat)


def key_gen_leaky(rng, n, k, leak_threshold=5, leak_repeats=1500):
    """Identical to sklearn version — intentional secret-dependent branch."""
    q = rng.get_q()
    first_col = [rng.get_random(q) for _ in range(n)]
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        A[i][0] = first_col[i]
    x = list(first_col)
    for col in range(1, n):
        x = right_rotate(x, 1)
        x[0] = (-x[0]) % q
        for row in range(n):
            A[row][col] = x[row]

    s = [[rng.get_small_random() for _ in range(k)] for _ in range(n)]
    e = [[rng.get_small_random() for _ in range(k)] for _ in range(n)]
    As = mat_mult_mod(A, s, q)
    t  = mat_add_mod(As, e, q)

    weight, _ = _secret_properties(s)
    if weight > leak_threshold:
        dummy = A
        for _ in range(leak_repeats):
            dummy = mat_mult_mod(dummy, A, q)

    return {"private_key": s, "public_key_A": A, "public_key_t": t,
            "q": q, "n": n, "k": k}


def collect_dataset(numbers, keygen_fn, n=4, k=2, num_trials=300, verbose=True):
    """Unchanged from sklearn version."""
    rows = []
    cursor = 0
    for trial in range(num_trials):
        batch = numbers[cursor:cursor + 60]
        if len(batch) < 30:
            if verbose:
                print(f"  Stopped early at trial {trial}: out of QRNG numbers")
            break
        qrng = QRNGSource(numbers=list(batch))
        rng  = QEQSKERandom(qrng)

        t0 = time.perf_counter()
        keys = keygen_fn(rng, n, k)
        elapsed = time.perf_counter() - t0

        cursor += qrng.total_consumed
        weight, s_sum = _secret_properties(keys["private_key"])
        rows.append({"timing_us": elapsed * 1e6,
                     "secret_weight": weight, "secret_sum": s_sum})
    return rows


# ---------------------------------------------------------------------------
# PyTorch MLP for the MLP slot
# ---------------------------------------------------------------------------

def _torch_mlp_cv_scores(X, y, rskf, hidden=(16, 8), lr=1e-3,
                          epochs=100, batch_size=64, seed=42):
    """Returns dict of per-fold accuracy and roc_auc arrays."""
    accs, aucs = [], []
    for tr, te in rskf.split(X, y):
        torch.manual_seed(seed)
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
        Xtr = ((X[tr] - mu) / sd).astype(np.float32)
        Xte = ((X[te] - mu) / sd).astype(np.float32)
        ytr = y[tr].astype(np.float32)

        layers = []
        in_d = X.shape[1]
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h
        layers += [nn.Linear(in_d, 1), nn.Sigmoid()]
        net = nn.Sequential(*layers).to(DEVICE)

        opt    = torch.optim.Adam(net.parameters(), lr=lr)
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

        pred = (prob >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], pred))
        try:
            aucs.append(roc_auc_score(y[te], prob))
        except ValueError:
            aucs.append(0.5)

    return {"accuracy": np.array(accs), "roc_auc": np.array(aucs)}


# ---------------------------------------------------------------------------
# ML runner
# ---------------------------------------------------------------------------

def run_ml_with_cv(rows, label_name="secret_weight",
                   n_splits=5, n_repeats=10, scenario_name="clean"):
    """Predicts above/below-median label using repeated stratified k-fold CV."""
    X    = np.array([[r["timing_us"]] for r in rows])
    vals = np.array([r[label_name] for r in rows])
    y    = (vals > np.median(vals)).astype(int)

    if len(set(y)) < 2:
        print("  Not enough class variation to run classification.")
        return None

    baseline = max(np.mean(y == 0), np.mean(y == 1))
    rskf     = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=42)

    print(f"\n  Device: {DEVICE}")
    print(f"  Baseline (majority-class) accuracy = {baseline*100:.2f}%")
    print(f"  {'Model':<24}{'Accuracy (mean+-std)':>26}{'ROC-AUC (mean+-std)':>24}")

    results = {}

    # sklearn models
    sklearn_models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)":           make_pipeline(StandardScaler(),
                                             SVC(kernel="rbf", probability=False)),
    }
    for name, model in sklearn_models.items():
        ml_stats = attack_c_stats.compute_ml_stats(model, X, y, rskf)
        acc_mean = ml_stats["accuracy"]["mean"]
        acc_std  = ml_stats["accuracy"]["sd"]
        auc_mean = ml_stats["roc_auc"]["mean"]
        auc_std  = ml_stats["roc_auc"]["sd"]
        results[name] = {"acc_mean": acc_mean, "acc_std": acc_std,
                          "auc_mean": auc_mean, "auc_std": auc_std}
        print(f"  {name:<24}{acc_mean*100:>9.2f}% +/- {acc_std*100:>5.2f}%"
              f"{'':>2}{auc_mean:>10.3f} +/- {auc_std:.3f}")
        attack_c_stats.update_json_results(
            "../results/attack_c_full_stats_torch.json",
            f"{scenario_name}_ml_{name}", ml_stats)

    # PyTorch MLP
    cv_torch = _torch_mlp_cv_scores(X, y, rskf)
    acc_t = cv_torch["accuracy"].mean()
    std_t = cv_torch["accuracy"].std()
    auc_t = cv_torch["roc_auc"].mean()
    asd_t = cv_torch["roc_auc"].std()
    name_t = f"MLP (PyTorch {DEVICE})"
    results[name_t] = {"acc_mean": acc_t, "acc_std": std_t,
                        "auc_mean": auc_t, "auc_std": asd_t}
    print(f"  {name_t:<24}{acc_t*100:>9.2f}% +/- {std_t*100:>5.2f}%"
          f"{'':>2}{auc_t:>10.3f} +/- {asd_t:.3f}")

    return results, baseline


if __name__ == "__main__":
    with open("../qrng_combined.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("#" * 70)
    print("POSITIVE CONTROL: QEQSKE-clean vs. QEQSKE-leaky  [PyTorch version]")
    print("#" * 70)

    print("\n--- Collecting QEQSKE-CLEAN dataset (300 trials) ---")
    clean_rows = collect_dataset(numbers, key_gen_clean, num_trials=300)
    print(f"Collected {len(clean_rows)} clean trials.")

    print("\n--- Collecting QEQSKE-LEAKY dataset (300 trials) ---")
    leaky_rows = collect_dataset(numbers, key_gen_leaky, num_trials=300)
    print(f"Collected {len(leaky_rows)} leaky trials.")

    print("\n" + "=" * 70)
    print("ML ATTACK on QEQSKE-CLEAN (expect: near baseline)")
    print("=" * 70)
    run_ml_with_cv(clean_rows, scenario_name="clean")

    print("\n" + "=" * 70)
    print("ML ATTACK on QEQSKE-LEAKY (expect: clearly above baseline)")
    print("=" * 70)
    run_ml_with_cv(leaky_rows, scenario_name="leaky")
