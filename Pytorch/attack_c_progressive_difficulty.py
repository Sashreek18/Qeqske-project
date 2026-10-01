"""
ATTACK C — Progressive Difficulty Ladder  [PyTorch version]
============================================================
Drop-in replacement for sk-learn/attack_c_progressive_difficulty.py.

Changes vs. sklearn version:
  - Logistic Regression: UNCHANGED (sklearn)
  - Random Forest: UNCHANGED (sklearn)
  - NEW: PyTorch MLP added as a third classifier per difficulty level.
    Results from all three are compared, and the best is reported.

All data collection and target-labelling logic are identical.
"""

import time
import tracemalloc
import os
import numpy as np
import psutil
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
from qeqske_core import QEQSKERandom, key_gen
from nn_torch import get_device

DEVICE = get_device()


# ---------------------------------------------------------------------------
# Data collection (identical to sklearn version)
# ---------------------------------------------------------------------------

def collect_combined_dataset(numbers, n=4, k=2, num_trials=1000, verbose=True):
    proc = psutil.Process(os.getpid())
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

        tracemalloc.start()
        proc.cpu_percent(interval=None)
        cpu_before = proc.cpu_times()
        t0 = time.perf_counter()

        keys = key_gen(rng, n=n, k=k)

        elapsed    = time.perf_counter() - t0
        cpu_after  = proc.cpu_times()
        current, peak = tracemalloc.get_traced_memory()
        snapshot   = tracemalloc.take_snapshot()
        num_allocs = sum(stat.count for stat in snapshot.statistics("lineno"))
        tracemalloc.stop()

        cursor += qrng.total_consumed
        s = keys["private_key"]
        s_flat = [v for row in s for v in row]
        weight = sum(1 for v in s_flat if v != 0)
        s_sum  = sum(s_flat)

        rows.append({
            "timing_us":       elapsed * 1e6,
            "cpu_user_us":     (cpu_after.user - cpu_before.user) * 1e6,
            "ram_peak_bytes":  peak,
            "num_allocations": num_allocs,
            "secret_weight":   weight,
            "secret_sum":      s_sum,
            "s00":             s[0][0],
        })

    if verbose:
        print(f"Collected {len(rows)} combined trials.")
    return rows


def _features(rows):
    return np.array([[r["timing_us"], r["cpu_user_us"],
                      r["ram_peak_bytes"], r["num_allocations"]]
                     for r in rows])


# ---------------------------------------------------------------------------
# PyTorch MLP helper for CV
# ---------------------------------------------------------------------------

def _torch_mlp_mean_acc(X, y, rskf, hidden=(16, 8), lr=1e-3,
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
        accs.append(((prob >= 0.5).astype(int) == y[te]).mean())

    return np.array(accs)


# ---------------------------------------------------------------------------
# Per-target evaluation
# ---------------------------------------------------------------------------

def evaluate_target(X, y, target_name, difficulty, n_splits=5, n_repeats=10):
    if len(set(y)) < 2:
        print(f"  {target_name:<32}{difficulty:<12}  SKIPPED (no class variation)")
        return None

    n_classes = len(set(y))
    baseline  = max(np.mean(y == c) for c in set(y))

    lr_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    rf       = RandomForestClassifier(n_estimators=200, random_state=42)
    rskf     = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=42)

    scoring = ("accuracy", "balanced_accuracy", "roc_auc", "f1") if n_classes == 2 \
              else ("accuracy", "balanced_accuracy", "roc_auc_ovr", "f1_macro")

    ml_stats_lr = attack_c_stats.compute_ml_stats(lr_model, X, y, rskf, scoring=scoring)
    ml_stats_rf = attack_c_stats.compute_ml_stats(rf,       X, y, rskf, scoring=scoring)

    # PyTorch MLP
    acc_mlp = _torch_mlp_mean_acc(X, y, rskf)

    acc_lr_mean  = ml_stats_lr["accuracy"]["mean"]
    acc_rf_mean  = ml_stats_rf["accuracy"]["mean"]
    acc_mlp_mean = acc_mlp.mean()
    acc_mlp_std  = acc_mlp.std(ddof=1)

    best_acc = max(acc_lr_mean, acc_rf_mean, acc_mlp_mean)
    if best_acc == acc_lr_mean:
        best_std  = ml_stats_lr["accuracy"]["sd"]
        best_stats = ml_stats_lr
        best_name  = "LR"
    elif best_acc == acc_rf_mean:
        best_std  = ml_stats_rf["accuracy"]["sd"]
        best_stats = ml_stats_rf
        best_name  = "RF"
    else:
        best_std  = acc_mlp_std
        best_stats = None
        best_name  = f"MLP({DEVICE})"

    margin = best_acc - baseline
    ci_low, ci_high = best_stats["accuracy"]["ci95"] if best_stats else (best_acc - best_std, best_acc + best_std)
    ci_margin = (ci_high - ci_low) / 2.0

    print(f"  {target_name:<32}{difficulty:<12}{n_classes:>3} classes  "
          f"baseline={baseline*100:5.1f}%  best_acc={best_acc*100:5.1f}%"
          f"+/-{best_std*100:4.1f}%  ({best_name})  "
          f"(95% CI +/-{ci_margin*100:.1f}pts)  margin={margin*100:+5.1f}pts")

    attack_c_stats.update_json_results(
        "../results/attack_c_full_stats_torch.json",
        f"progressive_ml_{target_name}_LR", ml_stats_lr)
    attack_c_stats.update_json_results(
        "../results/attack_c_full_stats_torch.json",
        f"progressive_ml_{target_name}_RF", ml_stats_rf)

    return {"target": target_name, "difficulty": difficulty, "n_classes": n_classes,
            "baseline": baseline, "best_acc": best_acc, "best_std": best_std,
            "ci95": ci_margin, "margin": margin, "best_model": best_name}


if __name__ == "__main__":
    with open("../qrng_combined.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("#" * 70)
    print("ATTACK C — Progressive Difficulty Ladder  [PyTorch version]")
    print(f"Device: {DEVICE}")
    print("#" * 70)
    rows = collect_combined_dataset(numbers, n=4, k=2, num_trials=1000)
    X = _features(rows)

    print(f"\n{'Target':<32}{'Difficulty':<12}{'Classes':>10}  "
          f"{'Baseline':>9}  {'Best Acc':>16}  {'Margin':>10}")
    print("-" * 100)

    results = []

    weights = np.array([r["secret_weight"] for r in rows])
    y_weight = (weights > np.median(weights)).astype(int)
    results.append(evaluate_target(X, y_weight, "Secret weight (binary)", "Easy"))

    sums = np.array([r["secret_sum"] for r in rows])
    y_sum = (sums > np.median(sums)).astype(int)
    results.append(evaluate_target(X, y_sum, "Secret sum (binary)", "Easy"))

    quartiles = np.percentile(weights, [25, 50, 75])
    y_quartile = np.digitize(weights, quartiles)
    results.append(evaluate_target(X, y_quartile, "Secret weight (multi-class)", "Medium"))

    s00_vals = np.array([r["s00"] for r in rows])
    y_sign = (s00_vals > 0).astype(int)
    results.append(evaluate_target(X, y_sign, "s[0][0] sign (positive vs not)", "Hard"))

    y_exact = (s00_vals + 1).astype(int)
    results.append(evaluate_target(X, y_exact, "s[0][0] exact value", "Very Hard"))

    print("\nDone.")
