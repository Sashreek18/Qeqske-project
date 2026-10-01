"""
ATTACK C (Part 4): Combined Side-Channel ML Attack  [PyTorch version]
======================================================================
Drop-in replacement for sk-learn/attack_c_combined_ml.py.

Changes vs. sklearn version:
  - Logistic Regression, Random Forest: UNCHANGED (sklearn)
  - SVM (RBF): UNCHANGED (sklearn)
  - Neural Network (MLP): replaced by PyTorch MLP (GPU/MPS accelerated,
    can be scaled to larger architectures)

All data collection (collect_combined_dataset) and significance reporting
(report_feature_significance) are identical.
"""

import time
import tracemalloc
import os
import numpy as np
import psutil
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline

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

def _secret_properties(s):
    s_flat = [val for row in s for val in row]
    return sum(1 for v in s_flat if v != 0), sum(s_flat)


def collect_combined_dataset(numbers, n=4, k=2, num_trials=500, verbose=True):
    proc = psutil.Process(os.getpid())
    rows = []
    cursor = 0

    for trial in range(num_trials):
        batch = numbers[cursor:cursor + 60]
        if len(batch) < 30:
            if verbose:
                print(f"  Stopping early at trial {trial}: ran out of QRNG numbers")
            break

        qrng = QRNGSource(numbers=list(batch))
        rng  = QEQSKERandom(qrng)

        tracemalloc.start()
        proc.cpu_percent(interval=None)
        cpu_before = proc.cpu_times()
        t0 = time.perf_counter()

        keys = key_gen(rng, n=n, k=k)

        elapsed   = time.perf_counter() - t0
        cpu_after = proc.cpu_times()
        current, peak = tracemalloc.get_traced_memory()
        snapshot  = tracemalloc.take_snapshot()
        num_allocs = sum(stat.count for stat in snapshot.statistics("lineno"))
        tracemalloc.stop()

        cursor += qrng.total_consumed
        secret_weight, secret_sum = _secret_properties(keys["private_key"])

        rows.append({
            "trial": trial,
            "timing_us":      elapsed * 1e6,
            "cpu_user_us":    (cpu_after.user - cpu_before.user) * 1e6,
            "ram_peak_bytes": peak,
            "num_allocations": num_allocs,
            "secret_weight":  secret_weight,
            "secret_sum":     secret_sum,
        })

    if verbose:
        print(f"Collected {len(rows)} combined trials.")

    return rows


# ---------------------------------------------------------------------------
# Per-signal significance (identical to sklearn version)
# ---------------------------------------------------------------------------

def report_feature_significance(rows):
    features = ["timing_us", "cpu_user_us", "ram_peak_bytes", "num_allocations"]
    weights  = [r["secret_weight"] for r in rows]

    print(f"\n{'=' * 68}")
    print("Statistical Significance — each signal vs. secret_weight")
    print(f"{'=' * 68}")
    print(f"  {'Signal':<18}{'Pearson r':>12}{'p-value':>12}  Conclusion")
    for feat in features:
        vals = [r[feat] for r in rows]
        if len(set(vals)) > 1 and len(set(weights)) > 1:
            stats_dict = attack_c_stats.compute_correlation_stats(vals, weights, len(features))
            r = stats_dict["pearson_r"]
            p = stats_dict["pearson_p"]
            bonf_p = stats_dict["bonferroni_p"]
            attack_c_stats.update_json_results(
                "../results/attack_c_full_stats_torch.json",
                f"combined_ml_signals_{feat}", stats_dict)
        else:
            r, p, bonf_p = 0.0, 1.0, 1.0

        sig = "SIGNIFICANT (Bonferroni p<0.05)" if bonf_p < 0.05 else "not significant"
        print(f"  {feat:<18}{r:>12.4f}{p:>12.4f}  {sig}")


# ---------------------------------------------------------------------------
# PyTorch MLP for the "Neural Network" slot
# ---------------------------------------------------------------------------

class _MLPCV:
    """
    Wraps a PyTorch MLP in a cross_validate-compatible interface using
    manual CV iteration (sklearn's cross_validate doesn't accept torch models).
    """
    def __init__(self, hidden=(16, 8), lr=1e-3, epochs=100,
                 batch_size=64, seed=42):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def _build(self, n_feat):
        layers = []
        in_d = n_feat
        for h in self.hidden:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h
        layers += [nn.Linear(in_d, 1), nn.Sigmoid()]
        return nn.Sequential(*layers).to(DEVICE)

    def cv_score(self, X, y, rskf):
        accs, precs, recs, f1s = [], [], [], []
        for tr, te in rskf.split(X, y):
            torch.manual_seed(self.seed)
            mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-8
            Xtr = ((X[tr] - mu) / sd).astype(np.float32)
            Xte = ((X[te] - mu) / sd).astype(np.float32)
            ytr = y[tr].astype(np.float32)

            net = self._build(X.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            ds  = TensorDataset(torch.from_numpy(Xtr).to(DEVICE),
                                 torch.from_numpy(ytr).to(DEVICE))
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
            net.train()
            for _ in range(self.epochs):
                for Xb, yb in loader:
                    opt.zero_grad()
                    nn.BCELoss()(net(Xb).squeeze(-1), yb).backward()
                    opt.step()

            with torch.no_grad():
                net.eval()
                prob = net(torch.from_numpy(Xte).to(DEVICE)).squeeze(-1).cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            yte  = y[te]
            accs.append(accuracy_score(yte, pred))
            precs.append(precision_score(yte, pred, zero_division=0))
            recs.append(recall_score(yte, pred, zero_division=0))
            f1s.append(f1_score(yte, pred, zero_division=0))

        return {
            "test_accuracy":  np.array(accs),
            "test_precision": np.array(precs),
            "test_recall":    np.array(recs),
            "test_f1":        np.array(f1s),
        }


# ---------------------------------------------------------------------------
# ML attack runner
# ---------------------------------------------------------------------------

def run_ml_attack(rows, verbose=True):
    X = np.array([[r["timing_us"], r["cpu_user_us"],
                   r["ram_peak_bytes"], r["num_allocations"]]
                  for r in rows])
    weights = np.array([r["secret_weight"] for r in rows])
    y = (weights > np.median(weights)).astype(int)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    sklearn_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)":           SVC(kernel="rbf"),
    }

    baseline_acc = max(np.mean(y == 0), np.mean(y == 1))
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    print(f"\n{'=' * 68}")
    print(f"Combined ML Attack  [PyTorch version] — Device: {DEVICE}")
    print(f"{'=' * 68}")
    print(f"  Dataset: {len(rows)} trials | Repeated CV (5 splits, 10 repeats)")
    print(f"  Random baseline = {baseline_acc*100:.2f}%")
    print(f"\n  {'Model':<24}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")

    results = {}

    # sklearn models
    for name, clf in sklearn_models.items():
        model = make_pipeline(StandardScaler(), clf)
        cv_r  = cross_validate(model, X, y, cv=rskf,
                               scoring=("accuracy", "precision", "recall", "f1"))
        acc  = cv_r["test_accuracy"].mean()
        prec = cv_r["test_precision"].mean()
        rec  = cv_r["test_recall"].mean()
        f1   = cv_r["test_f1"].mean()
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"  {name:<24}{acc*100:>9.2f}%{prec*100:>10.2f}%{rec*100:>8.2f}%{f1*100:>7.2f}%")

        ml_stats = attack_c_stats.compute_ml_stats(model, X, y, rskf)
        attack_c_stats.update_json_results(
            "../results/attack_c_full_stats_torch.json",
            f"combined_ml_models_{name}", ml_stats)

    # PyTorch MLP
    torch_mlp = _MLPCV(hidden=(16, 8), epochs=100)
    cv_torch  = torch_mlp.cv_score(X, y, rskf)
    acc_t  = cv_torch["test_accuracy"].mean()
    prec_t = cv_torch["test_precision"].mean()
    rec_t  = cv_torch["test_recall"].mean()
    f1_t   = cv_torch["test_f1"].mean()
    name_t = f"MLP (PyTorch {DEVICE})"
    results[name_t] = {"accuracy": acc_t, "precision": prec_t,
                       "recall": rec_t,   "f1": f1_t}
    print(f"  {name_t:<24}{acc_t*100:>9.2f}%{prec_t*100:>10.2f}%"
          f"{rec_t*100:>8.2f}%{f1_t*100:>7.2f}%")

    if verbose:
        rf = sklearn_models["Random Forest"]
        rf.fit(X_s, y)
        importances = rf.feature_importances_
        feat_names  = ["timing_us", "cpu_user_us", "ram_peak_bytes", "num_allocations"]
        print(f"\n  Random Forest feature importance:")
        for fname, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
            bar = "#" * int(imp * 40)
            print(f"    {fname:<18} {imp:.4f}  {bar}")

        best_acc = max(r["accuracy"] for r in results.values())
        margin   = best_acc - baseline_acc
        print(f"\n  Best model beat baseline by {margin*100:+.2f} percentage points.")

    return results, baseline_acc


if __name__ == "__main__":
    with open("../qrng_combined.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("\n" + "#" * 70)
    print("ATTACK C-4: Combined Side-Channel ML Attack  [PyTorch version]")
    print("#" * 70)
    rows = collect_combined_dataset(numbers, n=4, k=2, num_trials=500)

    report_feature_significance(rows)
    run_ml_attack(rows)
