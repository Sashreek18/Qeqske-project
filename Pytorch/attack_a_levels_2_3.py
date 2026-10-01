"""
ATTACK A — Level 2 and Level 3  [PyTorch version]
==================================================
Exact copy of sk-learn/attack_a_levels_2_3.py with:
  - Level 2 statistical comparison: unchanged (scipy KS test)
  - Level 3 ML classifiers: Logistic Regression (sklearn) + PyTorch MLP
    replacing RandomForestClassifier for GPU-accelerated inference

All data collection code (collect_level2_data) is identical.
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qrng_handler import QRNGSource
from qeqske_core import QEQSKERandom, key_gen
from nn_torch import get_device

DEVICE = get_device()


class MersenneQRNGSource:
    """Drop-in replacement for QRNGSource backed by NumPy Mersenne Twister."""
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.total_consumed = 0

    def call_qrng(self):
        self.total_consumed += 1
        return int(self.rng.randint(0, 65536))


def collect_level2_data(numbers, source_label, n=4, k=2,
                        num_trials=300, use_qrng=True, seed_base=0):
    """Unchanged from sklearn version."""
    rows = []
    cursor = 0
    for trial in range(num_trials):
        if use_qrng:
            batch = numbers[cursor:cursor + 60]
            if len(batch) < 30:
                break
            qrng = QRNGSource(numbers=list(batch))
        else:
            qrng = MersenneQRNGSource(seed=seed_base + trial)

        rng = QEQSKERandom(qrng)
        keys = key_gen(rng, n=n, k=k)

        if use_qrng:
            cursor += qrng.total_consumed

        s_flat = [v for row in keys["private_key"] for v in row]
        t_flat = [v for row in keys["public_key_t"] for v in row]

        rows.append({
            "source": source_label,
            "q": keys["q"],
            "secret_mean": float(np.mean(s_flat)),
            "secret_weight": sum(1 for v in s_flat if v != 0),
            "t_mean": float(np.mean(t_flat)) / keys["q"],
            "t_std": float(np.std(t_flat)) / keys["q"],
        })
    return rows


def compare_distributions(qrng_rows, mt_rows):
    """Unchanged from sklearn version."""
    print(f"\n{'=' * 70}")
    print("LEVEL 2 — QEQSKE-derived output comparison (QRNG-fed vs MT-fed)")
    print(f"{'=' * 70}")
    fields = ["q", "secret_mean", "secret_weight", "t_mean", "t_std"]
    print(f"  {'Field':<16}{'QRNG mean':>12}{'MT mean':>12}{'KS p-value':>12}  Conclusion")
    for field in fields:
        a = [r[field] for r in qrng_rows]
        b = [r[field] for r in mt_rows]
        ks_stat, p_val = stats.ks_2samp(a, b)
        sig = "DIFFERENT dist." if p_val < 0.05 / 5 else "same distribution"
        print(f"  {field:<16}{np.mean(a):>12.4f}{np.mean(b):>12.4f}{p_val:>12.4f}  {sig}")
        print("  Bonferroni-corrected alpha = 0.05/5 = 0.010 (5 simultaneous tests)")


# ---------------------------------------------------------------------------
# PyTorch MLP for Level 3
# ---------------------------------------------------------------------------

class _SmallMLP(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, 1),          nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _cv_torch_mlp(X, y, rskf, hidden=32, lr=5e-3, epochs=50,
                  batch_size=64, seed=42):
    """Cross-validated accuracy for the PyTorch MLP on tabular data."""
    accs = []
    for tr, te in rskf.split(X, y):
        torch.manual_seed(seed)
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0) + 1e-8
        Xtr = ((X[tr] - mu) / sd).astype(np.float32)
        Xte = ((X[te] - mu) / sd).astype(np.float32)
        ytr = y[tr].astype(np.float32)
        yte = y[te]

        net = _SmallMLP(X.shape[1], hidden).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        ds  = TensorDataset(torch.from_numpy(Xtr).to(DEVICE),
                             torch.from_numpy(ytr).to(DEVICE))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        net.train()
        for _ in range(epochs):
            for Xb, yb in loader:
                opt.zero_grad()
                nn.BCELoss()(net(Xb), yb).backward()
                opt.step()

        with torch.no_grad():
            net.eval()
            prob = net(torch.from_numpy(Xte).to(DEVICE)).cpu().numpy()
        accs.append(((prob >= 0.5).astype(int) == yte).mean())

    return np.array(accs)


def run_level3(qrng_rows, num_trials, n=4, k=2):
    """Level 3: predict secret_weight from public outputs using PyTorch MLP."""
    print(f"\n{'=' * 70}")
    print("LEVEL 3 — Adversarial prediction from PUBLIC outputs only (no side-channel)")
    print(f"  Device: {DEVICE}")
    print(f"{'=' * 70}")

    X = np.array([[r["q"], r["t_mean"], r["t_std"]] for r in qrng_rows],
                 dtype=np.float32)
    weights = np.array([r["secret_weight"] for r in qrng_rows])
    y = (weights > np.median(weights)).astype(int)

    if len(set(y)) < 2:
        print("  Not enough class variation to test.")
        return

    baseline = max(np.mean(y == 0), np.mean(y == 1))
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    # Logistic Regression (sklearn, fast baseline)
    lr_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    acc_lr = cross_val_score(lr_model, X, y, cv=rskf, scoring="accuracy")

    # PyTorch MLP (GPU/MPS accelerated)
    acc_mlp = _cv_torch_mlp(X, y, rskf)

    best = max(acc_lr.mean(), acc_mlp.mean())
    best_std = acc_lr.std() if acc_lr.mean() >= acc_mlp.mean() else acc_mlp.std()
    best_name = "Logistic Regression" if acc_lr.mean() >= acc_mlp.mean() else "MLP (PyTorch)"

    print(f"  Baseline (majority-class)  = {baseline*100:.2f}%")
    print(f"  Logistic Regression        = {acc_lr.mean()*100:.2f}% +/- {acc_lr.std()*100:.2f}%")
    print(f"  MLP (PyTorch, {DEVICE})  = {acc_mlp.mean()*100:.2f}% +/- {acc_mlp.std()*100:.2f}%")
    print(f"  Best model ({best_name})  accuracy = {best*100:.2f}% +/- {best_std*100:.2f}%")
    print(f"  Margin over baseline       = {(best-baseline)*100:+.2f} points")
    print(f"  NOTE: at n={n}, k={k} the secret is recoverable from public (A, t) by brute")
    print(f"  force in under a millisecond (see break_poc.py). A null result here therefore")
    print(f"  reflects these ML models, not LWE hardness.")


if __name__ == "__main__":
    with open("../qrng_combined.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("#" * 70)
    print("ATTACK A — Levels 2 and 3  [PyTorch version]")
    print("#" * 70)

    print("\nCollecting QEQSKE outputs fed by real QRNG (300 trials)...")
    qrng_rows = collect_level2_data(numbers, "QRNG", num_trials=300, use_qrng=True)

    print("Collecting QEQSKE outputs fed by Mersenne Twister (300 trials)...")
    mt_rows = collect_level2_data(numbers, "Mersenne", num_trials=300, use_qrng=False)

    compare_distributions(qrng_rows, mt_rows)
    run_level3(qrng_rows, num_trials=300)

    print("\nDone.")
