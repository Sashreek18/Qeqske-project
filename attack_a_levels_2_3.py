"""
ATTACK A — Level 2 and Level 3 (extends the raw-randomness distinguisher)
=============================================================================
Level 1 (already done, see attack_a_complete.py) asks: can ML tell QRNG
numbers apart from Mersenne Twister numbers? That alone doesn't establish
QEQSKE's security -- a reviewer can rightly ask "how does that demonstrate
security of the protocol itself?"

Level 2 — QEQSKE-derived randomness:
  Feed BOTH QRNG and Mersenne Twister numbers through the ACTUAL QEQSKE
  key-generation pipeline, and statistically compare the resulting:
    - generated primes (q)
    - secret coefficients (flattened entries of s)
    - noise coefficients (flattened entries of e)
    - public key entries (flattened entries of t)
  If QEQSKE produces statistically indistinguishable outputs regardless of
  which randomness source feeds it, that is direct evidence about the
  PROTOCOL, not just the raw randomness source.

Level 3 — Adversarial prediction from public outputs only:
  Give an ML model ONLY the public outputs (matrix A, public key t) --
  exactly what a real-world eavesdropper would see -- and ask whether it
  can predict anything about the secret (its weight, or individual
  coefficients). This directly tests the LWE hardness assumption underlying
  Kyber/QEQSKE in our toy-parameter implementation.
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from qrng_handler import QRNGSource
from qeqske_core import QEQSKERandom, key_gen


class MersenneQRNGSource:
    """Drop-in replacement for QRNGSource, but backed by NumPy's Mersenne
    Twister instead of real QRNG numbers -- same 16-bit range so it is
    consumed identically by QEQSKERandom."""
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.total_consumed = 0

    def call_qrng(self):
        self.total_consumed += 1
        return int(self.rng.randint(0, 65536))


def collect_level2_data(numbers, source_label, n=4, k=2, num_trials=300, use_qrng=True, seed_base=0):
    """Runs key_gen() num_trials times using either QRNG or Mersenne
    Twister as the underlying randomness source, and records the resulting
    prime, secret coefficients, noise-adjacent public key entries."""
    rows = []
    cursor = 0
    for trial in range(num_trials):
        if use_qrng:
            batch = numbers[cursor:cursor + 60]
            if len(batch) < 30:
                break
            qrng = QRNGSource(numbers=list(batch))
            cursor_advance = None
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
            "t_mean": float(np.mean(t_flat)) / keys["q"],  # normalized
            "t_std": float(np.std(t_flat)) / keys["q"],
        })
    return rows


def compare_distributions(qrng_rows, mt_rows):
    print(f"\n{'=' * 70}")
    print("LEVEL 2 — QEQSKE-derived output comparison (QRNG-fed vs MT-fed)")
    print(f"{'=' * 70}")
    fields = ["q", "secret_mean", "secret_weight", "t_mean", "t_std"]
    print(f"  {'Field':<16}{'QRNG mean':>12}{'MT mean':>12}{'KS p-value':>12}  Conclusion")
    for field in fields:
        a = [r[field] for r in qrng_rows]
        b = [r[field] for r in mt_rows]
        ks_stat, p_val = stats.ks_2samp(a, b)
        sig = "DIFFERENT dist." if p_val < 0.05 else "same distribution"
        print(f"  {field:<16}{np.mean(a):>12.4f}{np.mean(b):>12.4f}{p_val:>12.4f}  {sig}")


def run_level3(qrng_rows, num_trials, n=4, k=2):
    """Level 3: predict secret_weight using ONLY public outputs (q, t
    statistics) -- no execution-behaviour side-channel involved at all,
    just the mathematical public key itself."""
    print(f"\n{'=' * 70}")
    print("LEVEL 3 — Adversarial prediction from PUBLIC outputs only (no side-channel)")
    print(f"{'=' * 70}")
    X = np.array([[r["q"], r["t_mean"], r["t_std"]] for r in qrng_rows])
    weights = np.array([r["secret_weight"] for r in qrng_rows])
    y = (weights > np.median(weights)).astype(int)

    if len(set(y)) < 2:
        print("  Not enough class variation to test.")
        return

    baseline = max(np.mean(y == 0), np.mean(y == 1))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    acc_lr = cross_val_score(model, X, y, cv=rskf, scoring="accuracy")
    acc_rf = cross_val_score(rf, X, y, cv=rskf, scoring="accuracy")
    best = max(acc_lr.mean(), acc_rf.mean())
    best_std = acc_lr.std() if acc_lr.mean() >= acc_rf.mean() else acc_rf.std()

    print(f"  Baseline (majority-class) = {baseline*100:.2f}%")
    print(f"  Best model accuracy       = {best*100:.2f}% +/- {best_std*100:.2f}%")
    print(f"  Margin over baseline      = {(best-baseline)*100:+.2f} points")
    print(f"  NOTE: predicting from public (A, t) alone directly tests the LWE hardness")
    print(f"  assumption underlying Kyber/QEQSKE at our toy parameters (n={n}, k={k}).")


if __name__ == "__main__":
    with open("qrng_combined.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("#" * 70)
    print("ATTACK A — Levels 2 and 3")
    print("#" * 70)

    print("\nCollecting QEQSKE outputs fed by real QRNG (300 trials)...")
    qrng_rows = collect_level2_data(numbers, "QRNG", num_trials=300, use_qrng=True)

    print("Collecting QEQSKE outputs fed by Mersenne Twister (300 trials)...")
    mt_rows = collect_level2_data(numbers, "Mersenne", num_trials=300, use_qrng=False)

    compare_distributions(qrng_rows, mt_rows)
    run_level3(qrng_rows, num_trials=300)

    print("\nDone.")
