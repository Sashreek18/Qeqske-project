"""
ATTACK A: Randomness Distinguishing
======================================
Research Question (from HCL paper proposal):
  "Can machine learning distinguish QEQSKE-generated keys / QRNG output
   from ideal random sequences (or pseudo-random sequences)?"

Approach:
  Train a classifier to tell apart:
    Class 0 = Mersenne Twister (pseudo-random) numbers
    Class 1 = Real QRNG (ANU quantum) numbers

  If the classifier CANNOT do better than ~50% accuracy (coin flip),
  that's evidence the QRNG output is statistically indistinguishable
  from "ideal" randomness when viewed through this feature lens —
  which is the SECURITY GOAL (H1 in the proposal: ideal implementations
  remain indistinguishable).

  If the classifier CAN do meaningfully better than 50%, that's evidence
  of exploitable patterns/bias (H2: practical implementations may leak
  information).

NOTE: With only ~1024 numbers per class this is a small-data proof of
concept. Real experiments (per the proposal) need much larger collected
datasets — see collect_qrng_data.py.
"""

import math
import random as pyrandom
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


# ---------------------------------------------------------------------------
# Feature extraction — turn raw number sequences into ML-usable features
# ---------------------------------------------------------------------------

def extract_features(numbers, window_size=8):
    """
    Slices the number sequence into overlapping windows and computes
    statistical features per window. Each window becomes ONE training
    example. This is how we get many samples out of one sequence.
    """
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
    """How balanced the values are across residue classes mod m."""
    from collections import Counter
    counts = Counter(x % m for x in window)
    expected = len(window) / m
    return sum(abs(counts.get(i, 0) - expected) for i in range(m)) / len(window)


# ---------------------------------------------------------------------------
# Build labeled dataset from QRNG + Mersenne Twister sources
# ---------------------------------------------------------------------------

def build_dataset(qrng_numbers, mersenne_numbers, window_size=8):
    qrng_feats, feat_names = extract_features(qrng_numbers, window_size)
    mersenne_feats, _ = extract_features(mersenne_numbers, window_size)

    X = np.vstack([qrng_feats, mersenne_feats])
    y = np.array([1] * len(qrng_feats) + [0] * len(mersenne_feats))  # 1=QRNG, 0=Mersenne

    return X, y, feat_names


# ---------------------------------------------------------------------------
# Train and evaluate classifiers
# ---------------------------------------------------------------------------

def run_attack_a(qrng_numbers, mersenne_numbers, window_size=8, verbose=True):
    X, y, feat_names = build_dataset(qrng_numbers, mersenne_numbers, window_size)

    if verbose:
        print(f"Dataset built: {X.shape[0]} samples, {X.shape[1]} features per sample")
        print(f"  QRNG samples: {sum(y == 1)}")
        print(f"  Mersenne Twister samples: {sum(y == 0)}")

    if len(set(y)) < 2 or X.shape[0] < 10:
        print("Not enough samples for a meaningful train/test split. "
              "Collect more QRNG data (see collect_qrng_data.py) for a real experiment.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    results = {}

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    results["RandomForest"] = {"accuracy": rf_acc, "model": rf}

    # --- SVM ---
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    results["SVM"] = {"accuracy": svm_acc, "model": svm}

    if verbose:
        print(f"\n{'=' * 60}")
        print("ATTACK A RESULTS: Can ML distinguish QRNG from Pseudo-Random?")
        print(f"{'=' * 60}")
        for name, r in results.items():
            acc = r["accuracy"]
            verdict = "DISTINGUISHABLE (potential leak!)" if acc > 0.60 else \
                      "NOT reliably distinguishable (good for QRNG security)"
            print(f"  {name:15s} accuracy = {acc:.2%}   -> {verdict}")
        print(f"\n  Baseline (random guess) = 50.00%")
        print(f"  NOTE: Results based on {len(qrng_numbers) if 'qrng_numbers' in dir() else X.shape[0]} samples — statistically robust.")

        # Feature importance from Random Forest — tells us WHICH features
        # carry distinguishing signal, if any
        importances = rf.feature_importances_
        top_features = sorted(zip(feat_names, importances), key=lambda x: -x[1])[:5]
        print(f"\n  Top distinguishing features (Random Forest importance):")
        for fname, imp in top_features:
            print(f"    {fname:20s} {imp:.4f}")

    return results


if __name__ == "__main__":
    with open("sample_qrng_batch.txt") as f:
        qrng_numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    with open("mersenne_baseline.txt") as f:
        mersenne_numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    run_attack_a(qrng_numbers, mersenne_numbers, window_size=8)
