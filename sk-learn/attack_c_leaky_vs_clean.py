"""
ATTACK C — Positive Control: QEQSKE-clean vs. QEQSKE-leaky
=============================================================
Addresses the review comment: "check that the experiments actually detect
leakage from a leaky source." Right now, a "no leakage found" result could
mean either (a) QEQSKE genuinely has no exploitable side-channel, or
(b) our measurement + ML pipeline simply isn't sensitive enough to find one.

This experiment rules out (b): we build a QEQSKE-leaky variant with an
INTENTIONAL, DOCUMENTED secret-dependent branch:

    if secret_weight(s) > threshold:
        perform additional (redundant) computation

then run the exact same measurement + ML pipeline against both the clean
and leaky implementations. If the pipeline:
  - fails to beat baseline on QEQSKE-clean, AND
  - succeeds (high accuracy) on QEQSKE-leaky,
that demonstrates the apparatus is capable of detecting leakage when it
is actually present -- making the "no leakage" conclusion on the real
(clean) implementation far more credible.
"""

import time
import numpy as np
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

import attack_c_stats

from qrng_handler import QRNGSource
from qeqske_core import (
    QEQSKERandom, mat_mult_mod, mat_add_mod, right_rotate, key_gen as key_gen_clean
)


def _secret_properties(s):
    s_flat = [val for row in s for val in row]
    secret_weight = sum(1 for v in s_flat if v != 0)
    secret_sum = sum(abs(v) for v in s_flat)
    return secret_weight, secret_sum


def key_gen_leaky(rng, n, k, leak_threshold=5, leak_repeats=1500):
    """
    Identical to qeqske_core.key_gen(), EXCEPT: after generating the secret
    key s, if its weight exceeds `leak_threshold` (chosen near the median
    weight of ~5/8 for n=4,k=2, so roughly half of trials trigger the leak
    and half don't -- giving the classifier a real signal to find), it
    performs `leak_repeats` extra, genuinely-executed redundant matrix
    multiplications before returning. This is a deliberately, documentedly
    leaky research variant -- NOT a proposed fix or claim about the real
    QEQSKE system.
    """
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
    t = mat_add_mod(As, e, q)

    # --- INTENTIONAL LEAK: secret-dependent extra computation (always runs
    # when triggered -- no dead branches) ---
    weight, _ = _secret_properties(s)
    if weight > leak_threshold:
        dummy = A
        for _ in range(leak_repeats):
            dummy = mat_mult_mod(dummy, A, q)

    return {"private_key": s, "public_key_A": A, "public_key_t": t, "q": q, "n": n, "k": k}


def collect_dataset(numbers, keygen_fn, n=4, k=2, num_trials=300, verbose=True):
    """Collects (timing_us, secret_weight) pairs for a given key_gen function."""
    rows = []
    cursor = 0
    for trial in range(num_trials):
        batch = numbers[cursor:cursor + 60]
        if len(batch) < 30:
            if verbose:
                print(f"  Stopped early at trial {trial}: out of QRNG numbers")
            break
        qrng = QRNGSource(numbers=list(batch))
        rng = QEQSKERandom(qrng)

        t0 = time.perf_counter()
        keys = keygen_fn(rng, n, k)
        elapsed = time.perf_counter() - t0

        cursor += qrng.total_consumed
        weight, s_sum = _secret_properties(keys["private_key"])
        rows.append({"timing_us": elapsed * 1e6, "secret_weight": weight, "secret_sum": s_sum})

    return rows


def run_ml_with_cv(rows, label_name="secret_weight", n_splits=5, n_repeats=10, scenario_name="clean"):
    """Predicts above/below-median label using repeated stratified k-fold CV."""
    X = np.array([[r["timing_us"]] for r in rows])
    vals = np.array([r[label_name] for r in rows])
    median_v = np.median(vals)
    y = (vals > median_v).astype(int)

    if len(set(y)) < 2:
        print("  Not enough class variation to run classification.")
        return None

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=False)),
        "Neural Network (MLP)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42)),
    }

    baseline = max(np.mean(y == 0), np.mean(y == 1))
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    print(f"\n  Baseline (majority-class) accuracy = {baseline*100:.2f}%")
    print(f"  {'Model':<24}{'Accuracy (mean+-std)':>26}{'ROC-AUC (mean+-std)':>24}")
    results = {}
    for name, model in models.items():
        ml_stats = attack_c_stats.compute_ml_stats(model, X, y, rskf)
        acc_mean = ml_stats["accuracy"]["mean"]
        acc_std = ml_stats["accuracy"]["sd"]
        auc_mean = ml_stats["roc_auc"]["mean"]
        auc_std = ml_stats["roc_auc"]["sd"]
        
        results[name] = {"acc_mean": acc_mean, "acc_std": acc_std,
                          "auc_mean": auc_mean, "auc_std": auc_std}
        print(f"  {name:<24}{acc_mean*100:>9.2f}% +/- {acc_std*100:>5.2f}%"
              f"{'':<2}{auc_mean:>10.3f} +/- {auc_std:.3f}")
              
        attack_c_stats.update_json_results("../results/attack_c_full_stats.json", f"{scenario_name}_ml_{name}", ml_stats)

    return results, baseline


if __name__ == "__main__":
    with open("qrng_combined.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("#" * 70)
    print("POSITIVE CONTROL: QEQSKE-clean vs. QEQSKE-leaky")
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
