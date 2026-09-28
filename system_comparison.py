"""
System Comparison: Random Baseline vs. Standard ML-KEM vs. QEQSKE
=====================================================================
Addresses the review comment: compare QEQSKE against standard Kyber/ML-KEM
under the same experimental pipeline.

We use `kyber-py` (a pure-Python implementation of ML-KEM, the NIST FIPS 203
standard derived from CRYSTALS-Kyber) as the STANDARD reference -- not a
custom reimplementation -- so this is a genuine external comparison point,
not something we built ourselves.

For each system we run the SAME experiment: measure key-generation timing
across many trials, and (where a comparable secret-key property exists)
try to predict something about the secret using the same ML pipeline.

NOTE: ML-KEM's secret key is a byte string, not a small-integer LWE-style
matrix like QEQSKE's, so a directly analogous "secret weight" doesn't
exist. For ML-KEM we instead test whether TIMING ALONE reveals which
security parameter set (ML-KEM-512 vs 768) was used -- this is the closest
directly comparable "does timing leak an operational secret" experiment.
"""

import time
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import attack_c_stats

from qrng_handler import QRNGSource
from qeqske_core import QEQSKERandom, key_gen as qeqske_keygen
from attack_c_leaky_vs_clean import key_gen_leaky as qeqske_keygen_leaky, _secret_properties

from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768


def time_ml_kem(num_trials=300, param_set=ML_KEM_512):
    """
    Times ML_KEM keygen with a FIXED parameter set (so both label classes
    come from the SAME public configuration -- unlike distinguishing 512 vs
    768, which is not a real secret and would trivially leak via timing).

    The "secret" property tested is the bit-level Hamming weight of the
    decapsulation key (dk) bytes -- the closest available analogue to
    QEQSKE's secret_weight, since kyber-py does not expose the raw LWE
    secret polynomial coefficients through its public API.
    """
    rows = []
    for trial in range(num_trials):
        t0 = time.perf_counter()
        ek, dk = param_set.keygen()
        elapsed = time.perf_counter() - t0
        # Hamming weight of the decapsulation key bytes (secret-key analogue)
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
        rng = QEQSKERandom(qrng)
        t0 = time.perf_counter()
        keys = keygen_fn(rng, n, k)
        elapsed = time.perf_counter() - t0
        cursor += qrng.total_consumed
        weight, _ = _secret_properties(keys["private_key"])
        rows.append({"timing_us": elapsed * 1e6, "label": int(weight > 5)})
    return rows


def run_attack(rows, name):
    X = np.array([[r["timing_us"]] for r in rows])
    y = np.array([r["label"] for r in rows])
    if len(set(y)) < 2:
        return {"system": name, "baseline": None, "best_model": None, "best_acc": None}

    baseline = max(np.mean(y == 0), np.mean(y == 1))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    ml_stats_lr = attack_c_stats.compute_ml_stats(model, X, y, rskf)
    ml_stats_rf = attack_c_stats.compute_ml_stats(rf, X, y, rskf)

    acc_lr_mean = ml_stats_lr["accuracy"]["mean"]
    acc_rf_mean = ml_stats_rf["accuracy"]["mean"]

    best_name = "Logistic Regression" if acc_lr_mean >= acc_rf_mean else "Random Forest"
    best_acc = max(acc_lr_mean, acc_rf_mean)
    best_stats = ml_stats_lr if acc_lr_mean >= acc_rf_mean else ml_stats_rf
    best_std = best_stats["accuracy"]["sd"]

    print(f"  {name:<28}baseline={baseline*100:5.2f}%   "
          f"best={best_name} {best_acc*100:5.2f}% +/- {best_std*100:.2f}%   "
          f"margin={((best_acc-baseline)*100):+.2f}pts")
          
    attack_c_stats.update_json_results("results/attack_c_full_stats.json", f"system_comp_{name}_LR", ml_stats_lr)
    attack_c_stats.update_json_results("results/attack_c_full_stats.json", f"system_comp_{name}_RF", ml_stats_rf)

    return {"system": name, "baseline": baseline, "best_model": best_name,
            "best_acc": best_acc, "best_std": best_std, "margin": best_acc - baseline}


if __name__ == "__main__":
    with open("qrng_combined.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("#" * 70)
    print("SYSTEM COMPARISON: Random baseline vs ML-KEM vs QEQSKE-clean vs QEQSKE-leaky")
    print("#" * 70)

    print("\nCollecting timing data...")
    print("  - ML-KEM-512 (fixed param set), 300 trials, label = dk byte-Hamming-weight above/below median...")
    mlkem_rows = time_ml_kem(num_trials=300)

    print("  - QEQSKE-clean, 300 trials...")
    clean_rows = time_qeqske(numbers, qeqske_keygen, num_trials=300)

    print("  - QEQSKE-leaky, 300 trials...")
    leaky_rows = time_qeqske(numbers, qeqske_keygen_leaky, num_trials=300)

    print("\n" + "=" * 70)
    print("Timing-based ML attack results (predict secret-relevant label from timing alone)")
    print("=" * 70)
    results = []
    results.append(run_attack(mlkem_rows, "Standard ML-KEM-512 (fixed params)"))
    results.append(run_attack(clean_rows, "QEQSKE-clean"))
    results.append(run_attack(leaky_rows, "QEQSKE-leaky (positive control)"))

    print("\nDone.")
