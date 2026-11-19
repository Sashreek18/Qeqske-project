"""
ATTACK C-5: Decryption (decrypt_it) Side-Channel Profiling
=============================================================
Addresses the gap identified in Section 9.5 — decrypt_it() was never
profiled for timing, memory, or CPU side-channels.

RESEARCH QUESTION:
  Does Alice's decryption operation leak secret-key information through
  observable execution traces (timing, memory, CPU)?

METHODOLOGY:
  1. Generate a fresh key pair per trial via key_gen().
  2. Bob encrypts a fixed known plaintext using encrypt_it().
  3. Profile Alice's decrypt_it() measuring:
       - Wall-clock timing (µs)
       - CPU user-time (µs)
       - Peak RAM (bytes)
       - Number of allocations
  4. Record the secret key's weight (count of non-zero entries in s).
  5. Train ML classifiers to predict above/below-median secret weight
     from the 4D side-channel trace of the DECRYPTION operation.
  6. Report statistical significance (Pearson r with Bonferroni correction)
     and ML accuracy vs. random baseline.

WHAT THIS TELLS US:
  If decryption timing/memory correlates with the secret key, then an
  adversary who can observe Alice's machine during decryption (e.g. via
  a co-located VM, power analysis, or EM emanation) may learn partial
  information about s.  This is a REAL side-channel threat model, distinct
  from the passive statistical attacks (A/B) on the ciphertext itself.
"""

import time
import tracemalloc
import os
import numpy as np
import psutil
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import attack_c_stats
from qrng_handler import QRNGSource
from qeqske_core import QEQSKERandom, key_gen, encrypt_it, decrypt_it


# ---------------------------------------------------------------------------
# Step 1 — Collect decryption side-channel traces
# ---------------------------------------------------------------------------

FIXED_PLAINTEXT = "HELLO"  # Known fixed plaintext for all trials


def _secret_weight(s):
    return sum(1 for row in s for v in row if v != 0)


def collect_decrypt_traces(numbers, n=4, k=2, num_trials=500, verbose=True):
    """
    For each trial:
      1. key_gen() → (s, A, t, q)
      2. encrypt_it(FIXED_PLAINTEXT) → (u, v_queue)
      3. Profile decrypt_it(s, u, v_queue, q) for timing/RAM/CPU
      4. Record secret_weight(s) as the target label
    """
    proc = psutil.Process(os.getpid())
    rows = []
    cursor = 0

    for trial in range(num_trials):
        # Need ~981 QRNG numbers per full exchange (key_gen + encrypt_it)
        batch = numbers[cursor:cursor + 2000]
        if len(batch) < 1000:
            if verbose:
                print(f"  Stopping early at trial {trial}: ran out of QRNG numbers")
            break

        qrng = QRNGSource(numbers=list(batch))
        rng = QEQSKERandom(qrng)

        # Key generation (not profiled here — see attack_c_combined_ml.py for that)
        keys = key_gen(rng, n=n, k=k)
        s = keys["private_key"]
        A = keys["public_key_A"]
        t = keys["public_key_t"]
        q = keys["q"]

        # Encryption (Bob's side — not profiled here)
        ct = encrypt_it(rng, FIXED_PLAINTEXT, A, t, q, n, k)

        cursor += qrng.total_consumed

        # --- PROFILE DECRYPTION (Alice's side) ---
        u = ct["u"]
        v_queue = ct["v_queue"]

        tracemalloc.start()
        cpu_before = proc.cpu_times()
        t0 = time.perf_counter()

        # THIS is what we're profiling
        result = decrypt_it(s, u, v_queue, q)

        elapsed = time.perf_counter() - t0
        cpu_after = proc.cpu_times()
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        num_allocs = sum(stat.count for stat in snapshot.statistics("lineno"))
        tracemalloc.stop()

        weight = _secret_weight(s)
        rows.append({
            "trial": trial,
            "timing_us": elapsed * 1e6,
            "cpu_user_us": (cpu_after.user - cpu_before.user) * 1e6,
            "ram_peak_bytes": peak,
            "num_allocations": num_allocs,
            "secret_weight": weight,
            "decryption_correct": (result == FIXED_PLAINTEXT),
            "q": q,
        })

    if verbose:
        correct = sum(1 for r in rows if r["decryption_correct"])
        print(f"\nCollected {len(rows)} decrypt_it() side-channel traces.")
        print(f"  Decryption correctness: {correct}/{len(rows)} "
              f"({100*correct/max(1,len(rows)):.1f}%)")
        times = [r["timing_us"] for r in rows]
        print(f"  Timing: mean={np.mean(times):.2f} µs, "
              f"std={np.std(times):.2f} µs, "
              f"min={np.min(times):.2f} µs, max={np.max(times):.2f} µs")

    return rows


# ---------------------------------------------------------------------------
# Step 2 — Statistical significance of each signal vs secret_weight
# ---------------------------------------------------------------------------

def report_decrypt_feature_significance(rows):
    features = ["timing_us", "cpu_user_us", "ram_peak_bytes", "num_allocations"]
    weights = [r["secret_weight"] for r in rows]

    print(f"\n{'=' * 70}")
    print("DECRYPT Side-Channel — Statistical Significance per Signal")
    print(f"{'=' * 70}")
    print(f"  {'Signal':<18}{'Pearson r':>12}{'p-value':>12}  Conclusion")

    for feat in features:
        vals = [r[feat] for r in rows]
        if len(set(vals)) > 1 and len(set(weights)) > 1:
            stats_dict = attack_c_stats.compute_correlation_stats(
                vals, weights, len(features)
            )
            r = stats_dict["pearson_r"]
            p = stats_dict["pearson_p"]
            bonf_p = stats_dict["bonferroni_p"]

            attack_c_stats.update_json_results(
                "results/attack_c_full_stats.json",
                f"decrypt_sidechannel_{feat}", stats_dict
            )
        else:
            r, p, bonf_p = 0.0, 1.0, 1.0

        sig = "SIGNIFICANT" if bonf_p < 0.05 else "not significant"
        print(f"  {feat:<18}{r:>12.4f}{p:>12.4f}  {sig}")
        if bonf_p < 0.05 and abs(r) > 0 and abs(r) < 0.15:
            print(f"    (Note: r = {r:.4f} means under "
                  f"{(r**2)*100:.2f}% of variance explained)")


# ---------------------------------------------------------------------------
# Step 3 — ML attack on decrypt_it() side-channels
# ---------------------------------------------------------------------------

def run_decrypt_ml_attack(rows, verbose=True):
    X = np.array([
        [r["timing_us"], r["cpu_user_us"], r["ram_peak_bytes"], r["num_allocations"]]
        for r in rows
    ])
    weights = np.array([r["secret_weight"] for r in rows])
    median_w = np.median(weights)
    y = (weights > median_w).astype(int)

    baseline_acc = max(np.mean(y == 0), np.mean(y == 1))

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf"),
        "Neural Net (MLP)": MLPClassifier(
            hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42
        ),
    }

    print(f"\n{'=' * 70}")
    print("DECRYPT Side-Channel — ML Attack")
    print(f"{'=' * 70}")
    print(f"  Target: predict above/below-median secret weight from decrypt traces")
    print(f"  Dataset: {len(rows)} trials, 5×10 Repeated Stratified K-Fold CV")
    print(f"  Features: timing_us, cpu_user_us, ram_peak_bytes, num_allocations")
    print(f"  Random baseline accuracy = {baseline_acc*100:.2f}%")

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    print(f"\n  {'Model':<24}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    results = {}

    for name, clf in models.items():
        pipeline = make_pipeline(StandardScaler(), clf)
        cv_results = cross_validate(
            pipeline, X, y, cv=rskf,
            scoring=("accuracy", "precision", "recall", "f1")
        )
        acc = cv_results["test_accuracy"].mean()
        prec = cv_results["test_precision"].mean()
        rec = cv_results["test_recall"].mean()
        f1 = cv_results["test_f1"].mean()

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"  {name:<24}{acc*100:>9.2f}%{prec*100:>10.2f}%{rec*100:>8.2f}%{f1*100:>7.2f}%")

        ml_stats = attack_c_stats.compute_ml_stats(pipeline, X, y, rskf)
        attack_c_stats.update_json_results(
            "results/attack_c_full_stats.json",
            f"decrypt_sidechannel_ml_{name}", ml_stats
        )

    if verbose:
        # Feature importances from Random Forest
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_s, y)
        importances = rf.feature_importances_
        feat_names = ["timing_us", "cpu_user_us", "ram_peak_bytes", "num_allocations"]

        print(f"\n  Random Forest feature importance (decrypt traces):")
        for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
            bar = "#" * int(imp * 40)
            print(f"    {name:<18} {imp:.4f}  {bar}")

        best_acc = max(r["accuracy"] for r in results.values())
        margin = best_acc - baseline_acc
        print(f"\n  Best model beat baseline by {margin*100:+.2f} percentage points.")
        if margin > 0.10:
            risk = "MEDIUM-HIGH"
        elif margin > 0.05:
            risk = "LOW-MEDIUM"
        else:
            risk = "LOW"
        print(f"  Decrypt side-channel leakage risk: {risk}")

    return results, baseline_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open("qrng_combined.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("\n" + "#" * 70)
    print("ATTACK C-5: Decryption Side-Channel Profiling (decrypt_it)")
    print("#" * 70)
    print(f"  Profiling {FIXED_PLAINTEXT!r} across 500 key pairs")
    print(f"  This addresses the Section 9.5 limitation.")

    rows = collect_decrypt_traces(numbers, n=4, k=2, num_trials=500)
    report_decrypt_feature_significance(rows)
    run_decrypt_ml_attack(rows)

    print("\n" + "#" * 70)
    print("ATTACK C-5 COMPLETE")
    print("#" * 70)
