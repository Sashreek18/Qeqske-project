"""
ATTACK C (Part 4): Combined Side-Channel ML Attack Pipeline
==============================================================
This ties together the three Attack C legs (Time, RAM, CPU/Cache-proxy)
into ONE unified dataset per trial, then trains real ML classifiers on
it -- because a correlation coefficient is a statistical test, not a
machine learning attack, and the project title specifically claims
"Machine Learning-Based" attacks.

Renamed terminology (matches what the code has always actually measured):
  Attack C-1: Timing Side-Channel Analysis
  Attack C-2: Memory (RAM) Side-Channel Analysis
  Attack C-3: CPU Resource-Usage Side-Channel Analysis (software proxy for
              real cache/power hardware counters, which we don't have)
  Attack C-4 (this file): Combined ML Side-Channel Attack
              -- do these three *signals together* let an ML model predict
              something about the secret key, even if each signal alone
              (per attack_c_ram_sidechannel.py / attack_c_cache_sidechannel.py)
              showed only weak correlation?

WHAT WE PREDICT (not the full secret key):
  Predicting the entire secret key from side-channels is unrealistic for a
  software-only study. Instead we predict a defensible derived property:
  whether a trial's secret key has an ABOVE- or BELOW-median "weight"
  (count of non-zero entries in the private key matrix s). This is a
  binary classification problem, so a baseline of ~50% is the honest
  "no attack" floor -- same logic as Attack A.

METHODOLOGY:
  1. Run key_gen() once per trial (single real call, not a repeat-batch --
     unlike attack_c_cache_sidechannel.py's CPU script, here we need one
     row per REAL secret key, so timing/CPU are measured per single call).
  2. Record: elapsed time, CPU user-time delta, peak memory, allocation
     count -- all four signals for the SAME trial/secret key.
  3. Label each trial: secret_weight above or below the median (binary).
  4. Train/test split (70/30), train multiple classical ML models.
  5. Report Accuracy, Precision, Recall, F1, confusion matrix, feature
     importance (Random Forest), and compare against random-guess baseline.
  6. Report Pearson correlation + p-value for each individual feature vs.
     secret_weight, instead of a flat "correlation > 0.3" threshold.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from qrng_handler import QRNGSource
from qeqske_core import QEQSKERandom, key_gen


# ---------------------------------------------------------------------------
# Step 1 -- Combined per-trial data collection (Time + RAM + CPU together)
# ---------------------------------------------------------------------------

def _secret_properties(s):
    s_flat = [val for row in s for val in row]
    secret_weight = sum(1 for v in s_flat if v != 0)
    secret_sum = sum(s_flat)
    return secret_weight, secret_sum


def collect_combined_dataset(numbers, n=4, k=2, num_trials=500, verbose=True):
    """
    Runs key_gen() once per trial and measures Time + RAM + CPU together
    for that SAME call, so every row of the dataset describes one real
    secret key observed through all three side-channels at once.

    NOTE ON CPU-TIME RESOLUTION: a single key_gen() at toy parameters
    (n=4, k=2) finishes in microseconds, below the OS's per-process CPU
    clock tick (~1-10 ms on Linux). Unlike attack_c_cache_sidechannel.py
    (which repeats each call hundreds of times to get a measurable batch
    signal), here we need ONE call per secret key, so cpu_user_time will
    read as 0.0 for most trials. We keep the column anyway and report it
    honestly -- if the ML model's feature importance shows it contributes
    ~nothing, that itself is a valid, reportable finding (see Section 4
    of the writeup: "which signal contributes most information").
    """
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
        rng = QEQSKERandom(qrng)

        tracemalloc.start()
        proc.cpu_percent(interval=None)
        cpu_before = proc.cpu_times()
        t0 = time.perf_counter()

        keys = key_gen(rng, n=n, k=k)

        elapsed = time.perf_counter() - t0
        cpu_after = proc.cpu_times()
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        num_allocs = sum(stat.count for stat in snapshot.statistics("lineno"))
        tracemalloc.stop()

        cursor += qrng.total_consumed
        secret_weight, secret_sum = _secret_properties(keys["private_key"])

        rows.append({
            "trial": trial,
            "timing_us": elapsed * 1e6,
            "cpu_user_us": (cpu_after.user - cpu_before.user) * 1e6,
            "ram_peak_bytes": peak,
            "num_allocations": num_allocs,
            "secret_weight": secret_weight,
            "secret_sum": secret_sum,
        })

    if verbose:
        print(f"Collected {len(rows)} combined trials "
              f"(Time + RAM + CPU measured together per secret key).")

    return rows


# ---------------------------------------------------------------------------
# Step 2 -- Statistical significance per individual signal (replaces the
# old flat "correlation > 0.3" threshold with a real p-value)
# ---------------------------------------------------------------------------

def report_feature_significance(rows):
    features = ["timing_us", "cpu_user_us", "ram_peak_bytes", "num_allocations"]
    weights = [r["secret_weight"] for r in rows]

    print(f"\n{'=' * 68}")
    print("Statistical Significance — each signal vs. secret_weight")
    print(f"{'=' * 68}")
    print(f"  {'Signal':<18}{'Pearson r':>12}{'p-value':>12}  Conclusion")
    for feat in features:
        vals = [r[feat] for r in rows]
        if len(set(vals)) > 1 and len(set(weights)) > 1:
            r, p = stats.pearsonr(vals, weights)
        else:
            r, p = 0.0, 1.0
        sig = "SIGNIFICANT (p<0.0125)" if p < 0.0125 else "not significant (Bonferroni)"
        print(f"  {feat:<18}{r:>12.4f}{p:>12.4f}  {sig}")
        if p < 0.0125 and abs(r) > 0 and abs(r) < 0.15:
            print(f"    (Note: r = {r:.4f} means under {(r**2)*100:.2f}% of variance explained; detectable but practically useless)")


# ---------------------------------------------------------------------------
# Step 3 -- Real ML attack: predict above/below-median secret weight from
# the four combined side-channel features
# ---------------------------------------------------------------------------

def run_ml_attack(rows, verbose=True):
    X = np.array([[r["timing_us"], r["cpu_user_us"], r["ram_peak_bytes"], r["num_allocations"]]
                  for r in rows])
    weights = np.array([r["secret_weight"] for r in rows])
    median_w = np.median(weights)
    y = (weights > median_w).astype(int)  # binary label: above/below median weight

    from sklearn.pipeline import make_pipeline
    
    # We still keep a scaler for the full-dataset feature importances
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf"),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42),
    }

    baseline_acc = max(np.mean(y == 0), np.mean(y == 1))  # majority-class guess

    print(f"\n{'=' * 68}")
    print(f"Combined ML Attack — predicting above/below-median secret weight")
    print(f"{'=' * 68}")
    print(f"  Dataset: {len(rows)} trials -> Repeated Cross-Validation (5 splits, 10 repeats)")
    print(f"  Features: timing_us, cpu_user_us, ram_peak_bytes, num_allocations")
    print(f"  Random/majority-class baseline accuracy = {baseline_acc*100:.2f}%")

    results = {}
    print(f"\n  {'Model':<24}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    
    for name, clf in models.items():
        model = make_pipeline(StandardScaler(), clf)
        cv_results = cross_validate(model, X, y, cv=rskf, scoring=("accuracy", "precision", "recall", "f1"))
        acc = cv_results['test_accuracy'].mean()
        prec = cv_results['test_precision'].mean()
        rec = cv_results['test_recall'].mean()
        f1 = cv_results['test_f1'].mean()
        
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"  {name:<24}{acc*100:>9.2f}%{prec*100:>10.2f}%{rec*100:>8.2f}%{f1*100:>7.2f}%")

    if verbose:
        # Fit Random Forest on full dataset to extract overall feature importances
        rf = models["Random Forest"]
        rf.fit(X_s, y)
        importances = rf.feature_importances_
        feat_names = ["timing_us", "cpu_user_us", "ram_peak_bytes", "num_allocations"]
        print(f"\n  Random Forest feature importance (which signal matters most):")
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
        print(f"  Combined-signal leakage risk classification: {risk}")
        print(f"\n  NOTE: this predicts a derived property (above/below-median secret")
        print(f"  weight), NOT the secret key itself -- recovering the full key from")
        print(f"  software side-channels alone is not a realistic or claimed outcome.")

    return results, baseline_acc


if __name__ == "__main__":
    with open("qrng_combined.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("\n" + "#" * 70)
    print("ATTACK C-4: Combined Side-Channel Dataset Collection")
    print("#" * 70)
    rows = collect_combined_dataset(numbers, n=4, k=2, num_trials=500)

    report_feature_significance(rows)
    run_ml_attack(rows)
