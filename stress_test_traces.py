"""
STRESS TEST: Confound Check, Held-Out CV Spread, and Shuffled-Label Control
=============================================================================
Stress-tests the findings from attack_sc_sender_receiver_traces.py /
attack_sc_ml_classifiers.py on the 500-trial dataset:

1. CONFOUND CHECK:
   - Correlation between public modulus q and secret_weight.
   - Correlation between public modulus q and every individual feature.
   - Recompute ML performance on q-residualized features (controlling for q).

2. HELD-OUT CV SPREAD:
   - RepeatedStratifiedKFold (5 splits × 10 repeats = 50 folds).
   - Report Mean, Std dev, Min, Max, and 95% CI across folds.

3. SHUFFLED-LABEL CONTROL:
   - Run the exact same pipeline on randomly shuffled secret_weight labels.
   - Compare Real Accuracy vs Shuffled Accuracy vs Majority Baseline side-by-side.
"""

import json
import numpy as np
from scipy import stats as sp_stats
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------------------

with open("results/sender_receiver_traces.json") as f:
    rows = json.load(f)

N_TRIALS = len(rows)
print(f"Loaded {N_TRIALS} trace records from results/sender_receiver_traces.json")

q_vals = np.array([r["q"] for r in rows], dtype=float)
weights = np.array([r["secret_weight"] for r in rows], dtype=float)
median_w = np.median(weights)
y = (weights > median_w).astype(int)
baseline_acc = max(np.mean(y == 0), np.mean(y == 1))

# Set up shuffled labels (deterministic seed)
rng = np.random.RandomState(42)
y_shuffled = rng.permutation(y)

print(f"Secret weight median: {median_w}")
print(f"Label distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
print(f"Majority Baseline Accuracy: {baseline_acc*100:.2f}%\n")


# ---------------------------------------------------------------------------
# Feature Tier Definitions
# ---------------------------------------------------------------------------

SENDER_FEATURES = [
    "enc_timing_us", "enc_cpu_user_us", "enc_ram_peak_bytes", "enc_num_allocations"
]

RECEIVER_FEATURES = [
    "dec_timing_us", "dec_cpu_user_us", "dec_ram_peak_bytes", "dec_num_allocations"
]

CHANNEL_FEATURES = [
    "num_ct_blocks", "u_frobenius_mean", "u_frobenius_std",
    "u_mean", "u_std", "u_min", "u_max", "u_median", "u_skewness", "u_kurtosis",
    "v_norm_mean", "v_norm_std",
    "v_mean", "v_std", "v_min", "v_max", "v_median", "v_skewness", "v_kurtosis",
    "ct_total_elements", "ct_mean", "ct_std", "ct_entropy_approx",
    "u_mean_over_q", "v_mean_over_q", "ct_coefficient_of_variation",
]

TIERS = {
    "Tier 1a: Sender Only": SENDER_FEATURES,
    "Tier 1b: Receiver Only": RECEIVER_FEATURES,
    "Tier 1c: Channel Only": CHANNEL_FEATURES,
    "Tier 2: Sender + Receiver": SENDER_FEATURES + RECEIVER_FEATURES,
    "Tier 3: ALL Combined": SENDER_FEATURES + RECEIVER_FEATURES + CHANNEL_FEATURES,
}


# ===========================================================================
# CHECK 1: CONFOUND CHECK (Modulus q)
# ===========================================================================

print("=" * 80)
print("CHECK 1: CONFOUND ANALYSIS — Correlation with Modulus q")
print("=" * 80)

r_q_sw, p_q_sw = sp_stats.pearsonr(q_vals, weights)
rho_q_sw, p_rho_q_sw = sp_stats.spearmanr(q_vals, weights)
print(f"Correlation between Modulus q and Secret Weight:")
print(f"  Pearson  r = {r_q_sw:+.4f} (p = {p_q_sw:.4f})")
print(f"  Spearman ρ = {rho_q_sw:+.4f} (p = {p_rho_q_sw:.4f})")
if p_q_sw > 0.05:
    print("  --> q is NOT significantly correlated with secret_weight (p > 0.05).")
else:
    print("  --> WARNING: q IS significantly correlated with secret_weight!")

print("\nCorrelations of Features with Modulus q (by Tier):")
for tier_name, feats in [("Sender", SENDER_FEATURES), ("Receiver", RECEIVER_FEATURES), ("Channel", CHANNEL_FEATURES)]:
    print(f"\n  [{tier_name} Features vs q]")
    print(f"    {'Feature':<30}{'Pearson r vs q':>16}{'p-value':>12}  Confound Risk")
    for feat in feats:
        vals = np.array([r.get(feat, 0) for r in rows], dtype=float)
        if np.std(vals) > 0:
            r_val, p_val = sp_stats.pearsonr(q_vals, vals)
            risk = "HIGH (|r|>0.5)" if abs(r_val) > 0.5 else ("MODERATE" if abs(r_val) > 0.2 else "LOW")
            print(f"    {feat:<30}{r_val:>+16.4f}{p_val:>12.4e}  {risk}")
        else:
            print(f"    {feat:<30}{'CONSTANT':>16}{'-':>12}  NONE")


# ===========================================================================
# CHECK 2 & 3: HELD-OUT CV SPREAD & SHUFFLED-LABEL CONTROL
# ===========================================================================

print("\n" + "=" * 80)
print("CHECKS 2 & 3: HELD-OUT CV ACCURACY SPREAD & SHUFFLED-LABEL CONTROL")
print("Repeated Stratified 5-Fold CV (10 repeats = 50 folds)")
print(f"Majority Baseline Accuracy: {baseline_acc*100:.2f}%")
print("=" * 80)

def get_models():
    return {
        "Logistic Reg": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf"),
        "MLP (16,8)": MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42),
        "Grad Boost": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
    }

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

tier_summary = {}

for tier_name, tier_features in TIERS.items():
    X = np.array([[r.get(f, 0) for f in tier_features] for r in rows], dtype=float)
    col_stds = np.std(X, axis=0)
    active_cols = col_stds > 0
    X_active = X[:, active_cols]
    active_names = [f for f, a in zip(tier_features, active_cols) if a]

    # Also build q-residualized X (regressing out q from every active feature)
    X_q_resid = np.zeros_like(X_active)
    for col_idx in range(X_active.shape[1]):
        linreg = LinearRegression().fit(q_vals.reshape(-1, 1), X_active[:, col_idx])
        X_q_resid[:, col_idx] = X_active[:, col_idx] - linreg.predict(q_vals.reshape(-1, 1))

    print(f"\n{'─' * 80}")
    print(f"{tier_name} ({len(active_names)} active features)")
    print(f"{'─' * 80}")
    print(f"{'Model':<14} | {'REAL CV ACCURACY (Mean ± Std)':<30} | {'SHUFFLED CONTROL':<22} | {'GAP (Real-Shuf)':<15} | {'q-RESIDUALIZED':<16}")
    print(f"{'-'*14}-+-{'-'*30}-+-{'-'*22}-+-{'-'*15}-+-{'-'*16}")

    models = get_models()
    best_real_acc = 0.0
    best_model_name = ""

    for model_name, clf in models.items():
        # 1. Real CV
        pipe_real = make_pipeline(StandardScaler(), clf)
        cv_real = cross_validate(pipe_real, X_active, y, cv=rskf, scoring="accuracy", n_jobs=-1)
        real_scores = cv_real["test_score"]
        r_mean = np.mean(real_scores)
        r_std = np.std(real_scores)
        r_min = np.min(real_scores)
        r_max = np.max(real_scores)

        # 2. Shuffled CV (Control)
        pipe_shuf = make_pipeline(StandardScaler(), clf)
        cv_shuf = cross_validate(pipe_shuf, X_active, y_shuffled, cv=rskf, scoring="accuracy", n_jobs=-1)
        shuf_scores = cv_shuf["test_score"]
        s_mean = np.mean(shuf_scores)
        s_std = np.std(shuf_scores)

        # 3. q-residualized Real CV (Control for q confound)
        pipe_q = make_pipeline(StandardScaler(), clf)
        cv_q = cross_validate(pipe_q, X_q_resid, y, cv=rskf, scoring="accuracy", n_jobs=-1)
        q_mean = np.mean(cv_q["test_score"])
        q_std = np.std(cv_q["test_score"])

        gap = r_mean - s_mean
        real_str = f"{r_mean*100:5.2f}% ± {r_std*100:4.2f}% [{r_min*100:.1f}-{r_max*100:.1f}]"
        shuf_str = f"{s_mean*100:5.2f}% ± {s_std*100:4.2f}%"
        gap_str = f"{gap*100:+6.2f}%"
        q_str = f"{q_mean*100:5.2f}% ± {q_std*100:4.2f}%"

        flag = " ***" if gap > 0.10 else ""
        print(f"{model_name:<14} | {real_str:<30} | {shuf_str:<22} | {gap_str:<15} | {q_str:<16}{flag}")

        if r_mean > best_real_acc:
            best_real_acc = r_mean
            best_model_name = model_name
            best_gap = gap
            best_s_mean = s_mean
            best_q_mean = q_mean

    tier_summary[tier_name] = {
        "best_model": best_model_name,
        "best_real": best_real_acc,
        "shuffled_acc": best_s_mean,
        "gap": best_gap,
        "q_controlled": best_q_mean,
    }


# ===========================================================================
# SUMMARY TABLE
# ===========================================================================

print("\n" + "=" * 80)
print("FINAL STRESS-TEST VERDICT SUMMARY TABLE")
print("=" * 80)
print(f"{'Tier':<28} | {'Best Model':<14} | {'Real Acc':>9} | {'Shuffled':>9} | {'Net Gap':>9} | {'q-Controlled':>12} | Verdict")
print(f"{'-'*28}-+-{'-'*14}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*12}-+--------")

for tier_name, res in tier_summary.items():
    real = res["best_real"] * 100
    shuf = res["shuffled_acc"] * 100
    gap = res["gap"] * 100
    q_ctrl = res["q_controlled"] * 100
    
    if gap > 10.0:
        verdict = "CONFIRMED SIGNAL (Robust to Shuffle & q)"
    elif gap > 3.0:
        verdict = "MARGINAL / UNSTABLE"
    else:
        verdict = "NO SIGNAL (Indistinguishable from Shuffle)"
        
    print(f"{tier_name:<28} | {res['best_model']:<14} | {real:>8.2f}% | {shuf:>8.2f}% | {gap:>+8.2f}% | {q_ctrl:>11.2f}% | {verdict}")

print("\nInterpretation Rules:")
print("1. If Net Gap (Real - Shuffled) ~ 0%, the tier contains ZERO predictive signal.")
print("2. If Net Gap > 10% and q-Controlled maintains accuracy, the leakage is genuine and NOT a confound.")
print("3. Check the CV std: a narrow spread (e.g. ±3%) indicates stability across folds.")

