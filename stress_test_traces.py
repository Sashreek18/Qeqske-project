"""
STRESS TEST: Multi-Tier Confound Check, Held-Out CV Spread, Shuffled Control & Permutations
===========================================================================================
Evaluates Sender <-> Receiver <-> Public Channel <-> TCP Network RTT traces across:

1. SIGNAL DESCRIPTIVES & BOOTSTRAP CORRELATIONS:
   - Median (IQR), Mean (SD) for timing, CPU user-time, and TCP RTT signals
   - Spearman rho & Pearson r vs secret_weight with 2,000-resample bootstrap 95% CIs
   - Execution-order time-drift diagnostics & modulus-q confound diagnostics

2. 5-MODEL REPEATED STRATIFIED 5x10 CV (SEED=42):
   - Logistic Regression, Random Forest, SVM (RBF), MLP, Gradient Boosting
   - Majority-class baseline, Shuffled-label control, leak-free in-fold QResidualizer
   - Accuracy & ROC-AUC across all tiers (Sender, Receiver, Channel, TCP Orig, TCP CT, Combined)

3. PERMUTATION SIGNIFICANCE TESTING (200 PERMUTATIONS, 1/(N+1) FLOOR):
   - All 5 models tested on Tier 1b (Receiver Local) and Tier 1d (Network TCP RTT Original)
   - Best model tested on all remaining tiers

Saves:
  - results/stress_test_traces_5000.json (default)
  - results/stress_test_traces_5000_summary.md (default)
"""

import argparse
import json
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy import stats as sp_stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    permutation_test_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from attack_c_decrypt_5000 import (
    QResidualizer,
    _fold_stats,
    describe_array,
    paired_correlation_with_bootstrap,
    to_native,
)
from env_metadata import (
    format_perm_pvalue,
    get_environment_metadata,
    perm_pvalue_metadata,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

SEED = 42
N_BOOTSTRAP = 2000
DEFAULT_PERMUTATIONS = 200
DEFAULT_INPUT = "results/sender_receiver_traces_5000.json"
DEFAULT_OUTPUT = "results/stress_test_traces_5000.json"
DEFAULT_SUMMARY = "results/stress_test_traces_5000_summary.md"

CHANNEL_FEATURES = [
    "num_ct_blocks", "u_frobenius_mean", "u_frobenius_std",
    "u_mean", "u_std", "u_min", "u_max", "u_median", "u_skewness", "u_kurtosis",
    "v_norm_mean", "v_norm_std",
    "v_mean", "v_std", "v_min", "v_max", "v_median", "v_skewness", "v_kurtosis",
    "ct_total_elements", "ct_mean", "ct_std", "ct_entropy_approx",
    "u_mean_over_q", "v_mean_over_q", "ct_coefficient_of_variation",
]

SIGNAL_CANDIDATES = [
    "enc_timing_us",
    "enc_cpu_user_us",
    "dec_timing_us",
    "dec_cpu_user_us",
    "net_rtt_us",
    "net_ct_rtt_us",
]

DRIFT_PAIRS = [
    ("enc_timing_us", "enc_exec_order_pos"),
    ("enc_cpu_user_us", "enc_exec_order_pos"),
    ("dec_timing_us", "dec_exec_order_pos"),
    ("dec_cpu_user_us", "dec_exec_order_pos"),
    ("net_rtt_us", "net_exec_order_pos"),
    ("net_ct_rtt_us", "net_ct_exec_order_pos"),
]


def _get_models():
    return {
        "Logistic Reg": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=SEED, n_jobs=1
        ),
        "SVM (RBF)": lambda: SVC(kernel="rbf", random_state=SEED),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=SEED
        ),
        "Grad Boost": lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=SEED
        ),
    }


def _safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_rho": 0.0, "spearman_p": 1.0}
    pr, pp = sp_stats.pearsonr(x, y)
    sr, sp = sp_stats.spearmanr(x, y)
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


def build_tiers(available_keys):
    """Build feature tiers dynamically from available trace columns."""
    sender_cols = [
        f
        for f in ("enc_timing_us", "enc_cpu_user_us", "enc_ram_peak_bytes", "enc_num_allocations")
        if f in available_keys
    ]
    receiver_cols = [
        f
        for f in ("dec_timing_us", "dec_cpu_user_us", "dec_ram_peak_bytes", "dec_num_allocations")
        if f in available_keys
    ]
    channel_cols = [f for f in CHANNEL_FEATURES if f in available_keys]

    tiers = {}
    if sender_cols:
        tiers["Tier 1a: Sender Only (Clean)"] = sender_cols
    if receiver_cols:
        tiers["Tier 1b: Receiver Only (Clean Local)"] = receiver_cols
    if channel_cols:
        tiers["Tier 1c: Public Channel Only"] = channel_cols

    has_tcp_orig = "net_rtt_us" in available_keys
    has_tcp_ct = "net_ct_rtt_us" in available_keys
    tcp_orig_cols = [f for f in ("net_rtt_us", "net_payload_bytes") if f in available_keys]
    tcp_ct_cols = [f for f in ("net_ct_rtt_us", "net_payload_bytes") if f in available_keys]

    if has_tcp_orig:
        tiers["Tier 1d: Network TCP RTT (Original decrypt_it)"] = tcp_orig_cols
    if has_tcp_ct:
        tiers["Tier 1e: Network TCP RTT (Constant-Time)"] = tcp_ct_cols
    if sender_cols and receiver_cols:
        tiers["Tier 2: Sender + Receiver (Clean)"] = sender_cols + receiver_cols

    combined = list(sender_cols) + list(receiver_cols) + list(channel_cols)
    if has_tcp_orig:
        combined.extend(c for c in tcp_orig_cols if c not in combined)
    if combined:
        tiers["Tier 3: ALL Combined (Sender + Receiver + Channel + Net RTT)"] = combined

    tier_groups = {
        "Sender": sender_cols,
        "Receiver": receiver_cols,
        "Channel": channel_cols,
    }
    if has_tcp_orig or has_tcp_ct:
        net_group = []
        for c in ("net_rtt_us", "net_ct_rtt_us", "net_payload_bytes"):
            if c in available_keys:
                net_group.append(c)
        tier_groups["Network RTT"] = net_group

    return tiers, tier_groups


def write_summary_markdown(res, summary_path):
    """Write a concise, audit-friendly Markdown report from the saved JSON dict."""
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    env = res["environment"]
    n_trials = res["n_trials"]
    n_perm = res["n_permutations"]
    base_pct = res["majority_baseline"] * 100.0
    conf_q = res["confound_q_vs_secret_weight"]

    lines = [
        "# Sender <-> Receiver Multi-Tier Side-Channel Evaluation Summary",
        "",
        "## 1. Setup & Integrity",
        f"- **Input traces:** `{res['input_file']}` (`{n_trials}` trials, seed `{res['seed']}`)",
        f"- **Environment:** `{env['os']}`, CPU `{env.get('cpu_model', 'unknown')}`, Python `{env['python_version']}`, NumPy `{env['numpy_version']}`",
        f"- **Target:** `secret_weight > {res['median_secret_weight']}` (class 0: `{res['class_counts']['0']}`, class 1: `{res['class_counts']['1']}`, majority baseline: **{base_pct:.2f}%**)",
    ]

    corr_info = res.get("correctness", {})
    if corr_info:
        parts = [f"{k}: `{v}`" for k, v in corr_info.items()]
        lines.append(f"- **Correctness:** {', '.join(parts)}")

    lines.extend([
        "",
        "## 2. Signal Correlations vs. `secret_weight` (2,000-Resample Bootstrap 95% CIs)",
        "| Signal | Mean ± SD (µs) | Median (IQR) µs | Pearson r [95% CI] | Spearman ρ [95% CI] | Spearman p |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for sig, sdata in res["signal_correlations"].items():
        d = sdata["descriptives"]
        c = sdata["correlation"]
        p_lo, p_hi = c["pearson_boot_ci95"]
        s_lo, s_hi = c["spearman_boot_ci95"]
        lines.append(
            f"| `{sig}` | {d['mean']:.2f} ± {d['std']:.2f} | {d['median']:.2f} ({d['iqr']:.2f}) | "
            f"{c['pearson_r']:+.4f} [{p_lo:+.4f}, {p_hi:+.4f}] | "
            f"{c['spearman_rho']:+.4f} [{s_lo:+.4f}, {s_hi:+.4f}] | {c['spearman_p']:.2e} |"
        )

    lines.extend([
        "",
        "## 3. Confound & Execution-Order Time-Drift Checks",
        f"- **`q` vs `secret_weight`:** Pearson `r = {conf_q['pearson_r']:+.4f}` (`p = {conf_q['pearson_p']:.4f}`), Spearman `ρ = {conf_q['spearman_rho']:+.4f}` (`p = {conf_q['spearman_p']:.4f}`)",
    ])
    for dk, dv in res.get("time_drift_checks", {}).items():
        lines.append(f"- **Order drift (`{dk}`):** `r = {dv['pearson_r']:+.4f}` (`p = {dv['pearson_p']:.4f}`)")

    lines.extend([
        "",
        "## 4. Multi-Tier 5x10 Stratified CV Summary",
        f"Majority baseline: **{base_pct:.2f}%** | Permutation test resolution floor (`N={n_perm}`): `< 1/{n_perm + 1} ({1.0/(n_perm + 1):.4f})`",
        "",
        "| Tier | Active Features | Best Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized | Perm p-value | Verdict |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for tier_name, ts in res["tier_summary"].items():
        best_name = ts["best_model"]
        bm = res["tier_models"][tier_name][best_name]
        real_acc = bm["real"]["accuracy"]["mean"] * 100.0
        real_sd = bm["real"]["accuracy"]["sd"] * 100.0
        auc = bm["real"]["roc_auc"]["mean"]
        shuf_acc = bm["shuffled_control"]["accuracy"]["mean"] * 100.0
        gap = bm["gap_real_minus_shuffled"] * 100.0
        q_acc = bm["q_residualized"]["accuracy"]["mean"] * 100.0
        p_disp = bm["permutation_test"]["pvalue_display"] if "permutation_test" in bm else "—"
        lines.append(
            f"| **{tier_name}** | {ts['n_active_features']} | {best_name} | "
            f"{real_acc:.2f}% ± {real_sd:.2f}% | {auc:.4f} | {shuf_acc:.2f}% | "
            f"{gap:+.2f}% | {q_acc:.2f}% | {p_disp} | {ts['verdict']} |"
        )

    for focus_tier in (
        "Tier 1b: Receiver Only (Clean Local)",
        "Tier 1d: Network TCP RTT (Original decrypt_it)",
    ):
        if focus_tier not in res["tier_models"]:
            continue
        lines.extend([
            "",
            f"### Per-Model Breakdown: `{focus_tier}`",
            "| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized Acc | Perm p-value |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for m_name, m_data in res["tier_models"][focus_tier].items():
            ra = m_data["real"]["accuracy"]
            sa = m_data["shuffled_control"]["accuracy"]
            qa = m_data["q_residualized"]["accuracy"]
            auc = m_data["real"]["roc_auc"]["mean"]
            gap = m_data["gap_real_minus_shuffled"] * 100.0
            p_disp = (
                m_data["permutation_test"]["pvalue_display"]
                if "permutation_test" in m_data
                else "—"
            )
            lines.append(
                f"| {m_name} | {ra['mean']*100:.2f}% ± {ra['sd']*100:.2f}% | {auc:.4f} | "
                f"{sa['mean']*100:.2f}% ± {sa['sd']*100:.2f}% | {gap:+.2f}% | "
                f"{qa['mean']*100:.2f}% ± {qa['sd']*100:.2f}% | {p_disp} |"
            )

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_stress_test(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT,
                    summary_path=DEFAULT_SUMMARY, n_permutations=DEFAULT_PERMUTATIONS):
    t_all_0 = time.perf_counter()

    with open(input_path, "r") as f:
        raw_traces = json.load(f)
    rows = raw_traces["records"] if isinstance(raw_traces, dict) else raw_traces
    n_trials = len(rows)
    available_keys = set().union(*(r.keys() for r in rows))

    print(f"Loaded {n_trials} trace records from {input_path}")

    q_vals = np.array([r["q"] for r in rows], dtype=float)
    weights = np.array([r["secret_weight"] for r in rows], dtype=float)
    median_w = float(np.median(weights))
    y = (weights > median_w).astype(int)
    baseline_acc = float(max(np.mean(y == 0), np.mean(y == 1)))

    rng_shuf = np.random.RandomState(SEED)
    y_shuffled = rng_shuf.permutation(y)

    print(f"Secret weight median: {median_w} (0={int(np.sum(y==0))}, 1={int(np.sum(y==1))})")
    print(f"Majority Baseline Accuracy: {baseline_acc*100:.2f}%\n")

    tiers, tier_groups = build_tiers(available_keys)

    # --- Signal Descriptives & Bootstrap Correlations ---
    print("=" * 80)
    print("CHECK 1: SIGNAL CORRELATIONS & CONFOUND DIAGNOSTICS")
    print("=" * 80)

    signal_correlations = {}
    for sig in SIGNAL_CANDIDATES:
        if sig not in available_keys:
            continue
        vals = np.array([r[sig] for r in rows], dtype=float)
        desc = describe_array(vals)
        corr = paired_correlation_with_bootstrap(vals, weights, n_bootstrap=N_BOOTSTRAP, seed=SEED)
        signal_correlations[sig] = {"descriptives": desc, "correlation": corr}
        s_lo, s_hi = corr["spearman_boot_ci95"]
        print(
            f"  {sig:<18} med={desc['median']:7.2f} (IQR {desc['iqr']:5.2f}) | "
            f"rho={corr['spearman_rho']:+.4f} [{s_lo:+.4f}, {s_hi:+.4f}] (p={corr['spearman_p']:.2e})"
        )

    conf_q_sw = _safe_corr(q_vals, weights)
    print(
        f"\nModulus q vs secret_weight: Pearson r={conf_q_sw['pearson_r']:+.4f} (p={conf_q_sw['pearson_p']:.4f}), "
        f"Spearman rho={conf_q_sw['spearman_rho']:+.4f} (p={conf_q_sw['spearman_p']:.4f})"
    )

    time_drift_checks = {}
    seen_pos = set()
    for sig, pos_key in DRIFT_PAIRS:
        if sig in available_keys and pos_key in available_keys:
            pos_vals = np.array([r[pos_key] for r in rows], dtype=float)
            sig_vals = np.array([r[sig] for r in rows], dtype=float)
            time_drift_checks[f"{pos_key}_vs_{sig}"] = _safe_corr(pos_vals, sig_vals)
            if pos_key not in seen_pos:
                seen_pos.add(pos_key)
                time_drift_checks[f"{pos_key}_vs_secret_weight"] = _safe_corr(pos_vals, weights)

    per_feature_q = {}
    for gname, gfeats in tier_groups.items():
        per_feature_q[gname] = {}
        for feat in gfeats:
            vals = np.array([r.get(feat, 0.0) for r in rows], dtype=float)
            if np.std(vals) > 0:
                r_val, p_val = sp_stats.pearsonr(q_vals, vals)
                risk = "HIGH (|r|>0.5)" if abs(r_val) > 0.5 else ("MODERATE" if abs(r_val) > 0.2 else "LOW")
                per_feature_q[gname][feat] = {"pearson_r": float(r_val), "p_value": float(p_val), "risk": risk}
            else:
                per_feature_q[gname][feat] = {"pearson_r": None, "p_value": None, "risk": "NONE"}

    # --- Multi-Tier 5x10 CV + In-fold q-Residualizer + Permutation Tests ---
    print("\n" + "=" * 80)
    print("CHECKS 2-4: 5x10 REPEATED STRATIFIED CV, SHUFFLED CONTROL & PERMUTATION TESTS")
    print("=" * 80)

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    perm_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scoring = ("accuracy", "roc_auc")

    tier_models = {}
    tier_summary = {}

    for tier_name, tier_features in tiers.items():
        X_raw = np.array([[r.get(f, 0.0) for f in tier_features] for r in rows], dtype=float)
        col_stds = np.std(X_raw, axis=0)
        active_cols = col_stds > 0.0
        if not np.any(active_cols):
            print(f"\n  Skipping {tier_name}: all features constant")
            continue

        X_active = X_raw[:, active_cols]
        active_names = [f for f, a in zip(tier_features, active_cols) if a]
        X_with_q = np.column_stack([X_active, q_vals])

        print(f"\n{'─' * 80}")
        print(f"{tier_name} ({len(active_names)} active features)")
        print(f"{'─' * 80}")
        print(f"{'Model':<14} | {'REAL CV ACC':<28} | {'AUC':>6} | {'SHUFFLED':<16} | {'GAP':>8} | {'q-RESID':<16}")
        print(f"{'-'*14}-+-{'-'*28}-+-{'-'*6}-+-{'-'*16}-+-{'-'*8}-+-{'-'*16}")

        models_out = {}
        pipelines_real = {}

        for model_name, clf_fn in _get_models().items():
            pipe_real = make_pipeline(StandardScaler(), clf_fn())
            pipe_shuf = make_pipeline(StandardScaler(), clf_fn())
            pipe_q = make_pipeline(QResidualizer(), StandardScaler(), clf_fn())
            pipelines_real[model_name] = pipe_real

            cv_real = cross_validate(pipe_real, X_active, y, cv=rskf, scoring=scoring, n_jobs=-1)
            cv_shuf = cross_validate(pipe_shuf, X_active, y_shuffled, cv=rskf, scoring=scoring, n_jobs=-1)
            cv_q = cross_validate(pipe_q, X_with_q, y, cv=rskf, scoring=scoring, n_jobs=-1)

            def _pack(cv_dict):
                return {
                    met: _fold_stats(cv_dict[f"test_{met}"], clamp_unit=True)
                    for met in scoring
                }

            real_res = _pack(cv_real)
            shuf_res = _pack(cv_shuf)
            q_res = _pack(cv_q)

            r_acc = real_res["accuracy"]
            s_acc = shuf_res["accuracy"]
            q_acc = q_res["accuracy"]
            gap = r_acc["mean"] - s_acc["mean"]

            models_out[model_name] = {
                "real": real_res,
                "shuffled_control": shuf_res,
                "q_residualized": q_res,
                "delta_vs_baseline": float(r_acc["mean"] - baseline_acc),
                "gap_real_minus_shuffled": float(gap),
            }

            real_str = (
                f"{r_acc['mean']*100:5.2f}% ± {r_acc['sd']*100:4.2f}% "
                f"[{r_acc['min']*100:.1f}-{r_acc['max']*100:.1f}]"
            )
            shuf_str = f"{s_acc['mean']*100:5.2f}% ± {s_acc['sd']*100:4.2f}%"
            q_str = f"{q_acc['mean']*100:5.2f}% ± {q_acc['sd']*100:4.2f}%"
            print(
                f"{model_name:<14} | {real_str:<28} | {real_res['roc_auc']['mean']:6.4f} | "
                f"{shuf_str:<16} | {gap*100:+7.2f}% | {q_str:<16}"
            )

        # Permutation tests: all 5 models on primary Receiver & TCP Original tiers; best model otherwise
        best_model_name = max(
            models_out, key=lambda k: models_out[k]["real"]["accuracy"]["mean"]
        )
        if tier_name in (
            "Tier 1b: Receiver Only (Clean Local)",
            "Tier 1d: Network TCP RTT (Original decrypt_it)",
        ):
            perm_targets = list(models_out.keys())
        else:
            perm_targets = [best_model_name]

        for m_name in perm_targets:
            score, perm_scores, pval = permutation_test_score(
                pipelines_real[m_name],
                X_active,
                y,
                scoring="accuracy",
                cv=perm_cv,
                n_permutations=n_permutations,
                n_jobs=-1,
                random_state=SEED,
            )
            p_meta = perm_pvalue_metadata(pval, n_permutations)
            models_out[m_name]["permutation_test"] = {
                "scoring": "accuracy",
                "cv_folds": 3,
                "score": float(score),
                "perm_mean": float(np.mean(perm_scores)),
                "perm_std": float(np.std(perm_scores, ddof=1)),
                **p_meta,
            }
            print(f"  Perm Test [{m_name}]: score={score*100:.2f}%, p={p_meta['pvalue_display']}")

        tier_models[tier_name] = models_out
        bm = models_out[best_model_name]
        best_real = bm["real"]["accuracy"]["mean"]
        best_gap = bm["gap_real_minus_shuffled"]
        best_p = bm.get("permutation_test", {}).get("pvalue", 1.0)
        best_delta_base = bm["delta_vs_baseline"]

        if best_p < 0.01 and best_gap > 0.03 and best_delta_base > 0.03:
            verdict = "CONFIRMED SIGNAL (Perm p < 0.01, Robust to Shuffle & q)"
        elif best_p < 0.05 and best_gap > 0.02 and best_delta_base > 0.02:
            verdict = "WEAK / MODERATE SIGNAL"
        else:
            verdict = "NO SIGNAL (At Baseline / Indistinguishable from Shuffle)"

        tier_summary[tier_name] = {
            "active_features": active_names,
            "n_active_features": len(active_names),
            "best_model": best_model_name,
            "best_real": float(best_real),
            "best_roc_auc": float(bm["real"]["roc_auc"]["mean"]),
            "shuffled_acc": float(bm["shuffled_control"]["accuracy"]["mean"]),
            "gap": float(best_gap),
            "q_controlled": float(bm["q_residualized"]["accuracy"]["mean"]),
            "perm_pvalue": float(best_p),
            "perm_pvalue_display": bm.get("permutation_test", {}).get("pvalue_display", "—"),
            "verdict": verdict,
        }

    correctness = {}
    if "decryption_correct" in available_keys:
        c_dec = sum(1 for r in rows if r.get("decryption_correct"))
        correctness["local_decrypt"] = f"{c_dec}/{n_trials} ({100*c_dec/n_trials:.2f}%)"
    if "net_ack_correct" in available_keys:
        c_net = sum(1 for r in rows if r.get("net_ack_correct"))
        correctness["tcp_original_ack"] = f"{c_net}/{n_trials} ({100*c_net/n_trials:.2f}%)"
    if "net_ct_ack_correct" in available_keys:
        c_ct = sum(1 for r in rows if r.get("net_ct_ack_correct"))
        correctness["tcp_ct_ack"] = f"{c_ct}/{n_trials} ({100*c_ct/n_trials:.2f}%)"

    total_wall_s = time.perf_counter() - t_all_0

    output_dict = to_native({
        "environment": get_environment_metadata(),
        "input_file": input_path,
        "source_experiment": raw_traces.get("experiment") if isinstance(raw_traces, dict) else None,
        "qrng_stream": raw_traces.get("qrng_stream") if isinstance(raw_traces, dict) else None,
        "n_trials": n_trials,
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutations": n_permutations,
        "cv": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)",
        "median_secret_weight": median_w,
        "class_counts": {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))},
        "majority_baseline": baseline_acc,
        "correctness": correctness,
        "signal_correlations": signal_correlations,
        "confound_q_vs_secret_weight": conf_q_sw,
        "time_drift_checks": time_drift_checks,
        "per_feature_q_correlations": per_feature_q,
        "tier_models": tier_models,
        "tier_summary": tier_summary,
        "total_wall_seconds": total_wall_s,
    })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    write_summary_markdown(output_dict, summary_path)

    print("\n" + "=" * 80)
    print("FINAL STRESS-TEST VERDICT SUMMARY TABLE")
    print("=" * 80)
    for t_name, ts in tier_summary.items():
        print(
            f"  {t_name:<56} | {ts['best_model']:<13} | "
            f"{ts['best_real']*100:5.2f}% (gap {ts['gap']*100:+5.2f}%, "
            f"q-res {ts['q_controlled']*100:5.2f}%, p {ts['perm_pvalue_display']}) -> {ts['verdict']}"
        )
    print(f"\nSaved: {output_path}, {summary_path} ({total_wall_s:.1f}s)")
    return output_dict


def main():
    parser = argparse.ArgumentParser(description="Multi-tier stress test on Sender<->Receiver traces.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=str, default=DEFAULT_SUMMARY)
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_PERMUTATIONS)
    args = parser.parse_args()

    run_stress_test(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        n_permutations=args.n_permutations,
    )


if __name__ == "__main__":
    main()
