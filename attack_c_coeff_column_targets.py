#!/usr/bin/env python3
"""
Coefficient-Level, Column-Level, and Key-Recovery Bridge Analysis (5,000 Trials)
=================================================================================
Addresses two key questions connecting side-channel leakage to cryptanalysis:
  1. Do passive timing / channel traces leak *individual* secret coefficients
     s_{i,j} in {-1, 0, +1} or *column-level* structure (w(c_0), w(c_1),
     all-zero column detection) across 5,000 deterministically replayed trials?
  2. How much mutual information (in bits vs. H(S) = 8 * log2(3) = 12.68 bits)
     does passive single-trace leakage reveal, vs. chosen-ciphertext timing probes
     that isolate one secret row at a time (mitigated by decrypt_it_constant_time)?

Outputs: results/coeff_column_leak_5000.json
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import math
import time
import numpy as np
from scipy import stats as sp_stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, cross_validate, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from env_metadata import get_environment_metadata
from qeqske_core import QEQSKERandom, decrypt_it, decrypt_it_constant_time, encrypt_it, key_gen
from qrng_handler import MersenneSource, QRNGSource

SEED = 42


class QResidualizer(BaseEstimator, TransformerMixin):
    """In-fold linear OLS modulus-q residualizer on all preceding feature columns."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        feats, q = X[:, :-1], X[:, -1]
        q_mean = float(np.mean(q))
        qc = q - q_mean
        denom = float(np.dot(qc, qc))
        f_mean = np.mean(feats, axis=0)
        slopes = (qc @ (feats - f_mean)) / denom if denom > 0.0 else np.zeros(feats.shape[1])
        self.slopes_ = slopes
        self.intercepts_ = f_mean - slopes * q_mean
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, :-1] - (self.intercepts_[None, :] + X[:, -1:] * self.slopes_[None, :])


def replay_secrets_and_verify(records: list[dict], qrng_path: str) -> list[list[list[int]]]:
    """Deterministically replay 5,000 trials and verify 0 mismatches against saved records."""
    src = QRNGSource.from_file(qrng_path, cycle=True)
    rng = QEQSKERandom(src)
    secrets = []
    mismatches = 0
    for rec in records:
        kd = key_gen(rng, n=4, k=2)
        s = kd["private_key"]
        q = kd["q"]
        encrypt_it(rng, "HELLO", kd["public_key_A"], kd["public_key_t"], q, 4, 2)
        sw = sum(1 for row in s for v in row if v != 0)
        if q != rec["q"] or sw != rec["secret_weight"]:
            mismatches += 1
        secrets.append(s)
    if mismatches != 0:
        raise RuntimeError(f"Replay mismatch count={mismatches} for {qrng_path}")
    return secrets


def eval_binary_target(
    X: np.ndarray,
    q_vals: np.ndarray,
    y: np.ndarray,
    primary_feat: np.ndarray,
    n_perm: int = 100,
    detailed: bool = True,
) -> dict:
    counts = {str(k): int(np.sum(y == k)) for k in sorted(set(y.tolist()))}
    baseline = float(max(counts.values()) / len(y))
    rho_prim, p_prim = sp_stats.spearmanr(primary_feat, y)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED) if detailed else StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    perm_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scoring = ("accuracy", "balanced_accuracy", "roc_auc")
    X_q = np.column_stack([X, q_vals])

    clfs = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
        "HistGB": lambda: HistGradientBoostingClassifier(max_iter=80, max_depth=4, random_state=SEED),
        "SVM (RBF)": lambda: SVC(kernel="rbf", cache_size=500, random_state=SEED),
    }

    best_name, best_acc, best_bal, best_auc, best_pipe = None, -1.0, 0.0, 0.5, None
    for name, fn in clfs.items():
        pipe = make_pipeline(StandardScaler(), fn())
        sc = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        acc = float(np.mean(sc["test_accuracy"]))
        bal = float(np.mean(sc["test_balanced_accuracy"]))
        auc = float(np.mean(sc["test_roc_auc"]))
        if acc > best_acc:
            best_name, best_acc, best_bal, best_auc, best_pipe = name, acc, bal, auc, fn

    pipe_q = make_pipeline(QResidualizer(), StandardScaler(), best_pipe())
    sc_q = cross_validate(pipe_q, X_q, y, cv=cv, scoring=("accuracy", "balanced_accuracy"), n_jobs=-1)
    q_acc = float(np.mean(sc_q["test_accuracy"]))
    q_bal = float(np.mean(sc_q["test_balanced_accuracy"]))

    # Use best_pipe directly when detailed (global/column/CC-probe), else avoid nested slow SVM 100-perm repeats
    perm_fn = best_pipe if (detailed or best_name != "SVM (RBF)") else clfs["HistGB"]
    _, _, perm_p = permutation_test_score(
        make_pipeline(StandardScaler(), perm_fn()),
        X, y, scoring="accuracy", cv=perm_cv, n_permutations=n_perm, random_state=SEED, n_jobs=-1,
    )

    return {
        "n": int(len(y)),
        "class_counts": counts,
        "majority_baseline": baseline,
        "spearman_rho": float(rho_prim),
        "spearman_p": float(p_prim),
        "best_model": best_name,
        "best_accuracy": best_acc,
        "best_balanced_accuracy": best_bal,
        "best_roc_auc": best_auc,
        "margin_above_baseline": float(best_acc - baseline),
        "q_residualized_accuracy": q_acc,
        "q_residualized_balanced_accuracy": q_bal,
        "perm_pvalue": float(perm_p),
    }


def eval_multiclass_target(X: np.ndarray, y: np.ndarray, n_perm: int = 100) -> dict:
    _, cnts = np.unique(y, return_counts=True)
    baseline = float(cnts.max() / len(y))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    perm_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    clfs = {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
        "HistGB": lambda: HistGradientBoostingClassifier(max_iter=80, max_depth=4, random_state=SEED),
        "SVM (RBF)": lambda: SVC(kernel="rbf", cache_size=500, random_state=SEED),
    }

    best_name, best_acc, best_bal, best_fn = None, -1.0, 0.0, None
    for name, fn in clfs.items():
        sc = cross_validate(make_pipeline(StandardScaler(), fn()), X, y, cv=cv, scoring=("accuracy", "balanced_accuracy"), n_jobs=-1)
        acc = float(np.mean(sc["test_accuracy"]))
        bal = float(np.mean(sc["test_balanced_accuracy"]))
        if acc > best_acc:
            best_name, best_acc, best_bal, best_fn = name, acc, bal, fn

    perm_fn = best_fn if best_name != "SVM (RBF)" else clfs["HistGB"]
    _, _, perm_p = permutation_test_score(
        make_pipeline(StandardScaler(), perm_fn()),
        X, y, scoring="accuracy", cv=perm_cv, n_permutations=n_perm, random_state=SEED, n_jobs=-1,
    )
    return {
        "n": int(len(y)),
        "majority_baseline": baseline,
        "best_model": best_name,
        "best_accuracy": best_acc,
        "best_balanced_accuracy": best_bal,
        "margin_above_baseline": float(best_acc - baseline),
        "perm_pvalue": float(perm_p),
    }


def eval_regression_target(X: np.ndarray, y: np.ndarray, primary_feat: np.ndarray) -> dict:
    rho_prim, p_prim = sp_stats.spearmanr(primary_feat, y)
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    best_name, best_r2 = None, -1e9
    for name, fn in [
        ("Ridge", lambda: Ridge(alpha=1.0)),
        ("HistGB", lambda: HistGradientBoostingRegressor(max_iter=80, max_depth=4, random_state=SEED)),
    ]:
        sc = cross_validate(make_pipeline(StandardScaler(), fn()), X, y, cv=rkf, scoring="r2", n_jobs=-1)
        r2 = float(np.mean(sc["test_score"]))
        if r2 > best_r2:
            best_name, best_r2 = name, r2
    return {
        "spearman_rho": float(rho_prim),
        "spearman_p": float(p_prim),
        "best_model": best_name,
        "best_r2": best_r2,
    }


def discrete_mi_bits(x: np.ndarray, y: np.ndarray, n_bins: int = 16) -> float:
    """Quantile-binned discrete mutual information I(X; Y) in bits."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    bx = np.zeros_like(x, dtype=int) if len(edges) <= 2 else np.digitize(x, edges[1:-1], right=True)
    vals_x = np.unique(bx)
    vals_y = np.unique(y)
    N = float(len(x))
    mi = 0.0
    for vx in vals_x:
        mx = bx == vx
        px = np.sum(mx) / N
        for vy in vals_y:
            pxy = np.sum(mx & (y == vy)) / N
            if pxy > 0.0:
                py = np.sum(y == vy) / N
                mi += pxy * math.log2(pxy / (px * py))
    return float(max(0.0, mi))


def evaluate_suite(
    name: str,
    traces_path: str,
    qrng_path: str,
    feature_keys: list[str],
    primary_feat_key: str,
    full_8_coeffs: bool = True,
) -> dict:
    print(f"\n--- Evaluating {name} ({traces_path}, full_8_coeffs={full_8_coeffs}) ---")
    with open(traces_path, "r") as f:
        data = json.load(f)
    records = data["records"]
    secrets = replay_secrets_and_verify(records, qrng_path)
    print(f"  Replay verified: {len(secrets)}/5,000 trials matched (0 mismatches)")

    X = np.array([[r[k] for k in feature_keys] for r in records], dtype=float)
    q_vals = np.array([r["q"] for r in records], dtype=float)
    primary = np.array([r[primary_feat_key] for r in records], dtype=float)
    S = np.array(secrets, dtype=int)

    w_global = np.sum(S != 0, axis=(1, 2))
    y_glob = (w_global > np.median(w_global)).astype(int)
    glob_res = eval_binary_target(X, q_vals, y_glob, primary, detailed=True)
    glob_mi_knn = float(mutual_info_classif(primary[:, None], y_glob, n_neighbors=5, random_state=SEED)[0] / math.log(2.0))
    glob_mi_hist = discrete_mi_bits(primary, y_glob, n_bins=16)

    w_c0 = np.sum(S[:, :, 0] != 0, axis=1)
    w_c1 = np.sum(S[:, :, 1] != 0, axis=1)
    has_zero_col = ((w_c0 == 0) | (w_c1 == 0)).astype(int)

    col_res = {
        "col0_weight_binary": {
            **eval_binary_target(X, q_vals, (w_c0 > np.median(w_c0)).astype(int), primary, detailed=True),
            "regression": eval_regression_target(X, w_c0.astype(float), primary),
        },
        "col1_weight_binary": {
            **eval_binary_target(X, q_vals, (w_c1 > np.median(w_c1)).astype(int), primary, detailed=True),
            "regression": eval_regression_target(X, w_c1.astype(float), primary),
        },
        "has_zero_column": eval_binary_target(X, q_vals, has_zero_col, primary, detailed=True),
    }

    coeff_res = {}
    max_nz_acc = 0.0
    max_nz_bal = 0.0
    nz_mi_vals = []
    pairs = [(i, j) for i in range(4) for j in range(2)] if full_8_coeffs else [(0, 0), (0, 1)]

    for i, j in pairs:
        vals = S[:, i, j]
        y_nz = (vals != 0).astype(int)
        r_nz = eval_binary_target(X, q_vals, y_nz, primary, detailed=False)
        r_3c = eval_multiclass_target(X, vals + 1)
        mask_nz = vals != 0
        y_sign = (vals[mask_nz] > 0).astype(int)
        r_sign = eval_binary_target(X[mask_nz], q_vals[mask_nz], y_sign, primary[mask_nz], detailed=False)

        mi_knn = float(mutual_info_classif(primary[:, None], y_nz, n_neighbors=5, random_state=SEED)[0] / math.log(2.0))
        mi_hist = discrete_mi_bits(primary, y_nz, n_bins=16)
        nz_mi_vals.append(mi_hist)
        max_nz_acc = max(max_nz_acc, r_nz["best_accuracy"])
        max_nz_bal = max(max_nz_bal, r_nz["best_balanced_accuracy"])

        coeff_res[f"s[{i}][{j}]"] = {
            "nonzero_binary": r_nz,
            "exact_3class": r_3c,
            "sign_given_nonzero": r_sign,
            "mi_bits_knn": mi_knn,
            "mi_bits_hist16": mi_hist,
        }
        print(f"    s[{i}][{j}] nz={r_nz['best_accuracy']*100:.2f}% (base={r_nz['majority_baseline']*100:.2f}%, bal={r_nz['best_balanced_accuracy']*100:.2f}%, p={r_nz['perm_pvalue']:.3f}) | 3cls={r_3c['best_accuracy']*100:.2f}% | sign={r_sign['best_accuracy']*100:.2f}%")

    print(f"    Columns: w(c0)={col_res['col0_weight_binary']['best_accuracy']*100:.2f}% (R2={col_res['col0_weight_binary']['regression']['best_r2']:+.4f}) | w(c1)={col_res['col1_weight_binary']['best_accuracy']*100:.2f}% (R2={col_res['col1_weight_binary']['regression']['best_r2']:+.4f}) | zero_col bal={col_res['has_zero_column']['best_balanced_accuracy']*100:.2f}% auc={col_res['has_zero_column']['best_roc_auc']:.4f} p={col_res['has_zero_column']['perm_pvalue']:.3f}")

    return {
        "traces_file": traces_path,
        "qrng_file": qrng_path,
        "n_trials": int(len(records)),
        "zero_column_count": int(np.sum(has_zero_col)),
        "zero_column_pct": float(100.0 * np.mean(has_zero_col)),
        "global_weight_binary": {
            **glob_res,
            "mi_bits_knn": glob_mi_knn,
            "mi_bits_hist16": glob_mi_hist,
        },
        "column_targets": col_res,
        "coefficients": coeff_res,
        "summary": {
            "coeff_positions_evaluated": len(pairs),
            "max_coeff_nonzero_acc": float(max_nz_acc),
            "max_coeff_nonzero_bal_acc": float(max_nz_bal),
            "mean_coeff_nz_mi_bits_hist16": float(np.mean(nz_mi_vals)),
        },
    }


def run_chosen_ciphertext_probe(n_keys: int = 300, row_idx: int = 0) -> dict:
    """
    300-key chosen-ciphertext probe isolating row `row_idx` of S via sparse u_probe
    (only row `row_idx` nonzero). Compares decrypt_it() vs. decrypt_it_constant_time().
    """
    print(f"\n--- Running Chosen-Ciphertext Row-{row_idx} Probe ({n_keys} keys) ---")
    src = QRNGSource.from_file("qrng_combined.txt", cycle=True)
    rng = QEQSKERandom(src)
    trials = []
    for tid in range(n_keys):
        kd = key_gen(rng, n=4, k=2)
        s, q = kd["private_key"], kd["q"]
        w_row = int(sum(1 for v in s[row_idx] if v != 0))
        u_mat = [[0, 0, 0, 0], [0, 0, 0, 0]]
        u_mat[0][row_idx] = (tid * 997 + 12345) % q or 1
        u_mat[1][row_idx] = (tid * 991 + 23456) % q or 1
        u_q = [u_mat for _ in range(40)]
        v_q = [[[1, 1], [1, 1]] for _ in range(40)]
        trials.append((s, u_q, v_q, q, w_row))

    w_src = MersenneSource(seed=42099)
    w_rng = QEQSKERandom(w_src)
    w_kd = key_gen(w_rng, n=4, k=2)
    wu_q = [[[1, 0, 0, 0], [1, 0, 0, 0]] for _ in range(40)]
    wv_q = [[[1, 1], [1, 1]] for _ in range(40)]
    for fn in (decrypt_it, decrypt_it_constant_time):
        for _ in range(50):
            fn(w_kd["private_key"], wu_q, wv_q, w_kd["q"])

    out = {}
    for fn_label, fn in [("decrypt_it_leaky", decrypt_it), ("decrypt_it_constant_time", decrypt_it_constant_time)]:
        order = np.random.RandomState(SEED).permutation(n_keys)
        timings = np.zeros(n_keys, dtype=float)
        for idx in order:
            s, u_q, v_q, q, _ = trials[idx]
            t0 = time.perf_counter_ns()
            _ = fn(s, u_q, v_q, q)
            timings[idx] = (time.perf_counter_ns() - t0) / 1000.0

        w_rows = np.array([t[4] for t in trials], dtype=int)
        q_arr = np.array([t[3] for t in trials], dtype=float)
        y_nz = (w_rows > 0).astype(int)
        res = eval_binary_target(timings[:, None], q_arr, y_nz, timings, n_perm=100, detailed=True)
        rho_w, p_w = sp_stats.spearmanr(timings, w_rows)
        by_w = {str(w): float(np.median(timings[w_rows == w])) for w in (0, 1, 2)}
        out[fn_label] = {
            **res,
            "spearman_rho_vs_row_weight": float(rho_w),
            "spearman_p_vs_row_weight": float(p_w),
            "median_us_by_row_weight": by_w,
        }
        print(f"  CC-Probe [{fn_label}] row_0 nonzero acc={res['best_accuracy']*100:.2f}% (base={res['majority_baseline']*100:.2f}%, bal={res['best_balanced_accuracy']*100:.2f}%, p={res['perm_pvalue']:.3f}, rho_w={rho_w:+.4f}, med_us={by_w})")
    return out


def main():
    t0 = time.perf_counter()
    os.makedirs("results", exist_ok=True)
    H_full = 8.0 * math.log2(3.0)

    suites_cfg = [
        ("linux_i9_leaky", "results/decrypt_traces_5000_linux_i9_recollect.json", "qrng_combined.txt", ["dec_timing_us", "dec_cpu_user_us"], "dec_timing_us", True),
        ("mac_m2_leaky", "results/decrypt_traces_5000_mac_arm64.json", "qrng_combined.txt", ["dec_timing_us", "dec_cpu_user_us"], "dec_timing_us", False),
        ("ibmq_mac_m2_leaky", "results/decrypt_traces_5000_ibmq_mac_arm64.json", "ibmq_qrng_uint16.txt", ["dec_timing_us", "dec_cpu_user_us"], "dec_timing_us", True),
        ("linux_i9_ct", "results/decrypt_ct_traces_5000.json", "qrng_combined.txt", ["dec_timing_us", "dec_cpu_user_us"], "dec_timing_us", True),
        ("tier1b_receiver", "results/sender_receiver_traces_5000.json", "qrng_combined.txt", ["dec_timing_us", "dec_cpu_user_us"], "dec_timing_us", False),
        ("tier1c_public_channel", "results/sender_receiver_traces_5000.json", "qrng_combined.txt", [
            "num_ct_blocks", "u_frobenius_mean", "u_frobenius_std",
            "u_mean", "u_std", "u_min", "u_max", "u_median", "u_skewness", "u_kurtosis",
            "v_norm_mean", "v_norm_std", "v_mean", "v_std", "v_min", "v_max", "v_median", "v_skewness", "v_kurtosis",
            "ct_total_elements", "ct_mean", "ct_std", "ct_entropy_approx",
            "u_mean_over_q", "v_mean_over_q", "ct_coefficient_of_variation",
        ], "ct_entropy_approx", True),
    ]

    suites_out = {}
    for name, tpath, qpath, feats, pfeat, full8 in suites_cfg:
        if not os.path.exists(tpath):
            print(f"  [SKIP] {tpath} not found")
            continue
        suites_out[name] = evaluate_suite(name, tpath, qpath, feats, pfeat, full_8_coeffs=full8)

    cc_probe = run_chosen_ciphertext_probe(n_keys=300, row_idx=0)

    out = {
        "environment": get_environment_metadata(),
        "theoretical_toy_entropy_bits": float(H_full),
        "theoretical_toy_key_count": int(3 ** 8),
        "zero_column_reduced_key_count": int(3 ** 4),
        "total_wall_seconds": float(time.perf_counter() - t0),
        "suites": suites_out,
        "chosen_ciphertext_probe": cc_probe,
    }
    out_path = "results/coeff_column_leak_5000.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path} in {out['total_wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
