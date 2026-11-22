"""
ATTACK C-5 (5,000-Trial Rigorous Replication & Extension):
Decryption (decrypt_it) Side-Channel Leakage Evaluation
=============================================================================
Collects 5,000 decrypt_it() execution traces (fixed plaintext, fresh key per
trial) with randomized execution order and interleaved warm-up phases to
eliminate time-drift confounds.

Reports:
  1. Spearman rho and Pearson r with secret_weight for dec_timing_us and
     dec_cpu_user_us, with paired bootstrap 95% CIs (2,000 resamples).
  2. Time-drift and modulus-q confound diagnostics.
  3. 5-model binary classification (above- vs <=-median secret_weight) using
     5x10 Repeated Stratified K-Fold CV (seed=42), majority-class baseline,
     shuffled-label control, leak-free in-fold q-residualized features, and
     permutation test p-values (200 permutations).
  4. 5-model regression predicting exact secret_weight (0..8) with R^2, RMSE,
     MAE under 5x10 Repeated K-Fold CV, shuffled-label control, in-fold
     q-residualized features, and permutation test p-values.
  5. Transparent comparison between uninstrumented timing (primary) and
     tracemalloc-wrapped instrumentation (legacy repo method) on the same
     5,000 (key, ciphertext) pairs.

Saves:
  - results/decrypt_traces_5000.json
  - results/decrypt_leak_5000.json
  - results/decrypt_leak_5000_summary.md
"""

import gc
import json
import math
import os
import time
import tracemalloc
import warnings

import numpy as np
import psutil
from scipy import stats as sp_stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    permutation_test_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from qeqske_core import QEQSKERandom, decrypt_it, encrypt_it, key_gen, secret_weight
from qrng_handler import MersenneSource, QRNGSource

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ---------------------------------------------------------------------------
# Experiment Constants
# ---------------------------------------------------------------------------

N_TRIALS = 5000
N_PARAM = 4
K_PARAM = 2
FIXED_PLAINTEXT = "HELLO"
SEED = 42
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 200

CLEAN_BLOCK_SIZE = 250
CLEAN_INITIAL_WARMUP = 200
CLEAN_INTERLEAVED_WARMUP = 25

TM_BLOCK_SIZE = 500
TM_INITIAL_WARMUP = 50
TM_INTERLEAVED_WARMUP = 10

TRACES_PATH = "results/decrypt_traces_5000.json"
RESULTS_PATH = "results/decrypt_leak_5000.json"
SUMMARY_PATH = "results/decrypt_leak_5000_summary.md"


# ---------------------------------------------------------------------------
# JSON sanitization & validation
# ---------------------------------------------------------------------------

def to_native(obj):
    """Recursively convert NumPy scalars/arrays to finite native Python types."""
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite float encountered: {val}")
        return val
    return obj


# ---------------------------------------------------------------------------
# In-fold q-residualization transformer (leak-free)
# ---------------------------------------------------------------------------

class QResidualizer(BaseEstimator, TransformerMixin):
    """Regress out modulus q (last column) from all feature columns inside CV."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        feats = X[:, :-1]
        q = X[:, -1]
        q_mean = float(np.mean(q))
        q_centered = q - q_mean
        denom = float(np.dot(q_centered, q_centered))
        feat_means = np.mean(feats, axis=0)
        if denom > 0.0:
            slopes = (q_centered @ (feats - feat_means)) / denom
        else:
            slopes = np.zeros(feats.shape[1], dtype=float)
        self.slopes_ = slopes
        self.intercepts_ = feat_means - slopes * q_mean
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        feats = X[:, :-1]
        q = X[:, -1]
        return feats - (self.intercepts_[None, :] + q[:, None] * self.slopes_[None, :])


# ---------------------------------------------------------------------------
# Trace Generation & Collection
# ---------------------------------------------------------------------------

def pregenerate_trials(qrng_path="qrng_combined.txt"):
    """Pre-generate 5,000 fresh (key, ciphertext) pairs and 200 warm-up pairs."""
    src = QRNGSource.from_file(qrng_path, cycle=True)
    rng = QEQSKERandom(src)

    trials = []
    unique_keys = set()
    for trial_id in range(N_TRIALS):
        keys = key_gen(rng, n=N_PARAM, k=K_PARAM)
        s = keys["private_key"]
        A = keys["public_key_A"]
        t = keys["public_key_t"]
        q = keys["q"]
        ct = encrypt_it(rng, FIXED_PLAINTEXT, A, t, q, N_PARAM, K_PARAM)
        wt = secret_weight(s)
        unique_keys.add((q, tuple(v for row in s for v in row)))
        trials.append({
            "trial_id": trial_id,
            "s": s,
            "u": ct["u"],
            "v_queue": ct["v_queue"],
            "q": q,
            "secret_weight": wt,
        })

    qrng_meta = {
        "source_file": qrng_path,
        "cycle_enabled": True,
        "dataset_size": len(src._original),
        "total_consumed": src.total_consumed,
        "wraps": src.wraps,
        "unique_q_secret_pairs": len(unique_keys),
    }

    # Separate Mersenne pool for warm-up so warm-up never touches QRNG stream
    w_src = MersenneSource(seed=42000)
    w_rng = QEQSKERandom(w_src)
    warmup_pool = []
    for _ in range(CLEAN_INITIAL_WARMUP):
        w_keys = key_gen(w_rng, n=N_PARAM, k=K_PARAM)
        w_ct = encrypt_it(
            w_rng,
            FIXED_PLAINTEXT,
            w_keys["public_key_A"],
            w_keys["public_key_t"],
            w_keys["q"],
            N_PARAM,
            K_PARAM,
        )
        warmup_pool.append((w_keys["private_key"], w_ct["u"], w_ct["v_queue"], w_keys["q"]))

    return trials, warmup_pool, qrng_meta


def _run_clean_warmup(warmup_pool, count, start_idx=0, decrypt_fn=decrypt_it):
    n_pool = len(warmup_pool)
    for i in range(count):
        s, u, v_queue, q = warmup_pool[(start_idx + i) % n_pool]
        _ = decrypt_fn(s, u, v_queue, q)


def _run_tm_warmup(warmup_pool, count, start_idx=0, decrypt_fn=decrypt_it):
    n_pool = len(warmup_pool)
    for i in range(count):
        s, u, v_queue, q = warmup_pool[(start_idx + i) % n_pool]
        tracemalloc.start()
        _ = decrypt_fn(s, u, v_queue, q)
        _ = tracemalloc.get_traced_memory()
        _ = tracemalloc.take_snapshot()
        tracemalloc.stop()


def collect_traces(trials, warmup_pool, decrypt_fn=decrypt_it):
    """Run randomized, warm-up-interleaved clean & tracemalloc-wrapped passes."""
    proc = psutil.Process(os.getpid())
    records = [
        {
            "trial_id": tr["trial_id"],
            "q": int(tr["q"]),
            "secret_weight": int(tr["secret_weight"]),
        }
        for tr in trials
    ]

    # --- Pass 1: Clean uninstrumented timing & CPU user-time ---
    gc.collect()
    _run_clean_warmup(warmup_pool, CLEAN_INITIAL_WARMUP, start_idx=0, decrypt_fn=decrypt_fn)
    exec_order_clean = np.random.RandomState(SEED).permutation(N_TRIALS)

    t_pass1_0 = time.perf_counter()
    for pos, idx in enumerate(exec_order_clean):
        if pos % CLEAN_BLOCK_SIZE == 0:
            gc.collect()
            _run_clean_warmup(
                warmup_pool, CLEAN_INTERLEAVED_WARMUP, start_idx=pos, decrypt_fn=decrypt_fn
            )

        tr = trials[int(idx)]
        s, u, v_queue, q = tr["s"], tr["u"], tr["v_queue"], tr["q"]

        cpu_before = proc.cpu_times()
        t0 = time.perf_counter()
        dec_out = decrypt_fn(s, u, v_queue, q)
        elapsed = time.perf_counter() - t0
        cpu_after = proc.cpu_times()

        rec = records[int(idx)]
        rec["exec_order_pos"] = int(pos)
        rec["dec_timing_us"] = float(elapsed * 1e6)
        rec["dec_cpu_user_us"] = float((cpu_after.user - cpu_before.user) * 1e6)
        rec["decryption_correct"] = bool(dec_out == FIXED_PLAINTEXT)
    pass1_wall_s = time.perf_counter() - t_pass1_0

    # --- Pass 2: Legacy tracemalloc-wrapped instrumentation ---
    gc.collect()
    _run_tm_warmup(warmup_pool, TM_INITIAL_WARMUP, start_idx=0, decrypt_fn=decrypt_fn)
    exec_order_tm = np.random.RandomState(SEED + 100).permutation(N_TRIALS)

    t_pass2_0 = time.perf_counter()
    for pos, idx in enumerate(exec_order_tm):
        if pos % TM_BLOCK_SIZE == 0:
            gc.collect()
            _run_tm_warmup(
                warmup_pool, TM_INTERLEAVED_WARMUP, start_idx=pos, decrypt_fn=decrypt_fn
            )

        tr = trials[int(idx)]
        s, u, v_queue, q = tr["s"], tr["u"], tr["v_queue"], tr["q"]

        tracemalloc.start()
        cpu_before = proc.cpu_times()
        t0 = time.perf_counter()
        dec_out = decrypt_fn(s, u, v_queue, q)
        elapsed = time.perf_counter() - t0
        cpu_after = proc.cpu_times()
        _, peak = tracemalloc.get_traced_memory()
        snap = tracemalloc.take_snapshot()
        n_allocs = sum(stat.count for stat in snap.statistics("lineno"))
        tracemalloc.stop()

        rec = records[int(idx)]
        rec["tm_exec_order_pos"] = int(pos)
        rec["tm_dec_timing_us"] = float(elapsed * 1e6)
        rec["tm_dec_cpu_user_us"] = float((cpu_after.user - cpu_before.user) * 1e6)
        rec["dec_ram_peak_bytes"] = int(peak)
        rec["dec_num_allocations"] = int(n_allocs)
        rec["tm_decryption_correct"] = bool(dec_out == FIXED_PLAINTEXT)
    pass2_wall_s = time.perf_counter() - t_pass2_0

    return records, {
        "clean_pass_wall_seconds": pass1_wall_s,
        "tracemalloc_pass_wall_seconds": pass2_wall_s,
    }


# ---------------------------------------------------------------------------
# Statistical Summaries, Correlations & Bootstrap CIs
# ---------------------------------------------------------------------------

def describe_array(arr):
    arr = np.asarray(arr, dtype=float)
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _safe_pearson(x, y):
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return 0.0, 1.0
    r, p = sp_stats.pearsonr(x, y)
    return float(r), float(p)


def _safe_spearman(x, y):
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return 0.0, 1.0
    rho, p = sp_stats.spearmanr(x, y)
    return float(rho), float(p)


def paired_correlation_with_bootstrap(x, y, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Compute Pearson r and Spearman rho with paired percentile bootstrap 95% CIs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return {
            "pearson_r": 0.0,
            "pearson_p": 1.0,
            "pearson_boot_ci95": [0.0, 0.0],
            "pearson_boot_se": 0.0,
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
            "spearman_boot_ci95": [0.0, 0.0],
            "spearman_boot_se": 0.0,
            "n_bootstrap": int(n_bootstrap),
        }

    pearson_r, pearson_p = _safe_pearson(x, y)
    spearman_rho, spearman_p = _safe_spearman(x, y)

    rng = np.random.default_rng(seed)
    boot_p = np.empty(n_bootstrap, dtype=float)
    boot_s = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        if np.ptp(xb) == 0.0 or np.ptp(yb) == 0.0:
            boot_p[b] = 0.0
            boot_s[b] = 0.0
        else:
            boot_p[b] = sp_stats.pearsonr(xb, yb).statistic
            boot_s[b] = sp_stats.spearmanr(xb, yb).statistic

    p_lo, p_hi = np.percentile(boot_p, [2.5, 97.5])
    s_lo, s_hi = np.percentile(boot_s, [2.5, 97.5])

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "pearson_boot_ci95": [float(p_lo), float(p_hi)],
        "pearson_boot_se": float(np.std(boot_p, ddof=1)),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "spearman_boot_ci95": [float(s_lo), float(s_hi)],
        "spearman_boot_se": float(np.std(boot_s, ddof=1)),
        "n_bootstrap": int(n_bootstrap),
    }


def summarize_by_secret_weight(weights, timing_us, cpu_us):
    out = {}
    for w in sorted(set(int(v) for v in weights)):
        mask = weights == w
        t_sub = timing_us[mask]
        c_sub = cpu_us[mask]
        out[str(w)] = {
            "count": int(np.sum(mask)),
            "timing_mean_us": float(np.mean(t_sub)),
            "timing_median_us": float(np.median(t_sub)),
            "timing_std_us": float(np.std(t_sub, ddof=1)) if len(t_sub) > 1 else 0.0,
            "cpu_mean_us": float(np.mean(c_sub)),
            "cpu_median_us": float(np.median(c_sub)),
            "cpu_std_us": float(np.std(c_sub, ddof=1)) if len(c_sub) > 1 else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# CV & Permutation Helpers
# ---------------------------------------------------------------------------

def _fold_stats(arr, clamp_unit=False):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    t_crit = float(sp_stats.t.ppf(0.975, df=n - 1))
    margin = t_crit * sd / math.sqrt(n)
    lo = mean - margin
    hi = mean + margin
    if clamp_unit:
        lo = max(0.0, lo)
        hi = min(1.0, hi)
    return {
        "mean": mean,
        "sd": sd,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95": [float(lo), float(hi)],
    }


def _get_classifiers():
    return {
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100, random_state=SEED, n_jobs=1
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=SEED
        ),
        "MLP": lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=1000, random_state=SEED
        ),
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000, random_state=SEED
        ),
        "SVM (RBF)": lambda: SVC(kernel="rbf", random_state=SEED),
    }


def _get_regressors():
    return {
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=100, random_state=SEED, n_jobs=1
        ),
        "Gradient Boosting": lambda: GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=SEED
        ),
        "MLP": lambda: MLPRegressor(
            hidden_layer_sizes=(32, 16), max_iter=1000, random_state=SEED
        ),
        "Linear Regression": lambda: LinearRegression(),
        "SVR (RBF)": lambda: SVR(kernel="rbf"),
    }


def run_classification_suite(
    X,
    q_vals,
    y_cls,
    y_cls_shuf,
    majority_baseline,
    perm_all=True,
):
    """Run 5 classifiers with 5x10 CV (real, shuffled, q-resid) + permutation test."""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    perm_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    X_with_q = np.column_stack([X, q_vals])
    scoring = ("accuracy", "balanced_accuracy", "roc_auc", "f1")

    models_out = {}
    pipelines_real = {}
    for name, clf_fn in _get_classifiers().items():
        pipe_real = make_pipeline(StandardScaler(), clf_fn())
        pipe_shuf = make_pipeline(StandardScaler(), clf_fn())
        pipe_q = make_pipeline(QResidualizer(), StandardScaler(), clf_fn())
        pipelines_real[name] = pipe_real

        cv_real = cross_validate(pipe_real, X, y_cls, cv=rskf, scoring=scoring, n_jobs=-1)
        cv_shuf = cross_validate(pipe_shuf, X, y_cls_shuf, cv=rskf, scoring=scoring, n_jobs=-1)
        cv_q = cross_validate(pipe_q, X_with_q, y_cls, cv=rskf, scoring=scoring, n_jobs=-1)

        def _pack(cv_dict):
            return {
                met: _fold_stats(cv_dict[f"test_{met}"], clamp_unit=True)
                for met in scoring
            }

        real_res = _pack(cv_real)
        shuf_res = _pack(cv_shuf)
        q_res = _pack(cv_q)

        models_out[name] = {
            "real": real_res,
            "shuffled_control": shuf_res,
            "q_residualized": q_res,
            "delta_vs_baseline": float(real_res["accuracy"]["mean"] - majority_baseline),
            "gap_real_minus_shuffled": float(
                real_res["accuracy"]["mean"] - shuf_res["accuracy"]["mean"]
            ),
        }

    # Permutation tests (all 5 models for primary clean pass; best model for comparison)
    if perm_all:
        perm_targets = list(models_out.keys())
    else:
        best_name = max(
            models_out, key=lambda k: models_out[k]["real"]["accuracy"]["mean"]
        )
        perm_targets = [best_name]

    for name in perm_targets:
        score, perm_scores, pval = permutation_test_score(
            pipelines_real[name],
            X,
            y_cls,
            scoring="accuracy",
            cv=perm_cv,
            n_permutations=N_PERMUTATIONS,
            n_jobs=-1,
            random_state=SEED,
        )
        models_out[name]["permutation_test"] = {
            "scoring": "accuracy",
            "cv_folds": 3,
            "n_permutations": N_PERMUTATIONS,
            "score": float(score),
            "perm_mean": float(np.mean(perm_scores)),
            "perm_std": float(np.std(perm_scores, ddof=1)),
            "pvalue": float(pval),
        }

    return models_out


def run_regression_suite(
    X,
    q_vals,
    y_reg,
    y_reg_shuf,
    perm_all=True,
):
    """Run 5 regressors for exact secret_weight with 5x10 CV + permutation test."""
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    perm_cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
    X_with_q = np.column_stack([X, q_vals])
    scoring = ("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error")

    models_out = {}
    pipelines_real = {}
    for name, reg_fn in _get_regressors().items():
        pipe_real = make_pipeline(StandardScaler(), reg_fn())
        pipe_shuf = make_pipeline(StandardScaler(), reg_fn())
        pipe_q = make_pipeline(QResidualizer(), StandardScaler(), reg_fn())
        pipelines_real[name] = pipe_real

        cv_real = cross_validate(pipe_real, X, y_reg, cv=rkf, scoring=scoring, n_jobs=-1)
        cv_shuf = cross_validate(pipe_shuf, X, y_reg_shuf, cv=rkf, scoring=scoring, n_jobs=-1)
        cv_q = cross_validate(pipe_q, X_with_q, y_reg, cv=rkf, scoring=scoring, n_jobs=-1)

        def _pack(cv_dict):
            return {
                "r2": _fold_stats(cv_dict["test_r2"], clamp_unit=False),
                "rmse": _fold_stats(-cv_dict["test_neg_root_mean_squared_error"], clamp_unit=False),
                "mae": _fold_stats(-cv_dict["test_neg_mean_absolute_error"], clamp_unit=False),
            }

        real_res = _pack(cv_real)
        shuf_res = _pack(cv_shuf)
        q_res = _pack(cv_q)

        models_out[name] = {
            "real": real_res,
            "shuffled_control": shuf_res,
            "q_residualized": q_res,
            "gap_real_minus_shuffled_r2": float(
                real_res["r2"]["mean"] - shuf_res["r2"]["mean"]
            ),
        }

    if perm_all:
        perm_targets = list(models_out.keys())
    else:
        best_name = max(models_out, key=lambda k: models_out[k]["real"]["r2"]["mean"])
        perm_targets = [best_name]

    for name in perm_targets:
        score, perm_scores, pval = permutation_test_score(
            pipelines_real[name],
            X,
            y_reg,
            scoring="r2",
            cv=perm_cv,
            n_permutations=N_PERMUTATIONS,
            n_jobs=-1,
            random_state=SEED,
        )
        models_out[name]["permutation_test"] = {
            "scoring": "r2",
            "cv_folds": 3,
            "n_permutations": N_PERMUTATIONS,
            "score": float(score),
            "perm_mean": float(np.mean(perm_scores)),
            "perm_std": float(np.std(perm_scores, ddof=1)),
            "pvalue": float(pval),
        }

    return models_out


# ---------------------------------------------------------------------------
# Summary Markdown Generator (reads exclusively from saved JSON)
# ---------------------------------------------------------------------------

def generate_summary_markdown(json_path, md_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    meta = data["experiment"]
    qrng = data["qrng_stream"]
    target = data["targets"]
    clean = data["clean_uninstrumented"]
    tm2 = data["tracemalloc_instrumented_2feat"]
    tm4 = data["tracemalloc_instrumented_4feat"]

    lines = [
        f"# `{meta.get('decrypt_label', 'decrypt_it()')}` Side-Channel Leakage Evaluation (5,000 Trials)",
        "",
        "## 1. Experiment Setup & Data Integrity",
        f"- **Trials:** {meta['n_trials']} (fixed plaintext `{meta['fixed_plaintext']}`, parameters `n={meta['n']}`, `k={meta['k']}`, fresh key per trial, seed `{meta['seed']}`)",
        f"- **Decryption correctness:** Clean pass `{clean['decryption_correct_count']}/{meta['n_trials']}` ({clean['decryption_correct_rate']*100:.2f}%); `tracemalloc` pass `{tm2['decryption_correct_count']}/{meta['n_trials']}` ({tm2['decryption_correct_rate']*100:.2f}%)",
        f"- **QRNG stream:** `{qrng['source_file']}` (`{qrng['dataset_size']:,}` numbers, `cycle={qrng['cycle_enabled']}`, consumed `{qrng['total_consumed']:,}` draws across `{qrng['wraps']}` wraps; `{qrng['unique_q_secret_pairs']:,}` unique `(q, s)` pairs)",
        f"- **Binary target:** `secret_weight > {target['median_secret_weight']}` -> class 0: `{target['class_counts']['0']}`, class 1: `{target['class_counts']['1']}`, majority-class baseline: **{target['majority_baseline']*100:.2f}%**",
        f"- **Regression target:** exact `secret_weight` (range `{target['secret_weight_min']}..{target['secret_weight_max']}`, mean `{target['secret_weight_mean']:.4f}`, std `{target['secret_weight_std']:.4f}`)",
        "",
        "## 2. Primary Analysis: Clean Uninstrumented Traces (`dec_timing_us`, `dec_cpu_user_us`)",
        "",
        "### Correlations vs. `secret_weight` (2,000-Resample Paired Bootstrap 95% CIs)",
        "| Signal | Mean ± Std (µs) | Median (IQR) µs | Pearson r (95% Boot CI) | Pearson p | Spearman ρ (95% Boot CI) | Spearman p |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for feat in ["dec_timing_us", "dec_cpu_user_us"]:
        desc = clean["descriptives"][feat]
        c = clean["correlations"][feat]
        lines.append(
            f"| `{feat}` | {desc['mean']:.2f} ± {desc['std']:.2f} | "
            f"{desc['median']:.2f} ({desc['iqr']:.2f}) | "
            f"{c['pearson_r']:+.4f} [{c['pearson_boot_ci95'][0]:+.4f}, {c['pearson_boot_ci95'][1]:+.4f}] | "
            f"{c['pearson_p']:.2e} | "
            f"{c['spearman_rho']:+.4f} [{c['spearman_boot_ci95'][0]:+.4f}, {c['spearman_boot_ci95'][1]:+.4f}] | "
            f"{c['spearman_p']:.2e} |"
        )

    q_chk = clean["confound_q"]
    d_chk = clean["drift_check"]
    lines.extend([
        "",
        "### Confound & Time-Drift Checks (Clean Pass)",
        f"- **Modulus `q` vs `secret_weight`:** Pearson `r = {q_chk['q_vs_secret_weight']['pearson_r']:+.4f}` (`p = {q_chk['q_vs_secret_weight']['pearson_p']:.4f}`), Spearman `ρ = {q_chk['q_vs_secret_weight']['spearman_rho']:+.4f}` (`p = {q_chk['q_vs_secret_weight']['spearman_p']:.4f}`)",
        f"- **Modulus `q` vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = {q_chk['q_vs_dec_timing_us']['pearson_r']:+.4f}` (`p = {q_chk['q_vs_dec_timing_us']['pearson_p']:.4f}`) / `r = {q_chk['q_vs_dec_cpu_user_us']['pearson_r']:+.4f}` (`p = {q_chk['q_vs_dec_cpu_user_us']['pearson_p']:.4f}`)",
        f"- **Execution order vs `secret_weight`:** Pearson `r = {d_chk['exec_order_vs_secret_weight']['pearson_r']:+.4f}` (`p = {d_chk['exec_order_vs_secret_weight']['pearson_p']:.4f}`)",
        f"- **Execution order vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = {d_chk['exec_order_vs_dec_timing_us']['pearson_r']:+.4f}` (`p = {d_chk['exec_order_vs_dec_timing_us']['pearson_p']:.4f}`) / `r = {d_chk['exec_order_vs_dec_cpu_user_us']['pearson_r']:+.4f}` (`p = {d_chk['exec_order_vs_dec_cpu_user_us']['pearson_p']:.4f}`)",
        "",
        f"### Binary Classification (5×10 Repeated Stratified CV, Majority Baseline = {target['majority_baseline']*100:.2f}%)",
        "| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized Acc | Perm p-value |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m_name, m_res in clean["classification"].items():
        r_acc = m_res["real"]["accuracy"]
        r_auc = m_res["real"]["roc_auc"]
        s_acc = m_res["shuffled_control"]["accuracy"]
        q_acc = m_res["q_residualized"]["accuracy"]
        p_val = m_res["permutation_test"]["pvalue"]
        lines.append(
            f"| {m_name} | {r_acc['mean']*100:.2f}% ± {r_acc['sd']*100:.2f}% | "
            f"{r_auc['mean']:.4f} | "
            f"{s_acc['mean']*100:.2f}% ± {s_acc['sd']*100:.2f}% | "
            f"{m_res['gap_real_minus_shuffled']*100:+.2f}% | "
            f"{q_acc['mean']*100:.2f}% ± {q_acc['sd']*100:.2f}% | "
            f"{p_val:.4f} |"
        )

    lines.extend([
        "",
        "### Regression Target: Exact `secret_weight` (5×10 Repeated K-Fold CV)",
        "| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | q-Residualized R² | Perm p-value |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m_name, m_res in clean["regression"].items():
        r_r2 = m_res["real"]["r2"]
        r_rmse = m_res["real"]["rmse"]
        s_r2 = m_res["shuffled_control"]["r2"]
        q_r2 = m_res["q_residualized"]["r2"]
        p_val = m_res["permutation_test"]["pvalue"]
        lines.append(
            f"| {m_name} | {r_r2['mean']:+.4f} ± {r_r2['sd']:.4f} | "
            f"{r_rmse['mean']:.4f} | "
            f"{s_r2['mean']:+.4f} ± {s_r2['sd']:.4f} | "
            f"{q_r2['mean']:+.4f} ± {q_r2['sd']:.4f} | "
            f"{p_val:.4f} |"
        )

    lines.extend([
        "",
        "## 3. Legacy Instrumentation Comparison (`tracemalloc`-Wrapped Pass)",
        "",
        "### Correlations vs. `secret_weight` under `tracemalloc`",
        "| Signal | Mean ± Std | Pearson r (95% Boot CI) | Spearman ρ (95% Boot CI) |",
        "| :--- | :---: | :---: | :---: |",
    ])

    for feat in ["tm_dec_timing_us", "tm_dec_cpu_user_us", "dec_ram_peak_bytes", "dec_num_allocations"]:
        desc = tm4["descriptives"][feat]
        c = tm4["correlations"][feat]
        lines.append(
            f"| `{feat}` | {desc['mean']:.2f} ± {desc['std']:.2f} | "
            f"{c['pearson_r']:+.4f} [{c['pearson_boot_ci95'][0]:+.4f}, {c['pearson_boot_ci95'][1]:+.4f}] | "
            f"{c['spearman_rho']:+.4f} [{c['spearman_boot_ci95'][0]:+.4f}, {c['spearman_boot_ci95'][1]:+.4f}] |"
        )

    best_tm2_cls = max(tm2["classification"], key=lambda k: tm2["classification"][k]["real"]["accuracy"]["mean"])
    best_tm4_cls = max(tm4["classification"], key=lambda k: tm4["classification"][k]["real"]["accuracy"]["mean"])
    best_tm2_reg = max(tm2["regression"], key=lambda k: tm2["regression"][k]["real"]["r2"]["mean"])
    best_tm4_reg = max(tm4["regression"], key=lambda k: tm4["regression"][k]["real"]["r2"]["mean"])

    lines.extend([
        "",
        "### ML Summary under `tracemalloc`",
        f"- **2-feature (`tm_dec_timing_us`, `tm_dec_cpu_user_us`) Best Classifier ({best_tm2_cls}):** "
        f"Real Acc `{tm2['classification'][best_tm2_cls]['real']['accuracy']['mean']*100:.2f}%`, "
        f"Shuffled `{tm2['classification'][best_tm2_cls]['shuffled_control']['accuracy']['mean']*100:.2f}%`, "
        f"q-Residualized `{tm2['classification'][best_tm2_cls]['q_residualized']['accuracy']['mean']*100:.2f}%`, "
        f"Perm `p = {tm2['classification'][best_tm2_cls]['permutation_test']['pvalue']:.4f}`",
        f"- **2-feature Best Regressor ({best_tm2_reg}):** "
        f"Real R² `{tm2['regression'][best_tm2_reg]['real']['r2']['mean']:+.4f}`, "
        f"Shuffled R² `{tm2['regression'][best_tm2_reg]['shuffled_control']['r2']['mean']:+.4f}`, "
        f"q-Residualized R² `{tm2['regression'][best_tm2_reg]['q_residualized']['r2']['mean']:+.4f}`, "
        f"Perm `p = {tm2['regression'][best_tm2_reg]['permutation_test']['pvalue']:.4f}`",
        f"- **4-feature (+ `dec_ram_peak_bytes`, `dec_num_allocations`) Best Classifier ({best_tm4_cls}):** "
        f"Real Acc `{tm4['classification'][best_tm4_cls]['real']['accuracy']['mean']*100:.2f}%`, "
        f"Shuffled `{tm4['classification'][best_tm4_cls]['shuffled_control']['accuracy']['mean']*100:.2f}%`, "
        f"q-Residualized `{tm4['classification'][best_tm4_cls]['q_residualized']['accuracy']['mean']*100:.2f}%`, "
        f"Perm `p = {tm4['classification'][best_tm4_cls]['permutation_test']['pvalue']:.4f}`",
        f"- **4-feature Best Regressor ({best_tm4_reg}):** "
        f"Real R² `{tm4['regression'][best_tm4_reg]['real']['r2']['mean']:+.4f}`, "
        f"Shuffled R² `{tm4['regression'][best_tm4_reg]['shuffled_control']['r2']['mean']:+.4f}`, "
        f"q-Residualized R² `{tm4['regression'][best_tm4_reg]['q_residualized']['r2']['mean']:+.4f}`, "
        f"Perm `p = {tm4['regression'][best_tm4_reg]['permutation_test']['pvalue']:.4f}`",
    ])

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Analysis Orchestration & Reporting
# ---------------------------------------------------------------------------

def analyze_pass(
    records,
    feature_keys,
    timing_key,
    cpu_key,
    order_key,
    correct_key,
    y_cls,
    y_cls_shuf,
    y_reg,
    y_reg_shuf,
    majority_baseline,
    perm_all=True,
):
    weights = np.array([r["secret_weight"] for r in records], dtype=float)
    q_vals = np.array([r["q"] for r in records], dtype=float)
    order_vals = np.array([r[order_key] for r in records], dtype=float)
    timing_vals = np.array([r[timing_key] for r in records], dtype=float)
    cpu_vals = np.array([r[cpu_key] for r in records], dtype=float)

    descriptives = {k: describe_array([r[k] for r in records]) for k in feature_keys}
    correlations = {
        k: paired_correlation_with_bootstrap([r[k] for r in records], weights)
        for k in feature_keys
    }

    p_qw, pp_qw = _safe_pearson(q_vals, weights)
    s_qw, sp_qw = _safe_spearman(q_vals, weights)
    p_qt, pp_qt = _safe_pearson(q_vals, timing_vals)
    s_qt, sp_qt = _safe_spearman(q_vals, timing_vals)
    p_qc, pp_qc = _safe_pearson(q_vals, cpu_vals)
    s_qc, sp_qc = _safe_spearman(q_vals, cpu_vals)

    confound_q = {
        "q_vs_secret_weight": {
            "pearson_r": p_qw, "pearson_p": pp_qw,
            "spearman_rho": s_qw, "spearman_p": sp_qw,
        },
        f"q_vs_{timing_key}": {
            "pearson_r": p_qt, "pearson_p": pp_qt,
            "spearman_rho": s_qt, "spearman_p": sp_qt,
        },
        f"q_vs_{cpu_key}": {
            "pearson_r": p_qc, "pearson_p": pp_qc,
            "spearman_rho": s_qc, "spearman_p": sp_qc,
        },
    }

    p_ow, pp_ow = _safe_pearson(order_vals, weights)
    s_ow, sp_ow = _safe_spearman(order_vals, weights)
    p_ot, pp_ot = _safe_pearson(order_vals, timing_vals)
    s_ot, sp_ot = _safe_spearman(order_vals, timing_vals)
    p_oc, pp_oc = _safe_pearson(order_vals, cpu_vals)
    s_oc, sp_oc = _safe_spearman(order_vals, cpu_vals)

    drift_check = {
        "exec_order_vs_secret_weight": {
            "pearson_r": p_ow, "pearson_p": pp_ow,
            "spearman_rho": s_ow, "spearman_p": sp_ow,
        },
        f"exec_order_vs_{timing_key}": {
            "pearson_r": p_ot, "pearson_p": pp_ot,
            "spearman_rho": s_ot, "spearman_p": sp_ot,
        },
        f"exec_order_vs_{cpu_key}": {
            "pearson_r": p_oc, "pearson_p": pp_oc,
            "spearman_rho": s_oc, "spearman_p": sp_oc,
        },
    }

    X = np.column_stack([np.array([r[k] for r in records], dtype=float) for k in feature_keys])

    classification = run_classification_suite(
        X, q_vals, y_cls, y_cls_shuf, majority_baseline, perm_all=perm_all
    )
    regression = run_regression_suite(
        X, q_vals, y_reg, y_reg_shuf, perm_all=perm_all
    )

    correct_cnt = sum(1 for r in records if r[correct_key])
    return {
        "features": list(feature_keys),
        "decryption_correct_count": int(correct_cnt),
        "decryption_correct_rate": float(correct_cnt / len(records)),
        "descriptives": descriptives,
        "correlations": correlations,
        "by_secret_weight": summarize_by_secret_weight(weights, timing_vals, cpu_vals),
        "confound_q": confound_q,
        "drift_check": drift_check,
        "classification": classification,
        "regression": regression,
    }


def print_report(results):
    exp = results["experiment"]
    qrng = results["qrng_stream"]
    tgt = results["targets"]
    clean = results["clean_uninstrumented"]
    tm2 = results["tracemalloc_instrumented_2feat"]
    tm4 = results["tracemalloc_instrumented_4feat"]

    print("=" * 88)
    print(f"SIDE-CHANNEL EVALUATION: {exp.get('decrypt_label', 'decrypt_it')} (5,000 TRIALS)")
    print("=" * 88)
    print(f"Trials: {exp['n_trials']} | Plaintext: {exp['fixed_plaintext']!r} | Params: n={exp['n']}, k={exp['k']} | Seed: {exp['seed']}")
    print(f"QRNG: {qrng['source_file']} (size={qrng['dataset_size']:,}, consumed={qrng['total_consumed']:,}, wraps={qrng['wraps']}, unique_(q,s)={qrng['unique_q_secret_pairs']:,})")
    print(f"Decryption correctness: clean={clean['decryption_correct_count']}/{exp['n_trials']} ({clean['decryption_correct_rate']*100:.2f}%), tracemalloc={tm2['decryption_correct_count']}/{exp['n_trials']} ({tm2['decryption_correct_rate']*100:.2f}%)")
    print(f"Binary target: secret_weight > {tgt['median_secret_weight']} -> class0={tgt['class_counts']['0']}, class1={tgt['class_counts']['1']}, majority baseline={tgt['majority_baseline']*100:.2f}%")
    print(f"Regression target: secret_weight in [{tgt['secret_weight_min']}, {tgt['secret_weight_max']}], mean={tgt['secret_weight_mean']:.4f}, std={tgt['secret_weight_std']:.4f}")

    print("\n--- 1. CLEAN UNINSTRUMENTED CORRELATIONS WITH secret_weight (2,000 Bootstraps) ---")
    print(f"{'Signal':<18} {'Mean±Std (us)':>18} {'Pearson r [95% Boot CI]':>30} {'p-val':>10} {'Spearman rho [95% Boot CI]':>31} {'p-val':>10}")
    for feat in ["dec_timing_us", "dec_cpu_user_us"]:
        d = clean["descriptives"][feat]
        c = clean["correlations"][feat]
        ms = f"{d['mean']:.2f}±{d['std']:.2f}"
        pr = f"{c['pearson_r']:+.4f} [{c['pearson_boot_ci95'][0]:+.4f},{c['pearson_boot_ci95'][1]:+.4f}]"
        sr = f"{c['spearman_rho']:+.4f} [{c['spearman_boot_ci95'][0]:+.4f},{c['spearman_boot_ci95'][1]:+.4f}]"
        print(f"{feat:<18} {ms:>18} {pr:>30} {c['pearson_p']:>10.2e} {sr:>31} {c['spearman_p']:>10.2e}")

    qc = clean["confound_q"]
    dc = clean["drift_check"]
    print("\n--- 2. CONFOUND (q) & TIME-DRIFT (exec_order_pos) CHECKS [CLEAN PASS] ---")
    print(f"  q vs secret_weight       : Pearson r={qc['q_vs_secret_weight']['pearson_r']:+.4f} (p={qc['q_vs_secret_weight']['pearson_p']:.4f}), Spearman rho={qc['q_vs_secret_weight']['spearman_rho']:+.4f} (p={qc['q_vs_secret_weight']['spearman_p']:.4f})")
    print(f"  q vs dec_timing_us       : Pearson r={qc['q_vs_dec_timing_us']['pearson_r']:+.4f} (p={qc['q_vs_dec_timing_us']['pearson_p']:.4f}), Spearman rho={qc['q_vs_dec_timing_us']['spearman_rho']:+.4f} (p={qc['q_vs_dec_timing_us']['spearman_p']:.4f})")
    print(f"  q vs dec_cpu_user_us     : Pearson r={qc['q_vs_dec_cpu_user_us']['pearson_r']:+.4f} (p={qc['q_vs_dec_cpu_user_us']['pearson_p']:.4f}), Spearman rho={qc['q_vs_dec_cpu_user_us']['spearman_rho']:+.4f} (p={qc['q_vs_dec_cpu_user_us']['spearman_p']:.4f})")
    print(f"  order vs secret_weight   : Pearson r={dc['exec_order_vs_secret_weight']['pearson_r']:+.4f} (p={dc['exec_order_vs_secret_weight']['pearson_p']:.4f})")
    print(f"  order vs dec_timing_us   : Pearson r={dc['exec_order_vs_dec_timing_us']['pearson_r']:+.4f} (p={dc['exec_order_vs_dec_timing_us']['pearson_p']:.4f}), Spearman rho={dc['exec_order_vs_dec_timing_us']['spearman_rho']:+.4f} (p={dc['exec_order_vs_dec_timing_us']['spearman_p']:.4f})")
    print(f"  order vs dec_cpu_user_us : Pearson r={dc['exec_order_vs_dec_cpu_user_us']['pearson_r']:+.4f} (p={dc['exec_order_vs_dec_cpu_user_us']['pearson_p']:.4f}), Spearman rho={dc['exec_order_vs_dec_cpu_user_us']['spearman_rho']:+.4f} (p={dc['exec_order_vs_dec_cpu_user_us']['spearman_p']:.4f})")

    print(f"\n--- 3. CLEAN CLASSIFICATION (5x10 Stratified CV; Baseline={tgt['majority_baseline']*100:.2f}%) ---")
    print(f"{'Model':<20} {'Real Acc':>14} {'ROC-AUC':>8} {'Shuffled':>14} {'Net Gap':>9} {'q-Resid Acc':>14} {'Perm p':>8}")
    for m_name, m in clean["classification"].items():
        ra = m["real"]["accuracy"]
        auc = m["real"]["roc_auc"]
        sa = m["shuffled_control"]["accuracy"]
        qa = m["q_residualized"]["accuracy"]
        pv = m["permutation_test"]["pvalue"]
        print(
            f"{m_name:<20} {ra['mean']*100:6.2f}±{ra['sd']*100:4.2f}% "
            f"{auc['mean']:8.4f} "
            f"{sa['mean']*100:6.2f}±{sa['sd']*100:4.2f}% "
            f"{m['gap_real_minus_shuffled']*100:+8.2f}% "
            f"{qa['mean']*100:6.2f}±{qa['sd']*100:4.2f}% "
            f"{pv:8.4f}"
        )

    print("\n--- 4. CLEAN REGRESSION TARGET: EXACT secret_weight (5x10 Repeated K-Fold CV) ---")
    print(f"{'Model':<20} {'Real R^2':>16} {'RMSE':>8} {'Shuffled R^2':>16} {'q-Resid R^2':>16} {'Perm p':>8}")
    for m_name, m in clean["regression"].items():
        rr = m["real"]["r2"]
        rm = m["real"]["rmse"]
        sr = m["shuffled_control"]["r2"]
        qr = m["q_residualized"]["r2"]
        pv = m["permutation_test"]["pvalue"]
        print(
            f"{m_name:<20} {rr['mean']:+7.4f}±{rr['sd']:6.4f} "
            f"{rm['mean']:8.4f} "
            f"{sr['mean']:+7.4f}±{sr['sd']:6.4f} "
            f"{qr['mean']:+7.4f}±{qr['sd']:6.4f} "
            f"{pv:8.4f}"
        )

    print("\n--- 5. LEGACY TRACEMALLOC-WRAPPED COMPARISON (OBSERVER EFFECT CHECK) ---")
    for feat in ["tm_dec_timing_us", "tm_dec_cpu_user_us", "dec_ram_peak_bytes", "dec_num_allocations"]:
        d = tm4["descriptives"][feat]
        c = tm4["correlations"][feat]
        print(
            f"  {feat:<22}: mean={d['mean']:8.2f}, Pearson r={c['pearson_r']:+.4f} "
            f"[{c['pearson_boot_ci95'][0]:+.4f},{c['pearson_boot_ci95'][1]:+.4f}], "
            f"Spearman rho={c['spearman_rho']:+.4f} "
            f"[{c['spearman_boot_ci95'][0]:+.4f},{c['spearman_boot_ci95'][1]:+.4f}]"
        )
    print("  [tracemalloc 2-feat (timing, cpu) Classification & Regression]")
    for m_name, m in tm2["classification"].items():
        pt = f", perm_p={m['permutation_test']['pvalue']:.4f}" if "permutation_test" in m else ""
        print(f"    CLF {m_name:<19}: real={m['real']['accuracy']['mean']*100:.2f}%, shuf={m['shuffled_control']['accuracy']['mean']*100:.2f}%, q-res={m['q_residualized']['accuracy']['mean']*100:.2f}%{pt}")
    for m_name, m in tm2["regression"].items():
        pt = f", perm_p={m['permutation_test']['pvalue']:.4f}" if "permutation_test" in m else ""
        print(f"    REG {m_name:<19}: real_R2={m['real']['r2']['mean']:+.4f}, shuf_R2={m['shuffled_control']['r2']['mean']:+.4f}, q-res_R2={m['q_residualized']['r2']['mean']:+.4f}{pt}")
    print("  [tracemalloc 4-feat (+ram, allocs) Best Models]")
    b4c = max(tm4["classification"], key=lambda k: tm4["classification"][k]["real"]["accuracy"]["mean"])
    b4r = max(tm4["regression"], key=lambda k: tm4["regression"][k]["real"]["r2"]["mean"])
    m4c = tm4["classification"][b4c]
    m4r = tm4["regression"][b4r]
    print(f"    Best CLF ({b4c}): real={m4c['real']['accuracy']['mean']*100:.2f}%, shuf={m4c['shuffled_control']['accuracy']['mean']*100:.2f}%, q-res={m4c['q_residualized']['accuracy']['mean']*100:.2f}%, perm_p={m4c['permutation_test']['pvalue']:.4f}")
    print(f"    Best REG ({b4r}): real_R2={m4r['real']['r2']['mean']:+.4f}, shuf_R2={m4r['shuffled_control']['r2']['mean']:+.4f}, q-res_R2={m4r['q_residualized']['r2']['mean']:+.4f}, perm_p={m4r['permutation_test']['pvalue']:.4f}")
    print("=" * 88)


def run_decrypt_leak_pipeline(
    decrypt_fn=decrypt_it,
    traces_path=TRACES_PATH,
    results_path=RESULTS_PATH,
    summary_path=SUMMARY_PATH,
    label="decrypt_it (original)",
    pregenerated=None,
    verbose=True,
):
    """Run the full 5,000-trial side-channel leakage evaluation for decrypt_fn."""
    os.makedirs("results", exist_ok=True)
    t_all_0 = time.perf_counter()

    if pregenerated is None:
        trials, warmup_pool, qrng_meta = pregenerate_trials()
    else:
        trials, warmup_pool, qrng_meta = pregenerated

    records, pass_timings = collect_traces(trials, warmup_pool, decrypt_fn=decrypt_fn)

    if pregenerated is None:
        del trials
        gc.collect()

    with open(traces_path, "w") as f:
        json.dump(to_native(records), f, indent=2)

    weights = np.array([r["secret_weight"] for r in records], dtype=float)
    median_w = float(np.median(weights))
    y_cls = (weights > median_w).astype(int)
    majority_baseline = float(max(np.mean(y_cls == 0), np.mean(y_cls == 1)))

    shuf_rng = np.random.RandomState(SEED)
    y_cls_shuf = shuf_rng.permutation(y_cls)
    y_reg = weights.astype(float)
    y_reg_shuf = np.random.RandomState(SEED).permutation(y_reg)

    clean_res = analyze_pass(
        records=records,
        feature_keys=["dec_timing_us", "dec_cpu_user_us"],
        timing_key="dec_timing_us",
        cpu_key="dec_cpu_user_us",
        order_key="exec_order_pos",
        correct_key="decryption_correct",
        y_cls=y_cls,
        y_cls_shuf=y_cls_shuf,
        y_reg=y_reg,
        y_reg_shuf=y_reg_shuf,
        majority_baseline=majority_baseline,
        perm_all=True,
    )

    tm2_res = analyze_pass(
        records=records,
        feature_keys=["tm_dec_timing_us", "tm_dec_cpu_user_us"],
        timing_key="tm_dec_timing_us",
        cpu_key="tm_dec_cpu_user_us",
        order_key="tm_exec_order_pos",
        correct_key="tm_decryption_correct",
        y_cls=y_cls,
        y_cls_shuf=y_cls_shuf,
        y_reg=y_reg,
        y_reg_shuf=y_reg_shuf,
        majority_baseline=majority_baseline,
        perm_all=False,
    )

    tm4_res = analyze_pass(
        records=records,
        feature_keys=[
            "tm_dec_timing_us",
            "tm_dec_cpu_user_us",
            "dec_ram_peak_bytes",
            "dec_num_allocations",
        ],
        timing_key="tm_dec_timing_us",
        cpu_key="tm_dec_cpu_user_us",
        order_key="tm_exec_order_pos",
        correct_key="tm_decryption_correct",
        y_cls=y_cls,
        y_cls_shuf=y_cls_shuf,
        y_reg=y_reg,
        y_reg_shuf=y_reg_shuf,
        majority_baseline=majority_baseline,
        perm_all=False,
    )

    total_wall_s = time.perf_counter() - t_all_0

    full_results = {
        "experiment": {
            "decrypt_label": label,
            "decrypt_function": getattr(decrypt_fn, "__name__", str(decrypt_fn)),
            "n_trials": N_TRIALS,
            "n": N_PARAM,
            "k": K_PARAM,
            "fixed_plaintext": FIXED_PLAINTEXT,
            "plaintext_bits": len(FIXED_PLAINTEXT) * 8,
            "seed": SEED,
            "cv_classification": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)",
            "cv_regression": "RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)",
            "n_bootstrap": N_BOOTSTRAP,
            "n_permutations": N_PERMUTATIONS,
            "clean_warmup_initial": CLEAN_INITIAL_WARMUP,
            "clean_warmup_interleaved": CLEAN_INTERLEAVED_WARMUP,
            "clean_block_size": CLEAN_BLOCK_SIZE,
            "tm_warmup_initial": TM_INITIAL_WARMUP,
            "tm_warmup_interleaved": TM_INTERLEAVED_WARMUP,
            "tm_block_size": TM_BLOCK_SIZE,
            **pass_timings,
            "total_wall_seconds": total_wall_s,
            "traces_file": traces_path,
        },
        "qrng_stream": qrng_meta,
        "targets": {
            "median_secret_weight": median_w,
            "class_definition": f"1 if secret_weight > {median_w} else 0",
            "class_counts": {
                "0": int(np.sum(y_cls == 0)),
                "1": int(np.sum(y_cls == 1)),
            },
            "majority_baseline": majority_baseline,
            "secret_weight_min": int(np.min(weights)),
            "secret_weight_max": int(np.max(weights)),
            "secret_weight_mean": float(np.mean(weights)),
            "secret_weight_std": float(np.std(weights, ddof=1)),
        },
        "clean_uninstrumented": clean_res,
        "tracemalloc_instrumented_2feat": tm2_res,
        "tracemalloc_instrumented_4feat": tm4_res,
    }

    sanitized = to_native(full_results)
    with open(results_path, "w") as f:
        json.dump(sanitized, f, indent=2)

    generate_summary_markdown(results_path, summary_path)
    if verbose:
        print_report(sanitized)
        print(f"Saved: {traces_path}, {results_path}, {summary_path} (total {total_wall_s:.1f}s)")
    return sanitized


def main():
    run_decrypt_leak_pipeline()


if __name__ == "__main__":
    main()
