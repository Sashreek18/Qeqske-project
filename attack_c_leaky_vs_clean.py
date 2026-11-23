"""
ATTACK C — Extended Positive Control & Graded-Leak Sensitivity Threshold
=========================================================================
Extends the original clean-vs-leaky control to address two methodological checks:

1. Full-Feature Clean vs. Leaky Comparison (500 trials):
   Uses the exact same multi-dimensional feature set as the main pipeline
   (timing_us, cpu_user_us, ram_peak_bytes, num_allocations) across BOTH
   key_gen() and decrypt_it() (8D combined, plus 4D key_gen and 4D decrypt):
     - Clean baseline         : key_gen() + decrypt_it_constant_time()
     - Leaky key_gen control  : key_gen_leaky() + decrypt_it_constant_time()
     - Leaky decrypt control  : key_gen() + decrypt_it_constant_time() + 50 us delay
     - Natural decrypt_it     : key_gen() + unmitigated original decrypt_it()

2. Graded Secret-Dependent Delay Sweep (0.0, 0.5, 1.0, 2.0, 5.0 us):
   Injects a calibrated busy-wait delay into constant-time decryption whenever
   secret_weight > median, and evaluates 5 classifiers (5x10 Repeated Stratified
   K-Fold CV + 200-permutation tests) on both Full 8D and Decrypt-Side 4D traces
   to measure the pipeline's minimum detection threshold.

Saves:
  - results/attack_c_leaky_vs_clean.json
  - results/attack_c_full_stats.json (summary entries)
  - results/graded_leak_threshold.png
  - results/graded_leak_threshold.svg
"""

import gc
import json
import math
import os
import time
import tracemalloc
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import resource
except ImportError:
    resource = None

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import psutil
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import attack_c_stats
from env_metadata import get_environment_metadata
from qeqske_core import (
    QEQSKERandom,
    decrypt_it,
    decrypt_it_constant_time,
    encrypt_it,
    key_gen as key_gen_clean,
    mat_add_mod,
    mat_mult_mod,
    right_rotate,
)
from qrng_handler import MersenneSource, QRNGSource

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TRIALS = 500
N_PARAM = 4
K_PARAM = 2
FIXED_PLAINTEXT = "HELLO"
SEED = 42
N_SPLITS = 5
N_REPEATS = 10
GRADED_DELAYS_US = [0.0, 0.5, 1.0, 2.0, 5.0]
STRONG_DECRYPT_DELAY_US = 50.0

KEYGEN_FEATURES = [
    "kg_timing_us",
    "kg_cpu_user_us",
    "kg_ram_peak_bytes",
    "kg_num_allocations",
]
DECRYPT_FEATURES = [
    "dec_timing_us",
    "dec_cpu_user_us",
    "dec_ram_peak_bytes",
    "dec_num_allocations",
]
FULL_FEATURES = KEYGEN_FEATURES + DECRYPT_FEATURES

RESULTS_JSON_PATH = "results/attack_c_leaky_vs_clean.json"
PLOT_PNG_PATH = "results/graded_leak_threshold.png"
PLOT_SVG_PATH = "results/graded_leak_threshold.svg"


# ---------------------------------------------------------------------------
# Backwards-compatible helpers (imported by system_comparison.py)
# ---------------------------------------------------------------------------

def _secret_properties(s):
    s_flat = [val for row in s for val in row]
    s_weight = sum(1 for v in s_flat if v != 0)
    s_sum = sum(abs(v) for v in s_flat)
    return s_weight, s_sum


def key_gen_leaky(rng, n, k, leak_threshold=5, leak_repeats=1500):
    """
    Identical to qeqske_core.key_gen(), EXCEPT: after generating the secret
    key s, if its weight exceeds `leak_threshold`, it performs `leak_repeats`
    extra, genuinely-executed redundant matrix multiplications before returning.
    Retained for positive control & system_comparison.py compatibility.
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

    weight, _ = _secret_properties(s)
    if weight > leak_threshold:
        dummy = A
        for _ in range(leak_repeats):
            dummy = mat_mult_mod(dummy, A, q)

    return {"private_key": s, "public_key_A": A, "public_key_t": t, "q": q, "n": n, "k": k}


# ---------------------------------------------------------------------------
# Timing, CPU, & Allocation Measurement Helpers
# ---------------------------------------------------------------------------

class _SliceRecorder:
    """Records the exact QRNG integer slice consumed by a key_gen() call."""

    def __init__(self, base_src):
        self._base = base_src
        self.recorded = []

    def call_qrng(self):
        val = self._base.call_qrng()
        self.recorded.append(val)
        return val


def _get_cpu_user_seconds(proc):
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_utime
    return proc.cpu_times().user


def _inject_delay_us(delay_us: float, trigger: bool) -> None:
    if delay_us <= 0.0 or not trigger:
        return
    target_ns = time.perf_counter_ns() + int(round(delay_us * 1000.0))
    while time.perf_counter_ns() < target_ns:
        pass


def pregenerate_exchange_trials(qrng_path="qrng_combined.txt", n_trials=N_TRIALS):
    """Pre-generate 500 deterministic QRNG trials and a Mersenne warm-up pool."""
    src = QRNGSource.from_file(qrng_path, cycle=True)
    trials = []
    for trial_id in range(n_trials):
        recorder = _SliceRecorder(src)
        rng_kg = QEQSKERandom(recorder)
        keys = key_gen_clean(rng_kg, n=N_PARAM, k=K_PARAM)
        rng_enc = QEQSKERandom(src)
        ct = encrypt_it(
            rng_enc,
            FIXED_PLAINTEXT,
            keys["public_key_A"],
            keys["public_key_t"],
            keys["q"],
            N_PARAM,
            K_PARAM,
        )
        s = keys["private_key"]
        wt, s_sum = _secret_properties(s)
        trials.append({
            "trial_id": trial_id,
            "kg_draws": recorder.recorded,
            "s": s,
            "u": ct["u"],
            "v_queue": ct["v_queue"],
            "q": keys["q"],
            "secret_weight": wt,
            "secret_sum": s_sum,
        })

    w_src = MersenneSource(seed=42000)
    warmup_pool = []
    for _ in range(100):
        w_rec = _SliceRecorder(w_src)
        w_keys = key_gen_clean(QEQSKERandom(w_rec), n=N_PARAM, k=K_PARAM)
        w_ct = encrypt_it(
            QEQSKERandom(w_src),
            FIXED_PLAINTEXT,
            w_keys["public_key_A"],
            w_keys["public_key_t"],
            w_keys["q"],
            N_PARAM,
            K_PARAM,
        )
        warmup_pool.append((w_rec.recorded, w_keys["private_key"], w_ct["u"], w_ct["v_queue"], w_keys["q"]))

    return trials, warmup_pool


def collect_full_feature_dataset(
    trials,
    warmup_pool,
    keygen_fn,
    decrypt_fn,
    delay_us=0.0,
    median_threshold=5.0,
    cached_tracemalloc=None,
):
    """
    Collect 8D main-pipeline features (4 key_gen + 4 decrypt) on all trials:
      - Pass A (uninstrumented, randomized order, CPU-pinned, warm-up interleaved):
        captures kg_timing_us, kg_cpu_user_us, dec_timing_us, dec_cpu_user_us.
      - Pass B (separate tracemalloc pass on identical replay inputs, or reused
        when code & allocations match clean baseline):
        captures kg_ram_peak_bytes, kg_num_allocations, dec_ram_peak_bytes, dec_num_allocations.
    """
    proc = psutil.Process(os.getpid())
    n_trials = len(trials)
    rows = [
        {
            "trial_id": tr["trial_id"],
            "q": int(tr["q"]),
            "secret_weight": int(tr["secret_weight"]),
            "secret_sum": int(tr["secret_sum"]),
        }
        for tr in trials
    ]

    orig_affinity = None
    if hasattr(os, "sched_setaffinity"):
        orig_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {0})

    try:
        # Warm-up adaptive specialization before Pass A
        gc.collect()
        for kg_draws, s, u, v_queue, q in warmup_pool:
            _ = keygen_fn(QEQSKERandom(QRNGSource(numbers=list(kg_draws))), N_PARAM, K_PARAM)
            _ = decrypt_fn(s, u, v_queue, q)

        exec_order = np.random.RandomState(SEED).permutation(n_trials)
        for pos, idx in enumerate(exec_order):
            if pos % 100 == 0:
                gc.collect()
                for w_i in range(10):
                    kg_draws, s, u, v_queue, q = warmup_pool[(pos + w_i) % len(warmup_pool)]
                    _ = keygen_fn(QEQSKERandom(QRNGSource(numbers=list(kg_draws))), N_PARAM, K_PARAM)
                    _ = decrypt_fn(s, u, v_queue, q)

            tr = trials[int(idx)]
            rng_kg = QEQSKERandom(QRNGSource(numbers=list(tr["kg_draws"])))

            cpu0 = _get_cpu_user_seconds(proc)
            t0 = time.perf_counter_ns()
            _ = keygen_fn(rng_kg, N_PARAM, K_PARAM)
            kg_ns = time.perf_counter_ns() - t0
            kg_cpu_s = _get_cpu_user_seconds(proc) - cpu0

            trigger = bool(tr["secret_weight"] > median_threshold)
            cpu1 = _get_cpu_user_seconds(proc)
            t1 = time.perf_counter_ns()
            dec_out = decrypt_fn(tr["s"], tr["u"], tr["v_queue"], tr["q"])
            _inject_delay_us(delay_us, trigger)
            dec_ns = time.perf_counter_ns() - t1
            dec_cpu_s = _get_cpu_user_seconds(proc) - cpu1

            rec = rows[int(idx)]
            rec["kg_timing_us"] = float(kg_ns / 1000.0)
            rec["kg_cpu_user_us"] = float(kg_cpu_s * 1e6)
            rec["dec_timing_us"] = float(dec_ns / 1000.0)
            rec["dec_cpu_user_us"] = float(dec_cpu_s * 1e6)
            rec["decryption_correct"] = bool(dec_out == FIXED_PLAINTEXT)

        # Pass B: tracemalloc allocations & peak RAM (isolated from Pass A timings)
        if cached_tracemalloc is not None:
            for idx, tm_vals in enumerate(cached_tracemalloc):
                rows[idx].update(tm_vals)
        else:
            gc.collect()
            for kg_draws, s, u, v_queue, q in warmup_pool[:20]:
                tracemalloc.start()
                _ = keygen_fn(QEQSKERandom(QRNGSource(numbers=list(kg_draws))), N_PARAM, K_PARAM)
                _ = tracemalloc.get_traced_memory()
                _ = tracemalloc.take_snapshot()
                tracemalloc.stop()

                tracemalloc.start()
                _ = decrypt_fn(s, u, v_queue, q)
                _ = tracemalloc.get_traced_memory()
                _ = tracemalloc.take_snapshot()
                tracemalloc.stop()

            for idx, tr in enumerate(trials):
                rng_kg = QEQSKERandom(QRNGSource(numbers=list(tr["kg_draws"])))
                tracemalloc.start()
                _ = keygen_fn(rng_kg, N_PARAM, K_PARAM)
                _, kg_peak = tracemalloc.get_traced_memory()
                kg_snap = tracemalloc.take_snapshot()
                kg_allocs = sum(stat.count for stat in kg_snap.statistics("lineno"))
                tracemalloc.stop()

                tracemalloc.start()
                _ = decrypt_fn(tr["s"], tr["u"], tr["v_queue"], tr["q"])
                _, dec_peak = tracemalloc.get_traced_memory()
                dec_snap = tracemalloc.take_snapshot()
                dec_allocs = sum(stat.count for stat in dec_snap.statistics("lineno"))
                tracemalloc.stop()

                rows[idx]["kg_ram_peak_bytes"] = int(kg_peak)
                rows[idx]["kg_num_allocations"] = int(kg_allocs)
                rows[idx]["dec_ram_peak_bytes"] = int(dec_peak)
                rows[idx]["dec_num_allocations"] = int(dec_allocs)
    finally:
        if orig_affinity is not None:
            os.sched_setaffinity(0, orig_affinity)

    return rows


def collect_graded_delay_datasets(
    trials,
    warmup_pool,
    delays_us,
    cached_tracemalloc,
    median_threshold=5.0,
):
    """
    Collect all graded delay conditions in a single CPU-pinned, warm-up-interleaved
    pass using randomized (trial, delay) ordering, BEFORE any multi-core ML jobs run.
    This eliminates slow thermal / CPU frequency drift between delay conditions.
    """
    proc = psutil.Process(os.getpid())
    n_trials = len(trials)
    datasets = {
        float(d): [
            {
                "trial_id": tr["trial_id"],
                "q": int(tr["q"]),
                "secret_weight": int(tr["secret_weight"]),
                "secret_sum": int(tr["secret_sum"]),
                "kg_timing_us": 0.0,
                "kg_cpu_user_us": 0.0,
                "dec_timing_us": 0.0,
                "dec_cpu_user_us": 0.0,
                **cached_tracemalloc[i],
            }
            for i, tr in enumerate(trials)
        ]
        for d in delays_us
    }

    schedule = [(float(d), idx) for d in delays_us for idx in range(n_trials)]
    perm = np.random.RandomState(SEED + 7).permutation(len(schedule))

    orig_affinity = None
    if hasattr(os, "sched_setaffinity"):
        orig_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {0})

    try:
        gc.collect()
        for kg_draws, s, u, v_queue, q in warmup_pool:
            _ = key_gen_clean(QEQSKERandom(QRNGSource(numbers=list(kg_draws))), N_PARAM, K_PARAM)
            _ = decrypt_it_constant_time(s, u, v_queue, q)

        for step, s_idx in enumerate(perm):
            if step % 250 == 0:
                gc.collect()
                for w_i in range(10):
                    kg_draws, s, u, v_queue, q = warmup_pool[(step + w_i) % len(warmup_pool)]
                    _ = key_gen_clean(QEQSKERandom(QRNGSource(numbers=list(kg_draws))), N_PARAM, K_PARAM)
                    _ = decrypt_it_constant_time(s, u, v_queue, q)

            d_us, idx = schedule[int(s_idx)]
            tr = trials[idx]
            rng_kg = QEQSKERandom(QRNGSource(numbers=list(tr["kg_draws"])))

            cpu0 = _get_cpu_user_seconds(proc)
            t0 = time.perf_counter_ns()
            _ = key_gen_clean(rng_kg, N_PARAM, K_PARAM)
            kg_ns = time.perf_counter_ns() - t0
            kg_cpu_s = _get_cpu_user_seconds(proc) - cpu0

            trigger = bool(tr["secret_weight"] > median_threshold)
            cpu1 = _get_cpu_user_seconds(proc)
            t1 = time.perf_counter_ns()
            dec_out = decrypt_it_constant_time(tr["s"], tr["u"], tr["v_queue"], tr["q"])
            _inject_delay_us(d_us, trigger)
            dec_ns = time.perf_counter_ns() - t1
            dec_cpu_s = _get_cpu_user_seconds(proc) - cpu1

            rec = datasets[d_us][idx]
            rec["kg_timing_us"] = float(kg_ns / 1000.0)
            rec["kg_cpu_user_us"] = float(kg_cpu_s * 1e6)
            rec["dec_timing_us"] = float(dec_ns / 1000.0)
            rec["dec_cpu_user_us"] = float(dec_cpu_s * 1e6)
            rec["decryption_correct"] = bool(dec_out == FIXED_PLAINTEXT)
    finally:
        if orig_affinity is not None:
            os.sched_setaffinity(0, orig_affinity)

    return datasets


# ---------------------------------------------------------------------------
# Multi-Model Evaluation
# ---------------------------------------------------------------------------

def _get_classifiers():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
        ),
        "Random Forest": make_pipeline(
            StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
        ),
        "Gradient Boosting": make_pipeline(
            StandardScaler(), GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED)
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", random_state=SEED)
        ),
        "Neural Network (MLP)": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=SEED)
        ),
    }


def evaluate_feature_subset(rows, feature_names, label_name="secret_weight"):
    """Evaluate all 5 classifiers under 5x10 Repeated Stratified CV + 200 permutations."""
    X = np.array([[r[f] for f in feature_names] for r in rows], dtype=float)
    vals = np.array([r[label_name] for r in rows], dtype=float)
    median_v = float(np.median(vals))
    y = (vals > median_v).astype(int)
    baseline = float(max(np.mean(y == 0), np.mean(y == 1)))

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    models_out = {}
    for name, model in _get_classifiers().items():
        ml_stats = attack_c_stats.compute_ml_stats(model, X, y, rskf)
        acc_mean = ml_stats["accuracy"]["mean"]
        models_out[name] = {
            **ml_stats,
            "delta_vs_baseline": float(acc_mean - baseline),
        }

    best_model = max(models_out, key=lambda k: models_out[k]["accuracy"]["mean"])
    best_stats = models_out[best_model]
    return {
        "features": list(feature_names),
        "n_trials": int(len(rows)),
        "median_target": median_v,
        "majority_baseline": baseline,
        "models": models_out,
        "best_model": best_model,
        "best_accuracy": float(best_stats["accuracy"]["mean"]),
        "best_sd": float(best_stats["accuracy"]["sd"]),
        "best_delta_vs_baseline": float(best_stats["delta_vs_baseline"]),
        "best_roc_auc": float(best_stats["roc_auc"]["mean"]),
        "best_perm_pvalue": float(best_stats["permutation_test"]["pvalue"]),
        "best_perm_display": best_stats["permutation_test"]["pvalue_display"],
    }


def _timing_summary(rows, y):
    kg_t = np.array([r["kg_timing_us"] for r in rows], dtype=float)
    dec_t = np.array([r["dec_timing_us"] for r in rows], dtype=float)
    dec_0 = dec_t[y == 0]
    dec_1 = dec_t[y == 1]
    kg_0 = kg_t[y == 0]
    kg_1 = kg_t[y == 1]
    return {
        "kg_median_y0_us": float(np.median(kg_0)),
        "kg_median_y1_us": float(np.median(kg_1)),
        "kg_median_delta_us": float(np.median(kg_1) - np.median(kg_0)),
        "dec_median_y0_us": float(np.median(dec_0)),
        "dec_median_y1_us": float(np.median(dec_1)),
        "dec_median_delta_us": float(np.median(dec_1) - np.median(dec_0)),
        "dec_iqr_y0_us": float(np.percentile(dec_0, 75.0) - np.percentile(dec_0, 25.0)),
        "dec_iqr_y1_us": float(np.percentile(dec_1, 75.0) - np.percentile(dec_1, 25.0)),
        "dec_mean_delta_us": float(np.mean(dec_1) - np.mean(dec_0)),
    }


# ---------------------------------------------------------------------------
# Visualization: Detection Accuracy vs. Delay (PNG via Pillow + clean SVG)
# ---------------------------------------------------------------------------

MODEL_STYLES = [
    ("Gradient Boosting", "#2563EB", "circle"),
    ("SVM (RBF)", "#DC2626", "square"),
    ("Neural Network (MLP)", "#059669", "diamond"),
    ("Random Forest", "#D97706", "triangle"),
    ("Logistic Regression", "#7C3AED", "cross"),
]


def _load_fonts(scale=2):
    regular_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    bold_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    reg_p = next((p for p in regular_paths if os.path.exists(p)), None)
    bold_p = next((p for p in bold_paths if os.path.exists(p)), None)
    if reg_p and bold_p:
        return {
            "title": ImageFont.truetype(bold_p, 21 * scale),
            "subtitle": ImageFont.truetype(reg_p, 13 * scale),
            "panel_title": ImageFont.truetype(bold_p, 15 * scale),
            "axis": ImageFont.truetype(bold_p, 13 * scale),
            "tick": ImageFont.truetype(reg_p, 12 * scale),
            "legend": ImageFont.truetype(reg_p, 11 * scale),
            "annot": ImageFont.truetype(bold_p, 11 * scale),
            "annot_sm": ImageFont.truetype(reg_p, 10 * scale),
        }
    dflt = ImageFont.load_default()
    return {k: dflt for k in ("title", "subtitle", "panel_title", "axis", "tick", "legend", "annot", "annot_sm")}


def _draw_marker_pil(draw, shape, cx, cy, r, color):
    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline="#FFFFFF", width=2)
    elif shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=color, outline="#FFFFFF", width=2)
    elif shape == "diamond":
        pts = [(cx, cy - r - 1), (cx + r + 1, cy), (cx, cy + r + 1), (cx - r - 1, cy)]
        draw.polygon(pts, fill=color, outline="#FFFFFF")
    elif shape == "triangle":
        pts = [(cx, cy - r - 1), (cx + r + 1, cy + r), (cx - r - 1, cy + r)]
        draw.polygon(pts, fill=color, outline="#FFFFFF")
    else:
        draw.line((cx - r, cy - r, cx + r, cy + r), fill=color, width=3)
        draw.line((cx - r, cy + r, cx + r, cy - r), fill=color, width=3)


def _draw_dashed_line_pil(draw, x0, y0, x1, y1, color, width=2, dash=10, gap=7):
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    if dist <= 0:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    while pos < dist:
        end = min(pos + dash, dist)
        draw.line(
            (x0 + ux * pos, y0 + uy * pos, x0 + ux * end, y0 + uy * end),
            fill=color,
            width=width,
        )
        pos += dash + gap


def _draw_text_centered(draw, x, y, text, font, fill, anchor="mm"):
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)


def _draw_graded_plot_svg(delays, sweep_records, baseline_pct, natural_full_pct, threshold_summary, svg_path):
    w, h = 1480, 640
    x_min, x_max = -0.35, 5.45
    y_min, y_max = 45.0, 105.0

    panels = [
        (90, 695, 115, 545, "A. All 5 Classifiers (Full 8D Feature Set)"),
        (815, 1420, 115, 545, "B. Best-Model Threshold: Full 8D vs. Decrypt 4D"),
    ]

    def sx(px0, px1, val):
        return px0 + (val - x_min) / (x_max - x_min) * (px1 - px0)

    def sy(py0, py1, val):
        return py1 - (val - y_min) / (y_max - y_min) * (py1 - py0)

    def marker_svg(shape, cx, cy, r, color):
        if shape == "circle":
            return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" stroke="#FFFFFF" stroke-width="1.5"/>'
        if shape == "square":
            return f'<rect x="{cx-r:.1f}" y="{cy-r:.1f}" width="{2*r}" height="{2*r}" fill="{color}" stroke="#FFFFFF" stroke-width="1.5"/>'
        if shape == "diamond":
            return f'<polygon points="{cx:.1f},{cy-r-1:.1f} {cx+r+1:.1f},{cy:.1f} {cx:.1f},{cy+r+1:.1f} {cx-r-1:.1f},{cy:.1f}" fill="{color}" stroke="#FFFFFF" stroke-width="1.2"/>'
        if shape == "triangle":
            return f'<polygon points="{cx:.1f},{cy-r-1:.1f} {cx+r+1:.1f},{cy+r:.1f} {cx-r-1:.1f},{cy+r:.1f}" fill="{color}" stroke="#FFFFFF" stroke-width="1.2"/>'
        return (
            f'<line x1="{cx-r:.1f}" y1="{cy-r:.1f}" x2="{cx+r:.1f}" y2="{cy+r:.1f}" stroke="{color}" stroke-width="2.2"/>'
            f'<line x1="{cx-r:.1f}" y1="{cy+r:.1f}" x2="{cx+r:.1f}" y2="{cy-r:.1f}" stroke="{color}" stroke-width="2.2"/>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text { font-family: "DejaVu Sans", Arial, sans-serif; fill: #1E293B; }</style>',
        f'<text x="{w/2}" y="38" text-anchor="middle" font-size="21" font-weight="bold">Positive Control &amp; Graded Side-Channel Sensitivity Sweep (500 Trials, 5×10 CV)</text>',
        f'<text x="{w/2}" y="64" text-anchor="middle" font-size="13" fill="#475569">Constant-time decrypt + calibrated secret-dependent delay (0.0, 0.5, 1.0, 2.0, 5.0 µs) | Baseline = {baseline_pct:.2f}%</text>',
    ]

    for p_idx, (px0, px1, py0, py1, title) in enumerate(panels):
        parts.append(f'<text x="{(px0+px1)/2:.1f}" y="{py0-20}" text-anchor="middle" font-size="15" font-weight="bold">{title}</text>')
        parts.append(f'<rect x="{px0}" y="{py0}" width="{px1-px0}" height="{py1-py0}" fill="#FAFAFA" stroke="#CBD5E1" stroke-width="1.2"/>')

        for y_tick in [50, 60, 70, 80, 90, 100]:
            yy = sy(py0, py1, y_tick)
            parts.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" stroke="#E2E8F0" stroke-width="1"/>')
            parts.append(f'<text x="{px0-10}" y="{yy+4:.1f}" text-anchor="end" font-size="12" fill="#475569">{y_tick}%</text>')

        for d_val in delays:
            xx = sx(px0, px1, d_val)
            parts.append(f'<line x1="{xx:.1f}" y1="{py0}" x2="{xx:.1f}" y2="{py1}" stroke="#F1F5F9" stroke-width="1"/>')
            lbl = f"{d_val:.1f}" if d_val == 0.5 else f"{int(d_val)}" if d_val in (0.0, 1.0, 2.0, 5.0) else f"{d_val:g}"
            parts.append(f'<text x="{xx:.1f}" y="{py1+22}" text-anchor="middle" font-size="12" fill="#475569">{lbl}</text>')

        parts.append(f'<text x="{(px0+px1)/2:.1f}" y="{py1+54}" text-anchor="middle" font-size="13" font-weight="bold">Injected Secret-Dependent Delay (µs)</text>')
        parts.append(f'<text x="{px0-56}" y="{(py0+py1)/2:.1f}" text-anchor="middle" font-size="13" font-weight="bold" transform="rotate(-90 {px0-56} {(py0+py1)/2:.1f})">5×10 Stratified CV Accuracy (%)</text>')

        # Baseline reference line
        yb = sy(py0, py1, baseline_pct)
        parts.append(f'<line x1="{px0}" y1="{yb:.1f}" x2="{px1}" y2="{yb:.1f}" stroke="#64748B" stroke-width="1.6" stroke-dasharray="7,5"/>')
        parts.append(f'<text x="{px1-8}" y="{yb-7:.1f}" text-anchor="end" font-size="11" fill="#475569">Majority Baseline ({baseline_pct:.1f}%)</text>')

        if p_idx == 0:
            yn = sy(py0, py1, natural_full_pct)
            parts.append(f'<line x1="{px0}" y1="{yn:.1f}" x2="{px1}" y2="{yn:.1f}" stroke="#0F172A" stroke-width="1.5" stroke-dasharray="3,4"/>')
            parts.append(f'<text x="{px1-8}" y="{yn-7:.1f}" text-anchor="end" font-size="11" fill="#0F172A">Unmitigated decrypt_it ({natural_full_pct:.1f}%)</text>')

            for m_name, color, shape in MODEL_STYLES:
                pts = []
                for rec in sweep_records:
                    xx = sx(px0, px1, rec["delay_us"])
                    acc = rec["full_8feat"]["models"][m_name]["accuracy"]["mean"] * 100.0
                    yy = sy(py0, py1, acc)
                    pts.append((xx, yy))
                poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{poly}"/>')
                for xx, yy in pts:
                    parts.append(marker_svg(shape, xx, yy, 5, color))

            # Legend box top-left (strictly clear of curves and reference lines)
            lx, ly, lw, lh = px0 + 12, py0 + 12, 200, 118
            parts.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="6" fill="#FFFFFF" fill-opacity="0.95" stroke="#CBD5E1"/>')
            for i, (m_name, color, shape) in enumerate(MODEL_STYLES):
                ry = ly + 18 + i * 21
                parts.append(f'<line x1="{lx+12}" y1="{ry}" x2="{lx+34}" y2="{ry}" stroke="{color}" stroke-width="2.4"/>')
                parts.append(marker_svg(shape, lx + 23, ry, 4, color))
                parts.append(f'<text x="{lx+42}" y="{ry+4}" font-size="11">{m_name}</text>')
        else:
            # Panel B: threshold shade & Full 8D vs Decrypt 4D best curves with ±1 SD
            thresh_us = threshold_summary.get("min_significant_delay_us")
            if thresh_us is not None:
                xt = sx(px0, px1, thresh_us)
                parts.append(f'<rect x="{xt-10:.1f}" y="{py0}" width="{px1-(xt-10):.1f}" height="{py1-py0}" fill="#ECFDF5" fill-opacity="0.55"/>')
                parts.append(f'<line x1="{xt:.1f}" y1="{py0}" x2="{xt:.1f}" y2="{py1}" stroke="#059669" stroke-width="1.5" stroke-dasharray="5,4"/>')
                parts.append(f'<text x="{xt+8:.1f}" y="{py0+20}" font-size="11" font-weight="bold" fill="#047857">Threshold: {thresh_us:g} µs (p &lt; 1/201)</text>')

            series_specs = [
                ("Full 8D (KeyGen + Decrypt)", "full_8feat", "#0F766E", "circle"),
                ("Decrypt-Side 4D Only", "decrypt_4feat", "#E11D48", "diamond"),
            ]
            for label, key, color, shape in series_specs:
                pts = []
                for rec in sweep_records:
                    sub = rec[key]
                    xx = sx(px0, px1, rec["delay_us"])
                    acc = sub["best_accuracy"] * 100.0
                    sd = sub["best_sd"] * 100.0
                    yy = sy(py0, py1, acc)
                    y_lo = sy(py0, py1, max(y_min, acc - sd))
                    y_hi = sy(py0, py1, min(y_max, acc + sd))
                    pts.append((xx, yy, y_lo, y_hi, acc, sub["best_perm_pvalue"] <= 0.01))
                for xx, yy, y_lo, y_hi, _, _ in pts:
                    parts.append(f'<line x1="{xx:.1f}" y1="{y_lo:.1f}" x2="{xx:.1f}" y2="{y_hi:.1f}" stroke="{color}" stroke-width="1.6" stroke-opacity="0.7"/>')
                    parts.append(f'<line x1="{xx-4:.1f}" y1="{y_lo:.1f}" x2="{xx+4:.1f}" y2="{y_lo:.1f}" stroke="{color}" stroke-width="1.4"/>')
                    parts.append(f'<line x1="{xx-4:.1f}" y1="{y_hi:.1f}" x2="{xx+4:.1f}" y2="{y_hi:.1f}" stroke="{color}" stroke-width="1.4"/>')
                poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, *_ in pts)
                parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.8" points="{poly}"/>')
                for idx_pt, (xx, yy, y_lo, y_hi, acc, is_sig) in enumerate(pts):
                    parts.append(marker_svg(shape, xx, yy, 5.5, color))
                    star = "*" if is_sig else ""
                    if key == "full_8feat":
                        parts.append(f'<text x="{xx:.1f}" y="{y_hi-10:.1f}" text-anchor="middle" font-size="10.5" font-weight="bold" fill="{color}">{acc:.1f}%{star}</text>')
                    else:
                        parts.append(f'<text x="{xx:.1f}" y="{y_lo+14:.1f}" text-anchor="middle" font-size="10.5" font-weight="bold" fill="{color}">{acc:.1f}%{star}</text>')

            lx, ly, lw, lh = px1 - 245, py1 - 64, 235, 54
            parts.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" rx="6" fill="#FFFFFF" fill-opacity="0.95" stroke="#CBD5E1"/>')
            for i, (label, _, color, shape) in enumerate(series_specs):
                ry = ly + 18 + i * 21
                parts.append(f'<line x1="{lx+12}" y1="{ry}" x2="{lx+36}" y2="{ry}" stroke="{color}" stroke-width="2.8"/>')
                parts.append(marker_svg(shape, lx + 24, ry, 4.5, color))
                parts.append(f'<text x="{lx+44}" y="{ry+4}" font-size="11">{label}</text>')

    parts.append("</svg>\n")
    with open(svg_path, "w") as f:
        f.write("\n".join(parts))


def generate_threshold_plot(sweep_records, baseline_pct, natural_full_pct, threshold_summary, png_path, svg_path):
    """Render both pure-Python SVG and 2x-supersampled Pillow PNG without matplotlib."""
    delays = [r["delay_us"] for r in sweep_records]
    _draw_graded_plot_svg(delays, sweep_records, baseline_pct, natural_full_pct, threshold_summary, svg_path)

    # Pillow 2x-supersampled renderer
    S = 2
    W, H = 1480 * S, 640 * S
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img, "RGBA")
    fonts = _load_fonts(scale=S)

    x_min, x_max = -0.35, 5.45
    y_min, y_max = 45.0, 105.0
    panels = [
        (90 * S, 695 * S, 115 * S, 545 * S, "A. All 5 Classifiers (Full 8D Feature Set)"),
        (815 * S, 1420 * S, 115 * S, 545 * S, "B. Best-Model Threshold: Full 8D vs. Decrypt 4D"),
    ]

    def sx(px0, px1, val):
        return px0 + (val - x_min) / (x_max - x_min) * (px1 - px0)

    def sy(py0, py1, val):
        return py1 - (val - y_min) / (y_max - y_min) * (py1 - py0)

    _draw_text_centered(
        draw, W / 2, 36 * S,
        "Positive Control & Graded Side-Channel Sensitivity Sweep (500 Trials, 5×10 CV)",
        fonts["title"], "#0F172A",
    )
    _draw_text_centered(
        draw, W / 2, 66 * S,
        f"Constant-time decrypt + calibrated secret-dependent delay (0.0, 0.5, 1.0, 2.0, 5.0 µs) | Majority Baseline = {baseline_pct:.2f}%",
        fonts["subtitle"], "#475569",
    )

    for p_idx, (px0, px1, py0, py1, title) in enumerate(panels):
        _draw_text_centered(draw, (px0 + px1) / 2, py0 - 20 * S, title, fonts["panel_title"], "#1E293B")
        draw.rectangle((px0, py0, px1, py1), fill="#FAFAFA", outline="#CBD5E1", width=2)

        for y_tick in [50, 60, 70, 80, 90, 100]:
            yy = sy(py0, py1, y_tick)
            draw.line((px0, yy, px1, yy), fill="#E2E8F0", width=2)
            _draw_text_centered(draw, px0 - 10 * S, yy, f"{y_tick}%", fonts["tick"], "#475569", anchor="rm")

        for d_val in delays:
            xx = sx(px0, px1, d_val)
            draw.line((xx, py0, xx, py1), fill="#F1F5F9", width=2)
            lbl = f"{d_val:.1f}" if d_val == 0.5 else f"{int(d_val)}" if d_val in (0.0, 1.0, 2.0, 5.0) else f"{d_val:g}"
            _draw_text_centered(draw, xx, py1 + 20 * S, lbl, fonts["tick"], "#475569", anchor="mm")

        _draw_text_centered(draw, (px0 + px1) / 2, py1 + 52 * S, "Injected Secret-Dependent Delay (µs)", fonts["axis"], "#1E293B")
        if p_idx == 0:
            # Rotated Y-axis label rendered via temp image
            txt_img = Image.new("RGBA", (420 * S, 40 * S), (255, 255, 255, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            _draw_text_centered(txt_draw, 210 * S, 20 * S, "5×10 Stratified CV Accuracy (%)", fonts["axis"], "#1E293B")
            rot = txt_img.rotate(90, expand=True)
            img.paste(rot, (10 * S, int((py0 + py1) / 2 - rot.height / 2)), rot)

        yb = sy(py0, py1, baseline_pct)
        _draw_dashed_line_pil(draw, px0, yb, px1, yb, "#64748B", width=3, dash=14, gap=10)
        _draw_text_centered(draw, px1 - 8 * S, yb - 12 * S, f"Majority Baseline ({baseline_pct:.1f}%)", fonts["annot_sm"], "#475569", anchor="rm")

        if p_idx == 0:
            yn = sy(py0, py1, natural_full_pct)
            _draw_dashed_line_pil(draw, px0, yn, px1, yn, "#0F172A", width=3, dash=6, gap=8)
            _draw_text_centered(draw, px1 - 8 * S, yn - 12 * S, f"Unmitigated decrypt_it ({natural_full_pct:.1f}%)", fonts["annot_sm"], "#0F172A", anchor="rm")

            for m_name, color, shape in MODEL_STYLES:
                pts = []
                for rec in sweep_records:
                    xx = sx(px0, px1, rec["delay_us"])
                    acc = rec["full_8feat"]["models"][m_name]["accuracy"]["mean"] * 100.0
                    yy = sy(py0, py1, acc)
                    pts.append((xx, yy))
                for i in range(len(pts) - 1):
                    draw.line((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]), fill=color, width=5)
                for xx, yy in pts:
                    _draw_marker_pil(draw, shape, xx, yy, 5 * S, color)

            lx0, ly0, lx1, ly1 = px0 + 12 * S, py0 + 12 * S, px0 + 212 * S, py0 + 130 * S
            draw.rounded_rectangle((lx0, ly0, lx1, ly1), radius=8 * S, fill=(255, 255, 255, 244), outline="#CBD5E1", width=2)
            for i, (m_name, color, shape) in enumerate(MODEL_STYLES):
                ry = ly0 + (18 + i * 21) * S
                draw.line((lx0 + 12 * S, ry, lx0 + 34 * S, ry), fill=color, width=5)
                _draw_marker_pil(draw, shape, lx0 + 23 * S, ry, 4 * S, color)
                _draw_text_centered(draw, lx0 + 42 * S, ry, m_name, fonts["legend"], "#1E293B", anchor="lm")
        else:
            thresh_us = threshold_summary.get("min_significant_delay_us")
            if thresh_us is not None:
                xt = sx(px0, px1, thresh_us)
                draw.rectangle((xt - 10 * S, py0 + 2, px1 - 2, py1 - 2), fill=(236, 253, 245, 145))
                _draw_dashed_line_pil(draw, xt, py0, xt, py1, "#059669", width=3, dash=10, gap=8)
                _draw_text_centered(draw, xt + 8 * S, py0 + 18 * S, f"Threshold: {thresh_us:g} µs (p < 1/201)", fonts["annot"], "#047857", anchor="lm")

            series_specs = [
                ("Full 8D (KeyGen + Decrypt)", "full_8feat", "#0F766E", "circle"),
                ("Decrypt-Side 4D Only", "decrypt_4feat", "#E11D48", "diamond"),
            ]
            for label, key, color, shape in series_specs:
                pts = []
                for rec in sweep_records:
                    sub = rec[key]
                    xx = sx(px0, px1, rec["delay_us"])
                    acc = sub["best_accuracy"] * 100.0
                    sd = sub["best_sd"] * 100.0
                    yy = sy(py0, py1, acc)
                    y_lo = sy(py0, py1, max(y_min, acc - sd))
                    y_hi = sy(py0, py1, min(y_max, acc + sd))
                    pts.append((xx, yy, y_lo, y_hi, acc, sub["best_perm_pvalue"] <= 0.01))
                for xx, yy, y_lo, y_hi, _, _ in pts:
                    draw.line((xx, y_lo, xx, y_hi), fill=color, width=3)
                    draw.line((xx - 5 * S, y_lo, xx + 5 * S, y_lo), fill=color, width=3)
                    draw.line((xx - 5 * S, y_hi, xx + 5 * S, y_hi), fill=color, width=3)
                for i in range(len(pts) - 1):
                    draw.line((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]), fill=color, width=6)
                for idx_pt, (xx, yy, y_lo, y_hi, acc, is_sig) in enumerate(pts):
                    _draw_marker_pil(draw, shape, xx, yy, 5.5 * S, color)
                    star = "*" if is_sig else ""
                    lbl_y = (y_hi - 11 * S) if key == "full_8feat" else (y_lo + 14 * S)
                    _draw_text_centered(draw, xx, lbl_y, f"{acc:.1f}%{star}", fonts["annot"], color, anchor="mm")

            lx0, ly0, lx1, ly1 = px1 - 245 * S, py1 - 64 * S, px1 - 10 * S, py1 - 10 * S
            draw.rounded_rectangle((lx0, ly0, lx1, ly1), radius=8 * S, fill=(255, 255, 255, 244), outline="#CBD5E1", width=2)
            for i, (label, _, color, shape) in enumerate(series_specs):
                ry = ly0 + (18 + i * 21) * S
                draw.line((lx0 + 12 * S, ry, lx0 + 36 * S, ry), fill=color, width=6)
                _draw_marker_pil(draw, shape, lx0 + 24 * S, ry, 4.5 * S, color)
                _draw_text_centered(draw, lx0 + 44 * S, ry, label, fonts["legend"], "#1E293B", anchor="lm")

    final_img = img.resize((1480, 640), Image.Resampling.LANCZOS)
    final_img.save(png_path, format="PNG")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def _print_eval_block(title, ev):
    print(f"\n  [{title}] (features={len(ev['features'])}, baseline={ev['majority_baseline']*100:.2f}%)")
    print(f"    {'Model':<22} {'Acc (Mean ± SD)':>18} {'ROC-AUC':>9} {'Delta':>9} {'Perm p':>18}")
    for m_name, m in ev["models"].items():
        acc = m["accuracy"]
        auc = m["roc_auc"]["mean"]
        p_disp = m["permutation_test"]["pvalue_display"]
        print(
            f"    {m_name:<22} {acc['mean']*100:6.2f}% ± {acc['sd']*100:4.2f}% "
            f"{auc:9.4f} {m['delta_vs_baseline']*100:+8.2f}% {p_disp:>18}"
        )


def run_extended_positive_control():
    os.makedirs("results", exist_ok=True)
    env = get_environment_metadata()

    print("#" * 78)
    print("ATTACK C — Extended Full-Feature Positive Control & Graded Leak Threshold")
    print("#" * 78)
    print(f"Environment: {env['os']} | CPU: {env['cpu_model']} | Python {env['python_version']} | NumPy {env['numpy_version']}")

    trials, warmup_pool = pregenerate_exchange_trials("qrng_combined.txt", N_TRIALS)
    weights = np.array([tr["secret_weight"] for tr in trials], dtype=float)
    median_w = float(np.median(weights))
    y = (weights > median_w).astype(int)
    baseline = float(max(np.mean(y == 0), np.mean(y == 1)))
    print(f"Prepared {len(trials)} trials (median secret_weight={median_w}, class0={np.sum(y==0)}, class1={np.sum(y==1)}, baseline={baseline*100:.2f}%)")

    # -----------------------------------------------------------------------
    # Collect ALL trace datasets BEFORE running multi-core ML evaluations
    # (prevents 24-core permutation sweeps from inducing CPU frequency/thermal drift)
    # -----------------------------------------------------------------------
    print("\nCollecting all Part 1 & Part 2 traces (isolated, CPU-pinned, interleaved)...")
    clean_rows = collect_full_feature_dataset(
        trials, warmup_pool, key_gen_clean, decrypt_it_constant_time, delay_us=0.0, median_threshold=median_w
    )
    ct_tm_cache = [
        {
            "kg_ram_peak_bytes": r["kg_ram_peak_bytes"],
            "kg_num_allocations": r["kg_num_allocations"],
            "dec_ram_peak_bytes": r["dec_ram_peak_bytes"],
            "dec_num_allocations": r["dec_num_allocations"],
        }
        for r in clean_rows
    ]
    leaky_kg_rows = collect_full_feature_dataset(
        trials, warmup_pool, key_gen_leaky, decrypt_it_constant_time, delay_us=0.0, median_threshold=median_w
    )
    leaky_dec_rows = collect_full_feature_dataset(
        trials, warmup_pool, key_gen_clean, decrypt_it_constant_time, delay_us=STRONG_DECRYPT_DELAY_US,
        median_threshold=median_w, cached_tracemalloc=ct_tm_cache,
    )
    natural_orig_rows = collect_full_feature_dataset(
        trials, warmup_pool, key_gen_clean, decrypt_it, delay_us=0.0, median_threshold=median_w
    )
    graded_datasets = collect_graded_delay_datasets(
        trials, warmup_pool, GRADED_DELAYS_US, cached_tracemalloc=ct_tm_cache, median_threshold=median_w
    )

    # -----------------------------------------------------------------------
    # Part 1: Full-Feature Clean vs. Leaky Comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PART 1: FULL-FEATURE CLEAN vs. LEAKY COMPARISON (Timing, CPU, RAM, Allocs)")
    print("=" * 78)

    part1 = {
        "clean_baseline": {
            "timing_summary": _timing_summary(clean_rows, y),
            "keygen_4feat": evaluate_feature_subset(clean_rows, KEYGEN_FEATURES),
            "decrypt_4feat": evaluate_feature_subset(clean_rows, DECRYPT_FEATURES),
            "full_8feat": evaluate_feature_subset(clean_rows, FULL_FEATURES),
        },
        "leaky_keygen": {
            "timing_summary": _timing_summary(leaky_kg_rows, y),
            "keygen_4feat": evaluate_feature_subset(leaky_kg_rows, KEYGEN_FEATURES),
            "full_8feat": evaluate_feature_subset(leaky_kg_rows, FULL_FEATURES),
        },
        "leaky_decrypt_50us": {
            "delay_us": STRONG_DECRYPT_DELAY_US,
            "timing_summary": _timing_summary(leaky_dec_rows, y),
            "decrypt_4feat": evaluate_feature_subset(leaky_dec_rows, DECRYPT_FEATURES),
            "full_8feat": evaluate_feature_subset(leaky_dec_rows, FULL_FEATURES),
        },
        "natural_original_decrypt": {
            "timing_summary": _timing_summary(natural_orig_rows, y),
            "decrypt_4feat": evaluate_feature_subset(natural_orig_rows, DECRYPT_FEATURES),
            "full_8feat": evaluate_feature_subset(natural_orig_rows, FULL_FEATURES),
        },
    }

    _print_eval_block("Clean Baseline — Full 8D", part1["clean_baseline"]["full_8feat"])
    _print_eval_block("Clean Baseline — Decrypt 4D", part1["clean_baseline"]["decrypt_4feat"])
    _print_eval_block("Leaky KeyGen Control — Full 8D", part1["leaky_keygen"]["full_8feat"])
    _print_eval_block(f"Leaky Decrypt ({STRONG_DECRYPT_DELAY_US:g} us) — Full 8D", part1["leaky_decrypt_50us"]["full_8feat"])
    _print_eval_block("Natural Unmitigated decrypt_it — Full 8D", part1["natural_original_decrypt"]["full_8feat"])
    _print_eval_block("Natural Unmitigated decrypt_it — Decrypt 4D", part1["natural_original_decrypt"]["decrypt_4feat"])

    # -----------------------------------------------------------------------
    # Part 2: Graded Delay Sweep (0.0, 0.5, 1.0, 2.0, 5.0 us)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PART 2: GRADED SECRET-DEPENDENT DELAY SWEEP (0.0, 0.5, 1.0, 2.0, 5.0 us)")
    print("=" * 78)

    sweep_records = []
    for d_us in GRADED_DELAYS_US:
        d_rows = graded_datasets[float(d_us)]
        t_sum = _timing_summary(d_rows, y)
        ev_dec = evaluate_feature_subset(d_rows, DECRYPT_FEATURES)
        ev_full = evaluate_feature_subset(d_rows, FULL_FEATURES)
        sweep_records.append({
            "delay_us": float(d_us),
            "timing_summary": t_sum,
            "decrypt_4feat": ev_dec,
            "full_8feat": ev_full,
        })
        print(
            f"  Delay={d_us:3.1f} us (meas delta={t_sum['dec_median_delta_us']:+5.2f} us) | "
            f"Full 8D: {ev_full['best_accuracy']*100:5.2f}% ({ev_full['best_model']}, p {ev_full['best_perm_display']}) | "
            f"Decrypt 4D: {ev_dec['best_accuracy']*100:5.2f}% ({ev_dec['best_model']}, p {ev_dec['best_perm_display']})"
        )

    def _first_delay(pred_fn):
        for r in sweep_records:
            if r["delay_us"] > 0.0 and pred_fn(r):
                return r["delay_us"]
        return None

    threshold_summary = {
        "min_significant_delay_us": _first_delay(lambda r: r["full_8feat"]["best_perm_pvalue"] < 0.01),
        "min_5pt_margin_full_8d_us": _first_delay(lambda r: r["full_8feat"]["best_delta_vs_baseline"] >= 0.05),
        "min_10pt_margin_full_8d_us": _first_delay(lambda r: r["full_8feat"]["best_delta_vs_baseline"] >= 0.10),
        "min_80pct_acc_full_8d_us": _first_delay(lambda r: r["full_8feat"]["best_accuracy"] >= 0.80),
        "min_5pt_margin_decrypt_4d_us": _first_delay(lambda r: r["decrypt_4feat"]["best_delta_vs_baseline"] >= 0.05),
        "min_80pct_acc_decrypt_4d_us": _first_delay(lambda r: r["decrypt_4feat"]["best_accuracy"] >= 0.80),
        "baseline_jitter_iqr_us": sweep_records[0]["timing_summary"]["dec_iqr_y0_us"],
    }

    natural_full_pct = part1["natural_original_decrypt"]["full_8feat"]["best_accuracy"] * 100.0
    generate_threshold_plot(
        sweep_records,
        baseline_pct=baseline * 100.0,
        natural_full_pct=natural_full_pct,
        threshold_summary=threshold_summary,
        png_path=PLOT_PNG_PATH,
        svg_path=PLOT_SVG_PATH,
    )

    payload = {
        "environment": env,
        "experiment": {
            "n_trials": N_TRIALS,
            "n": N_PARAM,
            "k": K_PARAM,
            "fixed_plaintext": FIXED_PLAINTEXT,
            "seed": SEED,
            "cv": f"RepeatedStratifiedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS}, random_state={SEED})",
            "n_permutations": 200,
            "median_secret_weight": median_w,
            "class_counts": {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))},
            "majority_baseline": baseline,
            "keygen_features": KEYGEN_FEATURES,
            "decrypt_features": DECRYPT_FEATURES,
            "full_features": FULL_FEATURES,
            "graded_delays_us": GRADED_DELAYS_US,
        },
        "part1_clean_vs_leaky": part1,
        "part2_graded_delay_sweep": sweep_records,
        "threshold_summary": threshold_summary,
        "plots": {"png": PLOT_PNG_PATH, "svg": PLOT_SVG_PATH},
    }

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    # Also record compact summaries in attack_c_full_stats.json (preserving original environment)
    for tag, sub in [
        ("clean_baseline_full8d", part1["clean_baseline"]["full_8feat"]),
        ("clean_baseline_dec4d", part1["clean_baseline"]["decrypt_4feat"]),
        ("leaky_keygen_full8d", part1["leaky_keygen"]["full_8feat"]),
        ("leaky_decrypt_50us_full8d", part1["leaky_decrypt_50us"]["full_8feat"]),
        ("natural_original_decrypt_full8d", part1["natural_original_decrypt"]["full_8feat"]),
    ]:
        attack_c_stats.update_json_results(
            "results/attack_c_full_stats.json",
            f"extended_{tag}_best",
            {
                "best_model": sub["best_model"],
                **sub["models"][sub["best_model"]],
            },
        )
    for rec in sweep_records:
        d_str = f"{rec['delay_us']:g}us".replace(".", "p")
        sub = rec["full_8feat"]
        attack_c_stats.update_json_results(
            "results/attack_c_full_stats.json",
            f"extended_graded_{d_str}_full8d_best",
            {
                "delay_us": rec["delay_us"],
                "measured_delta_us": rec["timing_summary"]["dec_median_delta_us"],
                "best_model": sub["best_model"],
                **sub["models"][sub["best_model"]],
            },
        )

    print(f"\nThreshold summary: {json.dumps(threshold_summary, indent=2)}")
    print(f"Saved: {RESULTS_JSON_PATH}, {PLOT_PNG_PATH}, {PLOT_SVG_PATH}")


if __name__ == "__main__":
    run_extended_positive_control()
