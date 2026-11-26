"""
Audit Evidence Generator (Items B1, B2, B3)
=========================================
Regenerates and persists the three previously unsourced empirical claims into
results/ so every quantitative claim in README.md and PAPER_OUTLINE.md is
backed by a committed results/*.json artifact:

  1. results/tier1c_zero_column_analysis.json
     Replays the 5,000 Alice<->Bob exchanges matching
     results/sender_receiver_traces_5000.json, counts degenerate keys with an
     all-zero column in s (119/5,000 = 2.38%), and compares channel-feature
     Pearson correlations before vs. after removing those keys.

  2. results/break_poc_and_algebraic_lwe.json
     Runs break_poc.py (Break 1 + Break 2 over 200 trials) and
     attack_e_algebraic_lwe.py recovery (300 trials), recording exact recovery
     rates and per-key timings.

  3. results/reproduce_all_fast_log.json
     Executes `reproduce_all.py --mode fast`, recording per-step timings,
     total runtime, validated JSON count, and captured stdout.
"""

import contextlib
import io
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

import attack_e_algebraic_lwe
import break_poc
from env_metadata import get_environment_metadata
from qeqske_core import QEQSKERandom, decrypt_it, encrypt_it, key_gen
from qrng_handler import QRNGSource
from reproduce_all import _check_finite


RESULTS_DIR = Path("results")


def generate_tier1c_zero_column_analysis() -> Path:
    """Verify 119/5,000 all-zero-column keys and channel correlations (B1)."""
    traces_path = RESULTS_DIR / "sender_receiver_traces_5000.json"
    with traces_path.open("r", encoding="utf-8") as f:
        sr5 = json.load(f)
    recs = sr5["records"]

    src = QRNGSource.from_file("qrng_combined.txt", cycle=True)
    rng = QEQSKERandom(src)

    zero_col_flags = []
    mismatches = 0
    for rec in recs:
        kd = key_gen(rng, 4, 2)
        s = kd["private_key"]
        q = kd["q"]
        encrypt_it(rng, "HELLO", kd["public_key_A"], kd["public_key_t"], q, 4, 2)
        sw = sum(1 for row in s for v in row if v != 0)
        if q != rec["q"] or sw != rec["secret_weight"]:
            mismatches += 1
        has_zero_col = any(all(s[row][c] == 0 for row in range(4)) for c in range(2))
        zero_col_flags.append(has_zero_col)

    if mismatches != 0:
        raise RuntimeError(f"Found {mismatches} trace mismatches while replaying {traces_path}")

    zc = np.array(zero_col_flags, dtype=bool)
    sw_arr = np.array([rec["secret_weight"] for rec in recs], dtype=float)
    zc_count = int(zc.sum())
    if zc_count != 119:
        raise RuntimeError(f"Expected 119 zero-column keys, got {zc_count}")

    features = [
        "ct_entropy_approx",
        "ct_coefficient_of_variation",
        "v_mean_over_q",
        "v_kurtosis",
        "v_skewness",
    ]
    feature_correlations = {}
    for feat in features:
        vals = np.array([rec[feat] for rec in recs], dtype=float)
        r_all, p_all = pearsonr(vals, sw_arr)
        r_nz, p_nz = pearsonr(vals[~zc], sw_arr[~zc])
        feature_correlations[feat] = {
            "all_trials": {"pearson_r": float(r_all), "pearson_p": float(p_all)},
            "excluding_zero_col_keys": {"pearson_r": float(r_nz), "pearson_p": float(p_nz)},
        }

    out = {
        "environment": get_environment_metadata(),
        "source_traces_file": str(traces_path),
        "n_trials": len(recs),
        "verified_trace_mismatches": mismatches,
        "zero_column_key_count": zc_count,
        "zero_column_key_pct": round(100.0 * zc_count / len(recs), 4),
        "non_zero_column_key_count": int((~zc).sum()),
        "zero_column_max_secret_weight": int(sw_arr[zc].max()),
        "feature_correlations_vs_secret_weight": feature_correlations,
    }
    _check_finite(out, "tier1c_zero_column_analysis")

    out_path = RESULTS_DIR / "tier1c_zero_column_analysis.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out_path


def generate_break_poc_and_algebraic_lwe() -> Path:
    """Record break_poc.py and attack_e_algebraic_lwe.py recovery metrics (B2)."""
    src_bp = QRNGSource.from_file("qrng_combined.txt", cycle=True)
    rng_bp = QEQSKERandom(src_bp)
    with contextlib.redirect_stdout(io.StringIO()):
        hit_vuln, hit_fixed = break_poc.demo_break_1(rng_bp)
        rec_bp, tot_bp, elapsed_bp = break_poc.demo_break_2(rng_bp, trials=200)

    src_alg = QRNGSource.from_file("qrng_combined.txt", cycle=True)
    rng_alg = QEQSKERandom(src_alg)
    alg_trials = attack_e_algebraic_lwe.TRIALS
    n = attack_e_algebraic_lwe.N
    k = attack_e_algebraic_lwe.K
    msg = attack_e_algebraic_lwe.MESSAGE

    alg_recovered = 0
    alg_decrypt_verified = 0
    alg_timings_ms = []
    alg_cands = []
    t_alg_start = time.perf_counter()
    for _ in range(alg_trials):
        kd = key_gen(rng_alg, n, k)
        A, t, q, s = kd["public_key_A"], kd["public_key_t"], kd["q"], kd["private_key"]
        t0 = time.perf_counter()
        s_hat, ncands = attack_e_algebraic_lwe.recover_full_secret(A, t, q, n, k, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        alg_timings_ms.append(float(elapsed_ms))
        alg_cands.append(int(ncands))
        if s_hat is not None and attack_e_algebraic_lwe.keys_match(s, s_hat, n, k):
            alg_recovered += 1
            ct = encrypt_it(rng_alg, msg, A, t, q, n, k)
            if decrypt_it(s_hat, ct["u"], ct["v_queue"], q) == msg:
                alg_decrypt_verified += 1
    alg_total_s = time.perf_counter() - t_alg_start

    out = {
        "environment": get_environment_metadata(),
        "break_poc": {
            "break_1_vulnerable_recovered": bool(hit_vuln),
            "break_1_fixed_recovered": bool(hit_fixed),
            "break_2_recovered": int(rec_bp),
            "break_2_trials": int(tot_bp),
            "break_2_recovery_rate": float(rec_bp / tot_bp),
            "break_2_total_seconds": float(elapsed_bp),
            "break_2_ms_per_key": float(1000.0 * elapsed_bp / tot_bp),
        },
        "attack_e_algebraic_lwe": {
            "recovered": int(alg_recovered),
            "trials": int(alg_trials),
            "recovery_rate": float(alg_recovered / alg_trials),
            "decrypt_verified": int(alg_decrypt_verified),
            "total_seconds": float(alg_total_s),
            "mean_ms_per_key": float(statistics.mean(alg_timings_ms)),
            "median_ms_per_key": float(statistics.median(alg_timings_ms)),
            "max_ms_per_key": float(max(alg_timings_ms)),
            "mean_candidates_checked": float(statistics.mean(alg_cands)),
        },
    }
    _check_finite(out, "break_poc_and_algebraic_lwe")

    out_path = RESULTS_DIR / "break_poc_and_algebraic_lwe.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out_path


def generate_reproduce_all_fast_log() -> Path:
    """Run reproduce_all.py --mode fast and persist structured log + timings (B3)."""
    out_path = RESULTS_DIR / "reproduce_all_fast_log.json"
    # Pre-seed a valid finite JSON placeholder so reproduce_all's glob count includes this file.
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"status": "running"}, f)

    cmd = [sys.executable, "reproduce_all.py", "--mode", "fast"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"reproduce_all.py --mode fast failed (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    steps = []
    current_cmd = None
    for line in proc.stdout.splitlines():
        if line.startswith("$ ") and not line.startswith("$ validating"):
            current_cmd = line[2:].strip()
        elif current_cmd is not None:
            m = re.search(r"\[OK in ([0-9.]+)s\]", line)
            if m:
                steps.append({"command": current_cmd, "seconds": float(m.group(1))})
                current_cmd = None

    val_match = re.search(r"\[OK:\s*(\d+)\s*JSON files validated\]", proc.stdout)
    validated_count = int(val_match.group(1)) if val_match else 0

    tot_match = re.search(r"All checks finished in ([0-9.]+)s", proc.stdout)
    reported_runtime_s = float(tot_match.group(1)) if tot_match else float(wall_s)

    out = {
        "environment": get_environment_metadata(),
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "total_wall_seconds": float(wall_s),
        "reported_runtime_seconds": reported_runtime_s,
        "validated_json_count": validated_count,
        "steps": steps,
        "stdout": proc.stdout,
    }
    _check_finite(out, "reproduce_all_fast_log")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out_path


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p1 = generate_tier1c_zero_column_analysis()
    print(f"Wrote {p1}")
    p2 = generate_break_poc_and_algebraic_lwe()
    print(f"Wrote {p2}")
    p3 = generate_reproduce_all_fast_log()
    print(f"Wrote {p3}")


if __name__ == "__main__":
    main()
