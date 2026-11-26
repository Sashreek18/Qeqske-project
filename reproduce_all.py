"""
One-Command Reproduction Runner for QEQSKE Side-Channel & ML Evaluation
=======================================================================
Usage:
  python3 reproduce_all.py --mode fast   # Verifies tests, PoCs, cross-host comparison,
                                         # ML-KEM detrending, and finite integrity of all
                                         # committed results/*.json files (~1-2 min).
  python3 reproduce_all.py --mode full   # Re-runs every benchmark & ML pipeline end-to-end.
"""

import argparse
import glob
import json
import math
import subprocess
import sys
import time


def _run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, check=False)
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (exit {res.returncode}): {' '.join(cmd)}")
    print(f"  [OK in {dt:.1f}s]")


def _check_finite(obj, path=""):
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"Non-finite float at {path}: {obj}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_finite(v, f"{path}[{i}]")


def validate_results_jsons():
    print("\n$ validating all results/*.json files for syntax & finiteness")
    files = sorted(glob.glob("results/*.json"))
    for path in files:
        with open(path, "r") as f:
            data = json.load(f)
        _check_finite(data, path)
    print(f"  [OK: {len(files)} JSON files validated]")


def main():
    parser = argparse.ArgumentParser(description="One-command reproduction script for QEQSKE evaluation.")
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="fast: sanity checks + offline comparisons + JSON audit; full: regenerate all runs.",
    )
    args = parser.parse_args()

    t_all = time.perf_counter()
    py = sys.executable

    fast_cmds = [
        [py, "test_qeqske.py"],
        [py, "break_poc.py"],
        [py, "attack_c_decrypt_5000.py", "--compare", "results/decrypt_leak_5000.json", "results/decrypt_leak_5000_mac_arm64.json"],
        [py, "attack_c_mlkem_decrypt_5000.py", "--detrend-existing"],
    ]

    full_cmds = [
        [py, "test_qeqske.py"],
        [py, "break_poc.py"],
        [py, "attack_a_level1_full_stats.py"],
        [py, "attack_b_entropy_leakage.py"],
        [py, "attack_c_leaky_vs_clean.py"],
        [py, "attack_c_decrypt_5000.py"],
        [py, "attack_c_decrypt_hypothesis.py"],
        [py, "attack_sc_sender_receiver_traces.py"],
        [py, "stress_test_traces.py"],
        [py, "attack_c_mlkem_decrypt_5000.py", "--variant", "both", "--k", "2", "--n-trials", "5000"],
        [py, "attack_c_mlkem_decrypt_5000.py", "--detrend-existing"],
        [py, "attack_c_mlkem_decrypt_5000.py", "--positive-control", "--n-trials", "5000"],
    ]

    cmds = fast_cmds if args.mode == "fast" else full_cmds
    for cmd in cmds:
        _run(cmd)

    validate_results_jsons()
    print(f"\nAll checks finished in {time.perf_counter() - t_all:.1f}s (mode={args.mode}).")


if __name__ == "__main__":
    main()
