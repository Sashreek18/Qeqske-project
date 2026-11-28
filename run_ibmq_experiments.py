#!/usr/bin/env python3
"""
HCL IBMQ Multi-QRNG Side-Channel Experiments (Apple M2 mac_arm64)
=================================================================
Runs the 5,000-trial timing side-channel pipeline using HCL Software's
IBMQ Superconducting QRNG stream (ibmq_qrng_uint16.txt) across:
  1. Toy QEQSKE leaky decrypt_it()        -> results/decrypt_leak_5000_ibmq_mac_arm64.json
  2. Toy QEQSKE constant-time NumPy CT    -> results/decrypt_ct_leak_5000_ibmq_mac_arm64.json
  3. ML-KEM-512 QEQSKE-Variant (wraps=0)  -> results/mlkem512_qeqske_ibmq_mac_arm64_decrypt_leak_5000.json
"""

import time
import attack_c_decrypt_5000 as dec5k
import attack_c_mlkem_decrypt_5000 as mlk5k
from qeqske_core import decrypt_it, decrypt_it_constant_time


def main():
    t0 = time.perf_counter()
    ibmq_path = "ibmq_qrng_uint16.txt"

    print("=== 1/3 Pre-generating 5,000 Toy Trials from IBMQ QRNG ===")
    pre = dec5k.pregenerate_trials(qrng_path=ibmq_path, seed=42)

    print("=== 2/3 Running Leaky Toy decrypt_it() on IBMQ QRNG ===")
    dec5k.run_decrypt_leak_pipeline(
        decrypt_fn=decrypt_it,
        traces_path="results/decrypt_traces_5000_ibmq_mac_arm64.json",
        results_path="results/decrypt_leak_5000_ibmq_mac_arm64.json",
        summary_path="results/decrypt_leak_5000_ibmq_mac_arm64_summary.md",
        label="decrypt_it (original, IBMQ QRNG, mac_arm64)",
        pregenerated=pre,
        seed=42,
    )

    print("=== 3/3 Running Constant-Time NumPy decrypt_it_constant_time() on IBMQ QRNG ===")
    dec5k.run_decrypt_leak_pipeline(
        decrypt_fn=decrypt_it_constant_time,
        traces_path="results/decrypt_ct_traces_5000_ibmq_mac_arm64.json",
        results_path="results/decrypt_ct_leak_5000_ibmq_mac_arm64.json",
        summary_path="results/decrypt_ct_leak_5000_ibmq_mac_arm64_summary.md",
        label="decrypt_it_constant_time (NumPy CT, IBMQ QRNG, mac_arm64)",
        pregenerated=pre,
        seed=42,
    )

    del pre

    print("=== 4/4 Running ML-KEM-512 QEQSKE-Variant (5,000 trials, IBMQ QRNG) ===")
    mlk_res = mlk5k.run_mlkem_pipeline(
        variant="qeqske",
        k_param=2,
        n_trials=5000,
        seed=42,
        skip_tracemalloc=False,
        tag="ibmq_mac_arm64",
        qrng_path=ibmq_path,
    )
    wraps = mlk_res["stream"]["wraps"]
    consumed = mlk_res["stream"]["total_consumed_uint16"]
    assert wraps == 0, f"Expected wraps==0 on IBMQ ML-KEM-512, got {wraps}"
    print(f"Verified IBMQ ML-KEM-512 stream: {consumed:,} uint16 draws, wraps={wraps}")
    print(f"Total run_ibmq_experiments.py elapsed: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
