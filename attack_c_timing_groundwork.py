"""
ATTACK C (Groundwork): Timing Side-Channel Measurement
==========================================================
Research Question (from HCL paper proposal):
  "Can ML-assisted side-channel attacks infer secret information during
   CRYSTALS-KYBER key generation, encapsulation, or decapsulation?"

This module measures EXECUTION TIME of key_gen/encrypt/decrypt operations
across many runs, with varying secret key values, to see if timing
correlates with secret data (a classic side-channel vulnerability).

IMPORTANT — what real side-channel attacks need (per proposal Phase 2):
  - Power consumption traces  -> needs hardware (oscilloscope on real device)
  - Cache activity traces     -> needs specialized profiling tools
  - Timing traces             -> CAN do this in software (what we build here)

So this script gives you the SOFTWARE-MEASURABLE part of Attack C now.
Power/cache traces are the part to ask HCL about (as discussed earlier).
"""

import time
import numpy as np
from qrng_handler import QRNGSource
from qeqske_core import QEQSKERandom, key_gen, encrypt_it, decrypt_it


def measure_keygen_timing(numbers, n=4, k=2, num_trials=20, verbose=True):
    """
    Runs key_gen() repeatedly, measuring wall-clock time each run.
    Records timing alongside properties of the generated secret key,
    to later test: does timing leak info about the secret?
    """
    results = []
    cursor = 0

    for trial in range(num_trials):
        # Use a fresh slice of QRNG numbers for each trial
        batch = numbers[cursor:cursor + 200]
        if len(batch) < 50:
            if verbose:
                print(f"  Stopping early at trial {trial}: ran out of QRNG numbers")
            break
        cursor += 200

        qrng = QRNGSource(numbers=batch)
        rng = QEQSKERandom(qrng)

        start = time.perf_counter()
        keys = key_gen(rng, n=n, k=k)
        elapsed = time.perf_counter() - start

        # Compute simple properties of the secret key (for correlation analysis)
        s_flat = [val for row in keys["private_key"] for val in row]
        secret_weight = sum(1 for v in s_flat if v != 0)  # "Hamming weight"-like measure
        secret_sum = sum(s_flat)

        results.append({
            "trial": trial,
            "elapsed_seconds": elapsed,
            "q": keys["q"],
            "secret_weight": secret_weight,
            "secret_sum": secret_sum,
            "qrng_numbers_used": qrng.total_consumed,
        })

    if verbose:
        times = [r["elapsed_seconds"] for r in results]
        print(f"\n{'=' * 60}")
        print(f"Timing Measurements: key_gen() across {len(results)} trials")
        print(f"{'=' * 60}")
        print(f"  Mean time   = {np.mean(times)*1000:.4f} ms")
        print(f"  Std dev     = {np.std(times)*1000:.4f} ms")
        print(f"  Min / Max   = {np.min(times)*1000:.4f} / {np.max(times)*1000:.4f} ms")

        weights = [r["secret_weight"] for r in results]
        correlation = np.corrcoef(times, weights)[0, 1] if len(set(weights)) > 1 else 0.0
        print(f"\n  Correlation(timing, secret_weight) = {correlation:.4f}")
        if abs(correlation) > 0.3:
            print("  -> MEANINGFUL correlation detected — potential timing side-channel!")
        else:
            print("  -> Weak/no correlation — timing doesn't obviously leak secret weight "
                  "in this software implementation.")
        print(f"\n  NOTE: Real side-channel attacks (per proposal) also need power and")
        print(f"  cache traces from actual hardware execution — ask HCL whether they")
        print(f"  can provide an oscilloscope/profiler setup or pre-collected traces.")

    return results


if __name__ == "__main__":
    with open("sample_qrng_batch.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    measure_keygen_timing(numbers, n=4, k=2, num_trials=5)
