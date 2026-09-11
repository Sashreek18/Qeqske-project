"""
ATTACK C (Part 2): RAM / Memory Side-Channel Measurement
==========================================================
Research Question (from HCL paper proposal):
  "Can ML-assisted side-channel attacks infer secret information during
   CRYSTALS-KYBER key generation, encapsulation, or decapsulation?"

This module measures MEMORY ALLOCATION behaviour of key_gen/encrypt/decrypt
across many runs, with varying secret key values, to see if memory usage
correlates with secret data (a software-observable proxy for the kind of
information a real memory/cache side-channel attack would try to exploit).

Per Rajib's instruction: since we don't have hardware to collect real
power/cache traces, we use SOFTWARE PROXIES instead:
  - tracemalloc  -> traces every memory allocation made by Python during
                    a block of code, giving us peak memory and allocation
                    COUNT (how many objects were allocated) -- this stands
                    in for "how much memory activity did this secret-dependent
                    computation cause?"

This mirrors attack_c_timing_groundwork.py in structure and output format,
so results can sit side by side in the report.
"""

import tracemalloc
import numpy as np
from qrng_handler import QRNGSource
from qeqske_core import QEQSKERandom, key_gen, encrypt_it, decrypt_it


def _secret_properties(s):
    """Same secret-key summary stats used in the timing script, so all
    three side-channel legs (Time / RAM / Cache) can be compared directly."""
    s_flat = [val for row in s for val in row]
    secret_weight = sum(1 for v in s_flat if v != 0)  # "Hamming weight"-like measure
    secret_sum = sum(s_flat)
    return secret_weight, secret_sum


def measure_keygen_memory(numbers, n=4, k=2, num_trials=20, verbose=True):
    """
    Runs key_gen() repeatedly under tracemalloc, recording:
      - peak memory used during the call (bytes)
      - number of individual memory allocations (blocks) made
    alongside properties of the generated secret key, to test whether
    memory behaviour leaks info about the secret.
    """
    results = []
    cursor = 0

    for trial in range(num_trials):
        batch = numbers[cursor:cursor + 200]
        if len(batch) < 50:
            if verbose:
                print(f"  Stopping early at trial {trial}: ran out of QRNG numbers")
            break
        cursor += 200

        qrng = QRNGSource(numbers=batch)
        rng = QEQSKERandom(qrng)

        tracemalloc.start()
        keys = key_gen(rng, n=n, k=k)
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        num_blocks = sum(stat.count for stat in snapshot.statistics("lineno"))
        tracemalloc.stop()

        secret_weight, secret_sum = _secret_properties(keys["private_key"])

        results.append({
            "trial": trial,
            "peak_memory_bytes": peak,
            "current_memory_bytes": current,
            "num_allocations": num_blocks,
            "q": keys["q"],
            "secret_weight": secret_weight,
            "secret_sum": secret_sum,
            "qrng_numbers_used": qrng.total_consumed,
        })

    if verbose:
        _report(results, stage_name="key_gen()")

    return results


def measure_encrypt_memory(numbers, n=4, k=2, num_trials=20, message="Hi", verbose=True):
    """
    Same idea as measure_keygen_memory, but profiles encrypt_it() instead.
    Here the "secret" correlated against is Bob's ephemeral private key r
    (the closest analogue of a secret during encryption).
    """
    results = []
    cursor = 0

    for trial in range(num_trials):
        batch = numbers[cursor:cursor + 400]
        if len(batch) < 150:
            if verbose:
                print(f"  Stopping early at trial {trial}: ran out of QRNG numbers")
            break
        cursor += 400

        qrng = QRNGSource(numbers=batch)
        rng = QEQSKERandom(qrng)
        keys = key_gen(rng, n=n, k=k)

        tracemalloc.start()
        ct = encrypt_it(rng, message, keys["public_key_A"], keys["public_key_t"], keys["q"], n, k)
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        num_blocks = sum(stat.count for stat in snapshot.statistics("lineno"))
        tracemalloc.stop()

        secret_weight, secret_sum = _secret_properties(keys["private_key"])

        results.append({
            "trial": trial,
            "peak_memory_bytes": peak,
            "current_memory_bytes": current,
            "num_allocations": num_blocks,
            "num_bits_encrypted": ct["num_bits"],
            "secret_weight": secret_weight,
            "secret_sum": secret_sum,
            "qrng_numbers_used": qrng.total_consumed,
        })

    if verbose:
        _report(results, stage_name="encrypt_it()")

    return results


def _report(results, stage_name):
    peaks = [r["peak_memory_bytes"] for r in results]
    allocs = [r["num_allocations"] for r in results]
    weights = [r["secret_weight"] for r in results]

    print(f"\n{'=' * 60}")
    print(f"RAM/Memory Measurements: {stage_name} across {len(results)} trials")
    print(f"{'=' * 60}")
    print(f"  Mean peak memory   = {np.mean(peaks):.1f} bytes")
    print(f"  Std dev peak       = {np.std(peaks):.1f} bytes")
    print(f"  Min / Max peak     = {np.min(peaks)} / {np.max(peaks)} bytes")
    print(f"  Mean allocations   = {np.mean(allocs):.1f}")

    corr_peak = np.corrcoef(peaks, weights)[0, 1] if len(set(weights)) > 1 and len(set(peaks)) > 1 else 0.0
    corr_alloc = np.corrcoef(allocs, weights)[0, 1] if len(set(weights)) > 1 and len(set(allocs)) > 1 else 0.0

    print(f"\n  Correlation(peak_memory, secret_weight)  = {corr_peak:.4f}")
    print(f"  Correlation(num_allocations, secret_weight) = {corr_alloc:.4f}")

    if max(abs(corr_peak), abs(corr_alloc)) > 0.3:
        print("  -> MEANINGFUL correlation detected — potential memory side-channel!")
    else:
        print("  -> Weak/no correlation — memory allocation pattern doesn't obviously")
        print("     leak secret weight in this software implementation.")

    print(f"\n  NOTE: tracemalloc measures Python-level allocations as a SOFTWARE PROXY")
    print(f"  for real memory/cache side-channels. A hardware cache-timing attack")
    print(f"  observes CPU cache lines directly; this is the closest software-only")
    print(f"  approximation available without specialized profiling hardware.")


if __name__ == "__main__":
    with open("sample_qrng_batch.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("\n" + "#" * 70)
    print("ATTACK C (Part 2): RAM Side-Channel — key_gen()")
    print("#" * 70)
    measure_keygen_memory(numbers, n=4, k=2, num_trials=5)

    print("\n" + "#" * 70)
    print("ATTACK C (Part 2): RAM Side-Channel — encrypt_it()")
    print("#" * 70)
    measure_encrypt_memory(numbers, n=4, k=2, num_trials=5)
