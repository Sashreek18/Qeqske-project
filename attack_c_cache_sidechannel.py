"""
ATTACK C (Part 3): Cache / Power Side-Channel Measurement (software proxy)
============================================================================
Research Question (from HCL paper proposal):
  "Can ML-assisted side-channel attacks infer secret information during
   CRYSTALS-KYBER key generation, encapsulation, or decapsulation?"

Real cache-timing and power-analysis attacks need hardware we don't have
(cache-miss counters, an oscilloscope on the device). Per Rajib's
instruction, we approximate this in software using `psutil`:

  - CPU utilisation (%) sampled tightly around the operation is used as a
    PROXY for power draw (more computation/branching -> more CPU load).
  - CPU *times* (user/system seconds actually spent by THIS process,
    from /proc via psutil) is a more precise, low-noise proxy that isolates
    our process from other system activity -- this is the number we
    correlate against the secret key.

This is not a real cache attack (no L1/L2/L3 hit/miss data is available
in pure Python), but it gives a measurable, repeatable software signal in
the same spirit: "does resource-usage behaviour during the secret-dependent
computation leak information about the secret?"
"""

import psutil
import os
import numpy as np
from scipy import stats as sp_stats
import attack_c_stats
from qrng_handler import QRNGSource
from qeqske_core import QEQSKERandom, key_gen, encrypt_it, decrypt_it


def _secret_properties(s):
    s_flat = [val for row in s for val in row]
    secret_weight = sum(1 for v in s_flat if v != 0)
    secret_sum = sum(s_flat)
    return secret_weight, secret_sum


def measure_keygen_cpu(numbers, n=4, k=2, num_trials=20, repeats_per_trial=500, verbose=True):
    """
    Runs key_gen() repeatedly, sampling this process's CPU time
    (psutil.Process().cpu_times()) immediately before and after each call,
    as a software proxy for power/cache activity.

    NOTE ON RESOLUTION: with small toy parameters (n=4, k=2), a single
    key_gen() call finishes in microseconds -- below the OS's CPU-time
    tick resolution (~1ms on Linux), so a single-call measurement reads
    as 0.0000 every time. To get a measurable signal we re-run key_gen()
    `repeats_per_trial` times back-to-back using the SAME secret key's
    randomness cursor position, and measure the accumulated CPU time for
    the whole batch. This is standard practice in software side-channel
    work when operations are faster than the timer/clock resolution.
    """
    proc = psutil.Process(os.getpid())
    results = []
    cursor = 0
    needed = repeats_per_trial * 25  # rough upper bound on numbers per key_gen call

    for trial in range(num_trials):
        batch = numbers[cursor:cursor + needed]
        if len(batch) < 100:
            if verbose:
                print(f"  Stopping early at trial {trial}: ran out of QRNG numbers")
            break

        qrng = QRNGSource(numbers=list(batch), generator_fn=lambda: list(batch))
        rng = QEQSKERandom(qrng)

        # Keep the FIRST key_gen's secret key as the one we correlate against,
        # then repeat the same call to accumulate measurable CPU time.
        proc.cpu_percent(interval=None)
        cpu_before = proc.cpu_times()

        first_keys = None
        reps_done = 0
        for _ in range(repeats_per_trial):
            try:
                keys = key_gen(rng, n=n, k=k)
            except RuntimeError:
                break
            if first_keys is None:
                first_keys = keys
            reps_done += 1

        cpu_after = proc.cpu_times()
        cpu_percent_sample = proc.cpu_percent(interval=None)

        cursor += qrng.total_consumed

        if first_keys is None or reps_done == 0:
            if verbose:
                print(f"  Stopping early at trial {trial}: ran out of QRNG numbers")
            break

        user_delta = cpu_after.user - cpu_before.user
        system_delta = cpu_after.system - cpu_before.system

        secret_weight, secret_sum = _secret_properties(first_keys["private_key"])

        results.append({
            "trial": trial,
            "cpu_user_seconds": user_delta,
            "cpu_system_seconds": system_delta,
            "cpu_user_seconds_per_call": user_delta / reps_done,
            "cpu_percent_sample": cpu_percent_sample,
            "reps_done": reps_done,
            "q": first_keys["q"],
            "secret_weight": secret_weight,
            "secret_sum": secret_sum,
            "qrng_numbers_used": qrng.total_consumed,
        })

    if verbose:
        _report(results, stage_name="key_gen()")

    return results


def measure_encrypt_cpu(numbers, n=4, k=2, num_trials=20, message="Hi", repeats_per_trial=100, verbose=True):
    """
    Same idea as measure_keygen_cpu, but profiles encrypt_it(). Encryption
    does more work per call (one matrix op per bit), so fewer repeats are
    needed to accumulate a measurable CPU-time sample.
    """
    proc = psutil.Process(os.getpid())
    results = []
    cursor = 0
    needed = repeats_per_trial * 420

    for trial in range(num_trials):
        batch = numbers[cursor:cursor + needed]
        if len(batch) < 450:
            if verbose:
                print(f"  Stopping early at trial {trial}: ran out of QRNG numbers")
            break

        qrng = QRNGSource(numbers=list(batch), generator_fn=lambda: list(batch))
        rng = QEQSKERandom(qrng)
        first_keys = key_gen(rng, n=n, k=k)

        proc.cpu_percent(interval=None)
        cpu_before = proc.cpu_times()

        reps_done = 0
        last_ct = None
        for _ in range(repeats_per_trial):
            try:
                last_ct = encrypt_it(rng, message, first_keys["public_key_A"],
                                      first_keys["public_key_t"], first_keys["q"], n, k)
            except RuntimeError:
                break
            reps_done += 1

        cpu_after = proc.cpu_times()
        cpu_percent_sample = proc.cpu_percent(interval=None)

        cursor += qrng.total_consumed

        if reps_done == 0:
            if verbose:
                print(f"  Stopping early at trial {trial}: ran out of QRNG numbers")
            break

        user_delta = cpu_after.user - cpu_before.user
        system_delta = cpu_after.system - cpu_before.system

        secret_weight, secret_sum = _secret_properties(first_keys["private_key"])

        results.append({
            "trial": trial,
            "cpu_user_seconds": user_delta,
            "cpu_system_seconds": system_delta,
            "cpu_user_seconds_per_call": user_delta / reps_done,
            "cpu_percent_sample": cpu_percent_sample,
            "reps_done": reps_done,
            "num_bits_encrypted": last_ct["num_bits"] if last_ct else 0,
            "secret_weight": secret_weight,
            "secret_sum": secret_sum,
            "qrng_numbers_used": qrng.total_consumed,
        })

    if verbose:
        _report(results, stage_name="encrypt_it()")

    return results


def _report(results, stage_name):
    user_times = [r["cpu_user_seconds_per_call"] for r in results]
    sys_times = [r["cpu_system_seconds"] for r in results]
    weights = [r["secret_weight"] for r in results]

    print(f"\n{'=' * 60}")
    print(f"Cache/Power Proxy Measurements: {stage_name} across {len(results)} trials")
    print(f"(each trial = {results[0]['reps_done']} repeated calls, to get above clock resolution)")
    print(f"{'=' * 60}")
    print(f"  Mean CPU user time (per call) = {np.mean(user_times)*1e6:.2f} µs")
    print(f"  Std dev user time (per call)  = {np.std(user_times)*1e6:.2f} µs")
    print(f"  Mean CPU system time (batch)  = {np.mean(sys_times)*1000:.4f} ms")

    corr_user = np.corrcoef(user_times, weights)[0, 1] if len(set(weights)) > 1 and len(set(user_times)) > 1 else 0.0
    corr_sys = np.corrcoef(sys_times, weights)[0, 1] if len(set(weights)) > 1 and len(set(sys_times)) > 1 else 0.0

    print(f"\n  Correlation(cpu_user_time, secret_weight)   = {corr_user:.4f}")
    print(f"  Correlation(cpu_system_time, secret_weight) = {corr_sys:.4f}")

    if max(abs(corr_user), abs(corr_sys)) > 0.3:
        print("  -> MEANINGFUL correlation detected — potential cache/power side-channel!")
    else:
        print("  -> Weak/no correlation — CPU usage pattern doesn't obviously leak")
        print("     secret weight in this software implementation.")

    print(f"\n  NOTE: psutil CPU time is a SOFTWARE PROXY for power/cache behaviour.")
    print(f"  Real cache-timing attacks (e.g. Flush+Reload) need hardware performance")
    print(f"  counters (perf_event_open on Linux) or an oscilloscope for power traces.")
    print(f"  This proxy captures the same underlying idea -- does secret-dependent")
    print(f"  branching/looping cause measurably different resource usage -- using")
    print(f"  only what's available in a standard Python environment.")


def _summarize_cpu(results):
    user_times = [r["cpu_user_seconds_per_call"] for r in results]
    sys_times = [r["cpu_system_seconds"] for r in results]
    weights = [r["secret_weight"] for r in results]
    if len(set(weights)) > 1 and len(set(user_times)) > 1:
        r_user, p_user = sp_stats.pearsonr(user_times, weights)
    else:
        r_user, p_user = 0.0, 1.0
    if len(set(weights)) > 1 and len(set(sys_times)) > 1:
        r_sys, p_sys = sp_stats.pearsonr(sys_times, weights)
    else:
        r_sys, p_sys = 0.0, 1.0
    return {
        "n_trials": int(len(results)),
        "reps_per_trial": int(results[0]["reps_done"]) if results else 0,
        "mean_user_us_per_call": float(np.mean(user_times) * 1e6),
        "std_user_us_per_call": float(np.std(user_times) * 1e6),
        "pearson_user_r": float(r_user),
        "pearson_user_p": float(p_user),
        "pearson_sys_r": float(r_sys),
        "pearson_sys_p": float(p_sys),
    }


if __name__ == "__main__":
    dataset_path = "qrng_combined.txt" if os.path.exists("qrng_combined.txt") else "sample_qrng_batch.txt"
    with open(dataset_path) as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("\n" + "#" * 70)
    print(f"ATTACK C (Part 3): Cache/Power Proxy — key_gen() [{dataset_path}]")
    print("#" * 70)
    kg_res = measure_keygen_cpu(numbers, n=4, k=2, num_trials=20, repeats_per_trial=500)

    print("\n" + "#" * 70)
    print(f"ATTACK C (Part 3): Cache/Power Proxy — encrypt_it() [{dataset_path}]")
    print("#" * 70)
    enc_res = measure_encrypt_cpu(numbers, n=4, k=2, num_trials=20, repeats_per_trial=100)

    attack_c_stats.update_json_results(
        "results/attack_c_ram_cache.json",
        "cache_sidechannel",
        {
            "dataset": dataset_path,
            "key_gen": _summarize_cpu(kg_res),
            "encrypt_it": _summarize_cpu(enc_res),
        },
    )
