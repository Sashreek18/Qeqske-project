"""
ATTACK SC-1: Sender-to-Receiver Communication Trace Collection
=================================================================
Implements the full Alice↔Bob QEQSKE key exchange and captures traces
at BOTH endpoints and in the communication channel.

RESEARCH QUESTION (per Prof. Rajib Mall):
  "Can ML-based side-channel attacks detect any information leakage
   from sender-to-receiver communication traces during key exchange?"

TRACE MODEL:
  The QEQSKE key exchange proceeds as:
    1. Alice: key_gen() → (s, A, t, q)       [Alice's side]
    2. Alice → Bob: public keys (A, t, q)     [public channel]
    3. Bob: encrypt_it(msg, A, t, q) → ct     [Bob's side]
    4. Bob → Alice: ciphertext (u, v_queue)   [public channel]
    5. Alice: decrypt_it(s, u, v_queue, q)    [Alice's side]

  Traces captured at each stage:
    SENDER (Bob) traces:
      - Encryption timing (µs)
      - Encryption CPU user-time (µs)
      - Encryption peak RAM (bytes)
      - Encryption allocation count

    RECEIVER (Alice) traces:
      - Decryption timing (µs)
      - Decryption CPU user-time (µs)
      - Decryption peak RAM (bytes)
      - Decryption allocation count

    COMMUNICATION CHANNEL traces (what an eavesdropper sees):
      - Ciphertext size (total elements in u + v_queue)
      - u matrix: Frobenius norm, mean, std, min, max
      - v_queue: mean norm across blocks, std of norms
      - q value (public modulus)
      - Number of ciphertext blocks (= message length in bits)
      - Statistical moments of flattened ciphertext coefficients

  Labels (what we try to predict — secret information):
      - Secret key weight (non-zero entries in s)
      - Secret key sum
      - Secret key L2 norm

THIS IS THE CORE OF THE PROJECT:
  Using QEQSKE as the base protocol, we collect multi-dimensional traces
  from sender, receiver, and the communication channel, then use ML models
  to detect if any of these traces leak information about the secret key.
"""

import time
import tracemalloc
import os
import json
import numpy as np
import psutil
from scipy import stats

from qrng_handler import QRNGSource
from qeqske_core import QEQSKERandom, key_gen, encrypt_it, decrypt_it


# ---------------------------------------------------------------------------
# Secret key property extraction
# ---------------------------------------------------------------------------

def extract_secret_properties(s):
    """Extract multiple properties of the secret key for labeling."""
    s_flat = [v for row in s for v in row]
    return {
        "secret_weight": sum(1 for v in s_flat if v != 0),
        "secret_sum": sum(s_flat),
        "secret_l2_norm": float(np.sqrt(sum(v**2 for v in s_flat))),
        "secret_max_abs": max(abs(v) for v in s_flat),
    }


# ---------------------------------------------------------------------------
# Ciphertext (communication channel) feature extraction
# ---------------------------------------------------------------------------

def extract_channel_features(ct, q):
    """
    Extract observable features from the ciphertext — what an eavesdropper
    on the communication channel between Bob and Alice can see.
    """
    u = ct["u"]
    v_queue = ct["v_queue"]
    per_bit = isinstance(u, list) and len(u) > 0 and isinstance(u[0], list) \
        and len(u[0]) > 0 and isinstance(u[0][0], list)

    features = {}
    features["q"] = q
    features["num_ct_blocks"] = len(v_queue)

    # --- u matrix features ---
    if per_bit:
        # Per-bit u: list of matrices, take statistics across all
        all_u_vals = []
        u_norms = []
        for u_i in u:
            u_flat = [v for row in u_i for v in row]
            all_u_vals.extend(u_flat)
            u_norms.append(np.sqrt(sum(v**2 for v in u_flat)))
        features["u_frobenius_mean"] = float(np.mean(u_norms))
        features["u_frobenius_std"] = float(np.std(u_norms))
    else:
        # Single u matrix
        u_flat = [v for row in u for v in row]
        all_u_vals = u_flat
        features["u_frobenius_mean"] = float(np.sqrt(sum(v**2 for v in u_flat)))
        features["u_frobenius_std"] = 0.0

    u_arr = np.array(all_u_vals, dtype=float)
    features["u_mean"] = float(np.mean(u_arr))
    features["u_std"] = float(np.std(u_arr))
    features["u_min"] = float(np.min(u_arr))
    features["u_max"] = float(np.max(u_arr))
    features["u_median"] = float(np.median(u_arr))
    features["u_skewness"] = float(stats.skew(u_arr))
    features["u_kurtosis"] = float(stats.kurtosis(u_arr))

    # --- v_queue features ---
    v_norms = []
    all_v_vals = []
    for v_block in v_queue:
        v_flat = [v for row in v_block for v in row]
        all_v_vals.extend(v_flat)
        v_norms.append(np.sqrt(sum(v**2 for v in v_flat)))

    features["v_norm_mean"] = float(np.mean(v_norms))
    features["v_norm_std"] = float(np.std(v_norms))

    v_arr = np.array(all_v_vals, dtype=float)
    features["v_mean"] = float(np.mean(v_arr))
    features["v_std"] = float(np.std(v_arr))
    features["v_min"] = float(np.min(v_arr))
    features["v_max"] = float(np.max(v_arr))
    features["v_median"] = float(np.median(v_arr))
    features["v_skewness"] = float(stats.skew(v_arr))
    features["v_kurtosis"] = float(stats.kurtosis(v_arr))

    # --- Combined ciphertext statistics ---
    combined = np.concatenate([u_arr, v_arr])
    features["ct_total_elements"] = len(combined)
    features["ct_mean"] = float(np.mean(combined))
    features["ct_std"] = float(np.std(combined))
    features["ct_entropy_approx"] = float(
        stats.entropy(np.histogram(combined, bins=50, density=True)[0] + 1e-10)
    )

    # Ratio features (normalized by q)
    features["u_mean_over_q"] = features["u_mean"] / max(q, 1)
    features["v_mean_over_q"] = features["v_mean"] / max(q, 1)
    features["ct_coefficient_of_variation"] = (
        features["ct_std"] / max(abs(features["ct_mean"]), 1e-10)
    )

    return features


# ---------------------------------------------------------------------------
# Full key exchange simulation with trace collection
# ---------------------------------------------------------------------------

FIXED_PLAINTEXT = "HELLO"


def simulate_key_exchange_with_traces(numbers, n=4, k=2, num_trials=500,
                                       verbose=True):
    """
    Simulate num_trials full Alice↔Bob key exchanges.

    For each exchange, capture:
      - Bob's encryption side-channel traces
      - Alice's decryption side-channel traces
      - Communication channel (ciphertext) features
      - Secret key properties (labels for ML)
    """
    proc = psutil.Process(os.getpid())
    rows = []
    cursor = 0

    for trial in range(num_trials):
        # ~981 QRNG numbers per full exchange (key_gen ~21 + encrypt_it ~960)
        batch = numbers[cursor:cursor + 2000]
        if len(batch) < 1000:
            if verbose:
                print(f"  Stopping early at trial {trial}: ran out of QRNG numbers")
            break

        qrng = QRNGSource(numbers=list(batch))
        rng = QEQSKERandom(qrng)

        # ========= ALICE: Key Generation =========
        keys = key_gen(rng, n=n, k=k)
        s = keys["private_key"]
        A = keys["public_key_A"]
        t = keys["public_key_t"]
        q = keys["q"]

        # Alice sends (A, t, q) to Bob over public channel
        # (We don't profile key_gen here — it's already done in attack_c_combined_ml.py)

        # ========= BOB: Encryption (sender side-channel) =========
        tracemalloc.start()
        cpu_before_enc = proc.cpu_times()
        t0_enc = time.perf_counter()

        ct = encrypt_it(rng, FIXED_PLAINTEXT, A, t, q, n, k)

        elapsed_enc = time.perf_counter() - t0_enc
        cpu_after_enc = proc.cpu_times()
        _, peak_enc = tracemalloc.get_traced_memory()
        snap_enc = tracemalloc.take_snapshot()
        allocs_enc = sum(stat.count for stat in snap_enc.statistics("lineno"))
        tracemalloc.stop()

        # Bob sends (u, v_queue) to Alice over public channel

        # ========= CHANNEL: Ciphertext features (eavesdropper) =========
        channel_features = extract_channel_features(ct, q)

        # ========= ALICE: Decryption (receiver side-channel) =========
        u = ct["u"]
        v_queue = ct["v_queue"]

        tracemalloc.start()
        cpu_before_dec = proc.cpu_times()
        t0_dec = time.perf_counter()

        result = decrypt_it(s, u, v_queue, q)

        elapsed_dec = time.perf_counter() - t0_dec
        cpu_after_dec = proc.cpu_times()
        _, peak_dec = tracemalloc.get_traced_memory()
        snap_dec = tracemalloc.take_snapshot()
        allocs_dec = sum(stat.count for stat in snap_dec.statistics("lineno"))
        tracemalloc.stop()

        cursor += qrng.total_consumed

        # ========= Record everything =========
        secret_props = extract_secret_properties(s)

        row = {
            "trial": trial,
            # Bob's encryption traces (sender)
            "enc_timing_us": elapsed_enc * 1e6,
            "enc_cpu_user_us": (cpu_after_enc.user - cpu_before_enc.user) * 1e6,
            "enc_ram_peak_bytes": peak_enc,
            "enc_num_allocations": allocs_enc,
            # Alice's decryption traces (receiver)
            "dec_timing_us": elapsed_dec * 1e6,
            "dec_cpu_user_us": (cpu_after_dec.user - cpu_before_dec.user) * 1e6,
            "dec_ram_peak_bytes": peak_dec,
            "dec_num_allocations": allocs_dec,
            # Communication channel features
            **channel_features,
            # Secret key properties (labels)
            **secret_props,
            # Validation
            "decryption_correct": (result == FIXED_PLAINTEXT),
            "q": q,
        }
        rows.append(row)

        if verbose and (trial + 1) % 100 == 0:
            print(f"  Trial {trial + 1}/{num_trials} complete "
                  f"(QRNG consumed: {cursor})")

    if verbose:
        correct = sum(1 for r in rows if r["decryption_correct"])
        print(f"\n{'=' * 70}")
        print(f"Sender↔Receiver Trace Collection Complete")
        print(f"{'=' * 70}")
        print(f"  Trials: {len(rows)}")
        print(f"  Decryption correctness: {correct}/{len(rows)} "
              f"({100*correct/max(1,len(rows)):.1f}%)")
        print(f"  QRNG numbers consumed: {cursor}")

        # Quick summary stats
        enc_times = [r["enc_timing_us"] for r in rows]
        dec_times = [r["dec_timing_us"] for r in rows]
        print(f"\n  Sender (encrypt) timing: "
              f"mean={np.mean(enc_times):.1f} µs, std={np.std(enc_times):.1f} µs")
        print(f"  Receiver (decrypt) timing: "
              f"mean={np.mean(dec_times):.1f} µs, std={np.std(dec_times):.1f} µs")

        weights = [r["secret_weight"] for r in rows]
        print(f"\n  Secret weight distribution: "
              f"min={min(weights)}, max={max(weights)}, "
              f"mean={np.mean(weights):.2f}, std={np.std(weights):.2f}")

    return rows


# ---------------------------------------------------------------------------
# Save traces to disk for reproducibility
# ---------------------------------------------------------------------------

def save_traces(rows, filepath="results/sender_receiver_traces.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  Saved {len(rows)} trace records to {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with open("qrng_combined.txt") as f:
        numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    print("\n" + "#" * 70)
    print("ATTACK SC-1: Sender↔Receiver Communication Trace Collection")
    print("  Using QEQSKE key exchange protocol as the base")
    print("#" * 70)
    print(f"  Plaintext: {FIXED_PLAINTEXT!r}")
    print(f"  Parameters: n=4, k=2 (toy, per HCL paper)")
    print(f"  QRNG dataset: {len(numbers)} numbers available")

    rows = simulate_key_exchange_with_traces(
        numbers, n=4, k=2, num_trials=500, verbose=True
    )
    save_traces(rows)

    print("\n  Trace dimensions per trial:")
    sample = rows[0]
    sender_feats = [k for k in sample if k.startswith("enc_")]
    receiver_feats = [k for k in sample if k.startswith("dec_")]
    channel_feats = [k for k in sample if k not in sender_feats + receiver_feats
                     and k not in ["trial", "secret_weight", "secret_sum",
                                   "secret_l2_norm", "secret_max_abs",
                                   "decryption_correct", "q"]]
    print(f"    Sender (Bob) traces:   {len(sender_feats)} features")
    print(f"    Receiver (Alice) traces: {len(receiver_feats)} features")
    print(f"    Channel features:      {len(channel_feats)} features")
    print(f"    Total feature vector:  {len(sender_feats) + len(receiver_feats) + len(channel_feats)}D")

    print("\n" + "#" * 70)
    print("TRACE COLLECTION COMPLETE — Run attack_sc_ml_classifiers.py next")
    print("#" * 70)
