"""
ATTACK SC-1: Sender-to-Receiver Communication Trace Collection (Clean 5,000 + TCP RTT)
======================================================================================
Implements the full Alice<->Bob QEQSKE key exchange and captures uninstrumented
traces across BOTH endpoints, the public communication channel, and a live
localhost TCP socket round-trip channel:

  1. Sender (Bob) clean uninstrumented encrypt_it(): timing & CPU user-time
  2. Receiver (Alice) clean uninstrumented decrypt_it(): timing & CPU user-time
  3. Public Channel (eavesdropper): 26 ciphertext statistical features (u, v_queue)
  4. Real Localhost TCP Socket RTT:
     - Original decrypt_it() behind a persistent 127.0.0.1 TCP server
     - Constant-time decrypt_it_constant_time() behind the same TCP server

Design safeguards:
  - No tracemalloc inside timed execution passes (avoids 11.7x observer artifact)
  - Randomized execution order per pass (deterministic seeds 42..45)
  - Separate Mersenne warm-up pool (seed=42000) so warm-up never consumes QRNG draws
  - Initial 200-call warm-up + gc.collect() & 25 interleaved warm-ups every 250 trials
  - High-resolution resource.getrusage(RUSAGE_SELF).ru_utime CPU user timer

Saves:
  - results/sender_receiver_traces_5000.json (default)
"""

import argparse
import gc
import json
import math
import os
import socket
import struct
import threading
import time
import numpy as np
import psutil
from scipy import stats

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import resource
except ImportError:
    resource = None

from env_metadata import get_environment_metadata
from qeqske_core import (
    QEQSKERandom,
    decrypt_it,
    decrypt_it_constant_time,
    encrypt_it,
    key_gen,
)
from qrng_handler import MersenneSource, QRNGSource


FIXED_PLAINTEXT = "HELLO"
DEFAULT_TRIALS = 5000
DEFAULT_SEED = 42
BLOCK_SIZE = 250
INITIAL_WARMUP = 200
INTERLEAVED_WARMUP = 25
DEFAULT_OUTPUT_PATH = "results/sender_receiver_traces_5000.json"


# ---------------------------------------------------------------------------
# Helpers
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


def _get_cpu_user_seconds(proc):
    """Return high-resolution process user CPU time in seconds."""
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_utime
    return proc.cpu_times().user


def extract_secret_properties(s):
    """Extract secret key summary properties for ML targets."""
    s_flat = [v for row in s for v in row]
    return {
        "secret_weight": sum(1 for v in s_flat if v != 0),
        "secret_sum": sum(s_flat),
        "secret_l2_norm": float(np.sqrt(sum(v**2 for v in s_flat))),
        "secret_max_abs": max(abs(v) for v in s_flat),
    }


def extract_channel_features(ct, q):
    """
    Extract observable features from the ciphertext (u, v_queue) -- what an
    eavesdropper on the communication channel between Bob and Alice can see.
    """
    u = ct["u"]
    v_queue = ct["v_queue"]
    per_bit = (
        isinstance(u, list)
        and len(u) > 0
        and isinstance(u[0], list)
        and len(u[0]) > 0
        and isinstance(u[0][0], list)
    )

    features = {
        "q": q,
        "num_ct_blocks": len(v_queue),
    }

    if per_bit:
        all_u_vals = []
        u_norms = []
        for u_i in u:
            u_flat = [v for row in u_i for v in row]
            all_u_vals.extend(u_flat)
            u_norms.append(np.sqrt(sum(v**2 for v in u_flat)))
        features["u_frobenius_mean"] = float(np.mean(u_norms))
        features["u_frobenius_std"] = float(np.std(u_norms))
    else:
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

    combined = np.concatenate([u_arr, v_arr])
    features["ct_total_elements"] = len(combined)
    features["ct_mean"] = float(np.mean(combined))
    features["ct_std"] = float(np.std(combined))
    features["ct_entropy_approx"] = float(
        stats.entropy(np.histogram(combined, bins=50, density=True)[0] + 1e-10)
    )

    features["u_mean_over_q"] = features["u_mean"] / max(q, 1)
    features["v_mean_over_q"] = features["v_mean"] / max(q, 1)
    features["ct_coefficient_of_variation"] = (
        features["ct_std"] / max(abs(features["ct_mean"]), 1e-10)
    )

    return features


# ---------------------------------------------------------------------------
# Pre-generation (Pass 0)
# ---------------------------------------------------------------------------

def pregenerate_exchanges(numbers, n=4, k=2, num_trials=DEFAULT_TRIALS,
                          qrng_path="qrng_combined.txt"):
    """Pre-generate num_trials Alice<->Bob key exchanges and 200 warm-up pairs."""
    src = QRNGSource(numbers=numbers, cycle=True)
    rng = QEQSKERandom(src)
    n_orig = len(numbers)

    trials = []
    unique_keys = set()
    for trial_id in range(num_trials):
        keys = key_gen(rng, n=n, k=k)
        s = keys["private_key"]
        A = keys["public_key_A"]
        t = keys["public_key_t"]
        q = keys["q"]

        enc_start = src.total_consumed
        ct = encrypt_it(rng, FIXED_PLAINTEXT, A, t, q, n, k)
        enc_end = src.total_consumed
        enc_qrng_slice = [numbers[i % n_orig] for i in range(enc_start, enc_end)]

        channel_features = extract_channel_features(ct, q)
        secret_props = extract_secret_properties(s)
        unique_keys.add((q, tuple(v for row in s for v in row)))

        payload = json.dumps(
            {"u": ct["u"], "v": ct["v_queue"], "q": q},
            separators=(",", ":"),
        ).encode("utf-8")

        trials.append({
            "trial_id": trial_id,
            "s": s,
            "A": A,
            "t": t,
            "q": q,
            "enc_qrng_slice": enc_qrng_slice,
            "u": ct["u"],
            "v_queue": ct["v_queue"],
            "payload": payload,
            "net_payload_bytes": len(payload),
            "channel_features": channel_features,
            "secret_props": secret_props,
        })

    qrng_meta = {
        "source_file": qrng_path,
        "cycle_enabled": True,
        "dataset_size": n_orig,
        "total_consumed": int(src.total_consumed),
        "wraps": int(src.wraps),
        "unique_q_secret_pairs": len(unique_keys),
    }

    # Separate Mersenne pool for warm-up so warm-up never touches the QRNG stream
    w_src = MersenneSource(seed=42000)
    w_rng = QEQSKERandom(w_src)
    warmup_pool = []
    for _ in range(INITIAL_WARMUP):
        w_keys = key_gen(w_rng, n=n, k=k)
        w_s = w_keys["private_key"]
        w_A = w_keys["public_key_A"]
        w_t = w_keys["public_key_t"]
        w_q = w_keys["q"]

        w_enc_nums = [w_src.call_qrng() for _ in range(len(trials[0]["enc_qrng_slice"]))]
        w_enc_rng = QEQSKERandom(QRNGSource(numbers=w_enc_nums))
        w_ct = encrypt_it(w_enc_rng, FIXED_PLAINTEXT, w_A, w_t, w_q, n, k)
        w_payload = json.dumps(
            {"u": w_ct["u"], "v": w_ct["v_queue"], "q": w_q},
            separators=(",", ":"),
        ).encode("utf-8")
        warmup_pool.append({
            "s": w_s,
            "A": w_A,
            "t": w_t,
            "q": w_q,
            "enc_qrng_slice": w_enc_nums,
            "u": w_ct["u"],
            "v_queue": w_ct["v_queue"],
            "payload": w_payload,
        })

    return trials, warmup_pool, qrng_meta


# ---------------------------------------------------------------------------
# Localhost TCP Server Helper for Pass 3 & Pass 4
# ---------------------------------------------------------------------------

def _recv_exact(sock, n_bytes):
    buf = bytearray()
    while len(buf) < n_bytes:
        chunk = sock.recv(n_bytes - len(buf))
        if not chunk:
            raise ConnectionError("Unexpected EOF on TCP socket")
        buf.extend(chunk)
    return bytes(buf)


class _AliceTCPServer:
    """Single-connection persistent localhost TCP receiver server for Alice."""

    def __init__(self, secret_lookup, decrypt_fn):
        self._secrets = secret_lookup
        self._decrypt_fn = decrypt_fn
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._error = None
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _serve(self):
        conn = None
        try:
            conn, _ = self._srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                try:
                    hdr = _recv_exact(conn, 8)
                except ConnectionError:
                    break
                slot_id, length = struct.unpack("!II", hdr)
                if slot_id == 0xFFFFFFFF and length == 0:
                    break
                payload = _recv_exact(conn, length)
                msg = json.loads(payload)
                s = self._secrets[slot_id]
                dec_out = self._decrypt_fn(s, msg["u"], msg["v"], msg["q"])
                ack = b"\x01" if dec_out == FIXED_PLAINTEXT else b"\x00"
                conn.sendall(ack)
        except Exception as exc:
            self._error = exc
        finally:
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
            try:
                self._srv.close()
            except OSError:
                pass

    def finish(self):
        self._thread.join(timeout=10.0)
        if self._error is not None:
            raise RuntimeError(f"Alice TCP server failed: {self._error}") from self._error


def _run_tcp_rtt_pass(trials, warmup_pool, records, decrypt_fn, seed,
                      pos_key, rtt_key, ack_key, n, k):
    """Measure end-to-end localhost TCP socket RTT for all trials in randomized order."""
    n_trials = len(trials)
    n_warmup = len(warmup_pool)
    secret_lookup = {tr["trial_id"]: tr["s"] for tr in trials}
    for i, w in enumerate(warmup_pool):
        secret_lookup[n_trials + i] = w["s"]

    warmup_frames = [
        struct.pack("!II", n_trials + i, len(w["payload"])) + w["payload"]
        for i, w in enumerate(warmup_pool)
    ]

    server = _AliceTCPServer(secret_lookup, decrypt_fn).start()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect(("127.0.0.1", server.port))

    def _warmup_tcp(count, start_idx=0):
        for i in range(count):
            frame = warmup_frames[(start_idx + i) % n_warmup]
            sock.sendall(frame)
            _ = _recv_exact(sock, 1)

    t_pass_0 = time.perf_counter()
    try:
        gc.collect()
        _warmup_tcp(INITIAL_WARMUP, start_idx=0)
        order = np.random.RandomState(seed).permutation(n_trials)

        for pos, idx in enumerate(order):
            if pos % BLOCK_SIZE == 0:
                gc.collect()
                _warmup_tcp(INTERLEAVED_WARMUP, start_idx=pos)

            tr = trials[int(idx)]
            frame = struct.pack("!II", tr["trial_id"], len(tr["payload"])) + tr["payload"]

            t0 = time.perf_counter()
            sock.sendall(frame)
            ack = _recv_exact(sock, 1)
            elapsed = time.perf_counter() - t0

            rec = records[int(idx)]
            rec[pos_key] = int(pos)
            rec[rtt_key] = float(elapsed * 1e6)
            rec[ack_key] = bool(ack == b"\x01")

        # Graceful shutdown sentinel
        sock.sendall(struct.pack("!II", 0xFFFFFFFF, 0))
    finally:
        try:
            sock.close()
        except OSError:
            pass
        server.finish()

    return time.perf_counter() - t_pass_0


# ---------------------------------------------------------------------------
# Full Uninstrumented Multi-Pass Collection
# ---------------------------------------------------------------------------

def collect_full_traces(numbers, n=4, k=2, num_trials=DEFAULT_TRIALS,
                        seed=DEFAULT_SEED, include_tcp_rtt=True,
                        qrng_path="qrng_combined.txt", verbose=True):
    """
    Collect clean uninstrumented Sender, Receiver, Channel, and (optional)
    localhost TCP RTT traces over num_trials Alice<->Bob key exchanges.
    """
    proc = psutil.Process(os.getpid())
    if verbose:
        print(f"  Pass 0: Pre-generating {num_trials} QRNG exchanges + {INITIAL_WARMUP} warm-ups...")

    trials, warmup_pool, qrng_meta = pregenerate_exchanges(
        numbers, n=n, k=k, num_trials=num_trials, qrng_path=qrng_path
    )
    n_warmup = len(warmup_pool)

    records = [
        {
            "trial_id": int(tr["trial_id"]),
            "q": int(tr["q"]),
            "net_payload_bytes": int(tr["net_payload_bytes"]),
            **tr["secret_props"],
            **{f: val for f, val in tr["channel_features"].items() if f != "q"},
        }
        for tr in trials
    ]

    # --- Pass 1: Sender (encrypt_it) clean uninstrumented ---
    if verbose:
        print("  Pass 1: Collecting clean uninstrumented Sender (encrypt_it) traces...")

    def _warmup_enc(count, start_idx=0):
        for i in range(count):
            w = warmup_pool[(start_idx + i) % n_warmup]
            w_rng = QEQSKERandom(QRNGSource(numbers=list(w["enc_qrng_slice"])))
            _ = encrypt_it(w_rng, FIXED_PLAINTEXT, w["A"], w["t"], w["q"], n, k)

    gc.collect()
    _warmup_enc(INITIAL_WARMUP, start_idx=0)
    enc_order = np.random.RandomState(seed).permutation(num_trials)

    t_p1_0 = time.perf_counter()
    for pos, idx in enumerate(enc_order):
        if pos % BLOCK_SIZE == 0:
            gc.collect()
            _warmup_enc(INTERLEAVED_WARMUP, start_idx=pos)

        tr = trials[int(idx)]
        rng_enc = QEQSKERandom(QRNGSource(numbers=list(tr["enc_qrng_slice"])))
        A, t, q = tr["A"], tr["t"], tr["q"]

        cpu_before = _get_cpu_user_seconds(proc)
        t0 = time.perf_counter()
        _ = encrypt_it(rng_enc, FIXED_PLAINTEXT, A, t, q, n, k)
        elapsed = time.perf_counter() - t0
        cpu_after = _get_cpu_user_seconds(proc)

        rec = records[int(idx)]
        rec["enc_exec_order_pos"] = int(pos)
        rec["enc_timing_us"] = float(elapsed * 1e6)
        rec["enc_cpu_user_us"] = float((cpu_after - cpu_before) * 1e6)
    pass1_wall_s = time.perf_counter() - t_p1_0

    # --- Pass 2: Receiver (decrypt_it) clean uninstrumented ---
    if verbose:
        print("  Pass 2: Collecting clean uninstrumented Receiver (decrypt_it) traces...")

    def _warmup_dec(count, start_idx=0, decrypt_fn=decrypt_it):
        for i in range(count):
            w = warmup_pool[(start_idx + i) % n_warmup]
            _ = decrypt_fn(w["s"], w["u"], w["v_queue"], w["q"])

    gc.collect()
    _warmup_dec(INITIAL_WARMUP, start_idx=0, decrypt_fn=decrypt_it)
    dec_order = np.random.RandomState(seed + 1).permutation(num_trials)

    t_p2_0 = time.perf_counter()
    for pos, idx in enumerate(dec_order):
        if pos % BLOCK_SIZE == 0:
            gc.collect()
            _warmup_dec(INTERLEAVED_WARMUP, start_idx=pos, decrypt_fn=decrypt_it)

        tr = trials[int(idx)]
        s, u, v_queue, q = tr["s"], tr["u"], tr["v_queue"], tr["q"]

        cpu_before = _get_cpu_user_seconds(proc)
        t0 = time.perf_counter()
        dec_out = decrypt_it(s, u, v_queue, q)
        elapsed = time.perf_counter() - t0
        cpu_after = _get_cpu_user_seconds(proc)

        rec = records[int(idx)]
        rec["dec_exec_order_pos"] = int(pos)
        rec["dec_timing_us"] = float(elapsed * 1e6)
        rec["dec_cpu_user_us"] = float((cpu_after - cpu_before) * 1e6)
        rec["decryption_correct"] = bool(dec_out == FIXED_PLAINTEXT)
    pass2_wall_s = time.perf_counter() - t_p2_0

    pass_timings = {
        "sender_pass_wall_seconds": float(pass1_wall_s),
        "receiver_pass_wall_seconds": float(pass2_wall_s),
    }

    # --- Pass 3 & Pass 4: Localhost TCP Socket RTT ---
    if include_tcp_rtt:
        if verbose:
            print("  Pass 3: Measuring localhost TCP socket RTT (original decrypt_it)...")
        pass3_wall_s = _run_tcp_rtt_pass(
            trials=trials,
            warmup_pool=warmup_pool,
            records=records,
            decrypt_fn=decrypt_it,
            seed=seed + 2,
            pos_key="net_exec_order_pos",
            rtt_key="net_rtt_us",
            ack_key="net_ack_correct",
            n=n,
            k=k,
        )
        pass_timings["tcp_orig_pass_wall_seconds"] = float(pass3_wall_s)

        if verbose:
            print("  Pass 4: Measuring localhost TCP socket RTT (constant-time decrypt_it)...")
        pass4_wall_s = _run_tcp_rtt_pass(
            trials=trials,
            warmup_pool=warmup_pool,
            records=records,
            decrypt_fn=decrypt_it_constant_time,
            seed=seed + 3,
            pos_key="net_ct_exec_order_pos",
            rtt_key="net_ct_rtt_us",
            ack_key="net_ct_ack_correct",
            n=n,
            k=k,
        )
        pass_timings["tcp_ct_pass_wall_seconds"] = float(pass4_wall_s)

    exp_meta = {
        "n_trials": int(num_trials),
        "n": int(n),
        "k": int(k),
        "fixed_plaintext": FIXED_PLAINTEXT,
        "seed": int(seed),
        "instrumentation": "none (clean uninstrumented perf_counter + ru_utime)",
        "include_tcp_rtt": bool(include_tcp_rtt),
        "cpu_user_timer": (
            "resource.getrusage(RUSAGE_SELF).ru_utime"
            if resource is not None
            else "psutil.Process.cpu_times.user"
        ),
        "warmup_initial": INITIAL_WARMUP,
        "warmup_interleaved": INTERLEAVED_WARMUP,
        "block_size": BLOCK_SIZE,
        **pass_timings,
    }

    if verbose:
        correct = sum(1 for r in records if r["decryption_correct"])
        enc_t = [r["enc_timing_us"] for r in records]
        dec_t = [r["dec_timing_us"] for r in records]
        print(f"\n{'=' * 74}")
        print("Sender<->Receiver Clean Trace Collection Complete")
        print(f"{'=' * 74}")
        print(f"  Trials: {len(records)} (unique (q, s): {qrng_meta['unique_q_secret_pairs']})")
        print(f"  Local decrypt correctness: {correct}/{len(records)} ({100*correct/max(1, len(records)):.2f}%)")
        print(f"  Sender (encrypt_it):       mean={np.mean(enc_t):.2f} us, median={np.median(enc_t):.2f} us")
        print(f"  Receiver (decrypt_it):     mean={np.mean(dec_t):.2f} us, median={np.median(dec_t):.2f} us")
        if include_tcp_rtt:
            net_t = [r["net_rtt_us"] for r in records]
            net_ct_t = [r["net_ct_rtt_us"] for r in records]
            ack_match = sum(1 for r in records if r["net_ack_correct"] == r["decryption_correct"])
            ack_ct_match = sum(1 for r in records if r["net_ct_ack_correct"] == r["decryption_correct"])
            print(f"  TCP RTT (original):        mean={np.mean(net_t):.2f} us, median={np.median(net_t):.2f} us (match={ack_match}/{len(records)})")
            print(f"  TCP RTT (constant-time):   mean={np.mean(net_ct_t):.2f} us, median={np.median(net_ct_t):.2f} us (match={ack_ct_match}/{len(records)})")

    return to_native(records), to_native(exp_meta), to_native(qrng_meta)


def simulate_key_exchange_with_traces(numbers, n=4, k=2, num_trials=DEFAULT_TRIALS,
                                      verbose=True, seed=DEFAULT_SEED,
                                      include_tcp_rtt=True):
    """Backward-compatible entry point returning the list of trace record dicts."""
    records, _, _ = collect_full_traces(
        numbers=numbers,
        n=n,
        k=k,
        num_trials=num_trials,
        seed=seed,
        include_tcp_rtt=include_tcp_rtt,
        verbose=verbose,
    )
    return records


def save_traces(rows, filepath=DEFAULT_OUTPUT_PATH, experiment=None, qrng_stream=None):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    payload = {
        "environment": get_environment_metadata(),
        "records": to_native(rows),
    }
    if experiment is not None:
        payload["experiment"] = to_native(experiment)
    if qrng_stream is not None:
        payload["qrng_stream"] = to_native(qrng_stream)
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Saved {len(rows)} trace records to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean 5,000-trial Sender<->Receiver + TCP RTT trace collector."
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--qrng", type=str, default="qrng_combined.txt")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-tcp", action="store_true", help="Skip localhost TCP socket RTT passes.")
    args = parser.parse_args()

    src = QRNGSource.from_file(args.qrng, cycle=True)
    numbers = list(src._original)

    print("\n" + "#" * 74)
    print("ATTACK SC-1: Clean Sender<->Receiver + TCP RTT Trace Collection")
    print(f"  Trials: {args.n_trials}, Parameters: n=4, k=2, Seed: {args.seed}")
    print(f"  QRNG dataset: {len(numbers)} numbers")
    print("#" * 74)

    records, exp_meta, qrng_meta = collect_full_traces(
        numbers=numbers,
        n=4,
        k=2,
        num_trials=args.n_trials,
        seed=args.seed,
        include_tcp_rtt=not args.no_tcp,
        qrng_path=args.qrng,
        verbose=True,
    )
    save_traces(records, filepath=args.output, experiment=exp_meta, qrng_stream=qrng_meta)


if __name__ == "__main__":
    main()
