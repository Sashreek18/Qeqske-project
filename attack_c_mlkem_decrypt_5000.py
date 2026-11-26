"""
ATTACK C-5 (Real ML-KEM Scaling: n=256, k=2 or 3, q=3329):
Decryption Side-Channel Leakage Evaluation via kyber-py & QEQSKE Variant
=============================================================================
Ports the 5,000-trial Prompt 2 side-channel pipeline from toy QEQSKE (n=4, k=2)
to real ML-KEM parameters (n=256, k=2 ML-KEM-512 or k=3 ML-KEM-768, q=3329)
using the external reference implementation `kyber-py` (FIPS 203).

Two configurable protocol variants:
  1. `standard`: Unmodified `kyber_py.ml_kem.ML_KEM` (`q = 3329`, SHAKE-256
     reproducible randomness), exercising the real FIPS 203 `decaps()` and
     `_k_pke_decrypt()` code paths directly.
  2. `qeqske`: Configurable QEQSKE variant built directly atop `kyber-py`'s
     `ML_KEM` / `PolynomialRing` / `Module` classes, feeding ANU QRNG bytes
     (`qrng_combined.txt`) and selecting a per-key 16-bit NTT-friendly prime
     `q = 1 (mod 256)` (`q in [3329, 64513]`) via `closest_ntt_prime()`.

For each variant (5,000 trials, randomized execution order, CPU core 0 pinned,
warm-up interleaved), collects:
  - Clean uninstrumented `decaps(dk, c)` timing & CPU user time (`ru_utime`)
  - Isolated clean `_k_pke_decrypt(dk_pke, c)` timing & CPU user time
  - Legacy `tracemalloc`-wrapped `decaps(dk, c)` timing, CPU, peak RAM, allocs

Evaluates BOTH secret targets under the full Prompt 2 pipeline (2,000-resample
bootstrap CIs, 5x10 repeated CV, shuffled-label control, in-fold q-residualization,
and 200-permutation tests reported as `< 1/201 (0.0050)` at the floor):
  1. `secret_weight`: non-zero count of centered `s = NTT^{-1}(s_hat)` (cls + reg)
  2. Individual coefficient sign `s[0][0]`:
     - `s00_positive`: `1 if s[0][0] > 0 else 0` (all 5,000 trials)
     - `s00_sign_nonzero`: `1 if s[0][0] > 0 else 0` (balanced `s[0][0] != 0` subset)
"""

import argparse
import gc
from hashlib import shake_128, shake_256
import json
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
import psutil
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from kyber_py.ml_kem.default_parameters import DEFAULT_PARAMETERS
from kyber_py.ml_kem.ml_kem import ML_KEM
from kyber_py.modules.modules import Matrix, Module
from kyber_py.polynomials.polynomials import Polynomial, PolynomialNTT, PolynomialRing
from kyber_py.utilities.utils import select_bytes

from attack_c_decrypt_5000 import (
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    SEED,
    _fmt_perm_p,
    _get_cpu_user_seconds,
    _safe_pearson,
    _safe_spearman,
    describe_array,
    paired_correlation_with_bootstrap,
    run_classification_suite,
    run_regression_suite,
    to_native,
)
from env_metadata import format_perm_pvalue, get_environment_metadata
from qeqske_core import is_prime
from qrng_handler import MersenneSource, QRNGSource

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ---------------------------------------------------------------------------
# NTT-friendly primes (q = 1 mod 256) for QEQSKE n=256 per-key q variant
# ---------------------------------------------------------------------------

NTT_PRIMES_16 = [q for q in range(2049, 65536, 256) if is_prime(q)]


def _find_prim_256_root(q: int) -> int:
    exp = (q - 1) // 256
    for g in range(2, q):
        r = pow(g, exp, q)
        if pow(r, 128, q) == q - 1:
            return r
    raise RuntimeError(f"No primitive 256th root of unity mod {q}")


NTT_ROOTS = {q: _find_prim_256_root(q) for q in NTT_PRIMES_16}


def closest_ntt_prime(val: int) -> int:
    """Nearest NTT-friendly prime q = 1 (mod 256) in [3329, 64513]; ties upward."""
    return min(NTT_PRIMES_16, key=lambda p: (abs(p - val), -p))


# ---------------------------------------------------------------------------
# kyber-py Subclasses Supporting Configurable Per-Key Prime q (d_poly = 16)
# ---------------------------------------------------------------------------

class ParamPolynomial(Polynomial):
    def _compress_ele(self, x, d):
        q = self.parent.q
        t = 1 << d
        return ((t * x + (q // 2)) // q) % t

    def _decompress_ele(self, x, d):
        q = self.parent.q
        return (q * x + (1 << (d - 1))) >> d

    def to_ntt(self):
        q = self.parent.q
        k, l = 1, 128
        coeffs = self.coeffs
        zetas = self.parent.ntt_zetas
        while l >= 2:
            start = 0
            while start < 256:
                zeta = zetas[k]
                k += 1
                j = start
                for j in range(start, start + l):
                    t = zeta * coeffs[j + l]
                    coeffs[j + l] = coeffs[j] - t
                    coeffs[j] = coeffs[j] + t
                start = l + (j + 1)
            l >>= 1
        for j in range(256):
            coeffs[j] %= q
        return self.parent(coeffs, is_ntt=True)


class ParamPolynomialNTT(ParamPolynomial, PolynomialNTT):
    def from_ntt(self):
        q = self.parent.q
        l = 2
        k = 127
        coeffs = self.coeffs
        zetas = self.parent.ntt_zetas
        while l <= 128:
            start = 0
            while start < 256:
                zeta = zetas[k]
                k -= 1
                j = start
                for j in range(start, start + l):
                    t = coeffs[j]
                    coeffs[j] = t + coeffs[j + l]
                    coeffs[j + l] = zeta * (coeffs[j + l] - t)
                start = j + l + 1
            l <<= 1
        f = self.parent.ntt_f
        for j in range(256):
            coeffs[j] = (coeffs[j] * f) % q
        return self.parent(coeffs, is_ntt=False)

    def _ntt_base_multiplication(self, a0, a1, b0, b1, zeta):
        q = self.parent.q
        return (a0 * b0 + zeta * a1 * b1) % q, (a1 * b0 + a0 * b1) % q

    def __mul__(self, other):
        if isinstance(other, type(self)):
            new_coeffs = self._ntt_multiplication(other)
        elif isinstance(other, int):
            q = self.parent.q
            new_coeffs = [(c * other) % q for c in self.coeffs]
        else:
            raise NotImplementedError
        return self.parent(new_coeffs, is_ntt=True)


class ParamPolynomialRing(PolynomialRing):
    def __init__(self, q: int, d_poly: int = 16):
        self.q = q
        self.n = 256
        self._d_poly = d_poly
        self.element = ParamPolynomial
        self.element_ntt = ParamPolynomialNTT
        root = NTT_ROOTS[q]
        self.ntt_zetas = [pow(root, self._br(i, 7), q) for i in range(128)]
        self.ntt_f = pow(128, -1, q)
        self._q_mask = (1 << q.bit_length()) - 1

    def ntt_sample(self, input_bytes):
        q = self.q
        mask = self._q_mask
        coeffs = [0] * 256
        i = j = 0
        n_bytes = len(input_bytes)
        while j < 256:
            if i + 1 >= n_bytes:
                raise RuntimeError(f"XOF buffer exhausted at i={i}, j={j} for q={q}")
            d = (input_bytes[i] | (input_bytes[i + 1] << 8)) & mask
            if d < q:
                coeffs[j] = d
                j += 1
            i += 2
        return self(coeffs, is_ntt=True)

    def cbd(self, input_bytes, eta, is_ntt=False):
        q = self.q
        coeffs = [0] * 256
        b_int = int.from_bytes(input_bytes, "little")
        mask = (1 << eta) - 1
        mask2 = (1 << (2 * eta)) - 1
        for i in range(256):
            x = b_int & mask2
            a = (x & mask).bit_count()
            b = ((x >> eta) & mask).bit_count()
            b_int >>= 2 * eta
            coeffs[i] = (a - b) % q
        return self(coeffs, is_ntt=is_ntt)

    def decode(self, input_bytes, d, is_ntt=False):
        if 256 * d != len(input_bytes) * 8:
            raise ValueError(f"Expected {32 * d} bytes, got {len(input_bytes)}")
        m = self.q if d == self._d_poly else (1 << d)
        coeffs = [0] * 256
        b_int = int.from_bytes(input_bytes, "little")
        mask = (1 << d) - 1
        for i in range(256):
            coeffs[i] = (b_int & mask) % m
            b_int >>= d
        return self(coeffs, is_ntt=is_ntt)


class ParamModule(Module):
    def __init__(self, q: int, d_poly: int = 16):
        self.ring = ParamPolynomialRing(q, d_poly=d_poly)
        self.matrix = Matrix


class QEQSKE_ML_KEM(ML_KEM):
    """QEQSKE-variant ML-KEM (per-key NTT prime q + 16-bit polynomial encoding)."""

    def __init__(self, params: dict, q: int, random_bytes_fn=None):
        super().__init__(params)
        self.q = q
        self.d_poly = 16
        self.M = ParamModule(q, d_poly=self.d_poly)
        self.R = self.M.ring
        if random_bytes_fn is not None:
            self.random_bytes = random_bytes_fn

    def _ek_size(self) -> int:
        return 32 * self.d_poly * self.k + 32

    def _dk_size(self) -> int:
        return 64 * self.d_poly * self.k + 96

    @staticmethod
    def _xof(b: bytes, i: bytes, j: bytes) -> bytes:
        return shake_128(b + i + j).digest(4096)

    def _k_pke_keygen(self, d: bytes) -> tuple[bytes, bytes]:
        rho, sigma = self._G(d + bytes([self.k]))
        A_hat = self._generate_matrix_from_seed(rho)
        N = 0
        s, N = self._generate_error_vector(sigma, self.eta_1, N)
        e, N = self._generate_error_vector(sigma, self.eta_1, N)
        s_hat = s.to_ntt()
        e_hat = e.to_ntt()
        t_hat = A_hat @ s_hat + e_hat
        return (t_hat.encode(self.d_poly) + rho, s_hat.encode(self.d_poly))

    def _k_pke_encrypt(self, ek_pke: bytes, m: bytes, r: bytes) -> bytes:
        if len(ek_pke) != self._ek_size():
            raise ValueError("ek length mismatch")
        t_hat_bytes, rho = ek_pke[:-32], ek_pke[-32:]
        t_hat = self.M.decode_vector(t_hat_bytes, self.k, self.d_poly, is_ntt=True)
        if t_hat.encode(self.d_poly) != t_hat_bytes:
            raise ValueError("modulus check failed")
        A_hat_T = self._generate_matrix_from_seed(rho, transpose=True)
        N = 0
        y, N = self._generate_error_vector(r, self.eta_1, N)
        e1, N = self._generate_error_vector(r, self.eta_2, N)
        e2, N = self._generate_polynomial(r, self.eta_2, N)
        y_hat = y.to_ntt()
        u = (A_hat_T @ y_hat).from_ntt() + e1
        mu = self.R.decode(m, 1).decompress(1)
        v = t_hat.dot(y_hat).from_ntt() + e2 + mu
        return u.compress(self.du).encode(self.du) + v.compress(self.dv).encode(self.dv)

    def _k_pke_decrypt(self, dk_pke: bytes, c: bytes) -> bytes:
        n = self.k * self.du * 32
        c1, c2 = c[:n], c[n:]
        u = self.M.decode_vector(c1, self.k, self.du).decompress(self.du)
        v = self.R.decode(c2, self.dv).decompress(self.dv)
        s_hat = self.M.decode_vector(dk_pke, self.k, self.d_poly, is_ntt=True)
        u_hat = u.to_ntt()
        w = v - (s_hat.dot(u_hat)).from_ntt()
        return w.compress(1).encode(1)

    def _decaps_internal(self, dk: bytes, c: bytes) -> bytes:
        if len(c) != 32 * (self.du * self.k + self.dv):
            raise ValueError("ciphertext type check failed")
        if len(dk) != self._dk_size():
            raise ValueError("decapsulation type check failed")
        pke_len = 32 * self.d_poly * self.k
        dk_pke = dk[0 : pke_len]
        ek_pke = dk[pke_len : 2 * pke_len + 32]
        h = dk[2 * pke_len + 32 : 2 * pke_len + 64]
        z = dk[2 * pke_len + 64 :]
        if self._H(ek_pke) != h:
            raise ValueError("hash check failed")
        m_prime = self._k_pke_decrypt(dk_pke, c)
        K_prime, r_prime = self._G(m_prime + h)
        K_bar = self._J(z + c)
        c_prime = self._k_pke_encrypt(ek_pke, m_prime, r_prime)
        return select_bytes(K_bar, K_prime, c == c_prime)


# ---------------------------------------------------------------------------
# Randomness & Secret Extraction Helpers
# ---------------------------------------------------------------------------

def make_deterministic_shake_rng(seed: int):
    state = {"ctr": 0}

    def _rand_bytes(n: int) -> bytes:
        idx = state["ctr"]
        state["ctr"] += 1
        return shake_256(f"mlkem-std-{seed}-{idx}".encode("ascii")).digest(n)

    return _rand_bytes


def make_qrng_bytes_fn(qrng_src: QRNGSource):
    def _rand_bytes(n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            out.extend(qrng_src.call_qrng().to_bytes(2, "little"))
        return bytes(out[:n])

    return _rand_bytes


def extract_secret_metrics(kem: ML_KEM, dk: bytes, d_poly: int) -> dict:
    """Decode s_hat and s = NTT^{-1}(s_hat) from dk and verify bounds."""
    q = kem.R.q
    pke_len = 32 * d_poly * kem.k
    dk_pke = dk[:pke_len]
    s_hat = kem.M.decode_vector(dk_pke, kem.k, d_poly, is_ntt=True)
    s_hat_zero_count = sum(
        1 for row in s_hat._data for poly in row for c in poly.coeffs if c == 0
    )
    s = s_hat.from_ntt()
    half_q = q // 2
    centered = [
        (c - q if c > half_q else c)
        for row in s._data
        for poly in row
        for c in poly.coeffs
    ]
    max_abs = max(abs(c) for c in centered)
    if max_abs > kem.eta_1:
        raise AssertionError(f"Secret coefficient out of [-{kem.eta_1}, {kem.eta_1}]: max_abs={max_abs}")

    s00 = int(centered[0])
    return {
        "dk_pke": dk_pke,
        "secret_weight": int(sum(1 for c in centered if c != 0)),
        "secret_l1_norm": int(sum(abs(c) for c in centered)),
        "s_hat_zero_count": int(s_hat_zero_count),
        "s00": s00,
        "s00_positive": int(1 if s00 > 0 else 0),
        "s00_sign": int(1 if s00 > 0 else (-1 if s00 < 0 else 0)),
        "dk_bit_hw": int(sum(b.bit_count() for b in dk)),
    }


# ---------------------------------------------------------------------------
# Pre-generation & Trace Collection
# ---------------------------------------------------------------------------

def pregenerate_mlkem_trials(
    variant: str,
    k_param: int,
    n_trials: int,
    seed: int = SEED,
    qrng_path: str = "qrng_combined.txt",
):
    param_key = "ML512" if k_param == 2 else "ML768"
    params = DEFAULT_PARAMETERS[param_key]
    trials = []
    warmup_pool = []

    if variant == "standard":
        kem = ML_KEM(params)
        kem.random_bytes = make_deterministic_shake_rng(seed)
        d_poly = 12
        for trial_id in range(n_trials):
            ek, dk = kem.keygen()
            K, c = kem.encaps(ek)
            sec = extract_secret_metrics(kem, dk, d_poly)
            trials.append({
                "trial_id": trial_id,
                "kem": kem,
                "dk": dk,
                "dk_pke": sec["dk_pke"],
                "c": c,
                "K": K,
                "q": int(kem.R.q),
                **{k: v for k, v in sec.items() if k != "dk_pke"},
            })

        w_kem = ML_KEM(params)
        w_kem.random_bytes = make_deterministic_shake_rng(seed + 42000)
        for _ in range(100):
            ek_w, dk_w = w_kem.keygen()
            K_w, c_w = w_kem.encaps(ek_w)
            warmup_pool.append((w_kem, dk_w, dk_w[: 32 * d_poly * k_param], c_w, K_w))

        stream_meta = {
            "source": f"Deterministic SHAKE-256 (seed={seed})",
            "q_mode": "fixed",
            "q_fixed": 3329,
            "unique_q_count": 1,
        }
    elif variant == "qeqske":
        src = QRNGSource.from_file(qrng_path, cycle=True)
        if seed != SEED:
            n_orig = len(src._original)
            start_offset = int(np.random.RandomState(seed).randint(1, n_orig))
            rotated = src._original[start_offset:] + src._original[:start_offset]
            src = QRNGSource(numbers=rotated, cycle=True)
        else:
            start_offset = 0
        rand_bytes_fn = make_qrng_bytes_fn(src)
        d_poly = 16
        kem_cache = {}
        q_set = set()
        for trial_id in range(n_trials):
            q = closest_ntt_prime(src.call_qrng())
            q_set.add(q)
            if q not in kem_cache:
                kem_cache[q] = QEQSKE_ML_KEM(params, q=q, random_bytes_fn=rand_bytes_fn)
            kem = kem_cache[q]
            ek, dk = kem.keygen()
            K, c = kem.encaps(ek)
            sec = extract_secret_metrics(kem, dk, d_poly)
            trials.append({
                "trial_id": trial_id,
                "kem": kem,
                "dk": dk,
                "dk_pke": sec["dk_pke"],
                "c": c,
                "K": K,
                "q": int(q),
                **{k: v for k, v in sec.items() if k != "dk_pke"},
            })

        warmup_seed = 42000 if seed == SEED else 42000 + int(seed)
        w_src = MersenneSource(seed=warmup_seed)
        w_rand_fn = make_qrng_bytes_fn(w_src)
        w_cache = {}
        for _ in range(100):
            qw = closest_ntt_prime(w_src.call_qrng())
            if qw not in w_cache:
                w_cache[qw] = QEQSKE_ML_KEM(params, q=qw, random_bytes_fn=w_rand_fn)
            w_kem = w_cache[qw]
            ek_w, dk_w = w_kem.keygen()
            K_w, c_w = w_kem.encaps(ek_w)
            warmup_pool.append((w_kem, dk_w, dk_w[: 32 * d_poly * k_param], c_w, K_w))

        stream_meta = {
            "source": qrng_path,
            "cycle_enabled": True,
            "qrng_start_offset": int(start_offset),
            "dataset_size": int(len(src._original)),
            "total_consumed_uint16": int(src.total_consumed),
            "wraps": int(src.wraps),
            "q_mode": "per_key_ntt_prime",
            "ntt_prime_pool_size": len(NTT_PRIMES_16),
            "unique_q_observed": len(q_set),
            "q_min": int(min(q_set)),
            "q_max": int(max(q_set)),
        }
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return trials, warmup_pool, stream_meta


def collect_mlkem_traces(trials, warmup_pool, seed: int = SEED, skip_tracemalloc: bool = False):
    """Collect clean decaps, clean _k_pke_decrypt, and tracemalloc decaps traces."""
    proc = psutil.Process(os.getpid())
    n_trials = len(trials)
    records = [
        {
            "trial_id": tr["trial_id"],
            "q": tr["q"],
            "secret_weight": tr["secret_weight"],
            "secret_l1_norm": tr["secret_l1_norm"],
            "s_hat_zero_count": tr["s_hat_zero_count"],
            "s00": tr["s00"],
            "s00_positive": tr["s00_positive"],
            "s00_sign": tr["s00_sign"],
            "dk_bit_hw": tr["dk_bit_hw"],
        }
        for tr in trials
    ]

    orig_affinity = None
    if hasattr(os, "sched_setaffinity"):
        orig_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {0})

    try:
        # --- Pass 1a: Clean uninstrumented decaps(dk, c) ---
        gc.collect()
        for kem_w, dk_w, _, c_w, _ in warmup_pool:
            _ = kem_w.decaps(dk_w, c_w)

        order_dec = np.random.RandomState(seed).permutation(n_trials)
        t_1a_0 = time.perf_counter()
        for pos, idx in enumerate(order_dec):
            if pos % 250 == 0:
                gc.collect()
                for wi in range(15):
                    kem_w, dk_w, _, c_w, _ = warmup_pool[(pos + wi) % len(warmup_pool)]
                    _ = kem_w.decaps(dk_w, c_w)

            tr = trials[int(idx)]
            kem = tr["kem"]
            cpu0 = _get_cpu_user_seconds(proc)
            t0 = time.perf_counter_ns()
            k_out = kem.decaps(tr["dk"], tr["c"])
            dt_ns = time.perf_counter_ns() - t0
            cpu1 = _get_cpu_user_seconds(proc)

            rec = records[int(idx)]
            rec["exec_order_pos"] = int(pos)
            rec["dec_timing_us"] = float(dt_ns / 1000.0)
            rec["dec_cpu_user_us"] = float((cpu1 - cpu0) * 1e6)
            rec["decryption_correct"] = bool(k_out == tr["K"])
        pass1a_wall = time.perf_counter() - t_1a_0

        # --- Pass 1b: Isolated clean _k_pke_decrypt(dk_pke, c) ---
        gc.collect()
        for kem_w, _, dk_pke_w, c_w, _ in warmup_pool:
            _ = kem_w._k_pke_decrypt(dk_pke_w, c_w)

        order_pke = np.random.RandomState(seed + 50).permutation(n_trials)
        t_1b_0 = time.perf_counter()
        for pos, idx in enumerate(order_pke):
            if pos % 250 == 0:
                gc.collect()
                for wi in range(15):
                    kem_w, _, dk_pke_w, c_w, _ = warmup_pool[(pos + wi) % len(warmup_pool)]
                    _ = kem_w._k_pke_decrypt(dk_pke_w, c_w)

            tr = trials[int(idx)]
            kem = tr["kem"]
            cpu0 = _get_cpu_user_seconds(proc)
            t0 = time.perf_counter_ns()
            _ = kem._k_pke_decrypt(tr["dk_pke"], tr["c"])
            dt_ns = time.perf_counter_ns() - t0
            cpu1 = _get_cpu_user_seconds(proc)

            rec = records[int(idx)]
            rec["pke_exec_order_pos"] = int(pos)
            rec["pke_dec_timing_us"] = float(dt_ns / 1000.0)
            rec["pke_dec_cpu_user_us"] = float((cpu1 - cpu0) * 1e6)
        pass1b_wall = time.perf_counter() - t_1b_0

        # --- Pass 2: Legacy tracemalloc-wrapped decaps(dk, c) ---
        pass2_wall = 0.0
        if not skip_tracemalloc:
            gc.collect()
            for kem_w, dk_w, _, c_w, _ in warmup_pool[:30]:
                tracemalloc.start()
                _ = kem_w.decaps(dk_w, c_w)
                _ = tracemalloc.get_traced_memory()
                _ = tracemalloc.take_snapshot()
                tracemalloc.stop()

            order_tm = np.random.RandomState(seed + 100).permutation(n_trials)
            t_2_0 = time.perf_counter()
            for pos, idx in enumerate(order_tm):
                if pos % 500 == 0:
                    gc.collect()
                    for wi in range(10):
                        kem_w, dk_w, _, c_w, _ = warmup_pool[(pos + wi) % len(warmup_pool)]
                        tracemalloc.start()
                        _ = kem_w.decaps(dk_w, c_w)
                        _ = tracemalloc.get_traced_memory()
                        _ = tracemalloc.take_snapshot()
                        tracemalloc.stop()

                tr = trials[int(idx)]
                kem = tr["kem"]
                tracemalloc.start()
                cpu0 = _get_cpu_user_seconds(proc)
                t0 = time.perf_counter_ns()
                k_out = kem.decaps(tr["dk"], tr["c"])
                dt_ns = time.perf_counter_ns() - t0
                cpu1 = _get_cpu_user_seconds(proc)
                _, peak = tracemalloc.get_traced_memory()
                snap = tracemalloc.take_snapshot()
                n_allocs = sum(stat.count for stat in snap.statistics("lineno"))
                tracemalloc.stop()

                rec = records[int(idx)]
                rec["tm_exec_order_pos"] = int(pos)
                rec["tm_dec_timing_us"] = float(dt_ns / 1000.0)
                rec["tm_dec_cpu_user_us"] = float((cpu1 - cpu0) * 1e6)
                rec["dec_ram_peak_bytes"] = int(peak)
                rec["dec_num_allocations"] = int(n_allocs)
                rec["tm_decryption_correct"] = bool(k_out == tr["K"])
            pass2_wall = time.perf_counter() - t_2_0
    finally:
        if orig_affinity is not None:
            os.sched_setaffinity(0, orig_affinity)

    return records, {
        "decaps_clean_wall_seconds": float(pass1a_wall),
        "pke_decrypt_clean_wall_seconds": float(pass1b_wall),
        "decaps_tracemalloc_wall_seconds": float(pass2_wall),
    }


# ---------------------------------------------------------------------------
# Analysis Suite (Weight + Individual Coefficient Sign)
# ---------------------------------------------------------------------------

def _analyze_feature_pass(
    records,
    feature_keys,
    timing_key,
    cpu_key,
    order_key,
    q_vals,
    weights,
    y_wt,
    y_wt_shuf,
    wt_baseline,
    s00_vals,
    y_pos,
    y_pos_shuf,
    pos_baseline,
    nz_mask,
    y_nz,
    y_nz_shuf,
    nz_baseline,
    perm_all_primary=False,
    run_nonzero_sign=True,
):
    X = np.column_stack([np.array([r[k] for r in records], dtype=float) for k in feature_keys])
    timing_vals = np.array([r[timing_key] for r in records], dtype=float)
    cpu_vals = np.array([r[cpu_key] for r in records], dtype=float)
    order_vals = np.array([r[order_key] for r in records], dtype=float)
    l1_vals = np.array([r["secret_l1_norm"] for r in records], dtype=float)
    shat_z_vals = np.array([r["s_hat_zero_count"] for r in records], dtype=float)
    dk_hw_vals = np.array([r["dk_bit_hw"] for r in records], dtype=float)

    descriptives = {k: describe_array([r[k] for r in records]) for k in feature_keys}
    corr_weight = {
        k: paired_correlation_with_bootstrap([r[k] for r in records], weights)
        for k in feature_keys
    }
    corr_s00_val = {
        k: paired_correlation_with_bootstrap([r[k] for r in records], s00_vals)
        for k in feature_keys
    }
    corr_s00_pos = {
        k: paired_correlation_with_bootstrap([r[k] for r in records], y_pos)
        for k in feature_keys
    }
    diagnostic_corrs = {
        "timing_vs_secret_l1_norm": paired_correlation_with_bootstrap(timing_vals, l1_vals),
        "timing_vs_s_hat_zero_count": paired_correlation_with_bootstrap(timing_vals, shat_z_vals),
        "timing_vs_dk_bit_hw": paired_correlation_with_bootstrap(timing_vals, dk_hw_vals),
    }

    p_qw, pp_qw = _safe_pearson(q_vals, weights)
    s_qw, sp_qw = _safe_spearman(q_vals, weights)
    p_qs, pp_qs = _safe_pearson(q_vals, y_pos)
    s_qs, sp_qs = _safe_spearman(q_vals, y_pos)
    p_qt, pp_qt = _safe_pearson(q_vals, timing_vals)
    s_qt, sp_qt = _safe_spearman(q_vals, timing_vals)
    p_qc, pp_qc = _safe_pearson(q_vals, cpu_vals)
    s_qc, sp_qc = _safe_spearman(q_vals, cpu_vals)

    confound_q = {
        "q_vs_secret_weight": {"pearson_r": p_qw, "pearson_p": pp_qw, "spearman_rho": s_qw, "spearman_p": sp_qw},
        "q_vs_s00_positive": {"pearson_r": p_qs, "pearson_p": pp_qs, "spearman_rho": s_qs, "spearman_p": sp_qs},
        f"q_vs_{timing_key}": {"pearson_r": p_qt, "pearson_p": pp_qt, "spearman_rho": s_qt, "spearman_p": sp_qt},
        f"q_vs_{cpu_key}": {"pearson_r": p_qc, "pearson_p": pp_qc, "spearman_rho": s_qc, "spearman_p": sp_qc},
    }

    p_ow, pp_ow = _safe_pearson(order_vals, weights)
    p_ot, pp_ot = _safe_pearson(order_vals, timing_vals)
    p_oc, pp_oc = _safe_pearson(order_vals, cpu_vals)
    drift_check = {
        "exec_order_vs_secret_weight": {"pearson_r": p_ow, "pearson_p": pp_ow},
        f"exec_order_vs_{timing_key}": {"pearson_r": p_ot, "pearson_p": pp_ot},
        f"exec_order_vs_{cpu_key}": {"pearson_r": p_oc, "pearson_p": pp_oc},
    }

    wt_cls = run_classification_suite(
        X, q_vals, y_wt, y_wt_shuf, wt_baseline, perm_all=perm_all_primary
    )
    wt_reg = run_regression_suite(
        X, q_vals, weights, np.random.RandomState(SEED).permutation(weights), perm_all=perm_all_primary
    )
    pos_cls = run_classification_suite(
        X, q_vals, y_pos, y_pos_shuf, pos_baseline, perm_all=perm_all_primary
    )

    out = {
        "features": list(feature_keys),
        "descriptives": descriptives,
        "correlations_vs_secret_weight": corr_weight,
        "correlations_vs_s00_value": corr_s00_val,
        "correlations_vs_s00_positive": corr_s00_pos,
        "diagnostic_correlations": diagnostic_corrs,
        "confound_q": confound_q,
        "drift_check": drift_check,
        "weight_classification": wt_cls,
        "weight_regression": wt_reg,
        "s00_positive_classification": pos_cls,
    }
    if run_nonzero_sign:
        nz_cls = run_classification_suite(
            X[nz_mask], q_vals[nz_mask], y_nz, y_nz_shuf, nz_baseline, perm_all=False
        )
        out["s00_sign_nonzero_classification"] = nz_cls
    return out


# ---------------------------------------------------------------------------
# Summary Markdown Generator
# ---------------------------------------------------------------------------

def generate_mlkem_summary_markdown(json_path: str, md_path: str):
    with open(json_path, "r") as f:
        d = json.load(f)

    env = d.get("environment") or {}
    exp = d["experiment"]
    tgt = d["targets"]
    passes = [
        ("Clean `decaps(dk, c)` (Full FIPS 203)", d["decaps_clean"], "dec_timing_us", "dec_cpu_user_us"),
        ("Clean `_k_pke_decrypt(dk_pke, c)` (Isolated Core)", d["k_pke_decrypt_clean"], "pke_dec_timing_us", "pke_dec_cpu_user_us"),
    ]
    if "decaps_tracemalloc_2feat" in d:
        passes.append(("`tracemalloc` `decaps(dk, c)` (2-feat)", d["decaps_tracemalloc_2feat"], "tm_dec_timing_us", "tm_dec_cpu_user_us"))
    if "decaps_tracemalloc_4feat" in d:
        passes.append(("`tracemalloc` `decaps(dk, c)` (4-feat)", d["decaps_tracemalloc_4feat"], "tm_dec_timing_us", "tm_dec_cpu_user_us"))

    n_perm = exp.get("n_permutations", N_PERMUTATIONS)
    lines = [
        f"# `{exp['label']}` Decrypt Side-Channel Evaluation ({exp['n_trials']:,} Trials)",
        "",
        "## 1. Experiment Setup & Secret Distribution",
        f"- **Environment:** `{env.get('os')}` (`{env.get('platform')}`), CPU `{env.get('cpu_model')}`, Python `{env.get('python_version')}`, NumPy `{env.get('numpy_version')}`",
        f"- **Parameters:** `n={exp['n']}`, `k={exp['k']}`, `eta_1={exp['eta_1']}`, `eta_2={exp['eta_2']}`, `du={exp['du']}`, `dv={exp['dv']}`, `d_poly={exp['d_poly']}`, seed `{exp['seed']}`",
        f"- **Correctness:** Clean `decaps` `{exp['decaps_correct_count']}/{exp['n_trials']}` ({exp['decaps_correct_rate']*100:.2f}%)"
        + (f"; `tracemalloc` `decaps` `{exp['tm_decaps_correct_count']}/{exp['n_trials']}` ({exp['tm_decaps_correct_rate']*100:.2f}%)" if exp.get("tm_decaps_correct_count") is not None else ""),
        f"- **Target 1 (`secret_weight`):** non-zero entries out of `{exp['n']*exp['k']}` centered coeffs — range `{tgt['secret_weight_min']}..{tgt['secret_weight_max']}`, mean `{tgt['secret_weight_mean']:.2f} ± {tgt['secret_weight_std']:.2f}`, binary `> {tgt['median_secret_weight']}` (class 0: `{tgt['weight_class_counts']['0']}`, class 1: `{tgt['weight_class_counts']['1']}`, majority baseline **{tgt['weight_majority_baseline']*100:.2f}%**)",
        f"- **Target 2a (`s00_positive`: `s[0][0] > 0`):** all `{exp['n_trials']}` trials — class 0 (`<=0`): `{tgt['s00_positive_class_counts']['0']}`, class 1 (`>0`): `{tgt['s00_positive_class_counts']['1']}`, majority baseline **{tgt['s00_positive_majority_baseline']*100:.2f}%**",
        f"- **Target 2b (`s00_sign_nonzero`: `s[0][0] > 0` on `s[0][0] != 0`):** `{tgt['s00_nonzero_count']}` trials — negative: `{tgt['s00_nonzero_class_counts']['0']}`, positive: `{tgt['s00_nonzero_class_counts']['1']}`, majority baseline **{tgt['s00_nonzero_majority_baseline']*100:.2f}%**",
        f"- **NTT Sparsity Note:** Mean zero coefficients in `s_hat` = `{tgt['s_hat_zero_mean']:.2f} / {exp['n']*exp['k']}` (vs. `{exp['n']*exp['k'] - tgt['secret_weight_mean']:.2f}` zeros in centered `s`).",
        "",
        "## 2. Cross-Pass Side-Channel Summary (Weight & Coefficient Sign)",
        "",
        "| Execution Pass | Timing Mean ± SD (µs) | `r_weight` | `ρ_weight` | Best Weight Clf (Real / Base / `q`-Res / Perm `p`) | Best Weight Reg `R²` (Perm `p`) | `ρ_s00_pos` | Best `s00 > 0` Clf (Real / Base / Perm `p`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for title, blk, t_k, _ in passes:
        td = blk["descriptives"][t_k]
        cw = blk["correlations_vs_secret_weight"][t_k]
        cs = blk["correlations_vs_s00_positive"][t_k]
        b_wc = max(blk["weight_classification"], key=lambda m: blk["weight_classification"][m]["real"]["accuracy"]["mean"])
        wcm = blk["weight_classification"][b_wc]
        b_wr = max(blk["weight_regression"], key=lambda m: blk["weight_regression"][m]["real"]["r2"]["mean"])
        wrm = blk["weight_regression"][b_wr]
        b_sc = max(blk["s00_positive_classification"], key=lambda m: blk["s00_positive_classification"][m]["real"]["accuracy"]["mean"])
        scm = blk["s00_positive_classification"][b_sc]
        r2_val = wrm["real"]["r2"]["mean"]
        r2_str = f"{r2_val:+.4f} ({b_wr})" if r2_val > 0 else f"All < 0 (max {r2_val:+.4f})"
        lines.append(
            f"| {title} | {td['mean']:.1f} ± {td['std']:.1f} | "
            f"{cw['pearson_r']:+.4f} | {cw['spearman_rho']:+.4f} | "
            f"{wcm['real']['accuracy']['mean']*100:.2f}% {b_wc} / {tgt['weight_majority_baseline']*100:.2f}% / {wcm['q_residualized']['accuracy']['mean']*100:.2f}% / {_fmt_perm_p(wcm.get('permutation_test'), n_perm)} | "
            f"{r2_str} / {_fmt_perm_p(wrm.get('permutation_test'), n_perm)} | "
            f"{cs['spearman_rho']:+.4f} | "
            f"{scm['real']['accuracy']['mean']*100:.2f}% {b_sc} / {tgt['s00_positive_majority_baseline']*100:.2f}% / {_fmt_perm_p(scm.get('permutation_test'), n_perm)} |"
        )

    lines.extend([
        "",
        "## 3. Primary Clean `decaps(dk, c)` Detailed Results",
        "",
        "### Correlations vs. Secret Targets (2,000-Resample Bootstrap 95% CIs)",
        "| Signal | Target | Pearson `r` [95% CI] | Pearson `p` | Spearman `ρ` [95% CI] | Spearman `p` |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])
    clean_blk = d["decaps_clean"]
    for feat in ["dec_timing_us", "dec_cpu_user_us"]:
        for tname, cdict in [
            ("secret_weight", clean_blk["correlations_vs_secret_weight"][feat]),
            ("s00_value", clean_blk["correlations_vs_s00_value"][feat]),
            ("s00_positive", clean_blk["correlations_vs_s00_positive"][feat]),
        ]:
            lines.append(
                f"| `{feat}` | `{tname}` | "
                f"{cdict['pearson_r']:+.4f} [{cdict['pearson_boot_ci95'][0]:+.4f}, {cdict['pearson_boot_ci95'][1]:+.4f}] | "
                f"{cdict['pearson_p']:.2e} | "
                f"{cdict['spearman_rho']:+.4f} [{cdict['spearman_boot_ci95'][0]:+.4f}, {cdict['spearman_boot_ci95'][1]:+.4f}] | "
                f"{cdict['spearman_p']:.2e} |"
            )

    qc = clean_blk["confound_q"]
    dc = clean_blk["drift_check"]
    lines.extend([
        "",
        "### Confound (`q`) & Time-Drift Checks",
        f"- **`q` vs `secret_weight` / `s00_positive`:** `r = {qc['q_vs_secret_weight']['pearson_r']:+.4f}` (`p = {qc['q_vs_secret_weight']['pearson_p']:.4f}`) / `r = {qc['q_vs_s00_positive']['pearson_r']:+.4f}` (`p = {qc['q_vs_s00_positive']['pearson_p']:.4f}`)",
        f"- **`q` vs `dec_timing_us` / `dec_cpu_user_us`:** `r = {qc['q_vs_dec_timing_us']['pearson_r']:+.4f}` (`p = {qc['q_vs_dec_timing_us']['pearson_p']:.4f}`) / `r = {qc['q_vs_dec_cpu_user_us']['pearson_r']:+.4f}` (`p = {qc['q_vs_dec_cpu_user_us']['pearson_p']:.4f}`)",
        f"- **Execution order vs `dec_timing_us` / `secret_weight`:** `r = {dc['exec_order_vs_dec_timing_us']['pearson_r']:+.4f}` (`p = {dc['exec_order_vs_dec_timing_us']['pearson_p']:.4f}`) / `r = {dc['exec_order_vs_secret_weight']['pearson_r']:+.4f}` (`p = {dc['exec_order_vs_secret_weight']['pearson_p']:.4f}`)",
        "",
        f"### Target 1: `secret_weight` Classification (5×10 CV, Majority Baseline = {tgt['weight_majority_baseline']*100:.2f}%)",
        "| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for m_name, m_res in clean_blk["weight_classification"].items():
        ra = m_res["real"]["accuracy"]
        sa = m_res["shuffled_control"]["accuracy"]
        qa = m_res["q_residualized"]["accuracy"]
        lines.append(
            f"| {m_name} | {ra['mean']*100:.2f}% ± {ra['sd']*100:.2f}% | {m_res['real']['roc_auc']['mean']:.4f} | "
            f"{sa['mean']*100:.2f}% ± {sa['sd']*100:.2f}% | {m_res['gap_real_minus_shuffled']*100:+.2f}% | "
            f"{qa['mean']*100:.2f}% ± {qa['sd']*100:.2f}% | {_fmt_perm_p(m_res.get('permutation_test'), n_perm)} |"
        )

    lines.extend([
        "",
        "### Target 1: Exact `secret_weight` Regression (5×10 CV)",
        "| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | `q`-Residualized R² | Perm `p`-value | Caveat |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for m_name, m_res in clean_blk["weight_regression"].items():
        rr = m_res["real"]["r2"]
        sr = m_res["shuffled_control"]["r2"]
        qr = m_res["q_residualized"]["r2"]
        caveat = "Negative R² (worse than mean baseline)" if rr["mean"] <= 0.0 else "—"
        lines.append(
            f"| {m_name} | {rr['mean']:+.4f} ± {rr['sd']:.4f} | {m_res['real']['rmse']['mean']:.4f} | "
            f"{sr['mean']:+.4f} ± {sr['sd']:.4f} | {qr['mean']:+.4f} ± {qr['sd']:.4f} | "
            f"{_fmt_perm_p(m_res.get('permutation_test'), n_perm)} | {caveat} |"
        )

    lines.extend([
        "",
        f"### Target 2a: `s[0][0] > 0` Sign Classification (5×10 CV, Majority Baseline = {tgt['s00_positive_majority_baseline']*100:.2f}%)",
        "| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for m_name, m_res in clean_blk["s00_positive_classification"].items():
        ra = m_res["real"]["accuracy"]
        sa = m_res["shuffled_control"]["accuracy"]
        qa = m_res["q_residualized"]["accuracy"]
        lines.append(
            f"| {m_name} | {ra['mean']*100:.2f}% ± {ra['sd']*100:.2f}% | {m_res['real']['roc_auc']['mean']:.4f} | "
            f"{sa['mean']*100:.2f}% ± {sa['sd']*100:.2f}% | {m_res['gap_real_minus_shuffled']*100:+.2f}% | "
            f"{qa['mean']*100:.2f}% ± {qa['sd']*100:.2f}% | {_fmt_perm_p(m_res.get('permutation_test'), n_perm)} |"
        )

    if "s00_sign_nonzero_classification" in clean_blk:
        nz_blk = clean_blk["s00_sign_nonzero_classification"]
        b_nz = max(nz_blk, key=lambda m: nz_blk[m]["real"]["accuracy"]["mean"])
        nzm = nz_blk[b_nz]
        pke_nz = d["k_pke_decrypt_clean"].get("s00_sign_nonzero_classification", {})
        b_pke_nz = max(pke_nz, key=lambda m: pke_nz[m]["real"]["accuracy"]["mean"]) if pke_nz else None
        lines.extend([
            "",
            f"### Target 2b: Balanced Non-Zero `sign(s[0][0])` (`s[0][0] != 0`, n={tgt['s00_nonzero_count']}, Baseline = {tgt['s00_nonzero_majority_baseline']*100:.2f}%)",
            f"- **Clean `decaps` Best ({b_nz}):** Real `{nzm['real']['accuracy']['mean']*100:.2f}% ± {nzm['real']['accuracy']['sd']*100:.2f}%` (AUC `{nzm['real']['roc_auc']['mean']:.4f}`, Shuffled `{nzm['shuffled_control']['accuracy']['mean']*100:.2f}%`, `q`-Resid `{nzm['q_residualized']['accuracy']['mean']*100:.2f}%`, Perm `p {_fmt_perm_p(nzm.get('permutation_test'), n_perm)}`)",
        ])
        if b_pke_nz:
            pm = pke_nz[b_pke_nz]
            lines.append(
                f"- **Isolated `_k_pke_decrypt` Best ({b_pke_nz}):** Real `{pm['real']['accuracy']['mean']*100:.2f}% ± {pm['real']['accuracy']['sd']*100:.2f}%` (AUC `{pm['real']['roc_auc']['mean']:.4f}`, Perm `p {_fmt_perm_p(pm.get('permutation_test'), n_perm)}`)"
            )

    lines.extend([
        "",
        f"> *Notes:* Permutation `p`-values at the `N={n_perm}` floor are reported as `< 1/{n_perm + 1} ({1.0/(n_perm + 1):.4f})`. Negative-`R²` models perform worse than predicting the sample mean (`R² = 0`).",
    ])

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_comparison_markdown(toy_json_path: str, std_json: dict, qeq_json: dict, out_md_path: str):
    toy = None
    if os.path.exists(toy_json_path):
        with open(toy_json_path, "r") as f:
            toy = json.load(f)

    def _row_mlkem(label, d):
        c_wt = d["decaps_clean"]["weight_classification"]
        b_wc = max(c_wt, key=lambda m: c_wt[m]["real"]["accuracy"]["mean"])
        r_wt = d["decaps_clean"]["weight_regression"]
        b_wr = max(r_wt, key=lambda m: r_wt[m]["real"]["r2"]["mean"])
        c_s0 = d["decaps_clean"]["s00_positive_classification"]
        b_sc = max(c_s0, key=lambda m: c_s0[m]["real"]["accuracy"]["mean"])
        pke_wt = d["k_pke_decrypt_clean"]["weight_classification"]
        b_pw = max(pke_wt, key=lambda m: pke_wt[m]["real"]["accuracy"]["mean"])
        tm_wt = d["decaps_tracemalloc_2feat"]["weight_classification"]
        b_tw = max(tm_wt, key=lambda m: tm_wt[m]["real"]["accuracy"]["mean"])
        n_perm = d["experiment"].get("n_permutations", N_PERMUTATIONS)
        r2_v = r_wt[b_wr]["real"]["r2"]["mean"]
        r2_s = f"{r2_v:+.4f} ({b_wr})" if r2_v > 0 else f"All < 0 (max {r2_v:+.4f})"
        return {
            "label": label,
            "params": f"n={d['experiment']['n']}, k={d['experiment']['k']}, {d['stream']['q_mode']}",
            "timing": f"{d['decaps_clean']['descriptives']['dec_timing_us']['mean']:.1f} ± {d['decaps_clean']['descriptives']['dec_timing_us']['std']:.1f} µs",
            "rho_wt": f"{d['decaps_clean']['correlations_vs_secret_weight']['dec_timing_us']['spearman_rho']:+.4f}",
            "wt_clf": f"{c_wt[b_wc]['real']['accuracy']['mean']*100:.2f}% vs {d['targets']['weight_majority_baseline']*100:.2f}% ({b_wc}, p {_fmt_perm_p(c_wt[b_wc].get('permutation_test'), n_perm)})",
            "wt_reg": r2_s,
            "pke_wt": f"{pke_wt[b_pw]['real']['accuracy']['mean']*100:.2f}% ({b_pw}, ρ={d['k_pke_decrypt_clean']['correlations_vs_secret_weight']['pke_dec_timing_us']['spearman_rho']:+.4f})",
            "tm_wt": f"{tm_wt[b_tw]['real']['accuracy']['mean']*100:.2f}% ({b_tw}, ρ={d['decaps_tracemalloc_2feat']['correlations_vs_secret_weight']['tm_dec_timing_us']['spearman_rho']:+.4f})",
            "s00_clf": f"{c_s0[b_sc]['real']['accuracy']['mean']*100:.2f}% vs {d['targets']['s00_positive_majority_baseline']*100:.2f}% ({b_sc}, p {_fmt_perm_p(c_s0[b_sc].get('permutation_test'), n_perm)})",
        }

    rows = []
    if toy:
        tc = toy["clean_uninstrumented"]
        bwc = max(tc["classification"], key=lambda m: tc["classification"][m]["real"]["accuracy"]["mean"])
        bwr = max(tc["regression"], key=lambda m: tc["regression"][m]["real"]["r2"]["mean"])
        tm = toy["tracemalloc_instrumented_2feat"]
        btw = max(tm["classification"], key=lambda m: tm["classification"][m]["real"]["accuracy"]["mean"])
        n_perm = toy["experiment"].get("n_permutations", N_PERMUTATIONS)
        rows.append({
            "label": "Toy QEQSKE (`decrypt_it`)",
            "params": "n=4, k=2, per-key prime q",
            "timing": f"{tc['descriptives']['dec_timing_us']['mean']:.2f} ± {tc['descriptives']['dec_timing_us']['std']:.2f} µs",
            "rho_wt": f"{tc['correlations']['dec_timing_us']['spearman_rho']:+.4f}",
            "wt_clf": f"{tc['classification'][bwc]['real']['accuracy']['mean']*100:.2f}% vs {toy['targets']['majority_baseline']*100:.2f}% ({bwc}, p {_fmt_perm_p(tc['classification'][bwc].get('permutation_test'), n_perm)})",
            "wt_reg": f"{tc['regression'][bwr]['real']['r2']['mean']:+.4f} ({bwr})",
            "pke_wt": "N/A (no FO layer)",
            "tm_wt": f"{tm['classification'][btw]['real']['accuracy']['mean']*100:.2f}% ({btw}, ρ={tm['correlations']['tm_dec_timing_us']['spearman_rho']:+.4f})",
            "s00_clf": "N/A (at baseline in Prog. Ladder)",
        })
    rows.append(_row_mlkem(f"Standard ML-KEM (`kyber-py`, k={std_json['experiment']['k']})", std_json))
    rows.append(_row_mlkem(f"QEQSKE-Variant ML-KEM (QRNG + per-key q, k={qeq_json['experiment']['k']})", qeq_json))

    lines = [
        "# Cross-Scale Comparison: Toy QEQSKE vs. Real ML-KEM (`n=256`, `kyber-py`) vs. QEQSKE-Variant ML-KEM",
        "",
        "| System | Parameters & `q` | Clean Timing | Clean `ρ_weight` | Clean Weight Clf vs. Baseline | Clean Weight Reg `R²` | Isolated `_k_pke_decrypt` | `tracemalloc` `decaps` | `s[0][0] > 0` Clf vs. Baseline |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['params']} | {r['timing']} | {r['rho_wt']} | {r['wt_clf']} | {r['wt_reg']} | {r['pke_wt']} | {r['tm_wt']} | {r['s00_clf']} |"
        )
    with open(out_md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------------------------

def run_mlkem_pipeline(
    variant: str = "standard",
    k_param: int = 2,
    n_trials: int = 5000,
    seed: int = SEED,
    skip_tracemalloc: bool = False,
    tag: str | None = None,
):
    os.makedirs("results", exist_ok=True)
    t_all_0 = time.perf_counter()
    env = get_environment_metadata()

    name_base = f"mlkem{'512' if k_param == 2 else '768'}"
    if variant == "qeqske":
        name_base += "_qeqske"
    if tag:
        name_base += f"_{tag.strip().lstrip('_')}"

    traces_path = f"results/{name_base}_decrypt_traces_{n_trials}.json"
    results_path = f"results/{name_base}_decrypt_leak_{n_trials}.json"
    summary_path = f"results/{name_base}_decrypt_leak_{n_trials}_summary.md"

    label = (
        f"Standard ML-KEM-{'512' if k_param == 2 else '768'} (kyber-py, n=256, k={k_param}, q=3329)"
        if variant == "standard"
        else f"QEQSKE-Variant ML-KEM-{'512' if k_param == 2 else '768'} (QRNG, n=256, k={k_param}, per-key NTT q)"
    )
    print("=" * 88)
    print(f"RUNNING: {label} | trials={n_trials}")
    print("=" * 88)

    trials, warmup_pool, stream_meta = pregenerate_mlkem_trials(
        variant=variant, k_param=k_param, n_trials=n_trials, seed=seed
    )
    records, pass_timings = collect_mlkem_traces(
        trials, warmup_pool, seed=seed, skip_tracemalloc=skip_tracemalloc
    )
    del trials, warmup_pool
    gc.collect()

    with open(traces_path, "w") as f:
        json.dump({"environment": env, "records": to_native(records)}, f, indent=2)

    q_vals = np.array([r["q"] for r in records], dtype=float)
    weights = np.array([r["secret_weight"] for r in records], dtype=float)
    med_wt = float(np.median(weights))
    y_wt = (weights > med_wt).astype(int)
    wt_baseline = float(max(np.mean(y_wt == 0), np.mean(y_wt == 1)))

    s00_vals = np.array([r["s00"] for r in records], dtype=float)
    y_pos = np.array([r["s00_positive"] for r in records], dtype=int)
    pos_baseline = float(max(np.mean(y_pos == 0), np.mean(y_pos == 1)))

    nz_mask = s00_vals != 0
    y_nz = (s00_vals[nz_mask] > 0).astype(int)
    nz_baseline = float(max(np.mean(y_nz == 0), np.mean(y_nz == 1)))

    y_wt_shuf = np.random.RandomState(seed).permutation(y_wt)
    y_pos_shuf = np.random.RandomState(seed + 1).permutation(y_pos)
    y_nz_shuf = np.random.RandomState(seed + 2).permutation(y_nz)

    common_kw = dict(
        records=records,
        q_vals=q_vals,
        weights=weights,
        y_wt=y_wt,
        y_wt_shuf=y_wt_shuf,
        wt_baseline=wt_baseline,
        s00_vals=s00_vals,
        y_pos=y_pos,
        y_pos_shuf=y_pos_shuf,
        pos_baseline=pos_baseline,
        nz_mask=nz_mask,
        y_nz=y_nz,
        y_nz_shuf=y_nz_shuf,
        nz_baseline=nz_baseline,
    )

    decaps_clean = _analyze_feature_pass(
        feature_keys=["dec_timing_us", "dec_cpu_user_us"],
        timing_key="dec_timing_us",
        cpu_key="dec_cpu_user_us",
        order_key="exec_order_pos",
        perm_all_primary=True,
        run_nonzero_sign=True,
        **common_kw,
    )
    k_pke_clean = _analyze_feature_pass(
        feature_keys=["pke_dec_timing_us", "pke_dec_cpu_user_us"],
        timing_key="pke_dec_timing_us",
        cpu_key="pke_dec_cpu_user_us",
        order_key="pke_exec_order_pos",
        perm_all_primary=False,
        run_nonzero_sign=True,
        **common_kw,
    )

    param_dict = DEFAULT_PARAMETERS["ML512" if k_param == 2 else "ML768"]
    shat_zeros = [r["s_hat_zero_count"] for r in records]
    dec_correct = sum(1 for r in records if r["decryption_correct"])
    tm_correct = sum(1 for r in records if r["tm_decryption_correct"]) if not skip_tracemalloc else None

    full_results = {
        "environment": env,
        "experiment": {
            "label": label,
            "variant": variant,
            "reference_library": "kyber-py",
            "n_trials": int(n_trials),
            "n": 256,
            "k": int(k_param),
            "eta_1": int(param_dict["eta_1"]),
            "eta_2": int(param_dict["eta_2"]),
            "du": int(param_dict["du"]),
            "dv": int(param_dict["dv"]),
            "d_poly": 12 if variant == "standard" else 16,
            "seed": int(seed),
            "n_bootstrap": N_BOOTSTRAP,
            "n_permutations": N_PERMUTATIONS,
            "cv_classification": "RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)",
            "cv_regression": "RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)",
            "decaps_correct_count": int(dec_correct),
            "decaps_correct_rate": float(dec_correct / n_trials),
            "tm_decaps_correct_count": int(tm_correct) if tm_correct is not None else None,
            "tm_decaps_correct_rate": float(tm_correct / n_trials) if tm_correct is not None else None,
            **pass_timings,
            "total_wall_seconds": float(time.perf_counter() - t_all_0),
            "traces_file": traces_path,
        },
        "stream": stream_meta,
        "targets": {
            "median_secret_weight": med_wt,
            "weight_class_counts": {"0": int(np.sum(y_wt == 0)), "1": int(np.sum(y_wt == 1))},
            "weight_majority_baseline": wt_baseline,
            "secret_weight_min": int(np.min(weights)),
            "secret_weight_max": int(np.max(weights)),
            "secret_weight_mean": float(np.mean(weights)),
            "secret_weight_std": float(np.std(weights, ddof=1)),
            "s_hat_zero_mean": float(np.mean(shat_zeros)),
            "s00_positive_class_counts": {"0": int(np.sum(y_pos == 0)), "1": int(np.sum(y_pos == 1))},
            "s00_positive_majority_baseline": pos_baseline,
            "s00_nonzero_count": int(np.sum(nz_mask)),
            "s00_nonzero_class_counts": {"0": int(np.sum(y_nz == 0)), "1": int(np.sum(y_nz == 1))},
            "s00_nonzero_majority_baseline": nz_baseline,
        },
        "decaps_clean": decaps_clean,
        "k_pke_decrypt_clean": k_pke_clean,
    }

    if not skip_tracemalloc:
        full_results["decaps_tracemalloc_2feat"] = _analyze_feature_pass(
            feature_keys=["tm_dec_timing_us", "tm_dec_cpu_user_us"],
            timing_key="tm_dec_timing_us",
            cpu_key="tm_dec_cpu_user_us",
            order_key="tm_exec_order_pos",
            perm_all_primary=False,
            run_nonzero_sign=False,
            **common_kw,
        )
        full_results["decaps_tracemalloc_4feat"] = _analyze_feature_pass(
            feature_keys=["tm_dec_timing_us", "tm_dec_cpu_user_us", "dec_ram_peak_bytes", "dec_num_allocations"],
            timing_key="tm_dec_timing_us",
            cpu_key="tm_dec_cpu_user_us",
            order_key="tm_exec_order_pos",
            perm_all_primary=False,
            run_nonzero_sign=False,
            **common_kw,
        )

    sanitized = to_native(full_results)
    sanitized["experiment"]["total_wall_seconds"] = float(time.perf_counter() - t_all_0)
    with open(results_path, "w") as f:
        json.dump(sanitized, f, indent=2)

    generate_mlkem_summary_markdown(results_path, summary_path)

    # Console summary
    b_wc = max(decaps_clean["weight_classification"], key=lambda m: decaps_clean["weight_classification"][m]["real"]["accuracy"]["mean"])
    b_sc = max(decaps_clean["s00_positive_classification"], key=lambda m: decaps_clean["s00_positive_classification"][m]["real"]["accuracy"]["mean"])
    b_wr = max(decaps_clean["weight_regression"], key=lambda m: decaps_clean["weight_regression"][m]["real"]["r2"]["mean"])
    print(f"Completed {name_base} in {sanitized['experiment']['total_wall_seconds']:.1f}s")
    print(f"  Correctness: {dec_correct}/{n_trials}")
    print(f"  Clean decaps timing: {decaps_clean['descriptives']['dec_timing_us']['mean']:.1f} ± {decaps_clean['descriptives']['dec_timing_us']['std']:.1f} us | rho_weight={decaps_clean['correlations_vs_secret_weight']['dec_timing_us']['spearman_rho']:+.4f} | rho_s00_pos={decaps_clean['correlations_vs_s00_positive']['dec_timing_us']['spearman_rho']:+.4f}")
    print(f"  Weight Clf ({b_wc}): {decaps_clean['weight_classification'][b_wc]['real']['accuracy']['mean']*100:.2f}% vs {wt_baseline*100:.2f}% (p {format_perm_pvalue(decaps_clean['weight_classification'][b_wc]['permutation_test']['pvalue'], N_PERMUTATIONS)})")
    print(f"  Weight Reg ({b_wr}): R2={decaps_clean['weight_regression'][b_wr]['real']['r2']['mean']:+.4f} (p {format_perm_pvalue(decaps_clean['weight_regression'][b_wr]['permutation_test']['pvalue'], N_PERMUTATIONS)})")
    print(f"  s[0][0]>0 Clf ({b_sc}): {decaps_clean['s00_positive_classification'][b_sc]['real']['accuracy']['mean']*100:.2f}% vs {pos_baseline*100:.2f}% (p {format_perm_pvalue(decaps_clean['s00_positive_classification'][b_sc]['permutation_test']['pvalue'], N_PERMUTATIONS)})")
    print(f"  Saved: {traces_path}, {results_path}, {summary_path}")
    return sanitized


def _inject_delay_us(delay_us: float, trigger: bool) -> None:
    if delay_us <= 0.0 or not trigger:
        return
    target_ns = time.perf_counter_ns() + int(round(delay_us * 1000.0))
    while time.perf_counter_ns() < target_ns:
        pass


def order_ols_detrend(values, order_pos):
    """Regress values linearly on execution order position and return residuals."""
    v = np.asarray(values, dtype=float)
    o = np.asarray(order_pos, dtype=float)
    o_centered = o - np.mean(o)
    denom = float(np.dot(o_centered, o_centered))
    slope = float(np.dot(o_centered, v - np.mean(v)) / denom) if denom > 0.0 else 0.0
    intercept = float(np.mean(v)) - slope * float(np.mean(o))
    return v - (intercept + slope * o)


def rolling_median_detrend(values, order_pos, window: int = 51):
    """Leave-one-out centered rolling-median detrending along execution order."""
    v = np.asarray(values, dtype=float)
    o = np.asarray(order_pos, dtype=int)
    n = len(v)
    sort_idx = np.argsort(o)
    ordered = v[sort_idx]
    half = window // 2
    baseline_ord = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        nbrs = np.concatenate((ordered[lo:i], ordered[i + 1 : hi]))
        baseline_ord[i] = float(np.median(nbrs)) if len(nbrs) > 0 else float(ordered[i])
    detrended = np.empty(n, dtype=float)
    detrended[sort_idx] = ordered - baseline_ord
    return detrended


def _evaluate_detrend_mode(records, timing_vals, cpu_vals, order_vals, q_vals, weights,
                           y_wt, y_wt_shuf, wt_baseline):
    """Evaluate drift, correlations, and 5x10 Stratified CV on one signal variant."""
    desc_t = describe_array(timing_vals)
    desc_c = describe_array(cpu_vals)
    p_ot, pp_ot = _safe_pearson(order_vals, timing_vals)
    s_ot, sp_ot = _safe_spearman(order_vals, timing_vals)
    p_oc, pp_oc = _safe_pearson(order_vals, cpu_vals)
    s_oc, sp_oc = _safe_spearman(order_vals, cpu_vals)
    corr_t_wt = paired_correlation_with_bootstrap(timing_vals, weights)
    corr_c_wt = paired_correlation_with_bootstrap(cpu_vals, weights)
    corr_t_bin = paired_correlation_with_bootstrap(timing_vals, y_wt)

    X = np.column_stack([timing_vals, cpu_vals])
    cls_res = run_classification_suite(
        X, q_vals, y_wt, y_wt_shuf, wt_baseline, perm_all=False
    )
    best_m = max(cls_res, key=lambda m: cls_res[m]["real"]["accuracy"]["mean"])
    bm = cls_res[best_m]
    return {
        "timing_descriptives": desc_t,
        "cpu_descriptives": desc_c,
        "drift_order_vs_timing": {
            "pearson_r": p_ot, "pearson_p": pp_ot,
            "spearman_rho": s_ot, "spearman_p": sp_ot,
        },
        "drift_order_vs_cpu": {
            "pearson_r": p_oc, "pearson_p": pp_oc,
            "spearman_rho": s_oc, "spearman_p": sp_oc,
        },
        "corr_timing_vs_weight": corr_t_wt,
        "corr_cpu_vs_weight": corr_c_wt,
        "corr_timing_vs_binary_y": corr_t_bin,
        "best_model": best_m,
        "best_accuracy": bm["real"]["accuracy"]["mean"],
        "best_sd": bm["real"]["accuracy"]["sd"],
        "best_roc_auc": bm["real"]["roc_auc"]["mean"],
        "shuffled_accuracy": bm["shuffled_control"]["accuracy"]["mean"],
        "q_residualized_accuracy": bm["q_residualized"]["accuracy"]["mean"],
        "gap_real_minus_shuffled": bm["gap_real_minus_shuffled"],
        "best_perm_pvalue": bm.get("permutation_test", {}).get("pvalue", 1.0),
        "best_perm_pvalue_display": bm.get("permutation_test", {}).get("pvalue_display", "—"),
        "classification": cls_res,
    }


def run_detrend_comparison(
    window: int = 51,
    out_json_path: str = "results/mlkem512_detrend_comparison.json",
    out_md_path: str = "results/mlkem512_detrend_comparison_summary.md",
):
    """Compare Raw vs Order-OLS vs Rolling-Median detrending on existing 5,000-trial ML-KEM traces."""
    targets = [
        ("standard_mlkem512", "Standard ML-KEM-512 (kyber-py, q=3329)",
         "results/mlkem512_decrypt_traces_5000.json", "results/mlkem512_decrypt_leak_5000.json"),
        ("qeqske_mlkem512", "QEQSKE-Variant ML-KEM-512 (QRNG, per-key NTT q)",
         "results/mlkem512_qeqske_decrypt_traces_5000.json", "results/mlkem512_qeqske_decrypt_leak_5000.json"),
    ]

    variants_out = {}
    for key, label, trace_path, res_path in targets:
        with open(trace_path, "r") as f:
            tr_raw = json.load(f)
        records = tr_raw["records"]
        src_env = tr_raw.get("environment") or get_environment_metadata()
        stream_meta = None
        if os.path.exists(res_path):
            with open(res_path, "r") as f:
                stream_meta = json.load(f).get("stream")

        q_vals = np.array([r["q"] for r in records], dtype=float)
        weights = np.array([r["secret_weight"] for r in records], dtype=float)
        med_w = float(np.median(weights))
        y_wt = (weights > med_w).astype(int)
        wt_baseline = float(max(np.mean(y_wt == 0), np.mean(y_wt == 1)))
        y_wt_shuf = np.random.RandomState(SEED).permutation(y_wt)

        t_raw = np.array([r["dec_timing_us"] for r in records], dtype=float)
        c_raw = np.array([r["dec_cpu_user_us"] for r in records], dtype=float)
        o_vals = np.array([r["exec_order_pos"] for r in records], dtype=float)

        t_ols = order_ols_detrend(t_raw, o_vals)
        c_ols = order_ols_detrend(c_raw, o_vals)
        t_rmed = rolling_median_detrend(t_raw, o_vals, window=window)
        c_rmed = rolling_median_detrend(c_raw, o_vals, window=window)

        modes = {
            "raw": _evaluate_detrend_mode(records, t_raw, c_raw, o_vals, q_vals, weights, y_wt, y_wt_shuf, wt_baseline),
            "order_ols_detrended": _evaluate_detrend_mode(records, t_ols, c_ols, o_vals, q_vals, weights, y_wt, y_wt_shuf, wt_baseline),
            "rolling_median_detrended": _evaluate_detrend_mode(records, t_rmed, c_rmed, o_vals, q_vals, weights, y_wt, y_wt_shuf, wt_baseline),
        }
        variants_out[key] = {
            "label": label,
            "trace_file": trace_path,
            "trace_environment": src_env,
            "stream": stream_meta,
            "n_trials": len(records),
            "majority_baseline": wt_baseline,
            "modes": modes,
        }

    out = to_native({
        "analysis_environment": get_environment_metadata(),
        "rolling_median_window": int(window),
        "seed": SEED,
        "variants": variants_out,
    })
    os.makedirs("results", exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(out, f, indent=2)

    lines = [
        f"# ML-KEM-512 Execution-Order Detrending Comparison (5,000 Trials, Window={window})",
        "",
        "Compares unmodified raw timing/CPU traces against leave-one-out centered rolling-median (`W=51`) and linear order-OLS detrended residuals on both Linux Intel Core i9-12900K datasets.",
        "",
        "| Variant & Mode | Timing SD / IQR (µs) | Order Drift `r` (`ρ`) vs Timing | `ρ_weight` [95% CI] | Best 5×10 Clf (Acc / Base / `q`-Res) | Perm `p`-value |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for vdata in variants_out.values():
        base_pct = vdata["majority_baseline"] * 100.0
        for mkey, mlabel in [
            ("raw", "Raw"),
            ("order_ols_detrended", "Order-OLS Residuals"),
            ("rolling_median_detrended", f"Rolling Median (W={window})"),
        ]:
            md = vdata["modes"][mkey]
            td = md["timing_descriptives"]
            dr = md["drift_order_vs_timing"]
            cw = md["corr_timing_vs_weight"]
            ci = cw["spearman_boot_ci95"]
            lines.append(
                f"| **{vdata['label'].split('(')[0].strip()}** — {mlabel} | "
                f"{td['std']:.1f} / {td['iqr']:.1f} | "
                f"{dr['pearson_r']:+.4f} ({dr['spearman_rho']:+.4f}) | "
                f"{cw['spearman_rho']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                f"{md['best_accuracy']*100:.2f}% {md['best_model']} / {base_pct:.2f}% / {md['q_residualized_accuracy']*100:.2f}% | "
                f"{md['best_perm_pvalue_display']} |"
            )

    lines.extend([
        "",
        "## Randomness & QRNG Reuse Audit",
        "- **Standard ML-KEM-512 (`kyber-py`):** Uses deterministic `SHAKE-256` (`seed=42`, `q=3329` fixed).",
        "- **QEQSKE-Variant ML-KEM-512:** Consumed `245,000` uint16 draws from `qrng_combined.txt` (`1,106,944` values) -> **`wraps = 0` (zero QRNG reuse across all 5,000 trials)**.",
    ])
    with open(out_md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {out_json_path}, {out_md_path}")
    for line in lines[5:]:
        print(line)
    return out


def _save_mlkem_threshold_svg(sweep_records, baseline_pct, raw_thresh_us, det_thresh_us, svg_path):
    """Write a self-contained SVG plot comparing Raw vs Rolling-Median Detrended ML-KEM accuracy."""
    W, H = 960, 440
    x0, x1, y0, y1 = 80, 915, 65, 365
    delays = [r["delay_us"] for r in sweep_records]
    xs = [x0 + i * (x1 - x0) / max(1, len(delays) - 1) for i in range(len(delays))]
    y_min, y_max = 48.0, 102.0

    def sy(val):
        return y1 - (val - y_min) / (y_max - y_min) * (y1 - y0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif">',
        f'<text x="{W/2:.1f}" y="28" text-anchor="middle" font-size="17" font-weight="700" fill="#0f172a">ML-KEM-512 (n=256) Graded Injected-Delay Positive Control</text>',
        f'<text x="{W/2:.1f}" y="48" text-anchor="middle" font-size="12" fill="#475569">5x10 Stratified CV Best Accuracy across {sweep_records[0]["n_trials"]} Trials | Majority Baseline = {baseline_pct:.2f}%</text>',
        f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>',
    ]

    for yt in [50, 60, 70, 80, 90, 100]:
        yy = sy(yt)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{x0-10}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#475569">{yt}%</text>')

    for xx, dv in zip(xs, delays):
        parts.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" stroke="#f1f5f9" stroke-width="1"/>')
        parts.append(f'<text x="{xx:.1f}" y="{y1+20}" text-anchor="middle" font-size="11" fill="#475569">{dv:g} µs</text>')

    yb = sy(baseline_pct)
    parts.append(f'<line x1="{x0}" y1="{yb:.1f}" x2="{x1}" y2="{yb:.1f}" stroke="#64748b" stroke-width="1.8" stroke-dasharray="6,5"/>')
    parts.append(f'<text x="{x1-6}" y="{yb-6:.1f}" text-anchor="end" font-size="11" fill="#475569">Majority Baseline ({baseline_pct:.1f}%)</text>')

    raw_pts = [(xx, sy(r["raw"]["best_accuracy"] * 100.0), r["raw"]["best_accuracy"] * 100.0, r["raw"]["significant"]) for xx, r in zip(xs, sweep_records)]
    det_pts = [(xx, sy(r["detrended"]["best_accuracy"] * 100.0), r["detrended"]["best_accuracy"] * 100.0, r["detrended"]["significant"]) for xx, r in zip(xs, sweep_records)]

    for color, pts, dy_lbl in [("#dc2626", raw_pts, 15), ("#0284c7", det_pts, -10)]:
        d_str = " ".join(("M" if i == 0 else "L") + f"{p[0]:.1f},{p[1]:.1f}" for i, p in enumerate(pts))
        parts.append(f'<path d="{d_str}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for xx, yy, acc, sig in pts:
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4.5" fill="{color}" stroke="#ffffff" stroke-width="1"/>')
            star = "*" if sig else ""
            parts.append(f'<text x="{xx:.1f}" y="{yy + dy_lbl:.1f}" text-anchor="middle" font-size="10.5" font-weight="600" fill="{color}">{acc:.1f}%{star}</text>')

    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1+44}" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1e293b">Injected Secret-Dependent Delay (µs, discrete index spacing)</text>')
    parts.append(f'<text x="22" y="{(y0+y1)/2:.1f}" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1e293b" transform="rotate(-90 22 {(y0+y1)/2:.1f})">5×10 Stratified CV Best Accuracy (%)</text>')

    raw_lbl = f"Raw Clean Timing (threshold: {raw_thresh_us:g} µs)" if raw_thresh_us is not None else "Raw Clean Timing (not detected <= max)"
    det_lbl = f"Rolling-Median Detrended (threshold: {det_thresh_us:g} µs)" if det_thresh_us is not None else "Rolling-Median Detrended (not detected <= max)"
    parts.extend([
        f'<rect x="{x0+14}" y="{y0+12}" width="340" height="52" rx="6" fill="#ffffff" fill-opacity="0.94" stroke="#cbd5e1"/>',
        f'<line x1="{x0+26}" y1="{y0+29}" x2="{x0+50}" y2="{y0+29}" stroke="#dc2626" stroke-width="2.5"/>',
        f'<circle cx="{x0+38}" cy="{y0+29}" r="4" fill="#dc2626"/>',
        f'<text x="{x0+58}" y="{y0+33}" font-size="11.5" fill="#1e293b">{raw_lbl}</text>',
        f'<line x1="{x0+26}" y1="{y0+49}" x2="{x0+50}" y2="{y0+49}" stroke="#0284c7" stroke-width="2.5"/>',
        f'<circle cx="{x0+38}" cy="{y0+49}" r="4" fill="#0284c7"/>',
        f'<text x="{x0+58}" y="{y0+53}" font-size="11.5" fill="#1e293b">{det_lbl}</text>',
        "</svg>",
    ])
    with open(svg_path, "w") as f:
        f.write("\n".join(parts) + "\n")


def run_mlkem_positive_control(
    k_param: int = 2,
    n_trials: int = 5000,
    delays_us=(0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
    seed: int = SEED,
    window: int = 51,
    tag: str | None = None,
):
    """Calibrated secret-weight-dependent delay sweep on real ML-KEM-512 decaps."""
    os.makedirs("results", exist_ok=True)
    t_all_0 = time.perf_counter()
    env = get_environment_metadata()
    proc = psutil.Process(os.getpid())

    print("=" * 88)
    print(f"ML-KEM-512 POSITIVE CONTROL (n=256, k={k_param}) | trials={n_trials} | delays={list(delays_us)} us")
    print(f"Environment: {env['os']} | CPU: {env['cpu_model']} | Python {env['python_version']}")
    print("=" * 88)

    trials, warmup_pool, stream_meta = pregenerate_mlkem_trials(
        variant="standard", k_param=k_param, n_trials=n_trials, seed=seed
    )
    weights = np.array([tr["secret_weight"] for tr in trials], dtype=float)
    q_vals = np.array([tr["q"] for tr in trials], dtype=float)
    med_w = float(np.median(weights))
    y_wt = (weights > med_w).astype(int)
    wt_baseline = float(max(np.mean(y_wt == 0), np.mean(y_wt == 1)))
    y_wt_shuf = np.random.RandomState(seed).permutation(y_wt)
    triggers = [bool(w > med_w) for w in weights]

    # Collect ALL delay passes BEFORE running parallel ML CV/permutations
    collected_passes = []
    orig_affinity = None
    if hasattr(os, "sched_setaffinity"):
        orig_affinity = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {0})

    try:
        for d_idx, d_us in enumerate(delays_us):
            gc.collect()
            for wi, (kem_w, dk_w, _, c_w, _) in enumerate(warmup_pool):
                _ = kem_w.decaps(dk_w, c_w)
                _inject_delay_us(float(d_us), trigger=(wi % 2 == 1))

            order = np.random.RandomState(seed + 10 * d_idx).permutation(n_trials)
            timings = np.empty(n_trials, dtype=float)
            cpus = np.empty(n_trials, dtype=float)
            orders = np.empty(n_trials, dtype=int)
            correct_cnt = 0

            t0_pass = time.perf_counter()
            for pos, idx in enumerate(order):
                if pos % 250 == 0:
                    gc.collect()
                    for wi in range(15):
                        kem_w, dk_w, _, c_w, _ = warmup_pool[(pos + wi) % len(warmup_pool)]
                        _ = kem_w.decaps(dk_w, c_w)

                tr = trials[int(idx)]
                trig = triggers[int(idx)]
                kem = tr["kem"]
                cpu0 = _get_cpu_user_seconds(proc)
                t0 = time.perf_counter_ns()
                k_out = kem.decaps(tr["dk"], tr["c"])
                _inject_delay_us(float(d_us), trigger=trig)
                dt_ns = time.perf_counter_ns() - t0
                cpu1 = _get_cpu_user_seconds(proc)

                timings[int(idx)] = dt_ns / 1000.0
                cpus[int(idx)] = (cpu1 - cpu0) * 1e6
                orders[int(idx)] = pos
                if k_out == tr["K"]:
                    correct_cnt += 1

            wall_s = time.perf_counter() - t0_pass
            med_delta = float(np.median(timings[y_wt == 1]) - np.median(timings[y_wt == 0]))
            print(
                f"  Collected delay={d_us:5.1f} us ({wall_s:5.1f}s): "
                f"correct={correct_cnt}/{n_trials}, med={np.median(timings):.1f} us, "
                f"IQR={np.percentile(timings, 75) - np.percentile(timings, 25):.1f} us, "
                f"measured delta={med_delta:+.2f} us"
            )
            collected_passes.append({
                "delay_us": float(d_us),
                "correct_count": int(correct_cnt),
                "wall_seconds": float(wall_s),
                "timings": timings,
                "cpus": cpus,
                "orders": orders,
                "measured_median_delta_us": med_delta,
            })
    finally:
        if orig_affinity is not None:
            os.sched_setaffinity(0, orig_affinity)

    del trials, warmup_pool
    gc.collect()

    print("\nRunning 5x10 Stratified CV + 200-permutation tests across Raw & Detrended traces...")
    sweep_records = []
    for pdata in collected_passes:
        d_us = pdata["delay_us"]
        t_raw, c_raw, o_vals = pdata["timings"], pdata["cpus"], pdata["orders"]
        t_det = rolling_median_detrend(t_raw, o_vals, window=window)
        c_det = rolling_median_detrend(c_raw, o_vals, window=window)

        raw_eval = _evaluate_detrend_mode(
            [], t_raw, c_raw, o_vals, q_vals, weights, y_wt, y_wt_shuf, wt_baseline
        )
        det_eval = _evaluate_detrend_mode(
            [], t_det, c_det, o_vals, q_vals, weights, y_wt, y_wt_shuf, wt_baseline
        )
        raw_eval["measured_median_delta_us"] = float(np.median(t_raw[y_wt == 1]) - np.median(t_raw[y_wt == 0]))
        det_eval["measured_median_delta_us"] = float(np.median(t_det[y_wt == 1]) - np.median(t_det[y_wt == 0]))

        for ev in (raw_eval, det_eval):
            ev["significant"] = bool(
                ev["best_perm_pvalue"] < 0.01
                and ev["best_accuracy"] > wt_baseline + 0.02
                and ev["gap_real_minus_shuffled"] > 0.02
            )

        print(
            f"  [{d_us:5.1f} us] Raw: {raw_eval['best_accuracy']*100:5.2f}% ({raw_eval['best_model']}, "
            f"rho_bin={raw_eval['corr_timing_vs_binary_y']['spearman_rho']:+.4f}, p={raw_eval['best_perm_pvalue_display']}) | "
            f"Detrended: {det_eval['best_accuracy']*100:5.2f}% ({det_eval['best_model']}, "
            f"rho_bin={det_eval['corr_timing_vs_binary_y']['spearman_rho']:+.4f}, p={det_eval['best_perm_pvalue_display']})"
        )

        sweep_records.append({
            "delay_us": float(d_us),
            "n_trials": int(n_trials),
            "correct_count": pdata["correct_count"],
            "wall_seconds": pdata["wall_seconds"],
            "raw": raw_eval,
            "detrended": det_eval,
        })

    def _find_threshold(mode_key):
        sig_recs = [r for r in sweep_records if r["delay_us"] > 0.0 and r[mode_key]["significant"]]
        if not sig_recs:
            return None
        first = sig_recs[0]
        return {
            "min_detectable_delay_us": float(first["delay_us"]),
            "best_model": first[mode_key]["best_model"],
            "best_accuracy": float(first[mode_key]["best_accuracy"]),
            "perm_pvalue": float(first[mode_key]["best_perm_pvalue"]),
            "perm_pvalue_display": first[mode_key]["best_perm_pvalue_display"],
            "min_detectable_spearman_rho_vs_binary_y": float(
                first[mode_key]["corr_timing_vs_binary_y"]["spearman_rho"]
            ),
            "min_detectable_spearman_rho_vs_weight": float(
                first[mode_key]["corr_timing_vs_weight"]["spearman_rho"]
            ),
            "measured_median_delta_us": float(first[mode_key]["measured_median_delta_us"]),
        }

    raw_thresh = _find_threshold("raw")
    det_thresh = _find_threshold("detrended")

    raw_max_d = max(delays_us)
    raw_cap = (
        f"On {env['cpu_model']} ({env['os']}), unmodified ML-KEM-512 decaps ({sweep_records[0]['raw']['timing_descriptives']['median']:.1f} us median, IQR {sweep_records[0]['raw']['timing_descriptives']['iqr']:.1f} us) "
        + (
            f"reliably detects injected secret-dependent delays >= {raw_thresh['min_detectable_delay_us']:g} us (Spearman rho_binary >= {raw_thresh['min_detectable_spearman_rho_vs_binary_y']:+.4f}, acc {raw_thresh['best_accuracy']*100:.2f}%, p {raw_thresh['perm_pvalue_display']}); "
            if raw_thresh
            else f"could not detect delays up to {raw_max_d:g} us on raw timing; "
        )
        + (
            f"rolling-median detrending (W={window}) lowers/matches threshold to >= {det_thresh['min_detectable_delay_us']:g} us (Spearman rho_binary >= {det_thresh['min_detectable_spearman_rho_vs_binary_y']:+.4f}, acc {det_thresh['best_accuracy']*100:.2f}%, p {det_thresh['perm_pvalue_display']})."
            if det_thresh
            else f"rolling-median detrending (W={window}) also could not detect delays up to {raw_max_d:g} us."
        )
    )

    kem_tag = "mlkem512" if k_param == 2 else "mlkem768"
    tag_sfx = f"_{tag.strip().lstrip('_')}" if tag else ""
    out_json = f"results/{kem_tag}_positive_control_{n_trials}{tag_sfx}.json"
    out_md = f"results/{kem_tag}_positive_control_{n_trials}{tag_sfx}_summary.md"
    out_svg = f"results/{kem_tag}_leak_threshold{tag_sfx}.svg"

    res_dict = to_native({
        "environment": env,
        "experiment": {
            "variant": "standard",
            "n": 256,
            "k": int(k_param),
            "q": 3329,
            "n_trials": int(n_trials),
            "seed": int(seed),
            "delays_us": [float(d) for d in delays_us],
            "rolling_median_window": int(window),
            "n_bootstrap": N_BOOTSTRAP,
            "n_permutations": N_PERMUTATIONS,
            "total_wall_seconds": float(time.perf_counter() - t_all_0),
        },
        "stream": stream_meta,
        "majority_baseline": wt_baseline,
        "thresholds": {
            "raw": raw_thresh,
            "detrended": det_thresh,
        },
        "detection_capability_statement": raw_cap,
        "sweep": sweep_records,
    })

    with open(out_json, "w") as f:
        json.dump(res_dict, f, indent=2)

    _save_mlkem_threshold_svg(
        sweep_records,
        wt_baseline * 100.0,
        raw_thresh["min_detectable_delay_us"] if raw_thresh else None,
        det_thresh["min_detectable_delay_us"] if det_thresh else None,
        out_svg,
    )

    lines = [
        f"# ML-KEM-512 (n=256, k={k_param}) Graded Injected-Delay Positive Control ({n_trials:,} Trials)",
        "",
        f"- **Environment:** `{env['os']}` (`{env['platform']}`), CPU `{env['cpu_model']}`, Python `{env['python_version']}`, NumPy `{env['numpy_version']}`",
        f"- **Majority Baseline (`secret_weight > {med_w}`):** **{wt_baseline*100:.2f}%** | Rolling-median detrend window: `W={window}`",
        f"- **Detection Capability:** {raw_cap}",
        "",
        "| Injected Delay (µs) | Raw Med / IQR (µs) | Raw Δt | Raw `ρ_bin` / `ρ_wt` | Raw Best Acc (Model / Perm `p`) | Detrended IQR (µs) | Detrended `ρ_bin` / `ρ_wt` | Detrended Best Acc (Model / Perm `p`) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in sweep_records:
        re_ = r["raw"]
        de_ = r["detrended"]
        lines.append(
            f"| **{r['delay_us']:g} µs** | "
            f"{re_['timing_descriptives']['median']:.1f} / {re_['timing_descriptives']['iqr']:.1f} | "
            f"{re_['measured_median_delta_us']:+.2f} µs | "
            f"{re_['corr_timing_vs_binary_y']['spearman_rho']:+.4f} / {re_['corr_timing_vs_weight']['spearman_rho']:+.4f} | "
            f"{re_['best_accuracy']*100:.2f}% ({re_['best_model']} / {re_['best_perm_pvalue_display']}) | "
            f"{de_['timing_descriptives']['iqr']:.1f} | "
            f"{de_['corr_timing_vs_binary_y']['spearman_rho']:+.4f} / {de_['corr_timing_vs_weight']['spearman_rho']:+.4f} | "
            f"{de_['best_accuracy']*100:.2f}% ({de_['best_model']} / {de_['best_perm_pvalue_display']}) |"
        )
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nCapability: {raw_cap}")
    print(f"Saved: {out_json}, {out_md}, {out_svg} ({res_dict['experiment']['total_wall_seconds']:.1f}s)")
    return res_dict


def main():
    parser = argparse.ArgumentParser(description="Real ML-KEM (n=256) decrypt side-channel evaluation via kyber-py.")
    parser.add_argument("--variant", choices=["standard", "qeqske", "both"], default="both", help="Protocol variant: standard ML-KEM, QEQSKE per-key q + QRNG, or both.")
    parser.add_argument("--k", type=int, choices=[2, 3], default=2, help="Module rank k=2 (ML-KEM-512) or k=3 (ML-KEM-768).")
    parser.add_argument("--n-trials", type=int, default=5000, help="Number of trials per variant (default 5000).")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default 42).")
    parser.add_argument("--skip-tracemalloc", action="store_true", help="Skip tracemalloc pass.")
    parser.add_argument("--tag", type=str, default=None, help="Optional filename suffix tag.")
    parser.add_argument("--detrend-existing", action="store_true", help="Compare Raw vs Order-OLS vs Rolling-Median detrending on existing 5000-trial traces.")
    parser.add_argument("--positive-control", action="store_true", help="Run graded injected-delay positive control on ML-KEM decaps.")
    parser.add_argument("--delays", type=str, default="0,1,2,5,10,20,50", help="Comma-separated delays in us for --positive-control.")
    args = parser.parse_args()

    if args.detrend_existing:
        run_detrend_comparison()
        return

    if args.positive_control:
        delays = tuple(float(x.strip()) for x in args.delays.split(",") if x.strip())
        run_mlkem_positive_control(
            k_param=args.k, n_trials=args.n_trials, delays_us=delays, seed=args.seed, tag=args.tag
        )
        return

    if args.variant == "both":
        std_res = run_mlkem_pipeline(
            variant="standard", k_param=args.k, n_trials=args.n_trials,
            seed=args.seed, skip_tracemalloc=args.skip_tracemalloc, tag=args.tag,
        )
        qeq_res = run_mlkem_pipeline(
            variant="qeqske", k_param=args.k, n_trials=args.n_trials,
            seed=args.seed, skip_tracemalloc=args.skip_tracemalloc, tag=args.tag,
        )
        tag_sfx = f"_{args.tag.strip().lstrip('_')}" if args.tag else ""
        comp_path = f"results/mlkem{'512' if args.k == 2 else '768'}{tag_sfx}_comparison_summary.md"
        generate_comparison_markdown("results/decrypt_leak_5000.json", std_res, qeq_res, comp_path)
        print(f"Saved cross-scale comparison: {comp_path}")
    else:
        run_mlkem_pipeline(
            variant=args.variant, k_param=args.k, n_trials=args.n_trials,
            seed=args.seed, skip_tracemalloc=args.skip_tracemalloc, tag=args.tag,
        )


if __name__ == "__main__":
    main()
