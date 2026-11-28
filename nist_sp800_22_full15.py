#!/usr/bin/env python3
"""
Full 15-Test NIST SP 800-22 Rev. 1a Randomness Evaluation
=========================================================
Runs all 15 statistical tests from NIST SP 800-22 Rev. 1a on matched
1,000,000-bit streams (and multi-block pass proportions across full corpora) for:
  1. ANU Optical QRNG (qrng_combined.txt)
  2. IBMQ Superconducting QRNG (fixbit-IBMQ.txt, shared by HCL Software)
  3. Mersenne Twister MT19937 Baseline (mersenne_large_baseline.txt)

Outputs: results/nist_sp800_22_full15.json
"""

import json
import math
import os
import time
import numpy as np
from scipy.special import erfc, gammaincc

from env_metadata import get_environment_metadata

ALPHA = 0.01
N_BITS_SLICE = 1_000_000


def load_uint16_as_bits(path: str, limit_bits: int | None = None) -> np.ndarray:
    with open(path, "r", encoding="ascii") as f:
        nums = np.array([int(x.strip()) for x in f.read().split(",") if x.strip()], dtype=np.uint16)
    shifts = np.arange(15, -1, -1, dtype=np.uint16)
    bits = ((nums[:, None] >> shifts[None, :]) & 1).astype(np.uint8).ravel()
    if limit_bits is not None:
        bits = bits[:limit_bits]
    return bits


def load_bitstring_file(path: str, limit_bits: int | None = None) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read().strip()
    if limit_bits is not None:
        raw = raw[:limit_bits]
    arr = np.frombuffer(raw, dtype=np.uint8) - ord("0")
    return arr.astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. Frequency (Monobit) Test
# ---------------------------------------------------------------------------
def test_01_monobit(bits: np.ndarray) -> dict:
    n = len(bits)
    s = int(2 * int(bits.sum()) - n)
    s_obs = abs(s) / math.sqrt(n)
    p = float(erfc(s_obs / math.sqrt(2.0)))
    return {"test": "1. Frequency (Monobit)", "p_value": p, "pass": bool(p >= ALPHA), "stat": float(s_obs)}


# ---------------------------------------------------------------------------
# 2. Block Frequency Test (M = 128)
# ---------------------------------------------------------------------------
def test_02_block_frequency(bits: np.ndarray, M: int = 128) -> dict:
    n = len(bits)
    N = n // M
    blocks = bits[: N * M].reshape(N, M)
    pi = blocks.mean(axis=1)
    chi2 = float(4.0 * M * np.sum((pi - 0.5) ** 2))
    p = float(gammaincc(N / 2.0, chi2 / 2.0))
    return {"test": "2. Block Frequency (M=128)", "p_value": p, "pass": bool(p >= ALPHA), "chi2": chi2}


# ---------------------------------------------------------------------------
# 3. Runs Test
# ---------------------------------------------------------------------------
def test_03_runs(bits: np.ndarray) -> dict:
    n = len(bits)
    pi = float(bits.mean())
    tau = 2.0 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return {"test": "3. Runs", "p_value": 0.0, "pass": False, "note": "Monobit prerequisite failed"}
    v_obs = int(np.count_nonzero(bits[1:] != bits[:-1])) + 1
    num = abs(v_obs - 2.0 * n * pi * (1.0 - pi))
    den = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)
    p = float(erfc(num / den)) if den > 0.0 else 0.0
    return {"test": "3. Runs", "p_value": p, "pass": bool(p >= ALPHA), "v_obs": v_obs}


# ---------------------------------------------------------------------------
# 4. Longest Run of Ones in a Block (M = 10000, K = 6)
# ---------------------------------------------------------------------------
def _longest_run_ones_in_block(block: np.ndarray) -> int:
    if not block.any():
        return 0
    padded = np.empty(len(block) + 2, dtype=np.uint8)
    padded[0] = 0
    padded[1:-1] = block
    padded[-1] = 0
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    return int((ends - starts).max()) if len(starts) else 0


def test_04_longest_run(bits: np.ndarray, M: int = 10000) -> dict:
    # NIST SP 800-22 Rev. 1a constants for M = 10000 (n >= 750,000)
    N = len(bits) // M
    K = 6
    pi = np.array([0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727], dtype=float)
    blocks = bits[: N * M].reshape(N, M)
    max_runs = np.array([_longest_run_ones_in_block(b) for b in blocks], dtype=int)
    v = np.array([
        int(np.sum(max_runs <= 10)),
        int(np.sum(max_runs == 11)),
        int(np.sum(max_runs == 12)),
        int(np.sum(max_runs == 13)),
        int(np.sum(max_runs == 14)),
        int(np.sum(max_runs == 15)),
        int(np.sum(max_runs >= 16)),
    ], dtype=float)
    expected = N * pi
    chi2 = float(np.sum(((v - expected) ** 2) / expected))
    p = float(gammaincc(K / 2.0, chi2 / 2.0))
    return {"test": "4. Longest Run of Ones", "p_value": p, "pass": bool(p >= ALPHA), "chi2": chi2}


# ---------------------------------------------------------------------------
# 5. Binary Matrix Rank Test (32 x 32 over GF(2))
# ---------------------------------------------------------------------------
def _gf2_rank_32(rows: list[int]) -> int:
    rank = 0
    for bit_pos in range(31, -1, -1):
        mask = 1 << bit_pos
        piv = -1
        for r in range(rank, len(rows)):
            if rows[r] & mask:
                piv = r
                break
        if piv == -1:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        piv_row = rows[rank]
        for r in range(len(rows)):
            if r != rank and (rows[r] & mask):
                rows[r] ^= piv_row
        rank += 1
        if rank == 32:
            break
    return rank


def test_05_binary_matrix_rank(bits: np.ndarray, M: int = 32, Q: int = 32) -> dict:
    blk_bits = M * Q
    N = len(bits) // blk_bits
    # Pack each 32-bit row into Python uint32 using numpy
    weights = (1 << np.arange(31, -1, -1, dtype=np.uint32))
    mat_bits = bits[: N * blk_bits].reshape(N, M, Q)
    packed = (mat_bits @ weights).astype(np.uint32)

    f32 = 0
    f31 = 0
    for row_arr in packed.tolist():
        rk = _gf2_rank_32(row_arr)
        if rk == 32:
            f32 += 1
        elif rk == 31:
            f31 += 1
    f_le30 = N - f32 - f31

    # Exact NIST SP 800-22 probabilities for 32x32 over GF(2)
    p32 = 0.2887880950866024
    p31 = 0.5775761901732048
    p30 = 1.0 - p32 - p31
    chi2 = ((f32 - N * p32) ** 2) / (N * p32) + ((f31 - N * p31) ** 2) / (N * p31) + ((f_le30 - N * p30) ** 2) / (N * p30)
    p = float(math.exp(-chi2 / 2.0))
    return {"test": "5. Binary Matrix Rank (32x32)", "p_value": p, "pass": bool(p >= ALPHA), "chi2": float(chi2)}


# ---------------------------------------------------------------------------
# 6. Discrete Fourier Transform (Spectral) Test
# ---------------------------------------------------------------------------
def test_06_dft_spectral(bits: np.ndarray) -> dict:
    n = len(bits)
    if n % 2 == 1:
        bits = bits[:-1]
        n -= 1
    x = 2.0 * bits.astype(np.float64) - 1.0
    m = np.abs(np.fft.rfft(x))[: n // 2]
    T = math.sqrt(math.log(1.0 / 0.05) * n)
    N0 = 0.95 * n / 2.0
    N1 = int(np.sum(m < T))
    d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    p = float(erfc(abs(d) / math.sqrt(2.0)))
    return {"test": "6. DFT (Spectral)", "p_value": p, "pass": bool(p >= ALPHA), "N1": N1, "N0": float(N0)}


# ---------------------------------------------------------------------------
# 7. Non-Overlapping Template Matching Test (m = 9, B = 000000001, N = 8)
# ---------------------------------------------------------------------------
def test_07_non_overlapping_template(bits: np.ndarray, m: int = 9, N: int = 8) -> dict:
    n = len(bits)
    M = n // N
    # Template B = 000000001 (integer value 1 in m-bit window)
    target = 1
    mask = (1 << m) - 1
    blocks = bits[: N * M].reshape(N, M)
    powers = (1 << np.arange(m - 1, -1, -1, dtype=np.uint32))

    W = np.zeros(N, dtype=float)
    for b_idx in range(N):
        blk = blocks[b_idx]
        # Convolution computes rolling m-bit integers cleanly
        # Using bit-rolling in numpy:
        vals = np.convolve(blk.astype(np.uint32), powers[::-1], mode="valid")
        cand = np.flatnonzero(vals == target)
        cnt = 0
        last_end = -1
        for pos in cand:
            if pos >= last_end:
                cnt += 1
                last_end = pos + m
        W[b_idx] = cnt

    mu = (M - m + 1) / (2.0 ** m)
    var = M * ((1.0 / (2.0 ** m)) - ((2.0 * m - 1.0) / (2.0 ** (2 * m))))
    chi2 = float(np.sum(((W - mu) ** 2) / var))
    p = float(gammaincc(N / 2.0, chi2 / 2.0))
    return {"test": "7. Non-Overlapping Template (m=9)", "p_value": p, "pass": bool(p >= ALPHA), "chi2": chi2}


# ---------------------------------------------------------------------------
# 8. Overlapping Template Matching Test (m = 9, B = 111111111, M = 1032, K = 5)
# ---------------------------------------------------------------------------
def test_08_overlapping_template(bits: np.ndarray, m: int = 9, M: int = 1032) -> dict:
    # NIST SP 800-22 Rev. 1a Table / Section 2.8 corrected pi for M=1032, m=9 (lambda=2, eta=1)
    N = len(bits) // M
    K = 5
    pi = np.array([0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865], dtype=float)
    pi /= pi.sum()
    blocks = bits[: N * M].reshape(N, M)
    # Count overlapping runs of 9 consecutive ones in each block
    # Using convolution of sum of m consecutive bits == m
    kernel = np.ones(m, dtype=np.uint16)
    counts = np.zeros(N, dtype=int)
    for i in range(N):
        s = np.convolve(blocks[i].astype(np.uint16), kernel, mode="valid")
        counts[i] = int(np.sum(s == m))
    v = np.array([
        int(np.sum(counts == 0)),
        int(np.sum(counts == 1)),
        int(np.sum(counts == 2)),
        int(np.sum(counts == 3)),
        int(np.sum(counts == 4)),
        int(np.sum(counts >= 5)),
    ], dtype=float)
    expected = N * pi
    chi2 = float(np.sum(((v - expected) ** 2) / expected))
    p = float(gammaincc(K / 2.0, chi2 / 2.0))
    return {"test": "8. Overlapping Template (m=9)", "p_value": p, "pass": bool(p >= ALPHA), "chi2": chi2}


# ---------------------------------------------------------------------------
# 9. Maurer's Universal Statistical Test (L = 7, Q = 1280 for n >= 904,960)
# ---------------------------------------------------------------------------
def test_09_maurers_universal(bits: np.ndarray, L: int = 7, Q: int = 1280) -> dict:
    expected_val = 6.1962507
    variance = 3.125
    n = len(bits)
    K = (n // L) - Q
    total_blocks = Q + K
    weights = (1 << np.arange(L - 1, -1, -1, dtype=np.int32))
    block_vals = (bits[: total_blocks * L].reshape(total_blocks, L) @ weights).tolist()

    T = [0] * (1 << L)
    for i in range(Q):
        T[block_vals[i]] = i + 1

    log2_sum = 0.0
    for i in range(Q, total_blocks):
        v = block_vals[i]
        idx = i + 1
        log2_sum += math.log2(idx - T[v])
        T[v] = idx

    fn = log2_sum / K
    c = 0.7 - 0.8 / L + (4.0 + 32.0 / L) * (K ** (-3.0 / L)) / 15.0
    sigma = c * math.sqrt(variance / K)
    p = float(erfc(abs(fn - expected_val) / (math.sqrt(2.0) * sigma)))
    return {"test": "9. Maurer's Universal (L=7)", "p_value": p, "pass": bool(p >= ALPHA), "fn": float(fn)}


# ---------------------------------------------------------------------------
# 10. Linear Complexity Test (M = 500, K = 6)
# ---------------------------------------------------------------------------
def _berlekamp_massey_bitmask(block_bits: list[int], M: int) -> int:
    # Fast GF(2) Berlekamp-Massey storing binary polynomials in Python arbitrary-precision ints
    c = 1
    b = 1
    L = 0
    m = -1
    for n in range(M):
        # d = s[n] ^ parity((c >> 1) & window) -> compute discrepancy
        # Evaluate sum_{i=0..L} c_i * s[n - i] mod 2
        d = block_bits[n]
        cc = c >> 1
        k = n - 1
        while cc:
            if cc & 1:
                d ^= block_bits[k]
            cc >>= 1
            k -= 1
        if d:
            t = c
            c ^= b << (n - m)
            if 2 * L <= n:
                L = n + 1 - L
                m = n
                b = t
    return L


def test_10_linear_complexity(bits: np.ndarray, M: int = 500) -> dict:
    N = len(bits) // M
    K = 6
    pi = np.array([0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833], dtype=float)
    pi /= pi.sum()
    mu = M / 2.0 + (9.0 + ((-1) ** (M + 1))) / 36.0 - ((M / 3.0 + 2.0 / 9.0) / (2.0 ** M))

    blocks = bits[: N * M].reshape(N, M).tolist()
    sign = (-1) ** M
    T = np.array([sign * (_berlekamp_massey_bitmask(blk, M) - mu) + 2.0 / 9.0 for blk in blocks], dtype=float)
    v = np.array([
        int(np.sum(T <= -2.5)),
        int(np.sum((T > -2.5) & (T <= -1.5))),
        int(np.sum((T > -1.5) & (T <= -0.5))),
        int(np.sum((T > -0.5) & (T <= 0.5))),
        int(np.sum((T > 0.5) & (T <= 1.5))),
        int(np.sum((T > 1.5) & (T <= 2.5))),
        int(np.sum(T > 2.5)),
    ], dtype=float)
    expected = N * pi
    chi2 = float(np.sum(((v - expected) ** 2) / expected))
    p = float(gammaincc(K / 2.0, chi2 / 2.0))
    return {"test": "10. Linear Complexity (M=500)", "p_value": p, "pass": bool(p >= ALPHA), "chi2": chi2}


# ---------------------------------------------------------------------------
# 11. Serial Test (m = 16)
# ---------------------------------------------------------------------------
def _psi2(bits: np.ndarray, m: int) -> float:
    if m <= 0:
        return 0.0
    n = len(bits)
    ext = np.concatenate([bits, bits[: m - 1]])
    # Accumulate m-bit indices cleanly
    idx = np.zeros(n, dtype=np.int32)
    for k in range(m):
        idx = (idx << 1) | ext[k : k + n].astype(np.int32)
    counts = np.bincount(idx, minlength=(1 << m)).astype(np.float64)
    return float(((1 << m) / n) * np.sum(counts * counts) - n)


def test_11_serial(bits: np.ndarray, m: int = 16) -> dict:
    psi_m = _psi2(bits, m)
    psi_m1 = _psi2(bits, m - 1)
    psi_m2 = _psi2(bits, m - 2)
    del1 = psi_m - psi_m1
    del2 = psi_m - 2.0 * psi_m1 + psi_m2
    p1 = float(gammaincc(2 ** (m - 2), max(0.0, del1) / 2.0))
    p2 = float(gammaincc(2 ** (m - 3), max(0.0, del2) / 2.0))
    p_min = min(p1, p2)
    return {
        "test": "11. Serial (m=16)",
        "p_value": p1,
        "p_value_2": p2,
        "min_p_value": p_min,
        "pass": bool(p1 >= ALPHA and p2 >= ALPHA),
    }


# ---------------------------------------------------------------------------
# 12. Approximate Entropy Test (m = 10)
# ---------------------------------------------------------------------------
def _phi(bits: np.ndarray, m: int) -> float:
    n = len(bits)
    ext = np.concatenate([bits, bits[: m - 1]])
    idx = np.zeros(n, dtype=np.int32)
    for k in range(m):
        idx = (idx << 1) | ext[k : k + n].astype(np.int32)
    counts = np.bincount(idx, minlength=(1 << m)).astype(np.float64)
    nz = counts[counts > 0] / n
    return float(np.sum(nz * np.log(nz)))


def test_12_approximate_entropy(bits: np.ndarray, m: int = 10) -> dict:
    n = len(bits)
    phi_m = _phi(bits, m)
    phi_m1 = _phi(bits, m + 1)
    apen = phi_m - phi_m1
    chi2 = float(2.0 * n * (math.log(2.0) - apen))
    p = float(gammaincc(2 ** (m - 1), max(0.0, chi2) / 2.0))
    return {"test": "12. Approximate Entropy (m=10)", "p_value": p, "pass": bool(p >= ALPHA), "apen": float(apen)}


# ---------------------------------------------------------------------------
# 13. Cumulative Sums (Cusum) Test (Forward & Backward)
# ---------------------------------------------------------------------------
def _cusum_p(n: int, z: int) -> float:
    if z <= 0:
        return 1.0
    sqrt_n = math.sqrt(n)
    k_min1 = int(math.floor((-n / z + 1.0) / 4.0))
    k_max1 = int(math.floor((n / z - 1.0) / 4.0))
    s1 = 0.0
    for k in range(k_min1, k_max1 + 1):
        a = (4 * k + 1) * z / sqrt_n
        b = (4 * k - 1) * z / sqrt_n
        s1 += 0.5 * (math.erfc(-a / math.sqrt(2.0)) - math.erfc(-b / math.sqrt(2.0)))

    k_min2 = int(math.floor((-n / z - 3.0) / 4.0))
    k_max2 = int(math.floor((n / z - 1.0) / 4.0))
    s2 = 0.0
    for k in range(k_min2, k_max2 + 1):
        a = (4 * k + 3) * z / sqrt_n
        b = (4 * k + 1) * z / sqrt_n
        s2 += 0.5 * (math.erfc(-a / math.sqrt(2.0)) - math.erfc(-b / math.sqrt(2.0)))

    return float(max(0.0, min(1.0, 1.0 - s1 + s2)))


def test_13_cumulative_sums(bits: np.ndarray) -> dict:
    n = len(bits)
    x = 2 * bits.astype(np.int32) - 1
    z_fwd = int(np.max(np.abs(np.cumsum(x))))
    z_bwd = int(np.max(np.abs(np.cumsum(x[::-1]))))
    p_fwd = _cusum_p(n, z_fwd)
    p_bwd = _cusum_p(n, z_bwd)
    p_min = min(p_fwd, p_bwd)
    return {
        "test": "13. Cumulative Sums",
        "p_value": p_fwd,
        "p_value_backward": p_bwd,
        "min_p_value": p_min,
        "pass": bool(p_fwd >= ALPHA and p_bwd >= ALPHA),
    }


# ---------------------------------------------------------------------------
# 14 & 15. Random Excursions & Random Excursions Variant
# ---------------------------------------------------------------------------
def _cycle_positions(bits: np.ndarray):
    x = 2 * bits.astype(np.int32) - 1
    S = np.empty(len(x) + 2, dtype=np.int32)
    S[0] = 0
    np.cumsum(x, out=S[1:-1])
    S[-1] = 0
    zero_idx = np.flatnonzero(S == 0)
    J = len(zero_idx) - 1
    return S, zero_idx, J


def _excursions_pi(ax: int) -> np.ndarray:
    p0 = 1.0 - 1.0 / (2.0 * ax)
    pi = np.empty(6, dtype=float)
    pi[0] = p0
    base = 1.0 - 1.0 / (2.0 * ax)
    for k in range(1, 5):
        pi[k] = (1.0 / (4.0 * ax * ax)) * (base ** (k - 1))
    pi[5] = (1.0 / (2.0 * ax)) * (base ** 4)
    return pi


def test_14_random_excursions(bits: np.ndarray) -> dict:
    S, zero_idx, J = _cycle_positions(bits)
    if J < 500:
        return {"test": "14. Random Excursions (8 states)", "p_value": None, "pass": False, "J": int(J), "note": "Insufficient cycles J < 500"}
    states = [-4, -3, -2, -1, 1, 2, 3, 4]
    # Fast vectorized cycle-by-state visit counting using bincount per cycle
    z_starts = zero_idx[:-1]
    z_ends = zero_idx[1:]
    # Mask S values in [-4..4]
    # Collect frequencies per state using loop over cycles (~500-2000 cycles, very fast in python/numpy)
    state_v = {st: [0] * 6 for st in states}
    s_list = S.tolist()
    for c in range(J):
        seg = s_list[z_starts[c] + 1 : z_ends[c]]
        if not seg:
            for st in states:
                state_v[st][0] += 1
            continue
        cnt_map = {}
        for val in seg:
            if -4 <= val <= 4:
                cnt_map[val] = cnt_map.get(val, 0) + 1
        for st in states:
            k = cnt_map.get(st, 0)
            state_v[st][k if k < 5 else 5] += 1

    pvals = {}
    for st in states:
        pi = _excursions_pi(abs(st))
        v = np.array(state_v[st], dtype=float)
        exp = J * pi
        chi2 = float(np.sum(((v - exp) ** 2) / exp))
        pvals[str(st)] = float(gammaincc(2.5, chi2 / 2.0))

    min_p = min(pvals.values())
    passed_states = sum(1 for pv in pvals.values() if pv >= ALPHA)
    return {
        "test": "14. Random Excursions (8 states)",
        "p_value": float(min_p),
        "median_p_value": float(np.median(list(pvals.values()))),
        "passed_states": f"{passed_states}/8",
        "pass": bool(passed_states == 8),
        "J": int(J),
        "state_pvalues": pvals,
    }


def test_15_random_excursions_variant(bits: np.ndarray) -> dict:
    S, _, J = _cycle_positions(bits)
    if J < 500:
        return {"test": "15. Random Excursions Var (18 states)", "p_value": None, "pass": False, "J": int(J), "note": "Insufficient cycles J < 500"}
    states = list(range(-9, 0)) + list(range(1, 10))
    S_shift = S[1:-1] + 9
    mask = (S_shift >= 0) & (S_shift <= 18)
    counts = np.bincount(S_shift[mask], minlength=19)
    pvals = {}
    for st in states:
        xi = int(counts[st + 9])
        pvals[str(st)] = float(erfc(abs(xi - J) / math.sqrt(2.0 * J * (4.0 * abs(st) - 2.0))))
    min_p = min(pvals.values())
    passed_states = sum(1 for pv in pvals.values() if pv >= ALPHA)
    return {
        "test": "15. Random Excursions Var (18 states)",
        "p_value": float(min_p),
        "median_p_value": float(np.median(list(pvals.values()))),
        "passed_states": f"{passed_states}/18",
        "pass": bool(passed_states == 18),
        "J": int(J),
        "state_pvalues": pvals,
    }


TEST_FNS = [
    test_01_monobit,
    test_02_block_frequency,
    test_03_runs,
    test_04_longest_run,
    test_05_binary_matrix_rank,
    test_06_dft_spectral,
    test_07_non_overlapping_template,
    test_08_overlapping_template,
    test_09_maurers_universal,
    test_10_linear_complexity,
    test_11_serial,
    test_12_approximate_entropy,
    test_13_cumulative_sums,
    test_14_random_excursions,
    test_15_random_excursions_variant,
]


def run_suite_on_slice(bits: np.ndarray) -> list[dict]:
    return [fn(bits) for fn in TEST_FNS]


def run_multiblock_proportions(full_bits: np.ndarray, max_blocks: int = 9, slice_len: int = N_BITS_SLICE) -> list[dict]:
    n_blocks = min(max_blocks, len(full_bits) // slice_len)
    pass_counts = [0] * len(TEST_FNS)
    eligible_counts = [0] * len(TEST_FNS)
    pvals_per_test = [[] for _ in TEST_FNS]

    for b in range(n_blocks):
        sl = full_bits[b * slice_len : (b + 1) * slice_len]
        res = run_suite_on_slice(sl)
        for ti, r in enumerate(res):
            pv = r.get("p_value")
            if pv is not None:
                eligible_counts[ti] += 1
                pvals_per_test[ti].append(pv)
                if r["pass"]:
                    pass_counts[ti] += 1

    # NIST SP 800-22 minimum pass proportion threshold (Section 4.2.1):
    # p_hat - 3 * sqrt(p_hat*(1-p_hat)/m), p_hat = 1 - alpha = 0.99
    summary = []
    for ti, fn in enumerate(TEST_FNS):
        m = eligible_counts[ti]
        k = pass_counts[ti]
        threshold = (0.99 - 3.0 * math.sqrt(0.99 * 0.01 / m)) if m > 0 else 0.0
        prop = (k / m) if m > 0 else 0.0
        # For small m (<10), NIST allows 1 failure out of ~9? Check exact NIST formula: 0.99 - 3*sqrt(0.0099/9) = 0.8905 (8/9 = 0.8889)
        summary.append({
            "test": run_suite_on_slice(full_bits[:slice_len])[ti]["test"] if False else f"Test {ti+1}",
            "passed_blocks": int(k),
            "eligible_blocks": int(m),
            "proportion": float(prop),
            "min_p_across_blocks": float(min(pvals_per_test[ti])) if pvals_per_test[ti] else None,
            "mean_p_across_blocks": float(np.mean(pvals_per_test[ti])) if pvals_per_test[ti] else None,
        })
    return summary, n_blocks


def main():
    t0 = time.perf_counter()
    os.makedirs("results", exist_ok=True)
    env = get_environment_metadata()

    print("Loading ANU QRNG, HCL IBMQ QRNG, and Mersenne Twister bitstreams...")
    anu_full = load_uint16_as_bits("qrng_combined.txt")
    ibmq_full = load_bitstring_file("fixbit-IBMQ.txt")
    mt_full = load_uint16_as_bits("mersenne_large_baseline.txt")

    sources = {
        "anu_qrng": {"label": "ANU Optical QRNG (qrng_combined.txt)", "bits": anu_full},
        "ibmq_qrng": {"label": "HCL IBMQ Superconducting QRNG (fixbit-IBMQ.txt)", "bits": ibmq_full},
        "mersenne_twister": {"label": "Mersenne Twister MT19937 (mersenne_large_baseline.txt)", "bits": mt_full},
    }

    slice_results = {}
    multiblock_results = {}
    for key, info in sources.items():
        ts = time.perf_counter()
        full_b = info["bits"]
        sl = full_b[:N_BITS_SLICE]
        res = run_suite_on_slice(sl)
        passed = sum(1 for r in res if r["pass"])
        slice_results[key] = {
            "label": info["label"],
            "total_available_bits": int(len(full_b)),
            "slice_bits": int(len(sl)),
            "passed_count": int(passed),
            "total_tests": len(res),
            "tests": res,
        }
        print(f"  {key:<18s} 1M-bit slice: {passed}/{len(res)} passed ({time.perf_counter()-ts:.1f}s)")
        for r in res:
            pv = r.get("p_value")
            pstr = f"{pv:.4f}" if pv is not None else "N/A"
            print(f"    [{'PASS' if r['pass'] else 'FAIL'}] {r['test']:<36s} p={pstr}")

        # Run across all disjoint 1M-bit blocks (up to 9 blocks)
        n_disjoint = len(full_b) // N_BITS_SLICE
        mb_list = []
        max_b = min(9, n_disjoint)
        # Collect block results
        per_test_pass = [0] * len(TEST_FNS)
        per_test_elig = [0] * len(TEST_FNS)
        per_test_pvals = [[] for _ in TEST_FNS]
        for bi in range(max_b):
            b_slice = full_b[bi * N_BITS_SLICE : (bi + 1) * N_BITS_SLICE]
            b_res = res if bi == 0 else run_suite_on_slice(b_slice)
            for ti, rr in enumerate(b_res):
                if rr.get("p_value") is not None:
                    per_test_elig[ti] += 1
                    per_test_pvals[ti].append(rr["p_value"])
                    if rr["pass"]:
                        per_test_pass[ti] += 1
        for ti, rr in enumerate(res):
            m = per_test_elig[ti]
            k = per_test_pass[ti]
            mb_list.append({
                "test": rr["test"],
                "passed_blocks": int(k),
                "eligible_blocks": int(m),
                "proportion": float(k / m) if m else 0.0,
                "mean_p": float(np.mean(per_test_pvals[ti])) if per_test_pvals[ti] else None,
            })
        multiblock_results[key] = {
            "n_blocks": int(max_b),
            "bits_per_block": N_BITS_SLICE,
            "tests": mb_list,
        }

    out = {
        "environment": env,
        "alpha": ALPHA,
        "slice_bits": N_BITS_SLICE,
        "total_wall_seconds": float(time.perf_counter() - t0),
        "single_slice_1m": slice_results,
        "multiblock_1m": multiblock_results,
    }
    out_path = "results/nist_sp800_22_full15.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path} in {out['total_wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
