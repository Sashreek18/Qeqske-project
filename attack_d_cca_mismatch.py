"""
attack_d_cca_mismatch.py  --  Route 1: Chosen-Ciphertext Mismatch Attack on QEQSKE
====================================================================================
Author : Security Analysis Suite for QEQSKE
Target : decrypt_it() in qeqske_core.py

BACKGROUND & THREAT MODEL
--------------------------
In a Chosen-Ciphertext Attack (CCA2), an active adversary acts as a man-in-the-
middle.  She crafts arbitrary (u*, v*) pairs and submits them to Alice's decryption
oracle, then observes the returned decrypted bit (0 or 1).

Standard CRYSTALS-Kyber / ML-KEM defends against this using a Fujisaki–Okamoto
(FO) transform: before returning any plaintext, Alice *re-encrypts* the tentative
result under a deterministic hash of the message and compares it to the received
ciphertext.  Any crafted ciphertext fails this check and returns ⊥.

QEQSKE has NO FO transform and NO hash re-encryption check in decrypt_it().
The oracle accepts ANY (u*, v*) unconditionally and leaks secret-dependent
information through each returned bit.  This is the vulnerability we exploit.

MATHEMATICAL STRUCTURE OF THE ATTACK
--------------------------------------
Alice's decryption for each bit (Algorithm 10) computes:

    diff  =  v  -  u · s          (all mod q)
    val   =  diff[0][0]
    bit   =  0  if  val  is closer to 0 (mod q)
             1  if  val  is closer to q/2 (mod q)

The rounding threshold sits at q/4 and 3q/4.  Concretely:

    bit = 1   iff   q/4 < val < 3q/4

The inner product that feeds val is:

    (u · s)[0][0]  =  sum_{m=0}^{n-1}  u[0][m] · s[m][0]       ... (*)

This sum only involves COLUMN 0 of the secret matrix s (shape n × k).
Columns 1 through k-1 are unreachable via diff[0][0] alone.

ATTACK DESIGN — TWO-PHASE FULL SECRET RECOVERY
-----------------------------------------------
Phase A  —  CCA binary search  (recovers column 0 of s):
    1. Build u* where u*[0][j] = 1 and all other entries are 0.
       Then (u* · s)[0][0] = s[j][0].
    2. Binary-search v*[0][0] = δ from 0 to q.
       The oracle bit at value δ is:
           0  if  (δ − s[j][0])  mod q  ≤ q/4   or  ≥ 3q/4
           1  otherwise
    3. The lowest δ where the oracle flips 0→1 satisfies:
           δ* ≡ s[j][0] + q/4   (mod q)
       so  s[j][0] = (δ* − q/4) mod q,   mapped back to {−1, 0, 1}.
    4. Repeat for j = 0 … n−1.
    Oracle cost per coefficient: ceil(log₂ q) ≈ 13–16 queries.
    Total Phase A cost: n × ceil(log₂ q) ≈ 4 × 15 = 60 queries.

Phase B  —  Brute-force column search  (recovers columns 1 … k−1 of s):
    For each remaining column c of s, search over 3^n = 81 candidates
    cand ∈ {−1,0,1}^n and find the one that satisfies:
        max_i  min((A·cand − t[:,c])_i  mod q,  q − …)  ≤ noise_bound (= 1)
    Since the noise magnitude is 1 and q is O(10^4), this is unambiguous.
    Phase B is free of oracle calls; it uses only the PUBLIC key (A, t).

Combined cost: ~60 adaptive oracle queries + 81 public-key checks → full key.
Compare to exhaustive LWE search: 3^(n×k) = 3^8 = 6561 candidates minimum.
CCA reduces adaptive work by 50% and uses public structure for the rest.

WHY THIS IS CCA2-RELEVANT
--------------------------
CRYSTALS-Kyber/ML-KEM is proven CCA2-secure under the Module-LWE assumption via
the FO transform.  The same Module-LWE hardness assumption applies to QEQSKE's
key-generation parameters.  The broken element is NOT the LWE problem itself —
it is the missing authentication layer around decrypt_it().

This attack is structurally identical to:
  • Bleichenbacher (1998) — PKCS#1 v1.5 RSA padding oracle
  • Manger (2001)        — RSA-OAEP oracle
  • Vaudenay (2002)      — CBC-mode padding oracle

All share the same root cause: an unauthenticated decryption oracle whose
bit-output is an algebraic function of the secret key.

RUN:
    python3 attack_d_cca_mismatch.py

Outputs five phases:
  Phase 0 — Legitimate exchange (sanity check)
  Phase 1 — Single-coefficient binary-search demo (verbose)
  Phase 2 — Full secret recovery (Phase A + B combined)
  Phase 3 — Verification: decrypt a fresh ciphertext with stolen key
  Phase 4 — Statistical evaluation over TRIALS independent keys
  Phase 5 — Root-cause and remediation guide
"""

import math
import time
import statistics
import itertools
from math import ceil

from qrng_handler import QRNGSource, MersenneSource
from qeqske_core import (
    QEQSKERandom,
    key_gen,
    encrypt_it,
    decrypt_it,
    mat_mult_mod,
    mat_sub_mod,
    bits_to_string,
    secret_weight,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N, K        = 4, 2          # Toy parameters (same as rest of suite)
MESSAGE     = "ATTACK!"     # Test plaintext for verification phase
TRIALS      = 300           # Number of independent keys for statistics phase
NOISE_BOUND = 1             # Max |e_i| used in QEQSKE key_gen


# ---------------------------------------------------------------------------
# Pretty banner
# ---------------------------------------------------------------------------

def banner(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CCA Decryption Oracle
# ---------------------------------------------------------------------------

def _raw_oracle(s, u_star, v_star, q):
    """
    Simulate Alice's decrypt_it() on a crafted (u*, v*) pair.

    u_star : list[list[int]]  shape k × n
    v_star : list[list[int]]  shape k × k

    Returns 0 or 1 exactly as Algorithm 10 does, with NO authenticity check.
    This is the security gap: any (u*, v*) is accepted unconditionally.
    """
    diff = mat_sub_mod(v_star, mat_mult_mod(u_star, s, q), q)
    val  = diff[0][0]
    return 0 if min(val, q - val) < abs(val - q // 2) else 1


class OracleCounter:
    """Tracks oracle invocations across all phases."""
    def __init__(self):
        self.calls = 0

    def query(self, s, u_star, v_star, q):
        self.calls += 1
        return _raw_oracle(s, u_star, v_star, q)

    def reset(self):
        self.calls = 0


# ---------------------------------------------------------------------------
# Probe vector builders
# ---------------------------------------------------------------------------

def _unit_u(probe_col: int, target_row: int, n: int, k: int) -> list:
    """
    Build a k × n matrix u where u[probe_col][target_row] = 1, rest = 0.

    With this u, the decryption inner product becomes:
        (u · s)[0][0]  =  u[probe_col=0][target_row] · s[target_row][0]
                       =  s[target_row][0]

    probe_col must be 0 for the result to land at diff[0][0].
    For probing other columns of s via Phase B (brute force), probe_col > 0
    would map to diff[probe_col][0] which the oracle does not expose.
    """
    assert probe_col == 0, "Oracle only reads diff[0][0]; probe_col must be 0."
    u = [[0] * n for _ in range(k)]
    u[0][target_row] = 1
    return u


def _scalar_v(val: int, k: int) -> list:
    """k × k matrix with v[0][0] = val, all else 0."""
    v = [[0] * k for _ in range(k)]
    v[0][0] = int(val)
    return v


# ---------------------------------------------------------------------------
# Phase A — Binary search for one coefficient in column 0 of s
# ---------------------------------------------------------------------------

def recover_col0_coeff(s, row_idx: int, q: int, n: int, k: int,
                        oracle: OracleCounter, verbose: bool = False) -> int:
    """
    Recover s[row_idx][0] via ceil(log₂ q) oracle calls.

    Mechanics:
        u* = unit matrix with u*[0][row_idx] = 1
        oracle reads  val = (δ − s[row_idx][0]) mod q
        oracle returns 1  iff  q/4 < val < 3q/4

    Binary search for the smallest δ where oracle transitions 0 → 1.
    That transition point δ* satisfies:
        (δ* − s[row_idx][0]) mod q  = ⌈q/4⌉
    so  s[row_idx][0] = (δ* − ⌈q/4⌉) mod q.

    CRITICAL NOTE: The oracle condition uses INTEGER arithmetic:
        bit = 0  if  min(val, q-val)  <  abs(val - q//2)
        bit = 1  otherwise  (≥, not >)

    The transition from 0→1 occurs at the smallest val where:
        min(val, q-val)  ≥  abs(val - q//2)
    For val < q//2 this becomes: val ≥ q//2 - val → 2·val ≥ q//2 → val ≥ ⌈q/4⌉.
    So the ACTUAL boundary threshold is ⌈q/4⌉ = ceil(q/4), NOT q//4 (floor).
    Using floor causes a systematic ±1 error on all coefficients — this is the
    precise reason a naïve implementation recovers ~56% of keys instead of 100%.

    Map result back to {−1, 0, 1}: if raw > q/2, subtract q.
    """
    u_star = _unit_u(probe_col=0, target_row=row_idx, n=n, k=k)
    # The actual oracle transition threshold (not floor, but ceiling)
    # Exact threshold: smallest integer val where the oracle returns 1.
    # Oracle returns 1 iff min(val, q-val) >= abs(val - q//2).
    # For val < q//2 this reduces to: val >= q//2 - val => 2*val >= q//2.
    # Smallest such integer: ceil((q//2) / 2) = (q//2 + 1) // 2.
    # This is NOT the same as ceil(q/4) or floor(q/4); it depends on q//2.
    threshold = (q // 2 + 1) // 2

    # Binary search over [0, q) for the lower rounding boundary (δ* = s_i + threshold)
    lo, hi = 0, q
    while lo < hi:
        mid = (lo + hi) // 2
        bit = oracle.query(s, u_star, _scalar_v(mid, k), q)
        if bit == 0:
            lo = mid + 1
        else:
            hi = mid

    delta_star = lo % q
    raw = (delta_star - threshold) % q
    # Map to signed: values > q/2 wrap to negative
    recovered = raw if raw <= q // 2 else raw - q

    if verbose:
        true_val = s[row_idx][0]
        mark = "✓" if recovered == true_val else "✗"
        print(f"    s[{row_idx}][0]: δ*={delta_star:6d}  raw={raw:6d}  "
              f"recovered={recovered:+d}  true={true_val:+d}  {mark}")

    return recovered


# ---------------------------------------------------------------------------
# Phase B — Brute-force remaining columns from public key
# ---------------------------------------------------------------------------

def recover_remaining_cols_bf(A, t, q: int, n: int, k: int,
                               noise_bound: int = NOISE_BOUND,
                               verbose: bool = False) -> list:
    """
    For each column c ∈ {1, …, k−1} of s, exhaustively search {−1,0,1}^n.

    Uses public (A, t): the LWE relation is t[:,c] = A · s[:,c] + e[:,c].
    The correct s[:,c] is the unique candidate with max residual ≤ noise_bound.

    Cost: (k−1) × 3^n candidate checks — no oracle calls.
    At n=4, k=2: 1 × 81 = 81 checks.
    """
    result_cols = {}

    for col in range(1, k):
        t_col = [t[i][col] for i in range(n)]

        for cand in itertools.product([-1, 0, 1], repeat=n):
            As = [sum(A[i][j] * cand[j] for j in range(n)) % q for i in range(n)]
            resid = [min((As[i] - t_col[i]) % q,
                         (t_col[i] - As[i]) % q) for i in range(n)]
            if max(resid) <= noise_bound:
                result_cols[col] = list(cand)
                if verbose:
                    print(f"    s[:,{col}]: {list(cand)} (residual ≤ {noise_bound})")
                break
        else:
            result_cols[col] = None   # Failed (should not happen at toy params)
            if verbose:
                print(f"    s[:,{col}]: NOT FOUND (noise_bound={noise_bound} too tight?)")

    return result_cols


# ---------------------------------------------------------------------------
# Full secret recovery: Phase A + Phase B
# ---------------------------------------------------------------------------

def recover_full_secret(s, A, t, q: int, n: int, k: int,
                         oracle: OracleCounter,
                         verbose: bool = False) -> list:
    """
    Combine Phase A (CCA binary search, column 0) and Phase B (brute force,
    columns 1…k−1) to reconstruct the entire n × k secret matrix.

    Returns s_hat: a list-of-lists matching the shape of the true secret.
    """
    # Phase A: CCA oracle, column 0
    if verbose:
        print("  [Phase A] CCA binary search — column 0 of s:")
    col0 = []
    for row in range(n):
        coeff = recover_col0_coeff(s, row, q, n, k, oracle, verbose=verbose)
        col0.append(coeff)

    # Phase B: brute force from public key, columns 1 … k-1
    if verbose:
        print(f"\n  [Phase B] Public-key brute force — columns 1…{k-1} of s "
              f"(3^{n}={3**n} candidates each):")
    extra_cols = recover_remaining_cols_bf(A, t, q, n, k, verbose=verbose)

    # Assemble full n × k matrix
    s_hat = [[0] * k for _ in range(n)]
    for row in range(n):
        s_hat[row][0] = col0[row]
    for col, coeff_list in extra_cols.items():
        if coeff_list is not None:
            for row in range(n):
                s_hat[row][col] = coeff_list[row]

    return s_hat


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def keys_match(s_true, s_hat, n: int, k: int) -> bool:
    return all(s_true[i][j] == s_hat[i][j] for i in range(n) for j in range(k))


# ---------------------------------------------------------------------------
# PHASE 0 — Legitimate exchange sanity check
# ---------------------------------------------------------------------------

def phase0_sanity(rng) -> bool:
    banner("PHASE 0 — Legitimate Alice ↔ Bob key exchange (sanity check)")
    kd  = key_gen(rng, N, K)
    A, t, q, s = kd["public_key_A"], kd["public_key_t"], kd["q"], kd["private_key"]
    ct  = encrypt_it(rng, MESSAGE, A, t, q, N, K)
    dec = decrypt_it(s, ct["u"], ct["v_queue"], q)
    ok  = dec == MESSAGE
    print(f"  Plaintext   : {MESSAGE!r}")
    print(f"  Decrypted   : {dec!r}")
    print(f"  q           : {q}")
    print(f"  True secret : {s}")
    print(f"  Match       : {'✓ YES' if ok else '✗ NO'}")
    return ok


# ---------------------------------------------------------------------------
# PHASE 1 — Single-coefficient binary-search demo
# ---------------------------------------------------------------------------

def phase1_single_coefficient_demo(rng):
    banner("PHASE 1 — Binary-search probe: recover one secret coefficient")

    kd  = key_gen(rng, N, K)
    s, q = kd["private_key"], kd["q"]
    n, k = kd["n"], kd["k"]
    oracle = OracleCounter()

    target = 2   # probe s[2][0]
    true_val = s[target][0]
    print(f"  Target coefficient  : s[{target}][0] = {true_val:+d}")
    print(f"  q                   : {q}")
    print(f"  Expected queries    : ceil(log₂ {q}) = {math.ceil(math.log2(q))}")
    print()

    t0 = time.perf_counter()
    recovered = recover_col0_coeff(s, target, q, n, k, oracle, verbose=True)
    elapsed = (time.perf_counter() - t0) * 1000

    hit = (recovered == true_val)
    print()
    print(f"  True value      : {true_val:+d}")
    print(f"  Recovered value : {recovered:+d}")
    print(f"  Oracle calls    : {oracle.calls}")
    print(f"  Time            : {elapsed:.3f} ms")
    print(f"  Result          : {'✓ CORRECT' if hit else '✗ WRONG'}")
    return hit, oracle.calls


# ---------------------------------------------------------------------------
# PHASE 2+3 — Full recovery + verification decryption
# ---------------------------------------------------------------------------

def phase2_full_recovery(rng):
    banner("PHASE 2 — Full secret key recovery (Phase A: CCA + Phase B: BF)")

    kd  = key_gen(rng, N, K)
    A, t, q, s = kd["public_key_A"], kd["public_key_t"], kd["q"], kd["private_key"]
    n, k = kd["n"], kd["k"]

    oracle = OracleCounter()
    cca_queries = n * math.ceil(math.log2(q))
    bf_checks   = (k - 1) * (3 ** n)

    print(f"  True secret s     : {s}")
    print(f"  q                 : {q}")
    print(f"  Phase A (CCA)     : {n} coefficients × "
          f"ceil(log₂{q})={math.ceil(math.log2(q))} queries = ~{cca_queries} queries")
    print(f"  Phase B (BF)      : {k-1} column(s) × 3^{n}={3**n} candidates = {bf_checks} checks")
    print()

    t0 = time.perf_counter()
    s_hat = recover_full_secret(s, A, t, q, n, k, oracle, verbose=True)
    elapsed = (time.perf_counter() - t0) * 1000

    exact = keys_match(s, s_hat, n, k)
    print()
    print(f"\n  Recovered secret  : {s_hat}")
    print(f"  True secret       : {s}")
    print(f"  Exact match       : {'✓ YES' if exact else '✗ NO'}")
    print(f"  Oracle calls used : {oracle.calls}")
    print(f"  Time              : {elapsed:.3f} ms")

    banner("PHASE 3 — Verification: decrypt fresh ciphertext with stolen key")
    ct  = encrypt_it(rng, MESSAGE, A, t, q, N, K)
    dec = decrypt_it(s_hat, ct["u"], ct["v_queue"], q)
    verified = (dec == MESSAGE)
    print(f"  Plaintext         : {MESSAGE!r}")
    print(f"  Decrypted         : {dec!r}")
    print(f"  Verified          : {'✓ DECRYPTION SUCCEEDED' if verified else '✗ FAILED'}")

    return exact, verified, oracle.calls, elapsed


# ---------------------------------------------------------------------------
# PHASE 4 — Statistical evaluation over many keys
# ---------------------------------------------------------------------------

def phase4_statistics(rng, trials: int = TRIALS):
    banner(f"PHASE 4 — Statistical evaluation over {trials} independent keys")

    exact_recoveries = 0
    verified_decrypts = 0
    oracle_counts  = []
    timings_ms     = []

    print(f"  {'Trial':>6}  {'Exact':>6}  {'Decrypt':>7}  "
          f"{'Oracle#':>8}  {'ms':>8}  {'Success%':>9}")
    print("  " + "-" * 55)

    for trial in range(1, trials + 1):
        kd  = key_gen(rng, N, K)
        A, t, q, s = (kd["public_key_A"], kd["public_key_t"],
                      kd["q"], kd["private_key"])
        n, k = kd["n"], kd["k"]

        oracle = OracleCounter()
        t0     = time.perf_counter()
        s_hat  = recover_full_secret(s, A, t, q, n, k, oracle, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000

        exact    = keys_match(s, s_hat, n, k)
        ct       = encrypt_it(rng, MESSAGE, A, t, q, N, K)
        dec      = decrypt_it(s_hat, ct["u"], ct["v_queue"], q)
        verified = (dec == MESSAGE)

        if exact:    exact_recoveries  += 1
        if verified: verified_decrypts += 1
        oracle_counts.append(oracle.calls)
        timings_ms.append(elapsed)

        if trial <= 10 or trial % 50 == 0 or trial == trials:
            print(f"  {trial:>6}  {'✓' if exact else '✗':>6}  "
                  f"{'✓' if verified else '✗':>7}  "
                  f"{oracle.calls:>8d}  {elapsed:>8.1f}  "
                  f"{100.0 * exact_recoveries / trial:>8.1f}%")

    # Summary
    banner("PHASE 4 — SUMMARY STATISTICS")
    print(f"  Trials                   : {trials}")
    print(f"  Exact key recoveries     : {exact_recoveries}/{trials}  "
          f"({100.0 * exact_recoveries / trials:.1f}%)")
    print(f"  Verified decryptions     : {verified_decrypts}/{trials}  "
          f"({100.0 * verified_decrypts / trials:.1f}%)")
    print()
    print(f"  Oracle calls  — mean     : {statistics.mean(oracle_counts):.1f}")
    print(f"                — median   : {statistics.median(oracle_counts):.1f}")
    print(f"                — min/max  : {min(oracle_counts)} / {max(oracle_counts)}")
    print(f"                — stdev    : {statistics.stdev(oracle_counts):.2f}")
    print()
    print(f"  Time per key  — mean     : {statistics.mean(timings_ms):.2f} ms")
    print(f"                — median   : {statistics.median(timings_ms):.2f} ms")
    print(f"                — min/max  : {min(timings_ms):.2f} / {max(timings_ms):.2f} ms")
    print()
    # Efficiency ratio vs brute force
    bf_total = 3 ** (N * K)
    avg_oracle = statistics.mean(oracle_counts)
    print(f"  Brute-force space        : 3^(n×k) = 3^{N*K} = {bf_total} candidates")
    print(f"  CCA oracle calls (avg)   : {avg_oracle:.1f}")
    print(f"  Reduction factor         : {bf_total / avg_oracle:.0f}× fewer oracle calls than BF")
    return exact_recoveries, trials


# ---------------------------------------------------------------------------
# PHASE 5 — Root-cause analysis and remediation
# ---------------------------------------------------------------------------

def phase5_root_cause():
    banner("PHASE 5 — Root-cause: Why QEQSKE is CCA-vulnerable & how to fix it")
    print("""
  VULNERABILITY: Missing Fujisaki–Okamoto (FO) transform in decrypt_it()
  -----------------------------------------------------------------------
  Standard ML-KEM (FIPS 203) wraps the PKE layer with an FO transform:

    Encaps(pk):
        m  ← {0,1}^256                    # random message
        (K, r) = G(m ∥ H(pk))             # key derivation
        (u, v)  = PKE.Encrypt(pk, m; r)   # deterministic re-encryption
        K_shared = KDF(K ∥ H(u, v))
        return (K_shared, ciphertext=(u,v))

    Decaps(sk, (u, v)):
        m'      = PKE.Decrypt(sk, (u,v))   # tentative plaintext
        (K',r') = G(m' ∥ H(pk))
        (u*,v*) = PKE.Encrypt(pk, m'; r')  # re-encrypt deterministically
        if CT_EQ((u,v), (u*,v*)):          # constant-time comparison
            return KDF(K' ∥ H(u,v))
        else:
            return KDF(z ∥ H(u,v))         # z is a random rejection key

  The CT_EQ check ensures that any adversarially crafted ciphertext that
  differs by even a single bit will produce a *random-looking* output key,
  giving the adversary no information about s.

  QEQSKE's decrypt_it() (Algorithm 10 in the HCL paper):
  -------------------------------------------------------
    diff = v - u·s   (mod q)
    val  = diff[0][0]
    return 0 if closer to 0, else 1

  There is no re-encryption, no hash check, no rejection.  The function
  accepts ANY (u*, v*) and reveals an algebraic function of s via the bit.

  EXPLOITATION CHAIN (this attack file):
  ----------------------------------------
  Phase A — CCA binary search (column 0 of s):
    1. Craft u* = e_j (unit vector at position j), v* = scalar δ.
    2. Oracle computes val = (δ − s[j][0]) mod q.
    3. Binary-search δ until oracle transitions 0→1 at δ* ≈ s[j][0] + q/4.
    4. s[j][0] = (δ* − q/4) mod q,  mapped to {−1, 0, 1}.
    Cost: n × ceil(log₂ q) ≈ 60 oracle queries.

  Phase B — Public-key brute force (columns 1…k-1 of s):
    For each column c > 0, check 3^n = 81 candidates against t[:,c] ≈ A·s[:,c].
    No oracle calls needed; uses public (A, t) only.
    Cost: (k−1) × 81 = 81 candidate checks.

  WHY COLUMN 0 ONLY FROM CCA:
    diff[0][0] = v[0][0] − (u·s)[0][0]
    (u·s)[0][0] = Σ_m u[0][m]·s[m][0]   ← ONLY column 0 of s contributes
    Columns 1…k-1 of s appear only in diff[0][1], diff[0][2], … which are
    never read by the oracle.  Phase B closes the gap using the public key.

  REQUIRED FIX (minimum viable, non-IND-CCA2 grade):
  ----------------------------------------------------
  At minimum, add ciphertext authentication (hash MAC):
    encrypt_it():  append HMAC-SHA3(key=shared_secret, msg=u∥v) to ciphertext.
    decrypt_it():  verify the MAC before decrypting; reject on failure.

  PRODUCTION FIX (IND-CCA2 security, recommended):
  --------------------------------------------------
  Implement the full Fujisaki–Okamoto transform as in FIPS 203 / ML-KEM:
    a) Derive (K, r) = G(m ∥ H(pk)) inside encapsulation.
    b) Re-encrypt m under pk with randomness r in decapsulation.
    c) Constant-time compare re-encrypted (u*,v*) with received (u,v).
    d) Return a pseudorandom rejection key on mismatch (never ⊥ in timing).

  Without fix (a)+(b)+(c)+(d), ANY decryption oracle is equivalent to a
  secret-key oracle via the binary-search attack demonstrated here.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    banner("CCA MISMATCH ATTACK on QEQSKE  —  Route 1 (Conference-Grade PoC)")
    print("""
  Attack Type : Adaptive Chosen-Ciphertext (CCA2) — Mismatch / Padding Oracle
  Target      : decrypt_it()  in qeqske_core.py
  Root Cause  : No Fujisaki–Okamoto transform; unauthenticated decryption oracle
  Impact      : Full n×k secret key recovery in ~60 adaptive oracle queries
  Comparable  : Bleichenbacher '98 (RSA PKCS#1), Manger '01 (RSA-OAEP),
                Vaudenay '02 (CBC Padding Oracle), Ronen et al. (EM-LWE)
""")

    src = QRNGSource.from_file("qrng_combined.txt", cycle=True)
    rng = QEQSKERandom(src)

    # Phase 0 — confirm honest exchange works
    ok = phase0_sanity(rng)
    assert ok, "Legitimate decryption failed — check parameters or QRNG file"

    # Phase 1 — visualise binary search on one coefficient
    hit1, calls1 = phase1_single_coefficient_demo(rng)

    # Phase 2+3 — full recovery + verification decryption
    exact, verified, calls2, t2 = phase2_full_recovery(rng)

    # Phase 4 — statistics over many keys
    recovered, total = phase4_statistics(rng, trials=TRIALS)

    # Phase 5 — root-cause write-up
    phase5_root_cause()

    # Final verdict table
    banner("FINAL VERDICT")
    max_theoretical = N * math.ceil(math.log2(60000))   # q can be up to ~60000
    print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │  CCA MISMATCH ATTACK ON QEQSKE — RESULTS                          │
  ├────────────────────────────────────────────────────────────────────┤
  │  Phase 1 single-coefficient recovery   : {"✓ PASS" if hit1 else "✗ FAIL":>7}                │
  │  Phase 2 full key recovery             : {"✓ PASS" if exact else "✗ FAIL":>7}                │
  │  Phase 3 verification decrypt          : {"✓ PASS" if verified else "✗ FAIL":>7}                │
  │  Phase 4 success rate ({total} trials) : {100.0*recovered/total:>5.1f}%                 │
  ├────────────────────────────────────────────────────────────────────┤
  │  Phase A oracle calls (this run)       : {calls2:>7}                │
  │  Theoretical max (n×⌈log₂q_max⌉)      : {max_theoretical:>7}                │
  │  Brute-force search space              : {3**(N*K):>7}                │
  ├────────────────────────────────────────────────────────────────────┤
  │  QEQSKE is IND-CCA2-secure?            :      NO                  │
  │  Root cause                            : Missing FO transform      │
  │  Minimum fix                           : HMAC on ciphertext        │
  │  Production fix                        : Full FO (as in ML-KEM)   │
  └────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
