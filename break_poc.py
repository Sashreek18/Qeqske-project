"""
break_poc.py -- Self-audit of the QEQSKE implementation
========================================================
Two independent breaks we found in our own code, with working demonstrations.
Both are properties of the implementation / parameter choice, not of the
underlying lattice problem.

BREAK 1 -- Pad reuse in encrypt_it (implementation bug, now FIXED)
    The original encrypt_it computed the mask res_v_1 = r*t + e1 once and
    reused it for every message bit. Since M[0][0] = bit * (q//2) is the only
    bit-dependent term, any two ciphertext blocks differ by exactly
    (bit_i - bit_j) * (q//2). Subtracting them recovers the entire plaintext
    up to a single global complement, using no key material at all.
    Fixed by sampling fresh r, e1 per bit (qeqske_core.encrypt_it).

BREAK 2 -- Brute-forceable secret at toy parameters (parameter choice)
    With n=4, k=2, secret coefficients drawn from {-1,0,1} and q ~ 30000, the
    secret has only 3^n = 81 candidates per column and the noise (bounded by 1)
    is negligible against q. The correct candidate is unambiguous. This is NOT
    fixed -- it is inherent to the toy parameters, and it is the reason no
    result at n=4 can be read as evidence about LWE hardness.

Run:  python3 break_poc.py
"""

import itertools
import time

from qrng_handler import QRNGSource
from qeqske_core import (QEQSKERandom, key_gen, encrypt_it, decrypt_it,
                         bits_to_string)

N, K = 4, 2
MESSAGE = "SecretKey42"


def rule(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ---------------------------------------------------------------------------
# BREAK 1
# ---------------------------------------------------------------------------

def recover_plaintext_no_key(v_queue, q):
    """
    Adversary sees only v_queue. No key, no u, no secret.
    Returns both complement candidates; one of them is the plaintext.
    """
    ref = v_queue[0][0][0]
    half = q // 2
    bits = []
    for v in v_queue:
        d = (v[0][0] - ref) % q
        bits.append(0 if min(d, q - d) < abs(d - half) else 1)
    return bits_to_string(bits), bits_to_string([1 - b for b in bits])


def demo_break_1(rng):
    rule("BREAK 1 -- plaintext recovery from ciphertext alone (no key)")
    K_ = key_gen(rng, N, K)
    A, t, q, s = K_["public_key_A"], K_["public_key_t"], K_["q"], K_["private_key"]

    print(f"  plaintext                    : {MESSAGE!r}")
    print(f"  modulus q                    : {q}")

    print("\n  -- vulnerable version (legacy_pad_reuse=True) --")
    ct = encrypt_it(rng, MESSAGE, A, t, q, N, K, legacy_pad_reuse=True)
    c1, c2 = recover_plaintext_no_key(ct["v_queue"], q)
    hit = MESSAGE in (c1, c2)
    print(f"  legitimate decrypt (with key): {decrypt_it(s, ct['u'], ct['v_queue'], q)!r}")
    print(f"  attacker candidate A         : {c1!r}")
    print(f"  attacker candidate B         : {c2!r}")
    print(f"  RECOVERED WITHOUT ANY KEY    : {hit}")

    print("\n  -- fixed version (fresh r, e1 per bit) --")
    ct = encrypt_it(rng, MESSAGE, A, t, q, N, K, legacy_pad_reuse=False)
    c1, c2 = recover_plaintext_no_key(ct["v_queue"], q)
    hit_fixed = MESSAGE in (c1, c2)
    print(f"  legitimate decrypt (with key): {decrypt_it(s, ct['u'], ct['v_queue'], q)!r}")
    print(f"  attacker candidate A         : {c1!r}")
    print(f"  RECOVERED WITHOUT ANY KEY    : {hit_fixed}")

    return hit, hit_fixed


# ---------------------------------------------------------------------------
# BREAK 2
# ---------------------------------------------------------------------------

def brute_force_secret(A, t, q, n, k, noise_bound=1):
    """
    Recover s from public (A, t, q) by exhaustive search over {-1,0,1}^n per
    column, accepting the candidate whose residual is within the noise bound.
    3^n = 81 candidates per column at n=4.
    """
    cols = []
    for c in range(k):
        tcol = [t[i][c] for i in range(n)]
        for cand in itertools.product([-1, 0, 1], repeat=n):
            As = [sum(A[i][j] * cand[j] for j in range(n)) % q for i in range(n)]
            resid = [min((As[i] - tcol[i]) % q, (tcol[i] - As[i]) % q)
                     for i in range(n)]
            if max(resid) <= noise_bound:
                cols.append(list(cand))
                break
        else:
            cols.append(None)
    if any(c is None for c in cols):
        return None
    return [[cols[c][i] for c in range(k)] for i in range(n)]


def demo_break_2(rng, trials=200):
    rule("BREAK 2 -- secret key recovery from the PUBLIC key, by brute force")
    print(f"  search space                 : 3^{N} = {3 ** N} candidates per column")
    print(f"  trials                       : {trials}")

    recovered = 0
    decrypt_ok = 0
    t0 = time.perf_counter()
    for _ in range(trials):
        Kd = key_gen(rng, N, K)
        A, t, q, s = Kd["public_key_A"], Kd["public_key_t"], Kd["q"], Kd["private_key"]
        guess = brute_force_secret(A, t, q, N, K)
        if guess == s:
            recovered += 1
            ct = encrypt_it(rng, "ok", A, t, q, N, K)
            if decrypt_it(guess, ct["u"], ct["v_queue"], q) == "ok":
                decrypt_ok += 1
    elapsed = time.perf_counter() - t0

    print(f"\n  exact key recoveries         : {recovered}/{trials} "
          f"({100.0 * recovered / trials:.1f}%)")
    print(f"  decrypted with stolen key    : {decrypt_ok}/{trials}")
    print(f"  total time                   : {elapsed:.3f} s "
          f"({1000 * elapsed / trials:.2f} ms per key)")
    return recovered, trials, elapsed


# ---------------------------------------------------------------------------

def main():
    src = QRNGSource.from_file("qrng_combined.txt", cycle=True)
    rng = QEQSKERandom(src)

    hit, hit_fixed = demo_break_1(rng)
    rec, tot, elapsed = demo_break_2(rng)

    rule("IMPLICATIONS FOR THE REPORT")
    print(f"""
  Break 1 was a genuine implementation bug and is now fixed. It does not
  affect any reported experimental result, because every attack in the study
  measures key generation and encryption behaviour, not ciphertext structure.

  Break 2 is NOT fixed and cannot be fixed at n=4. It directly contradicts the
  Attack A Level 3 conclusion as originally written. That section reported no
  ML edge over baseline and read it as "consistent with the LWE hardness
  assumption holding even at these toy parameters." The secret is in fact
  recoverable from public data in ~{1000 * elapsed / tot:.1f} ms, {100.0 * rec / tot:.0f}% of the time.

  Correct conclusion: Level 3 shows that the specific ML models tested did not
  extract the secret from public outputs. It says nothing about LWE hardness,
  because at n=4 there is no hardness to speak of. Any claim about hardness
  requires re-running at production parameters (n=256).
""")


if __name__ == "__main__":
    main()
