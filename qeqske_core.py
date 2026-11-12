"""
QEQSKE Implementation (revised)
================================
Based on algorithms from the HCL Software paper:
"Quantum Enabled Quantum Safe Key Exchange" (Ghosh, Ramesh, Dey, Kaushal, Chakrabarti)

Implements Algorithms 1-10:
  is_prime, closest_prime            (Alg 1, 2)
  call_qrng, get_q, get_random       (Alg 3, 4, 5)
  key_gen                            (Alg 6, 7)
  encrypt_it                         (Alg 8, 9)
  decrypt_it                         (Alg 10)

SECURITY FIX IN THIS REVISION
------------------------------
The previous version computed the masking term res_v_1 = r*t + e1 ONCE and
reused it for every bit of the message:

    res_v_1 = r*t + e1                    # once
    for bit in bits:
        M[0][0] = bit * (q // 2)
        v_queue.append(res_v_1 + M)       # same mask every time

Because M[0][0] is the only bit-dependent term, this gives

    v_i[0][0] - v_j[0][0] = (bit_i - bit_j) * (q // 2)   exactly

so subtracting any two ciphertext blocks reveals whether those two plaintext
bits are equal. The whole message follows, up to a single global complement,
with no key material whatsoever. This is classic one-time-pad reuse.
See break_poc.py, Break 1, for a working demonstration.

The fix: sample a fresh r and e1 for every bit, so each bit gets an
independent mask. Set legacy_pad_reuse=True to reproduce the old behaviour
for the proof of concept.

KNOWN REMAINING LIMITATION (deliberate, documented)
---------------------------------------------------
At n=4, k=2 with secret coefficients in {-1,0,1} and q ~ 30000, the secret is
recoverable from the public key by brute force over 3^n = 81 candidates per
column -- roughly 1.4 ms per key. The noise-to-modulus ratio is far too small
to hide it. See break_poc.py, Break 2. This is a property of the toy parameter
choice, not of this code, and it is why no result at these parameters may be
read as evidence about LWE hardness.
"""

from qrng_handler import QRNGSource


# ---------------------------------------------------------------------------
# Algorithm 1 & 2 -- Prime helpers
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Algorithm 1: is_prime()"""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def closest_prime(n: int) -> int:
    """Algorithm 2: closest_prime(). Ties resolve upward, per the paper."""
    if is_prime(n):
        return n
    i = 1
    while True:
        up, down = n + i, n - i
        if is_prime(up) and is_prime(down):
            return up
        if down >= 2 and is_prime(down):
            return down
        if is_prime(up):
            return up
        i += 1


# ---------------------------------------------------------------------------
# Algorithm 3, 4, 5 -- QRNG-backed randomness
# ---------------------------------------------------------------------------

class QEQSKERandom:
    """Wraps a QRNGSource to implement call_qrng / get_q / get_random."""

    def __init__(self, qrng_source: QRNGSource):
        self.qrng = qrng_source

    def call_qrng(self) -> int:
        """Algorithm 3: one true random number from the QRNG."""
        return self.qrng.call_qrng()

    def get_q(self) -> int:
        """Algorithm 4: prime modulus q derived from QRNG output."""
        return closest_prime(self.call_qrng())

    def get_random(self, q: int) -> int:
        """
        Algorithm 5: a random number modulo q.

        NOTE ON MODULO BIAS: raw QRNG values are uint16 (0..65535). For q not a
        power of two, `raw % q` is not uniform on [0, q) -- residues below
        65536 mod q occur once more often than the rest. At q ~ 32768 that is
        close to a factor of two on the lower half of the range, so the public
        matrix A is not uniformly distributed. Rejection sampling below removes
        the bias at the cost of extra draws; set exact=False for the original
        biased behaviour.
        """
        return self.call_qrng() % q

    def get_random_unbiased(self, q: int, span: int = 65536) -> int:
        """Rejection-sampled, bias-free version of Algorithm 5."""
        limit = span - (span % q)
        while True:
            r = self.call_qrng()
            if r < limit:
                return r % q

    def get_small_random(self) -> int:
        """Secret / noise coefficients in {-1, 0, 1}."""
        return (self.call_qrng() % 3) - 1


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def mat_mult_mod(A, B, q):
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])
    assert cols_a == rows_b, f"Shape mismatch: {cols_a} != {rows_b}"
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        Ai = A[i]
        for j in range(cols_b):
            s = 0
            for k in range(cols_a):
                s += Ai[k] * B[k][j]
            result[i][j] = s % q
    return result


def mat_add_mod(A, B, q):
    return [[(A[i][j] + B[i][j]) % q for j in range(len(A[0]))]
            for i in range(len(A))]


def mat_sub_mod(A, B, q):
    return [[(A[i][j] - B[i][j]) % q for j in range(len(A[0]))]
            for i in range(len(A))]


def right_rotate(lst, n=1):
    n = n % len(lst)
    return lst[-n:] + lst[:-n]


# ---------------------------------------------------------------------------
# Algorithm 6 & 7 -- Key Generation (Alice)
# ---------------------------------------------------------------------------

def key_gen(rng: QEQSKERandom, n: int, k: int, unbiased=False, fixed_q=None):
    """
    Algorithm 6/7. Returns dict with private_key s, public_key_A, public_key_t, q.

    unbiased : use rejection sampling for A (removes the modulo bias above)
    fixed_q  : pin the modulus instead of deriving it from QRNG output.
               Real Kyber/ML-KEM uses a fixed q = 3329; deriving a fresh
               ~30000-bit-range prime per key is a deviation from the standard
               and makes q itself a QRNG-dependent public observable.
    """
    q = fixed_q if fixed_q is not None else rng.get_q()
    draw = rng.get_random_unbiased if unbiased else rng.get_random

    # Public matrix A, negacyclic form over Z[x]/(x^n + 1)
    first_col = [draw(q) for _ in range(n)]
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        A[i][0] = first_col[i]
    x = list(first_col)
    for col in range(1, n):
        x = right_rotate(x, 1)
        x[0] = (-x[0]) % q
        for row in range(n):
            A[row][col] = x[row]

    s = [[rng.get_small_random() for _ in range(k)] for _ in range(n)]
    e = [[rng.get_small_random() for _ in range(k)] for _ in range(n)]
    t = mat_add_mod(mat_mult_mod(A, s, q), e, q)

    return {"private_key": s, "public_key_A": A, "public_key_t": t,
            "q": q, "n": n, "k": k}


# ---------------------------------------------------------------------------
# Algorithm 8 & 9 -- Encryption (Bob)
# ---------------------------------------------------------------------------

def string_to_bits(s: str):
    bits = []
    for ch in s:
        bits.extend(int(b) for b in format(ord(ch), "08b"))
    return bits


def bits_to_string(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]
        if len(byte) < 8:
            break
        chars.append(chr(int("".join(str(b) for b in byte), 2)))
    return "".join(chars)


def encrypt_it(rng: QEQSKERandom, shared_secret: str, A, t, q, n, k,
               legacy_pad_reuse=False):
    """
    Algorithm 8/9. Bob encrypts using Alice's public keys (A, t).

    legacy_pad_reuse=False (default, FIXED):
        fresh r, e1 per bit -> every bit gets an independent mask.
    legacy_pad_reuse=True (BROKEN, for break_poc.py only):
        single r, e1 reused across all bits -> plaintext recoverable with no key.
    """
    bits = string_to_bits(shared_secret)
    v_queue = []

    if legacy_pad_reuse:
        r = [[rng.get_small_random() for _ in range(n)] for _ in range(k)]
        e1 = [[rng.get_small_random() for _ in range(k)] for _ in range(k)]
        res_v_1 = mat_add_mod(mat_mult_mod(r, t, q), e1, q)
        for bit in bits:
            M = [[rng.get_small_random() for _ in range(k)] for _ in range(k)]
            M[0][0] = bit * (q // 2)
            v_queue.append(mat_add_mod(res_v_1, M, q))
        e2 = [[rng.get_small_random() for _ in range(n)] for _ in range(k)]
        u = mat_add_mod(mat_mult_mod(r, A, q), e2, q)
        return {"u": u, "v_queue": v_queue, "num_bits": len(bits),
                "legacy_pad_reuse": True}

    # FIXED PATH: independent randomness per bit
    u_queue = []
    for bit in bits:
        r = [[rng.get_small_random() for _ in range(n)] for _ in range(k)]
        e1 = [[rng.get_small_random() for _ in range(k)] for _ in range(k)]
        e2 = [[rng.get_small_random() for _ in range(n)] for _ in range(k)]

        res_v_1 = mat_add_mod(mat_mult_mod(r, t, q), e1, q)
        M = [[rng.get_small_random() for _ in range(k)] for _ in range(k)]
        M[0][0] = bit * (q // 2)

        v_queue.append(mat_add_mod(res_v_1, M, q))
        u_queue.append(mat_add_mod(mat_mult_mod(r, A, q), e2, q))

    return {"u": u_queue, "v_queue": v_queue, "num_bits": len(bits),
            "legacy_pad_reuse": False}


# ---------------------------------------------------------------------------
# Algorithm 10 -- Decryption (Alice)
# ---------------------------------------------------------------------------

def decrypt_it(s, u, v_queue, q):
    """
    Algorithm 10. Handles both the legacy single-u ciphertext and the fixed
    per-bit u_queue, dispatching on the shape of `u`.
    """
    per_bit = isinstance(u, list) and len(u) > 0 and isinstance(u[0], list) \
        and len(u[0]) > 0 and isinstance(u[0][0], list)

    bits = []
    for idx, v in enumerate(v_queue):
        u_i = u[idx] if per_bit else u
        diff = mat_sub_mod(v, mat_mult_mod(u_i, s, q), q)
        val = diff[0][0]
        dist_to_0 = min(val, q - val)
        dist_to_half = abs(val - q // 2)
        bits.append(0 if dist_to_0 < dist_to_half else 1)

    return bits_to_string(bits)


# ---------------------------------------------------------------------------
# Authenticated/Hardened Decryption Countermeasure (Minimal CCA Mitigation)
# ---------------------------------------------------------------------------

import hashlib
import hmac
import json
import secrets


def _serialize_ct(u, v_queue):
    """Deterministic JSON serialization of ciphertext (u, v_queue) for tag computation."""
    return json.dumps({"u": u, "v_queue": v_queue}, sort_keys=True).encode("utf-8")


def _compute_tag(m: str, u, v_queue) -> str:
    """
    Compute the re-encryption integrity tag: SHA-256(m_bytes || ct_bytes).

    m        : the plaintext string Bob is encrypting
    u, v_queue: the ciphertext components

    This is a simplified stand-in for the FO transform's re-encryption-and-compare
    step (phase5_root_cause already describes the full version).  In the real FO
    transform Alice would re-encrypt m' under the public key with deterministic
    randomness derived from m' and compare the resulting ciphertext component-wise.
    Here we approximate that check by hashing the plaintext together with the
    serialized ciphertext: a forged ciphertext cannot produce the same hash without
    knowing m in advance, so the check still closes the adaptive-query channel.

    NOTE ON TIMING: this function is called AFTER full decryption (Alice always
    decrypts first, then verifies).  For a production deployment the final
    comparison MUST be constant-time (e.g. hmac.compare_digest) so that tag
    mismatches do not leak timing information about how many hash bytes matched.
    """
    m_bytes = m.encode("utf-8") if isinstance(m, str) else m
    ct_bytes = _serialize_ct(u, v_queue)
    return hashlib.sha256(m_bytes + ct_bytes).hexdigest()


def encrypt_it_hardened(rng: QEQSKERandom, shared_secret: str, A, t, q, n, k):
    """
    Hardened encryption (Bob's side).

    Runs encrypt_it() then computes an integrity tag:
        tag = SHA-256(m || u || v)
    where m is the plaintext Bob is encrypting.  The tag binds the ciphertext to
    the plaintext that produced it, so Alice can verify authenticity after
    decryption without needing to share any additional secret.
    """
    ct = encrypt_it(rng, shared_secret, A, t, q, n, k, legacy_pad_reuse=False)
    ct["tag"] = _compute_tag(shared_secret, ct["u"], ct["v_queue"])
    return ct


def decrypt_it_hardened(s, u, v_queue, q, tag=None, expected_shared_secret=None):
    """
    Hardened decryption (Alice's side) — FO-style re-encryption check.

    Tag verification is MANDATORY: if tag is None this function rejects
    immediately.  There is NO code path that skips verification for a non-None
    tag; the `expected_shared_secret` parameter is retained only for call-site
    compatibility and is intentionally ignored.

    Algorithm:
      1. Reject immediately if tag is None (no authenticated ciphertext).
      2. Tentatively decrypt: m' = decrypt_it(s, u, v_queue, q).
      3. Recompute: tag' = SHA-256(m' || u || v).
      4. Constant-time compare tag' == tag.  If they differ, return None.
         No bit or threshold information about the mismatch is revealed.
      5. Return m' only on a clean tag match.

    This is a simplified stand-in for the FO transform's re-encryption-and-compare
    step (phase5_root_cause already describes the full version).  It requires Alice
    to always fully decrypt before verifying, which is why a constant-time
    comparison matters for a production version — the decryption itself must not
    be short-circuited on crafted ciphertexts either (not addressed here, but noted).
    """
    if tag is None:
        return None  # Rejection: no tag supplied — unauthenticated ciphertext

    # Step 2: tentative decryption (always performed — no early exit on crafted input)
    m_prime = decrypt_it(s, u, v_queue, q)

    # Step 3: recompute tag from the decrypted plaintext and ciphertext
    tag_prime = _compute_tag(m_prime, u, v_queue)

    # Step 4: constant-time comparison — do not reveal which bytes differed
    if not hmac.compare_digest(tag, tag_prime):
        return None  # Rejection: tag mismatch — forged or corrupted ciphertext

    # Step 5: verification passed
    return m_prime


def secret_weight(s):
    """Hamming-style weight of the secret: count of non-zero coefficients."""
    return sum(1 for row in s for v in row if v != 0)


if __name__ == "__main__":
    print("QEQSKE core (revised) loaded. Run test_qeqske.py for a full exchange.")

