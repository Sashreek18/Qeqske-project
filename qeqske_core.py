"""
QEQSKE Implementation
======================
Based on algorithms from the HCL Software paper:
"Quantum Enabled Quantum Safe Key Exchange" (Ghosh, Ramesh, Dey, Kaushal, Chakrabarti)

Implements Algorithms 1-10 from the paper:
- is_prime, closest_prime (Algorithm 1, 2)
- call_qrng, get_q, get_random (Algorithm 3, 4, 5)
- key_gen (Algorithm 6, 7)
- encrypt_it (Algorithm 8, 9)
- decrypt_it (Algorithm 10)

This version is wired to use YOUR real QRNG numbers instead of Python's
pseudo-random generator, making it "Quantum Enabled".
"""

from qrng_handler import QRNGSource


# ---------------------------------------------------------------------------
# Algorithm 1 & 2 — Prime helpers
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Algorithm 1 from the paper: is_prime()"""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def closest_prime(n: int) -> int:
    """Algorithm 2 from the paper: closest_prime()"""
    if is_prime(n):
        return n
    i = 1
    while True:
        if is_prime(n + i) and is_prime(n - i):
            return n + i  # paper returns the larger of equidistant primes
        if is_prime(n - i):
            return n - i
        if is_prime(n + i):
            return n + i
        i += 1


# ---------------------------------------------------------------------------
# Algorithm 3, 4, 5 — QRNG-backed randomness
# ---------------------------------------------------------------------------

class QEQSKERandom:
    """
    Wraps a QRNGSource to implement:
      - call_qrng()  -> Algorithm 3
      - get_q()      -> Algorithm 4
      - get_random(q)-> Algorithm 5
    """

    def __init__(self, qrng_source: QRNGSource):
        self.qrng = qrng_source

    def call_qrng(self) -> int:
        """Algorithm 3: returns ONE true random number from the QRNG."""
        return self.qrng.call_qrng()

    def get_q(self) -> int:
        """Algorithm 4: generate the prime modulus q from QRNG output."""
        raw = self.call_qrng()
        return closest_prime(raw)

    def get_random(self, q: int) -> int:
        """Algorithm 5: a true random number modulo q."""
        return self.call_qrng() % q

    def get_small_random(self) -> int:
        """
        Used for filling secret/noise matrices (paper uses get_random(3) - 1
        to get values in {-1, 0, 1}).
        """
        return (self.call_qrng() % 3) - 1


# ---------------------------------------------------------------------------
# Minimal matrix helpers (pure python, no numpy dependency needed)
# ---------------------------------------------------------------------------

def mat_mult_mod(A, B, q):
    """Matrix multiplication A x B, all entries mod q."""
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])
    assert cols_a == rows_b, f"Shape mismatch: {cols_a} != {rows_b}"
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0
            for k in range(cols_a):
                s += A[i][k] * B[k][j]
            result[i][j] = s % q
    return result


def mat_add_mod(A, B, q):
    rows, cols = len(A), len(A[0])
    return [[(A[i][j] + B[i][j]) % q for j in range(cols)] for i in range(rows)]


def mat_sub_mod(A, B, q):
    rows, cols = len(A), len(A[0])
    return [[(A[i][j] - B[i][j]) % q for j in range(cols)] for i in range(rows)]


def right_rotate(lst, n=1):
    n = n % len(lst)
    return lst[-n:] + lst[:-n]


# ---------------------------------------------------------------------------
# Algorithm 6 & 7 — Key Generation (Alice)
# ---------------------------------------------------------------------------

def key_gen(rng: QEQSKERandom, n: int, k: int):
    """
    Algorithm 6/7 from the paper.
    n = highest degree of polynomial ring (matrix dimension)
    k = width of secret key

    Returns: (private_key_s, public_key_A, public_key_t, q)
    """
    q = rng.get_q()

    # --- Build public matrix A in Ring-LWE form ---
    first_col = [rng.get_random(q) for _ in range(n)]
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        A[i][0] = first_col[i]

    x = list(first_col)
    for col in range(1, n):
        x = right_rotate(x, 1)
        x[0] = (-x[0]) % q
        for row in range(n):
            A[row][col] = x[row]

    # --- Secret key s (n x k), small values in {-1, 0, 1} ---
    s = [[rng.get_small_random() for _ in range(k)] for _ in range(n)]

    # --- Noise matrix e (n x k) ---
    e = [[rng.get_small_random() for _ in range(k)] for _ in range(n)]

    # --- t = A*s + e (mod q) ---
    As = mat_mult_mod(A, s, q)
    t = mat_add_mod(As, e, q)

    return {
        "private_key": s,
        "public_key_A": A,
        "public_key_t": t,
        "q": q,
        "n": n,
        "k": k,
    }


# ---------------------------------------------------------------------------
# Algorithm 8 & 9 — Encryption (Bob)
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


def encrypt_it(rng: QEQSKERandom, shared_secret: str, A, t, q, n, k):
    """
    Algorithm 8/9 from the paper.
    Bob encrypts `shared_secret` using Alice's public keys (A, t).

    Returns ciphertext: (u, v_queue)
    """
    # Bob's private key r (k x n), small values
    r = [[rng.get_small_random() for _ in range(n)] for _ in range(k)]

    # Noise e1 (k x k)
    e1 = [[rng.get_small_random() for _ in range(k)] for _ in range(k)]

    # res_v_1 = r*t + e1 (mod q)
    rt = mat_mult_mod(r, t, q)
    res_v_1 = mat_add_mod(rt, e1, q)

    # Convert shared secret to bits
    bits = string_to_bits(shared_secret)

    v_queue = []
    for bit in bits:
        # Build message matrix M (k x k), first entry encodes the bit
        M = [[rng.get_small_random() for _ in range(k)] for _ in range(k)]
        M[0][0] = bit * (q // 2)

        res = mat_add_mod(res_v_1, M, q)
        v_queue.append(res)

    # Noise e2 (k x n)
    e2 = [[rng.get_small_random() for _ in range(n)] for _ in range(k)]

    # u = r*A + e2 (mod q)
    rA = mat_mult_mod(r, A, q)
    u = mat_add_mod(rA, e2, q)

    return {"u": u, "v_queue": v_queue, "num_bits": len(bits)}


# ---------------------------------------------------------------------------
# Algorithm 10 — Decryption (Alice)
# ---------------------------------------------------------------------------

def decrypt_it(s, u, v_queue, q):
    """
    Algorithm 10 from the paper.
    Alice decrypts the ciphertext using her private key s.
    """
    us = mat_mult_mod(u, s, q)  # shape: (k x k) since u is (k x n), s is (n x k)

    bits = []
    for v in v_queue:
        diff = mat_sub_mod(v, us, q)
        val = diff[0][0]
        # Classify: close to 0 -> bit 0, close to q/2 -> bit 1
        dist_to_0 = min(val, q - val)
        dist_to_half = abs(val - q // 2)
        bits.append(0 if dist_to_0 < dist_to_half else 1)

    return bits_to_string(bits)


if __name__ == "__main__":
    print("QEQSKE core module loaded successfully.")
    print("Run test_qeqske.py to perform a full key exchange test.")
