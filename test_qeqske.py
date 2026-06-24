"""
QEQSKE End-to-End Test
========================
Uses YOUR real QRNG numbers to run a full Alice <-> Bob key exchange.

Pipeline:
  Alice: key_gen()           -> generates (private s, public A, t, q)
  Bob:   encrypt_it()        -> encrypts a shared secret using Alice's public key
  Alice: decrypt_it()        -> recovers the shared secret
  Check: does decrypted secret == original secret?
"""

from qrng_handler import QRNGSource
from qeqske_core import QEQSKERandom, key_gen, encrypt_it, decrypt_it


def load_qrng_numbers(filepath="sample_qrng_batch.txt"):
    with open(filepath, "r") as f:
        content = f.read().strip()
    return [int(x.strip()) for x in content.split(",") if x.strip()]


def run_test(message="Hi", n=4, k=2, verbose=True):
    """
    n, k kept SMALL (4, 2) for this first test so we don't burn through
    your 1024 QRNG numbers too fast. The paper uses much bigger n (e.g. 256+)
    for production-grade security; we'll scale up once basic correctness
    is verified.
    """
    numbers = load_qrng_numbers()
    if verbose:
        print(f"Loaded {len(numbers)} real QRNG numbers from your batch.\n")

    qrng = QRNGSource(numbers=numbers)
    rng = QEQSKERandom(qrng)

    # --- Alice: Key Generation ---
    if verbose:
        print("=" * 60)
        print("STEP 1: Alice generates keys (Algorithm 6/7)")
        print("=" * 60)

    keys = key_gen(rng, n=n, k=k)
    if verbose:
        print(f"  Prime modulus q (from QRNG)       = {keys['q']}")
        print(f"  Public key A (shape {n}x{n})        = generated")
        print(f"  Public key t (shape {n}x{k})        = generated")
        print(f"  Private key s (shape {n}x{k}, secret)= generated")
        print(f"  QRNG numbers used so far           = {qrng.total_consumed}")
        print(f"  QRNG numbers remaining              = {qrng.remaining()}\n")

    # --- Bob: Encryption ---
    if verbose:
        print("=" * 60)
        print(f"STEP 2: Bob encrypts shared secret: '{message}' (Algorithm 8/9)")
        print("=" * 60)

    ciphertext = encrypt_it(
        rng, message,
        keys["public_key_A"], keys["public_key_t"], keys["q"],
        keys["n"], keys["k"]
    )
    if verbose:
        print(f"  Message bits encrypted             = {ciphertext['num_bits']} bits")
        print(f"  Ciphertext component u (shape {k}x{n})= generated")
        print(f"  Ciphertext component v_queue        = {len(ciphertext['v_queue'])} matrices")
        print(f"  QRNG numbers used so far           = {qrng.total_consumed}")
        print(f"  QRNG numbers remaining              = {qrng.remaining()}\n")

    # --- Alice: Decryption ---
    if verbose:
        print("=" * 60)
        print("STEP 3: Alice decrypts ciphertext (Algorithm 10)")
        print("=" * 60)

    recovered = decrypt_it(
        keys["private_key"], ciphertext["u"], ciphertext["v_queue"], keys["q"]
    )
    if verbose:
        print(f"  Original message                    = '{message}'")
        print(f"  Recovered message                   = '{recovered}'")
        print(f"  MATCH                               = {recovered == message}\n")

    return {
        "success": recovered == message,
        "original": message,
        "recovered": recovered,
        "qrng_numbers_used": qrng.total_consumed,
        "qrng_numbers_remaining": qrng.remaining(),
        "q": keys["q"],
    }


if __name__ == "__main__":
    result = run_test(message="Hi", n=4, k=2)

    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    if result["success"]:
        print("✅ SUCCESS — QEQSKE key exchange worked correctly!")
        print(f"   Your real QRNG numbers successfully generated keys,")
        print(f"   encrypted, and decrypted the message '{result['original']}'")
    else:
        print("❌ FAILED — decrypted message did not match original.")
        print(f"   Original:  {result['original']}")
        print(f"   Recovered: {result['recovered']}")

    print(f"\n   Total QRNG numbers consumed: {result['qrng_numbers_used']} / 1024")
