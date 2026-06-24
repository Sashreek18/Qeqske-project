"""
Pseudo-Random Baseline Generator
===================================
Generates numbers from Python's Mersenne Twister (random module) in the
SAME format/range as your QRNG numbers (uint16: 0-65535).

This gives us a baseline to compare against your real QRNG data —
exactly like Section 6.1 of the HCL paper compares QRNG vs Mersenne Twister.
"""

import random


def generate_mersenne_batch(count: int, seed=None, max_val: int = 65535):
    """Generate `count` pseudo-random integers in [0, max_val], uint16-style."""
    rng = random.Random(seed)
    return [rng.randint(0, max_val) for _ in range(count)]


def save_to_file(numbers, filepath):
    with open(filepath, "w") as f:
        f.write(", ".join(str(n) for n in numbers))


if __name__ == "__main__":
    # Generate same size as a typical QRNG batch for fair comparison
    batch = generate_mersenne_batch(1024, seed=42)
    save_to_file(batch, "mersenne_baseline.txt")
    print(f"Generated {len(batch)} Mersenne Twister numbers -> mersenne_baseline.txt")
    print(f"First 5: {batch[:5]}")
