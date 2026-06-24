"""
QRNG Handler Module
====================
Handles Quantum Random Number Generator output.
Your QRNG generates 1024 numbers at a time in the terminal.

This module:
1. Loads/buffers your QRNG numbers
2. Provides them on-demand to the QEQSKE algorithm (like call_qrng() in the paper)
3. Falls back to refilling the buffer when numbers run out
"""

import random


class QRNGSource:
    """
    Wraps your quantum random number generator output.

    USAGE:
    ------
    Option A — If you have numbers saved to a file (one per line or comma separated):
        qrng = QRNGSource.from_file("my_qrng_numbers.txt")

    Option B — If you paste a Python list directly (e.g. copied from terminal output):
        qrng = QRNGSource(numbers=[123, 456, 789, ...])

    Option C — If you have a function/API that generates a new batch of 1024 numbers:
        qrng = QRNGSource(generator_fn=my_qrng_generate_function)
    """

    def __init__(self, numbers=None, generator_fn=None, batch_size=1024):
        self.buffer = list(numbers) if numbers else []
        self.generator_fn = generator_fn
        self.batch_size = batch_size
        self.total_consumed = 0

    @classmethod
    def from_file(cls, filepath, delimiter=","):
        """Load QRNG numbers from a text/CSV file."""
        with open(filepath, "r") as f:
            content = f.read().strip()
        if delimiter in content:
            numbers = [int(x.strip()) for x in content.split(delimiter) if x.strip()]
        else:
            numbers = [int(x.strip()) for x in content.splitlines() if x.strip()]
        return cls(numbers=numbers)

    def _refill(self):
        """Refill buffer using the generator function, if provided."""
        if self.generator_fn is not None:
            new_batch = self.generator_fn()
            self.buffer.extend(new_batch)
        else:
            raise RuntimeError(
                "QRNG buffer is empty and no generator_fn was provided to refill it. "
                "Either supply more numbers, or pass a generator_fn that produces a new "
                "1024-number batch from your QRNG terminal source."
            )

    def call_qrng(self):
        """
        Equivalent to call_qrng() in the HCL QEQSKE paper (Algorithm 3).
        Returns ONE true random number from your QRNG batch.
        """
        if not self.buffer:
            self._refill()
        self.total_consumed += 1
        return self.buffer.pop(0)

    def remaining(self):
        return len(self.buffer)


if __name__ == "__main__":
    # Quick test with dummy numbers (replace with your real QRNG batch)
    sample_batch = [random.randint(0, 2**16) for _ in range(1024)]
    qrng = QRNGSource(numbers=sample_batch)

    print(f"Loaded {qrng.remaining()} QRNG numbers")
    print("First 5 numbers pulled via call_qrng():")
    for _ in range(5):
        print(" ", qrng.call_qrng())
    print(f"Remaining after pulling 5: {qrng.remaining()}")
