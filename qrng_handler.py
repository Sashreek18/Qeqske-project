"""
QRNG Handler Module (revised)
=============================
Serves QRNG numbers one at a time to the QEQSKE core (Algorithm 3, call_qrng).

CHANGES vs the original version
-------------------------------
1. Uses collections.deque + popleft() instead of list.pop(0).
   list.pop(0) is O(n) in the buffer length. With ~200k buffered numbers this
   made every call_qrng() cost time proportional to the *remaining* buffer,
   which drains monotonically during a run. That injected a systematic
   downward drift into every wall-clock timing measurement in Attack C --
   a confound entirely unrelated to the secret key. deque.popleft() is O(1).

2. Optional cycle=True. Experiments that need >197,632 draws (e.g. the
   1,000-trial Attack C ladder with per-bit fresh randomness) would otherwise
   exhaust the dataset. With cycle=True the dataset is reused from the start.
   This is a documented limitation, not true fresh entropy: any experiment
   using it must say so.
"""

import random
from collections import deque


class QRNGBufferExhausted(RuntimeError):
    """Raised when the buffer empties and no refill strategy is available."""


class QRNGSource:
    """Wraps quantum random number generator output."""

    def __init__(self, numbers=None, generator_fn=None, batch_size=1024,
                 cycle=False):
        self._original = list(numbers) if numbers else []
        self.buffer = deque(self._original)
        self.generator_fn = generator_fn
        self.batch_size = batch_size
        self.cycle = cycle
        self.total_consumed = 0
        self.wraps = 0

    @classmethod
    def from_file(cls, filepath, delimiter=",", cycle=False):
        """Load QRNG numbers from a text/CSV file."""
        with open(filepath, "r") as f:
            content = f.read().strip()
        if delimiter in content:
            numbers = [int(x.strip()) for x in content.split(delimiter) if x.strip()]
        else:
            numbers = [int(x.strip()) for x in content.splitlines() if x.strip()]
        return cls(numbers=numbers, cycle=cycle)

    def _refill(self):
        if self.generator_fn is not None:
            self.buffer.extend(self.generator_fn())
            return
        if self.cycle and self._original:
            self.buffer.extend(self._original)
            self.wraps += 1
            return
        raise QRNGBufferExhausted(
            f"QRNG buffer empty after {self.total_consumed} draws. Supply more "
            "numbers, pass generator_fn, or construct with cycle=True (and "
            "document that the dataset was reused)."
        )

    def call_qrng(self):
        """Algorithm 3 from the HCL paper: return ONE random number."""
        if not self.buffer:
            self._refill()
        self.total_consumed += 1
        return self.buffer.popleft()

    def remaining(self):
        return len(self.buffer)

    def stats(self):
        return {
            "consumed": self.total_consumed,
            "remaining": self.remaining(),
            "dataset_size": len(self._original),
            "wraps": self.wraps,
        }


class MersenneSource(QRNGSource):
    """
    Drop-in classical baseline: same interface, Mersenne Twister instead of QRNG.
    Used as the control arm in Attack A Level 2, where both sources are pushed
    through the identical QEQSKE key-generation pipeline.
    """

    def __init__(self, seed=None, hi=65535):
        super().__init__(numbers=[])
        self._rng = random.Random(seed)
        self._hi = hi

    def call_qrng(self):
        self.total_consumed += 1
        return self._rng.randint(0, self._hi)

    def remaining(self):
        return float("inf")


if __name__ == "__main__":
    q = QRNGSource.from_file("qrng_combined.txt")
    print("Loaded", q.remaining(), "QRNG numbers")
    print("First 5:", [q.call_qrng() for _ in range(5)])
    print("Stats:", q.stats())
