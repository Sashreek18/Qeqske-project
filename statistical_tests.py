"""
Statistical Randomness Tests
===============================
Lightweight versions of tests from the NIST Statistical Test Suite,
used in Section 6.1 of the HCL paper (Table 3/4/5) to compare
QRNG vs Mersenne Twister.

These are simplified, dependency-light implementations (no need for
the full NIST STS binary) — enough to demonstrate the comparison
methodology for your project.
"""

import math
from collections import Counter


def numbers_to_bits(numbers, bit_width=16):
    """Convert a list of uint16 numbers into a flat list of bits."""
    bits = []
    for n in numbers:
        bits.extend(int(b) for b in format(n, f"0{bit_width}b"))
    return bits


# ---------------------------------------------------------------------------
# Test 1: Frequency (Monobit) Test
# ---------------------------------------------------------------------------
def frequency_test(bits):
    """
    Checks if the proportion of 0s and 1s is close to 50/50.
    Returns a p-value-like score; closer to 1.0 = more random.
    """
    n = len(bits)
    s = sum(1 if b == 1 else -1 for b in bits)
    s_obs = abs(s) / math.sqrt(n)
    # Using complementary error function approximation
    p_value = math.erfc(s_obs / math.sqrt(2))
    return {"test": "Frequency (Monobit)", "p_value": round(p_value, 4),
            "pass": p_value >= 0.01}


# ---------------------------------------------------------------------------
# Test 2: Runs Test
# ---------------------------------------------------------------------------
def runs_test(bits):
    """
    Checks if the number of "runs" (consecutive same bits) matches
    what's expected for random data.
    """
    n = len(bits)
    pi = sum(bits) / n
    if abs(pi - 0.5) >= (2 / math.sqrt(n)):
        return {"test": "Runs", "p_value": 0.0, "pass": False,
                "note": "Frequency test prerequisite failed"}

    v_obs = 1
    for i in range(1, n):
        if bits[i] != bits[i - 1]:
            v_obs += 1

    numerator = abs(v_obs - 2 * n * pi * (1 - pi))
    denominator = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    p_value = math.erfc(numerator / denominator) if denominator != 0 else 0.0
    return {"test": "Runs", "p_value": round(p_value, 4), "pass": p_value >= 0.01}


# ---------------------------------------------------------------------------
# Test 3: Longest Run of Ones
# ---------------------------------------------------------------------------
def longest_run_test(bits, block_size=8):
    """Checks the longest run of consecutive 1s within blocks."""
    max_run = 0
    current_run = 0
    for b in bits:
        if b == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    # Simplified pass criterion: max run shouldn't be wildly outside
    # expected range for random data of this length
    n = len(bits)
    expected_max = math.log2(n) if n > 0 else 0
    deviation = abs(max_run - expected_max) / max(expected_max, 1)
    passed = deviation < 1.5  # loose threshold for a simplified test
    return {"test": "Longest Run", "longest_run": max_run,
            "expected_approx": round(expected_max, 2), "pass": passed}


# ---------------------------------------------------------------------------
# Test 4: Chi-Square Uniformity Test (on the raw numbers, not bits)
# ---------------------------------------------------------------------------
def chi_square_uniformity(numbers, num_bins=16, max_val=65535):
    """
    Checks if numbers are uniformly distributed across the value range.
    True randomness should spread fairly evenly across bins.
    """
    bin_width = (max_val + 1) / num_bins
    bins = [0] * num_bins
    for n in numbers:
        idx = min(int(n / bin_width), num_bins - 1)
        bins[idx] += 1

    expected = len(numbers) / num_bins
    chi_sq = sum((observed - expected) ** 2 / expected for observed in bins)

    # Rough p-value approximation using chi-square survival function
    # via Wilson-Hilferty approximation (avoids needing scipy)
    df = num_bins - 1
    p_value = _chi_sq_p_value_approx(chi_sq, df)

    return {"test": "Chi-Square Uniformity", "chi_square": round(chi_sq, 2),
            "p_value": round(p_value, 4), "pass": p_value >= 0.01,
            "bin_counts": bins}


def _chi_sq_p_value_approx(chi_sq, df):
    """Wilson-Hilferty approximation for chi-square p-value (no scipy needed)."""
    if df <= 0:
        return 0.0
    term = (chi_sq / df) ** (1 / 3)
    z = (term - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    # Approximate upper-tail standard normal probability
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    return max(0.0, min(1.0, p_value))


# ---------------------------------------------------------------------------
# Test 5: Serial Correlation Test
# ---------------------------------------------------------------------------
def serial_correlation_test(numbers):
    """
    Checks correlation between consecutive numbers.
    Random sequences should have near-zero correlation.
    """
    n = len(numbers)
    if n < 2:
        return {"test": "Serial Correlation", "correlation": None, "pass": False}

    mean = sum(numbers) / n
    x = numbers[:-1]
    y = numbers[1:]
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if denom_x == 0 or denom_y == 0:
        correlation = 0.0
    else:
        correlation = numerator / (denom_x * denom_y)

    passed = abs(correlation) < 0.1  # near-zero correlation = good
    return {"test": "Serial Correlation", "correlation": round(correlation, 4),
            "pass": passed}


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
def run_all_tests(numbers, label="Dataset"):
    bits = numbers_to_bits(numbers)

    results = [
        frequency_test(bits),
        runs_test(bits),
        longest_run_test(bits),
        chi_square_uniformity(numbers),
        serial_correlation_test(numbers),
    ]

    print(f"\n{'=' * 60}")
    print(f"Statistical Test Results: {label}")
    print(f"{'=' * 60}")
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        extra = {k: v for k, v in r.items() if k not in ("test", "pass", "bin_counts")}
        print(f"  [{status}] {r['test']:25s} {extra}")

    pass_count = sum(1 for r in results if r["pass"])
    print(f"\n  Summary: {pass_count}/{len(results)} tests passed")

    return results


if __name__ == "__main__":
    # Quick self-test with your real QRNG data
    with open("sample_qrng_batch.txt") as f:
        qrng_numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    with open("mersenne_baseline.txt") as f:
        mersenne_numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    run_all_tests(qrng_numbers, label="Your Real QRNG Data (ANU)")
    run_all_tests(mersenne_numbers, label="Mersenne Twister (Pseudo-Random)")
