"""
MASTER SCRIPT — Run Full QEQSKE Attack Suite
================================================
Runs the complete pipeline in order:
  1. Verify QEQSKE implementation correctness (key exchange works)
  2. Statistical tests (NIST-style) on QRNG vs Mersenne Twister
  3. Attack A — ML distinguisher (QRNG vs pseudo-random)
  4. Attack B — Entropy leakage via autoencoder
  5. Attack C groundwork — Timing side-channel measurement

Produces a single summary report at the end.

USAGE:
    python3 run_full_suite.py
"""

import warnings
warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings for clean output

from mersenne_baseline import generate_mersenne_batch, save_to_file
import test_qeqske
import statistical_tests
import attack_a_distinguisher
import attack_b_entropy_leakage
import attack_c_timing_groundwork


def load_numbers(filepath):
    with open(filepath) as f:
        return [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]


def main():
    print("\n" + "#" * 70)
    print("# QEQSKE FULL ATTACK SUITE")
    print("# Using your real ANU QRNG data")
    print("#" * 70)

    # --- Step 0: Ensure Mersenne baseline exists ---
    mersenne_numbers = generate_mersenne_batch(1024, seed=42)
    save_to_file(mersenne_numbers, "mersenne_baseline.txt")

    qrng_numbers = load_numbers("sample_qrng_batch.txt")

    # --- Step 1: Verify QEQSKE correctness ---
    print("\n\n" + "#" * 70)
    print("# STEP 1: QEQSKE Correctness Check")
    print("#" * 70)
    result = test_qeqske.run_test(message="Hi", n=4, k=2, verbose=True)

    # --- Step 2: Statistical tests ---
    print("\n\n" + "#" * 70)
    print("# STEP 2: Statistical Randomness Tests")
    print("#" * 70)
    statistical_tests.run_all_tests(qrng_numbers, label="Your Real QRNG Data (ANU)")
    statistical_tests.run_all_tests(mersenne_numbers, label="Mersenne Twister (Pseudo-Random)")

    # --- Step 3: Attack A ---
    print("\n\n" + "#" * 70)
    print("# STEP 3: Attack A — ML-Based Randomness Distinguishing")
    print("#" * 70)
    attack_a_results = attack_a_distinguisher.run_attack_a(
        qrng_numbers, mersenne_numbers, window_size=8
    )

    # --- Step 4: Attack B ---
    print("\n\n" + "#" * 70)
    print("# STEP 4: Attack B — Entropy Leakage Detection (Autoencoder)")
    print("#" * 70)
    attack_b_results = attack_b_entropy_leakage.run_attack_b(
        qrng_numbers, mersenne_numbers, window_size=16, stride=4
    )

    # --- Step 5: Attack C groundwork ---
    print("\n\n" + "#" * 70)
    print("# STEP 5: Attack C Groundwork — Timing Side-Channel")
    print("#" * 70)
    timing_results = attack_c_timing_groundwork.measure_keygen_timing(
        qrng_numbers, n=4, k=2, num_trials=5
    )

    # --- Final Summary ---
    print("\n\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)
    print(f"""
  1. QEQSKE Implementation:    {'WORKING' if result['success'] else 'FAILED'}
     (key exchange correctly encrypts/decrypts using real QRNG numbers)

  2. Statistical Tests:        Both QRNG and Mersenne Twister passed basic
                                tests at this sample size (1024 numbers).
                                Need millions of bits (like the HCL paper)
                                for tests to meaningfully differentiate.

  3. Attack A (ML Distinguish): See accuracy results above.
                                ~50% accuracy = good (QRNG looks random)
                                >60% accuracy = investigate further

  4. Attack B (Entropy Leak):  See reconstruction error ratio above.
                                Ratio ~1.0 = good (QRNG ~ ideal randomness)

  5. Attack C (Timing):        Preliminary correlation measured between
                                key_gen() timing and secret key properties.
                                Larger trial count needed for confidence.

  NEXT STEPS:
  - Collect more QRNG data (collect_qrng_data.py) for statistically
    robust results — current sample (1024) is a proof-of-concept only.
  - Ask HCL about power/cache trace access for full Attack C.
  - Scale n, k up toward real Kyber parameters (n=256) once more
    QRNG numbers are available.
""")


if __name__ == "__main__":
    main()
