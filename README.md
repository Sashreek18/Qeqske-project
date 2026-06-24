# QEQSKE ML Attack Suite — Project README

## What's In This Folder

| File | What It Does |
|---|---|
| `qrng_handler.py` | Wraps your QRNG numbers, serves them one at a time (Algorithm 3 from HCL paper) |
| `qeqske_core.py` | Core CRYSTALS-KYBER implementation (Algorithms 1-10 from HCL paper) |
| `test_qeqske.py` | Verifies the implementation works (Alice ↔ Bob key exchange) |
| `sample_qrng_batch.txt` | Your real 1024 numbers from ANU QRNG |
| `collect_qrng_data.py` | Calls ANU API repeatedly to build a bigger dataset (run this with your API key) |
| `mersenne_baseline.py` | Generates Mersenne Twister (pseudo-random) numbers for comparison |
| `statistical_tests.py` | NIST-style statistical randomness tests (Frequency, Runs, Chi-Square, etc.) |
| `attack_a_distinguisher.py` | **Attack A**: ML classifier trying to tell QRNG apart from pseudo-random |
| `attack_b_entropy_leakage.py` | **Attack B**: Autoencoder checking if QRNG has learnable structure |
| `attack_c_timing_groundwork.py` | **Attack C groundwork**: Timing measurements during key generation |
| `run_full_suite.py` | Runs everything above in sequence with a final summary |

## How To Run

```bash
# Run everything at once:
python3 run_full_suite.py

# Or run individual pieces:
python3 test_qeqske.py
python3 statistical_tests.py
python3 attack_a_distinguisher.py
python3 attack_b_entropy_leakage.py
python3 attack_c_timing_groundwork.py
```

## To Collect More QRNG Data (Recommended Before Real Experiments)

```bash
python3 collect_qrng_data.py --api-key YOUR_ANU_API_KEY --batches 50 --output qrng_dataset.txt
```

This makes 50 API calls (1024 numbers each = 51,200 total) and saves them to one file.
Then update the scripts to load from `qrng_dataset.txt` instead of `sample_qrng_batch.txt`
for more statistically robust results.

## How To Explain Results To Rajib

### 1. QEQSKE Implementation — "Does it work?"
> "We implemented Algorithms 1-10 from the HCL paper in Python. Alice generates keys,
> Bob encrypts a message, Alice decrypts it — using real QRNG numbers throughout.
> The decrypted message matches the original exactly, confirming correctness."

### 2. Statistical Tests — "Is QRNG actually random?"
> "We ran 5 NIST-style tests (Frequency, Runs, Longest Run, Chi-Square, Serial
> Correlation) on both our QRNG data and Mersenne Twister pseudo-random data.
> At 1024 numbers both pass — we need much larger samples (like the paper's
> 10 million bits) to see meaningful differences."

### 3. Attack A — "Can ML tell QRNG from fake random?"
> "We trained Random Forest and SVM classifiers on statistical features (mean,
> entropy, autocorrelation, etc.) extracted from sliding windows of both QRNG
> and Mersenne Twister data. Accuracy came out near 50% — meaning ML cannot
> reliably distinguish them, which is the SECURITY GOAL (H1 in our proposal:
> ideal implementations should remain indistinguishable from true randomness)."

### 4. Attack B — "Does QRNG leak hidden patterns?"
> "We trained an autoencoder — a neural network that learns to compress and
> reconstruct data — on QRNG sequences. If QRNG had hidden structure, the
> autoencoder would reconstruct it suspiciously well. Instead, QRNG
> reconstruction error was nearly identical to pure theoretical random noise
> (ratio ~0.96), suggesting no obvious entropy leakage at this scale."

### 5. Attack C — "Can timing reveal the secret key?"
> "We measured execution time of key generation across multiple trials and
> correlated it with properties of the generated secret key. We found a
> notable correlation in this small sample — this is a preliminary signal
> worth investigating further with more trials, and is exactly the kind of
> software-level timing side-channel the proposal describes. For full
> Attack C (power/cache traces) we'd need either real hardware measurement
> tools or HCL-provided trace data."

## Honest Limitations (Be Upfront About These)

- **Sample size**: 1024 numbers is a proof-of-concept, not a publication-ready
  experiment. The HCL paper itself used 10 million bits for its NIST tests.
- **Toy parameters**: We used n=4, k=2 for the lattice dimensions (real Kyber
  uses n=256+) to conserve QRNG numbers during development. Scale up once you
  have more data.
- **No real hardware side-channels yet**: Power consumption and cache traces
  need actual hardware profiling — this is the open question for HCL.

## Next Steps

1. Run `collect_qrng_data.py` to get thousands-millions of QRNG numbers
2. Re-run `run_full_suite.py` with the bigger dataset for robust results
3. Scale `n` and `k` up toward real Kyber parameters (n=256)
4. Ask HCL whether they can provide power/cache trace data or hardware access
5. Implement Attack D (fault injection) once A, B, C are solid
