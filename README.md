# QEQSKE Security Analysis

Empirical security evaluation of HCL Software's **QEQSKE** (Quantum Enabled
Quantum Safe Key Exchange) using machine learning and software-observable
side-channel analysis.

**Course:** CSD493 Project-1, B.Tech Final Year, Shiv Nadar University
**Supervisor:** Prof. Rajib Mall
**Industry Partner:** HCL Software
**Dataset:** 197,632 quantum random numbers from the ANU QRNG (two accounts)

---

## What this project claims, and what it does not

The defensible claim is narrow and worth stating precisely:

> Under a bounded software-only threat model, at toy parameters (n=4, k=2), with
> a pipeline validated by a positive control, **no exploitable leakage was
> detected** in QEQSKE's randomness handling or key generation.

What this project does **not** claim:

- It does not claim QEQSKE is secure. Absence of detected leakage under one
  threat model is not a security proof.
- It does not claim anything about LWE hardness. At n=4 the secret is
  brute-forceable from the public key in well under a millisecond (see below),
  so no result at these parameters can speak to lattice hardness.
- It does not claim anything about hardware side channels. Power and cache
  measurement require physical equipment we do not have; `tracemalloc` and
  `psutil` are software proxies, not substitutes.

---

## Self-audit: two breaks found in our own implementation

Run `python3 break_poc.py`.

**Break 1 — pad reuse in `encrypt_it` (implementation bug, FIXED).**
The original `encrypt_it` computed the mask `res_v_1 = r*t + e1` once and reused
it for every message bit. Since `M[0][0] = bit * (q//2)` was the only
bit-dependent term, any two ciphertext blocks differed by exactly
`(bit_i - bit_j) * (q//2)`. Subtracting them recovers the whole plaintext, up to
a single global complement, **with no key material at all**. This is classic
one-time-pad reuse. Fixed by sampling fresh `r` and `e1` per bit;
`legacy_pad_reuse=True` reproduces the original behaviour for demonstration, so
all earlier results remain reproducible.

**Break 2 — brute-forceable secret at toy parameters (NOT fixed, inherent).**
With n=4, k=2, coefficients in {-1,0,1} and q ≈ 30,000, the secret has 3⁴ = 81
candidates per column and the noise is negligible against q. Measured:
**200/200 exact key recoveries in 0.074 s (0.37 ms per key)**, and the stolen key
decrypts correctly. This is a property of the parameter choice, not of the code,
and it cannot be fixed at n=4.

Break 2 corrects a claim in an earlier draft. Attack A Level 3 reported no ML
edge over baseline and read that as consistent with LWE hardness holding. That
inference does not follow: the ML models failed, but the information was
recoverable by other means. Level 3 shows something about the models, not about
lattice hardness.

---

## Files

### Core implementation

| File | Purpose |
| --- | --- |
| `qeqske_core.py` | QEQSKE Algorithms 1–10. Pad reuse fixed; optional unbiased sampling and fixed `q`. |
| `qrng_handler.py` | Serves QRNG numbers (Algorithm 3). `deque.popleft()` instead of `list.pop(0)`. |
| `test_qeqske.py` | End-to-end Alice ↔ Bob correctness check. |

### Attacks and experiments

| File | Purpose |
| --- | --- |
| `break_poc.py` | Both cryptographic breaks, with working demonstrations. |
| `statistical_tests.py` | NIST-style randomness tests on the raw QRNG stream. |
| `attack_a_distinguisher.py` | Attack A Level 1 — raw QRNG vs Mersenne Twister. |
| `attack_a_levels_2_3.py` | Attack A Levels 2 and 3 — KS tests on QEQSKE-derived fields, and prediction from public outputs. |
| `attack_a_sequence.py` | Attack A RQ4 — LSTM/GRU/Transformer on QRNG draw order. |
| `attack_b_entropy_leakage.py` | Attack B — autoencoder reconstruction-error leakage test. |
| `attack_c_timing_groundwork.py` | Attack C — timing measurements. |
| `attack_c_ram_sidechannel.py` | Attack C — memory allocation proxy (`tracemalloc`). |
| `attack_c_cache_sidechannel.py` | Attack C — CPU-time proxy (`psutil`). |
| `attack_c_combined_ml.py` | Attack C — combined Time + RAM + CPU feature set. |
| `attack_c_leaky_vs_clean.py` | Positive control: clean vs deliberately-leaky QEQSKE. |
| `attack_c_progressive_difficulty.py` | Attack C — five prediction targets of increasing difficulty. |

### Infrastructure

| File | Purpose |
| --- | --- |
| `nn_from_scratch.py` | LSTM, GRU, Transformer in pure NumPy with analytic backprop. |
| `verify_gradients.py` | Central-difference check that the hand-written backward passes are correct. |
| `Large_Scale_Run_Script.py` | Runs the original five-step pipeline at full dataset size. |
| `run_full_suite.py` | Earlier combined runner. |
| `collect_qrng_data.py` | Bulk collection from the ANU QRNG API. |
| `mersenne_baseline.py` | Generates the classical baseline stream. |

### Data

`qrng_combined.txt` (197,632 numbers), `qrng_large_dataset.txt` (95,232),
`qrng_batch2.txt` (102,400), `sample_qrng_batch.txt` (1,024),
`mersenne_large_baseline.txt` (197,632 classical baseline).

---

## Setup

```bash
pip install -r requirements.txt
```

All experiments read QRNG data from local files. **No experiment requires network
access**, so nothing fails if the ANU QRNG API is rate-limited or down.

## Running

Quick checks (seconds):

```bash
python3 test_qeqske.py             # end-to-end correctness
python3 break_poc.py               # both cryptographic breaks
python3 verify_gradients.py        # numerical check of hand-written backprop
```

Attack A:

```bash
python3 statistical_tests.py
python3 attack_a_distinguisher.py          # Level 1
python3 attack_a_levels_2_3.py             # Levels 2 and 3
python3 attack_a_sequence.py --windows 400 --window-len 32 --epochs 40
```

Attack B and C:

```bash
python3 attack_b_entropy_leakage.py
python3 attack_c_timing_groundwork.py
python3 attack_c_ram_sidechannel.py
python3 attack_c_cache_sidechannel.py
python3 attack_c_combined_ml.py
python3 attack_c_leaky_vs_clean.py         # positive control
python3 attack_c_progressive_difficulty.py # 1,000-trial ladder (slow)
```

Runtime note: `attack_c_progressive_difficulty.py` collects 1,000 trials and
evaluates five targets under 5-fold × 10-repeat CV. Budget 15–30 minutes. Run it
ahead of any demo rather than live.

---

## Methodology

**Never report raw accuracy alone.** Every result is scored against its
majority-class baseline. 59% accuracy against a 56% baseline is a 3-point margin,
not a 59% success. Only the margin speaks to exploitability.

**Repeated cross-validation, not a single split.** 5-fold × 10-repeat stratified
CV (50 splits), reporting mean ± SD with a 95% CI computed over the fold scores.

**Positive controls.** "No leakage detected" is ambiguous between "there is no
leak" and "the pipeline is blind." `attack_c_leaky_vs_clean.py` runs a variant
with a known, deliberate secret-dependent branch through the identical pipeline.
If the pipeline missed that too, every null result here would be worthless.

---

## Results

### Correctness
End-to-end key exchange succeeds using real QRNG numbers.

### Attack A Level 2 — QEQSKE-derived fields (300 trials/source)

| Field | QRNG mean | MT mean | KS p | Conclusion |
| --- | --- | --- | --- | --- |
| generated prime q | 29,286.95 | 32,515.83 | 0.0125 | see note |
| secret coeff mean | −0.0138 | 0.0054 | 0.9993 | same distribution |
| secret weight | 5.3167 | 5.3233 | 1.0000 | same distribution |
| public key t mean | 0.5061 | 0.5028 | 0.8483 | same distribution |
| public key t SD | 0.2728 | 0.2628 | 0.2488 | same distribution |

**Note on the prime.** Five fields are tested simultaneously, so α must be
Bonferroni-corrected to 0.05/5 = 0.01. At p = 0.0125 the prime is **not**
significant after correction. Testing five fields at an uncorrected α = 0.05
inflates the family-wise false-positive rate to roughly 23%.

### Attack A Level 3 — public outputs only (300 trials)

| Approach | Result |
| --- | --- |
| Best ML model | 54.60% ± 5.12 vs 56.33% baseline → **−1.73 points** |
| Brute-force reference (`break_poc.py`) | **200/200 exact recoveries, 0.37 ms per key** |

The ML models find nothing; the secret is nonetheless fully recoverable from
public data. The null ML result reflects the models, not the hardness of the
instance.

### Attack C — positive control (300 trials/arm, timing feature)

| Arm | Baseline | Best accuracy | Margin | ROC-AUC |
| --- | --- | --- | --- | --- |
| QEQSKE-clean | 56.33% | 59.00% ± 6.55 | +2.67 | 0.584 |
| QEQSKE-leaky | 56.33% | **100.00% ± 0.00** | **+43.67** | **1.000** |

The pipeline detects a real secret-dependent timing branch perfectly while
leaving the clean implementation near baseline. This is what licenses reading the
null result on QEQSKE-clean as informative rather than as pipeline blindness.

Scope: this control uses the timing feature only. Extending it to the full
Time + RAM + CPU feature set used in `attack_c_combined_ml.py` would give feature
parity with the main Attack C pipeline.

### Attack C — progressive difficulty ladder (1,000 trials, Time+RAM+CPU)

| Target | Difficulty | Baseline | Best | Margin |
| --- | --- | --- | --- | --- |
| secret weight (binary) | Easy | 53.9% | 53.3% ± 1.0 | −0.6 |
| secret sum (binary) | Easy | 57.7% | 57.5% ± 0.6 | −0.2 |
| secret weight (multi-class) | Medium | 46.1% | 45.5% ± 0.7 | −0.6 |
| s[0][0] zero/non-zero | Hard | 65.9% | 65.7% ± 0.3 | −0.2 |
| s[0][0] exact value | Very Hard | 34.1% | 35.3% ± 3.0 | +1.2 |

No level shows an exploitable margin. Four of five targets fall below baseline
and the fifth is +1.2 points against a ±3.0 SD. This is a materially stronger
claim than testing secret weight alone, because it holds from a coarse binary
target down to an individual coefficient's exact value.

### Attack A RQ4 — sequence models

`attack_a_sequence.py` trains LSTM, GRU and Transformer models (pure NumPy,
analytic backprop verified by `verify_gradients.py`) on windows of consecutive
QRNG draws, against a Mersenne Twister control and an autocorrelated positive
control. Run it to regenerate; results are written to `results/`.

---

## Limitations

- **Toy parameters.** n=4, k=2 against real Kyber's n=256. The secret is
  brute-forceable at this size. Scaling up is the single highest-value next step.
- **Non-standard modulus.** QEQSKE derives a fresh ~30,000-range prime per key
  from QRNG output; ML-KEM uses a fixed q=3329. This makes q itself a
  QRNG-dependent public observable, which real Kyber does not have.
- **Modulo bias.** `get_random(q) = raw % q` with uint16 raw values is not
  uniform on [0,q). `get_random_unbiased` provides rejection sampling;
  results above use the original path for continuity.
- **Software proxies only.** `tracemalloc` and `psutil` approximate memory and
  cache behaviour. Real power and cache-trace analysis needs physical equipment
  (pending HCL access).
- **Positive control uses timing only**, not the full combined feature set.
- **No shuffled-label controls.** Margins are compared against baseline but not
  against permuted-label runs, which would distinguish real signal from
  evaluation artifacts.
- **Attack D (fault injection) not implemented.**

---

## Status

| Task | Status |
| --- | --- |
| QEQSKE implementation | Complete (pad reuse fixed) |
| Self-audit / break PoC | Complete |
| QRNG collection (197,632) | Complete |
| Statistical tests | Complete |
| Attack A Levels 1–3 | Complete |
| Attack A RQ4 sequence models | Complete |
| Attack B entropy leakage | Complete |
| Attack C time / RAM / CPU | Complete |
| Attack C positive control | Complete (timing feature only) |
| Attack C difficulty ladder | Complete |
| Attack C real hardware traces | Pending HCL equipment access |
| Attack D fault injection | Not started |
| Scale to n=256 | Not started |
| Defence mechanisms | Not started |
