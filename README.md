# QEQSKE ML Attack Suite — Project README

## Project Overview
This project evaluates the security of HCL Software's patented **QEQSKE (Quantum Enabled Quantum Safe Key Exchange)** system using machine learning-based attacks.

We implemented the QEQSKE algorithm from the HCL paper and ran ML-based attacks to check if real quantum random numbers (from ANU QRNG) show any exploitable patterns or leakage.

**Supervisor:** Prof. Rajib Mall
**Industry Partner:** HCL Software
**Dataset:** 1,97,632 real quantum random numbers from ANU QRNG (2 accounts)

---

## What's In This Folder

| File | What It Does |
|---|---|
| `qrng_handler.py` | Wraps QRNG numbers, serves them one at a time (Algorithm 3 from HCL paper) |
| `qeqske_core.py` | Core CRYSTALS-KYBER implementation (Algorithms 1-10 from HCL paper) |
| `test_qeqske.py` | Verifies the implementation works (Alice ↔ Bob key exchange) |
| `sample_qrng_batch.txt` | First 1024 numbers from ANU QRNG (initial test batch) |
| `qrng_large_dataset.txt` | 95,232 real quantum numbers (ANU account 1) |
| `qrng_batch2.txt` | 1,02,400 real quantum numbers (ANU account 2) |
| `qrng_combined.txt` | Combined dataset — 1,97,632 total quantum numbers |
| `collect_qrng_data.py` | Calls ANU QRNG API repeatedly to collect numbers in bulk |
| `mersenne_baseline.py` | Generates Mersenne Twister (pseudo-random) numbers for comparison |
| `mersenne_large_baseline.txt` | 1,97,632 Mersenne Twister numbers (comparison baseline) |
| `statistical_tests.py` | NIST-style statistical randomness tests |
| `attack_a_distinguisher.py` | **Attack A**: ML classifier (Random Forest + SVM) |
| `attack_b_entropy_leakage.py` | **Attack B**: Autoencoder-based entropy leakage detection |
<<<<<<< Updated upstream
| `attack_c_timing_groundwork.py` | **Attack C groundwork**: Timing side-channel measurements |
=======
| `attack_c_timing_groundwork.py` | **Attack C (Time)**: Timing side-channel measurements |
| `attack_c_ram_sidechannel.py` | **Attack C (RAM)**: Memory allocation side-channel (tracemalloc) |
| `attack_c_cache_sidechannel.py` | **Attack C (Cache/Power proxy)**: CPU-time side-channel (psutil) |
>>>>>>> Stashed changes
| `Large_Scale_Run_Script.py` | ✅ Main script — runs all 5 steps with 1,97,632 numbers |

---

## How To Run

```bash
# Run everything at once (recommended):
python3 Large_Scale_Run_Script.py

# Or run individual pieces:
python3 test_qeqske.py
python3 statistical_tests.py
python3 attack_a_distinguisher.py
python3 attack_b_entropy_leakage.py
python3 attack_c_timing_groundwork.py
python3 attack_c_ram_sidechannel.py
python3 attack_c_cache_sidechannel.py
```

---

## Current Results (at 1,97,632 QRNG numbers)

### Step 1 — QEQSKE Correctness ✅
- Key exchange works correctly using real quantum random numbers
- "Hi" → encrypted → decrypted → "Hi" — perfect match

### Step 2 — Statistical Tests ✅
| Test | QRNG | Mersenne Twister |
|---|---|---|
| Frequency (Monobit) | PASS (p=0.0222) | PASS (p=0.3523) |
| Runs | PASS (p=0.589) | PASS (p=0.1375) |
| Longest Run | PASS (run=23) | PASS (run=19) |
| Chi-Square Uniformity | PASS (p=0.2624) | PASS (p=0.7436) |
| Serial Correlation | PASS (corr=0.0) | PASS (corr=-0.0009) |

> QRNG has **literally zero serial correlation** — better than Mersenne Twister

### Step 3 — Attack A: ML Distinguisher ✅
| Model | Accuracy | Verdict |
|---|---|---|
| Random Forest | 50.23% | NOT distinguishable |
| SVM | 50.33% | NOT distinguishable |
| Random Guess (baseline) | 50.00% | — |

> ML cannot distinguish QRNG from pseudo-random — **H1 confirmed**
> Dataset: 24,704 samples per class (49,408 total)

### Step 4 — Attack B: Entropy Leakage ✅
| Source | Reconstruction Error |
|---|---|
| QRNG (held-out) | 0.06274 |
| Mersenne Twister | 0.06268 |
| Pure Random Noise | 0.06256 |
| **QRNG / Pure Random ratio** | **1.003** |

> Ratio ~1.0 means QRNG behaves like ideal randomness — **no entropy leakage detected**

<<<<<<< Updated upstream
### Step 5 — Attack C: Timing Side-Channel ✅
| Metric | Value |
|---|---|
| Mean execution time | 0.0218 ms |
| Timing std deviation | 0.0209 ms |
| Correlation with secret key | 0.2184 |

> Weak/no correlation — **no timing side-channel found at software level**
=======
### Step 5 — Attack C: Side-Channel Analysis (Time + RAM + Cache/Power) ✅
Per Rajib's instruction, Attack C covers three legs, since real hardware power/cache
counters aren't available — RAM and Cache/Power are approximated with software proxies
(`tracemalloc` and `psutil` respectively).

| Leg | Proxy Tool | Trials | Correlation w/ secret key | Verdict |
|---|---|---|---|---|
| Time | `time.perf_counter` | 50 | 0.2184 | Weak/none |
| RAM — key_gen() | `tracemalloc` | 150 | -0.0512 | Weak/none |
| RAM — encrypt_it() | `tracemalloc` | 150 | -0.0082 | Weak/none |
| Cache/Power — key_gen() | `psutil` (CPU time) | 19 | -0.1083 | Weak/none |
| Cache/Power — encrypt_it() | `psutil` (CPU time) | 24 | 0.1435 | Weak/none |

> All three side-channel legs converge to weak/no correlation at scale — same
> stabilization pattern seen in Attack A/B as sample size grows. No exploitable
> software-level side-channel found.
> Note: at low trial counts (~5-20), some legs briefly showed a "meaningful"
> correlation (>0.3) that vanished once trials were scaled up — consistent with
> earlier findings that small-sample correlations here are statistical noise, not
> real signal.
>>>>>>> Stashed changes

---

## Results Progression (Shows Stability)

| | 1,024 numbers | 95,232 numbers | 1,97,632 numbers |
|---|---|---|---|
| Attack A RF accuracy | 54.55% | 49.39% | 50.23% |
| Attack A SVM accuracy | 49.35% | 49.97% | 50.33% |
| Attack B ratio | 0.958 | 0.996 | 1.003 |
| Attack C correlation | 0.8778 | -0.0622 | 0.2184 |
| Trustworthy? | ❌ | ✅ | ✅✅ |

> Results stabilize as sample size increases — confirming statistical robustness

---

## How To Collect More QRNG Data

```bash
# Each ANU account gives ~1 lakh numbers per day
python3 collect_qrng_data.py --api-key YOUR_ANU_API_KEY --batches 100 --output qrng_batch3.txt

# Then combine with existing data:
python3 -c "
files = ['qrng_combined.txt', 'qrng_batch3.txt']
all_numbers = []
for f in files:
    with open(f) as fp:
        nums = [x.strip() for x in fp.read().strip().split(',') if x.strip()]
        all_numbers.extend(nums)
        print(f'{f}: {len(nums)} numbers')
with open('qrng_combined.txt', 'w') as f:
    f.write(', '.join(all_numbers))
print(f'Combined total: {len(all_numbers)} numbers')
"
```

---

## Current Status

| Task | Status |
|---|---|
| QEQSKE implementation | ✅ Complete |
| QRNG data collection (1,97,632 numbers) | ✅ Complete |
| Statistical tests | ✅ Complete |
| Attack A — ML Distinguisher | ✅ Complete |
| Attack B — Entropy Leakage | ✅ Complete |
| Attack C — Software Timing | ✅ Complete |
<<<<<<< Updated upstream
| Attack C — Hardware (power/cache traces) | ⏳ Pending HCL |
=======
| Attack C — Software RAM (tracemalloc) | ✅ Complete |
| Attack C — Software Cache/Power proxy (psutil) | ✅ Complete |
| Attack C — Real hardware (power/cache traces) | ⏳ Pending HCL |
>>>>>>> Stashed changes
| Attack D — Fault Injection | ⏳ Next to build |
| Defense Mechanisms | ⏳ After A/B/C/D |
| Scale to real Kyber n=256 | ⏳ Needs more QRNG data |

---

## Key Finding So Far

> **QEQSKE's QRNG component shows strong resistance to all ML-based attacks at the software level.**
<<<<<<< Updated upstream
> Statistical tests, ML distinguishers, autoencoder leakage detection, and timing analysis
> all fail to find exploitable patterns — confirming that HCL's implementation behaves
> like ideal randomness. The remaining open question is hardware-level side channels
=======
> Statistical tests, ML distinguishers, autoencoder leakage detection, and a full
> 3-leg side-channel analysis (timing, memory, and CPU/cache-proxy) all fail to find
> exploitable patterns — confirming that HCL's implementation behaves like ideal
> randomness. The remaining open question is real hardware-level side channels
>>>>>>> Stashed changes
> (power, cache traces) which require physical measurement equipment.

---

## Limitations

- **Parameters**: We used n=4, k=2 (toy size) — real Kyber uses n=256+. Scale up once more QRNG data is available.
<<<<<<< Updated upstream
- **Hardware side-channels**: Power consumption and cache traces need actual hardware — pending HCL access.
- **Attack D**: Fault injection not yet implemented.
=======
- **Hardware side-channels**: Real power consumption and cache-hit/miss traces need actual hardware — pending HCL access. Our RAM/Cache legs are software proxies (tracemalloc, psutil), not direct hardware measurements.
- **Attack C — Cache/Power trial count**: only reached 19-24 trials (not 150) because each trial's internal repeat-loop consumes QRNG numbers fast; correlation had already dropped near zero by then, but more trials would strengthen confidence.
- **Attack D**: Fault injection not yet implemented.
>>>>>>> Stashed changes
