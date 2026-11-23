# ML & Side-Channel Leakage Evaluation of Post-Quantum Key Exchange (QEQSKE)

Empirical security evaluation of HCL Software's **QEQSKE** (*Quantum-Enabled Quantum-Safe Key Exchange*, Ghosh et al.) as a representative post-quantum key-exchange protocol using statistical randomness testing, machine-learning distinguishers, and software-observable side-channel analysis.

- **Course:** CSD493 Final-Year Capstone Project, Shiv Nadar University
- **Faculty Supervisor:** Prof. Rajib Mall
- **Industry Partner:** HCL Software
- **Team:** Sri Surya Srikar, Sashreek Kandagatla, Sane Yateesh Reddy
- **Dataset:** 1,106,944 quantum random numbers from the Australian National University (ANU) QRNG API (`qrng_combined.txt`, uint16 stream)

---

## 1. What This Project Claims, and What It Does Not

The empirical evidence across `results/` separates **public channel / key generation / randomness handling** (no ML leakage detected) from **receiver-side decapsulation (`decrypt_it`)** (strong software side-channel leakage detected):

> **Positive finding (Endpoint Leakage):** Under a bounded software-only side-channel threat model at toy parameters (`n=4, k=2`), receiver-side execution traces from `decrypt_it()` (`dec_timing_us` and `dec_cpu_user_us` in `results/sender_receiver_traces.json`, `results/stress_test_traces.json`, and `results/attack_c_full_stats.json`) predict above/below-median `secret_weight` at **80.42% ± 3.27%** (Random Forest) and **82.18% ± 3.38%** (Gradient Boosting) under 5×10 repeated stratified CV — against a **55.40%** majority-class baseline and **48.36%** / **51.84%** shuffled-label controls (**+32.06** / **+30.34 percentage-point** net gap, permutation `p = 0.0050`). This signal persists unchanged after controlling for variable modulus `q` (`81.70%` RF, `82.68%` GB).

> **Negative findings (No ML Leakage Detected):** Across the same evaluation methodology (seed `42`, 5×10 repeated stratified CV, shuffled-label control, and `q`-confound checks), **no exploitable ML leakage was detected** in:
> 1. Raw ANU QRNG vs. Mersenne Twister (`attack_a_distinguisher.py`, `attack_a_sequence.py` — ~48–50% accuracy),
> 2. `key_gen()` software side-channel traces (`attack_c_combined_ml.py` and the 5-level `attack_c_progressive_difficulty.py` ladder — all targets at or below majority-class baseline),
> 3. **Sender (`encrypt_it`)** execution traces (`Tier 1a`: 53.68% best vs. 55.40% baseline and 53.86% shuffled control), or
> 4. **Public Channel** ciphertext statistical summaries (`Tier 1c`: 52.84% best vs. 55.40% baseline and 51.18% shuffled control).

### What This Project Does NOT Claim
- **Not a proof of cryptographic security.** Negative ML results under one bounded software feature set are not security proofs; see Section 2 for two full breaks at toy parameters.
- **Not evidence about LWE hardness.** At `n=4, k=2`, the secret matrix is recoverable from the public key `(A, t)` in under a millisecond (`break_poc.py`, `attack_e_algebraic_lwe.py`, `attack_e_lll_lattice.py`).
- **Not hardware power or cache-line measurements.** `tracemalloc` and `psutil` are Python software proxies, not oscilloscope power traces or hardware performance counters. Note also that in the 500-trial collectors (`attack_sc_sender_receiver_traces.py`, `attack_c_decrypt_sidechannel.py`), `tracemalloc.start()` wraps `decrypt_it()` during timing collection, which adds allocation-tracking instrumentation overhead alongside the underlying variable-time CPython arithmetic.
- **Fault injection is out of scope / ignored.** Fault-injection attacks (Attack D in early proposals) are intentionally excluded from this evaluation. (`attack_d_cca_mismatch.py` documents an unrelated active chosen-ciphertext / missing Fujisaki–Okamoto check observation in the appendix.)

---

## 2. Self-Audit: Two Cryptographic Breaks Found in Our Implementation

Run `python3 break_poc.py`:

1. **Break 1 — One-Time-Pad Reuse in `encrypt_it` (FIXED):**
   Earlier code sampled `r` and `e1` once per message and reused `res_v_1 = r*t + e1` across all ciphertext blocks `v_queue[i]`, so subtracting two blocks yielded `(bit_i - bit_j) * (q // 2)` and recovered the plaintext (up to global complement) with no key material. Fixed in `qeqske_core.py` by sampling fresh `r, e1, e2` per bit (`legacy_pad_reuse=False` default; `legacy_pad_reuse=True` preserved for reproduction).
2. **Break 2 — Brute-Forceable Secret at Toy Parameters `n=4, k=2` (INHERENT, NOT FIXED):**
   With `s` in `{-1, 0, 1}^{4×2}` and `q ~ 30,000`, each column has only `3^4 = 81` candidates and unit-bounded noise is negligible relative to `q`, enabling exact key recovery from `(A, t)` in <1 ms (`break_poc.py`, `attack_e_algebraic_lwe.py`, `attack_e_lll_lattice.py`).

---

## 3. Empirical Results (All Figures From `results/*.json`)

### 3.1 End-to-End Sender ↔ Receiver Side-Channel Evaluation (`results/stress_test_traces.json`, `results/attack_c_full_stats.json`)

500 Alice ↔ Bob sessions (`FIXED_PLAINTEXT = "HELLO"`, `n=4, k=2`), binary target `secret_weight > 5.0` (`0=277`, `1=223`, **Majority Baseline = 55.40%**), evaluated with `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`.

| Observational Tier | Active Features | Best Model (`stress_test_traces.json`) | Real CV Acc (Mean ± SD) | Shuffled Control | Net Gap (Real − Shuf) | `q`-Residualized Acc | Verdict |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1a: Sender (`encrypt_it`)** | 4 | Logistic Regression | 53.68% ± 2.03% | 53.86% ± 2.28% | −0.18% | 53.74% ± 2.25% | **No Signal (At Baseline)** |
| **Tier 1b: Receiver (`decrypt_it`)** | 4 | Gradient Boosting | **82.18% ± 3.38%** | 51.84% ± 3.41% | **+30.34%** | **82.68% ± 3.51%** | **Confirmed Leakage** |
| **Tier 1b: Receiver (`decrypt_it`)** | 4 | Random Forest | **80.42% ± 3.27%** | 48.36% ± 4.14% | **+32.06%** | **81.70% ± 3.29%** | **Confirmed Leakage** |
| **Tier 1c: Public Channel (`u, v`)** | 24 | SVM (RBF) | 52.84% ± 3.28% | 51.18% ± 3.41% | +1.66% | 50.14% ± 3.86% | **No Signal (At Baseline)** |
| **Tier 2: Sender + Receiver** | 8 | Gradient Boosting | **81.98% ± 2.99%** | 53.36% ± 4.54% | **+28.62%** | **83.02% ± 3.58%** | **Confirmed (Receiver-Driven)** |
| **Tier 3: ALL Combined** | 32 | Gradient Boosting | **81.96% ± 3.69%** | 48.98% ± 3.96% | **+32.98%** | **81.66% ± 4.01%** | **Confirmed (Receiver-Driven)** |

All 5 models on **Tier 1b: Receiver Only** in `results/stress_test_traces.json` and permutation tests in `results/attack_c_full_stats.json` (200 permutations, 3-fold stratified CV):

| Tier 1b Model | `stress_test_traces.json` Real Acc | Shuffled Control | Gap | `q`-Resid Acc | `attack_c_full_stats.json` Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | 82.18% ± 3.38% `[75.0–90.0%]` | 51.84% ± 3.41% | +30.34% | 82.68% ± 3.51% | 81.14% ± 3.22% | **0.0050** |
| **Random Forest** | 80.42% ± 3.27% `[75.0–90.0%]` | 48.36% ± 4.14% | +32.06% | 81.70% ± 3.29% | 80.26% ± 3.69% | **0.0050** |
| **MLP** | 78.04% ± 4.96% `[68.0–87.0%]` | 54.06% ± 2.57% | +23.98% | 76.94% ± 4.60% | 80.26% ± 4.31% | **0.0050** |
| **SVM (RBF)** | 61.88% ± 6.31% `[51.0–79.0%]` | 54.24% ± 2.29% | +7.64% | 59.02% ± 4.14% | 61.88% ± 6.37% | **0.0050** |
| **Logistic Regression** | 55.60% ± 4.04% `[44.0–67.0%]` | 54.98% ± 1.58% | +0.62% | 55.94% ± 4.07% | 55.60% ± 4.09% | 0.1393 |

**Modulus `q` confound check (`results/stress_test_traces.json`):** `q` is uncorrelated with `secret_weight` (Pearson `r = +0.0384, p = 0.3911`; Spearman `ρ = +0.0240, p = 0.5919`), and all four Receiver features have negligible correlation with `q` (`|r| ≤ 0.0705`, all `p ≥ 0.1152`).

**Independent `decrypt_it()` profiling (`attack_c_decrypt_sidechannel.py`, 500 trials in `results/attack_c_full_stats.json`):**
- `cpu_user_us`: Pearson `r = 0.7405` (`p = 5.53e-88`), Spearman `ρ = 0.7786` (`p = 7.46e-103`), Bonferroni `p = 2.21e-87`
- `timing_us`: Pearson `r = 0.6494` (`p = 3.20e-61`), Spearman `ρ = 0.7580` (`p = 1.76e-94`), Bonferroni `p = 1.28e-60`
- `ram_peak_bytes`: Pearson `r = 0.0312` (`p = 0.4863`) — not significant
- `num_allocations`: Pearson `r = -0.0655` (`p = 0.1438`) — not significant
- ML accuracy (`5×10` CV): Logistic Regression **81.86% ± 2.60%**, MLP **81.54% ± 2.87%**, SVM (RBF) **81.46% ± 2.82%**, Random Forest **78.22% ± 3.30%** (all permutation `p = 0.0050`).

---

### 3.2 Attack C — Progressive Difficulty Ladder on `key_gen()` (`results/attack_c_progressive_difficulty.json`)

1,000 trials (`Timing + RAM + CPU` of `key_gen()`), `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`.

**Reconciliation with `results/full_run.log`:** An earlier README reported `s[0][0] exact value` as `35.3% ± 3.0` (`+1.2` margin) from a pre-rerun draft, whereas `results/full_run.log` recorded `31.8% ± 2.6%` (`-2.3 pts` margin) from the Sep 11 run. Re-running `attack_c_progressive_difficulty.py` (which now exports `results/attack_c_progressive_difficulty.json`) produces `-1.0 pts` (`33.11% ± 3.00%` vs `34.10%` baseline, permutation `p = 0.4328`). Across all three runs, **all five ladder targets show no signal above baseline**:

| Target | Difficulty | Classes | Baseline | Best Model | Best Acc (Mean ± SD) | 95% CI | Margin | Perm `p`-value |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| `Secret weight (binary)` | Easy | 2 | 53.90% | Logistic Regression | 53.43% ± 1.13% | ±0.32 pts | **−0.47 pts** | 0.9652 |
| `Secret sum (binary)` | Easy | 2 | 57.70% | Logistic Regression | 57.40% ± 0.54% | ±0.15 pts | **−0.30 pts** | 0.4925 |
| `Secret weight (multi-class)` | Medium | 3 | 46.10% | Logistic Regression | 45.71% ± 0.56% | ±0.16 pts | **−0.39 pts** | 0.2886 |
| `s[0][0] sign (positive vs not)` | Hard | 2 | 65.90% | Logistic Regression | 65.67% ± 0.46% | ±0.13 pts | **−0.23 pts** | 0.7562 |
| `s[0][0] exact value` | Very Hard | 3 | 34.10% | Random Forest | 33.11% ± 3.00% | ±0.85 pts | **−0.99 pts** | 0.4328 |

---

### 3.3 Attack C Controls, Key-Gen Combined ML, RAM/Cache Proxies & ML-KEM Comparison

- **Positive Control (`attack_c_leaky_vs_clean.py`, `results/attack_c_full_stats.json`):**
  - `QEQSKE-clean`: Logistic Regression `56.10% ± 1.33%` (`perm p = 0.7413`), SVM `53.77% ± 4.18%` (`p = 0.9652`), MLP `54.53% ± 3.82%` (`p = 0.9751`), RF `44.70% ± 5.91%` (`p = 0.8905`).
  - `QEQSKE-leaky` (injected secret-dependent branch): **100.00% ± 0.00%** across all four models (`perm p = 0.0050`).
- **Combined `key_gen()` Side-Channel ML (`attack_c_combined_ml.py`, 500 trials, `results/attack_c_full_stats.json`):**
  - Correlations with `secret_weight`: `timing_us r = 0.0803 (p = 0.0727)`, `cpu_user_us r = 0.0866 (p = 0.0529)`, `ram_peak_bytes r = 0.0926 (Bonferroni p = 0.1539)`, `num_allocations r = 0.0933 (Bonferroni p = 0.1478)` — all negligible effect sizes.
  - Classifiers: SVM `55.42% ± 1.49%` (`p = 0.6418`), LR `54.76% ± 1.44%` (`p = 0.8856`), RF `53.66% ± 4.71%` (`p = 0.4179`), MLP `53.58% ± 3.55%` (`p = 0.9154`).
- **System Comparison (`system_comparison.py`, `results/attack_c_full_stats.json`):**
  - `QEQSKE-clean`: LR `56.20% ± 1.12%` (`p = 0.7612`), RF `51.60% ± 5.42%` (`p = 0.5572`).
  - `Standard ML-KEM-512 (fixed params)`: LR `46.40% ± 5.41%` (`p = 0.8259`), RF `49.17% ± 6.72%` (`p = 0.3085`).
  - `QEQSKE-leaky (positive control)`: **100.00% ± 0.00%** (`p = 0.0050`).
- **Fixed RAM & Cache/CPU Software Proxies (`results/attack_c_ram_cache.json`):**
  - Previously `results/full_run.log` failed with `[Errno 2] No such file or directory` looking for pre-rename `Attack C RAM Sidechannel.py` / `Attack Cache Sidechannel.py`, and the renamed scripts still hardcoded `sample_qrng_batch.txt` with pre-Break-1 400-number slice budgets (`QRNGBufferExhausted`). Both scripts now run cleanly on `qrng_combined.txt`:
    - `attack_c_ram_sidechannel.py` (100 trials): `key_gen()` peak `1780.5 ± 99.5` bytes (`r_peak = 0.1093, p = 0.2791`; `r_alloc = 0.0329, p = 0.7454`); `encrypt_it()` peak `12762.2 ± 454.5` bytes (`r_peak = 0.2497, p = 0.0122`; `r_alloc = 0.2376, p = 0.0173`).
    - `attack_c_cache_sidechannel.py` (20 trials): `key_gen()` (500 reps/trial) `19.28 ± 3.33 µs/call` (`r_user = 0.3939, p = 0.0858` — not statistically significant at `n=20`); `encrypt_it()` (100 reps/trial) `139.43 ± 1.30 µs/call` (`r_user = 0.0807, p = 0.7351`).

---

### 3.4 Attack A — QRNG vs. Mersenne Twister Distinguisher (`results/attack_a_level1_full_stats.json`)

| Model | Evaluation Protocol | Sample Size | Accuracy (Mean ± SD) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Random Forest** | 5×5 Repeated Stratified CV | 49,408 windows | 49.93% ± 0.67% | 0.5000 |
| **XGBoost** | 5×5 Repeated Stratified CV | 49,408 windows | 49.89% ± 0.45% | 0.4992 |
| **SVM** | 5×5 Repeated Stratified CV | 15,000 subsample | 49.87% ± 0.90% | 0.4996 |
| **GRU** (`nn_from_scratch.py`) | 10 Repeated Holdouts (75/25) | 2,000 windows | 50.00% ± 2.52% | 0.4890 |
| **LSTM** (`nn_from_scratch.py`) | 10 Repeated Holdouts (75/25) | 2,000 windows | 48.20% ± 3.92% | 0.4831 |
| **Transformer** (`nn_from_scratch.py`) | 10 Repeated Holdouts (75/25) | 2,000 windows | 48.10% ± 1.45% | 0.4991 |

---

## 4. Complete Repository Script Inventory

### Core Protocol, Randomness & Neural Infrastructure
| Script | Purpose |
| :--- | :--- |
| `qeqske_core.py` | QEQSKE Algorithms 1–10 (`is_prime`, `closest_prime`, `key_gen`, `encrypt_it`, `decrypt_it`, hardened tag helpers, and branch-free NumPy `decrypt_it_constant_time`). |
| `qrng_handler.py` | `O(1)` `deque`-backed `QRNGSource` (optional `cycle=True`) and `MersenneSource` baseline adapter. |
| `test_qeqske.py` | End-to-end Alice ↔ Bob key exchange correctness test. |
| `collect_qrng_data.py` | Fetches raw `uint16` quantum random batches from the ANU QRNG API. |
| `merge_qrng.py` | Merges raw QRNG files (`qrng_numbers*.txt`) into `qrng_combined.txt` (1,106,944 values). |
| `mersenne_baseline.py` | Generates deterministic Mersenne Twister comparison batches (`seed=42`). |
| `statistical_tests.py` | NIST-style statistical randomness suite (Frequency, Runs, Longest Run, Chi-Square, Serial Correlation). |
| `nn_from_scratch.py` | Pure-NumPy LSTM, GRU, and Transformer implementations with analytical BPTT. |
| `verify_gradients.py` | Numerical finite-difference gradient check for `nn_from_scratch.py`. |
| `collect_reproducibility_info.py` | Captures software versions, hardware specs, seeds, and hyperparameters to `results/reproducibility.json`. |
| `run_full_suite.py` | Earlier 5-step runner covering correctness, NIST tests, Attack A, Attack B, and Attack C timing groundwork. |
| `Large_Scale_Run_Script.py` | Extended 12-step runner across full QRNG dataset, Attack A/B/C pipelines, system comparison, and metadata collection. |

### Self-Audit, Attack Suites & Sender/Receiver Side-Channel Pipeline
| Script | Purpose |
| :--- | :--- |
| `break_poc.py` | Demonstrates Break 1 (legacy `encrypt_it` pad reuse) and Break 2 (toy `n=4, k=2` brute-force key recovery). |
| `attack_a_distinguisher.py` | Attack A Level 1 — classical ML distinguisher on 15 window summary features (QRNG vs. Mersenne Twister). |
| `attack_a_level1_full_stats.py` | Full 5×5 CV and 10-holdout statistical evaluation saving `results/attack_a_level1_full_stats.json`. |
| `attack_a_levels_2_3.py` | Attack A Levels 2 & 3 — distribution comparisons on QEQSKE-derived fields and public-output `(A, t)` prediction. |
| `attack_a_sequence.py` | Attack A RQ4 — sequence models (LSTM, GRU, Transformer) on QRNG draw order vs. MT and autocorrelated control. |
| `attack_b_entropy_leakage.py` | Attack B — autoencoder reconstruction-error entropy leakage test (QRNG vs. MT vs. uniform random). |
| `attack_c_timing_groundwork.py` | Attack C Part 1 — wall-clock timing correlation measurements on `key_gen()`. |
| `attack_c_ram_sidechannel.py` | Attack C Part 2 — `tracemalloc` peak memory & allocation proxy on `key_gen()` and `encrypt_it()` (`results/attack_c_ram_cache.json`). |
| `attack_c_cache_sidechannel.py` | Attack C Part 3 — `psutil` CPU user/system time cache/power proxy on `key_gen()` and `encrypt_it()` (`results/attack_c_ram_cache.json`). |
| `attack_c_combined_ml.py` | Attack C Part 4 — combined 4D (`timing, cpu, ram, allocs`) ML classification on `key_gen()`. |
| `attack_c_leaky_vs_clean.py` | Positive control comparing clean QEQSKE vs. deliberately timing-leaky QEQSKE (`100%` detection). |
| `attack_c_progressive_difficulty.py` | Attack C — 5-level progressive difficulty ladder on `key_gen()` (`results/attack_c_progressive_difficulty.json`). |
| `attack_c_stats.py` | Shared statistical helper (`compute_correlation_stats`, `compute_ml_stats`, `update_json_results`) writing `results/attack_c_full_stats.json`. |
| `attack_c_decrypt_sidechannel.py` | Attack C-5 — profiles `decrypt_it()` timing/CPU/RAM/allocs across 500 trials (`results/attack_c_full_stats.json`). |
| `attack_c_decrypt_5000.py` | Extended 5,000-trial `decrypt_it()` leakage harness (randomized order, interleaved warm-up, bootstrap CIs, classification + exact-weight regression, clean vs. `tracemalloc` comparison). *(Standalone script; full 5,000-trial run pending execution.)* |
| `attack_c_decrypt_hypothesis.py` | Hypothesis test (`mat_mult_mod` weight 0..8 microbenchmark, `cProfile`/`dis`, and constant-time `decrypt_it_constant_time` verification). *(Requires `results/decrypt_leak_5000.json` from `attack_c_decrypt_5000.py`.)* |
| `attack_sc_sender_receiver_traces.py` | Attack SC-1 — end-to-end Alice ↔ Bob trace collector capturing 34 sender, receiver, and channel features (`results/sender_receiver_traces.json`). |
| `attack_sc_ml_classifiers.py` | Attack SC-2 — 3-tier ML classifier evaluation (Sender, Receiver, Channel, Combined) across 5 model families (`results/attack_c_full_stats.json`). |
| `stress_test_traces.py` | Confound regression (`q`), 50-fold CV spread, and shuffled-label controls on `results/sender_receiver_traces.json` (`results/stress_test_traces.json`). |
| `system_comparison.py` | Compares QEQSKE-clean and QEQSKE-leaky against `kyber-py` `ML_KEM_512` under identical timing ML evaluation (`results/attack_c_full_stats.json`). |
| `attack_d_cca_mismatch.py` | Appendix observation — active chosen-ciphertext binary-search recovery against unhardened `decrypt_it` (no Fujisaki–Okamoto check) and verification of `decrypt_it_hardened`. |
| `attack_e_algebraic_lwe.py` | Route 2 algebraic attack — Gaussian elimination (`A^{-1} mod q`) + `3^n` noise enumeration recovering toy `n=4, k=2` secrets. |
| `attack_e_lll_lattice.py` | Route 2 primal LLL lattice basis reduction attack on toy `n=4, k=2` LWE instances. |
| `attack_f_timing_sidechannel.py` | Route 3 timing analysis of `decrypt_it()` (0-bit vs. 1-bit timing, secret-weight correlation, boundary-probe timing). |
| `attack_g_modulo_bias.py` | Route 2 modulo bias analysis of `call_qrng() % q` and validation of `get_random_unbiased()`. |
| `attack_h_trace_leakage.py` | Eavesdropper model evaluating summary ciphertext/public-key statistics + timing without exposing raw matrix entries. |

---

## 5. Countermeasures & Hardening in `qeqske_core.py`

1. **Per-Bit Ephemeral Masking (Default in `encrypt_it`):** Samples independent `r, e1, e2` per bit, eliminating Break 1 pad reuse.
2. **Rejection-Sampled Modular Arithmetic (`get_random_unbiased`):** Eliminates `uint16 % q` modulo bias (`attack_g_modulo_bias.py`).
3. **Authenticated Re-Encryption Tag (`encrypt_it_hardened` / `decrypt_it_hardened`):** Computes `SHA-256(m || u || v)` and enforces constant-time `hmac.compare_digest` verification after full tentative decryption (`attack_d_cca_mismatch.py`).
4. **Branch-Free Fixed-Dtype Decapsulation (`decrypt_it_constant_time`):** Replaces variable-time Python bigint arithmetic and `if` rounding branches with vectorized NumPy `int64` `matmul` and bitwise mask selection (retaining `decrypt_it` unmodified for baseline comparison).

---

## 6. How to Run Core Suites

```bash
pip install -r requirements.txt

# Correctness and self-audit breaks
python3 test_qeqske.py
python3 break_poc.py

# Sender <-> Receiver trace collection, ML tiers, and stress tests
python3 attack_sc_sender_receiver_traces.py
python3 attack_sc_ml_classifiers.py
python3 stress_test_traces.py

# Attack C progressive difficulty ladder & RAM/Cache proxies
python3 attack_c_progressive_difficulty.py
python3 attack_c_ram_sidechannel.py
python3 attack_c_cache_sidechannel.py
```

---

## 7. Project Status

| Task | Status | Notes / Result File |
| :--- | :--- | :--- |
| QEQSKE core (`qeqske_core.py`, Alg 1–10) | Complete | Pad reuse fixed; `get_random_unbiased`, `decrypt_it_hardened`, `decrypt_it_constant_time` added |
| Self-audit / Break PoC (`break_poc.py`) | Complete | Break 1 fixed; Break 2 inherent to `n=4, k=2` |
| ANU QRNG collection (`qrng_combined.txt`) | Complete | 1,106,944 real quantum random numbers |
| Attack A (Levels 1–3 & Sequence RQ4) | Complete | ~50% distinguisher accuracy (`results/attack_a_level1_full_stats.json`) |
| Attack B (Autoencoder Entropy) | Complete | No structure detected (error ratio ~1.0) |
| Attack C (`key_gen` Time / RAM / CPU / Ladder) | Complete | No leakage above baseline (`results/attack_c_progressive_difficulty.json`, `results/attack_c_ram_cache.json`) |
| Attack C Positive Control & ML-KEM Comparison | Complete | 100.00% on leaky control; clean & ML-KEM-512 at baseline (`results/attack_c_full_stats.json`) |
| **Sender / Channel / Receiver ML & Stress Test** | **Complete** | **Receiver `decrypt_it` leaks at ~80–82% (RF/GB) vs. 55.4% baseline; Sender & Channel at baseline (`results/stress_test_traces.json`)** |
| Attack C-5 5,000-trial & Constant-Time Verification | Scripts Ready / Run Pending | `attack_c_decrypt_5000.py` & `attack_c_decrypt_hypothesis.py` ready to run |
| Attack C Real Hardware Power/Cache Traces | Pending HCL Access | Requires physical oscilloscope / hardware PMU counters |
| Attack D (Fault Injection) | Out of Scope (Ignored) | Excluded per project scope; `attack_d_cca_mismatch.py` retained as appendix CCA note |
| Scale to `n=256` | Not Started | Toy `n=4, k=2` parameters only |
