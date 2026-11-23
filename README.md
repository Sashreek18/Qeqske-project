# ML & Side-Channel Leakage Evaluation of Post-Quantum Key Exchange (QEQSKE)

Empirical security evaluation of HCL Software's **QEQSKE** (*Quantum-Enabled Quantum-Safe Key Exchange*, Ghosh et al.) as a representative post-quantum key-exchange protocol using statistical randomness testing, machine-learning distinguishers, and software-observable side-channel analysis.

- **Course:** CSD493 Final-Year Capstone Project, Shiv Nadar University
- **Faculty Supervisor:** Prof. Rajib Mall
- **Industry Partner:** HCL Software
- **Team:** Sri Surya Srikar, Sashreek Kandagatla, Sane Yateesh Reddy
- **Dataset:** 1,106,944 quantum random numbers from the Australian National University (ANU) QRNG API (`qrng_combined.txt`, uint16 stream)

---

## 1. Executive Headline & What This Project Claims

The empirical evidence across `results/` cleanly separates **receiver-side decapsulation (`decrypt_it`)** (a genuine, mechanistically verified CPython timing side-channel) from **public channel / randomness handling** (no ML leakage detected), and separates **uninstrumented execution** from **`tracemalloc` instrumentation artifacts**:

> **Primary Finding (5,000-Trial Uninstrumented Evaluation, `results/decrypt_leak_5000_summary.md`):**
> Under a clean, uninstrumented software timing threat model at toy parameters (`n=4, k=2`, `5,000` QRNG trials, seed `42`, `53.70%` majority baseline, Linux x86_64 Intel Core i9-12900K), unmodified `decrypt_it()` leaks `secret_weight` at **65.25% ± 1.49%** binary accuracy (`MLP`, ROC-AUC `0.7076`, shuffled control `53.46%`, `+11.79` percentage-point net gap, `q`-residualized `64.79%`, permutation `p < 1/201 (0.0050)`) and **R² = +0.2260** continuous exact-weight regression (`SVR (RBF)`, `p < 1/201 (0.0050)`), driven by Spearman `ρ = +0.4166` (`p = 3.36e-209`) between `dec_timing_us` (`75.21 ± 6.37 µs`) and `secret_weight`.

> **Observer Effect (`tracemalloc` Amplifies the Leak):**
> Wrapping `decrypt_it()` inside Python's `tracemalloc.start()/stop()` — as done in earlier 500-trial collectors (`results/stress_test_traces.json`, `attack_c_decrypt_sidechannel.py`) — slows decryption **`~11.7×`** (`75.21 µs → 882.02 µs`) and **artificially amplifies** the side-channel signal: Spearman `ρ` jumps from `+0.4166` to **`+0.8241`**, classification accuracy rises from `65.25%` to **`84.37%`** (2-feature timing+CPU) / **`84.42%`** (4-feature), and regression `R²` rises from `+0.2260` to **`+0.6877`** / **`+0.7051`**. The root cause is CPython's allocation hook intercepting every new `PyLongObject` produced by `x * (±1)` while skipping cached small-int `0` products (`x * 0`). Accordingly, all earlier 500-trial `~80–82%` figures are explicitly marked **Legacy (`tracemalloc`-wrapped)** below.

> **Constant-Time Mitigation (`decrypt_it_constant_time`, `results/decrypt_ct_leak_5000_summary.md`):**
> Replacing pure-Python `mat_mult_mod` and conditional rounding branches with fixed-width vectorized NumPy `int64` arithmetic and bitwise selection eliminates the leak on the identical 5,000 QRNG trials: clean timing correlation drops to `ρ = -0.0258` (`p = 0.0677`), best classifier accuracy drops to **`53.56%`** (`p = 0.7065`, at `53.70%` majority baseline), and **all 5 regression models yield `R² < 0`** (no predictive skill).

> **Negative Findings (No ML Leakage Detected):**
> Across the same evaluation methodology (seed `42`, repeated CV, shuffled-label control, and `q`-confound checks):
> 1. Raw ANU QRNG vs. Mersenne Twister (`attack_a_distinguisher.py`, `attack_a_sequence.py`, `results/attack_a_level1_full_stats.json` — `48.10–50.00%` accuracy).
> 2. **Sender (`encrypt_it`)** execution traces (`Tier 1a`, legacy 500 trials: `53.68%` best vs. `55.40%` baseline and `53.86%` shuffled control).
> 3. **Public Channel** ciphertext statistical summaries (`Tier 1c`, legacy 500 trials: `52.84%` best vs. `55.40%` baseline and `51.18%` shuffled control).
> 4. `key_gen()` under `tracemalloc` (`attack_c_combined_ml.py` and the 5-level `attack_c_progressive_difficulty.py` ladder — all targets at or below majority baseline). *(Note: when measured uninstrumented with CPU core pinning in the extended positive control, `key_gen()` exhibits a subtle `+0.53 µs` shift from its single `mat_mult_mod(A, s, q)` call, `58.52%` MLP vs. `55.40%` baseline.)*
> 5. **Real ML-KEM-512 (`n=256, k=2, q=3329` via `kyber-py`) & QEQSKE Variant (per-key NTT `q`, QRNG-fed) Decapsulation (`attack_c_mlkem_decrypt_5000.py`):** Neither `secret_weight` nor individual coefficient sign `s[0][0]` leaks in clean `decaps`, isolated `_k_pke_decrypt`, or `tracemalloc` passes (all classifiers at majority baseline `51.05–52.24%` / `64.28–65.62%`, `p >= 0.1592`, all regression `R² < 0`). NTT transformation (`s_hat = NTT(s)`) eliminates coefficient-domain zero sparsity (`0.04–0.15 / 512` zeros in `s_hat` vs. `~160` zeros in `s`), extinguishing the CPython `*0` fast-path leak.

### Methodological Notes & Scope Boundaries
- **Permutation p-value convention (`< 1/(N+1)`):** With `N = 200` permutations in `permutation_test_score`, observing `0` permutations exceeding the real score hits the Phipson–Smyth resolution floor `1 / (N + 1) = 1/201 ≈ 0.004975` and is reported throughout as **`< 1/201 (0.0050)`** rather than an exact `0.0050`.
- **Negative-R² regression caveat:** Out-of-fold `R² <= 0` means a regressor performs *worse* than predicting the constant sample mean (`RMSE >= std(y) = 1.3320`). Negative-`R²` models (such as unpruned `Random Forest` on noisy clean timing, `R² = -0.0776`, and all 5 regressors on `decrypt_it_constant_time`) carry no valid continuous predictive skill and are explicitly caveated.
- **Not a proof of cryptographic security or LWE hardness:** At toy parameters `n=4, k=2`, the secret matrix is recoverable from public `(A, t)` in `<1 ms` (`break_poc.py`, `attack_e_algebraic_lwe.py`, `attack_e_lll_lattice.py`).
- **Software proxies, not hardware PMU/oscilloscope traces:** Timing (`perf_counter`), process CPU (`resource.getrusage` on Linux/macOS; legacy 500-trial scripts used `psutil`), and `tracemalloc` are software-level proxies.
- **Fault injection excluded:** Fault-injection attacks (Attack D) are intentionally out of scope (`attack_d_cca_mismatch.py` documents an unrelated missing Fujisaki–Okamoto check in the appendix).

---

## 2. Self-Audit: Two Cryptographic Breaks Found in Our Implementation

Run `python3 break_poc.py`:

1. **Break 1 — One-Time-Pad Reuse in `encrypt_it` (FIXED):**
   Earlier code sampled `r` and `e1` once per message and reused `res_v_1 = r*t + e1` across all ciphertext blocks `v_queue[i]`, so subtracting two blocks yielded `(bit_i - bit_j) * (q // 2)` and recovered the plaintext (up to global complement) with no key material. Fixed in `qeqske_core.py` by sampling fresh `r, e1, e2` per bit (`legacy_pad_reuse=False` default; `legacy_pad_reuse=True` preserved for reproduction).
2. **Break 2 — Brute-Forceable Secret at Toy Parameters `n=4, k=2` (INHERENT, NOT FIXED):**
   With `s` in `{-1, 0, 1}^{4×2}` and `q ~ 30,000`, each column has only `3^4 = 81` candidates and unit-bounded noise is negligible relative to `q`, enabling exact key recovery from `(A, t)` in `<1 ms` (`break_poc.py`, `attack_e_algebraic_lwe.py`, `attack_e_lll_lattice.py`).

---

## 3. Empirical Results (All Figures From `results/*.json`)

### 3.1 Primary 5,000-Trial `decrypt_it()` Evaluation: Clean vs. `tracemalloc` vs. Constant-Time

Evaluated on `5,000` fresh `(q, s, u, v_queue)` trials (`FIXED_PLAINTEXT = "HELLO"`, `n=4, k=2`, seed `42`; binary target `secret_weight > 5.0`, class 0 = `2,685`, class 1 = `2,315`, **Majority Baseline = 53.70%**; regression `secret_weight ∈ [0, 8]`, mean `5.3266`, SD `1.3320`) on `Linux 7.0.0-31-generic`, CPU `12th Gen Intel(R) Core(TM) i9-12900K`, Python `3.12.3`, NumPy `2.5.2` (`results/decrypt_leak_5000.json`, `results/decrypt_ct_leak_5000.json`):

| Metric (5,000 Identical QRNG Trials; Baseline = 53.70%) | Original `decrypt_it` (Clean Uninstrumented) | Original `decrypt_it` (`tracemalloc`-Wrapped) | Constant-Time `decrypt_it_constant_time` (Clean Uninstrumented) |
| :--- | :---: | :---: | :---: |
| **Decryption Correctness** | `4997/5000` (`99.94%`) | `4997/5000` (`99.94%`) | `4997/5000` (`99.94%`; `5000/5000` exact match with `decrypt_it`) |
| **Timing Mean ± SD (µs)** | `75.21 ± 6.37` | **`882.02 ± 92.90`** *(11.7× slower)* | `40.01 ± 2.56` *(191.60 ± 5.68 under tm)* |
| **Timing Median (IQR) µs** | `73.99 (3.23)` | — | `39.56 (0.95)` |
| **Timing Pearson `r` [95% Boot CI]** | `+0.1727 [+0.1437, +0.2034]` (`p = 9.09e-35`) | **`+0.6398 [+0.5939, +0.6868]`** | `-0.0238 [-0.0527, +0.0039]` (`p = 9.18e-02`) |
| **Timing Spearman `ρ` [95% Boot CI]** | **`+0.4166 [+0.3916, +0.4404]`** (`p = 3.36e-209`) | **`+0.8241 [+0.8113, +0.8352]`** *(amplified)* | **`-0.0258 [-0.0541, +0.0010]`** (`p = 6.77e-02`; tm `+0.0103`) |
| **CPU User `ρ` (`ru_utime`)** | `+0.4023` (`p = 6.99e-194`) | `+0.8223` | `-0.0322` (`p = 2.30e-02`; `r = -0.0236`) |
| **Best Classifier (5×10 CV)** | **65.25% ± 1.49%** (`MLP`, AUC `0.7076`) | **84.37%** (`GB`, 2-feat) / **84.42%** (`GB`, 4-feat) | **53.56% ± 0.20%** (`Logistic Reg.`, AUC `0.5056`; tm `53.93%`) |
| **Shuffled Control / `q`-Resid Acc** | `53.46%` / `64.79%` (`+11.79%` gap) | `52.62%` / `84.34%` (2-feat) | `53.66%` / `53.56%` (`-0.10%` gap) |
| **Best Classifier Perm `p`** | **`< 1/201 (0.0050)`** | **`< 1/201 (0.0050)`** | `0.7065` *(no signal; tm 2-feat `0.0498`)* |
| **Best Regression `R²`** | **`+0.2260 ± 0.0256`** (`SVR (RBF)`, RMSE `1.1706`) | **`+0.6877`** (`MLP`, 2f) / **`+0.7051`** (`GB`, 4f) | **All models `R² < 0`** *(best `-0.0020`, Linear Reg.; tm `-0.0026`)* |
| **Best Regression Perm `p`** | **`< 1/201 (0.0050)`** | **`< 1/201 (0.0050)`** | `0.7164` *(no signal; worse than mean baseline)* |

> **Why `tracemalloc` amplifies the leak:** Under `tracemalloc`, `dec_ram_peak_bytes` (`r = +0.0672`) and `dec_num_allocations` (`4.00 ± 0.01`, `r = -0.0178`) contribute almost no direct discriminative feature beyond timing (`84.37%` with 2 timing/CPU features vs. `84.42%` with all 4 features). Instead, `tracemalloc` hooks every Python heap allocation inside `mat_mult_mod`, amplifying the timing divergence between `x * 0` (CPython's cached small-integer `0` object) and `x * (±1)` (newly allocated `PyLongObject` products/sums) by an order of magnitude (`~11.7×` runtime inflation).

#### Full 5-Model Clean Uninstrumented Breakdown (`results/decrypt_leak_5000_summary.md`)

**Binary Classification (`secret_weight > 5.0`, Majority Baseline = `53.70%`):**

| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP** | **65.25% ± 1.49%** | 0.7076 | 53.46% ± 0.38% | **+11.79%** | 64.79% ± 1.25% | **< 1/201 (0.0050)** |
| **SVM (RBF)** | **65.21% ± 1.37%** | 0.7047 | 53.34% ± 0.39% | **+11.87%** | 65.22% ± 1.21% | **< 1/201 (0.0050)** |
| **Gradient Boosting** | **64.76% ± 1.63%** | 0.6957 | 53.07% ± 1.39% | **+11.69%** | 64.53% ± 1.35% | **< 1/201 (0.0050)** |
| **Random Forest** | 56.81% ± 1.18% | 0.6103 | 51.97% ± 1.25% | +4.84% | 60.33% ± 1.37% | **< 1/201 (0.0050)** |
| **Logistic Regression** | 56.00% ± 0.69% | 0.6980 | 53.64% ± 0.16% | +2.37% | 56.22% ± 0.75% | **< 1/201 (0.0050)** |

**Regression Target (Exact `secret_weight`, Sample SD = `1.3320`):**

| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | `q`-Residualized R² | Perm `p`-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **SVR (RBF)** | **+0.2260 ± 0.0256** | **1.1706** | -0.0338 ± 0.0121 | +0.2217 ± 0.0249 | **< 1/201 (0.0050)** | — |
| **MLP** | **+0.2246 ± 0.0307** | 1.1716 | -0.0160 ± 0.0733 | +0.2198 ± 0.0295 | **< 1/201 (0.0050)** | — |
| **Gradient Boosting** | **+0.2206 ± 0.0240** | 1.1747 | -0.0188 ± 0.0104 | +0.2190 ± 0.0242 | **< 1/201 (0.0050)** | — |
| **Linear Regression** | +0.0251 ± 0.0217 | 1.3141 | -0.0033 ± 0.0068 | +0.0252 ± 0.0223 | **< 1/201 (0.0050)** | — |
| **Random Forest** | -0.0776 ± 0.0404 | 1.3810 | -0.3859 ± 0.0467 | +0.0879 ± 0.0358 | < 1/201 (0.0050) | **Negative R² (worse than mean baseline)** |

> *Negative-R² caveat:* Unpruned `Random Forest` overfits the 2D timing/CPU scatter (`R² = -0.0776 < 0`, `RMSE = 1.3810 > 1.3320`); its permutation `p < 1/201` merely reflects beating an even worse shuffled fit (`R² = -0.3859`), not usable continuous prediction. On `decrypt_it_constant_time`, *all 5 regressors* yield `R² < 0` (`-0.0020` to `-0.3749`, `p = 0.6667–0.8507`), confirming zero regression signal.

**Confound & drift checks (clean original pass):** `q` is uncorrelated with `secret_weight` (`r = -0.0031, p = 0.8284`; `ρ = -0.0042, p = 0.7692`), execution order is uncorrelated with `secret_weight` (`r = -0.0082, p = 0.5619`) and `dec_timing_us` (`r = -0.0101, p = 0.4761`), and `q`-residualized ML accuracies (`64.79–65.22%`) match the unresidualized signal.

#### Root-Cause Microbenchmark & Explaining the Residual NumPy Weight Trend (`results/decrypt_hypothesis_ct_summary.md`)

1. **Why original `decrypt_it` leaks (`attack_c_decrypt_hypothesis.py`):**
   - `cProfile` across `2,000` calls (`0.3090 s`) shows `mat_mult_mod` consumes `42.0%` of total execution time (`0.1298 s`) and accounts for `82%` (`+0.0183 s` of `+0.0223 s`) of the paired `w=8` vs `w=0` difference on identical ciphertexts.
   - In a CPU-pinned (`core 0`), randomized 40-call batch microbenchmark (`q = 32771`, `300` reps/weight), pure-Python `mat_mult_mod` scales monotonically from `34.25 µs` (`w=0`) to `44.25 µs` (`w=8`, Spearman `ρ = +0.7306 [+0.7048, +0.7561]`).
   - Operand-isolation controls confirm `all_+1` (`40.32 µs`) and `all_-1` (`40.88 µs`) are `+8.42 µs` (`p = 4.53e-120`) and `+8.98 µs` (`p = 1.43e-122`) slower than `all_0` (`31.90 µs`), matching CPython `long_mul` / `long_add` micro-op timings (`x*0 = 11.01 ns` vs `x*1 = 16.15 ns` and `x*(-1) = 16.54 ns`).
2. **Explaining the residual NumPy `int64` weight trend (`w=0: 2.79 µs` → `w=8: 3.13 µs`, `Δ = +0.34 µs`, `ρ = +0.4354`):**
   Isolating every sub-operation of `(v_arr - np.matmul(u_arr, np.asarray(s, dtype=np.int64))) % q64` reveals that **neither `np.matmul` nor subtraction produces the trend** — the trend arises almost entirely from **signed floor-modulo on negative intermediate differences**:

| Isolated NumPy Component | `w=0` Med (µs) | `w=8` Med (µs) | Δ (`w8 − w0`) | Spearman `ρ` vs. Weight | Spearman `p` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `np.matmul(u_arr, s_np)` | 1.215 | 1.177 | `-0.038 µs` | `-0.1235` | `1.22e-10` |
| `v_arr - np.matmul(u_arr, s_np)` | 1.502 | 1.452 | `-0.050 µs` | `-0.1491` | `6.84e-15` |
| **`diff % q64` (can be negative)** | **0.916** | **1.172** | **`+0.256 µs`** | **`+0.3393`** | **`9.56e-74`** |
| **`(diff + 4*q64) % q64` (all `>= 0`)** | **1.329** | **1.313** | **`-0.015 µs`** | **`-0.0186`** | **`3.35e-01`** |
| `np.asarray(s_py, dtype=np.int64)` | 0.743 | 0.771 | `+0.028 µs` | `+0.2736` | `1.51e-47` |
| `(v - u@s_np + 4*q64) % q64` (pre-converted, `>= 0`) | 2.651 | 2.620 | `-0.030 µs` | `-0.1449` | `3.83e-14` |

   - At `w = 0`, `u @ s = 0`, so `diff = v_arr >= 0` (`0.0%` negative entries). As `w` increases to `8`, `u @ s` can exceed `v_arr`, pushing `P(diff < 0)` up monotonically (`w=1: 8.8%`, `w=2: 23.0%`, `w=4: 18.5%`, `w=8: 34.2%`).
   - NumPy's `int64` `%` implements Python floor-modulo using a data-dependent sign-fixup branch when the remainder is negative (`if rem != 0 && (rem < 0) != (q < 0): rem += q`) plus signed hardware `idiv`, causing branch mispredictions as negative elements increase.
   - Shifting `diff + 4*q64` before `% q64` forces every dividend `>= 0` and completely eliminates the `% q64` trend (`ρ = -0.0186, p = 0.335`). The remaining `~0.03 µs` comes from `np.asarray(s_py)` unpacking Python `int` `0` vs `±1`.
   - Across full `~40.0 µs` `decrypt_it_constant_time` calls (`IQR = 0.95 µs`, `SD = 2.56 µs`), this `~0.34 µs` extreme-weight micro-trend (and `<0.1 µs` across typical QRNG weights `4..7`) is buried beneath wrapper/OS jitter (`ρ = -0.0258, p = 0.0677`, `53.56%` accuracy); production deployments should nonetheless replace `%` / `idiv` with branch-free Barrett or Montgomery reduction.

#### Cross-Platform Effect-Size Replication Status
Every `results/*.json` file now records standardized `"environment"` metadata (`platform`, `os`, `cpu_model`, `python_version`, `numpy_version`). `attack_c_decrypt_5000.py` provides `--tag <host>` and `--compare <json_a> <json_b>` to measure how effect size varies across CPUs, OSes, and Python/NumPy builds. Machine 1 (`Linux 7.0.0-31-generic`, `12th Gen Intel(R) Core(TM) i9-12900K`, Python `3.12.3`, NumPy `2.5.2`) is complete above; Machine 2 (`attack_c_decrypt_5000.py --tag <host2>`) must be executed on a second physical machine (only one physical host is accessible in this environment).

---

### 3.2 Legacy 500-Trial Sender ↔ Receiver Side-Channel Evaluation (`tracemalloc`-Wrapped, macOS arm64)

> **Legacy Marker:** The 500-trial datasets in `results/sender_receiver_traces.json`, `results/stress_test_traces.json`, and `results/attack_c_full_stats.json` were collected on `macOS-26.6.2-arm64` (Python `3.13.1`, NumPy `2.2.6`) with `tracemalloc.start()` wrapping `decrypt_it()` and `encrypt_it()`. As proven in Section 3.1, **`tracemalloc` amplifies the receiver leak** (`65.25%` clean → `84.37%` `tracemalloc` at `n=5,000`), which explains why the legacy Tier 1b accuracy sits at `~80–82%`. The null findings on **Sender (`Tier 1a`)** and **Public Channel (`Tier 1c`)** remain unaffected.

500 Alice ↔ Bob sessions (`FIXED_PLAINTEXT = "HELLO"`, `n=4, k=2`), binary target `secret_weight > 5.0` (`0=277`, `1=223`, **Majority Baseline = 55.40%**), `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`:

| Observational Tier *(Legacy 500-Trial, `tracemalloc`-Wrapped)* | Active Features | Best Model (`stress_test_traces.json`) | Real CV Acc (Mean ± SD) | Shuffled Control | Net Gap (Real − Shuf) | `q`-Residualized Acc | Verdict |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1a: Sender (`encrypt_it`)** | 4 | Logistic Regression | 53.68% ± 2.03% | 53.86% ± 2.28% | −0.18% | 53.74% ± 2.25% | **No Signal (At Baseline)** |
| **Tier 1b: Receiver (`decrypt_it`, legacy tm)** | 4 | Gradient Boosting | **82.18% ± 3.38%** | 51.84% ± 3.41% | **+30.34%** | **82.68% ± 3.51%** | **Confirmed Leak (tm-Amplified)** |
| **Tier 1b: Receiver (`decrypt_it`, legacy tm)** | 4 | Random Forest | **80.42% ± 3.27%** | 48.36% ± 4.14% | **+32.06%** | **81.70% ± 3.29%** | **Confirmed Leak (tm-Amplified)** |
| **Tier 1c: Public Channel (`u, v`)** | 24 | SVM (RBF) | 52.84% ± 3.28% | 51.18% ± 3.41% | +1.66% | 50.14% ± 3.86% | **No Signal (At Baseline)** |
| **Tier 2: Sender + Receiver** | 8 | Gradient Boosting | **81.98% ± 2.99%** | 53.36% ± 4.54% | **+28.62%** | **83.02% ± 3.58%** | **Confirmed (Receiver-Driven)** |
| **Tier 3: ALL Combined** | 32 | Gradient Boosting | **81.96% ± 3.69%** | 48.98% ± 3.96% | **+32.98%** | **81.66% ± 4.01%** | **Confirmed (Receiver-Driven)** |

All 5 models on legacy **Tier 1b: Receiver Only** (`results/stress_test_traces.json`, `results/attack_c_full_stats.json`, 200 permutations):

| Tier 1b Model *(Legacy 500-Trial, `tracemalloc`)* | `stress_test_traces.json` Real Acc | Shuffled Control | Gap | `q`-Resid Acc | `attack_c_full_stats.json` Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | 82.18% ± 3.38% `[75.0–90.0%]` | 51.84% ± 3.41% | +30.34% | 82.68% ± 3.51% | 81.14% ± 3.22% | **< 1/201 (0.0050)** |
| **Random Forest** | 80.42% ± 3.27% `[75.0–90.0%]` | 48.36% ± 4.14% | +32.06% | 81.70% ± 3.29% | 80.26% ± 3.69% | **< 1/201 (0.0050)** |
| **MLP** | 78.04% ± 4.96% `[68.0–87.0%]` | 54.06% ± 2.57% | +23.98% | 76.94% ± 4.60% | 80.26% ± 4.31% | **< 1/201 (0.0050)** |
| **SVM (RBF)** | 61.88% ± 6.31% `[51.0–79.0%]` | 54.24% ± 2.29% | +7.64% | 59.02% ± 4.14% | 61.88% ± 6.37% | **< 1/201 (0.0050)** |
| **Logistic Regression** | 55.60% ± 4.04% `[44.0–67.0%]` | 54.98% ± 1.58% | +0.62% | 55.94% ± 4.07% | 55.60% ± 4.09% | 0.1393 |

**Legacy independent `decrypt_it()` profiling (`attack_c_decrypt_sidechannel.py`, 500 `tracemalloc`-wrapped trials in `results/attack_c_full_stats.json`):** `cpu_user_us` Pearson `r = 0.7405` (`ρ = 0.7786`), `timing_us` `r = 0.6494` (`ρ = 0.7580`), `ram_peak_bytes` `r = 0.0312` (`p = 0.4863`), `num_allocations` `r = -0.0655` (`p = 0.1438`); ML accuracy (`5×10` CV): LR `81.86% ± 2.60%`, MLP `81.54% ± 2.87%`, SVM `81.46% ± 2.82%`, RF `78.22% ± 3.30%` (all permutation `p < 1/201 (0.0050)`).

---

### 3.3 Extended Positive Control (Full 8D + Decrypt Features) & Graded Leak Threshold (`results/attack_c_leaky_vs_clean.json`)

`attack_c_leaky_vs_clean.py` now evaluates the **full main-pipeline feature set** (`timing_us`, `cpu_user_us`, `ram_peak_bytes`, `num_allocations`) across **both `key_gen()` and `decrypt_it()`** (`Full 8D`, plus `KeyGen 4D` and `Decrypt 4D` subsets, `500` QRNG trials, majority baseline `55.40%`, `5×10` Stratified CV, CPU-pinned uninstrumented timing with isolated `tracemalloc` allocation capture on Intel i9-12900K):

**Part 1 — Clean vs. Leaky Comparison (500 Trials):**
- **Clean Baseline (`key_gen_clean` + `decrypt_it_constant_time`):**
  - `Decrypt 4D`: all 5 models at or below baseline — `Logistic Regression 54.34% ± 2.00%` (`p = 0.9453`), `MLP 53.88%` (`p = 0.9104`), `SVM 53.86%` (`p = 0.9303`), `GB 49.92%`, `RF 48.90%`.
  - `Full 8D`: `LR 54.44%` (`p = 0.8706`), `SVM 56.58%` (`p = 0.2886`), `GB 56.84%` (`p = 0.1692`), `RF 57.00%` (`p = 0.0746`), `MLP 57.48% ± 5.22%` (`p < 1/201 (0.0050)`, driven by `KeyGen 4D` `58.52%` where uninstrumented `key_gen_clean` executes one `mat_mult_mod(A, s, q)` call with `+0.53 µs` median shift).
- **Leaky KeyGen Control (`key_gen_leaky` + `decrypt_it_constant_time`):** **`100.00% ± 0.00%`** across all 5 models on both `KeyGen 4D` and `Full 8D` (`p < 1/201 (0.0050)`).
- **Leaky Decrypt Control (`+50.0 µs` secret-dependent delay on `decrypt_it_constant_time`):** **`99.80% ± 0.40%`** (`RF`, `Full 8D`) and **`99.60% ± 0.53%`** (`SVM`, `Decrypt 4D`; all 5 models `98.22–99.80%`, `p < 1/201 (0.0050)`).
- **Natural Unmitigated `decrypt_it`:** `62.42% ± 4.42%` (`MLP`, `Full 8D`, `p < 1/201 (0.0050)`) and `61.48% ± 4.45%` (`MLP`, `Decrypt 4D`, `p < 1/201 (0.0050)`).

**Part 2 — Graded Secret-Dependent Delay Sensitivity Sweep (`0.0, 0.5, 1.0, 2.0, 5.0 µs`):**
Injecting a calibrated busy-wait delay whenever `secret_weight > 5.0` into `decrypt_it_constant_time` (baseline decrypt IQR = `1.91 µs`, interleaved pinned measurement):

| Injected Delay (µs) | Measured Median Δt (µs) | Full 8D Best Acc (Model) | Full 8D Perm `p` | Decrypt 4D Best Acc (Model) | Decrypt 4D Perm `p` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 µs** | −0.13 µs | 57.56% ± 4.97% (`MLP`) | 0.0398 | 55.44% ± 0.97% (`Logistic Reg.`) | 0.4876 |
| **0.5 µs** | +0.64 µs | **61.62% ± 3.70%** (`Random Forest`) | **< 1/201 (0.0050)** | **60.12% ± 3.97%** (`MLP`) | **< 1/201 (0.0050)** |
| **1.0 µs** | +1.18 µs | **69.86% ± 4.54%** (`Gradient Boost`) | **< 1/201 (0.0050)** | **67.80% ± 4.94%** (`MLP`) | **< 1/201 (0.0050)** |
| **2.0 µs** | +2.17 µs | **83.14% ± 3.23%** (`MLP`) | **< 1/201 (0.0050)** | **82.66% ± 2.94%** (`MLP`) | **< 1/201 (0.0050)** |
| **5.0 µs** | +5.19 µs | **93.20% ± 2.61%** (`Random Forest`) | **< 1/201 (0.0050)** | **93.54% ± 2.60%** (`MLP`) | **< 1/201 (0.0050)** |

- **Empirical Detection Threshold:** Despite single-call timing jitter of `1.91 µs` IQR (`~40.7 µs` median), the pipeline already detects a **`0.5 µs` secret-dependent leak** with statistical significance (`p < 1/201 (0.0050)` on both Full 8D `61.62%` and Decrypt 4D `60.12%`), clears a `+10` percentage-point margin at **`1.0 µs`** (`69.86%`), and exceeds `80%` accuracy by **`2.0 µs`** (`83.14%` / `82.66%`).

![Graded Leak Detection Threshold](results/graded_leak_threshold.png)

---

### 3.4 Attack C Ladder, Key-Gen Combined ML, RAM/Cache Proxies, ML-KEM & Attack A

- **Attack C — Progressive Difficulty Ladder on `key_gen()` (Legacy macOS arm64, `results/attack_c_progressive_difficulty.json`):**
  1,000 trials (`Timing + RAM + CPU` under `tracemalloc`), `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`. Reconciled against `results/full_run.log`: all five targets remain below majority baseline (`-0.23` to `-0.99` pts):

| Target | Difficulty | Classes | Baseline | Best Model | Best Acc (Mean ± SD) | 95% CI | Margin | Perm `p`-value |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| `Secret weight (binary)` | Easy | 2 | 53.90% | Logistic Regression | 53.43% ± 1.13% | ±0.32 pts | **−0.47 pts** | 0.9652 |
| `Secret sum (binary)` | Easy | 2 | 57.70% | Logistic Regression | 57.40% ± 0.54% | ±0.15 pts | **−0.30 pts** | 0.4925 |
| `Secret weight (multi-class)` | Medium | 3 | 46.10% | Logistic Regression | 45.71% ± 0.56% | ±0.16 pts | **−0.39 pts** | 0.2886 |
| `s[0][0] sign (positive vs not)` | Hard | 2 | 65.90% | Logistic Regression | 65.67% ± 0.46% | ±0.13 pts | **−0.23 pts** | 0.7562 |
| `s[0][0] exact value` | Very Hard | 3 | 34.10% | Random Forest | 33.11% ± 3.00% | ±0.85 pts | **−0.99 pts** | 0.4328 |

- **Combined `key_gen()` Side-Channel ML (`attack_c_combined_ml.py`, Legacy 500 trials, `results/attack_c_full_stats.json`):**
  Correlations negligible (`|r| <= 0.0933`, all Bonferroni `p >= 0.1478`); classifiers at baseline: SVM `55.42% ± 1.49%` (`p = 0.6418`), LR `54.76% ± 1.44%` (`p = 0.8856`), RF `53.66% ± 4.71%` (`p = 0.4179`), MLP `53.58% ± 3.55%` (`p = 0.9154`).
- **System Comparison (`system_comparison.py`, Legacy `results/attack_c_full_stats.json`):**
  `QEQSKE-clean`: LR `56.20% ± 1.12%` (`p = 0.7612`), RF `51.60% ± 5.42%` (`p = 0.5572`); `Standard ML-KEM-512`: LR `46.40% ± 5.41%` (`p = 0.8259`), RF `49.17% ± 6.72%` (`p = 0.3085`); `QEQSKE-leaky`: **`100.00% ± 0.00%`** (`p < 1/201 (0.0050)`).
- **Fixed RAM & Cache/CPU Software Proxies (`results/attack_c_ram_cache.json`, Legacy macOS arm64):**
  - `attack_c_ram_sidechannel.py` (100 trials): `key_gen()` peak `1780.5 ± 99.5` bytes (`r_peak = 0.1093, p = 0.2791`; `r_alloc = 0.0329, p = 0.7454`); `encrypt_it()` peak `12762.2 ± 454.5` bytes (`r_peak = 0.2497, p = 0.0122`; `r_alloc = 0.2376, p = 0.0173`).
  - `attack_c_cache_sidechannel.py` (20 trials): `key_gen()` `19.28 ± 3.33 µs/call` (`r_user = 0.3939, p = 0.0858`); `encrypt_it()` `139.43 ± 1.30 µs/call` (`r_user = 0.0807, p = 0.7351`).
- **Attack A — QRNG vs. Mersenne Twister Distinguisher (`results/attack_a_level1_full_stats.json`, Legacy macOS arm64):**

| Model | Evaluation Protocol | Sample Size | Accuracy (Mean ± SD) | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Random Forest** | 5×5 Repeated Stratified CV | 49,408 windows | 49.93% ± 0.67% | 0.5000 |
| **XGBoost** | 5×5 Repeated Stratified CV | 49,408 windows | 49.89% ± 0.45% | 0.4992 |
| **SVM** | 5×5 Repeated Stratified CV | 15,000 subsample | 49.87% ± 0.90% | 0.4996 |
| **GRU** (`nn_from_scratch.py`) | 10 Repeated Holdouts (75/25) | 2,000 windows | 50.00% ± 2.52% | 0.4890 |
| **LSTM** (`nn_from_scratch.py`) | 10 Repeated Holdouts (75/25) | 2,000 windows | 48.20% ± 3.92% | 0.4831 |
| **Transformer** (`nn_from_scratch.py`) | 10 Repeated Holdouts (75/25) | 2,000 windows | 48.10% ± 1.45% | 0.4991 |

---

### 3.5 Scaling to Real ML-KEM Parameters (`n=256, k=2, q=3329` via `kyber-py`) vs. QEQSKE Variant

To test whether the toy `decrypt_it` weight leak survives at real post-quantum parameters, `attack_c_mlkem_decrypt_5000.py` ports the full 5,000-trial Prompt 2 pipeline to `n=256, k=2` (`eta_1=3, eta_2=2, du=10, dv=4`, configurable to `--k 3` for ML-KEM-768) using unmodified `kyber_py.ml_kem.ML_KEM` (`q=3329` fixed) alongside a configurable **QEQSKE Variant** (`--variant qeqske`, ANU QRNG-fed `qrng_combined.txt`, per-key 16-bit NTT-friendly prime `q ≡ 1 mod 256` across all 45 primes in `[3329, 64513]`, `d_poly=16`). Both variants achieved **`5000/5000` (100.00%)** decapsulation correctness on Linux x86_64 Intel Core i9-12900K (`results/mlkem512_decrypt_leak_5000_summary.md`, `results/mlkem512_qeqske_decrypt_leak_5000_summary.md`, `results/mlkem512_comparison_summary.md`):

| System (5,000 Trials) | Clean `decaps` Timing | Clean `ρ_weight` | Clean Weight Clf vs. Baseline | Clean Weight Reg `R²` | Isolated `_k_pke_decrypt` | `tracemalloc` `decaps` | `s[0][0] > 0` Clf vs. Baseline | Balanced `s[0][0] != 0` Sign Clf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Toy QEQSKE (`decrypt_it`)** (`n=4, k=2`) | 75.21 ± 6.37 µs | **+0.4166** | **65.25%** vs 53.70% (`MLP`, `p < 1/201 (0.0050)`) | **+0.2260** (`SVR (RBF)`) | N/A (no FO) | **84.37%** (`GB`, `ρ=+0.8241`) | N/A (`Prog. Ladder`: baseline) | N/A |
| **Standard ML-KEM-512 (`kyber-py`)** (`n=256, k=2, q=3329`) | 2097.0 ± 295.0 µs | +0.0147 (`p=0.300`) | 52.12% vs 52.02% (`SVM`, `p=0.1592`) | **All < 0** (max `-0.0019`, `p=0.5871`) | 52.24% (`MLP`, `ρ=+0.0006`, `p=0.7711`) | 51.84% (`SVM`, `ρ=-0.0166`, `p=0.4726`) | 65.62% vs 65.62% (`LR`, `p=0.6716`) | 51.10% vs 50.29% (`GB`, `p=0.1542`, `n=3,418`) |
| **QEQSKE-Variant ML-KEM-512** (`n=256, k=2`, per-key NTT `q`, QRNG) | 1983.0 ± 97.5 µs | +0.0120 (`p=0.396`) | 51.09% vs 51.14% (`LR`, `p=0.2438`) | **All < 0** (max `-0.0016`, `p=0.9005`) | 51.33% (`SVM`, `ρ=-0.0098`, `p=0.7313`) | 51.05% (`SVM`, `ρ=+0.0177`, `p=0.5224`) | 64.28% vs 64.30% (`SVM`, `p=0.3085`) | 51.76% vs 51.78% (`LR`, `p=0.4328`, `n=3,447`) |

**Does the leak survive at `n=256`? No — it disappears completely:**
1. **Why NTT extinguishes the CPython `*0` weight leak:** In toy QEQSKE, `decrypt_it` multiplied directly against raw centered secret entries `s ∈ {-1, 0, 1}` (~33% zeros), exposing CPython's `x * 0` fast-path. In real ML-KEM (`kyber-py`), decapsulation multiplies in the **NTT domain** (`s_hat.dot(u_hat)`). Although centered `s` averages `160.04` zeros (`standard`) and `159.88` zeros (`qeqske`) out of `512` coefficients (`secret_weight` `351.96 ± 10.45` and `352.12 ± 10.67`), the NTT representation `s_hat` is pseudo-uniform over `Z_q` with only **`0.15 / 512` zeros** (`standard`) and **`0.04 / 512` zeros** (`qeqske`) — practically zero. Consequently, neither clean `decaps`, isolated `_k_pke_decrypt` (`585.7 µs` / `563.8 µs`), nor `tracemalloc`-wrapped `decaps` (`21.5 ms` / `22.3 ms`) leaks `secret_weight` or `s[0][0]` sign.
2. **Per-key `q` timing confound vs. zero secret leakage in QEQSKE-Variant ML-KEM:** In `mlkem512_qeqske`, variable prime `q` correlates with clean execution timing (`q` vs `dec_timing_us` `r = +0.1493`, `dec_cpu_user_us` `r = +0.3129`, `pke_dec_timing_us` `r = +0.1578`) and induces a spurious correlation with `dk_bit_hw` (`ρ = +0.3082, p = 1.90e-110`) because larger `q` widens public/encoded integer bit lengths. Crucially, `q` is uncorrelated with `secret_weight` (`r = +0.0161, p = 0.2557`) and `s00_positive` (`r = +0.0129, p = 0.3604`), and `q`-residualized ML remains strictly at baseline (`51.12%` vs. `51.14%`), confirming zero secret-key leakage. *(Caveat: `kyber-py` is a pure-Python educational reference implementation; absence of a secret-weight timing leak in `kyber-py` under software tracing does not replace masked hardware power/EM side-channel evaluations.)*

---

## 4. Complete Repository Script Inventory

### Core Protocol, Randomness, Metadata & Neural Infrastructure
| Script | Purpose |
| :--- | :--- |
| `qeqske_core.py` | QEQSKE Algorithms 1–10 (`is_prime`, `closest_prime`, `key_gen`, `encrypt_it`, `decrypt_it`, hardened tag helpers, and branch-free NumPy `decrypt_it_constant_time`). |
| `qrng_handler.py` | `O(1)` `deque`-backed `QRNGSource` (optional `cycle=True`) and `MersenneSource` baseline adapter. |
| `env_metadata.py` | Shared environment metadata collector (`platform`, `os`, `cpu_model`, `python_version`, `numpy_version`) and `< 1/(N+1)` permutation p-value formatting helpers. |
| `test_qeqske.py` | End-to-end Alice ↔ Bob key exchange correctness test. |
| `collect_qrng_data.py` | Fetches raw `uint16` quantum random batches from the ANU QRNG API. |
| `merge_qrng.py` | Merges raw QRNG files (`qrng_numbers*.txt`) into `qrng_combined.txt` (1,106,944 values). |
| `mersenne_baseline.py` | Generates deterministic Mersenne Twister comparison batches (`seed=42`). |
| `statistical_tests.py` | NIST-style statistical randomness suite (Frequency, Runs, Longest Run, Chi-Square, Serial Correlation). |
| `nn_from_scratch.py` | Pure-NumPy LSTM, GRU, and Transformer implementations with analytical BPTT. |
| `verify_gradients.py` | Numerical finite-difference gradient check for `nn_from_scratch.py`. |
| `collect_reproducibility_info.py` | Captures software versions, hardware specs, CPU model, seeds, and hyperparameters to `results/reproducibility.json`. |
| `run_full_suite.py` | Earlier 5-step runner covering correctness, NIST tests, Attack A, Attack B, and Attack C timing groundwork. |
| `Large_Scale_Run_Script.py` | Extended 12-step runner across full QRNG dataset, Attack A/B/C pipelines, system comparison, and metadata collection. |

### Self-Audit, Attack Suites & Side-Channel Pipelines
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
| `attack_c_leaky_vs_clean.py` | Extended positive control & sensitivity threshold suite: full 8D (`key_gen` + `decrypt`) clean vs. leaky comparisons + graded `0.0–5.0 µs` delay sweep (`results/attack_c_leaky_vs_clean.json`, `results/graded_leak_threshold.png`). |
| `attack_c_progressive_difficulty.py` | Attack C — 5-level progressive difficulty ladder on `key_gen()` (`results/attack_c_progressive_difficulty.json`). |
| `attack_c_stats.py` | Shared statistical helper (`compute_correlation_stats`, `compute_ml_stats`, `update_json_results`) writing `results/attack_c_full_stats.json`. |
| `attack_c_decrypt_sidechannel.py` | Legacy 500-trial `decrypt_it()` profiling under `tracemalloc` (`results/attack_c_full_stats.json`). |
| `attack_c_decrypt_5000.py` | Primary 5,000-trial `decrypt_it()` leakage harness (clean vs. `tracemalloc`, randomized order, warm-up, bootstrap CIs, classification + regression, `--tag` / `--compare` cross-host replication; `results/decrypt_leak_5000.json`). |
| `attack_c_mlkem_decrypt_5000.py` | Real ML-KEM (`n=256, k=2/3, q=3329` via `kyber-py`) + configurable QEQSKE variant (`--variant`, per-key NTT `q`, QRNG) 5,000-trial decrypt evaluation targeting `secret_weight` and `s[0][0]` sign (`results/mlkem512*_decrypt_leak_5000.json`). |
| `attack_c_decrypt_hypothesis.py` | Hypothesis test (`mat_mult_mod` weight 0..8 microbenchmark, NumPy residual-trend decomposition, `cProfile`/`dis`, and 5,000-trial `decrypt_it_constant_time` verification; `results/decrypt_hypothesis_ct.json`). |
| `attack_sc_sender_receiver_traces.py` | Legacy Attack SC-1 — 500-trial Alice ↔ Bob `tracemalloc`-wrapped trace collector capturing 34 features (`results/sender_receiver_traces.json`). |
| `attack_sc_ml_classifiers.py` | Legacy Attack SC-2 — 3-tier ML classifier evaluation (Sender, Receiver, Channel, Combined) across 5 model families (`results/attack_c_full_stats.json`). |
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

# Primary 5,000-trial decrypt evaluation, hypothesis test, and constant-time verification
python3 attack_c_decrypt_5000.py
python3 attack_c_decrypt_hypothesis.py

# Second-machine replication & cross-host effect-size comparison
python3 attack_c_decrypt_5000.py --tag host2
python3 attack_c_decrypt_5000.py --compare results/decrypt_leak_5000.json results/decrypt_leak_5000_host2.json

# Extended full-feature positive control & graded delay threshold (0.5–5.0 us)
python3 attack_c_leaky_vs_clean.py

# Real ML-KEM (n=256, k=2 or 3) standard vs. QEQSKE-variant 5,000-trial evaluation
python3 attack_c_mlkem_decrypt_5000.py --variant both --k 2 --n-trials 5000

# Legacy 500-trial Sender <-> Receiver trace collection, ML tiers, and stress tests
python3 attack_sc_sender_receiver_traces.py
python3 attack_sc_ml_classifiers.py
python3 stress_test_traces.py
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
| Attack C (`key_gen` Time / RAM / CPU / Ladder) | Complete (Legacy) | No leakage above baseline under `tracemalloc` (`results/attack_c_progressive_difficulty.json`, `results/attack_c_ram_cache.json`) |
| **Attack C Extended Positive Control & Sensitivity Threshold** | **Complete** | **Full 8D + Decrypt 4D; 100% on leaky controls, `0.5 µs` statistically significant threshold (`p < 1/201`), `83.14%` at `2 µs` (`results/attack_c_leaky_vs_clean.json`, `results/graded_leak_threshold.png`)** |
| **500-Trial Sender / Channel / Receiver ML & Stress Test** | **Complete (Legacy — `tracemalloc`-Wrapped)** | **Receiver `80.42–82.18%` reflects `tracemalloc` amplification; Sender & Channel at baseline (`results/stress_test_traces.json`)** |
| **Attack C-5 5,000-Trial Clean vs. `tracemalloc` Evaluation** | **Complete** | **Clean `65.25%` (`ρ = +0.4166`, `R² = +0.2260`) vs. `tracemalloc` `84.37%` (`ρ = +0.8241`, `R² = +0.6877`) (`results/decrypt_leak_5000_summary.md`)** |
| **Root-Cause Microbenchmark & Constant-Time Verification** | **Complete** | **CPython `*0` vs `*±1` isolated; NumPy residual trend explained via signed `% q64`; `decrypt_it_constant_time` drops to `53.56%` (`p = 0.7065`, all `R² < 0`) (`results/decrypt_ct_leak_5000_summary.md`, `results/decrypt_hypothesis_ct_summary.md`)** |
| Cross-Machine 5,000-Trial Replication (`attack_c_decrypt_5000.py`) | Pending Second Machine | Machine 1 (Linux x86_64 Intel i9-12900K) complete; `--tag` / `--compare` CLI ready for Machine 2 run |
| Attack C Real Hardware Power/Cache Traces | Pending HCL Access | Requires physical oscilloscope / hardware PMU counters |
| Attack D (Fault Injection) | Out of Scope (Ignored) | Excluded per project scope; `attack_d_cca_mismatch.py` retained as appendix CCA note |
| **Scale to `n=256` (Real ML-KEM vs. QEQSKE Variant)** | **Complete** | **5,000 trials (`k=2`, `kyber-py` + QRNG per-key NTT `q`): leak vanishes (`52.12%` / `51.09%` weight, `65.62%` / `64.28%` `s[0][0]>0`, all `R² < 0`; `results/mlkem512_comparison_summary.md`)** |
