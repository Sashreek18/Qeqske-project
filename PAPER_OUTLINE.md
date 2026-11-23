# Paper Outline: Calibrated Machine-Learning Side-Channel Auditing of Post-Quantum Key Exchange (QEQSKE vs. ML-KEM-512)

## Working Title
**Mind the Profiler: Calibrated Machine-Learning Side-Channel Auditing of Post-Quantum Key Exchange from Toy Prototypes to ML-KEM-512**

## Abstract (Draft Summary)
Machine-learning (ML) classifiers are increasingly used to audit post-quantum cryptographic implementations for timing, memory, and channel side-channels, yet naive pipelines suffer from uncalibrated noise floors, public-parameter confounds, and profiler-induced observer artifacts. Using HCL Software's Lattice/LWE-based QEQSKE key exchange protocol alongside FIPS 203 ML-KEM-512 (`kyber-py`) across 1,106,944 ANU QRNG draws and two physical architectures (`Linux x86_64 Intel Core i9-12900K` and `macOS arm64 Apple M2`), we present a calibrated five-model ML auditing methodology with leak-free in-fold confound residualization (`QResidualizer`), shuffled-label controls, 200-permutation tests, and graded delay injection. Our evaluation reveals four concrete findings: (1) unmodified pure-Python `decrypt_it()` at toy parameters (`n=4, k=2`) leaks `secret_weight` at **65.25%** (`Linux i9`, `R² = +0.2260`, `p < 1/201`) and **63.65%** (`Apple M2`, `R² = +0.1768`, `p < 1/201`) via CPython's cached `x * 0` fast path (`34.25 → 44.25 µs` across weights `0..8`), and shrinks under live localhost TCP framing noise to **55.67%** (`ρ = +0.0700`); (2) Python's `tracemalloc` acts as a severe **observer effect**, inflating decryption latency `11.7×` (`75.21 → 882.02 µs`) and artificially boosting classifier accuracy from `65.25%` to `84.37%` (`85.21%` on `Apple M2`); (3) replacing pure-Python arithmetic with branch-free vectorized NumPy (`decrypt_it_constant_time`) removes the leak down to baseline (`53.56%`, `p = 0.7065`, all `R² < 0`; `53.92%` over TCP, `p = 0.0995`), while isolating a subtle `~0.26 µs` signed floor-modulo (`% q64`) micro-trend; and (4) scaling to real `n=256` NTT-domain ML-KEM-512 (`standard` and per-key `q` QRNG `QEQSKE` variant) extinguishes the coefficient-sparsity leak completely (`52.12%` / `51.09%`, `p >= 0.1592`, unchanged after rolling-median drift removal), even though a calibrated `n=256` positive control proves our detector reliably catches timing leaks as small as **`2.0 µs` (`< 0.1%` of `2.38 ms` `decaps`, `ρ_bin >= +0.1022`, `p < 1/201`)**.

---

## 1. Introduction & Threat Model
- **Research Question:** Where do exploitable timing, execution, and channel leaks emerge across an end-to-end post-quantum key exchange (Sender `encrypt_it`, Public Channel `(A, t, u, v_queue, q)`, Receiver `decrypt_it`), and how do ML auditing pipelines avoid both false positives (observer artifacts, public-modulus confounds) and false negatives (uncalibrated noise floors)?
- **Threat Model Tiers:**
  1. *QRNG / Channel Eavesdropper (`Tier 1c`, `Attack A/B`):* Passive observer inspecting ANU QRNG outputs and serialized ciphertexts `(u, v_queue, q)`.
  2. *Local Co-Resident Software Timing Observer (`Tier 1a/1b`, `Attack C`):* Unprivileged co-resident attacker measuring `perf_counter_ns` / `ru_utime` across sender encapsulation and receiver decapsulation.
  3. *Network Timing Observer (`Tier 1d/1e`):* Active sender / on-path observer measuring round-trip ACK latency over persistent TCP (`127.0.0.1`, `TCP_NODELAY=1`).
  4. *Excluded Scope:* Invasive fault injection (`Attack D`, noted only as a missing FO-check appendix observation) and hardware EM/power probing.

## 2. Related Work
- ML-based side-channel analysis on lattice KEMs (timing & cache attacks on Kyber / NewHope; KyberSlash constant-time division vulnerabilities).
- Variable-time big-integer and sparse-operand timing leaks in interpreted runtimes (CPython `PyLongObject` small-int cache and allocation hooks).
- Methodological pitfalls in ML side-channel evaluation: out-of-fold vs. in-fold confound residualization, thermal/frequency drift in high-precision timers, and permutation resolution floors (`1/(N+1)`).

## 3. Evaluation Methodology
- **Datasets & Randomness Governance:**
  - 1,106,944 ANU QRNG `uint16` draws (`qrng_combined.txt`); `mlkem512_qeqske` consumes `245,000` draws (`wraps = 0`, zero reuse); toy `n=4` consumes `4,905,000` draws (`wraps = 4`, yielding `5,000/5,000` unique `(q, s)` pairs).
- **Trace Collection Protocol:** Uninstrumented primary pass (`perf_counter` / `perf_counter_ns` + `ru_utime`) isolated from `tracemalloc` observer pass; randomized per-pass trial order (`seed=42..45`), separate Mersenne warm-up pool (`seed=42000`), interleaved GC + warm-up blocks every 250 trials, and CPU core 0 pinning (`os.sched_setaffinity`) where supported (`Linux`).
- **Drift Mitigation at `n=256`:** Compare Raw vs. linear Order-OLS vs. leave-one-out centered Rolling-Median (`W=51`) residuals (`--detrend-existing`).
- **Statistical & ML Pipeline (`5` Model Families):** Logistic/Linear Regression, Random Forest, Gradient Boosting, SVM/SVR (RBF), MLP evaluated over `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)` (classification `secret_weight > median`) and `RepeatedKFold(5x10)` (continuous regression `R²`), validated against majority-class baseline, shuffled-label control, leak-free in-fold `QResidualizer`, 2,000-resample paired bootstrap 95% CIs, and 200-permutation tests (`< 1/201 (0.0050)` resolution floor).
- **Calibrated Positive Control Sensitivity Floor:** Inject secret-dependent busy-wait delays (`0.0–5.0 µs` on `decrypt_it_constant_time`; `0, 1, 2, 5, 10, 20, 50 µs` on real ML-KEM-512 `decaps`) to establish empirically what size leak the setup can detect.

## 4. Empirical Results
1. **Toy QEQSKE (`n=4, k=2`) Uninstrumented Timing Leak & Cross-Host Replication:**
   - Linux `i9-12900K`: `65.25%` (`MLP`, `AUC 0.7076`), `R² = +0.2260` (`SVR`), `ρ = +0.4166` (`p < 1/201`).
   - macOS `Apple M2`: `63.65%` (`Grad Boost`, `AUC 0.6798`), `R² = +0.1768` (`MLP`), `ρ = +0.3418` timing / `+0.3706` CPU (`p < 1/201`).
2. **End-to-End Sender ↔ Receiver & Localhost TCP RTT (`5,000` clean trials, `Apple M2`):**
   - Sender `53.68%` (`p = 0.3682`); Receiver Local `65.29%` (`SVM`, `p < 1/201`); Public Channel `55.59%` (`p < 1/201`, explained by `119/5,000` toy-`n=4` zero-column degenerate keys collapsing ciphertext entropy); TCP Orig attenuated to `55.67%` (`ρ = +0.0700`, `p < 1/201`) vs. TCP Constant-Time `53.92%` (`ρ = -0.0033`, `p = 0.0995`).
3. **The `tracemalloc` Observer Effect (`11.7×` Runtime & `+19%` Accuracy Inflation):**
   - Clean `75.21 µs` (`65.25%`) → `tracemalloc` `882.02 µs` (`84.37%` Linux / `85.21%` Apple M2, `ρ = +0.8241` / `+0.8302`, `R² = +0.6877` / `+0.7247`) while `ram_peak_bytes` / `num_allocations` carry near-zero direct predictive power (`84.37%` 2f vs `84.42%` 4f).
4. **Root-Cause Isolation & Constant-Time Mitigation:**
   - `cProfile` (`mat_mult_mod` = `42.0%` runtime, `82%` of paired `w=8 − w=0` gap) + pinned microbenchmark (`34.25 → 44.25 µs`, `ρ = +0.7306`; `x*0 = 11.01 ns` vs. `x*±1 = 16.15–16.54 ns`).
   - `decrypt_it_constant_time` eliminates leak (`53.56%`, `ρ = -0.0258`, `p = 0.7065`, all `R² < 0`); component isolation uncovers signed NumPy floor-modulo branch (`+0.256 µs`, `ρ = +0.3393`) removed by non-negative offset `+4*q64` (`ρ = -0.0186`).
   - Toy sensitivity floor: `0.5 µs` (`61.62%` Full 8D / `60.12%` Decrypt 4D, `p < 1/201`).
5. **Scaling to Real ML-KEM-512 (`n=256`, `kyber-py`) + QEQSKE Variant:**
   - NTT destroys operand zero-sparsity (`160/512` zeros in `s` → `0.04–0.15/512` in `s_hat`). Both Standard (`52.12%`, `p = 0.1592`) and QEQSKE-Variant (`51.09%`, `p = 0.2438`; despite `q`-timing correlation `r = +0.1493` cleanly controlled by `QResidualizer`) sit at baseline.
   - **Detrending check (`--detrend-existing`):** Rolling-median (`W=51`) eliminates non-linear execution-order drift on Standard ML-KEM (`ρ` `-0.4668 → -0.0108`) and cuts timing SD nearly in half (`295.0 → 155.9 µs`), yet both variants remain strictly at baseline (`52.01%`, `p = 0.3284` and `51.03%`, `p = 0.3731`).
   - **ML-KEM Positive Control (`n=256`, `0–50 µs` sweep):** Pipeline reliably detects injected delays down to **`2.0 µs`** (`54.10%` Raw MLP / `55.40%` Detrended SVM, `ρ_bin >= +0.1022`, `p < 1/201 (0.0050)`), bounding any undetected `kyber-py` `decaps` leak below `< 2 µs` (`< 0.1%` of runtime).

## 5. Limitations
1. **Software Timing Proxies Only:** Evaluates `perf_counter_ns`, `ru_utime`, `tracemalloc`, and loopback TCP RTT — not hardware PMU counters (`perf`), oscilloscope power traces, or EM probes; absence of timing leakage does not imply DPA/CPA power resistance without masking.
2. **Pure-Python Target Implementations (`qeqske_core.py` & `kyber-py`):** Educational Python implementations execute in microseconds/milliseconds with interpreter allocation jitter rather than cycle-accurate compiled C/AVX2 assembly (where constant-time hardware division / compiler optimization bugs like KyberSlash occur).
3. **Toy Parameters (`n=4, k=2`) Are Algebraically Insecure:** Secret matrices at `n=4` are directly brute-forceable from public `(A, t)` in `<1 ms` (`break_poc.py`, `attack_e_*`), and `2.38%` of keys exhibit degenerate all-zero columns (`Tier 1c`); toy results probe timing side-channel mechanics, not LWE hardness.
4. **Fresh-Key Global Secret Weight vs. Fixed-Key Full Recovery:** Primary ML targets binary/continuous global `secret_weight` (plus `s[0][0]` sign at `n=256`) from 1 trace per fresh key rather than multi-query template/CPA recovery of every secret coefficient against a single static key.
5. **Localhost Loopback TCP vs. Real Network Jitter:** Localhost `127.0.0.1` TCP (`IQR 34.5 µs`) already attenuates the local leak (`65.29% → 55.67%`); wide-area internet jitter (milliseconds) would overwhelm sub-10 µs software timing differences.
6. **QRNG Stream Reuse on Toy Runs (`4` Wraps):** While `mlkem512_qeqske` uses zero QRNG reuse (`245,000 / 1,106,944`, `wraps = 0`), the 5,000-trial toy `n=4` runs consume `4,905,000` draws (`wraps = 4`, still producing `5,000/5,000` unique `(q, s)` pairs).
7. **Cross-Architecture Coverage:** Core toy decrypt replication spans two distinct architectures (`Intel Core i9-12900K Linux x86_64` and `Apple M2 macOS arm64`); `n=256` detrending analyzed Linux traces while `n=256` positive-control threshold sensitivity (`>= 2 µs`) was calibrated on Apple M2 (`darwin` lacks `os.sched_setaffinity`).

## 6. Reproducibility & Artifacts
- One-command reproduction runner: `python3 reproduce_all.py --mode fast` (verification + cross-host comparison + detrending) and `python3 reproduce_all.py --mode full` (full regeneration).
- Pinned dependencies in `requirements.txt` and per-host provenance embedded in all `results/*.json`.
