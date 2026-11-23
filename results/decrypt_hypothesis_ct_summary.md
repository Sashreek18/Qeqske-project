# Hypothesis Verification, Profiling & Constant-Time `decrypt_it` Report

- **Environment:** `Linux 7.0.0-31-generic` (`Linux-7.0.0-31-generic-x86_64-with-glibc2.39`), CPU `12th Gen Intel(R) Core(TM) i9-12900K`, Python `3.12.3`, NumPy `2.5.2`

## Part (a): `mat_mult_mod` Weight 0..8 Microbenchmark
- **CPU Pinning:** `True` (Pinned to CPU core 0 via os.sched_setaffinity(0, {0}).)
- **Batching:** `40` calls/batch (`1 HELLO` decrypt worth), `300` randomized reps per weight (`0..8`)
- **Pure-Python `mat_mult_mod` vs weight:** Spearman `ρ = +0.7306` (`[+0.7048, +0.7561]`, `p = 0.00e+00`), Pearson `r = +0.5079` (`p = 4.83e-177`)
- **NumPy `int64` `matmul` vs weight:** Spearman `ρ = +0.4354` (`[+0.4023, +0.4704]`, `p = 2.38e-125`), Pearson `r = +0.0775`

| Weight | `mat_mult_mod` Batch Med (µs) | Batch IQR (µs) | Per-Call Med (µs) | NumPy `int64` Batch Med (µs) | NumPy IQR (µs) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 34.25 | 2.20 | 0.856 | 2.79 | 0.08 |
| 1 | 36.87 | 2.51 | 0.922 | 2.87 | 0.21 |
| 2 | 38.93 | 2.54 | 0.973 | 2.98 | 0.24 |
| 3 | 40.37 | 2.40 | 1.009 | 3.03 | 0.28 |
| 4 | 41.39 | 2.32 | 1.035 | 3.01 | 0.25 |
| 5 | 42.21 | 2.37 | 1.055 | 3.03 | 0.28 |
| 6 | 42.69 | 2.64 | 1.067 | 3.08 | 0.27 |
| 7 | 43.66 | 2.65 | 1.091 | 3.09 | 0.30 |
| 8 | 44.25 | 2.97 | 1.106 | 3.13 | 0.24 |

### Explaining the Residual NumPy Weight Trend
- **Root cause:** np.matmul(u_arr, s_np) and (v_arr - u_arr @ s_np) alone show zero positive weight trend. Almost the entire ~0.3 us residual trend comes from signed floor-modulo (`diff % q64`): at weight=0, diff = v_arr >= 0 (0% negative), whereas at weight=8 ~34% of diff entries are negative; NumPy int64 remainder executes a sign-fixup branch when the dividend is negative, causing branch mispredictions as negative elements increase. Adding a non-negative offset (+n*q64) before `% q64` eliminates the modulo trend; the remaining ~0.03 us difference in full expressions comes from np.asarray(s) unpacking Python int 0 vs +/-1. Across full 40 us decrypt_it_constant_time calls, this sub-microsecond micro-trend is buried beneath wrapper/OS jitter (5,000-trial Spearman rho=-0.0258, p=0.068).
- **Negative-dividend fraction `P(v - u@s < 0)` by weight:** `w=0: 0.0%`, `w=1: 8.8%`, `w=2: 23.0%`, `w=4: 18.5%`, `w=8: 34.2%`.

| Isolated Component | `w=0` Med (µs) | `w=8` Med (µs) | Δ (`w8 − w0`) | Spearman ρ vs. Weight | Spearman p |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `np.matmul(u_arr, s_np)` | 1.215 | 1.177 | -0.038 µs | -0.1235 | 1.22e-10 |
| `v_arr - np.matmul(u_arr, s_np)` | 1.502 | 1.452 | -0.050 µs | -0.1491 | 6.84e-15 |
| `diff % q64 (can be negative)` | 0.916 | 1.172 | +0.256 µs | +0.3393 | 9.56e-74 |
| `(diff + 4*q64) % q64 (all >= 0)` | 1.329 | 1.313 | -0.015 µs | -0.0186 | 3.35e-01 |
| `np.asarray(s_py, dtype=np.int64)` | 0.743 | 0.771 | +0.028 µs | +0.2736 | 1.51e-47 |
| `(v - u@asarray(s) + 4*q64) % q64` | 3.271 | 3.316 | +0.045 µs | +0.1622 | 2.28e-17 |
| `(v - u@s_np + 4*q64) % q64` | 2.651 | 2.620 | -0.030 µs | -0.1449 | 3.83e-14 |

### Operand-Value Isolation Controls
- **`mat_mult_mod` (40 calls):** `all_0` median = `31.90 µs` (IQR `1.78`), `all_+1` median = `40.32 µs` (IQR `1.63`), `all_-1` median = `40.88 µs` (IQR `1.81`)
- **Deltas (`all_+1 - all_0`, `all_-1 - all_0`):** `+8.42 µs` (`p = 4.53e-120`), `+8.98 µs` (`p = 1.43e-122`)
- **CPython micro-ops (ns/op):** `x*0 = 11.01 ns`, `x*1 = 16.15 ns`, `x*(-1) = 16.54 ns`, `acc+0 = 15.81 ns`, `acc+x = 16.13 ns`

## Part (b): `cProfile` & `dis` Findings
- **Realistic `cProfile` (2,000 `decrypt_it` calls, total `0.3090 s`):**
  - `qeqske_core.py:137(mat_mult_mod)`: `80000` calls, `tottime = 0.1298 s` (`42.0%`)
  - `qeqske_core.py:273(decrypt_it)`: `2000` calls, `tottime = 0.0605 s` (`19.6%`)
  - `qeqske_core.py:157(mat_sub_mod)`: `80000` calls, `tottime = 0.0481 s` (`15.6%`)
  - `~:0(<built-in method builtins.len>)`: `576000` calls, `tottime = 0.0212 s` (`6.9%`)
  - `~:0(<built-in method builtins.min>)`: `80000` calls, `tottime = 0.0103 s` (`3.3%`)
  - `qeqske_core.py:221(<genexpr>)`: `90000` calls, `tottime = 0.0098 s` (`3.2%`)
- **Paired `w=0` vs `w=8` (identical `(u, v_queue, q)`):** `w=0` total `0.2897 s` vs `w=8` total `0.3119 s` (`Δ = +0.0223 s`)

## Part (c): Constant-Time `decrypt_it_constant_time` Before vs. After (5,000 Trials)
- **Correctness checks:** Legacy single-`u` `50/50`, boundary `39/39`, 5,000 QRNG trials `5000/5000`

| Metric (Clean Uninstrumented, Majority Baseline = 53.70%) | Original `decrypt_it` | Constant-Time `decrypt_it_constant_time` |
| :--- | :---: | :---: |
| `dec_timing_us` Pearson r [Boot CI] | +0.1727 [+0.1437, +0.2034] | -0.0238 [-0.0527, +0.0039] |
| `dec_timing_us` Spearman ρ [Boot CI] | +0.4166 [+0.3916, +0.4404] | -0.0258 [-0.0541, +0.0010] |
| `dec_cpu_user_us` Pearson r | +0.1579 | -0.0236 |
| `dec_cpu_user_us` Spearman ρ | +0.4023 | -0.0322 |
| Best Classifier Accuracy (Model) | 65.25% (MLP) | 53.56% (Logistic Regression) |
| Best Classifier Perm p-value | < 1/201 (0.0050) | 0.7065 |
| Best Regressor R² (Model) | +0.2260 (SVR (RBF)) | All models R² < 0 (best -0.0020, Linear Regression; worse than mean baseline) |
| Best Regressor Perm p-value | < 1/201 (0.0050) | 0.7164 |
| `tracemalloc` 2-feat Timing Spearman ρ | +0.8241 | +0.0103 |
| `tracemalloc` 2-feat Best Clf Acc | 84.37% (p < 1/201 (0.0050)) | 53.93% (p 0.0498) |

> *Notes:* Permutation p-values at the `N=200` floor are reported as `< 1/201 (0.0050)`. On `decrypt_it_constant_time`, all 5 regression models yield `R² < 0` (worse than predicting the sample mean), confirming zero regression skill; on original `decrypt_it`, unpruned Random Forest (`R² = -0.0776`) is similarly caveated while SVR/MLP/GB reach `R² = +0.2206` to `+0.2260`.
