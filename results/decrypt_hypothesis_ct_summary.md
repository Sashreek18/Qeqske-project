# Hypothesis Verification, Profiling & Constant-Time `decrypt_it` Report

## Part (a): `mat_mult_mod` Weight 0..8 Microbenchmark
- **CPU Pinning:** `True` (Pinned to CPU core 0 via os.sched_setaffinity(0, {0}).)
- **Batching:** `40` calls/batch (`1 HELLO` decrypt worth), `300` randomized reps per weight (`0..8`)
- **Pure-Python `mat_mult_mod` vs weight:** Spearman `ρ = +0.7668` (`[+0.7412, +0.7914]`, `p = 0.00e+00`), Pearson `r = +0.2382` (`p = 3.84e-36`)
- **NumPy `int64` `matmul` vs weight:** Spearman `ρ = +0.4307` (`[+0.3956, +0.4666]`, `p = 2.11e-122`), Pearson `r = +0.0733`

| Weight | `mat_mult_mod` Batch Med (µs) | Batch IQR (µs) | Per-Call Med (µs) | NumPy `int64` Batch Med (µs) | NumPy IQR (µs) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 33.86 | 1.57 | 0.847 | 2.78 | 0.09 |
| 1 | 36.47 | 1.81 | 0.912 | 2.85 | 0.23 |
| 2 | 38.07 | 2.04 | 0.952 | 2.99 | 0.27 |
| 3 | 39.57 | 1.95 | 0.989 | 3.02 | 0.27 |
| 4 | 40.53 | 1.88 | 1.013 | 3.02 | 0.25 |
| 5 | 41.17 | 1.97 | 1.029 | 3.04 | 0.26 |
| 6 | 41.88 | 2.21 | 1.047 | 3.09 | 0.27 |
| 7 | 42.73 | 2.10 | 1.068 | 3.08 | 0.30 |
| 8 | 43.35 | 2.79 | 1.084 | 3.12 | 0.27 |

### Operand-Value Isolation Controls
- **`mat_mult_mod` (40 calls):** `all_0` median = `33.04 µs` (IQR `1.60`), `all_+1` median = `40.98 µs` (IQR `1.98`), `all_-1` median = `41.13 µs` (IQR `1.63`)
- **Deltas (`all_+1 - all_0`, `all_-1 - all_0`):** `+7.94 µs` (`p = 3.75e-143`), `+8.09 µs` (`p = 2.06e-143`)
- **CPython micro-ops (ns/op):** `x*0 = 11.37 ns`, `x*1 = 16.53 ns`, `x*(-1) = 17.09 ns`, `acc+0 = 16.36 ns`, `acc+x = 16.54 ns`

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
| Best Classifier Perm p-value | 0.0050 | 0.7065 |
| Best Regressor R² (Model) | +0.2260 (SVR (RBF)) | -0.0020 (Linear Regression) |
| Best Regressor Perm p-value | 0.0050 | 0.7164 |
| `tracemalloc` 2-feat Timing Spearman ρ | +0.8241 | +0.0103 |
| `tracemalloc` 2-feat Best Clf Acc | 84.37% (p=0.0050) | 53.93% (p=0.0498) |
