# `decrypt_it_constant_time (NumPy int64 branch-free)` Side-Channel Leakage Evaluation (5,000 Trials)

## 1. Experiment Setup & Data Integrity
- **Environment:** `Linux 7.0.0-31-generic` (`Linux-7.0.0-31-generic-x86_64-with-glibc2.39`), CPU `12th Gen Intel(R) Core(TM) i9-12900K`, Python `3.12.3`, NumPy `2.5.2`
- **Trials:** 5000 (fixed plaintext `HELLO`, parameters `n=4`, `k=2`, fresh key per trial, seed `42`)
- **Decryption correctness:** Clean pass `4997/5000` (99.94%); `tracemalloc` pass `4997/5000` (99.94%)
- **QRNG stream:** `qrng_combined.txt` (`1,106,944` numbers, `cycle=True`, consumed `4,905,000` draws across `4` wraps; `5,000` unique `(q, s)` pairs)
- **Binary target:** `secret_weight > 5.0` -> class 0: `2685`, class 1: `2315`, majority-class baseline: **53.70%**
- **Regression target:** exact `secret_weight` (range `0..8`, mean `5.3266`, std `1.3320`)

## 2. Primary Analysis: Clean Uninstrumented Traces (`dec_timing_us`, `dec_cpu_user_us`)

### Correlations vs. `secret_weight` (2,000-Resample Paired Bootstrap 95% CIs)
| Signal | Mean ± Std (µs) | Median (IQR) µs | Pearson r (95% Boot CI) | Pearson p | Spearman ρ (95% Boot CI) | Spearman p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dec_timing_us` | 40.01 ± 2.56 | 39.56 (0.95) | -0.0238 [-0.0527, +0.0039] | 9.18e-02 | -0.0258 [-0.0541, +0.0010] | 6.77e-02 |
| `dec_cpu_user_us` | 40.06 ± 5.07 | 40.00 (1.00) | -0.0236 [-0.0518, +0.0057] | 9.53e-02 | -0.0322 [-0.0594, -0.0051] | 2.30e-02 |

### Confound & Time-Drift Checks (Clean Pass)
- **Modulus `q` vs `secret_weight`:** Pearson `r = -0.0031` (`p = 0.8284`), Spearman `ρ = -0.0042` (`p = 0.7692`)
- **Modulus `q` vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = -0.0518` (`p = 0.0002`) / `r = -0.0228` (`p = 0.1071`)
- **Execution order vs `secret_weight`:** Pearson `r = -0.0082` (`p = 0.5619`)
- **Execution order vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = -0.0002` (`p = 0.9910`) / `r = +0.0477` (`p = 0.0007`)

### Binary Classification (5×10 Repeated Stratified CV, Majority Baseline = 53.70%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized Acc | Perm p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 51.60% ± 1.69% | 0.5161 | 49.13% ± 1.47% | +2.47% | 51.12% ± 1.44% | 0.1841 |
| Gradient Boosting | 52.75% ± 1.08% | 0.4997 | 51.83% ± 0.87% | +0.92% | 52.48% ± 1.09% | 0.4229 |
| MLP | 53.53% ± 0.26% | 0.4948 | 53.52% ± 0.27% | +0.01% | 53.53% ± 0.25% | 0.7960 |
| Logistic Regression | 53.56% ± 0.20% | 0.5056 | 53.66% ± 0.13% | -0.10% | 53.56% ± 0.21% | 0.7065 |
| SVM (RBF) | 53.38% ± 0.27% | 0.4999 | 53.43% ± 0.34% | -0.05% | 53.38% ± 0.25% | 0.9751 |

> *Permutation p-value note:* With `N = 200` permutations, zero shuffled scores exceeding the observed score hits the Phipson–Smyth resolution floor `1/(N+1)` and is reported as `< 1/201 (0.0050)`.

### Regression Target: Exact `secret_weight` (5×10 Repeated K-Fold CV)
| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | q-Residualized R² | Perm p-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random Forest | -0.3749 ± 0.0523 | 1.5599 | -0.3985 ± 0.0401 | -0.1869 ± 0.0308 | 0.7164 | Negative R² (worse than mean baseline) |
| Gradient Boosting | -0.0187 ± 0.0086 | 1.3433 | -0.0220 ± 0.0078 | -0.0228 ± 0.0088 | 0.8010 | Negative R² (worse than mean baseline) |
| MLP | -0.0049 ± 0.0048 | 1.3341 | -0.0046 ± 0.0032 | -0.0045 ± 0.0043 | 0.6667 | Negative R² (worse than mean baseline) |
| Linear Regression | -0.0020 ± 0.0030 | 1.3322 | -0.0024 ± 0.0018 | -0.0019 ± 0.0030 | 0.7164 | Negative R² (worse than mean baseline) |
| SVR (RBF) | -0.0344 ± 0.0116 | 1.3535 | -0.0343 ± 0.0126 | -0.0342 ± 0.0116 | 0.8507 | Negative R² (worse than mean baseline) |

> *Regression caveat:* All 5 models yield negative out-of-fold `R²` (RMSE ≥ sample SD `1.3320`), performing worse than predicting the constant sample mean (`R² = 0`). There is **no regression signal**.

## 3. Legacy Instrumentation Comparison (`tracemalloc`-Wrapped Pass)

### Correlations vs. `secret_weight` under `tracemalloc`
| Signal | Mean ± Std | Pearson r (95% Boot CI) | Spearman ρ (95% Boot CI) |
| :--- | :---: | :---: | :---: |
| `tm_dec_timing_us` | 191.60 ± 5.68 | +0.0032 [-0.0246, +0.0285] | +0.0103 [-0.0174, +0.0369] |
| `tm_dec_cpu_user_us` | 193.22 ± 9.68 | -0.0017 [-0.0323, +0.0270] | +0.0032 [-0.0242, +0.0301] |
| `dec_ram_peak_bytes` | 10008.00 ± 0.34 | -0.0178 [-0.0314, +0.0000] | -0.0193 [-0.0339, +0.0000] |
| `dec_num_allocations` | 5.00 ± 0.01 | -0.0178 [-0.0314, +0.0000] | -0.0193 [-0.0339, +0.0000] |

### ML Summary under `tracemalloc`
- **2-feature (`tm_dec_timing_us`, `tm_dec_cpu_user_us`) Best Classifier (SVM (RBF)):** Real Acc `53.93%`, Shuffled `53.78%`, q-Residualized `53.93%`, Perm `p 0.0498`
- **2-feature Best Regressor (Linear Regression):** Real R² `-0.0026`, Shuffled R² `-0.0018`, q-Residualized R² `-0.0026`, Perm `p 0.2388` *(all models R² < 0; no predictive skill)*
- **4-feature (+ `dec_ram_peak_bytes`, `dec_num_allocations`) Best Classifier (SVM (RBF)):** Real Acc `53.94%`, Shuffled `53.79%`, q-Residualized `53.94%`, Perm `p 0.0448`
- **4-feature Best Regressor (Linear Regression):** Real R² `-0.0026`, Shuffled R² `-0.0018`, q-Residualized R² `-0.0026`, Perm `p 0.2388` *(all models R² < 0; no predictive skill)*
