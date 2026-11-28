# `decrypt_it (original, IBMQ QRNG, mac_arm64)` Side-Channel Leakage Evaluation (5,000 Trials)

## 1. Experiment Setup & Data Integrity
- **Environment:** `Darwin 27.0.0` (`macOS-27.0.1-arm64-arm-64bit-Mach-O`), CPU `Apple M2`, Python `3.13.1`, NumPy `2.2.6`
- **Trials:** 5000 (fixed plaintext `HELLO`, parameters `n=4`, `k=2`, fresh key per trial, seed `42`)
- **Decryption correctness:** Clean pass `5000/5000` (100.00%); `tracemalloc` pass `5000/5000` (100.00%)
- **QRNG stream:** `ibmq_qrng_uint16.txt` (`589,193` numbers, `cycle=True`, consumed `4,905,000` draws across `8` wraps; `4,998` unique `(q, s)` pairs)
- **Binary target:** `secret_weight > 5.0` -> class 0: `2626`, class 1: `2374`, majority-class baseline: **52.52%**
- **Regression target:** exact `secret_weight` (range `0..8`, mean `5.3534`, std `1.3355`)

## 2. Primary Analysis: Clean Uninstrumented Traces (`dec_timing_us`, `dec_cpu_user_us`)

### Correlations vs. `secret_weight` (2,000-Resample Paired Bootstrap 95% CIs)
| Signal | Mean ± Std (µs) | Median (IQR) µs | Pearson r (95% Boot CI) | Pearson p | Spearman ρ (95% Boot CI) | Spearman p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dec_timing_us` | 76.22 ± 4.09 | 75.33 (3.00) | +0.2289 [+0.1996, +0.2610] | 1.98e-60 | +0.3939 [+0.3703, +0.4183] | 3.12e-185 |
| `dec_cpu_user_us` | 76.47 ± 3.89 | 76.00 (3.00) | +0.2358 [+0.2069, +0.2653] | 3.69e-64 | +0.3793 [+0.3552, +0.4034] | 7.78e-171 |

### Confound & Time-Drift Checks (Clean Pass)
- **Modulus `q` vs `secret_weight`:** Pearson `r = +0.0112` (`p = 0.4265`), Spearman `ρ = +0.0117` (`p = 0.4088`)
- **Modulus `q` vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = -0.0972` (`p = 0.0000`) / `r = -0.1014` (`p = 0.0000`)
- **Execution order vs `secret_weight`:** Pearson `r = -0.0037` (`p = 0.7930`)
- **Execution order vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = +0.0294` (`p = 0.0379`) / `r = +0.0205` (`p = 0.1465`)

### Binary Classification (5×10 Repeated Stratified CV, Majority Baseline = 52.52%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized Acc | Perm p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 60.80% ± 1.25% | 0.6395 | 49.47% ± 1.24% | +11.34% | 61.03% ± 1.22% | < 1/201 (0.0050) |
| Gradient Boosting | 65.28% ± 1.26% | 0.7032 | 51.33% ± 1.27% | +13.95% | 64.32% ± 1.45% | < 1/201 (0.0050) |
| MLP | 65.11% ± 1.33% | 0.7064 | 52.28% ± 0.60% | +12.83% | 64.55% ± 1.26% | < 1/201 (0.0050) |
| Logistic Regression | 56.99% ± 1.17% | 0.6822 | 52.48% ± 0.11% | +4.51% | 57.77% ± 1.15% | < 1/201 (0.0050) |
| SVM (RBF) | 65.03% ± 1.39% | 0.6982 | 52.29% ± 0.67% | +12.75% | 64.46% ± 1.27% | < 1/201 (0.0050) |

> *Permutation p-value note:* With `N = 200` permutations, zero shuffled scores exceeding the observed score hits the Phipson–Smyth resolution floor `1/(N+1)` and is reported as `< 1/201 (0.0050)`.

### Regression Target: Exact `secret_weight` (5×10 Repeated K-Fold CV)
| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | q-Residualized R² | Perm p-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random Forest | +0.0569 ± 0.0300 | 1.2957 | -0.2438 ± 0.0312 | +0.1089 ± 0.0340 | < 1/201 (0.0050) | — |
| Gradient Boosting | +0.2300 ± 0.0243 | 1.1709 | -0.0202 ± 0.0084 | +0.2387 ± 0.0253 | < 1/201 (0.0050) | — |
| MLP | +0.2382 ± 0.0252 | 1.1645 | -0.0042 ± 0.0041 | +0.2468 ± 0.0260 | < 1/201 (0.0050) | — |
| Linear Regression | +0.0517 ± 0.0166 | 1.2995 | -0.0019 ± 0.0017 | +0.0527 ± 0.0174 | < 1/201 (0.0050) | — |
| SVR (RBF) | +0.2319 ± 0.0263 | 1.1693 | -0.0369 ± 0.0103 | +0.2416 ± 0.0272 | < 1/201 (0.0050) | — |

## 3. Legacy Instrumentation Comparison (`tracemalloc`-Wrapped Pass)

### Correlations vs. `secret_weight` under `tracemalloc`
| Signal | Mean ± Std | Pearson r (95% Boot CI) | Spearman ρ (95% Boot CI) |
| :--- | :---: | :---: | :---: |
| `tm_dec_timing_us` | 903.87 ± 423.49 | +0.1558 [+0.0951, +0.6809] | +0.7970 [+0.7834, +0.8099] |
| `tm_dec_cpu_user_us` | 897.75 ± 94.01 | +0.6678 [+0.6101, +0.7306] | +0.8026 [+0.7895, +0.8152] |
| `dec_ram_peak_bytes` | 1532.85 ± 32.08 | +0.0054 [-0.0234, +0.0337] | +0.0027 [-0.0254, +0.0314] |
| `dec_num_allocations` | 4.00 ± 0.01 | -0.0174 [-0.0306, +0.0000] | -0.0192 [-0.0336, +0.0000] |

### ML Summary under `tracemalloc`
- **2-feature (`tm_dec_timing_us`, `tm_dec_cpu_user_us`) Best Classifier (Gradient Boosting):** Real Acc `82.70%`, Shuffled `52.22%`, q-Residualized `82.90%`, Perm `p < 1/201 (0.0050)`
- **2-feature Best Regressor (Gradient Boosting):** Real R² `+0.6670`, Shuffled R² `-0.0187`, q-Residualized R² `+0.6694`, Perm `p < 1/201 (0.0050)`
- **4-feature (+ `dec_ram_peak_bytes`, `dec_num_allocations`) Best Classifier (Logistic Regression):** Real Acc `82.88%`, Shuffled `52.39%`, q-Residualized `83.19%`, Perm `p < 1/201 (0.0050)`
- **4-feature Best Regressor (Gradient Boosting):** Real R² `+0.6811`, Shuffled R² `-0.0188`, q-Residualized R² `+0.7002`, Perm `p < 1/201 (0.0050)`
