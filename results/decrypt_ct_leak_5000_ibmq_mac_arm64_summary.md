# `decrypt_it_constant_time (NumPy CT, IBMQ QRNG, mac_arm64)` Side-Channel Leakage Evaluation (5,000 Trials)

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
| `dec_timing_us` | 49.01 ± 7.69 | 47.71 (2.62) | +0.0104 [-0.0129, +0.0334] | 4.62e-01 | +0.0205 [-0.0067, +0.0488] | 1.47e-01 |
| `dec_cpu_user_us` | 48.49 ± 4.49 | 47.00 (2.00) | +0.0076 [-0.0160, +0.0312] | 5.91e-01 | +0.0056 [-0.0217, +0.0328] | 6.93e-01 |

### Confound & Time-Drift Checks (Clean Pass)
- **Modulus `q` vs `secret_weight`:** Pearson `r = +0.0112` (`p = 0.4265`), Spearman `ρ = +0.0117` (`p = 0.4088`)
- **Modulus `q` vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = +0.0215` (`p = 0.1280`) / `r = +0.0148` (`p = 0.2946`)
- **Execution order vs `secret_weight`:** Pearson `r = -0.0037` (`p = 0.7930`)
- **Execution order vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = -0.0994` (`p = 0.0000`) / `r = -0.0714` (`p = 0.0000`)

### Binary Classification (5×10 Repeated Stratified CV, Majority Baseline = 52.52%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized Acc | Perm p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 50.69% ± 1.15% | 0.5031 | 51.17% ± 1.43% | -0.48% | 50.57% ± 1.21% | 0.3184 |
| Gradient Boosting | 51.19% ± 1.23% | 0.4921 | 51.63% ± 1.15% | -0.44% | 51.32% ± 1.18% | 0.4030 |
| MLP | 52.45% ± 0.35% | 0.4948 | 52.25% ± 0.63% | +0.20% | 52.47% ± 0.32% | 0.3881 |
| Logistic Regression | 52.54% ± 0.18% | 0.5056 | 52.48% ± 0.13% | +0.06% | 52.55% ± 0.18% | 0.0647 |
| SVM (RBF) | 52.44% ± 0.32% | 0.5011 | 52.29% ± 0.46% | +0.16% | 52.45% ± 0.31% | 0.8159 |

> *Permutation p-value note:* With `N = 200` permutations, zero shuffled scores exceeding the observed score hits the Phipson–Smyth resolution floor `1/(N+1)` and is reported as `< 1/201 (0.0050)`.

### Regression Target: Exact `secret_weight` (5×10 Repeated K-Fold CV)
| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | q-Residualized R² | Perm p-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random Forest | -0.2231 ± 0.0298 | 1.4758 | -0.2354 ± 0.0355 | -0.1981 ± 0.0357 | 0.6866 | Negative R² (worse than mean baseline) |
| Gradient Boosting | -0.0196 ± 0.0080 | 1.3476 | -0.0250 ± 0.0098 | -0.0209 ± 0.0081 | 0.6318 | Negative R² (worse than mean baseline) |
| MLP | -0.0128 ± 0.0304 | 1.3430 | -0.0130 ± 0.0217 | -0.0125 ± 0.0290 | 0.5572 | Negative R² (worse than mean baseline) |
| Linear Regression | -0.0018 ± 0.0019 | 1.3358 | -0.0023 ± 0.0027 | -0.0018 ± 0.0018 | 0.3980 | Negative R² (worse than mean baseline) |
| SVR (RBF) | -0.0357 ± 0.0124 | 1.3582 | -0.0392 ± 0.0109 | -0.0356 ± 0.0124 | 0.1045 | Negative R² (worse than mean baseline) |

> *Regression caveat:* All 5 models yield negative out-of-fold `R²` (RMSE ≥ sample SD `1.3355`), performing worse than predicting the constant sample mean (`R² = 0`). There is **no regression signal**.

## 3. Legacy Instrumentation Comparison (`tracemalloc`-Wrapped Pass)

### Correlations vs. `secret_weight` under `tracemalloc`
| Signal | Mean ± Std | Pearson r (95% Boot CI) | Spearman ρ (95% Boot CI) |
| :--- | :---: | :---: | :---: |
| `tm_dec_timing_us` | 186.43 ± 11.79 | +0.0131 [-0.0130, +0.0386] | +0.0193 [-0.0082, +0.0476] |
| `tm_dec_cpu_user_us` | 187.64 ± 7.71 | +0.0071 [-0.0200, +0.0343] | +0.0204 [-0.0077, +0.0495] |
| `dec_ram_peak_bytes` | 10008.00 ± 0.00 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| `dec_num_allocations` | 5.00 ± 0.00 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

### ML Summary under `tracemalloc`
- **2-feature (`tm_dec_timing_us`, `tm_dec_cpu_user_us`) Best Classifier (Logistic Regression):** Real Acc `52.57%`, Shuffled `52.40%`, q-Residualized `52.57%`, Perm `p 0.0647`
- **2-feature Best Regressor (Linear Regression):** Real R² `-0.0018`, Shuffled R² `-0.0016`, q-Residualized R² `-0.0018`, Perm `p 0.2388` *(all models R² < 0; no predictive skill)*
- **4-feature (+ `dec_ram_peak_bytes`, `dec_num_allocations`) Best Classifier (Logistic Regression):** Real Acc `52.57%`, Shuffled `52.40%`, q-Residualized `52.57%`, Perm `p 0.0647`
- **4-feature Best Regressor (Linear Regression):** Real R² `-0.0018`, Shuffled R² `-0.0016`, q-Residualized R² `-0.0018`, Perm `p 0.2388` *(all models R² < 0; no predictive skill)*
