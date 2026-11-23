# `decrypt_it (original)` Side-Channel Leakage Evaluation (5,000 Trials)

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
| `dec_timing_us` | 75.21 ± 6.37 | 73.99 (3.23) | +0.1727 [+0.1437, +0.2034] | 9.09e-35 | +0.4166 [+0.3916, +0.4404] | 3.36e-209 |
| `dec_cpu_user_us` | 75.52 ± 7.35 | 74.00 (3.00) | +0.1579 [+0.1286, +0.1862] | 2.87e-29 | +0.4023 [+0.3778, +0.4261] | 6.99e-194 |

### Confound & Time-Drift Checks (Clean Pass)
- **Modulus `q` vs `secret_weight`:** Pearson `r = -0.0031` (`p = 0.8284`), Spearman `ρ = -0.0042` (`p = 0.7692`)
- **Modulus `q` vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = -0.1017` (`p = 0.0000`) / `r = -0.0790` (`p = 0.0000`)
- **Execution order vs `secret_weight`:** Pearson `r = -0.0082` (`p = 0.5619`)
- **Execution order vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = -0.0101` (`p = 0.4761`) / `r = -0.0220` (`p = 0.1205`)

### Binary Classification (5×10 Repeated Stratified CV, Majority Baseline = 53.70%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized Acc | Perm p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 56.81% ± 1.18% | 0.6103 | 51.97% ± 1.25% | +4.84% | 60.33% ± 1.37% | < 1/201 (0.0050) |
| Gradient Boosting | 64.76% ± 1.63% | 0.6957 | 53.07% ± 1.39% | +11.69% | 64.53% ± 1.35% | < 1/201 (0.0050) |
| MLP | 65.25% ± 1.49% | 0.7076 | 53.46% ± 0.38% | +11.79% | 64.79% ± 1.25% | < 1/201 (0.0050) |
| Logistic Regression | 56.00% ± 0.69% | 0.6980 | 53.64% ± 0.16% | +2.37% | 56.22% ± 0.75% | < 1/201 (0.0050) |
| SVM (RBF) | 65.21% ± 1.37% | 0.7047 | 53.34% ± 0.39% | +11.87% | 65.22% ± 1.21% | < 1/201 (0.0050) |

> *Permutation p-value note:* With `N = 200` permutations, zero shuffled scores exceeding the observed score hits the Phipson–Smyth resolution floor `1/(N+1)` and is reported as `< 1/201 (0.0050)`.

### Regression Target: Exact `secret_weight` (5×10 Repeated K-Fold CV)
| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | q-Residualized R² | Perm p-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random Forest | -0.0776 ± 0.0404 | 1.3810 | -0.3859 ± 0.0467 | +0.0879 ± 0.0358 | < 1/201 (0.0050) | Negative R² (worse than mean baseline) |
| Gradient Boosting | +0.2206 ± 0.0240 | 1.1747 | -0.0188 ± 0.0104 | +0.2190 ± 0.0242 | < 1/201 (0.0050) | — |
| MLP | +0.2246 ± 0.0307 | 1.1716 | -0.0160 ± 0.0733 | +0.2198 ± 0.0295 | < 1/201 (0.0050) | — |
| Linear Regression | +0.0251 ± 0.0217 | 1.3141 | -0.0033 ± 0.0068 | +0.0252 ± 0.0223 | < 1/201 (0.0050) | — |
| SVR (RBF) | +0.2260 ± 0.0256 | 1.1706 | -0.0338 ± 0.0121 | +0.2217 ± 0.0249 | < 1/201 (0.0050) | — |

> *Regression caveat:* Negative-`R²` models (Random Forest (-0.0776)) perform worse than predicting the constant sample mean (`R² = 0`, `RMSE = 1.3320`): unpruned trees overfit two noisy timing features (`shuffled R² = -0.3859`), so their permutation `p` reflects beating an even worse shuffled fit rather than useful continuous prediction. Report the positive-`R²` models instead (best: SVR (RBF) `R² = +0.2260`).

## 3. Legacy Instrumentation Comparison (`tracemalloc`-Wrapped Pass)

### Correlations vs. `secret_weight` under `tracemalloc`
| Signal | Mean ± Std | Pearson r (95% Boot CI) | Spearman ρ (95% Boot CI) |
| :--- | :---: | :---: | :---: |
| `tm_dec_timing_us` | 882.02 ± 92.90 | +0.6398 [+0.5939, +0.6868] | +0.8241 [+0.8113, +0.8352] |
| `tm_dec_cpu_user_us` | 883.35 ± 95.51 | +0.6240 [+0.5789, +0.6728] | +0.8223 [+0.8097, +0.8336] |
| `dec_ram_peak_bytes` | 1531.71 ± 32.14 | +0.0672 [+0.0388, +0.0934] | +0.0681 [+0.0392, +0.0954] |
| `dec_num_allocations` | 4.00 ± 0.01 | -0.0178 [-0.0314, +0.0000] | -0.0193 [-0.0339, +0.0000] |

### ML Summary under `tracemalloc`
- **2-feature (`tm_dec_timing_us`, `tm_dec_cpu_user_us`) Best Classifier (Gradient Boosting):** Real Acc `84.37%`, Shuffled `52.62%`, q-Residualized `84.34%`, Perm `p < 1/201 (0.0050)`
- **2-feature Best Regressor (MLP):** Real R² `+0.6877`, Shuffled R² `-0.0044`, q-Residualized R² `+0.6880`, Perm `p < 1/201 (0.0050)`
- **4-feature (+ `dec_ram_peak_bytes`, `dec_num_allocations`) Best Classifier (Gradient Boosting):** Real Acc `84.42%`, Shuffled `52.72%`, q-Residualized `84.46%`, Perm `p < 1/201 (0.0050)`
- **4-feature Best Regressor (Gradient Boosting):** Real R² `+0.7051`, Shuffled R² `-0.0173`, q-Residualized R² `+0.7265`, Perm `p < 1/201 (0.0050)`
