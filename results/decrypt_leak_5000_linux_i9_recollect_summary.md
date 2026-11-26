# `decrypt_it (original, linux_i9_recollect)` Side-Channel Leakage Evaluation (5,000 Trials)

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
| `dec_timing_us` | 73.37 ± 5.47 | 72.54 (3.18) | +0.1939 [+0.1537, +0.2422] | 1.49e-43 | +0.4237 [+0.3993, +0.4478] | 4.59e-217 |
| `dec_cpu_user_us` | 73.29 ± 8.44 | 73.00 (3.00) | +0.1397 [+0.1126, +0.1679] | 3.37e-23 | +0.4060 [+0.3810, +0.4301] | 8.57e-198 |

### Confound & Time-Drift Checks (Clean Pass)
- **Modulus `q` vs `secret_weight`:** Pearson `r = -0.0031` (`p = 0.8284`), Spearman `ρ = -0.0042` (`p = 0.7692`)
- **Modulus `q` vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = -0.0673` (`p = 0.0000`) / `r = -0.0444` (`p = 0.0017`)
- **Execution order vs `secret_weight`:** Pearson `r = -0.0082` (`p = 0.5619`)
- **Execution order vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = +0.0398` (`p = 0.0048`) / `r = +0.0339` (`p = 0.0164`)

### Binary Classification (5×10 Repeated Stratified CV, Majority Baseline = 53.70%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized Acc | Perm p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 57.19% ± 1.05% | 0.6177 | 51.40% ± 1.63% | +5.78% | 61.02% ± 1.34% | < 1/201 (0.0050) |
| Gradient Boosting | 65.40% ± 1.53% | 0.7032 | 52.97% ± 1.08% | +12.44% | 65.87% ± 1.52% | < 1/201 (0.0050) |
| MLP | 65.60% ± 1.55% | 0.7124 | 53.90% ± 0.32% | +11.71% | 66.08% ± 1.64% | < 1/201 (0.0050) |
| Logistic Regression | 60.65% ± 0.99% | 0.7046 | 53.86% ± 0.33% | +6.79% | 60.74% ± 1.15% | < 1/201 (0.0050) |
| SVM (RBF) | 65.78% ± 1.51% | 0.7096 | 53.86% ± 0.30% | +11.92% | 66.16% ± 1.60% | < 1/201 (0.0050) |

> *Permutation p-value note:* With `N = 200` permutations, zero shuffled scores exceeding the observed score hits the Phipson–Smyth resolution floor `1/(N+1)` and is reported as `< 1/201 (0.0050)`.

### Regression Target: Exact `secret_weight` (5×10 Repeated K-Fold CV)
| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | q-Residualized R² | Perm p-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random Forest | -0.1202 ± 0.0523 | 1.4078 | -0.3785 ± 0.0456 | +0.0736 ± 0.0361 | < 1/201 (0.0050) | Negative R² (worse than mean baseline) |
| Gradient Boosting | +0.1969 ± 0.0303 | 1.1923 | -0.0124 ± 0.0071 | +0.2139 ± 0.0282 | < 1/201 (0.0050) | — |
| MLP | +0.2127 ± 0.0272 | 1.1805 | -0.0007 ± 0.0054 | +0.2173 ± 0.0267 | < 1/201 (0.0050) | — |
| Linear Regression | +0.0327 ± 0.0169 | 1.3089 | -0.0016 ± 0.0020 | +0.0327 ± 0.0171 | < 1/201 (0.0050) | — |
| SVR (RBF) | +0.2118 ± 0.0299 | 1.1811 | -0.0292 ± 0.0116 | +0.2167 ± 0.0297 | < 1/201 (0.0050) | — |

> *Regression caveat:* Negative-`R²` models (Random Forest (-0.1202)) perform worse than predicting the constant sample mean (`R² = 0`, `RMSE = 1.3320`): unpruned trees overfit two noisy timing features (`shuffled R² = -0.3785`), so their permutation `p` reflects beating an even worse shuffled fit rather than useful continuous prediction. Report the positive-`R²` models instead (best: MLP `R² = +0.2127`).

## 3. Legacy Instrumentation Comparison (`tracemalloc`-Wrapped Pass)

### Correlations vs. `secret_weight` under `tracemalloc`
| Signal | Mean ± Std | Pearson r (95% Boot CI) | Spearman ρ (95% Boot CI) |
| :--- | :---: | :---: | :---: |
| `tm_dec_timing_us` | 852.27 ± 74.87 | +0.7458 [+0.7030, +0.7835] | +0.8290 [+0.8177, +0.8393] |
| `tm_dec_cpu_user_us` | 853.65 ± 77.61 | +0.7214 [+0.6736, +0.7674] | +0.8300 [+0.8186, +0.8404] |
| `dec_ram_peak_bytes` | 1531.71 ± 32.14 | +0.0672 [+0.0388, +0.0934] | +0.0681 [+0.0392, +0.0954] |
| `dec_num_allocations` | 4.00 ± 0.01 | -0.0178 [-0.0314, +0.0000] | -0.0193 [-0.0339, +0.0000] |

### ML Summary under `tracemalloc`
- **2-feature (`tm_dec_timing_us`, `tm_dec_cpu_user_us`) Best Classifier (SVM (RBF)):** Real Acc `84.01%`, Shuffled `53.56%`, q-Residualized `83.62%`, Perm `p < 1/201 (0.0050)`
- **2-feature Best Regressor (SVR (RBF)):** Real R² `+0.6960`, Shuffled R² `-0.0331`, q-Residualized R² `+0.6954`, Perm `p < 1/201 (0.0050)`
- **4-feature (+ `dec_ram_peak_bytes`, `dec_num_allocations`) Best Classifier (SVM (RBF)):** Real Acc `83.99%`, Shuffled `53.51%`, q-Residualized `83.48%`, Perm `p < 1/201 (0.0050)`
- **4-feature Best Regressor (SVR (RBF)):** Real R² `+0.7121`, Shuffled R² `-0.0313`, q-Residualized R² `+0.7118`, Perm `p < 1/201 (0.0050)`
