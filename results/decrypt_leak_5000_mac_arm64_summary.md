# `decrypt_it (original, mac_arm64)` Side-Channel Leakage Evaluation (5,000 Trials)

## 1. Experiment Setup & Data Integrity
- **Environment:** `Darwin 27.0.0` (`macOS-27.0.1-arm64-arm-64bit-Mach-O`), CPU `Apple M2`, Python `3.13.1`, NumPy `2.2.6`
- **Trials:** 5000 (fixed plaintext `HELLO`, parameters `n=4`, `k=2`, fresh key per trial, seed `42`)
- **Decryption correctness:** Clean pass `4997/5000` (99.94%); `tracemalloc` pass `4997/5000` (99.94%)
- **QRNG stream:** `qrng_combined.txt` (`1,106,944` numbers, `cycle=True`, consumed `4,905,000` draws across `4` wraps; `5,000` unique `(q, s)` pairs)
- **Binary target:** `secret_weight > 5.0` -> class 0: `2685`, class 1: `2315`, majority-class baseline: **53.70%**
- **Regression target:** exact `secret_weight` (range `0..8`, mean `5.3266`, std `1.3320`)

## 2. Primary Analysis: Clean Uninstrumented Traces (`dec_timing_us`, `dec_cpu_user_us`)

### Correlations vs. `secret_weight` (2,000-Resample Paired Bootstrap 95% CIs)
| Signal | Mean ± Std (µs) | Median (IQR) µs | Pearson r (95% Boot CI) | Pearson p | Spearman ρ (95% Boot CI) | Spearman p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dec_timing_us` | 76.79 ± 6.34 | 75.33 (3.33) | +0.1370 [+0.1024, +0.1730] | 2.30e-22 | +0.3418 [+0.3167, +0.3681] | 4.71e-137 |
| `dec_cpu_user_us` | 76.18 ± 4.13 | 76.00 (3.00) | +0.2271 [+0.1855, +0.2716] | 1.71e-59 | +0.3706 [+0.3464, +0.3955] | 1.52e-162 |

### Confound & Time-Drift Checks (Clean Pass)
- **Modulus `q` vs `secret_weight`:** Pearson `r = -0.0031` (`p = 0.8284`), Spearman `ρ = -0.0042` (`p = 0.7692`)
- **Modulus `q` vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = -0.0687` (`p = 0.0000`) / `r = -0.1049` (`p = 0.0000`)
- **Execution order vs `secret_weight`:** Pearson `r = -0.0082` (`p = 0.5619`)
- **Execution order vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = +0.0619` (`p = 0.0000`) / `r = +0.0734` (`p = 0.0000`)

### Binary Classification (5×10 Repeated Stratified CV, Majority Baseline = 53.70%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized Acc | Perm p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 59.22% ± 1.35% | 0.6213 | 51.47% ± 1.32% | +7.75% | 58.88% ± 1.21% | < 1/201 (0.0050) |
| Gradient Boosting | 63.65% ± 1.30% | 0.6798 | 52.38% ± 1.07% | +11.27% | 63.29% ± 1.33% | < 1/201 (0.0050) |
| MLP | 63.61% ± 1.40% | 0.6880 | 53.49% ± 0.34% | +10.12% | 63.84% ± 1.40% | < 1/201 (0.0050) |
| Logistic Regression | 58.33% ± 1.06% | 0.6626 | 53.64% ± 0.46% | +4.69% | 59.58% ± 1.08% | < 1/201 (0.0050) |
| SVM (RBF) | 63.43% ± 1.38% | 0.6820 | 53.38% ± 0.46% | +10.05% | 63.77% ± 1.51% | < 1/201 (0.0050) |

> *Permutation p-value note:* With `N = 200` permutations, zero shuffled scores exceeding the observed score hits the Phipson–Smyth resolution floor `1/(N+1)` and is reported as `< 1/201 (0.0050)`.

### Regression Target: Exact `secret_weight` (5×10 Repeated K-Fold CV)
| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | q-Residualized R² | Perm p-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random Forest | -0.0161 ± 0.0531 | 1.3409 | -0.2300 ± 0.0365 | +0.0288 ± 0.0460 | < 1/201 (0.0050) | Negative R² (worse than mean baseline) |
| Gradient Boosting | +0.1677 ± 0.0342 | 1.2138 | -0.0164 ± 0.0087 | +0.1726 ± 0.0309 | < 1/201 (0.0050) | — |
| MLP | +0.1768 ± 0.0327 | 1.2071 | -0.0062 ± 0.0059 | +0.1754 ± 0.0322 | < 1/201 (0.0050) | — |
| Linear Regression | +0.0506 ± 0.0228 | 1.2967 | -0.0015 ± 0.0015 | +0.0509 ± 0.0237 | < 1/201 (0.0050) | — |
| SVR (RBF) | +0.1743 ± 0.0351 | 1.2089 | -0.0311 ± 0.0125 | +0.1749 ± 0.0337 | < 1/201 (0.0050) | — |

> *Regression caveat:* Negative-`R²` models (Random Forest (-0.0161)) perform worse than predicting the constant sample mean (`R² = 0`, `RMSE = 1.3320`): unpruned trees overfit two noisy timing features (`shuffled R² = -0.2300`), so their permutation `p` reflects beating an even worse shuffled fit rather than useful continuous prediction. Report the positive-`R²` models instead (best: MLP `R² = +0.1768`).

## 3. Legacy Instrumentation Comparison (`tracemalloc`-Wrapped Pass)

### Correlations vs. `secret_weight` under `tracemalloc`
| Signal | Mean ± Std | Pearson r (95% Boot CI) | Spearman ρ (95% Boot CI) |
| :--- | :---: | :---: | :---: |
| `tm_dec_timing_us` | 892.15 ± 109.01 | +0.5800 [+0.5036, +0.6717] | +0.8302 [+0.8186, +0.8411] |
| `tm_dec_cpu_user_us` | 886.35 ± 90.06 | +0.6919 [+0.6263, +0.7589] | +0.8462 [+0.8352, +0.8564] |
| `dec_ram_peak_bytes` | 1531.71 ± 32.14 | +0.0672 [+0.0388, +0.0934] | +0.0681 [+0.0392, +0.0954] |
| `dec_num_allocations` | 4.00 ± 0.01 | -0.0178 [-0.0314, +0.0000] | -0.0193 [-0.0339, +0.0000] |

### ML Summary under `tracemalloc`
- **2-feature (`tm_dec_timing_us`, `tm_dec_cpu_user_us`) Best Classifier (Logistic Regression):** Real Acc `85.21%`, Shuffled `53.67%`, q-Residualized `85.23%`, Perm `p < 1/201 (0.0050)`
- **2-feature Best Regressor (Gradient Boosting):** Real R² `+0.7247`, Shuffled R² `-0.0227`, q-Residualized R² `+0.7133`, Perm `p < 1/201 (0.0050)`
- **4-feature (+ `dec_ram_peak_bytes`, `dec_num_allocations`) Best Classifier (Logistic Regression):** Real Acc `85.26%`, Shuffled `53.67%`, q-Residualized `85.34%`, Perm `p < 1/201 (0.0050)`
- **4-feature Best Regressor (Gradient Boosting):** Real R² `+0.7461`, Shuffled R² `-0.0212`, q-Residualized R² `+0.7623`, Perm `p < 1/201 (0.0050)`
