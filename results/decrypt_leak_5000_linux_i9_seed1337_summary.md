# `decrypt_it (original, linux_i9_seed1337, seed=1337)` Side-Channel Leakage Evaluation (5,000 Trials)

## 1. Experiment Setup & Data Integrity
- **Environment:** `Linux 7.0.0-31-generic` (`Linux-7.0.0-31-generic-x86_64-with-glibc2.39`), CPU `12th Gen Intel(R) Core(TM) i9-12900K`, Python `3.12.3`, NumPy `2.5.2`
- **Trials:** 5000 (fixed plaintext `HELLO`, parameters `n=4`, `k=2`, fresh key per trial, seed `1337`)
- **Decryption correctness:** Clean pass `4996/5000` (99.92%); `tracemalloc` pass `4996/5000` (99.92%)
- **QRNG stream:** `qrng_combined.txt` (`1,106,944` numbers, `cycle=True`, consumed `4,905,000` draws across `4` wraps; `4,999` unique `(q, s)` pairs)
- **Binary target:** `secret_weight > 5.0` -> class 0: `2577`, class 1: `2423`, majority-class baseline: **51.54%**
- **Regression target:** exact `secret_weight` (range `0..8`, mean `5.3680`, std `1.3502`)

## 2. Primary Analysis: Clean Uninstrumented Traces (`dec_timing_us`, `dec_cpu_user_us`)

### Correlations vs. `secret_weight` (2,000-Resample Paired Bootstrap 95% CIs)
| Signal | Mean ± Std (µs) | Median (IQR) µs | Pearson r (95% Boot CI) | Pearson p | Spearman ρ (95% Boot CI) | Spearman p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dec_timing_us` | 72.78 ± 5.90 | 71.67 (3.23) | +0.1904 [+0.1619, +0.2215] | 5.07e-42 | +0.4099 [+0.3848, +0.4339] | 6.00e-202 |
| `dec_cpu_user_us` | 72.50 ± 9.46 | 72.00 (3.00) | +0.1286 [+0.0997, +0.1561] | 7.05e-20 | +0.3914 [+0.3657, +0.4159] | 1.04e-182 |

### Confound & Time-Drift Checks (Clean Pass)
- **Modulus `q` vs `secret_weight`:** Pearson `r = -0.0047` (`p = 0.7379`), Spearman `ρ = -0.0014` (`p = 0.9199`)
- **Modulus `q` vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = -0.0803` (`p = 0.0000`) / `r = -0.0452` (`p = 0.0014`)
- **Execution order vs `secret_weight`:** Pearson `r = +0.0220` (`p = 0.1195`)
- **Execution order vs `dec_timing_us` / `dec_cpu_user_us`:** Pearson `r = -0.1151` (`p = 0.0000`) / `r = -0.1525` (`p = 0.0000`)

### Binary Classification (5×10 Repeated Stratified CV, Majority Baseline = 51.54%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized Acc | Perm p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 56.71% ± 1.16% | 0.6183 | 49.52% ± 1.32% | +7.19% | 60.62% ± 1.34% | < 1/201 (0.0050) |
| Gradient Boosting | 65.26% ± 1.44% | 0.7011 | 51.51% ± 1.27% | +13.75% | 64.81% ± 1.68% | < 1/201 (0.0050) |
| MLP | 65.92% ± 1.44% | 0.7063 | 52.18% ± 1.00% | +13.74% | 65.34% ± 1.73% | < 1/201 (0.0050) |
| Logistic Regression | 60.26% ± 0.94% | 0.7001 | 51.72% ± 0.54% | +8.54% | 60.60% ± 1.08% | < 1/201 (0.0050) |
| SVM (RBF) | 65.72% ± 1.46% | 0.7033 | 52.10% ± 0.70% | +13.62% | 65.15% ± 1.64% | < 1/201 (0.0050) |

> *Permutation p-value note:* With `N = 200` permutations, zero shuffled scores exceeding the observed score hits the Phipson–Smyth resolution floor `1/(N+1)` and is reported as `< 1/201 (0.0050)`.

### Regression Target: Exact `secret_weight` (5×10 Repeated K-Fold CV)
| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | q-Residualized R² | Perm p-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random Forest | -0.1181 ± 0.0494 | 1.4257 | -0.4115 ± 0.0423 | +0.0810 ± 0.0428 | < 1/201 (0.0050) | Negative R² (worse than mean baseline) |
| Gradient Boosting | +0.2010 ± 0.0292 | 1.2054 | -0.0159 ± 0.0075 | +0.2100 ± 0.0304 | < 1/201 (0.0050) | — |
| MLP | +0.2070 ± 0.0300 | 1.2009 | -0.0099 ± 0.0112 | +0.2103 ± 0.0301 | < 1/201 (0.0050) | — |
| Linear Regression | +0.0322 ± 0.0154 | 1.3271 | -0.0016 ± 0.0016 | +0.0322 ± 0.0156 | 0.8458 | — |
| SVR (RBF) | +0.2061 ± 0.0307 | 1.2015 | -0.0348 ± 0.0106 | +0.2097 ± 0.0315 | < 1/201 (0.0050) | — |

> *Regression caveat:* Negative-`R²` models (Random Forest (-0.1181)) perform worse than predicting the constant sample mean (`R² = 0`, `RMSE = 1.3502`): unpruned trees overfit two noisy timing features (`shuffled R² = -0.4115`), so their permutation `p` reflects beating an even worse shuffled fit rather than useful continuous prediction. Report the positive-`R²` models instead (best: MLP `R² = +0.2070`).

## 3. Legacy Instrumentation Comparison (`tracemalloc`-Wrapped Pass)

### Correlations vs. `secret_weight` under `tracemalloc`
| Signal | Mean ± Std | Pearson r (95% Boot CI) | Spearman ρ (95% Boot CI) |
| :--- | :---: | :---: | :---: |
| `tm_dec_timing_us` | 864.43 ± 71.95 | +0.7846 [+0.7501, +0.8154] | +0.8615 [+0.8509, +0.8711] |
| `tm_dec_cpu_user_us` | 865.92 ± 72.92 | +0.7757 [+0.7402, +0.8095] | +0.8620 [+0.8513, +0.8714] |
| `dec_ram_peak_bytes` | 1531.78 ± 31.90 | -0.0115 [-0.0392, +0.0153] | -0.0094 [-0.0370, +0.0176] |
| `dec_num_allocations` | 4.00 ± 0.01 | -0.0171 [-0.0300, +0.0000] | -0.0189 [-0.0330, +0.0000] |

### ML Summary under `tracemalloc`
- **2-feature (`tm_dec_timing_us`, `tm_dec_cpu_user_us`) Best Classifier (Logistic Regression):** Real Acc `86.33%`, Shuffled `51.39%`, q-Residualized `86.21%`, Perm `p < 1/201 (0.0050)`
- **2-feature Best Regressor (Gradient Boosting):** Real R² `+0.7445`, Shuffled R² `-0.0170`, q-Residualized R² `+0.7434`, Perm `p < 1/201 (0.0050)`
- **4-feature (+ `dec_ram_peak_bytes`, `dec_num_allocations`) Best Classifier (SVM (RBF)):** Real Acc `86.44%`, Shuffled `50.69%`, q-Residualized `86.23%`, Perm `p < 1/201 (0.0050)`
- **4-feature Best Regressor (Gradient Boosting):** Real R² `+0.7724`, Shuffled R² `-0.0171`, q-Residualized R² `+0.7870`, Perm `p < 1/201 (0.0050)`
