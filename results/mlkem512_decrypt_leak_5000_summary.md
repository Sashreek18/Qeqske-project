# `Standard ML-KEM-512 (kyber-py, n=256, k=2, q=3329)` Decrypt Side-Channel Evaluation (5,000 Trials)

## 1. Experiment Setup & Secret Distribution
- **Environment:** `Linux 7.0.0-31-generic` (`Linux-7.0.0-31-generic-x86_64-with-glibc2.39`), CPU `12th Gen Intel(R) Core(TM) i9-12900K`, Python `3.12.3`, NumPy `2.5.2`
- **Parameters:** `n=256`, `k=2`, `eta_1=3`, `eta_2=2`, `du=10`, `dv=4`, `d_poly=12`, seed `42`
- **Correctness:** Clean `decaps` `5000/5000` (100.00%); `tracemalloc` `decaps` `5000/5000` (100.00%)
- **Target 1 (`secret_weight`):** non-zero entries out of `512` centered coeffs — range `316..390`, mean `351.96 ± 10.45`, binary `> 352.0` (class 0: `2601`, class 1: `2399`, majority baseline **52.02%**)
- **Target 2a (`s00_positive`: `s[0][0] > 0`):** all `5000` trials — class 0 (`<=0`): `3281`, class 1 (`>0`): `1719`, majority baseline **65.62%**
- **Target 2b (`s00_sign_nonzero`: `s[0][0] > 0` on `s[0][0] != 0`):** `3418` trials — negative: `1699`, positive: `1719`, majority baseline **50.29%**
- **NTT Sparsity Note:** Mean zero coefficients in `s_hat` = `0.15 / 512` (vs. `160.04` zeros in centered `s`).

## 2. Cross-Pass Side-Channel Summary (Weight & Coefficient Sign)

| Execution Pass | Timing Mean ± SD (µs) | `r_weight` | `ρ_weight` | Best Weight Clf (Real / Base / `q`-Res / Perm `p`) | Best Weight Reg `R²` (Perm `p`) | `ρ_s00_pos` | Best `s00 > 0` Clf (Real / Base / Perm `p`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean `decaps(dk, c)` (Full FIPS 203) | 2097.0 ± 295.0 | +0.0188 | +0.0147 | 52.12% SVM (RBF) / 52.02% / 52.12% / 0.1592 | All < 0 (max -0.0019) / 0.5871 | -0.0185 | 65.62% Logistic Regression / 65.62% / 0.6716 |
| Clean `_k_pke_decrypt(dk_pke, c)` (Isolated Core) | 585.7 ± 76.1 | -0.0075 | +0.0006 | 52.24% MLP / 52.02% / 52.24% / 0.7711 | All < 0 (max -0.0012) / 0.1493 | -0.0207 | 65.59% MLP / 65.62% / 0.9303 |
| `tracemalloc` `decaps(dk, c)` (2-feat) | 21519.5 ± 3062.7 | -0.0160 | -0.0166 | 51.84% SVM (RBF) / 52.02% / 51.84% / 0.4726 | All < 0 (max -0.0017) / 0.2388 | +0.0085 | 65.62% MLP / 65.62% / 0.9851 |
| `tracemalloc` `decaps(dk, c)` (4-feat) | 21519.5 ± 3062.7 | -0.0160 | -0.0166 | 51.89% SVM (RBF) / 52.02% / 51.89% / 0.5274 | All < 0 (max -0.0023) / 0.2637 | +0.0085 | 65.61% Logistic Regression / 65.62% / 0.1791 |

## 3. Primary Clean `decaps(dk, c)` Detailed Results

### Correlations vs. Secret Targets (2,000-Resample Bootstrap 95% CIs)
| Signal | Target | Pearson `r` [95% CI] | Pearson `p` | Spearman `ρ` [95% CI] | Spearman `p` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `dec_timing_us` | `secret_weight` | +0.0188 [-0.0080, +0.0474] | 1.83e-01 | +0.0147 [-0.0134, +0.0424] | 3.00e-01 |
| `dec_timing_us` | `s00_value` | +0.0036 [-0.0263, +0.0338] | 8.01e-01 | -0.0184 [-0.0485, +0.0102] | 1.94e-01 |
| `dec_timing_us` | `s00_positive` | +0.0083 [-0.0205, +0.0376] | 5.56e-01 | -0.0185 [-0.0478, +0.0109] | 1.90e-01 |
| `dec_cpu_user_us` | `secret_weight` | +0.0193 [-0.0077, +0.0472] | 1.73e-01 | +0.0142 [-0.0136, +0.0420] | 3.17e-01 |
| `dec_cpu_user_us` | `s00_value` | +0.0036 [-0.0260, +0.0344] | 7.99e-01 | -0.0194 [-0.0497, +0.0101] | 1.70e-01 |
| `dec_cpu_user_us` | `s00_positive` | +0.0084 [-0.0207, +0.0377] | 5.53e-01 | -0.0194 [-0.0490, +0.0094] | 1.71e-01 |

### Confound (`q`) & Time-Drift Checks
- **`q` vs `secret_weight` / `s00_positive`:** `r = +0.0000` (`p = 1.0000`) / `r = +0.0000` (`p = 1.0000`)
- **`q` vs `dec_timing_us` / `dec_cpu_user_us`:** `r = +0.0000` (`p = 1.0000`) / `r = +0.0000` (`p = 1.0000`)
- **Execution order vs `dec_timing_us` / `secret_weight`:** `r = -0.1523` (`p = 0.0000`) / `r = -0.0167` (`p = 0.2387`)

### Target 1: `secret_weight` Classification (5×10 CV, Majority Baseline = 52.02%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 50.18% ± 1.49% | 0.5072 | 49.16% ± 1.42% | +1.01% | 50.18% ± 1.49% | 0.8706 |
| Gradient Boosting | 51.20% ± 1.16% | 0.4876 | 49.96% ± 1.16% | +1.25% | 51.20% ± 1.16% | 0.8607 |
| MLP | 51.82% ± 0.53% | 0.5011 | 51.91% ± 0.29% | -0.09% | 51.82% ± 0.53% | 0.8259 |
| Logistic Regression | 51.86% ± 0.34% | 0.5036 | 51.88% ± 0.33% | -0.02% | 51.86% ± 0.34% | 0.2687 |
| SVM (RBF) | 52.12% ± 0.43% | 0.5079 | 51.78% ± 0.32% | +0.33% | 52.12% ± 0.43% | 0.1592 |

### Target 1: Exact `secret_weight` Regression (5×10 CV)
| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | `q`-Residualized R² | Perm `p`-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random Forest | -0.4251 ± 0.0467 | 12.4582 | -0.4294 ± 0.0442 | -0.4251 ± 0.0467 | 0.1940 | Negative R² (worse than mean baseline) |
| Gradient Boosting | -0.0248 ± 0.0114 | 10.5675 | -0.0214 ± 0.0106 | -0.0248 ± 0.0114 | 0.9801 | Negative R² (worse than mean baseline) |
| MLP | -0.0254 ± 0.0658 | 10.5660 | -0.0205 ± 0.0577 | -0.0254 ± 0.0658 | 0.8507 | Negative R² (worse than mean baseline) |
| Linear Regression | -0.0021 ± 0.0030 | 10.4500 | -0.0014 ± 0.0017 | -0.0021 ± 0.0030 | 0.3284 | Negative R² (worse than mean baseline) |
| SVR (RBF) | -0.0019 ± 0.0023 | 10.4489 | -0.0024 ± 0.0015 | -0.0019 ± 0.0023 | 0.5871 | Negative R² (worse than mean baseline) |

### Target 2a: `s[0][0] > 0` Sign Classification (5×10 CV, Majority Baseline = 65.62%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 54.86% ± 1.58% | 0.4956 | 55.70% ± 1.21% | -0.84% | 54.86% ± 1.58% | 0.7811 |
| Gradient Boosting | 65.05% ± 0.36% | 0.4983 | 64.96% ± 0.47% | +0.09% | 65.05% ± 0.36% | 0.6119 |
| MLP | 65.62% ± 0.04% | 0.5135 | 65.62% ± 0.05% | +0.00% | 65.62% ± 0.04% | 0.8109 |
| Logistic Regression | 65.62% ± 0.04% | 0.4891 | 65.60% ± 0.07% | +0.02% | 65.62% ± 0.04% | 0.6716 |
| SVM (RBF) | 65.53% ± 0.13% | 0.4812 | 65.60% ± 0.07% | -0.08% | 65.53% ± 0.13% | 0.5323 |

### Target 2b: Balanced Non-Zero `sign(s[0][0])` (`s[0][0] != 0`, n=3418, Baseline = 50.29%)
- **Clean `decaps` Best (Gradient Boosting):** Real `51.10% ± 1.50%` (AUC `0.5129`, Shuffled `50.47%`, `q`-Resid `51.10%`, Perm `p 0.1542`)
- **Isolated `_k_pke_decrypt` Best (Gradient Boosting):** Real `51.31% ± 1.68%` (AUC `0.5120`, Perm `p 0.0149`)

> *Notes:* Permutation `p`-values at the `N=200` floor are reported as `< 1/201 (0.0050)`. Negative-`R²` models perform worse than predicting the sample mean (`R² = 0`).
