# `QEQSKE-Variant ML-KEM-512 (QRNG, n=256, k=2, per-key NTT q)` Decrypt Side-Channel Evaluation (5,000 Trials)

## 1. Experiment Setup & Secret Distribution
- **Environment:** `Linux 7.0.0-31-generic` (`Linux-7.0.0-31-generic-x86_64-with-glibc2.39`), CPU `12th Gen Intel(R) Core(TM) i9-12900K`, Python `3.12.3`, NumPy `2.5.2`
- **Parameters:** `n=256`, `k=2`, `eta_1=3`, `eta_2=2`, `du=10`, `dv=4`, `d_poly=16`, seed `42`
- **Correctness:** Clean `decaps` `5000/5000` (100.00%); `tracemalloc` `decaps` `5000/5000` (100.00%)
- **Target 1 (`secret_weight`):** non-zero entries out of `512` centered coeffs — range `314..389`, mean `352.12 ± 10.67`, binary `> 352.0` (class 0: `2557`, class 1: `2443`, majority baseline **51.14%**)
- **Target 2a (`s00_positive`: `s[0][0] > 0`):** all `5000` trials — class 0 (`<=0`): `3215`, class 1 (`>0`): `1785`, majority baseline **64.30%**
- **Target 2b (`s00_sign_nonzero`: `s[0][0] > 0` on `s[0][0] != 0`):** `3447` trials — negative: `1662`, positive: `1785`, majority baseline **51.78%**
- **NTT Sparsity Note:** Mean zero coefficients in `s_hat` = `0.04 / 512` (vs. `159.88` zeros in centered `s`).

## 2. Cross-Pass Side-Channel Summary (Weight & Coefficient Sign)

| Execution Pass | Timing Mean ± SD (µs) | `r_weight` | `ρ_weight` | Best Weight Clf (Real / Base / `q`-Res / Perm `p`) | Best Weight Reg `R²` (Perm `p`) | `ρ_s00_pos` | Best `s00 > 0` Clf (Real / Base / Perm `p`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean `decaps(dk, c)` (Full FIPS 203) | 1983.0 ± 97.5 | +0.0142 | +0.0120 | 51.09% Logistic Regression / 51.14% / 51.12% / 0.2438 | All < 0 (max -0.0016) / 0.9005 | +0.0144 | 64.28% SVM (RBF) / 64.30% / 0.3085 |
| Clean `_k_pke_decrypt(dk_pke, c)` (Isolated Core) | 563.8 ± 30.0 | +0.0029 | -0.0098 | 51.33% SVM (RBF) / 51.14% / 51.10% / 0.7313 | All < 0 (max -0.0009) / 0.8358 | -0.0012 | 64.29% Logistic Regression / 64.30% / 0.6119 |
| `tracemalloc` `decaps(dk, c)` (2-feat) | 22328.6 ± 2196.4 | -0.0039 | +0.0177 | 51.05% SVM (RBF) / 51.14% / 51.05% / 0.5224 | All < 0 (max -0.0019) / 0.7811 | +0.0090 | 64.30% Logistic Regression / 64.30% / 0.9154 |
| `tracemalloc` `decaps(dk, c)` (4-feat) | 22328.6 ± 2196.4 | -0.0039 | +0.0177 | 51.15% MLP / 51.14% / 51.03% / 0.2090 | All < 0 (max -0.0021) / 0.6716 | +0.0090 | 64.26% Logistic Regression / 64.30% / 0.9055 |

## 3. Primary Clean `decaps(dk, c)` Detailed Results

### Correlations vs. Secret Targets (2,000-Resample Bootstrap 95% CIs)
| Signal | Target | Pearson `r` [95% CI] | Pearson `p` | Spearman `ρ` [95% CI] | Spearman `p` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `dec_timing_us` | `secret_weight` | +0.0142 [-0.0081, +0.0396] | 3.15e-01 | +0.0120 [-0.0152, +0.0408] | 3.96e-01 |
| `dec_timing_us` | `s00_value` | +0.0197 [-0.0060, +0.0493] | 1.64e-01 | -0.0003 [-0.0287, +0.0288] | 9.82e-01 |
| `dec_timing_us` | `s00_positive` | +0.0227 [-0.0047, +0.0548] | 1.09e-01 | +0.0144 [-0.0132, +0.0431] | 3.08e-01 |
| `dec_cpu_user_us` | `secret_weight` | +0.0181 [-0.0097, +0.0475] | 2.01e-01 | +0.0108 [-0.0162, +0.0392] | 4.46e-01 |
| `dec_cpu_user_us` | `s00_value` | +0.0015 [-0.0257, +0.0294] | 9.13e-01 | -0.0001 [-0.0288, +0.0296] | 9.96e-01 |
| `dec_cpu_user_us` | `s00_positive` | +0.0180 [-0.0094, +0.0476] | 2.04e-01 | +0.0136 [-0.0141, +0.0435] | 3.35e-01 |

### Confound (`q`) & Time-Drift Checks
- **`q` vs `secret_weight` / `s00_positive`:** `r = +0.0161` (`p = 0.2557`) / `r = +0.0129` (`p = 0.3604`)
- **`q` vs `dec_timing_us` / `dec_cpu_user_us`:** `r = +0.1493` (`p = 0.0000`) / `r = +0.3129` (`p = 0.0000`)
- **Execution order vs `dec_timing_us` / `secret_weight`:** `r = +0.0776` (`p = 0.0000`) / `r = +0.0178` (`p = 0.2092`)

### Target 1: `secret_weight` Classification (5×10 CV, Majority Baseline = 51.14%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 49.03% ± 1.35% | 0.4871 | 49.66% ± 1.48% | -0.63% | 49.94% ± 1.41% | 0.6169 |
| Gradient Boosting | 49.51% ± 1.59% | 0.4890 | 51.11% ± 1.47% | -1.59% | 50.78% ± 1.27% | 0.8209 |
| MLP | 50.11% ± 1.16% | 0.4937 | 51.24% ± 1.20% | -1.13% | 50.63% ± 0.90% | 0.9154 |
| Logistic Regression | 51.09% ± 0.23% | 0.4962 | 51.49% ± 1.39% | -0.40% | 51.12% ± 0.10% | 0.2438 |
| SVM (RBF) | 50.66% ± 0.70% | 0.4896 | 50.99% ± 1.11% | -0.33% | 50.98% ± 0.45% | 0.6766 |

### Target 1: Exact `secret_weight` Regression (5×10 CV)
| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | `q`-Residualized R² | Perm `p`-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random Forest | -0.4422 ± 0.0519 | 12.8034 | -0.4043 ± 0.0466 | -0.2790 ± 0.0330 | 0.8557 | Negative R² (worse than mean baseline) |
| Gradient Boosting | -0.0200 ± 0.0089 | 10.7708 | -0.0171 ± 0.0104 | -0.0203 ± 0.0078 | 0.8507 | Negative R² (worse than mean baseline) |
| MLP | -0.2335 ± 0.3500 | 11.7418 | -0.1934 ± 0.3038 | -0.2368 ± 0.3634 | 0.9154 | Negative R² (worse than mean baseline) |
| Linear Regression | -0.0016 ± 0.0021 | 10.6733 | -0.0022 ± 0.0032 | -0.0018 ± 0.0020 | 0.9005 | Negative R² (worse than mean baseline) |
| SVR (RBF) | -0.0032 ± 0.0025 | 10.6813 | -0.0045 ± 0.0045 | -0.0033 ± 0.0023 | 0.8706 | Negative R² (worse than mean baseline) |

### Target 2a: `s[0][0] > 0` Sign Classification (5×10 CV, Majority Baseline = 64.30%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 53.73% ± 1.34% | 0.4983 | 54.57% ± 1.08% | -0.84% | 56.18% ± 1.56% | 0.6219 |
| Gradient Boosting | 63.69% ± 0.47% | 0.5007 | 63.80% ± 0.42% | -0.11% | 63.48% ± 0.39% | 0.2090 |
| MLP | 64.22% ± 0.12% | 0.4844 | 64.29% ± 0.05% | -0.07% | 64.20% ± 0.13% | 0.8159 |
| Logistic Regression | 64.26% ± 0.11% | 0.5075 | 64.30% ± 0.03% | -0.04% | 64.26% ± 0.09% | 0.7164 |
| SVM (RBF) | 64.28% ± 0.15% | 0.5024 | 64.29% ± 0.07% | -0.01% | 64.28% ± 0.14% | 0.3085 |

### Target 2b: Balanced Non-Zero `sign(s[0][0])` (`s[0][0] != 0`, n=3447, Baseline = 51.78%)
- **Clean `decaps` Best (Logistic Regression):** Real `51.76% ± 0.12%` (AUC `0.4888`, Shuffled `51.69%`, `q`-Resid `51.76%`, Perm `p 0.4328`)
- **Isolated `_k_pke_decrypt` Best (Logistic Regression):** Real `51.67% ± 0.30%` (AUC `0.4944`, Perm `p 0.2388`)

> *Notes:* Permutation `p`-values at the `N=200` floor are reported as `< 1/201 (0.0050)`. Negative-`R²` models perform worse than predicting the sample mean (`R² = 0`).
