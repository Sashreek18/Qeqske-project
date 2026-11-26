# `QEQSKE-Variant ML-KEM-512 (QRNG, n=256, k=2, per-key NTT q)` Decrypt Side-Channel Evaluation (5,000 Trials)

## 1. Experiment Setup & Secret Distribution
- **Environment:** `Linux 7.0.0-31-generic` (`Linux-7.0.0-31-generic-x86_64-with-glibc2.39`), CPU `12th Gen Intel(R) Core(TM) i9-12900K`, Python `3.12.3`, NumPy `2.5.2`
- **Parameters:** `n=256`, `k=2`, `eta_1=3`, `eta_2=2`, `du=10`, `dv=4`, `d_poly=16`, seed `1337`
- **Correctness:** Clean `decaps` `5000/5000` (100.00%); `tracemalloc` `decaps` `5000/5000` (100.00%)
- **Target 1 (`secret_weight`):** non-zero entries out of `512` centered coeffs — range `313..388`, mean `351.66 ± 10.31`, binary `> 352.0` (class 0: `2597`, class 1: `2403`, majority baseline **51.94%**)
- **Target 2a (`s00_positive`: `s[0][0] > 0`):** all `5000` trials — class 0 (`<=0`): `3269`, class 1 (`>0`): `1731`, majority baseline **65.38%**
- **Target 2b (`s00_sign_nonzero`: `s[0][0] > 0` on `s[0][0] != 0`):** `3426` trials — negative: `1695`, positive: `1731`, majority baseline **50.53%**
- **NTT Sparsity Note:** Mean zero coefficients in `s_hat` = `0.04 / 512` (vs. `160.34` zeros in centered `s`).

## 2. Cross-Pass Side-Channel Summary (Weight & Coefficient Sign)

| Execution Pass | Timing Mean ± SD (µs) | `r_weight` | `ρ_weight` | Best Weight Clf (Real / Base / `q`-Res / Perm `p`) | Best Weight Reg `R²` (Perm `p`) | `ρ_s00_pos` | Best `s00 > 0` Clf (Real / Base / Perm `p`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean `decaps(dk, c)` (Full FIPS 203) | 2006.7 ± 79.3 | +0.0018 | -0.0120 | 52.52% MLP / 51.94% / 51.80% / 0.0697 | All < 0 (max -0.0010) / 0.0348 | -0.0128 | 65.38% Logistic Regression / 65.38% / 0.5323 |
| Clean `_k_pke_decrypt(dk_pke, c)` (Isolated Core) | 566.3 ± 32.2 | -0.0209 | -0.0179 | 51.88% Logistic Regression / 51.94% / 51.89% / 0.7214 | All < 0 (max -0.0012) / 0.1045 | +0.0041 | 65.34% Logistic Regression / 65.38% / 0.8010 |
| `tracemalloc` `decaps(dk, c)` (2-feat) | 22399.3 ± 1107.2 | +0.0213 | +0.0241 | 51.62% Logistic Regression / 51.94% / 51.63% / 0.9851 | All < 0 (max -0.0010) / 0.2687 | -0.0018 | 65.35% Logistic Regression / 65.38% / 0.8209 |
| `tracemalloc` `decaps(dk, c)` (4-feat) | 22399.3 ± 1107.2 | +0.0213 | +0.0241 | 51.54% Logistic Regression / 51.94% / 51.51% / 0.7960 | All < 0 (max -0.0014) / 0.2289 | -0.0018 | 65.34% Logistic Regression / 65.38% / 0.6567 |

## 3. Primary Clean `decaps(dk, c)` Detailed Results

### Correlations vs. Secret Targets (2,000-Resample Bootstrap 95% CIs)
| Signal | Target | Pearson `r` [95% CI] | Pearson `p` | Spearman `ρ` [95% CI] | Spearman `p` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `dec_timing_us` | `secret_weight` | +0.0018 [-0.0273, +0.0309] | 9.01e-01 | -0.0120 [-0.0411, +0.0163] | 3.97e-01 |
| `dec_timing_us` | `s00_value` | -0.0236 [-0.0515, +0.0048] | 9.47e-02 | -0.0140 [-0.0408, +0.0139] | 3.22e-01 |
| `dec_timing_us` | `s00_positive` | -0.0157 [-0.0412, +0.0118] | 2.68e-01 | -0.0128 [-0.0396, +0.0150] | 3.66e-01 |
| `dec_cpu_user_us` | `secret_weight` | -0.0216 [-0.0507, +0.0093] | 1.27e-01 | -0.0144 [-0.0434, +0.0139] | 3.08e-01 |
| `dec_cpu_user_us` | `s00_value` | -0.0056 [-0.0334, +0.0247] | 6.91e-01 | -0.0118 [-0.0391, +0.0156] | 4.04e-01 |
| `dec_cpu_user_us` | `s00_positive` | -0.0066 [-0.0335, +0.0204] | 6.43e-01 | -0.0113 [-0.0378, +0.0174] | 4.26e-01 |

### Confound (`q`) & Time-Drift Checks
- **`q` vs `secret_weight` / `s00_positive`:** `r = -0.0013` (`p = 0.9260`) / `r = -0.0050` (`p = 0.7219`)
- **`q` vs `dec_timing_us` / `dec_cpu_user_us`:** `r = +0.2416` (`p = 0.0000`) / `r = +0.2275` (`p = 0.0000`)
- **Execution order vs `dec_timing_us` / `secret_weight`:** `r = -0.0692` (`p = 0.0000`) / `r = -0.0230` (`p = 0.1043`)

### Target 1: `secret_weight` Classification (5×10 CV, Majority Baseline = 51.94%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 49.92% ± 1.51% | 0.5007 | 49.57% ± 1.22% | +0.35% | 50.29% ± 1.30% | 0.3781 |
| Gradient Boosting | 51.22% ± 1.14% | 0.5016 | 51.20% ± 1.10% | +0.02% | 50.76% ± 1.08% | 0.2488 |
| MLP | 52.52% ± 0.92% | 0.5162 | 51.86% ± 0.36% | +0.66% | 51.80% ± 0.57% | 0.0697 |
| Logistic Regression | 52.02% ± 0.34% | 0.5069 | 51.83% ± 0.64% | +0.19% | 51.97% ± 0.36% | 0.1343 |
| SVM (RBF) | 51.76% ± 0.49% | 0.5120 | 51.98% ± 0.34% | -0.22% | 51.81% ± 0.35% | 0.3284 |

### Target 1: Exact `secret_weight` Regression (5×10 CV)
| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | `q`-Residualized R² | Perm `p`-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random Forest | -0.4328 ± 0.0387 | 12.3270 | -0.4148 ± 0.0489 | -0.2675 ± 0.0304 | 0.6020 | Negative R² (worse than mean baseline) |
| Gradient Boosting | -0.0198 ± 0.0088 | 10.4019 | -0.0142 ± 0.0089 | -0.0192 ± 0.0074 | 0.8109 | Negative R² (worse than mean baseline) |
| MLP | -0.0437 ± 0.0459 | 10.5202 | -0.0365 ± 0.0345 | -0.0375 ± 0.0379 | 0.2786 | Negative R² (worse than mean baseline) |
| Linear Regression | -0.0010 ± 0.0022 | 10.3056 | -0.0018 ± 0.0017 | -0.0010 ± 0.0022 | 0.0348 | Negative R² (worse than mean baseline) |
| SVR (RBF) | -0.0037 ± 0.0047 | 10.3191 | -0.0044 ± 0.0030 | -0.0038 ± 0.0039 | 0.0398 | Negative R² (worse than mean baseline) |

### Target 2a: `s[0][0] > 0` Sign Classification (5×10 CV, Majority Baseline = 65.38%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 55.64% ± 1.32% | 0.5106 | 54.35% ± 1.44% | +1.29% | 57.55% ± 1.20% | 0.2189 |
| Gradient Boosting | 64.87% ± 0.45% | 0.5167 | 64.75% ± 0.49% | +0.13% | 64.63% ± 0.46% | 0.8557 |
| MLP | 65.36% ± 0.10% | 0.5146 | 65.38% ± 0.08% | -0.02% | 65.36% ± 0.09% | 0.3483 |
| Logistic Regression | 65.38% ± 0.04% | 0.5055 | 65.38% ± 0.04% | +0.00% | 65.38% ± 0.04% | 0.5323 |
| SVM (RBF) | 65.36% ± 0.07% | 0.5016 | 65.32% ± 0.09% | +0.03% | 65.34% ± 0.10% | 0.4428 |

### Target 2b: Balanced Non-Zero `sign(s[0][0])` (`s[0][0] != 0`, n=3426, Baseline = 50.53%)
- **Clean `decaps` Best (Gradient Boosting):** Real `52.10% ± 1.78%` (AUC `0.5185`, Shuffled `49.61%`, `q`-Resid `50.98%`, Perm `p 0.0448`)
- **Isolated `_k_pke_decrypt` Best (SVM (RBF)):** Real `51.99% ± 1.61%` (AUC `0.5212`, Perm `p 0.0249`)

> *Notes:* Permutation `p`-values at the `N=200` floor are reported as `< 1/201 (0.0050)`. Negative-`R²` models perform worse than predicting the sample mean (`R² = 0`).
