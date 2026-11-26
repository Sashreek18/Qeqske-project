# `Standard ML-KEM-512 (kyber-py, n=256, k=2, q=3329)` Decrypt Side-Channel Evaluation (5,000 Trials)

## 1. Experiment Setup & Secret Distribution
- **Environment:** `Linux 7.0.0-31-generic` (`Linux-7.0.0-31-generic-x86_64-with-glibc2.39`), CPU `12th Gen Intel(R) Core(TM) i9-12900K`, Python `3.12.3`, NumPy `2.5.2`
- **Parameters:** `n=256`, `k=2`, `eta_1=3`, `eta_2=2`, `du=10`, `dv=4`, `d_poly=12`, seed `1337`
- **Correctness:** Clean `decaps` `5000/5000` (100.00%); `tracemalloc` `decaps` `5000/5000` (100.00%)
- **Target 1 (`secret_weight`):** non-zero entries out of `512` centered coeffs — range `313..397`, mean `352.00 ± 10.52`, binary `> 352.0` (class 0: `2584`, class 1: `2416`, majority baseline **51.68%**)
- **Target 2a (`s00_positive`: `s[0][0] > 0`):** all `5000` trials — class 0 (`<=0`): `3317`, class 1 (`>0`): `1683`, majority baseline **66.34%**
- **Target 2b (`s00_sign_nonzero`: `s[0][0] > 0` on `s[0][0] != 0`):** `3458` trials — negative: `1775`, positive: `1683`, majority baseline **51.33%**
- **NTT Sparsity Note:** Mean zero coefficients in `s_hat` = `0.17 / 512` (vs. `160.00` zeros in centered `s`).

## 2. Cross-Pass Side-Channel Summary (Weight & Coefficient Sign)

| Execution Pass | Timing Mean ± SD (µs) | `r_weight` | `ρ_weight` | Best Weight Clf (Real / Base / `q`-Res / Perm `p`) | Best Weight Reg `R²` (Perm `p`) | `ρ_s00_pos` | Best `s00 > 0` Clf (Real / Base / Perm `p`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean `decaps(dk, c)` (Full FIPS 203) | 2028.9 ± 200.5 | +0.0269 | -0.0113 | 51.89% SVM (RBF) / 51.68% / 51.89% / 0.1244 | All < 0 (max -0.0012) / 0.5373 | -0.0137 | 66.34% Logistic Regression / 66.34% / 0.5920 |
| Clean `_k_pke_decrypt(dk_pke, c)` (Isolated Core) | 565.4 ± 29.2 | +0.0234 | -0.0167 | 51.79% Logistic Regression / 51.68% / 51.79% / 0.2040 | All < 0 (max -0.0004) / 0.0299 | -0.0321 | 66.34% MLP / 66.34% / 0.3134 |
| `tracemalloc` `decaps(dk, c)` (2-feat) | 20945.6 ± 941.0 | -0.0364 | -0.0382 | 51.51% Logistic Regression / 51.68% / 51.51% / 0.0398 | All < 0 (max -0.0000) / 0.0945 | +0.0014 | 66.32% SVM (RBF) / 66.34% / 0.1443 |
| `tracemalloc` `decaps(dk, c)` (4-feat) | 20945.6 ± 941.0 | -0.0364 | -0.0382 | 51.85% SVM (RBF) / 51.68% / 51.85% / 0.4229 | All < 0 (max -0.0004) / 0.0846 | +0.0014 | 66.32% Logistic Regression / 66.34% / 0.9900 |

## 3. Primary Clean `decaps(dk, c)` Detailed Results

### Correlations vs. Secret Targets (2,000-Resample Bootstrap 95% CIs)
| Signal | Target | Pearson `r` [95% CI] | Pearson `p` | Spearman `ρ` [95% CI] | Spearman `p` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `dec_timing_us` | `secret_weight` | +0.0269 [-0.0022, +0.0538] | 5.71e-02 | -0.0113 [-0.0381, +0.0168] | 4.23e-01 |
| `dec_timing_us` | `s00_value` | -0.0009 [-0.0211, +0.0202] | 9.47e-01 | -0.0142 [-0.0409, +0.0132] | 3.15e-01 |
| `dec_timing_us` | `s00_positive` | -0.0026 [-0.0281, +0.0269] | 8.56e-01 | -0.0137 [-0.0411, +0.0133] | 3.32e-01 |
| `dec_cpu_user_us` | `secret_weight` | +0.0037 [-0.0233, +0.0284] | 7.93e-01 | -0.0117 [-0.0392, +0.0169] | 4.09e-01 |
| `dec_cpu_user_us` | `s00_value` | -0.0027 [-0.0266, +0.0242] | 8.49e-01 | -0.0142 [-0.0414, +0.0130] | 3.16e-01 |
| `dec_cpu_user_us` | `s00_positive` | -0.0048 [-0.0296, +0.0231] | 7.32e-01 | -0.0132 [-0.0403, +0.0143] | 3.51e-01 |

### Confound (`q`) & Time-Drift Checks
- **`q` vs `secret_weight` / `s00_positive`:** `r = +0.0000` (`p = 1.0000`) / `r = +0.0000` (`p = 1.0000`)
- **`q` vs `dec_timing_us` / `dec_cpu_user_us`:** `r = +0.0000` (`p = 1.0000`) / `r = +0.0000` (`p = 1.0000`)
- **Execution order vs `dec_timing_us` / `secret_weight`:** `r = -0.1224` (`p = 0.0000`) / `r = -0.0037` (`p = 0.7951`)

### Target 1: `secret_weight` Classification (5×10 CV, Majority Baseline = 51.68%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 50.04% ± 1.35% | 0.4980 | 50.47% ± 1.47% | -0.43% | 50.04% ± 1.35% | 0.2687 |
| Gradient Boosting | 50.48% ± 1.25% | 0.4979 | 50.87% ± 1.12% | -0.39% | 50.48% ± 1.25% | 0.6965 |
| MLP | 51.66% ± 0.71% | 0.5118 | 51.48% ± 0.62% | +0.18% | 51.66% ± 0.71% | 0.0597 |
| Logistic Regression | 51.86% ± 0.39% | 0.4880 | 51.68% ± 0.14% | +0.19% | 51.86% ± 0.39% | 0.2687 |
| SVM (RBF) | 51.89% ± 0.30% | 0.5046 | 51.60% ± 0.32% | +0.29% | 51.89% ± 0.30% | 0.1244 |

### Target 1: Exact `secret_weight` Regression (5×10 CV)
| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | `q`-Residualized R² | Perm `p`-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random Forest | -0.3770 ± 0.0436 | 12.3339 | -0.4036 ± 0.0470 | -0.3770 ± 0.0436 | 0.3731 | Negative R² (worse than mean baseline) |
| Gradient Boosting | -0.0195 ± 0.0082 | 10.6151 | -0.0174 ± 0.0092 | -0.0195 ± 0.0082 | 0.7463 | Negative R² (worse than mean baseline) |
| MLP | -0.0351 ± 0.0393 | 10.6939 | -0.0411 ± 0.0507 | -0.0351 ± 0.0393 | 0.1045 | Negative R² (worse than mean baseline) |
| Linear Regression | -0.0012 ± 0.0026 | 10.5193 | -0.0016 ± 0.0018 | -0.0012 ± 0.0026 | 0.5373 | Negative R² (worse than mean baseline) |
| SVR (RBF) | -0.0021 ± 0.0040 | 10.5237 | -0.0027 ± 0.0027 | -0.0021 ± 0.0040 | 0.2587 | Negative R² (worse than mean baseline) |

### Target 2a: `s[0][0] > 0` Sign Classification (5×10 CV, Majority Baseline = 66.34%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 56.06% ± 1.21% | 0.5042 | 56.89% ± 1.28% | -0.83% | 56.06% ± 1.21% | 0.3383 |
| Gradient Boosting | 65.69% ± 0.49% | 0.5039 | 65.59% ± 0.44% | +0.09% | 65.69% ± 0.49% | 0.7662 |
| MLP | 66.32% ± 0.09% | 0.5050 | 66.29% ± 0.11% | +0.03% | 66.32% ± 0.09% | 0.7413 |
| Logistic Regression | 66.34% ± 0.05% | 0.4984 | 66.31% ± 0.09% | +0.03% | 66.34% ± 0.05% | 0.5920 |
| SVM (RBF) | 66.31% ± 0.11% | 0.4907 | 66.32% ± 0.10% | -0.02% | 66.31% ± 0.11% | 0.3085 |

### Target 2b: Balanced Non-Zero `sign(s[0][0])` (`s[0][0] != 0`, n=3458, Baseline = 51.33%)
- **Clean `decaps` Best (Gradient Boosting):** Real `51.52% ± 1.89%` (AUC `0.5088`, Shuffled `50.81%`, `q`-Resid `51.52%`, Perm `p 0.0398`)
- **Isolated `_k_pke_decrypt` Best (MLP):** Real `51.84% ± 1.33%` (AUC `0.5157`, Perm `p 0.0697`)

> *Notes:* Permutation `p`-values at the `N=200` floor are reported as `< 1/201 (0.0050)`. Negative-`R²` models perform worse than predicting the sample mean (`R² = 0`).
