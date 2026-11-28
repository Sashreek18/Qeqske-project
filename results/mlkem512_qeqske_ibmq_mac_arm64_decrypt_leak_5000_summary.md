# `QEQSKE-Variant ML-KEM-512 (QRNG=ibmq_qrng_uint16.txt, n=256, k=2, per-key NTT q)` Decrypt Side-Channel Evaluation (5,000 Trials)

## 1. Experiment Setup & Secret Distribution
- **Environment:** `Darwin 27.0.0` (`macOS-27.0.1-arm64-arm-64bit-Mach-O`), CPU `Apple M2`, Python `3.13.1`, NumPy `2.2.6`
- **Parameters:** `n=256`, `k=2`, `eta_1=3`, `eta_2=2`, `du=10`, `dv=4`, `d_poly=16`, seed `42`
- **Correctness:** Clean `decaps` `5000/5000` (100.00%); `tracemalloc` `decaps` `5000/5000` (100.00%)
- **Target 1 (`secret_weight`):** non-zero entries out of `512` centered coeffs — range `311..385`, mean `352.12 ± 10.41`, binary `> 352.0` (class 0: `2580`, class 1: `2420`, majority baseline **51.60%**)
- **Target 2a (`s00_positive`: `s[0][0] > 0`):** all `5000` trials — class 0 (`<=0`): `3296`, class 1 (`>0`): `1704`, majority baseline **65.92%**
- **Target 2b (`s00_sign_nonzero`: `s[0][0] > 0` on `s[0][0] != 0`):** `3399` trials — negative: `1695`, positive: `1704`, majority baseline **50.13%**
- **NTT Sparsity Note:** Mean zero coefficients in `s_hat` = `0.03 / 512` (vs. `159.88` zeros in centered `s`).

## 2. Cross-Pass Side-Channel Summary (Weight & Coefficient Sign)

| Execution Pass | Timing Mean ± SD (µs) | `r_weight` | `ρ_weight` | Best Weight Clf (Real / Base / `q`-Res / Perm `p`) | Best Weight Reg `R²` (Perm `p`) | `ρ_s00_pos` | Best `s00 > 0` Clf (Real / Base / Perm `p`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean `decaps(dk, c)` (Full FIPS 203) | 2559.7 ± 80.7 | -0.0164 | -0.0123 | 51.65% SVM (RBF) / 51.60% / 51.07% / 0.9552 | All < 0 (max -0.0017) / 0.1542 | -0.0101 | 65.90% MLP / 65.92% / 0.8856 |
| Clean `_k_pke_decrypt(dk_pke, c)` (Isolated Core) | 724.6 ± 24.3 | +0.0024 | -0.0080 | 51.54% SVM (RBF) / 51.60% / 51.01% / 0.6517 | All < 0 (max -0.0019) / 0.6368 | -0.0022 | 65.92% MLP / 65.92% / 0.9751 |
| `tracemalloc` `decaps(dk, c)` (2-feat) | 21930.2 ± 1267.1 | -0.0222 | -0.0115 | 51.59% Logistic Regression / 51.60% / 51.56% / 0.4428 | All < 0 (max -0.0010) / 0.0249 | -0.0110 | 65.92% MLP / 65.92% / 0.6468 |
| `tracemalloc` `decaps(dk, c)` (4-feat) | 21930.2 ± 1267.1 | -0.0222 | -0.0115 | 51.46% Logistic Regression / 51.60% / 51.14% / 0.3134 | All < 0 (max -0.0018) / 0.1194 | -0.0110 | 65.92% SVM (RBF) / 65.92% / 0.9950 |

## 3. Primary Clean `decaps(dk, c)` Detailed Results

### Correlations vs. Secret Targets (2,000-Resample Bootstrap 95% CIs)
| Signal | Target | Pearson `r` [95% CI] | Pearson `p` | Spearman `ρ` [95% CI] | Spearman `p` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `dec_timing_us` | `secret_weight` | -0.0164 [-0.0450, +0.0113] | 2.47e-01 | -0.0123 [-0.0413, +0.0152] | 3.85e-01 |
| `dec_timing_us` | `s00_value` | +0.0001 [-0.0253, +0.0274] | 9.92e-01 | -0.0059 [-0.0335, +0.0236] | 6.78e-01 |
| `dec_timing_us` | `s00_positive` | -0.0058 [-0.0338, +0.0233] | 6.82e-01 | -0.0101 [-0.0377, +0.0189] | 4.75e-01 |
| `dec_cpu_user_us` | `secret_weight` | -0.0179 [-0.0469, +0.0097] | 2.05e-01 | -0.0122 [-0.0408, +0.0150] | 3.90e-01 |
| `dec_cpu_user_us` | `s00_value` | -0.0015 [-0.0269, +0.0262] | 9.14e-01 | -0.0056 [-0.0332, +0.0238] | 6.91e-01 |
| `dec_cpu_user_us` | `s00_positive` | -0.0090 [-0.0368, +0.0206] | 5.25e-01 | -0.0099 [-0.0376, +0.0189] | 4.82e-01 |

### Confound (`q`) & Time-Drift Checks
- **`q` vs `secret_weight` / `s00_positive`:** `r = -0.0139` (`p = 0.3264`) / `r = -0.0150` (`p = 0.2879`)
- **`q` vs `dec_timing_us` / `dec_cpu_user_us`:** `r = +0.4151` (`p = 0.0000`) / `r = +0.4375` (`p = 0.0000`)
- **Execution order vs `dec_timing_us` / `secret_weight`:** `r = -0.0736` (`p = 0.0000`) / `r = -0.0012` (`p = 0.9335`)

### Target 1: `secret_weight` Classification (5×10 CV, Majority Baseline = 51.60%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 49.04% ± 1.45% | 0.4849 | 50.56% ± 1.56% | -1.52% | 49.30% ± 1.27% | 0.7811 |
| Gradient Boosting | 51.31% ± 1.39% | 0.5088 | 50.50% ± 1.10% | +0.81% | 50.66% ± 1.46% | 0.3085 |
| MLP | 51.52% ± 0.88% | 0.5040 | 51.17% ± 0.98% | +0.34% | 51.18% ± 0.89% | 0.9254 |
| Logistic Regression | 51.62% ± 0.37% | 0.5045 | 51.44% ± 0.61% | +0.18% | 51.57% ± 0.18% | 0.2488 |
| SVM (RBF) | 51.65% ± 0.83% | 0.5023 | 51.37% ± 0.58% | +0.28% | 51.07% ± 0.64% | 0.9552 |

### Target 1: Exact `secret_weight` Regression (5×10 CV)
| Model | Real R² (Mean ± SD) | Real RMSE | Shuffled R² | `q`-Residualized R² | Perm `p`-value | Caveat |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Random Forest | -0.3513 ± 0.0420 | 12.0882 | -0.3372 ± 0.0417 | -0.2655 ± 0.0302 | 0.7313 | Negative R² (worse than mean baseline) |
| Gradient Boosting | -0.0161 ± 0.0082 | 10.4840 | -0.0217 ± 0.0094 | -0.0201 ± 0.0097 | 0.6318 | Negative R² (worse than mean baseline) |
| MLP | -0.0149 ± 0.0100 | 10.4780 | -0.0133 ± 0.0084 | -0.0143 ± 0.0098 | 0.5920 | Negative R² (worse than mean baseline) |
| Linear Regression | -0.0017 ± 0.0019 | 10.4098 | -0.0019 ± 0.0024 | -0.0018 ± 0.0020 | 0.1542 | Negative R² (worse than mean baseline) |
| SVR (RBF) | -0.0022 ± 0.0023 | 10.4123 | -0.0039 ± 0.0038 | -0.0022 ± 0.0022 | 0.8458 | Negative R² (worse than mean baseline) |

### Target 2a: `s[0][0] > 0` Sign Classification (5×10 CV, Majority Baseline = 65.92%)
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | `q`-Residualized Acc | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest | 56.97% ± 1.59% | 0.5161 | 55.08% ± 1.54% | +1.89% | 57.69% ± 1.30% | 0.0398 |
| Gradient Boosting | 65.16% ± 0.51% | 0.5131 | 65.07% ± 0.41% | +0.09% | 65.21% ± 0.43% | 0.5124 |
| MLP | 65.90% ± 0.07% | 0.5038 | 65.90% ± 0.06% | -0.01% | 65.90% ± 0.06% | 0.8856 |
| Logistic Regression | 65.90% ± 0.08% | 0.5043 | 65.90% ± 0.08% | -0.00% | 65.90% ± 0.08% | 0.8259 |
| SVM (RBF) | 65.89% ± 0.07% | 0.4956 | 65.92% ± 0.04% | -0.03% | 65.91% ± 0.05% | 0.9154 |

### Target 2b: Balanced Non-Zero `sign(s[0][0])` (`s[0][0] != 0`, n=3399, Baseline = 50.13%)
- **Clean `decaps` Best (Gradient Boosting):** Real `51.57% ± 1.75%` (AUC `0.5206`, Shuffled `49.18%`, `q`-Resid `49.64%`, Perm `p 0.2289`)
- **Isolated `_k_pke_decrypt` Best (Random Forest):** Real `51.35% ± 1.92%` (AUC `0.5116`, Perm `p 0.2239`)

> *Notes:* Permutation `p`-values at the `N=200` floor are reported as `< 1/201 (0.0050)`. Negative-`R²` models perform worse than predicting the sample mean (`R² = 0`).
