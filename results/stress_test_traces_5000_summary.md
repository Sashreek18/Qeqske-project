# Sender <-> Receiver Multi-Tier Side-Channel Evaluation Summary

## 1. Setup & Integrity
- **Input traces:** `results/sender_receiver_traces_5000.json` (`5000` trials, seed `42`)
- **Environment:** `Darwin 27.0.0`, CPU `Apple M2`, Python `3.13.1`, NumPy `2.2.6`
- **Target:** `secret_weight > 5.0` (class 0: `2685`, class 1: `2315`, majority baseline: **53.70%**)
- **Correctness:** local_decrypt: `4997/5000 (99.94%)`, tcp_original_ack: `4997/5000 (99.94%)`, tcp_ct_ack: `4997/5000 (99.94%)`

## 2. Signal Correlations vs. `secret_weight` (2,000-Resample Bootstrap 95% CIs)
| Signal | Mean ± SD (µs) | Median (IQR) µs | Pearson r [95% CI] | Spearman ρ [95% CI] | Spearman p |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `enc_timing_us` | 348.69 ± 19.06 | 341.75 (9.42) | +0.0034 [-0.0211, +0.0290] | -0.0006 [-0.0288, +0.0262] | 9.64e-01 |
| `enc_cpu_user_us` | 345.68 ± 13.53 | 342.00 (5.00) | +0.0094 [-0.0156, +0.0349] | -0.0065 [-0.0333, +0.0199] | 6.48e-01 |
| `dec_timing_us` | 75.02 ± 5.05 | 73.96 (3.08) | +0.1804 [+0.1496, +0.2106] | +0.3989 [+0.3754, +0.4244] | 2.55e-190 |
| `dec_cpu_user_us` | 74.87 ± 3.93 | 74.00 (3.00) | +0.2382 [+0.2088, +0.2691] | +0.4113 [+0.3883, +0.4350] | 1.89e-203 |
| `net_rtt_us` | 168.14 ± 25.04 | 155.83 (34.50) | +0.0249 [-0.0039, +0.0534] | +0.0700 [+0.0414, +0.0983] | 7.17e-07 |
| `net_ct_rtt_us` | 138.14 ± 24.33 | 134.37 (36.73) | -0.0134 [-0.0401, +0.0176] | -0.0033 [-0.0303, +0.0262] | 8.15e-01 |

## 3. Confound & Execution-Order Time-Drift Checks
- **`q` vs `secret_weight`:** Pearson `r = -0.0031` (`p = 0.8284`), Spearman `ρ = -0.0042` (`p = 0.7692`)
- **Order drift (`enc_exec_order_pos_vs_enc_timing_us`):** `r = +0.0320` (`p = 0.0238`)
- **Order drift (`enc_exec_order_pos_vs_secret_weight`):** `r = -0.0082` (`p = 0.5619`)
- **Order drift (`enc_exec_order_pos_vs_enc_cpu_user_us`):** `r = +0.0432` (`p = 0.0023`)
- **Order drift (`dec_exec_order_pos_vs_dec_timing_us`):** `r = +0.0607` (`p = 0.0000`)
- **Order drift (`dec_exec_order_pos_vs_secret_weight`):** `r = +0.0000` (`p = 0.9975`)
- **Order drift (`dec_exec_order_pos_vs_dec_cpu_user_us`):** `r = +0.0559` (`p = 0.0001`)
- **Order drift (`net_exec_order_pos_vs_net_rtt_us`):** `r = -0.0570` (`p = 0.0001`)
- **Order drift (`net_exec_order_pos_vs_secret_weight`):** `r = -0.0040` (`p = 0.7772`)
- **Order drift (`net_ct_exec_order_pos_vs_net_ct_rtt_us`):** `r = -0.0744` (`p = 0.0000`)
- **Order drift (`net_ct_exec_order_pos_vs_secret_weight`):** `r = +0.0154` (`p = 0.2762`)

## 4. Multi-Tier 5x10 Stratified CV Summary
Majority baseline: **53.70%** | Permutation test resolution floor (`N=200`): `< 1/201 (0.0050)`

| Tier | Active Features | Best Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized | Perm p-value | Verdict |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tier 1a: Sender Only (Clean)** | 2 | Logistic Reg | 53.68% ± 0.09% | 0.4900 | 53.69% | -0.01% | 53.68% | 0.3682 | NO SIGNAL (At Baseline / Indistinguishable from Shuffle) |
| **Tier 1b: Receiver Only (Clean Local)** | 2 | SVM (RBF) | 65.29% ± 0.90% | 0.7080 | 53.42% | +11.87% | 65.48% | < 1/201 (0.0050) | CONFIRMED SIGNAL (Perm p < 0.01, Robust to Shuffle & q) |
| **Tier 1c: Public Channel Only** | 24 | SVM (RBF) | 55.59% ± 1.52% | 0.5717 | 53.09% | +2.50% | 55.31% | < 1/201 (0.0050) | NO SIGNAL (At Baseline / Indistinguishable from Shuffle) |
| **Tier 1d: Network TCP RTT (Original decrypt_it)** | 2 | MLP | 55.67% ± 0.98% | 0.5683 | 53.56% | +2.11% | 55.21% | < 1/201 (0.0050) | NO SIGNAL (At Baseline / Indistinguishable from Shuffle) |
| **Tier 1e: Network TCP RTT (Constant-Time)** | 2 | SVM (RBF) | 53.92% ± 0.65% | 0.5019 | 53.61% | +0.31% | 53.63% | 0.0995 | NO SIGNAL (At Baseline / Indistinguishable from Shuffle) |
| **Tier 2: Sender + Receiver (Clean)** | 4 | MLP | 65.92% ± 1.08% | 0.7134 | 52.70% | +13.22% | 65.72% | < 1/201 (0.0050) | CONFIRMED SIGNAL (Perm p < 0.01, Robust to Shuffle & q) |
| **Tier 3: ALL Combined (Sender + Receiver + Channel + Net RTT)** | 30 | Grad Boost | 66.18% ± 1.01% | 0.7277 | 51.70% | +14.48% | 66.32% | < 1/201 (0.0050) | CONFIRMED SIGNAL (Perm p < 0.01, Robust to Shuffle & q) |

### Per-Model Breakdown: `Tier 1b: Receiver Only (Clean Local)`
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized Acc | Perm p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Reg | 60.93% ± 1.08% | 0.6919 | 53.69% ± 0.04% | +7.24% | 60.14% ± 1.09% | < 1/201 (0.0050) |
| Random Forest | 60.91% ± 1.25% | 0.6539 | 49.84% ± 1.30% | +11.08% | 60.22% ± 1.34% | < 1/201 (0.0050) |
| SVM (RBF) | 65.29% ± 0.90% | 0.7080 | 53.42% ± 0.35% | +11.87% | 65.48% ± 1.02% | < 1/201 (0.0050) |
| MLP | 65.21% ± 0.98% | 0.7152 | 53.59% ± 0.22% | +11.62% | 65.34% ± 1.13% | < 1/201 (0.0050) |
| Grad Boost | 64.90% ± 1.22% | 0.7090 | 52.19% ± 1.02% | +12.71% | 64.74% ± 1.12% | < 1/201 (0.0050) |

### Per-Model Breakdown: `Tier 1d: Network TCP RTT (Original decrypt_it)`
| Model | Real Acc (Mean ± SD) | ROC-AUC | Shuffled Control | Net Gap | q-Residualized Acc | Perm p-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Reg | 53.59% ± 0.54% | 0.5202 | 53.70% ± 0.00% | -0.11% | 53.67% ± 0.57% | 0.9851 |
| Random Forest | 52.44% ± 1.38% | 0.5290 | 51.94% ± 1.40% | +0.50% | 51.55% ± 1.41% | 0.0100 |
| SVM (RBF) | 53.75% ± 0.79% | 0.5286 | 53.56% ± 0.47% | +0.20% | 53.59% ± 0.92% | 0.8259 |
| MLP | 55.67% ± 0.98% | 0.5683 | 53.56% ± 0.52% | +2.11% | 55.21% ± 1.24% | < 1/201 (0.0050) |
| Grad Boost | 55.51% ± 1.40% | 0.5774 | 53.45% ± 1.24% | +2.06% | 55.37% ± 1.24% | < 1/201 (0.0050) |
