# ML-KEM-512 Execution-Order Detrending Comparison (5,000 Trials, Window=51)

Compares unmodified raw timing/CPU traces against leave-one-out centered rolling-median (`W=51`) and linear order-OLS detrended residuals on both Linux Intel Core i9-12900K datasets.

| Variant & Mode | Timing SD / IQR (µs) | Order Drift `r` (`ρ`) vs Timing | `ρ_weight` [95% CI] | Best 5×10 Clf (Acc / Base / `q`-Res) | Perm `p`-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard ML-KEM-512** — Raw | 295.0 / 55.0 | -0.1523 (-0.4668) | +0.0147 [-0.0134, +0.0424] | 52.11% SVM (RBF) / 52.02% / 52.11% | 0.1592 |
| **Standard ML-KEM-512** — Order-OLS Residuals | 291.6 / 53.8 | +0.0000 (+0.4998) | +0.0127 [-0.0148, +0.0422] | 52.04% SVM (RBF) / 52.02% / 52.04% | 0.0846 |
| **Standard ML-KEM-512** — Rolling Median (W=51) | 155.9 / 30.4 | -0.1023 (-0.0108) | +0.0335 [+0.0055, +0.0622] | 52.01% SVM (RBF) / 52.02% / 52.01% | 0.3284 |
| **QEQSKE-Variant ML-KEM-512** — Raw | 97.5 / 58.1 | +0.0776 (+0.0652) | +0.0120 [-0.0152, +0.0408] | 51.09% Logistic Regression / 51.14% / 51.12% | 0.2438 |
| **QEQSKE-Variant ML-KEM-512** — Order-OLS Residuals | 97.2 / 58.3 | +0.0000 (-0.1104) | +0.0097 [-0.0173, +0.0392] | 51.09% Logistic Regression / 51.14% / 51.13% | 0.2637 |
| **QEQSKE-Variant ML-KEM-512** — Rolling Median (W=51) | 95.5 / 53.2 | +0.0542 (+0.0114) | +0.0220 [-0.0052, +0.0499] | 51.03% Logistic Regression / 51.14% / 50.52% | 0.3731 |

## Randomness & QRNG Reuse Audit
- **Standard ML-KEM-512 (`kyber-py`):** Uses deterministic `SHAKE-256` (`seed=42`, `q=3329` fixed).
- **QEQSKE-Variant ML-KEM-512:** Consumed `245,000` uint16 draws from `qrng_combined.txt` (`1,106,944` values) -> **`wraps = 0` (zero QRNG reuse across all 5,000 trials)**.
