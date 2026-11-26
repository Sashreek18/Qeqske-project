# ML-KEM-512 (n=256, k=2) Graded Injected-Delay Positive Control (5,000 Trials)

- **Environment:** `Linux 7.0.0-31-generic` (`Linux-7.0.0-31-generic-x86_64-with-glibc2.39`), CPU `12th Gen Intel(R) Core(TM) i9-12900K`, Python `3.12.3`, NumPy `2.5.2`
- **Majority Baseline (`secret_weight > 352.0`):** **52.02%** | Rolling-median detrend window: `W=51`
- **Detection Capability:** On 12th Gen Intel(R) Core(TM) i9-12900K (Linux 7.0.0-31-generic), unmodified ML-KEM-512 decaps (2084.8 us median, IQR 41.3 us) reliably detects injected secret-dependent delays >= 5 us (Spearman rho_binary >= +0.0844, acc 54.36%, p < 1/201 (0.0050)); rolling-median detrending (W=51) lowers/matches threshold to >= 5 us (Spearman rho_binary >= +0.0830, acc 54.16%, p < 1/201 (0.0050)).

| Injected Delay (µs) | Raw Med / IQR (µs) | Raw Δt | Raw `ρ_bin` / `ρ_wt` | Raw Best Acc (Model / Perm `p`) | Detrended IQR (µs) | Detrended `ρ_bin` / `ρ_wt` | Detrended Best Acc (Model / Perm `p`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 µs** | 2084.8 / 41.3 | -0.24 µs | +0.0024 / +0.0038 | 52.06% (Logistic Regression / 0.2289) | 36.6 | +0.0063 / +0.0129 | 52.07% (Logistic Regression / 0.2040) |
| **1 µs** | 2068.8 / 34.3 | +1.40 µs | +0.0135 / +0.0098 | 52.15% (Logistic Regression / 0.8109) | 33.0 | +0.0134 / +0.0119 | 52.10% (SVM (RBF) / 0.1393) |
| **2 µs** | 2067.0 / 32.2 | +1.28 µs | +0.0285 / +0.0242 | 51.98% (Logistic Regression / 0.5522) | 31.3 | +0.0277 / +0.0209 | 51.91% (Logistic Regression / 0.5224) |
| **5 µs** | 2064.4 / 36.7 | +4.01 µs | +0.0844 / +0.0855 | 54.36% (MLP / < 1/201 (0.0050)) | 36.2 | +0.0830 / +0.0823 | 54.16% (MLP / < 1/201 (0.0050)) |
| **10 µs** | 2070.4 / 40.9 | +9.92 µs | +0.1708 / +0.1552 | 58.99% (SVM (RBF) / < 1/201 (0.0050)) | 39.0 | +0.1746 / +0.1611 | 59.56% (SVM (RBF) / < 1/201 (0.0050)) |
| **20 µs** | 2075.2 / 46.3 | +20.89 µs | +0.3162 / +0.2771 | 68.46% (SVM (RBF) / < 1/201 (0.0050)) | 39.5 | +0.3258 / +0.2798 | 68.65% (Gradient Boosting / < 1/201 (0.0050)) |
| **50 µs** | 2098.6 / 62.0 | +50.71 µs | +0.5845 / +0.5032 | 88.29% (SVM (RBF) / < 1/201 (0.0050)) | 62.2 | +0.5786 / +0.4999 | 87.40% (MLP / < 1/201 (0.0050)) |
