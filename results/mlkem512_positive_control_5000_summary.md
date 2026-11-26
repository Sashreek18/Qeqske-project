# ML-KEM-512 (n=256, k=2) Graded Injected-Delay Positive Control (5,000 Trials)

- **Environment:** `Darwin 27.0.0` (`macOS-27.0.1-arm64-arm-64bit-Mach-O`), CPU `Apple M2`, Python `3.13.1`, NumPy `2.2.6`
- **Majority Baseline (`secret_weight > 352.0`):** **52.02%** | Rolling-median detrend window: `W=51`
- **Detection Capability:** On Apple M2 (Darwin 27.0.0), unmodified ML-KEM-512 decaps (2382.3 us median, IQR 16.3 us) reliably detects injected secret-dependent delays >= 2 us (Spearman rho_binary >= +0.1022, acc 54.10%, p < 1/201 (0.0050)); rolling-median detrending (W=51) lowers/matches threshold to >= 2 us (Spearman rho_binary >= +0.1069, acc 55.40%, p < 1/201 (0.0050)).

| Injected Delay (µs) | Raw Med / IQR (µs) | Raw Δt | Raw `ρ_bin` / `ρ_wt` | Raw Best Acc (Model / Perm `p`) | Detrended IQR (µs) | Detrended `ρ_bin` / `ρ_wt` | Detrended Best Acc (Model / Perm `p`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 µs** | 2382.3 / 16.3 | -0.12 µs | -0.0100 / -0.0078 | 51.98% (Logistic Regression / 0.9204) | 15.6 | -0.0094 / -0.0070 | 51.99% (Logistic Regression / 0.8856) |
| **1 µs** | 2381.1 / 16.6 | +1.17 µs | +0.0419 / +0.0450 | 52.51% (MLP / 0.0348) | 15.9 | +0.0396 / +0.0421 | 52.39% (MLP / 0.0249) |
| **2 µs** | 2371.6 / 16.0 | +2.50 µs | +0.1022 / +0.0951 | 54.10% (MLP / < 1/201 (0.0050)) | 14.8 | +0.1069 / +0.0972 | 55.40% (SVM (RBF) / < 1/201 (0.0050)) |
| **5 µs** | 2372.7 / 15.6 | +5.25 µs | +0.2226 / +0.1914 | 61.04% (MLP / < 1/201 (0.0050)) | 15.5 | +0.2242 / +0.1930 | 60.96% (MLP / < 1/201 (0.0050)) |
| **10 µs** | 2372.7 / 16.7 | +10.08 µs | +0.4136 / +0.3694 | 71.87% (MLP / < 1/201 (0.0050)) | 16.8 | +0.4056 / +0.3606 | 70.98% (MLP / < 1/201 (0.0050)) |
| **20 µs** | 2364.0 / 22.4 | +19.58 µs | +0.6445 / +0.5506 | 86.16% (MLP / < 1/201 (0.0050)) | 22.5 | +0.6297 / +0.5402 | 84.57% (MLP / < 1/201 (0.0050)) |
| **50 µs** | 2389.9 / 50.1 | +49.54 µs | +0.8352 / +0.7229 | 98.65% (MLP / < 1/201 (0.0050)) | 49.7 | +0.8117 / +0.7048 | 94.53% (MLP / < 1/201 (0.0050)) |
