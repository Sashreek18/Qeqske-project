# Cross-Scale Comparison: Toy QEQSKE vs. Real ML-KEM (`n=256`, `kyber-py`) vs. QEQSKE-Variant ML-KEM

> **Seed Note:** The first Toy QEQSKE row below is from the seed-42 run (`results/decrypt_leak_5000.json`), whereas the ML-KEM rows are from seed-1337, and the adjacent matching-seed Toy QEQSKE row is from `results/decrypt_leak_5000_linux_i9_seed1337.json` (seed-1337).

| System | Parameters & `q` | Clean Timing | Clean `ρ_weight` | Clean Weight Clf vs. Baseline | Clean Weight Reg `R²` | Isolated `_k_pke_decrypt` | `tracemalloc` `decaps` | `s[0][0] > 0` Clf vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Toy QEQSKE (`decrypt_it`, seed=42, `results/decrypt_leak_5000.json`) | n=4, k=2, per-key prime q | 75.21 ± 6.37 µs | +0.4166 | 65.25% vs 53.70% (MLP, p < 1/201 (0.0050)) | +0.2260 (SVR (RBF)) | N/A (no FO layer) | 84.37% (Gradient Boosting, ρ=+0.8241) | N/A (at baseline in Prog. Ladder) |
| Toy QEQSKE (`decrypt_it`, seed=1337, `results/decrypt_leak_5000_linux_i9_seed1337.json`) | n=4, k=2, per-key prime q | 72.78 ± 5.90 µs | +0.4099 | 65.92% vs 51.54% (MLP, p < 1/201 (0.0050)) | +0.2070 (MLP) | N/A (no FO layer) | 86.33% (Logistic Regression, ρ=+0.8615) | N/A (at baseline in Prog. Ladder) |
| Standard ML-KEM (`kyber-py`, k=2, seed=1337) | n=256, k=2, fixed | 2028.9 ± 200.5 µs | -0.0113 | 51.89% vs 51.68% (SVM (RBF), p 0.1244) | All < 0 (max -0.0012) | 51.79% (Logistic Regression, ρ=-0.0167) | 51.51% (Logistic Regression, ρ=-0.0382) | 66.34% vs 66.34% (Logistic Regression, p 0.5920) |
| QEQSKE-Variant ML-KEM (QRNG + per-key q, k=2, seed=1337) | n=256, k=2, per_key_ntt_prime | 2006.7 ± 79.3 µs | -0.0120 | 52.52% vs 51.94% (MLP, p 0.0697) | All < 0 (max -0.0010) | 51.88% (Logistic Regression, ρ=-0.0179) | 51.62% (Logistic Regression, ρ=+0.0241) | 65.38% vs 65.38% (Logistic Regression, p 0.5323) |
