# Cross-Scale Comparison: Toy QEQSKE vs. Real ML-KEM (`n=256`, `kyber-py`) vs. QEQSKE-Variant ML-KEM

| System | Parameters & `q` | Clean Timing | Clean `ρ_weight` | Clean Weight Clf vs. Baseline | Clean Weight Reg `R²` | Isolated `_k_pke_decrypt` | `tracemalloc` `decaps` | `s[0][0] > 0` Clf vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Toy QEQSKE (`decrypt_it`) | n=4, k=2, per-key prime q | 75.21 ± 6.37 µs | +0.4166 | 65.25% vs 53.70% (MLP, p < 1/201 (0.0050)) | +0.2260 (SVR (RBF)) | N/A (no FO layer) | 84.37% (Gradient Boosting, ρ=+0.8241) | N/A (at baseline in Prog. Ladder) |
| Standard ML-KEM (`kyber-py`, k=2) | n=256, k=2, fixed | 2097.0 ± 295.0 µs | +0.0147 | 52.12% vs 52.02% (SVM (RBF), p 0.1592) | All < 0 (max -0.0019) | 52.24% (MLP, ρ=+0.0006) | 51.84% (SVM (RBF), ρ=-0.0166) | 65.62% vs 65.62% (Logistic Regression, p 0.6716) |
| QEQSKE-Variant ML-KEM (QRNG + per-key q, k=2) | n=256, k=2, per_key_ntt_prime | 1983.0 ± 97.5 µs | +0.0120 | 51.09% vs 51.14% (Logistic Regression, p 0.2438) | All < 0 (max -0.0016) | 51.33% (SVM (RBF), ρ=-0.0098) | 51.05% (SVM (RBF), ρ=+0.0177) | 64.28% vs 64.30% (SVM (RBF), p 0.3085) |
