# Machine Learning and Side-Channel Leakage Evaluation of Post-Quantum Key Exchange Protocols

A comprehensive empirical evaluation framework analyzing physical execution telemetry, network-observable communication traces, and entropy sources in post-quantum key exchange, using **QEQSKE** (*Quantum-Enabled Quantum-Safe Key Exchange*, Ghosh et al., HCL Software) as a representative base algorithm.

- **Course:** CSD493 Final-Year Capstone Project, Shiv Nadar University
- **Faculty Supervisor:** Prof. Rajib Mall
- **Industry Partner:** HCL Software
- **Team:** Sri Surya Srikar, Sashreek Kandagatla, Sane Yateesh Reddy
- **Dataset:** 1,106,944 true quantum random numbers from the Australian National University (ANU) Quantum Optics QRNG API

---

## 1. Research Question & Academic Framing

Following supervisory guidance from Prof. Rajib Mall, the objective of this research is **not** to algebraically break the underlying encryption scheme, but rather to establish a **generalizable evaluation methodology**:

> **"Can modern machine learning classifiers and side-channel analysis detect information leakage across physical execution traces and communication messages in an end-to-end post-quantum key exchange session, and how can implementations be hardened against such leakage?"**

We evaluate an end-to-end Alice $\leftrightarrow$ Bob key exchange across three distinct physical and observational boundaries:
1. **The Sender (Bob):** Physical side-channel leakage during encapsulation (`encrypt_it`).
2. **The Public Channel (Eavesdropper):** Statistical moments, matrix norms, and distributions of transmitted ciphertexts ($\mathbf{u}, \mathbf{v}$).
3. **The Receiver (Alice):** Physical side-channel leakage during decapsulation (`decrypt_it`).

```
[Sender (Bob)]                [Public Communication Channel]               [Receiver (Alice)]
  encrypt_it()        ───►        Ciphertexts (u, v) + q         ───►        decrypt_it()
  • Latency                   • Matrix Frobenius norms                     • Latency
  • CPU user-time             • Moments (mean, std, skew, kurt)            • CPU user-time
  • RAM / Allocs              • Entropy approximations                     • RAM / Allocs
       │                                     │                                  │
       ▼                                     ▼                                  ▼
   TIER 1a:                               TIER 1c:                           TIER 1b:
  Sender Only                           Channel Only                       Receiver Only
 (53.7% | NO LEAK)                    (52.8% | NO LEAK)                  (82.2% | LEAK DETECTED)
```

---

## 2. Key Empirical Findings

*Baseline Accuracy (Majority Class Guess): **55.40%***

| Observational Tier | Features Profiled | Best ML Classifier | Real CV Accuracy | Shuffled Control | Net Gap | Leakage Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Tier 1a: Sender Only** | 4 | Logistic Regression | $53.68\% \pm 2.0\%$ | $53.86\%$ | $-0.18\%$ | **Zero Leakage (Secure)** |
| **Tier 1c: Channel Only** | 24 | SVM (RBF) | $52.84\% \pm 3.3\%$ | $51.18\%$ | $+1.66\%$ | **Zero Leakage (Secure)** |
| **Tier 1b: Receiver Only** | 4 | Gradient Boosting | $\mathbf{82.18\% \pm 3.4\%}$ | $\mathbf{51.84\%}$ | $\mathbf{+30.34\%}$ | **Severe Side-Channel Leakage ⚠️** |
| **Tier 2: Sender + Receiver** | 8 | Gradient Boosting | $\mathbf{81.98\% \pm 3.0\%}$ | $\mathbf{53.36\%}$ | $\mathbf{+28.62\%}$ | **Leakage Confirmed (Receiver-driven)** |
| **Tier 3: ALL Combined** | 32 | Gradient Boosting | $\mathbf{81.96\% \pm 3.7\%}$ | $\mathbf{48.98\%}$ | $\mathbf{+32.98\%}$ | **Leakage Confirmed (Receiver-driven)** |
| **Attack F Ext: Timing Alone** | 5 | SVM (RBF) | $\mathbf{68.94\% \pm 3.1\%}$ | $\mathbf{53.15\%}$ | $\mathbf{+15.79\%}$ | **Timing Side-Channel Confirmed** |

### Takeaways
1. **Mathematical Soundness of the Wire (Negative Result):** Transmitted ciphertexts exhibit zero exploitable ML leakage ($52.84\%$ vs. $51.18\%$ shuffled control). Eavesdroppers recording traffic cannot recover secret keys or message bits via ML.
2. **Physical Endpoint Vulnerability (Positive Result):** Software decapsulation exhibits an $82.18\%$ side-channel leakage vector. Feature-importance analysis confirms the signal is carried almost entirely by execution latency (`dec_timing_us`), traced to variable-time modular rounding branches.

---

## 3. Scientific Validation & Stress-Tests (`stress_test_traces.py`)

To ensure that the $82.2\%$ discovery was neither an artifact of sample size nor an external confound, we subjected the pipeline to three rigorous stress-tests:

1. **Permutation-Shuffled Label Control:**
   Randomly permuting the secret-key labels collapsed classifier accuracy to the chance floor ($51.84\%$), establishing a true $+30.34\%$ net accuracy gap for real labels.
2. **Modulus $q$ Confound Regression:**
   Deriving a variable prime $q$ introduces no confound into the secret key ($r = +0.0384, p = 0.3911$). Regressing out $q$ from every feature ($q$-residualization) left Receiver accuracy unaffected ($82.68\%$).
3. **50-Fold Cross-Validation Spread:**
   Evaluated using `RepeatedStratifiedKFold` (5 splits × 10 repeats = 50 folds). Accuracy variance was tightly bounded ($\pm 3.38\%$), with the worst single fold achieving $75.0\%$ (well above the $55.4\%$ baseline).

---

## 4. Repository Structure

### Core Protocol & Randomness
- `qeqske_core.py`: Implementation of Algorithms 1–10. Includes fix for pad reuse, constant-time modular decapsulation, and hardened re-encryption verification.
- `qrng_handler.py`: QRNG streaming engine backed by ANU Quantum Optics data with $O(1)$ deque consumption.
- `test_qeqske.py`: End-to-end Alice $\leftrightarrow$ Bob key exchange verification.

### Primary Experimental Suites
- `attack_sc_sender_receiver_traces.py`: Full Alice $\leftrightarrow$ Bob simulation capturing 34-dimensional telemetry (timing, CPU, memory, matrix moments).
- `attack_sc_ml_classifiers.py`: Multi-tier ML evaluation comparing 5 model families (GBDT, RF, SVM, MLP, LR).
- `stress_test_traces.py`: Confound regression, 50-fold variance reporting, and shuffled-label permutation controls.
- `attack_h_trace_leakage.py`: Eavesdropper wire traffic ML evaluation + ML timing classifier on Attack F data.
- `attack_c_decrypt_sidechannel.py`: Decryption endpoint profiling resolving the Section 9.5 midterm limitation.

### Baseline & Sequence Models
- `attack_a_distinguisher.py` & `attack_a_levels_2_3.py`: Distinguisher tests comparing QRNG vs. Mersenne Twister.
- `attack_a_sequence.py`: Deep sequence models (LSTM, GRU, Transformer) evaluating draw order.
- `nn_from_scratch.py`: Pure NumPy implementation of LSTM, GRU, and Transformer with analytical BPTT.
- `attack_b_entropy_leakage.py`: Autoencoder compression and reconstruction error tests (ratio = 1.004).
- `attack_f_timing_sidechannel.py`: Decryption timing side-channel analysis (Cohen's $d = 0.224$).
- `attack_g_modulo_bias.py`: Modulo bias detection and rejection sampling validation ($\chi^2 = 2055.8$).

### Appendix
- `attack_d_cca_mismatch.py`: Auxiliary observation documenting active chosen-ciphertext behavior in the absence of an FO transform (scoped outside the core thesis).
- `break_poc.py`: Demonstrations of initial self-audit findings (pad reuse in legacy mode and toy parameter brute force).

---

## 5. Countermeasures & Hardening

1. **Constant-Time Modular Arithmetic:**  
   Replaced branch-dependent distance comparisons (`0 if dist_to_0 < dist_to_half else 1`) with constant-time bitwise selection masks in `qeqske_core.py`, eliminating execution timing discrepancies across coefficient values.
2. **Hardened Re-Encryption Verification:**  
   Implemented `encrypt_it_hardened()` and `decrypt_it_hardened()`, enforcing mandatory constant-time tag verification ($\text{tag} = \text{SHA-256}(m \parallel u \parallel v)$).

---

## 6. How to Run the Suites

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Run full sender-receiver trace collection (500 full sessions)
python3 attack_sc_sender_receiver_traces.py

# 2. Run multi-tier ML evaluation
python3 attack_sc_ml_classifiers.py

# 3. Run scientific stress-tests (Shuffled control, CV spread, Confound regression)
python3 stress_test_traces.py

# 4. Run wire eavesdropper and timing classifier suite
python3 attack_h_trace_leakage.py
```
