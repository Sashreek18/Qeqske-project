
######################################################################
# STEP 1: QEQSKE Correctness Check
######################################################################
Loaded 1024 real QRNG numbers from your batch.

============================================================
STEP 1: Alice generates keys (Algorithm 6/7)
============================================================
  Prime modulus q (from QRNG)       = 38933
  Public key A (shape 4x4)        = generated
  Public key t (shape 4x2)        = generated
  Private key s (shape 4x2, secret)= generated
  QRNG numbers used so far           = 21
  QRNG numbers remaining              = 1003

============================================================
STEP 2: Bob encrypts shared secret: 'Hi' (Algorithm 8/9)
============================================================
  Message bits encrypted             = 16 bits
  Ciphertext component u (shape 2x4)= generated
  Ciphertext component v_queue        = 16 matrices
  QRNG numbers used so far           = 405
  QRNG numbers remaining              = 619

============================================================
STEP 3: Alice decrypts ciphertext (Algorithm 10)
============================================================
  Original message                    = 'Hi'
  Recovered message                   = 'Hi'
  MATCH                               = True


✅ STEP 1 PASSED — Key exchange works correctly with real QRNG numbers

######################################################################
# STEP 2: Statistical Tests at Scale
######################################################################
Generating Mersenne Twister baseline (1106944 numbers)...
Done

Loaded 1106944 real QRNG numbers from qrng_combined.txt


============================================================
Statistical Test Results: QRNG Data (1106944 numbers)
============================================================
  [PASS] Frequency (Monobit)       {'p_value': 0.0363}
  [PASS] Runs                      {'p_value': 0.0192}
  [PASS] Longest Run               {'longest_run': 23, 'expected_approx': 24.08}
  [PASS] Chi-Square Uniformity     {'chi_square': 26.52, 'p_value': 0.0329}
  [PASS] Serial Correlation        {'correlation': -0.0009}

  Summary: 5/5 tests passed

============================================================
Statistical Test Results: Mersenne Twister (1106944 numbers)
============================================================
  [PASS] Frequency (Monobit)       {'p_value': 0.0438}
  [PASS] Runs                      {'p_value': 0.325}
  [PASS] Longest Run               {'longest_run': 23, 'expected_approx': 24.08}
  [PASS] Chi-Square Uniformity     {'chi_square': 14.46, 'p_value': 0.491}
  [PASS] Serial Correlation        {'correlation': 0.0002}

  Summary: 5/5 tests passed

######################################################################
# STEP 3: Attack A — ML Distinguisher at Scale
######################################################################
Dataset built: 276736 samples, 15 features per sample
  QRNG samples: 138368
  Mersenne Twister samples: 138368
######################################################################
ATTACK A — Levels 2 and 3
######################################################################

Collecting QEQSKE outputs fed by real QRNG (300 trials)...
Collecting QEQSKE outputs fed by Mersenne Twister (300 trials)...

======================================================================
LEVEL 2 — QEQSKE-derived output comparison (QRNG-fed vs MT-fed)
======================================================================
  Field              QRNG mean     MT mean  KS p-value  Conclusion
  q                 29286.9533  32515.8333      0.0125  same distribution
  Bonferroni-corrected alpha = 0.05/5 = 0.010 (5 simultaneous tests)
  secret_mean          -0.0138      0.0054      0.9993  same distribution
  Bonferroni-corrected alpha = 0.05/5 = 0.010 (5 simultaneous tests)
  secret_weight         5.3167      5.3233      1.0000  same distribution
  Bonferroni-corrected alpha = 0.05/5 = 0.010 (5 simultaneous tests)
  t_mean                0.5061      0.5028      0.8483  same distribution
  Bonferroni-corrected alpha = 0.05/5 = 0.010 (5 simultaneous tests)
  t_std                 0.2728      0.2628      0.2488  same distribution
  Bonferroni-corrected alpha = 0.05/5 = 0.010 (5 simultaneous tests)

======================================================================
LEVEL 3 — Adversarial prediction from PUBLIC outputs only (no side-channel)
======================================================================
  Baseline (majority-class) = 56.33%
  Best model accuracy       = 54.60% +/- 5.12%
  Margin over baseline      = -1.73 points
  NOTE: at n=4, k=2 the secret is recoverable from public (A, t) by brute
  force in under a millisecond (see break_poc.py). A null result here therefore
  reflects these ML models, not LWE hardness.
  assumption underlying Kyber/QEQSKE at our toy parameters (n=4, k=2).

Done.

######################################################################
ATTACK C-4: Combined Side-Channel Dataset Collection
######################################################################
Collected 500 combined trials (Time + RAM + CPU measured together per secret key).

====================================================================
Statistical Significance — each signal vs. secret_weight
====================================================================
  Signal               Pearson r     p-value  Conclusion
  timing_us               0.0731      0.1024  not significant
  cpu_user_us             0.0678      0.1301  not significant
  ram_peak_bytes          0.0960      0.0319  not significant
  num_allocations         0.0997      0.0257  not significant

====================================================================
Combined ML Attack — predicting above/below-median secret weight
====================================================================
  Dataset: 500 trials -> Repeated Cross-Validation (5 splits, 10 repeats)
  Features: timing_us, cpu_user_us, ram_peak_bytes, num_allocations
  Random/majority-class baseline accuracy = 55.00%

  Model                     Accuracy  Precision   Recall      F1
  Logistic Regression         55.38%     55.33%    2.18%   4.14%
  Random Forest               48.20%     42.20%   41.20%  41.50%
  SVM (RBF)                   55.54%     56.63%    5.69%  10.13%
  Neural Network (MLP)        52.94%     46.18%   11.73%  16.77%

  Random Forest feature importance (which signal matters most):
    timing_us          0.9421  #####################################
    ram_peak_bytes     0.0269  #
    num_allocations    0.0251  #
    cpu_user_us        0.0059  

  Best model beat baseline by +0.54 percentage points.
  Combined-signal leakage risk classification: LOW

  NOTE: this predicts a derived property (above/below-median secret
  weight), NOT the secret key itself -- recovering the full key from
  software side-channels alone is not a realistic or claimed outcome.
######################################################################
ATTACK C — Progressive Difficulty Ladder (1000 trials, repeated CV)
######################################################################
Collected 1000 combined trials.

Target                          Difficulty     Classes   Baseline          Best Acc      Margin
----------------------------------------------------------------------------------------------------
  Secret weight (binary)          Easy          2 classes  baseline= 53.9%  best_acc= 53.3% +/-  1.3%  (95% CI +/-0.4pts)  margin= -0.6pts
  Secret sum (binary)             Easy          2 classes  baseline= 57.7%  best_acc= 57.3% +/-  0.6%  (95% CI +/-0.2pts)  margin= -0.4pts
  Secret weight (multi-class)     Medium        3 classes  baseline= 46.1%  best_acc= 45.6% +/-  0.6%  (95% CI +/-0.2pts)  margin= -0.5pts
  s[0][0] sign (positive vs not)  Hard          2 classes  baseline= 65.9%  best_acc= 65.7% +/-  0.5%  (95% CI +/-0.1pts)  margin= -0.2pts
  s[0][0] exact value             Very Hard     3 classes  baseline= 34.1%  best_acc= 36.8% +/-  3.1%  (95% CI +/-0.9pts)  margin= +2.6pts

Done.
######################################################################
POSITIVE CONTROL: QEQSKE-clean vs. QEQSKE-leaky
######################################################################

--- Collecting QEQSKE-CLEAN dataset (300 trials) ---
Collected 300 clean trials.

--- Collecting QEQSKE-LEAKY dataset (300 trials) ---
Collected 300 leaky trials.

======================================================================
ML ATTACK on QEQSKE-CLEAN (expect: near baseline)
======================================================================

  Baseline (majority-class) accuracy = 56.33%
  Model                         Accuracy (mean+-std)     ROC-AUC (mean+-std)
  Logistic Regression         56.27% +/-  0.79%       0.484 +/- 0.091
  Random Forest               50.13% +/-  5.89%       0.495 +/- 0.063
  SVM (RBF)                   54.27% +/-  3.91%       0.532 +/- 0.053
  Neural Network (MLP)        54.00% +/-  3.85%       0.562 +/- 0.062

======================================================================
ML ATTACK on QEQSKE-LEAKY (expect: clearly above baseline)
======================================================================

  Baseline (majority-class) accuracy = 56.33%
  Model                         Accuracy (mean+-std)     ROC-AUC (mean+-std)
  Logistic Regression        100.00% +/-  0.00%       1.000 +/- 0.000
  Random Forest              100.00% +/-  0.00%       1.000 +/- 0.000
  SVM (RBF)                  100.00% +/-  0.00%       1.000 +/- 0.000

  Neural Network (MLP)       100.00% +/-  0.00%       1.000 +/- 0.000
######################################################################
SYSTEM COMPARISON: Random baseline vs ML-KEM vs QEQSKE-clean vs QEQSKE-leaky
######################################################################

Collecting timing data...
  - ML-KEM-512 (fixed param set), 300 trials, label = dk byte-Hamming-weight above/below median...
  - QEQSKE-clean, 300 trials...
  - QEQSKE-leaky, 300 trials...

======================================================================
Timing-based ML attack results (predict secret-relevant label from timing alone)
======================================================================
  Standard ML-KEM-512 (fixed params)baseline=50.33%   best=Logistic Regression 47.53% +/- 4.07%   margin=-2.80pts
  QEQSKE-clean                baseline=56.33%   best=Logistic Regression 56.23% +/- 0.81%   margin=-0.10pts
  QEQSKE-leaky (positive control)baseline=56.33%   best=Logistic Regression 100.00% +/- 0.00%   margin=+43.67pts

Done.
Starting Task 1: Classical Models
Running classical model: Random Forest

push