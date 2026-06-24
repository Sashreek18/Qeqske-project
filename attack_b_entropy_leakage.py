"""
ATTACK B: Entropy Leakage Detection
======================================
Research Question (from HCL paper proposal):
  "Can deep neural networks detect hidden biases in QRNG implementations
   that pass NIST statistical tests?"

Approach:
  Train an AUTOENCODER on the QRNG data. An autoencoder learns to compress
  and reconstruct its training data. If it reconstructs QRNG sequences
  TOO WELL (very low error), that suggests there's learnable STRUCTURE
  in the data that shouldn't be there for true randomness — true random
  data should be hard to compress/predict.

  We compare reconstruction error on:
    - QRNG data (held-out test split)
    - Fresh Mersenne Twister data (different seed, unseen)
    - Pure uniform random noise (theoretical ideal)

  If QRNG reconstruction error is similar to pure random noise, that's
  GOOD (no leakage). If it's much lower (easier to reconstruct), that
  suggests exploitable structure (H3: deep learning models can identify
  subtle implementation biases missed by traditional statistical tests).
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def numbers_to_windows(numbers, window_size=16, stride=4):
    """Slice into overlapping windows, normalized to [0,1] for NN training."""
    windows = []
    for i in range(0, len(numbers) - window_size + 1, stride):
        windows.append(numbers[i:i + window_size])
    return np.array(windows, dtype=float)


def train_autoencoder(train_windows, hidden_size=4, max_iter=2000):
    """
    A simple autoencoder using MLPRegressor: input -> small hidden layer
    (bottleneck) -> output (reconstruction). The bottleneck forces the
    network to learn compressed structure if any exists.
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(train_windows)

    autoencoder = MLPRegressor(
        hidden_layer_sizes=(hidden_size,),
        activation="tanh",
        solver="adam",
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    autoencoder.fit(X_scaled, X_scaled)  # learns to reconstruct its own input

    return autoencoder, scaler


def reconstruction_error(autoencoder, scaler, windows):
    """Mean squared reconstruction error per window."""
    X_scaled = scaler.transform(windows)
    X_pred = autoencoder.predict(X_scaled)
    errors = np.mean((X_scaled - X_pred) ** 2, axis=1)
    return errors


def run_attack_b(qrng_numbers, mersenne_numbers, window_size=16, stride=4, verbose=True):
    qrng_windows = numbers_to_windows(qrng_numbers, window_size, stride)
    mersenne_windows = numbers_to_windows(mersenne_numbers, window_size, stride)

    if len(qrng_windows) < 10:
        print("Not enough QRNG windows for training. Collect more data.")
        return None

    # Split QRNG windows: train autoencoder on 70%, test reconstruction on 30% held-out
    n_train = int(0.7 * len(qrng_windows))
    qrng_train = qrng_windows[:n_train]
    qrng_test = qrng_windows[n_train:]

    if verbose:
        print(f"Training autoencoder on {len(qrng_train)} QRNG windows "
              f"(window_size={window_size})...")

    autoencoder, scaler = train_autoencoder(qrng_train, hidden_size=max(2, window_size // 4))

    # Generate pure uniform random noise as theoretical "ideal randomness" baseline
    rng = np.random.default_rng(seed=123)
    pure_random = rng.integers(0, 65536, size=(len(qrng_test), window_size)).astype(float)

    err_qrng_held_out = reconstruction_error(autoencoder, scaler, qrng_test)
    err_mersenne = reconstruction_error(autoencoder, scaler, mersenne_windows)
    err_pure_random = reconstruction_error(autoencoder, scaler, pure_random)

    if verbose:
        print(f"\n{'=' * 60}")
        print("ATTACK B RESULTS: Entropy Leakage via Autoencoder Reconstruction")
        print(f"{'=' * 60}")
        print(f"  (Lower error = autoencoder finds MORE structure/patterns = WORSE for security)")
        print(f"\n  QRNG held-out test error    = {np.mean(err_qrng_held_out):.5f} "
              f"(±{np.std(err_qrng_held_out):.5f})")
        print(f"  Mersenne Twister error      = {np.mean(err_mersenne):.5f} "
              f"(±{np.std(err_mersenne):.5f})")
        print(f"  Pure uniform random error   = {np.mean(err_pure_random):.5f} "
              f"(±{np.std(err_pure_random):.5f})")

        qrng_vs_pure_ratio = np.mean(err_qrng_held_out) / max(np.mean(err_pure_random), 1e-9)
        print(f"\n  QRNG error / Pure-random error ratio = {qrng_vs_pure_ratio:.3f}")
        if 0.85 <= qrng_vs_pure_ratio <= 1.15:
            print("  -> QRNG behaves like ideal randomness (ratio ~1.0). GOOD — no obvious leakage.")
        elif qrng_vs_pure_ratio < 0.85:
            print("  -> QRNG is MORE predictable than pure randomness. POTENTIAL LEAKAGE.")
        else:
            print("  -> QRNG is LESS predictable than pure randomness (unusual, investigate).")

    return {
        "qrng_error": err_qrng_held_out,
        "mersenne_error": err_mersenne,
        "pure_random_error": err_pure_random,
    }


if __name__ == "__main__":
    with open("sample_qrng_batch.txt") as f:
        qrng_numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    with open("mersenne_baseline.txt") as f:
        mersenne_numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    run_attack_b(qrng_numbers, mersenne_numbers, window_size=16, stride=4)
