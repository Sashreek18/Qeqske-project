"""
ATTACK B: Entropy Leakage Detection  [PyTorch version]
=======================================================
Replaces sklearn MLPRegressor autoencoder with a native PyTorch autoencoder
that trains on GPU/MPS. All data windowing and evaluation logic is identical.

Why PyTorch here: the autoencoder in the sklearn version is limited to CPU
and cannot be easily scaled (more layers, deeper bottleneck, attention-based
reconstruction). PyTorch allows future scaling to larger models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nn_torch import get_device

DEVICE = get_device()


# ---------------------------------------------------------------------------
# Data preparation (unchanged from sklearn version)
# ---------------------------------------------------------------------------

def numbers_to_windows(numbers, window_size=16, stride=4):
    """Slice into overlapping windows, returned as float32 numpy array."""
    windows = []
    for i in range(0, len(numbers) - window_size + 1, stride):
        windows.append(numbers[i:i + window_size])
    return np.array(windows, dtype=np.float32)


# ---------------------------------------------------------------------------
# PyTorch Autoencoder
# ---------------------------------------------------------------------------

class _Autoencoder(nn.Module):
    """
    Bottleneck autoencoder: input_dim -> hidden_size -> input_dim.
    Mirrors the sklearn MLPRegressor(hidden_layer_sizes=(hidden_size,))
    architecture, but runs on GPU/MPS and uses tanh activation.
    """
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(train_windows, hidden_size=4,
                      epochs=200, lr=1e-3, batch_size=256, seed=42):
    """
    Trains a bottleneck autoencoder on train_windows.
    Returns (model, mu, sd) — the normalisation stats are returned
    so reconstruction_error() can apply the same scaling.
    """
    torch.manual_seed(seed)
    input_dim = train_windows.shape[1]

    # Min-Max normalisation (mirrors sklearn MinMaxScaler)
    x_min = train_windows.min(axis=0)
    x_max = train_windows.max(axis=0)
    scale = (x_max - x_min) + 1e-8

    X_scaled = ((train_windows - x_min) / scale).astype(np.float32)

    net = _Autoencoder(input_dim, hidden_size).to(DEVICE)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Simple 85/15 val split for early stopping (mirrors sklearn default)
    n_val  = max(1, int(0.15 * len(X_scaled)))
    n_train = len(X_scaled) - n_val
    X_tr = torch.from_numpy(X_scaled[:n_train]).to(DEVICE)
    X_va = torch.from_numpy(X_scaled[n_train:]).to(DEVICE)

    loader = DataLoader(TensorDataset(X_tr, X_tr),
                        batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    patience = 20
    patience_count = 0

    net.train()
    for ep in range(epochs):
        for Xb, _ in loader:
            optimiser.zero_grad()
            criterion(net(Xb), Xb).backward()
            optimiser.step()

        with torch.no_grad():
            val_loss = criterion(net(X_va), X_va).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    return net, x_min, scale


@torch.no_grad()
def reconstruction_error(net, x_min, scale, windows):
    """Mean squared reconstruction error per window (on CPU-numpy output)."""
    net.eval()
    X_scaled = ((windows - x_min) / scale).astype(np.float32)
    Xt = torch.from_numpy(X_scaled).to(DEVICE)
    X_pred = net(Xt).cpu().numpy()
    errors = np.mean((X_scaled - X_pred) ** 2, axis=1)
    return errors


# ---------------------------------------------------------------------------
# Main attack runner
# ---------------------------------------------------------------------------

def run_attack_b(qrng_numbers, mersenne_numbers,
                 window_size=16, stride=4, verbose=True):
    print(f"  Device: {DEVICE}")

    qrng_windows    = numbers_to_windows(qrng_numbers, window_size, stride)
    mersenne_windows = numbers_to_windows(mersenne_numbers, window_size, stride)

    if len(qrng_windows) < 10:
        print("Not enough QRNG windows for training. Collect more data.")
        return None

    n_train = int(0.7 * len(qrng_windows))
    qrng_train = qrng_windows[:n_train]
    qrng_test  = qrng_windows[n_train:]

    hidden_size = max(2, window_size // 4)

    if verbose:
        print(f"Training PyTorch autoencoder on {len(qrng_train)} QRNG windows "
              f"(window_size={window_size}, hidden={hidden_size})...")

    net, x_min, scale = train_autoencoder(qrng_train, hidden_size=hidden_size)

    # Pure uniform random noise as theoretical ideal baseline
    rng = np.random.default_rng(seed=123)
    pure_random = rng.integers(0, 65536,
                               size=(len(qrng_test), window_size)).astype(np.float32)

    err_qrng     = reconstruction_error(net, x_min, scale, qrng_test)
    err_mersenne = reconstruction_error(net, x_min, scale, mersenne_windows)
    err_pure     = reconstruction_error(net, x_min, scale, pure_random)

    if verbose:
        print(f"\n{'=' * 60}")
        print("ATTACK B RESULTS: Entropy Leakage via Autoencoder  [PyTorch]")
        print(f"{'=' * 60}")
        print(f"  (Lower error = more structure = WORSE for security)")
        print(f"\n  QRNG held-out test error    = {np.mean(err_qrng):.5f} "
              f"(±{np.std(err_qrng):.5f})")
        print(f"  Mersenne Twister error      = {np.mean(err_mersenne):.5f} "
              f"(±{np.std(err_mersenne):.5f})")
        print(f"  Pure uniform random error   = {np.mean(err_pure):.5f} "
              f"(±{np.std(err_pure):.5f})")

        ratio = np.mean(err_qrng) / max(np.mean(err_pure), 1e-9)
        print(f"\n  QRNG error / Pure-random error ratio = {ratio:.3f}")
        if 0.85 <= ratio <= 1.15:
            print("  -> QRNG behaves like ideal randomness (ratio ~1.0). GOOD — no obvious leakage.")
        elif ratio < 0.85:
            print("  -> QRNG is MORE predictable than pure randomness. POTENTIAL LEAKAGE.")
        else:
            print("  -> QRNG is LESS predictable than pure randomness (unusual, investigate).")

    return {
        "qrng_error":       err_qrng,
        "mersenne_error":   err_mersenne,
        "pure_random_error": err_pure,
    }


if __name__ == "__main__":
    with open("../qrng_combined.txt") as f:
        qrng_numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    with open("../mersenne_large_baseline.txt") as f:
        mersenne_numbers = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

    run_attack_b(qrng_numbers, mersenne_numbers, window_size=16, stride=4)
