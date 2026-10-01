"""
Attack A / RQ4 -- sequence models on QRNG draw order  [PyTorch version]
========================================================================
Drop-in replacement for sk-learn/attack_a_sequence.py.

The data pipeline (load_qrng, mersenne_stream, autocorrelated_stream,
build_dataset, windows) is IDENTICAL.

The model classes (LSTMClassifier, GRUClassifier, TransformerClassifier)
are now imported from nn_torch.py (PyTorch + autograd, GPU/MPS support)
instead of nn_from_scratch.py (pure NumPy).

Training now runs on whatever GPU is available automatically.
"""

import argparse
import json
import os
import random

import numpy as np

from nn_torch import MODELS, get_device

DEVICE = get_device()
SEED = 42


def load_qrng(path="../qrng_combined.txt"):
    with open(path) as f:
        return np.array([int(x) for x in f.read().split(",") if x.strip()],
                        dtype=float)


def mersenne_stream(n, seed=SEED, hi=65535):
    r = random.Random(seed)
    return np.array([r.randint(0, hi) for _ in range(n)], dtype=float)


def autocorrelated_stream(n, seed=SEED, hi=65535, rho=0.6, match=None):
    """
    Positive control: each value is pulled toward the previous one.
    Rescaled to match target's mean and SD so only autocorrelation
    differs (not amplitude).
    """
    r = np.random.RandomState(seed)
    out = np.zeros(n)
    prev = r.randint(0, hi)
    for i in range(n):
        val = rho * prev + (1 - rho) * r.randint(0, hi)
        out[i] = val
        prev = val
    if match is not None:
        out = (out - out.mean()) / (out.std() + 1e-12)
        out = out * match.std() + match.mean()
    return out


def windows(stream, count, length, rng):
    """Non-overlapping-ish windows sampled from the stream."""
    max_start = len(stream) - length - 1
    starts = rng.choice(max_start, size=count, replace=False)
    return np.stack([stream[s:s + length] for s in starts])


def build_dataset(stream_a, stream_b, count, length, seed=SEED,
                  per_window_z=True):
    """
    per_window_z: standardise each window on its own mean/SD to strip
    amplitude differences, leaving only sequential structure.
    """
    rng = np.random.RandomState(seed)
    a = windows(stream_a, count, length, rng)
    b = windows(stream_b, count, length, rng)
    X = np.concatenate([a, b])
    if per_window_z:
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-9)
    X = X[:, :, None]                               # (2*count, length, 1)
    y = np.concatenate([np.ones(count), np.zeros(count)])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def holdout_eval(model_cls, X, y, seed=SEED, epochs=40, hidden=16):
    """
    Train/test split with a majority baseline.
    Models are now PyTorch (GPU/MPS accelerated).
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    cut = int(0.75 * len(X))
    tr, te = idx[:cut], idx[cut:]
    model = model_cls(hidden=hidden, lr=5e-3, epochs=epochs, seed=seed)
    model.fit(X[tr], y[tr])
    acc = model.score(X[te], y[te])
    vals, counts = np.unique(y[te], return_counts=True)
    base = counts.max() / len(y[te])
    return acc, base


def run_arm(name, stream_a, stream_b, count, length, epochs):
    print(f"\n  {name}")
    print("  " + "-" * 62)
    X, y = build_dataset(stream_a, stream_b, count, length)
    out = {}
    for mname, cls in MODELS.items():
        acc, base = holdout_eval(cls, X, y, epochs=epochs)
        margin = 100 * (acc - base)
        out[mname] = {"accuracy": float(acc), "baseline": float(base),
                      "margin_pts": float(margin)}
        print(f"  {mname:<14} acc {100 * acc:6.2f}%   baseline {100 * base:6.2f}%"
              f"   margin {margin:+6.2f} pts")
    best = max(out, key=lambda k: out[k]["margin_pts"])
    out["_best"] = best
    out["_best_margin"] = out[best]["margin_pts"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=400,
                    help="windows per class")
    ap.add_argument("--window-len", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    print("=" * 70)
    print("ATTACK A / RQ4 -- sequence models on QRNG draw order  [PyTorch]")
    print("=" * 70)
    print(f"  Device          : {DEVICE}")
    print(f"  windows/class   : {args.windows}   window length: "
          f"{args.window_len}   epochs: {args.epochs}")
    print("  models: LSTM, GRU, Transformer (PyTorch autograd)")

    need = args.windows + args.window_len + 2
    qrng = load_qrng()
    mt = mersenne_stream(len(qrng))
    auto = autocorrelated_stream(need * 4, match=mt)

    real = run_arm("REAL TASK -- QRNG vs Mersenne Twister",
                   qrng, mt, args.windows, args.window_len, args.epochs)
    ctrl = run_arm("POSITIVE CONTROL -- autocorrelated vs Mersenne Twister",
                   auto, mt[:len(auto)], min(args.windows, len(auto) // 4),
                   args.window_len, args.epochs)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  real task        best {real['_best']:<12} "
          f"margin {real['_best_margin']:+6.2f} pts")
    print(f"  positive control best {ctrl['_best']:<12} "
          f"margin {ctrl['_best_margin']:+6.2f} pts")

    validated = ctrl["_best_margin"] > 10
    if not validated:
        print(f"""
  PIPELINE NOT VALIDATED. The positive control only reached
  {ctrl['_best_margin']:+.2f} points. Raise --epochs or --windows and re-run.""")
    elif real["_best_margin"] > 5:
        print(f"""
  SEQUENCE STRUCTURE DETECTED on the real task ({real['_best_margin']:+.2f} pts). Needs replication.""")
    else:
        print(f"""
  Positive control detected at {ctrl['_best_margin']:+.2f} pts. On the real task the best
  margin is {real['_best_margin']:+.2f} pts -- no detectable order-dependent structure.""")

    os.makedirs("../results", exist_ok=True)
    with open("../results/attack_a_sequence_torch.json", "w") as f:
        json.dump({"config": vars(args), "real_task": real,
                   "positive_control": ctrl,
                   "pipeline_validated": bool(validated),
                   "device": str(DEVICE)}, f, indent=2)
    print("  [saved] results/attack_a_sequence_torch.json")


if __name__ == "__main__":
    main()
