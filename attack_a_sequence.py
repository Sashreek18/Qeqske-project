"""
Attack A / RQ4 -- sequence models on QRNG draw ORDER
=====================================================
Every other attack in this study treats a key generation as an unordered
feature vector. That design is blind to leakage that lives in the sequence
itself: autocorrelation, periodicity, or generator state carried across calls.

Here the raw draw sequence is fed to three models written from scratch in NumPy
(nn_from_scratch.py -- LSTM, GRU, single-head Transformer with positional
encoding). Task: given a window of consecutive draws, decide whether it came
from the ANU QRNG or from Mersenne Twister.

POSITIVE CONTROL
----------------
A null result from a hand-written network is worthless without evidence the
network can learn anything at all. So a third arm uses a deliberately
autocorrelated source -- a sequence where each value depends on the previous
one. The models MUST detect that. If they do not, the architectures or the
training schedule are broken and no null result here means anything.

Run:  python3 attack_a_sequence.py [--windows 400] [--window-len 32]
"""

import argparse
import json
import os
import random

import numpy as np

from nn_torch import MODELS

SEED = 42


def load_qrng(path="qrng_combined.txt"):
    with open(path) as f:
        return np.array([int(x) for x in f.read().split(",") if x.strip()],
                        dtype=float)


def mersenne_stream(n, seed=SEED, hi=65535):
    r = random.Random(seed)
    return np.array([r.randint(0, hi) for _ in range(n)], dtype=float)


def autocorrelated_stream(n, seed=SEED, hi=65535, rho=0.6, match=None):
    """
    Positive control: each value is pulled toward the previous one.

    IMPORTANT: the rho-smoothing shrinks the variance of the stream. Left
    uncorrected, the models would separate this arm from Mersenne Twister on
    amplitude alone -- which would validate nothing about their ability to read
    ORDER. So the stream is rescaled to match the target's mean and SD, leaving
    autocorrelation as the only distinguishing property.
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
    per_window_z: standardise each window on its own mean/SD. This strips out
    amplitude and offset differences between the two sources, so the only thing
    left for the model to use is the ORDER of values within the window -- which
    is the whole point of a sequence attack. Without it, a source with a
    slightly different marginal distribution is separable for reasons that have
    nothing to do with sequential structure.
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
    """Train/test split with a majority baseline. Simple holdout, since these
    models are slow -- CV would be better but is not affordable in pure NumPy."""
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
    print("ATTACK A / RQ4 -- sequence models on QRNG draw order")
    print("=" * 70)
    print(f"  windows per class: {args.windows}   window length: "
          f"{args.window_len}   epochs: {args.epochs}")
    print("  models: LSTM, GRU, Transformer (pure NumPy, analytic backprop)")
    print("  gradients verified against finite differences: see "
          "nn_from_scratch.py")

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
  {ctrl['_best_margin']:+.2f} points. Until an obviously autocorrelated stream is
  detected, the null result on the real task cannot be interpreted --
  it may simply mean these models are undertrained. Raise --epochs or
  --windows and re-run before reporting anything from this experiment.""")
    elif real["_best_margin"] > 5:
        print(f"""
  SEQUENCE STRUCTURE DETECTED on the real task ({real['_best_margin']:+.2f} pts). This would be a
  positive finding and needs replication at higher window counts, with
  different window lengths, and with the QRNG/MT assignment re-randomised
  before it is reported.""")
    else:
        print(f"""
  Positive control detected at {ctrl['_best_margin']:+.2f} points, confirming the models can learn
  sequence structure when it exists. On the real task the best margin is
  {real['_best_margin']:+.2f} points -- no detectable order-dependent structure separating
  ANU QRNG draws from Mersenne Twister at window length {args.window_len}.

  Scope: this tests the draw sequence as consumed by QEQSKE, at one window
  length, with small models. It is evidence against short-range sequential
  leakage, not against all sequential structure.""")

    os.makedirs("results", exist_ok=True)
    with open("results/attack_a_sequence.json", "w") as f:
        json.dump({"config": vars(args), "real_task": real,
                   "positive_control": ctrl,
                   "pipeline_validated": bool(validated)}, f, indent=2)
    print("  [saved] results/attack_a_sequence.json")


if __name__ == "__main__":
    main()
