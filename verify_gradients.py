"""
verify_gradients.py -- numerical check of the hand-written backprop
====================================================================
The LSTM, GRU and Transformer in nn_from_scratch.py implement backpropagation
analytically, by hand, with no autograd. A sign error or a missing residual term
in a backward pass does not raise an exception -- it silently produces a model
that trains poorly, and a model that trains poorly produces a null result that
looks exactly like "no leakage found".

That failure mode would be invisible without this check, and it would quietly
invalidate every RQ4 conclusion. So each analytic gradient is compared against a
central-difference estimate of the same quantity:

    df/dw  ~=  ( f(w + eps) - f(w - eps) ) / (2 * eps)

Agreement to ~1e-6 relative error means the backward pass matches the forward
pass. This runs in seconds and should be run whenever nn_from_scratch.py is
touched.

Run:  python3 verify_gradients.py
"""

import numpy as np

from nn_from_scratch import (GRUClassifier, LSTMClassifier,
                             TransformerClassifier)

EPS = 1e-5
TOL = 1e-4
N, T, F = 4, 6, 2


def check(model_cls, n_probe=3, seed=0):
    """Central-difference check on a few random entries of every parameter."""
    rs = np.random.RandomState(seed)
    X = rs.randn(N, T, F)
    y = rs.randint(0, 2, N).astype(float)

    model = model_cls(hidden=8, seed=seed)
    model._normalise(X, fit=True)
    Xn = model._normalise(X)
    model._init_params(F)

    _, grads, _ = model._forward_backward(Xn, y)

    worst, worst_key = 0.0, None
    for key, arr in model.params.items():
        if arr.size == 0 or key not in grads:
            continue
        flat = arr.ravel()
        gflat = np.asarray(grads[key]).ravel()
        idxs = rs.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + EPS
            lp, _, _ = model._forward_backward(Xn, y)
            flat[idx] = orig - EPS
            lm, _, _ = model._forward_backward(Xn, y)
            flat[idx] = orig

            numeric = (lp - lm) / (2 * EPS)
            analytic = gflat[idx]
            denom = max(abs(numeric) + abs(analytic), 1e-8)
            rel = abs(numeric - analytic) / denom
            if rel > worst:
                worst, worst_key = rel, f"{key}[{idx}]"
    return worst, worst_key


def main():
    print("=" * 70)
    print("GRADIENT VERIFICATION -- analytic backprop vs central differences")
    print("=" * 70)
    print(f"  eps={EPS}  tolerance={TOL}  batch={N}  seq_len={T}  features={F}\n")

    all_ok = True
    for cls in (LSTMClassifier, GRUClassifier, TransformerClassifier):
        worst, key = check(cls)
        ok = worst < TOL
        all_ok &= ok
        print(f"  {cls.name:<14}max relative error {worst:.3e}  "
              f"(worst: {key})   {'PASS' if ok else 'FAIL'}")

    print()
    if all_ok:
        print("  All backward passes agree with numerical gradients. The RQ4\n"
              "  models are training on correct gradients, so a null result\n"
              "  there reflects the data rather than a broken implementation.")
    else:
        print("  FAILURE. At least one backward pass disagrees with its\n"
              "  numerical gradient. Do NOT report any RQ4 result until this\n"
              "  passes -- a broken gradient produces a false null.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
