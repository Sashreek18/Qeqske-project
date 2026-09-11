"""
nn_from_scratch.py -- LSTM, GRU and Transformer in pure NumPy
==============================================================
Sequence models for Attack A / RQ4, implemented with analytic backpropagation
and Adam. No PyTorch, no TensorFlow -- NumPy only.

Why sequence models at all: Attacks A and C treat each key generation as an
independent feature vector. That cannot detect a leak that lives in the ORDER
of QRNG draws -- autocorrelation, periodicity, or state carried between calls.
A recurrent or attention model reads the draw sequence itself and can, in
principle, pick up structure a bag-of-features model is blind to.

All three share the SequenceClassifier interface:
    model.fit(X, y)          X shape (batch, timesteps, features)
    model.predict(X)         -> class labels
    model.predict_proba(X)   -> P(class 1)

Gradients are verified against finite differences in gradient_check() at the
bottom of this file -- run `python3 nn_from_scratch.py` to execute the check.
"""

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))),
                    np.exp(np.clip(x, -60, 60)) / (1.0 + np.exp(np.clip(x, -60, 60))))


def dsigmoid_from_out(s):
    return s * (1.0 - s)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.p:
            g = np.clip(grads[k], -5.0, 5.0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


class SequenceClassifier:
    """Common training loop: binary cross-entropy, Adam, mini-batches."""

    name = "base"

    def __init__(self, hidden=16, lr=1e-2, epochs=40, batch_size=32, seed=0):
        self.hidden, self.lr = hidden, lr
        self.epochs, self.batch_size = epochs, batch_size
        self.rng = np.random.RandomState(seed)
        self.params = None
        self._mu = self._sd = None

    # subclasses implement these two
    def _init_params(self, n_features):
        raise NotImplementedError

    def _forward_backward(self, X, y):
        """Returns (loss, grads, probs)."""
        raise NotImplementedError

    def _normalise(self, X, fit=False):
        if fit:
            self._mu = X.mean(axis=(0, 1), keepdims=True)
            self._sd = X.std(axis=(0, 1), keepdims=True) + 1e-8
        return (X - self._mu) / self._sd

    def fit(self, X, y, verbose=False):
        X = self._normalise(np.asarray(X, dtype=float), fit=True)
        y = np.asarray(y, dtype=float).ravel()
        self._init_params(X.shape[2])
        opt = Adam(self.params, lr=self.lr)
        n = len(X)
        for ep in range(self.epochs):
            idx = self.rng.permutation(n)
            tot = 0.0
            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]
                loss, grads, _ = self._forward_backward(X[b], y[b])
                opt.step(grads)
                tot += loss * len(b)
            if verbose and (ep + 1) % 10 == 0:
                print(f"    {self.name} epoch {ep + 1:3d}  loss {tot / n:.4f}")
        return self

    def predict_proba(self, X):
        X = self._normalise(np.asarray(X, dtype=float))
        _, _, p = self._forward_backward(X, np.zeros(len(X)))
        return p

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y).ravel()).mean())


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMClassifier(SequenceClassifier):
    name = "LSTM"

    def _init_params(self, d):
        h = self.hidden
        s = 1.0 / np.sqrt(h + d)
        r = self.rng
        self.params = {
            "Wf": r.randn(d + h, h) * s, "bf": np.ones(h),   # forget bias 1
            "Wi": r.randn(d + h, h) * s, "bi": np.zeros(h),
            "Wo": r.randn(d + h, h) * s, "bo": np.zeros(h),
            "Wg": r.randn(d + h, h) * s, "bg": np.zeros(h),
            "Wy": r.randn(h, 1) * s,     "by": np.zeros(1),
        }

    def _forward_backward(self, X, y):
        P = self.params
        B, T, d = X.shape
        h_dim = self.hidden
        h = np.zeros((B, h_dim))
        c = np.zeros((B, h_dim))
        cache = []

        for t in range(T):
            z = np.concatenate([X[:, t, :], h], axis=1)
            f = sigmoid(z @ P["Wf"] + P["bf"])
            i = sigmoid(z @ P["Wi"] + P["bi"])
            o = sigmoid(z @ P["Wo"] + P["bo"])
            g = np.tanh(z @ P["Wg"] + P["bg"])
            c_prev = c
            c = f * c_prev + i * g
            tc = np.tanh(c)
            h = o * tc
            cache.append((z, f, i, o, g, c_prev, c, tc))

        logit = (h @ P["Wy"] + P["by"]).ravel()
        p = sigmoid(logit)
        eps = 1e-9
        loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

        grads = {k: np.zeros_like(v) for k, v in P.items()}
        dlogit = ((p - y) / B).reshape(-1, 1)
        grads["Wy"] = h.T @ dlogit
        grads["by"] = dlogit.sum(axis=0)
        dh = dlogit @ P["Wy"].T
        dc = np.zeros_like(c)

        for t in reversed(range(T)):
            z, f, i, o, g, c_prev, c_t, tc = cache[t]
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev = dc * f

            da_f = df * dsigmoid_from_out(f)
            da_i = di * dsigmoid_from_out(i)
            da_o = do * dsigmoid_from_out(o)
            da_g = dg * (1 - g ** 2)

            for nm, da in (("f", da_f), ("i", da_i), ("o", da_o), ("g", da_g)):
                grads["W" + nm] += z.T @ da
                grads["b" + nm] += da.sum(axis=0)

            dz = (da_f @ P["Wf"].T + da_i @ P["Wi"].T +
                  da_o @ P["Wo"].T + da_g @ P["Wg"].T)
            dh = dz[:, d:]
            dc = dc_prev

        return loss, grads, p


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

class GRUClassifier(SequenceClassifier):
    name = "GRU"

    def _init_params(self, d):
        h = self.hidden
        s = 1.0 / np.sqrt(h + d)
        r = self.rng
        self.params = {
            "Wz": r.randn(d + h, h) * s, "bz": np.zeros(h),
            "Wr": r.randn(d + h, h) * s, "br": np.zeros(h),
            "Wh": r.randn(d + h, h) * s, "bh": np.zeros(h),
            "Wy": r.randn(h, 1) * s,     "by": np.zeros(1),
        }

    def _forward_backward(self, X, y):
        P = self.params
        B, T, d = X.shape
        h = np.zeros((B, self.hidden))
        cache = []

        for t in range(T):
            x = X[:, t, :]
            z_in = np.concatenate([x, h], axis=1)
            zt = sigmoid(z_in @ P["Wz"] + P["bz"])
            rt = sigmoid(z_in @ P["Wr"] + P["br"])
            hr = np.concatenate([x, rt * h], axis=1)
            hh = np.tanh(hr @ P["Wh"] + P["bh"])
            h_prev = h
            h = (1 - zt) * h_prev + zt * hh
            cache.append((x, z_in, zt, rt, hr, hh, h_prev))

        logit = (h @ P["Wy"] + P["by"]).ravel()
        p = sigmoid(logit)
        eps = 1e-9
        loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

        grads = {k: np.zeros_like(v) for k, v in P.items()}
        dlogit = ((p - y) / B).reshape(-1, 1)
        grads["Wy"] = h.T @ dlogit
        grads["by"] = dlogit.sum(axis=0)
        dh = dlogit @ P["Wy"].T

        for t in reversed(range(T)):
            x, z_in, zt, rt, hr, hh, h_prev = cache[t]
            dz = dh * (hh - h_prev)
            dhh = dh * zt
            dh_prev = dh * (1 - zt)

            da_hh = dhh * (1 - hh ** 2)
            grads["Wh"] += hr.T @ da_hh
            grads["bh"] += da_hh.sum(axis=0)
            dhr = da_hh @ P["Wh"].T
            drh = dhr[:, d:]
            dr = drh * h_prev
            dh_prev = dh_prev + drh * rt

            da_z = dz * dsigmoid_from_out(zt)
            da_r = dr * dsigmoid_from_out(rt)
            for nm, da in (("z", da_z), ("r", da_r)):
                grads["W" + nm] += z_in.T @ da
                grads["b" + nm] += da.sum(axis=0)

            dz_in = da_z @ P["Wz"].T + da_r @ P["Wr"].T
            dh = dh_prev + dz_in[:, d:]

        return loss, grads, p


# ---------------------------------------------------------------------------
# Transformer (single-head self-attention + mean pooling)
# ---------------------------------------------------------------------------

class TransformerClassifier(SequenceClassifier):
    name = "Transformer"

    def _init_params(self, d):
        h = self.hidden
        s = 1.0 / np.sqrt(max(d, h))
        r = self.rng
        self.params = {
            "Wq": r.randn(d, h) * s,
            "Wk": r.randn(d, h) * s,
            "Wv": r.randn(d, h) * s,
            "W1": r.randn(h, h) * s, "b1": np.zeros(h),
            "Wy": r.randn(h, 1) * s, "by": np.zeros(1),
        }

    def _forward_backward(self, X, y):
        P = self.params
        B, T, d = X.shape
        h = self.hidden
        scale = 1.0 / np.sqrt(h)

        # positional encoding so the model can use draw ORDER
        pos = np.arange(T)[None, :, None]
        Xp = X + 0.1 * np.sin(pos / max(T, 2))

        Q = Xp @ P["Wq"]                       # (B,T,h)
        Kk = Xp @ P["Wk"]
        V = Xp @ P["Wv"]
        scores = np.einsum("bth,bsh->bts", Q, Kk) * scale
        Att = softmax(scores, axis=-1)
        ctx = np.einsum("bts,bsh->bth", Att, V)

        ff_in = ctx @ P["W1"] + P["b1"]
        ff = np.maximum(ff_in, 0.0)            # ReLU
        pooled = ff.mean(axis=1)               # (B,h)
        logit = (pooled @ P["Wy"] + P["by"]).ravel()
        p = sigmoid(logit)
        eps = 1e-9
        loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

        dlogit = ((p - y) / B).reshape(-1, 1)
        grads = {k: np.zeros_like(v) for k, v in P.items()}
        grads["Wy"] = pooled.T @ dlogit
        grads["by"] = dlogit.sum(axis=0)

        dpooled = dlogit @ P["Wy"].T                       # (B,h)
        dff = np.repeat(dpooled[:, None, :], T, axis=1) / T
        dff_in = dff * (ff_in > 0)
        grads["W1"] = np.einsum("bth,btk->hk", ctx, dff_in)
        grads["b1"] = dff_in.sum(axis=(0, 1))
        dctx = dff_in @ P["W1"].T                          # (B,T,h)

        dAtt = np.einsum("bth,bsh->bts", dctx, V)
        dV = np.einsum("bts,bth->bsh", Att, dctx)
        # softmax Jacobian, row-wise
        dscores = Att * (dAtt - (dAtt * Att).sum(axis=-1, keepdims=True))
        dscores *= scale
        dQ = np.einsum("bts,bsh->bth", dscores, Kk)
        dK = np.einsum("bts,bth->bsh", dscores, Q)

        grads["Wq"] = np.einsum("btd,bth->dh", Xp, dQ)
        grads["Wk"] = np.einsum("btd,bth->dh", Xp, dK)
        grads["Wv"] = np.einsum("btd,bth->dh", Xp, dV)

        return loss, grads, p


MODELS = {"LSTM": LSTMClassifier, "GRU": GRUClassifier,
          "Transformer": TransformerClassifier}


# ---------------------------------------------------------------------------
# gradient check
# ---------------------------------------------------------------------------

def gradient_check(cls, seed=0, tol=1e-4):
    """Finite-difference check of the analytic gradients."""
    rng = np.random.RandomState(seed)
    X = rng.randn(6, 5, 3)
    y = rng.randint(0, 2, 6).astype(float)
    m = cls(hidden=4, seed=seed)
    m._mu, m._sd = 0.0, 1.0
    m._init_params(3)
    _, grads, _ = m._forward_backward(X, y)

    worst = 0.0
    for k in m.params:
        P = m.params[k]
        it = np.ndindex(P.shape)
        for _ in range(min(6, P.size)):
            idx = next(it)
            orig = P[idx]
            h = 1e-5
            P[idx] = orig + h
            lp, _, _ = m._forward_backward(X, y)
            P[idx] = orig - h
            lm, _, _ = m._forward_backward(X, y)
            P[idx] = orig
            num = (lp - lm) / (2 * h)
            ana = grads[k][idx]
            denom = max(abs(num), abs(ana), 1e-8)
            worst = max(worst, abs(num - ana) / denom)
    return worst, worst < tol


if __name__ == "__main__":
    print("Gradient check (finite differences vs analytic backprop)")
    print("-" * 58)
    for name, cls in MODELS.items():
        err, ok = gradient_check(cls)
        print(f"  {name:<14} max relative error {err:.2e}   "
              f"{'PASS' if ok else 'FAIL'}")
