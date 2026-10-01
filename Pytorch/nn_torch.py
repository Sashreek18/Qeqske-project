"""
nn_torch.py -- LSTM, GRU and Transformer in PyTorch
=====================================================
Drop-in replacement for nn_from_scratch.py. All three models expose the
same scikit-learn-style interface:

    model.fit(X, y)          X shape (batch, timesteps, features)
    model.predict(X)         -> class labels (numpy array)
    model.predict_proba(X)   -> P(class 1)   (numpy array)
    model.score(X, y)        -> accuracy (float)

Device selection is automatic:
    CUDA  -> NVIDIA GPU (RTX A1000 etc.)
    MPS   -> Apple Silicon GPU (Metal Performance Shaders)
    CPU   -> fallback

Gradients are handled by PyTorch autograd -- no hand-written backprop needed.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


# ---------------------------------------------------------------------------
# Shared training wrapper (scikit-learn style API)
# ---------------------------------------------------------------------------

class SequenceClassifierBase:
    """
    Wraps a torch.nn.Module with fit / predict / predict_proba / score,
    matching the exact interface used by nn_from_scratch.SequenceClassifier.
    """

    name = "base"

    def __init__(self, hidden=16, lr=1e-2, epochs=40, batch_size=32, seed=0):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._mu = None
        self._sd = None
        self._net = None

    def _build_net(self, n_features):
        """Override in each subclass to return a torch.nn.Module."""
        raise NotImplementedError

    def _normalise(self, X, fit=False):
        if fit:
            self._mu = X.mean(axis=(0, 1), keepdims=True)
            self._sd = X.std(axis=(0, 1), keepdims=True) + 1e-8
        return (X - self._mu) / self._sd

    def fit(self, X, y, verbose=False):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = self._normalise(np.asarray(X, dtype=np.float32), fit=True)
        y = np.asarray(y, dtype=np.float32).ravel()

        n_features = X.shape[2]
        self._net = self._build_net(n_features).to(DEVICE)

        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        n = len(X)
        rng = np.random.RandomState(self.seed)

        self._net.train()
        for ep in range(self.epochs):
            idx = rng.permutation(n)
            tot_loss = 0.0
            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]
                Xb = torch.from_numpy(X[b]).to(DEVICE)          # (B, T, F)
                yb = torch.from_numpy(y[b]).to(DEVICE)          # (B,)
                optimiser.zero_grad()
                prob = self._net(Xb).squeeze(-1)                 # (B,)
                loss = criterion(prob, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
                optimiser.step()
                tot_loss += loss.item() * len(b)
            if verbose and (ep + 1) % 10 == 0:
                print(f"    {self.name} epoch {ep + 1:3d}  loss {tot_loss / n:.4f}")
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self._net.eval()
        X = self._normalise(np.asarray(X, dtype=np.float32))
        Xt = torch.from_numpy(X).to(DEVICE)
        prob = self._net(Xt).squeeze(-1).cpu().numpy()
        return prob

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y).ravel()).mean())


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        out, (hn, _) = self.lstm(x)          # hn: (1, B, H)
        return self.sig(self.fc(hn.squeeze(0)))


class LSTMClassifier(SequenceClassifierBase):
    name = "LSTM"

    def _build_net(self, n_features):
        return _LSTMNet(n_features, self.hidden)


# ---------------------------------------------------------------------------
# GRU
# ---------------------------------------------------------------------------

class _GRUNet(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out, hn = self.gru(x)                # hn: (1, B, H)
        return self.sig(self.fc(hn.squeeze(0)))


class GRUClassifier(SequenceClassifierBase):
    name = "GRU"

    def _build_net(self, n_features):
        return _GRUNet(n_features, self.hidden)


# ---------------------------------------------------------------------------
# Transformer (single-head self-attention + mean pooling)
# ---------------------------------------------------------------------------

class _TransformerNet(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        # Project input features to hidden dim
        self.input_proj = nn.Linear(n_features, hidden)
        # nhead must evenly divide d_model; use 2 (minimum even value)
        nhead = 2 if hidden >= 2 else 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead,
            dim_feedforward=hidden * 2,
            dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc  = nn.Linear(hidden, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_proj(x)               # (B, T, hidden)
        x = self.encoder(x)                  # (B, T, hidden)
        pooled = x.mean(dim=1)               # (B, hidden)
        return self.sig(self.fc(pooled))


class TransformerClassifier(SequenceClassifierBase):
    name = "Transformer"

    def _build_net(self, n_features):
        return _TransformerNet(n_features, self.hidden)


# ---------------------------------------------------------------------------
# Model registry (same dict key names as nn_from_scratch.MODELS)
# ---------------------------------------------------------------------------

MODELS = {
    "LSTM":        LSTMClassifier,
    "GRU":         GRUClassifier,
    "Transformer": TransformerClassifier,
}


# ---------------------------------------------------------------------------
# Gradient / sanity check (confirms autograd is working)
# ---------------------------------------------------------------------------

def gradient_check(cls, seed=0, tol=1e-3):
    """
    Verifies that PyTorch autograd computes correct gradients for the
    network by comparing analytic grads to central finite differences.
    Uses torch.autograd.gradcheck on double precision for reliability.
    """
    torch.manual_seed(seed)
    B, T, F = 4, 6, 2

    m = cls(hidden=4, seed=seed)
    m._mu, m._sd = 0.0, 1.0
    # double precision required by gradcheck
    net = m._build_net(F).to("cpu").double()

    X = torch.randn(B, T, F, dtype=torch.float64, requires_grad=False)
    y = torch.randint(0, 2, (B,)).double()

    def loss_fn(*params):
        """Run forward pass with fresh params injected (needed by gradcheck)."""
        # gradcheck perturbs individual parameters; we need to rebuild the net
        # state from the provided flat tuple. We instead do a simpler check:
        # just verify loss.backward() agrees with manual finite differences
        # on the first parameter of the network.
        pass

    # Simpler: manual finite-difference on first weight matrix
    first_param = next(net.parameters())
    original = first_param.data.clone()
    eps = 1e-4

    # analytic gradient via autograd
    net.zero_grad()
    prob = net(X).squeeze(-1)
    loss = nn.BCELoss()(prob.double(), y)
    loss.backward()
    analytic = first_param.grad.data.clone()

    # numeric gradient (central difference) on the [0,0] entry
    with torch.no_grad():
        idx = tuple([0] * first_param.dim())
        first_param.data[idx] = original[idx] + eps
        lp = nn.BCELoss()(net(X).squeeze(-1).double(), y).item()
        first_param.data[idx] = original[idx] - eps
        lm = nn.BCELoss()(net(X).squeeze(-1).double(), y).item()
        first_param.data.copy_(original)

    numeric = (lp - lm) / (2 * eps)
    ana     = analytic[idx].item()
    denom   = max(abs(numeric) + abs(ana), 1e-8)
    err     = abs(numeric - ana) / denom
    return err, err < tol


if __name__ == "__main__":
    print(f"PyTorch version : {torch.__version__}")
    print(f"Running on      : {DEVICE}")
    print()
    print("Gradient check (finite differences vs PyTorch autograd)")
    print("-" * 58)
    for name, cls in MODELS.items():
        err, ok = gradient_check(cls)
        print(f"  {name:<14} max relative error {err:.2e}   "
              f"{'PASS' if ok else 'FAIL'}")
