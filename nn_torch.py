import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Detect device: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

class SequenceClassifier:
    """Base class for PyTorch sequence models. Interface matches nn_from_scratch.py."""
    name = "base"

    def __init__(self, hidden=16, lr=1e-2, epochs=40, batch_size=32, seed=0):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._mu = None
        self._sd = None
        self.model = None

    def _normalise(self, X, fit=False):
        if fit:
            self._mu = X.mean(axis=(0, 1), keepdims=True)
            self._sd = X.std(axis=(0, 1), keepdims=True) + 1e-8
        return (X - self._mu) / self._sd

    def _build_model(self, input_dim):
        raise NotImplementedError

    def fit(self, X, y, verbose=False):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        X = self._normalise(np.asarray(X, dtype=float), fit=True)
        y = np.asarray(y, dtype=float).ravel()
        
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        input_dim = X.shape[2]
        self.model = self._build_model(input_dim).to(DEVICE)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for ep in range(self.epochs):
            tot_loss = 0.0
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                probs = self.model(bx)
                loss = criterion(probs, by)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item() * bx.size(0)
                
            if verbose and (ep + 1) % 10 == 0:
                print(f"    {self.name} epoch {ep + 1:3d}  loss {tot_loss / len(X):.4f}")
        return self

    def predict_proba(self, X):
        X = self._normalise(np.asarray(X, dtype=float))
        X_t = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        probs_all = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(DEVICE)
                probs = self.model(bx)
                probs_all.append(probs.cpu().numpy())
        return np.vstack(probs_all).ravel()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y).ravel()).mean())


class PyTorchLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

class LSTMClassifier(SequenceClassifier):
    name = "LSTM"
    def _build_model(self, input_dim):
        return PyTorchLSTM(input_dim, self.hidden)


class PyTorchGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, hn = self.gru(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

class GRUClassifier(SequenceClassifier):
    name = "GRU"
    def _build_model(self, input_dim):
        return PyTorchGRU(input_dim, self.hidden)


class PyTorchTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Project input to hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=1, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is (B, T, input_dim)
        T = x.size(1)
        pos = torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        # Replicate the manual pos encoding from numpy implementation
        x_pos = x + 0.1 * torch.sin(pos / max(T, 2))
        
        proj_x = self.proj(x_pos)
        out = self.transformer(proj_x)
        
        # mean pooling
        pooled = out.mean(dim=1)
        return self.sigmoid(self.fc(pooled))

class TransformerClassifier(SequenceClassifier):
    name = "Transformer"
    def _build_model(self, input_dim):
        return PyTorchTransformer(input_dim, self.hidden)

MODELS = {"LSTM": LSTMClassifier, "GRU": GRUClassifier, "Transformer": TransformerClassifier}
