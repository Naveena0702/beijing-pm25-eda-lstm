"""Mine (deep learning): sliding windows, the PyTorch LSTM forecaster, training, metrics and feature analysis."""
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import SEED, HORIZON


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- windowing
def valid_origins(feat_ok, target_ok, lookback, horizon, lo, hi):
    """Forecast origins t in [lo, hi) whose input window (t-L+1 .. t) and target window
    (t+1 .. t+H) contain no missing values and whose targets stay before `hi`, so that no
    target leaks across a train / validation / test boundary."""
    bad_in = np.r_[0, np.cumsum(~feat_ok)]
    bad_out = np.r_[0, np.cumsum(~target_ok)]
    t = np.arange(max(lo, lookback - 1), hi - horizon)
    ok_in = (bad_in[t + 1] - bad_in[t + 1 - lookback]) == 0
    ok_out = (bad_out[t + 1 + horizon] - bad_out[t + 1]) == 0
    return t[ok_in & ok_out]


def make_windows(feats, target, origins, lookback, horizon=HORIZON):
    """X has shape (N, lookback, n_features); Y has shape (N, horizon)."""
    idx_in = origins[:, None] + np.arange(-lookback + 1, 1)[None, :]
    idx_out = origins[:, None] + np.arange(1, horizon + 1)[None, :]
    return feats[idx_in].astype(np.float32), target[idx_out].astype(np.float32)


# --------------------------------------------------------------------------- network
class LSTMForecaster(nn.Module):
    """Two stacked LSTM layers followed by a small dense head that emits all horizons at once.

    With `anchor_idx` set, the network uses residual learning: the most recent observed value of
    that input channel (standardised log PM2.5) is added to the head's output, so the LSTM only has
    to learn the change from the persistence forecast instead of the whole level.
    """

    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2, horizon=HORIZON, anchor_idx=None):
        super().__init__()
        self.anchor_idx = anchor_idx
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden // 2, horizon))

    def forward(self, x):
        out, _ = self.lstm(x)              # out: (batch, time, hidden)
        delta = self.head(out[:, -1, :])   # use the state after the most recent hour
        if self.anchor_idx is None:
            return delta
        return x[:, -1, self.anchor_idx].unsqueeze(1) + delta   # skip connection: persistence + learned change


@torch.no_grad()
def predict(model, X, batch=4096):
    model.eval()
    return np.concatenate([model(torch.from_numpy(X[i:i + batch])).numpy() for i in range(0, len(X), batch)])


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def train(Xtr, Ytr, Xva, Yva, inv, hidden=64, lr=1e-3, epochs=40, batch=256, patience=6, seed=SEED, verbose=True,
          anchor_idx=None):
    """Mini-batch Adam on a Huber (smooth L1) loss in standardised log space, with early stopping on
    validation loss. An L1-type loss targets the conditional median, which is what MAE rewards and which
    is unchanged by the expm1() back-transform. `inv` converts predictions back to ug/m3 for the MAE curves."""
    set_seed(seed)
    model = LSTMForecaster(Xtr.shape[2], hidden=hidden, anchor_idx=anchor_idx)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss(beta=0.05)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)), batch_size=batch,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    monitor = np.random.default_rng(seed).choice(len(Xtr), min(6000, len(Xtr)), replace=False)
    hist = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [], "lr": [], "seconds": []}
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guards against exploding gradients
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        p_va = predict(model, Xva)
        v_loss = float(loss_fn(torch.from_numpy(p_va), torch.from_numpy(Yva)))
        p_tr = predict(model, Xtr[monitor])
        hist["train_loss"].append(total / n)
        hist["val_loss"].append(v_loss)
        hist["train_mae"].append(mae(inv(p_tr), inv(Ytr[monitor])))
        hist["val_mae"].append(mae(inv(p_va), inv(Yva)))
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["seconds"].append(time.time() - t0)
        sched.step(v_loss)
        if verbose:
            print(f"epoch {ep + 1:02d}  train {total / n:.4f}  val {v_loss:.4f}  val MAE {hist['val_mae'][-1]:.2f} ug/m3")
        if v_loss < best - 1e-5:
            best, best_state, wait = v_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    hist["best_epoch"] = int(np.argmin(hist["val_loss"]) + 1)
    return model, hist


# --------------------------------------------------------------------------- evaluation
def horizon_table(y_true, y_pred):
    """MAE, RMSE, R^2 and mean bias for every forecast horizon (inputs in ug/m3)."""
    rows = []
    for h in range(y_true.shape[1]):
        yt, yp = y_true[:, h], y_pred[:, h]
        ok = ~np.isnan(yp)
        yt, yp = yt[ok], yp[ok]
        e = yp - yt
        rows.append({"horizon_h": h + 1, "MAE": mae(yp, yt), "RMSE": rmse(yp, yt),
                     "R2": float(1 - np.sum(e ** 2) / np.sum((yt - yt.mean()) ** 2)), "Bias": float(e.mean()),
                     "n": int(ok.sum())})
    return rows


def permutation_importance(model, X, Y, inv, names, repeats=5, seed=SEED):
    """Increase in validation RMSE (ug/m3, averaged over horizons) when one input channel is
    shuffled across windows; the whole 24-hour history of that feature is permuted together."""
    rng = np.random.default_rng(seed)
    truth = inv(Y)
    base = rmse(inv(predict(model, X)), truth)
    out = []
    for j, name in enumerate(names):
        incs = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, :, j] = X[rng.permutation(len(X)), :, j]
            incs.append(rmse(inv(predict(model, Xp)), truth) - base)
        out.append({"feature": name, "rmse_increase": float(np.mean(incs)), "sd": float(np.std(incs))})
    return base, sorted(out, key=lambda d: -d["rmse_increase"])
