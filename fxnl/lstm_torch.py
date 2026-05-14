"""Small single-layer LSTM classifier (PyTorch)."""

from __future__ import annotations

from typing import Any

import numpy as np


def sequences_for_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Train sequences end before ``n_train``; test sequences end at/after ``n_train`` (global row index)."""
    n = len(X)
    if n_train < seq_len + 5 or n - n_train < seq_len + 5:
        return None
    Xi_tr, yi_tr, Xi_te, yi_te = [], [], [], []
    for i in range(seq_len - 1, n):
        seq = X[i - seq_len + 1 : i + 1]
        yi = int(y[i])
        if i < n_train:
            Xi_tr.append(seq)
            yi_tr.append(yi)
        else:
            Xi_te.append(seq)
            yi_te.append(yi)
    if len(Xi_te) < 5 or len(Xi_tr) < 5:
        return None
    return (
        np.asarray(Xi_tr, dtype=np.float32),
        np.asarray(yi_tr, dtype=np.int64),
        np.asarray(Xi_te, dtype=np.float32),
        np.asarray(yi_te, dtype=np.int64),
    )


def fit_predict_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_train_rows: int,
    n_classes: int,
    seq_len: int = 20,
    hidden: int = 48,
    epochs: int = 35,
    lr: float = 0.02,
    random_state: int = 42,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    pack = sequences_for_split(X_full, y_full, n_train=n_train_rows, seq_len=seq_len)
    if pack is None:
        return {"error": "insufficient_rows_for_lstm"}
    Xtr, ytr, Xte, yte = pack

    torch.manual_seed(random_state)
    _, _, f = Xtr.shape
    device = torch.device("cpu")

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(f, hidden, batch_first=True, num_layers=1)
            self.fc = nn.Linear(hidden, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            o, _ = self.lstm(x)
            return self.fc(o[:, -1, :])

    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(Xt)
        loss = crit(logits, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        tr_logits = model(torch.from_numpy(Xtr).to(device)).cpu().numpy()
        te_logits = model(torch.from_numpy(Xte).to(device)).cpu().numpy()

    def _proba(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    tr_proba = _proba(tr_logits)
    te_proba = _proba(te_logits)
    tr_pred = tr_proba.argmax(axis=1)
    te_pred = te_proba.argmax(axis=1)
    return {
        "y_train_true": ytr,
        "y_test_true": yte,
        "y_train_pred": tr_pred,
        "y_test_pred": te_pred,
        "train_proba": tr_proba,
        "test_proba": te_proba,
        "seq_len": seq_len,
        "hidden": hidden,
        "epochs": epochs,
    }
