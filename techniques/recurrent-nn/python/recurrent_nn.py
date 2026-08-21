"""Recurrent Neural Network (Elman) with BPTT (Reference §27.3).

Simple recurrent cell:
    h_t = tanh(W_x x_t + W_h h_{t-1} + b_h)
    y_t = W_y h_t + b_y

Loss: sequence-level cross-entropy on the last output (many-to-one classifier).

Backprop-through-time (BPTT):
    dh_t = (W_y)^T dy_t + (W_h)^T dh_{t+1}
    dW_h += dh_t . tanh'(z_t) . h_{t-1}^T,   etc.

Well-known limitation: vanishing / exploding gradients across long sequences.
Solutions:
    * Gradient clipping (implemented).
    * LSTM / GRU gating (see `lstm-gru`).
    * Truncated BPTT (limit unroll length).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)


def fit_rnn(X, y, hidden: int = 16, lr: float = 0.05, epochs: int = 200,
            clip: float = 1.0, seed: int = 0) -> dict:
    """X: list of length-T sequences (T can vary), each token as an integer id.
       y: array of class labels."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    vocab = sorted({tok for seq in X for tok in seq})
    tok_ix = {t: i for i, t in enumerate(vocab)}
    V = len(vocab); n_classes = int(y.max()) + 1
    Wx = rng.normal(scale=0.1, size=(V, hidden))
    Wh = rng.normal(scale=0.1, size=(hidden, hidden))
    b_h = np.zeros(hidden)
    Wy = rng.normal(scale=0.1, size=(hidden, n_classes))
    b_y = np.zeros(n_classes)

    losses = []
    for ep in range(epochs):
        total_loss = 0.0
        for xi, yi in zip(X, y):
            T = len(xi)
            hs = [np.zeros(hidden)]
            zs = []
            for t in range(T):
                x_t = Wx[tok_ix[xi[t]]]
                z = x_t + hs[-1] @ Wh + b_h
                zs.append(z); hs.append(np.tanh(z))
            logits = hs[-1] @ Wy + b_y
            probs = _softmax(logits)
            total_loss += float(-np.log(probs[yi] + 1e-12))
            # backward
            dlogits = probs.copy(); dlogits[yi] -= 1.0
            dWy = np.outer(hs[-1], dlogits)
            db_y = dlogits
            dh = dlogits @ Wy.T
            dWx = np.zeros_like(Wx); dWh = np.zeros_like(Wh); db_h = np.zeros_like(b_h)
            for t in range(T - 1, -1, -1):
                dz = dh * (1 - np.tanh(zs[t]) ** 2)
                dWx[tok_ix[xi[t]]] += dz
                dWh += np.outer(hs[t], dz)
                db_h += dz
                dh = dz @ Wh.T
            # gradient clipping (elementwise)
            for grad in (dWx, dWh, dWy, db_h, db_y):
                np.clip(grad, -clip, clip, out=grad)
            Wx -= lr * dWx; Wh -= lr * dWh; Wy -= lr * dWy
            b_h -= lr * db_h; b_y -= lr * db_y
        losses.append(total_loss / len(X))

    return {"vocab": vocab, "tok_ix": tok_ix, "Wx": Wx, "Wh": Wh,
            "b_h": b_h, "Wy": Wy, "b_y": b_y,
            "losses": losses,
            "method": "Elman RNN + BPTT with gradient clipping"}


def predict_rnn(X, model):
    hidden = model["Wh"].shape[0]; preds = []
    for xi in X:
        h = np.zeros(hidden)
        for t in xi:
            x_t = model["Wx"][model["tok_ix"][t]]
            h = np.tanh(x_t + h @ model["Wh"] + model["b_h"])
        preds.append((h @ model["Wy"] + model["b_y"]).argmax())
    return np.array(preds)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # sequence classification: "does the sequence END with 'a' or 'b'?"
    # (only the LAST token matters — RNN handles short-range dependencies well.
    #  A "first-token" variant would exhibit the vanishing-gradient failure that
    #  motivates LSTM / GRU — see the caveats + `lstm-gru`.)
    n = 300
    X = []; y = []
    for _ in range(n):
        last = rng.choice(["a", "b"])
        length = rng.integers(4, 9)
        seq = list(rng.choice(list("abcdef"), size=length - 1)) + [last]
        X.append(seq); y.append(0 if last == "a" else 1)
    y = np.array(y)
    # 80 / 20 train / test split
    idx = rng.permutation(n); X = [X[i] for i in idx]; y = y[idx]
    n_tr = 240
    X_tr, X_te = X[:n_tr], X[n_tr:]; y_tr, y_te = y[:n_tr], y[n_tr:]

    m = fit_rnn(X_tr, y_tr, hidden=12, lr=0.05, epochs=40)
    acc_tr = float((predict_rnn(X_tr, m) == y_tr).mean())
    acc_te = float((predict_rnn(X_te, m) == y_te).mean())
    print(f"=== Elman RNN + BPTT on \"last-token = a?\" (n_train={n_tr}) ===")
    print(f"  train accuracy = {acc_tr:.3f}")
    print(f"  test  accuracy = {acc_te:.3f}")
    print(f"  final training loss = {m['losses'][-1]:.4f}")

    print("\n--- library cross-check (torch.nn.RNN + Linear head) ---")
    try:
        import torch
        import torch.nn as nn
        vocab = m["vocab"]; V = len(vocab); tok_ix = m["tok_ix"]
        def _pad(seqs, T=None):
            if T is None:
                T = max(len(s) for s in seqs)
            out = np.zeros((len(seqs), T), dtype=int)
            for i, s in enumerate(seqs):
                for j, t in enumerate(s):
                    out[i, j] = tok_ix[t]
            return out
        Xtr = torch.tensor(_pad(X_tr, T=15))
        Xte = torch.tensor(_pad(X_te, T=15))
        emb = nn.Embedding(V, 12)
        rnn = nn.RNN(12, 12, batch_first=True)
        head = nn.Linear(12, 2)
        params = list(emb.parameters()) + list(rnn.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=0.03)
        loss_fn = nn.CrossEntropyLoss()
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        for _ in range(60):
            opt.zero_grad()
            out, h = rnn(emb(Xtr))
            logits = head(out[:, -1])
            loss = loss_fn(logits, y_tr_t); loss.backward(); opt.step()
        with torch.no_grad():
            _, hte = rnn(emb(Xte))
            pred_te = head(_[:, -1]).argmax(dim=1).numpy()
            _, htr = rnn(emb(Xtr))
            pred_tr = head(_[:, -1]).argmax(dim=1).numpy()
        print(f"  torch RNN train = {(pred_tr == y_tr).mean():.3f}   test = {(pred_te == y_te).mean():.3f}")
    except ImportError:
        print("  (pytorch not installed)")
