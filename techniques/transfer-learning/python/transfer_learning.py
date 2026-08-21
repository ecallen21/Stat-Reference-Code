"""Transfer learning: feature extraction + fine-tuning (Reference §27.x extra).

Standard recipe: take a network pre-trained on a large upstream task, then
adapt to a downstream task with limited labels.

  * FEATURE EXTRACTION:  freeze the backbone, train only a fresh head.
  * FINE-TUNING:         also unfreeze the backbone (usually top layers first)
                          at a small LR.

Demo: pretrain a small MLP on a "source" 3-class classification task, then
transfer to a related "target" task with only 30 labelled examples.  Compare:
  (a) linear head trained from scratch on target features (no transfer)
  (b) frozen backbone + fresh head (feature extraction)
  (c) unfrozen backbone + fresh head (fine-tune) at low LR
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _relu(z): return np.maximum(z, 0.0)
def _relu_grad(z): return (z > 0).astype(float)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def train_mlp(X, y, hidden=(32, 16), n_classes: int = None,
              lr: float = 0.05, epochs: int = 300,
              init_W: dict = None, freeze_W: bool = False, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int); n, d = X.shape
    if n_classes is None: n_classes = int(y.max()) + 1
    sizes = [d] + list(hidden) + [n_classes]
    if init_W is None:
        Ws = [rng.normal(scale=np.sqrt(2.0 / sizes[i]),
                          size=(sizes[i], sizes[i + 1]))
              for i in range(len(sizes) - 1)]
    else:
        Ws = [w.copy() for w in init_W["Ws"]]
        # replace final layer with a fresh head sized to n_classes
        Ws[-1] = rng.normal(scale=np.sqrt(2.0 / sizes[-2]),
                             size=(sizes[-2], sizes[-1]))
    bs = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
    Y = np.zeros((n, n_classes)); Y[np.arange(n), y] = 1
    for ep in range(epochs):
        a = [X]; zs = []
        for k, (W, b) in enumerate(zip(Ws, bs)):
            z = a[-1] @ W + b; zs.append(z)
            a.append(_relu(z) if k < len(Ws) - 1 else _softmax(z))
        delta = (a[-1] - Y) / n
        for k in range(len(Ws) - 1, -1, -1):
            dW = a[k].T @ delta
            db = delta.sum(axis=0)
            if k > 0:
                delta = (delta @ Ws[k].T) * _relu_grad(zs[k - 1])
            # freeze backbone (all layers except the last)?
            if freeze_W and k < len(Ws) - 1:
                continue
            Ws[k] -= lr * dW; bs[k] -= lr * db
    logits = X
    for k, (W, b) in enumerate(zip(Ws, bs)):
        logits = _relu(logits @ W + b) if k < len(Ws) - 1 else logits @ W + b
    acc = float((logits.argmax(axis=1) == y).mean())
    return {"Ws": Ws, "bs": bs, "sizes": sizes, "train_acc": acc}


def predict(X, model):
    a = X
    for k, (W, b) in enumerate(zip(model["Ws"], model["bs"])):
        a = _relu(a @ W + b) if k < len(model["Ws"]) - 1 else a @ W + b
    return a.argmax(axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Source task: 3-class classification of 2D Gaussians (large data)
    def _make(n_per=200, centres=None, seed=0):
        rr = np.random.default_rng(seed)
        if centres is None:
            centres = [(0, 3), (-2, -1), (2, -1)]
        X = np.vstack([rr.normal(loc=c, size=(n_per, 2)) for c in centres])
        y = np.hstack([np.full(n_per, k) for k in range(len(centres))])
        return X, y

    X_src, y_src = _make(n_per=400, seed=0)
    src_model = train_mlp(X_src, y_src, hidden=(32, 16), lr=0.05, epochs=300)
    print(f"=== Source task trained: train acc = {src_model['train_acc']:.3f} ===")

    # Target task: 4-class classification in the same 2D space (related but different)
    tgt_centres = [(1, 3), (-2, 0), (2, -2), (-1, -3)]
    X_tgt, y_tgt = _make(n_per=200, centres=tgt_centres, seed=1)
    n_small = 12                                              # very few labels favours transfer
    idx = rng.permutation(len(X_tgt))
    X_tr, X_te = X_tgt[idx[:n_small]], X_tgt[idx[n_small:]]
    y_tr, y_te = y_tgt[idx[:n_small]], y_tgt[idx[n_small:]]

    # (a) scratch
    scratch = train_mlp(X_tr, y_tr, hidden=(32, 16), n_classes=4, lr=0.05, epochs=300)
    acc_scr = float((predict(X_te, scratch) == y_te).mean())

    # (b) feature extraction from source (freeze backbone, fresh head)
    feat = train_mlp(X_tr, y_tr, hidden=(32, 16), n_classes=4, lr=0.05, epochs=300,
                      init_W=src_model, freeze_W=True)
    acc_feat = float((predict(X_te, feat) == y_te).mean())

    # (c) fine-tune (unfrozen) at low LR
    ft = train_mlp(X_tr, y_tr, hidden=(32, 16), n_classes=4, lr=0.005, epochs=300,
                    init_W=src_model, freeze_W=False)
    acc_ft = float((predict(X_te, ft) == y_te).mean())

    print(f"\n=== Downstream: 4-class task, {n_small} training examples ===")
    print(f"  (a) from scratch          : test acc = {acc_scr:.3f}")
    print(f"  (b) feature extraction     : test acc = {acc_feat:.3f}")
    print(f"  (c) fine-tune (lr=0.005)   : test acc = {acc_ft:.3f}")

    print("\n--- library cross-check (torch: freeze via param.requires_grad = False) ---")
