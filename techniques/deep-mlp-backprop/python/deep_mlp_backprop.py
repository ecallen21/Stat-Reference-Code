"""Deep MLP with manual back-propagation (Reference §27.1).

Extends the single-hidden-layer perceptron (see `neural-network-mlp`) to
arbitrary depth.  Layers:

    z^l = W^l a^{l-1} + b^l
    a^l = phi(z^l)                    ReLU on hidden layers
    a^L = softmax(z^L)                on the output layer for classification

Loss: cross-entropy.  Back-prop:

    delta^L = a^L - y                 (softmax + CE combined gradient)
    delta^l = (W^{l+1})^T delta^{l+1} * phi'(z^l)
    dW^l = delta^l (a^{l-1})^T
    db^l = delta^l

Optimiser: mini-batch SGD (Adam / AdamW are covered in `adam-optimizer`).
He initialisation for ReLU layers to keep activations well-scaled at depth.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def _relu(z): return np.maximum(z, 0.0)
def _relu_grad(z): return (z > 0).astype(float)


def _he_init(rng, fan_in, fan_out):
    return rng.normal(scale=np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


def fit_mlp(X, y, hidden=(64, 64), n_classes: int = None,
            lr: float = 0.05, epochs: int = 200, batch: int = 64,
            seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    n, d = X.shape
    if n_classes is None:
        n_classes = int(y.max()) + 1
    sizes = [d] + list(hidden) + [n_classes]
    Ws = [_he_init(rng, sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    bs = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
    Y = np.zeros((n, n_classes)); Y[np.arange(n), y] = 1

    losses = []
    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch):
            b = idx[start: start + batch]
            xb = X[b]; yb = Y[b]
            # forward
            a = [xb]; zs = []
            for k, (W, bias) in enumerate(zip(Ws, bs)):
                z = a[-1] @ W + bias; zs.append(z)
                a.append(_relu(z) if k < len(Ws) - 1 else _softmax(z))
            # backward (softmax + CE combined)
            delta = (a[-1] - yb) / len(b)
            for k in range(len(Ws) - 1, -1, -1):
                dW = a[k].T @ delta
                db = delta.sum(axis=0)
                if k > 0:
                    delta = (delta @ Ws[k].T) * _relu_grad(zs[k - 1])
                Ws[k] -= lr * dW; bs[k] -= lr * db
        # epoch loss
        probs = _softmax(_forward_head(X, Ws, bs))
        losses.append(float(-np.log(probs[np.arange(n), y] + 1e-12).mean()))
    return {"Ws": Ws, "bs": bs, "losses": losses, "sizes": sizes,
            "method": f"Deep MLP {sizes} + ReLU + softmax-CE, SGD"}


def _forward_head(X, Ws, bs):
    a = X
    for k, (W, b) in enumerate(zip(Ws, bs)):
        z = a @ W + b
        a = _relu(z) if k < len(Ws) - 1 else z
    return a


def predict(X, model):
    return _softmax(_forward_head(X, model["Ws"], model["bs"])).argmax(axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 3-class 2D spirals
    n_per = 200; C = 3
    X = []; y = []
    for c in range(C):
        r = np.linspace(0.1, 1.0, n_per)
        theta = np.linspace(c * 4, (c + 1) * 4, n_per) + rng.normal(scale=0.2, size=n_per)
        X.append(np.column_stack([r * np.sin(theta), r * np.cos(theta)]))
        y.append(np.full(n_per, c))
    X = np.vstack(X); y = np.hstack(y)
    idx = rng.permutation(len(X)); X, y = X[idx], y[idx]
    n_tr = 450
    X_tr, X_te = X[:n_tr], X[n_tr:]; y_tr, y_te = y[:n_tr], y[n_tr:]

    m = fit_mlp(X_tr, y_tr, hidden=(32, 32), lr=0.1, epochs=400, batch=64)
    acc_tr = float((predict(X_tr, m) == y_tr).mean())
    acc_te = float((predict(X_te, m) == y_te).mean())
    print(f"=== Deep MLP (sizes {m['sizes']}, ReLU + softmax-CE) ===")
    print(f"  train accuracy = {acc_tr:.3f}")
    print(f"  test  accuracy = {acc_te:.3f}   (3-class 2D spirals, n_test={len(y_te)})")
    print(f"  final training cross-entropy = {m['losses'][-1]:.4f}")

    print("\n--- library cross-check (sklearn.neural_network.MLPClassifier) ---")
    try:
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=1500,
                            solver="adam", random_state=0).fit(X_tr, y_tr)
        print(f"  sklearn MLP test accuracy = {clf.score(X_te, y_te):.3f}")
    except ImportError:
        print("  (sklearn not installed)")
