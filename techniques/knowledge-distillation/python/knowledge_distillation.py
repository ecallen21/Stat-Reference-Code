"""Knowledge distillation (Hinton et al. 2015; Reference §27.x extra).

Train a small STUDENT model to match a large TEACHER's softmax outputs.

Combined loss:
    L = alpha * CE(y_true, p_student)
      + (1 - alpha) * T^2 * KL( softmax(z_teacher / T) || softmax(z_student / T) )

T (temperature) softens both distributions, exposing "dark knowledge"
(relative probabilities of wrong classes).

Demo: train a big teacher on a 3-class task; distil into a tiny student.
The distilled student should beat a student trained ONLY on hard labels
of the same size, especially when labels are scarce.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _relu(z): return np.maximum(z, 0.0)
def _relu_grad(z): return (z > 0).astype(float)


def _softmax(z, T: float = 1.0):
    z = z / T; z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def fit_mlp(X, y, hidden=(64,), n_classes: int = 3, lr: float = 0.05,
            epochs: int = 300, seed: int = 0,
            teacher_logits=None, alpha: float = 0.7, T: float = 3.0):
    """Fit an MLP.  If teacher_logits given, distil: mix CE + KL to teacher-softened logits."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int); n, d = X.shape
    sizes = [d] + list(hidden) + [n_classes]
    Ws = [rng.normal(scale=np.sqrt(2.0 / sizes[i]), size=(sizes[i], sizes[i + 1]))
          for i in range(len(sizes) - 1)]
    bs = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
    Y = np.zeros((n, n_classes)); Y[np.arange(n), y] = 1
    if teacher_logits is not None:
        Q_soft = _softmax(teacher_logits, T=T)                # teacher's soft targets
    for _ in range(epochs):
        a = [X]; zs = []
        for k, (W, b) in enumerate(zip(Ws, bs)):
            z = a[-1] @ W + b; zs.append(z)
            a.append(_relu(z) if k < len(Ws) - 1 else z)
        logits = a[-1]
        P_hard = _softmax(logits)
        d_logits_hard = (P_hard - Y) / n
        if teacher_logits is None:
            d_logits = d_logits_hard
        else:
            P_soft = _softmax(logits, T=T)
            # gradient of T^2 * KL(Q_soft || P_soft) wrt student's UN-scaled logits is
            # T * (P_soft - Q_soft) (chain-rule through the 1/T scaling)
            d_logits_kd = T * (P_soft - Q_soft) / n
            d_logits = alpha * d_logits_hard + (1 - alpha) * d_logits_kd
        delta = d_logits
        for k in range(len(Ws) - 1, -1, -1):
            dW = a[k].T @ delta
            db = delta.sum(axis=0)
            if k > 0:
                delta = (delta @ Ws[k].T) * _relu_grad(zs[k - 1])
            Ws[k] -= lr * dW; bs[k] -= lr * db
    def _pred(X):
        h = X
        for k, (W, b) in enumerate(zip(Ws, bs)):
            h = _relu(h @ W + b) if k < len(Ws) - 1 else h @ W + b
        return h
    return {"Ws": Ws, "bs": bs, "predict_logits": _pred,
            "method": "distilled MLP" if teacher_logits is not None else "hard-label MLP"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Task: 3-class 2D
    def _make(n_per=200, seed=0):
        rr = np.random.default_rng(seed)
        centres = [(0, 3), (-2, -1), (2, -1)]
        X = np.vstack([rr.normal(loc=c, size=(n_per, 2)) for c in centres])
        y = np.hstack([np.full(n_per, k) for k in range(3)])
        return X, y

    X, y = _make(n_per=300, seed=0)
    Xte, yte = _make(n_per=200, seed=1)
    idx = rng.permutation(len(X)); X, y = X[idx], y[idx]

    # 1. TEACHER: wide MLP, all labels available
    teacher = fit_mlp(X, y, hidden=(64, 32), epochs=400, lr=0.05)
    logits_te_teach = teacher["predict_logits"](Xte)
    acc_teacher = float((logits_te_teach.argmax(axis=1) == yte).mean())

    # 2. STUDENT (small), only HARD labels on a small subset
    n_small = 60
    Xs, ys = X[:n_small], y[:n_small]
    student_hard = fit_mlp(Xs, ys, hidden=(6,), epochs=500, lr=0.05)
    acc_hard = float((student_hard["predict_logits"](Xte).argmax(axis=1) == yte).mean())

    # 3. STUDENT (small), distilled from teacher on the SAME small subset
    tlogits_small = teacher["predict_logits"](Xs)
    student_kd = fit_mlp(Xs, ys, hidden=(6,), epochs=500, lr=0.05,
                          teacher_logits=tlogits_small, alpha=0.3, T=4.0)
    acc_kd = float((student_kd["predict_logits"](Xte).argmax(axis=1) == yte).mean())

    # 4. Bonus: distil on UNLABELED data too (transfer only via teacher probabilities)
    #    Small labelled + more unlabelled (transfer set).
    n_unlab = 240
    X_unlab = X[n_small: n_small + n_unlab]
    y_placeholder = np.zeros(len(X_unlab), dtype=int)     # not used in loss (alpha = 0)
    all_X = np.vstack([Xs, X_unlab])
    all_y = np.hstack([ys, y_placeholder])
    tlogits_big = teacher["predict_logits"](all_X)
    student_kd2 = fit_mlp(all_X, all_y, hidden=(6,), epochs=500, lr=0.05,
                           teacher_logits=tlogits_big, alpha=0.15, T=4.0)
    acc_kd2 = float((student_kd2["predict_logits"](Xte).argmax(axis=1) == yte).mean())

    print(f"=== Knowledge distillation ({n_small} labels; student hidden=(6,)) ===")
    print(f"  teacher (hidden=(64, 32), all labels): test acc = {acc_teacher:.3f}")
    print(f"  student hard-labels only              : test acc = {acc_hard:.3f}")
    print(f"  student distilled from teacher (KD)   : test acc = {acc_kd:.3f}")
    print(f"  student distilled + extra unlabelled  : test acc = {acc_kd2:.3f}")

    print("\n--- library cross-check (torch: KL(teacher || student) with temperature) ---")
