"""Online / streaming linear models via SGD (Reference §21.x extra).

One example at a time; parameter update:

    w <- w - eta_t * grad_loss(y, x, w)

Learning rate schedules:
  * constant:      eta_t = eta_0
  * inverse-time:  eta_t = eta_0 / (1 + t / t0)
  * inverse-sqrt:  eta_t = eta_0 / sqrt(t + 1)

Losses implemented:
  * squared error  (regression)
  * log-loss        (logistic regression)
  * hinge           (SVM-style)
  * passive-aggressive (Crammer et al. 2006)

Concept drift: with a fixed learning rate the model tracks the current
distribution; inverse-time collapses on the initial regime.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def sgd_online(X, y, loss: str = "squared",
               eta0: float = 0.05, schedule: str = "invsqrt") -> dict:
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    w = np.zeros(p)
    losses = np.zeros(n)
    for t in range(n):
        xi = X[t]; yi = y[t]
        if schedule == "constant":
            eta = eta0
        elif schedule == "invsqrt":
            eta = eta0 / np.sqrt(t + 1)
        elif schedule == "invtime":
            eta = eta0 / (1 + t / 100)
        else:
            raise ValueError(schedule)
        # gradient step per loss
        if loss == "squared":
            pred = xi @ w
            g = (pred - yi) * xi
            losses[t] = 0.5 * (pred - yi) ** 2
        elif loss == "log":
            pred = _sigmoid(xi @ w)
            g = (pred - yi) * xi
            losses[t] = -yi * np.log(pred + 1e-12) - (1 - yi) * np.log(1 - pred + 1e-12)
        elif loss == "hinge":
            margin = yi * (xi @ w)
            g = -yi * xi if margin < 1 else np.zeros_like(xi)
            losses[t] = max(0.0, 1.0 - margin)
        elif loss == "passive_aggressive":
            margin = yi * (xi @ w)
            loss_t = max(0.0, 1.0 - margin)
            tau = loss_t / (np.dot(xi, xi) + 1e-12)
            w += tau * yi * xi
            losses[t] = loss_t
            continue
        else:
            raise ValueError(loss)
        w -= eta * g
    return {"w": w, "losses": losses,
            "mean_loss_last_20pct": float(losses[int(0.8 * n):].mean()),
            "method": f"online SGD ({loss}, {schedule})"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 5000
    p = 5
    w_true = rng.normal(scale=1.5, size=p)

    # regression stream
    Xr = rng.normal(size=(n, p))
    yr = Xr @ w_true + rng.normal(scale=0.5, size=n)
    fit = sgd_online(Xr, yr, loss="squared", eta0=0.05, schedule="invsqrt")
    print(f"=== Online SGD, squared loss (n=5000, p=5) ===")
    print(f"  ||w_hat - w_true|| = {np.linalg.norm(fit['w'] - w_true):.4f}")
    print(f"  mean loss last 20% of stream = {fit['mean_loss_last_20pct']:.4f}   "
          f"(noise variance 0.5^2 / 2 = 0.125)")

    # logistic stream
    yb = (rng.uniform(size=n) < _sigmoid(Xr @ w_true)).astype(int)
    fit = sgd_online(Xr, yb, loss="log", eta0=0.05, schedule="invsqrt")
    print(f"\n=== Online SGD, log loss ===")
    print(f"  ||w_hat - w_true|| = {np.linalg.norm(fit['w'] - w_true):.4f}")
    print(f"  final mean log-loss = {fit['mean_loss_last_20pct']:.4f}")

    # hinge / SVM stream
    ys = np.where(Xr @ w_true > 0, 1.0, -1.0)
    fit = sgd_online(Xr, ys, loss="hinge", eta0=0.05, schedule="invsqrt")
    err = float((np.sign(Xr @ fit["w"]) != ys).mean())
    print(f"\n=== Online SGD, hinge loss ===")
    print(f"  classification error on stream = {err:.3f}")

    # passive-aggressive (no learning rate)
    fit = sgd_online(Xr, ys, loss="passive_aggressive")
    err = float((np.sign(Xr @ fit["w"]) != ys).mean())
    print(f"\n=== Passive-Aggressive online learner ===")
    print(f"  classification error on stream = {err:.3f}")

    # concept drift: flip sign of w_true mid-stream, use constant learning rate
    yr_drift = np.hstack([Xr[:n // 2] @ w_true, Xr[n // 2:] @ (-w_true)]) \
               + rng.normal(scale=0.5, size=n)
    fit = sgd_online(Xr, yr_drift, loss="squared", eta0=0.05, schedule="constant")
    print(f"\n=== Concept drift @ t = n/2 (constant lr recovers) ===")
    print(f"  ||w_hat - (-w_true)|| = {np.linalg.norm(fit['w'] + w_true):.4f}")

    print("\n--- library cross-check (sklearn SGDRegressor / SGDClassifier) ---")
    try:
        from sklearn.linear_model import SGDRegressor
        m = SGDRegressor(loss="squared_error", eta0=0.05, learning_rate="invscaling",
                          fit_intercept=False, max_iter=1, tol=None, random_state=0)
        m.partial_fit(Xr, yr)
        print(f"  sklearn ||w - w_true|| (single epoch) = "
              f"{np.linalg.norm(m.coef_ - w_true):.4f}")
    except ImportError:
        print("  (sklearn not installed)")
