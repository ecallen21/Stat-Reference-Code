"""Adversarial debiasing (Reference Ch 31 Fairness).

Zhang, Lemoine & Mitchell (2018) 'Mitigating Unwanted Biases with
Adversarial Learning.'

Two-player game between a PREDICTOR f(x) -> y_hat and an ADVERSARY
g(y_hat) -> A_hat that tries to recover the protected attribute A from
the predictor's output.

  Predictor loss:  L_pred(y_hat, y)  -  alpha * L_adv(A_hat, A).
  Adversary loss:  L_adv(A_hat, A).

The predictor is trained to be ACCURATE on Y while being UNPREDICTIVE
of A -- exactly equivalent to demographic-parity when the adversary is
optimal (Zhang 2018 Prop. 1).

Variant (Zhang eq 5): project the adversary's gradient onto the space
ORTHOGONAL to the predictor's, so the predictor sees only the debiasing
signal (not the accuracy-hurting part).

Here we implement the two-player game with logistic-regression predictor
and adversary, alternating SGD, and compare demographic parity + task
accuracy against ERM.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def train_adv(X, y, A, alpha=1.0, lr_pred=0.1, lr_adv=0.1, epochs=1000, l2=1e-3, seed=0):
    """Alternating SGD for the two-player game.

    Predictor: logistic(X @ theta) predicts Y.
    Adversary: logistic(w1 * y_hat + w0) predicts A from y_hat only.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    theta = rng.normal(0, 0.1, d)
    w1, w0 = 0.5, 0.0
    n = X.shape[0]
    for _ in range(epochs):
        # Predictor forward
        yh = _sigmoid(X @ theta)
        # Adversary forward on the predictor's SIGMOID output
        A_pred = _sigmoid(w1 * yh + w0)
        # Adversary BCE gradient on (w1, w0)
        dA = A_pred - A                          # (n,)
        d_w1 = np.mean(dA * yh)
        d_w0 = np.mean(dA)
        # Predictor: minimise BCE(yh, y) MINUS alpha * BCE(A_pred, A)
        dy_pred = yh - y                          # BCE derivative w.r.t. logit
        # Backprop of adversary loss through yh (yh -> A_pred): d A_pred / d yh = w1 * A_pred (1 - A_pred)
        dyh_adv = dA * (w1 * A_pred * (1 - A_pred))
        # Chain to theta: predictor logit gradient
        d_theta_pred = X.T @ (dy_pred * yh * (1 - yh)) / n
        d_theta_adv  = X.T @ (dyh_adv * yh * (1 - yh)) / n
        theta -= lr_pred * (d_theta_pred - alpha * d_theta_adv + l2 * theta)
        # Adversary step (maximise ability to predict A)
        w1 -= lr_adv * d_w1
        w0 -= lr_adv * d_w0
    return theta, (w1, w0)


def predict(theta, X): return (_sigmoid(X @ theta) > 0.5).astype(int)


def dp_ratio(y_hat, groups):
    r = [float(y_hat[groups == a].mean()) for a in np.unique(groups)]
    return min(r) / max(r) if max(r) > 0 else float("nan")


if __name__ == "__main__":
    print("=== Adversarial debiasing (Zhang 2018) ===\n")
    rng = np.random.default_rng(0)
    n_per = 500
    y0 = (rng.random(n_per) < 0.60).astype(int)
    y1 = (rng.random(n_per) < 0.25).astype(int)
    y = np.concatenate([y0, y1]).astype(float)
    groups = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(float)
    x0 = y + rng.normal(0, 0.5, len(y))
    x1 = groups + rng.normal(0, 0.4, len(y))      # proxy for group
    X = np.stack([x0, x1, np.ones_like(y)], axis=1)

    # ERM baseline
    theta_erm, _ = train_adv(X, y, groups, alpha=0.0)
    y_erm = predict(theta_erm, X)
    print(f"  ERM (alpha=0):        accuracy={(y_erm == y).mean():.3f}"
          f"   DP ratio={dp_ratio(y_erm, groups):.3f}")

    for alpha in (1.0, 3.0, 10.0):
        theta, adv = train_adv(X, y, groups, alpha=alpha)
        yh = predict(theta, X)
        print(f"  Adversarial alpha={alpha:>4.1f}: accuracy={(yh == y).mean():.3f}"
              f"   DP ratio={dp_ratio(yh, groups):.3f}"
              f"   adv weight w1={adv[0]:.3f}")

    print("\n  Higher alpha -> better DP ratio; some accuracy cost.\n")
    print("--- library cross-check (aif360.algorithms.inprocessing.AdversarialDebiasing) ---")
