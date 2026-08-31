"""Fast Gradient Sign Method attack (Reference Ch 30 Robustness).

Goodfellow, Shlens & Szegedy (2014) "Explaining and Harnessing Adversarial
Examples."  Cheapest first-order white-box attack:

  x_adv = x + eps * sign( grad_x L(f(x), y) )     under L_inf ball radius eps.

For untargeted attacks the perturbation moves each input coordinate one
step in the sign of the loss gradient. FGSM works because near-linear
behaviour of a network in a small neighbourhood makes the sign of the
gradient a good approximate first-order attack direction.

Empirically, tiny eps (a few percent of input range) is enough to flip
predictions on undefended models.

Here we implement FGSM against a logistic-regression classifier on a
synthetic 2-class problem so that the gradient step, sign quantisation
and epsilon budget are all fully transparent.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic(X, y, lr=0.5, epochs=400, l2=1e-3):
    d = X.shape[1]
    beta = np.zeros(d)
    n = X.shape[0]
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        g = X.T @ (p - y) / n + l2 * beta
        beta -= lr * g
    return beta


def predict(beta, X):
    return (_sigmoid(X @ beta) > 0.5).astype(int)


def fgsm(beta, x, y, eps):
    """FGSM perturbation for a single row x under logistic BCE loss."""
    p = _sigmoid(x @ beta)
    grad = (p - y) * beta                 # dL/dx for logistic BCE
    return x + eps * np.sign(grad)


if __name__ == "__main__":
    print("=== FGSM white-box attack on logistic regression (Goodfellow 2014) ===\n")
    rng = np.random.default_rng(0)
    d = 5
    # Two Gaussian blobs separated along a random direction.
    beta_true = rng.normal(0, 1, d)
    beta_true /= np.linalg.norm(beta_true)
    n = 400
    X = rng.normal(0, 1, (n, d))
    y = (X @ beta_true + rng.normal(0, 0.3, n) > 0).astype(int)

    beta = train_logistic(X, y)

    # Clean accuracy on a held-out set.
    X_te = rng.normal(0, 1, (500, d))
    y_te = (X_te @ beta_true + rng.normal(0, 0.3, 500) > 0).astype(int)
    y_hat = predict(beta, X_te)
    print(f"  clean accuracy: {(y_hat == y_te).mean():.3f}")

    for eps in (0.01, 0.05, 0.10, 0.20, 0.50):
        X_adv = np.stack([fgsm(beta, X_te[i], y_te[i], eps) for i in range(500)])
        y_adv = predict(beta, X_adv)
        adv_acc = (y_adv == y_te).mean()
        pert_norm = np.linalg.norm(X_adv - X_te, ord=np.inf, axis=1).mean()
        print(f"  eps={eps:>4.2f}  L_inf(pert)={pert_norm:.3f}  "
              f"adv accuracy={adv_acc:.3f}  drop={y_hat.mean() > 0 and (y_hat==y_te).mean()-adv_acc:.3f}")

    print("\n  Small eps is enough to flip most predictions on an undefended model.\n")
    print("--- library cross-check (foolbox / cleverhans / advertorch) ---")
