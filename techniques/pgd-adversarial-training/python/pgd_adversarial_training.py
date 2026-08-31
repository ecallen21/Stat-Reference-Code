"""PGD attack + adversarial training (Reference Ch 30 Robustness).

Madry, Makelov, Schmidt, Tsipras & Vladu (2018) "Towards Deep Learning
Models Resistant to Adversarial Attacks."

PROJECTED GRADIENT DESCENT is the multi-step generalisation of FGSM:

  Repeat T steps:
    x_adv <- Proj_{|x_adv - x|_inf <= eps} ( x_adv + alpha * sign( grad_x L ) )

Optionally start from a random point inside the eps ball (Madry's "random
start"), which prevents attack failure at exactly zero gradient.

ADVERSARIAL TRAINING is the saddle-point objective

  min_theta  E_(x,y) [ max_{|delta|_inf <= eps} L(f_theta(x + delta), y) ]

approximated by generating PGD adversarial examples during training and
minimising loss on THEM (not the clean examples).

Here we implement PGD attack + PGD-adversarial-training for logistic
regression, and compare clean/PGD accuracies with and without adversarial
training.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _bce_grad_x(beta, x, y):
    p = _sigmoid(x @ beta)
    return (p - y) * beta                      # dL/dx (per row)


def pgd_attack(beta, X, y, eps=0.20, alpha=0.05, T=20, rng=None):
    if rng is not None:
        X_adv = X + rng.uniform(-eps, eps, X.shape)
    else:
        X_adv = X.copy()
    for _ in range(T):
        # Per-row gradient
        p = _sigmoid((X_adv * beta).sum(axis=1))
        grad = (p - y)[:, None] * beta
        X_adv = X_adv + alpha * np.sign(grad)
        # Project onto L_inf ball around X
        X_adv = np.clip(X_adv, X - eps, X + eps)
    return X_adv


def _predict(beta, X):
    return (_sigmoid(X @ beta) > 0.5).astype(int)


def train_clean(X, y, lr=0.5, epochs=400, l2=1e-3):
    d = X.shape[1]
    beta = np.zeros(d)
    n = X.shape[0]
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        g = X.T @ (p - y) / n + l2 * beta
        beta -= lr * g
    return beta


def train_adversarial(X, y, eps=0.20, alpha=0.05, T=7, lr=0.5, epochs=400, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    beta = np.zeros(d)
    n = X.shape[0]
    for _ in range(epochs):
        # Inner max: generate PGD adversaries against current beta
        X_adv = pgd_attack(beta, X, y, eps=eps, alpha=alpha, T=T, rng=rng)
        # Outer min: SGD step on adversarial loss
        p = _sigmoid(X_adv @ beta)
        g = X_adv.T @ (p - y) / n + l2 * beta
        beta -= lr * g
    return beta


if __name__ == "__main__":
    print("=== PGD attack + Madry-style adversarial training ===\n")
    rng = np.random.default_rng(0)
    d = 5
    beta_true = rng.normal(0, 1, d); beta_true /= np.linalg.norm(beta_true)
    n = 400
    X = rng.normal(0, 1, (n, d))
    y = (X @ beta_true + rng.normal(0, 0.3, n) > 0).astype(int)
    X_te = rng.normal(0, 1, (500, d))
    y_te = (X_te @ beta_true + rng.normal(0, 0.3, 500) > 0).astype(int)

    beta_clean = train_clean(X, y)
    beta_adv = train_adversarial(X, y, eps=0.20, alpha=0.05, T=7, epochs=400)

    for name, beta in (("clean-trained", beta_clean), ("PGD-trained ", beta_adv)):
        clean_acc = (_predict(beta, X_te) == y_te).mean()
        for eps in (0.05, 0.10, 0.20, 0.40):
            X_adv = pgd_attack(beta, X_te, y_te, eps=eps, alpha=eps / 4, T=40,
                                rng=np.random.default_rng(42))
            adv_acc = (_predict(beta, X_adv) == y_te).mean()
            print(f"  {name}  eps={eps:.2f}  clean_acc={clean_acc:.3f}  PGD_acc={adv_acc:.3f}")
        print()

    print("  Adversarial training trades a little clean accuracy for much better PGD robustness.\n")
    print("--- library cross-check (foolbox / cleverhans / advertorch / torchattacks) ---")
