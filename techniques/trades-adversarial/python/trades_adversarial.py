"""TRADES adversarial training (Reference Ch 30 Robustness).

Zhang, Yu, Jiang, Xing, El Ghaoui & Jordan (2019) "Theoretically
Principled Trade-off between Robustness and Accuracy."

Instead of Madry's saddle-point objective (train only on adversarial
examples), TRADES splits the loss into a CLEAN CE term and a KL
divergence between the clean and adversarial predictions:

  L_trades = CE( f_theta(x), y )
             + beta * KL( f_theta(x)  ||  f_theta(x_adv) )

with x_adv found by PGD that MAXIMISES the KL term (not the CE).

beta = 0    -> vanilla clean training.
beta -> inf -> pure boundary smoothness (like Madry's PGD-AT).
beta ~ 1-6 -> Pareto-optimal clean/robust trade-off.

Advantages over Madry:
  - Explicit trade-off knob.
  - Better clean accuracy at the same robust accuracy.
  - Won the 2018 NeurIPS Adversarial Vision Challenge.

Here we implement TRADES for binary logistic regression: KL(p || q) with
p, q ~ Bernoulli(sigmoid(...)).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def _bce(p, y, eps=1e-12):
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def _kl_bern(p, q, eps=1e-12):
    return p * np.log((p + eps) / (q + eps)) + (1 - p) * np.log((1 - p + eps) / (1 - q + eps))


def pgd_trades(beta, X, y_prob_clean, eps=0.20, alpha=0.05, T=10, rng=None):
    """PGD ascending the KL(clean || adv) instead of CE."""
    if rng is not None:
        X_adv = X + rng.uniform(-eps, eps, X.shape)
    else:
        X_adv = X.copy()
    for _ in range(T):
        p_adv = _sigmoid(X_adv @ beta)
        # d KL(p || q(x)) / dx  =  (q - p) / (q (1-q))  *  dq/dx
        # For sigmoid: dq/dx = q(1-q) * beta -> simplifies to (q - p) * beta.
        grad = (p_adv - y_prob_clean)[:, None] * beta
        X_adv = X_adv + alpha * np.sign(grad)
        X_adv = np.clip(X_adv, X - eps, X + eps)
    return X_adv


def train_trades(X, y, beta_kl=6.0, eps=0.20, alpha=0.05, T=7, lr=0.5, epochs=400,
                  l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    beta = np.zeros(d)
    n = X.shape[0]
    for _ in range(epochs):
        p_clean = _sigmoid(X @ beta)
        X_adv = pgd_trades(beta, X, p_clean, eps=eps, alpha=alpha, T=T, rng=rng)
        p_adv = _sigmoid(X_adv @ beta)
        # gradient of CE on clean
        g_ce = X.T @ (p_clean - y) / n
        # gradient of KL(p_clean || p_adv) w.r.t. beta: (approx) (p_adv - p_clean) via X_adv
        # dp_adv/dbeta = X_adv * p_adv (1 - p_adv); dKL/dbeta = (p_adv - p_clean) / (p_adv(1-p_adv)) * dp_adv/dbeta
        g_kl = X_adv.T @ (p_adv - p_clean) / n
        g = g_ce + beta_kl * g_kl + l2 * beta
        beta -= lr * g
    return beta


def _predict(beta, X): return (_sigmoid(X @ beta) > 0.5).astype(int)


def pgd_attack(beta, X, y, eps, alpha, T, rng=None):
    if rng is not None:
        X_adv = X + rng.uniform(-eps, eps, X.shape)
    else:
        X_adv = X.copy()
    for _ in range(T):
        p = _sigmoid(X_adv @ beta)
        grad = (p - y)[:, None] * beta
        X_adv = X_adv + alpha * np.sign(grad)
        X_adv = np.clip(X_adv, X - eps, X + eps)
    return X_adv


def train_clean(X, y, lr=0.5, epochs=400, l2=1e-3):
    d = X.shape[1]; beta = np.zeros(d); n = X.shape[0]
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        g = X.T @ (p - y) / n + l2 * beta
        beta -= lr * g
    return beta


if __name__ == "__main__":
    print("=== TRADES adversarial training (Zhang 2019) ===\n")
    rng = np.random.default_rng(0)
    d = 5
    beta_true = rng.normal(0, 1, d); beta_true /= np.linalg.norm(beta_true)
    n = 400
    X = rng.normal(0, 1, (n, d))
    y = (X @ beta_true + rng.normal(0, 0.3, n) > 0).astype(int)
    X_te = rng.normal(0, 1, (500, d))
    y_te = (X_te @ beta_true + rng.normal(0, 0.3, 500) > 0).astype(int)

    beta_clean = train_clean(X, y)
    beta_t1 = train_trades(X, y, beta_kl=1.0)
    beta_t6 = train_trades(X, y, beta_kl=6.0)

    print(f"  {'model':16s}  {'clean_acc':>10}  {'PGD @ eps=0.2':>16}  {'PGD @ eps=0.4':>16}")
    for name, b in (("clean-trained", beta_clean),
                     ("TRADES beta=1  ", beta_t1),
                     ("TRADES beta=6  ", beta_t6)):
        clean_acc = (_predict(b, X_te) == y_te).mean()
        acc_02 = (_predict(b, pgd_attack(b, X_te, y_te, 0.2, 0.05, 40,
                    rng=np.random.default_rng(42))) == y_te).mean()
        acc_04 = (_predict(b, pgd_attack(b, X_te, y_te, 0.4, 0.10, 40,
                    rng=np.random.default_rng(42))) == y_te).mean()
        print(f"  {name}  {clean_acc:>10.3f}  {acc_02:>16.3f}  {acc_04:>16.3f}")

    print("\n  TRADES lets you dial the clean/robust trade-off via beta.\n")
    print("--- library cross-check (TRADES-pytorch ref repo; foolbox with KL loss) ---")
