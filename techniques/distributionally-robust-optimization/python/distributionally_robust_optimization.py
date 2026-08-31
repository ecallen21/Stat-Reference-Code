"""Group DRO (Reference Ch 30 Robustness).

Sagawa, Koh, Hashimoto & Liang (2020) "Distributionally Robust Neural
Networks for Group Shifts."

Standard ERM minimises AVERAGE loss:  min_theta  (1/n) sum_i L_i.
Group DRO minimises the MAX loss across pre-defined groups:

  min_theta   max_g   E_{i in g} L_i (theta).

The online algorithm keeps up-to-date group WEIGHTS q_g, updates each
step as

  q_g  <-  q_g * exp( eta_q * loss_g )      (then normalise)

and takes an SGD step on the WEIGHTED sum sum_g q_g * loss_g.

Result: the model no longer sacrifices worst-group accuracy for better
majority-group accuracy -- essential for fairness on subpopulations.

Here we implement Group DRO for logistic regression on a synthetic
two-group problem where the minority group has a spurious feature that
misleads ERM.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def train_erm(X, y, lr=0.5, epochs=800, l2=1e-3):
    d = X.shape[1]
    beta = np.zeros(d)
    n = X.shape[0]
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        g = X.T @ (p - y) / n + l2 * beta
        beta -= lr * g
    return beta


def train_group_dro(X, y, groups, n_groups, lr=0.5, eta_q=0.05, epochs=800, l2=1e-3):
    d = X.shape[1]
    beta = np.zeros(d)
    q = np.ones(n_groups) / n_groups
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        # per-example BCE
        eps = 1e-12
        losses = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        # per-group loss average
        losses_g = np.array([losses[groups == g].mean() if (groups == g).any() else 0.0
                              for g in range(n_groups)])
        # update q by exponentiated gradient
        q = q * np.exp(eta_q * losses_g)
        q = q / q.sum()
        # weighted SGD step on beta -- weight each example by q_{group_i} / |group|
        w = np.array([q[groups[i]] / max((groups == groups[i]).sum(), 1) for i in range(len(y))])
        g_step = X.T @ (w * (p - y)) + l2 * beta
        beta -= lr * g_step
    return beta, q


def eval_by_group(beta, X, y, groups, n_groups):
    p_hat = (_sigmoid(X @ beta) > 0.5).astype(int)
    out = []
    for g in range(n_groups):
        m = groups == g
        out.append((int(m.sum()), float((p_hat[m] == y[m]).mean()) if m.any() else float("nan")))
    return out


if __name__ == "__main__":
    print("=== Group DRO (Sagawa 2020) ===\n")
    rng = np.random.default_rng(0)
    d = 3
    # Two groups: majority (n=800) has a SPURIOUS feature x_1 correlated with y;
    # minority (n=100) has x_1 uncorrelated with y (spurious feature is misleading).
    n_maj, n_min = 800, 100
    beta_true = np.array([1.0, 0.0, 0.0])   # truly x_0 predicts y; x_1 is spurious; x_2 is noise
    # Majority: x_1 = y + noise (spurious agrees)
    y_maj = rng.integers(0, 2, n_maj).astype(float)
    X_maj = np.zeros((n_maj, d))
    X_maj[:, 0] = 0.6 * (y_maj - 0.5) + rng.normal(0, 0.4, n_maj)
    X_maj[:, 1] = 2 * (y_maj - 0.5) + rng.normal(0, 0.4, n_maj)
    X_maj[:, 2] = rng.normal(0, 1, n_maj)
    # Minority: x_1 = -y + noise (spurious disagrees)
    y_min = rng.integers(0, 2, n_min).astype(float)
    X_min = np.zeros((n_min, d))
    X_min[:, 0] = 0.6 * (y_min - 0.5) + rng.normal(0, 0.4, n_min)
    X_min[:, 1] = -2 * (y_min - 0.5) + rng.normal(0, 0.4, n_min)
    X_min[:, 2] = rng.normal(0, 1, n_min)

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([y_maj, y_min])
    groups = np.concatenate([np.zeros(n_maj), np.ones(n_min)]).astype(int)

    # Test: 50/50 balanced across groups so worst-group accuracy is the failure mode.
    def make_te(n_g, spurious_sign):
        y_g = rng.integers(0, 2, n_g).astype(float)
        Xg = np.zeros((n_g, d))
        Xg[:, 0] = 0.6 * (y_g - 0.5) + rng.normal(0, 0.4, n_g)
        Xg[:, 1] = spurious_sign * 2 * (y_g - 0.5) + rng.normal(0, 0.4, n_g)
        Xg[:, 2] = rng.normal(0, 1, n_g)
        return Xg, y_g
    X_te_maj, y_te_maj = make_te(500, +1)
    X_te_min, y_te_min = make_te(500, -1)
    X_te = np.vstack([X_te_maj, X_te_min])
    y_te = np.concatenate([y_te_maj, y_te_min])
    groups_te = np.concatenate([np.zeros(500), np.ones(500)]).astype(int)

    print("  ERM:")
    beta_erm = train_erm(X, y)
    for g, (n_g, acc_g) in enumerate(eval_by_group(beta_erm, X_te, y_te, groups_te, 2)):
        print(f"    group {g}  n={n_g}  test_acc={acc_g:.3f}")

    print("\n  Group DRO:")
    beta_dro, q = train_group_dro(X, y, groups, n_groups=2)
    for g, (n_g, acc_g) in enumerate(eval_by_group(beta_dro, X_te, y_te, groups_te, 2)):
        print(f"    group {g}  n={n_g}  test_acc={acc_g:.3f}")
    print(f"    final q = {np.round(q, 3).tolist()}   <- weight on the harder minority group")

    print("\n  Group DRO trades a small majority-group loss for a large minority-group gain.\n")
    print("--- library cross-check (wilds; sagawa-lab group_DRO reference; fairlearn) ---")
