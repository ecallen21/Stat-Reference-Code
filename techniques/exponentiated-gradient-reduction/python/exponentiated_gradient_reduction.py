"""Exponentiated-gradient reduction (Reference Ch 31 Fairness).

Agarwal, Beygelzimer, Dudik, Langford & Wallach (2018) 'A Reductions
Approach to Fair Classification.'  The core algorithm behind fairlearn.

REDUCE fair classification to a SEQUENCE of COST-SENSITIVE classification
problems solved by any base learner.  For demographic parity (DP), the
fairness constraints are

  E[Y_hat | A = a] - E[Y_hat]  =  0    for each group a.

Lagrangian:  L(h, lambda) = err(h) + sum_a lambda_a * (E[Y_hat|A=a] - E[Y_hat]).

EXPONENTIATED GRADIENT (Freund-Schapire 1997) on lambda + best-response by the
base learner on h gives a saddle-point solver:

  Repeat T rounds:
    (1) Refit base classifier with sample weights derived from current lambda.
    (2) Compute violation g_a = E[Y_hat|A=a] - E[Y_hat] per group.
    (3) Multiplicative update  lambda_a <- lambda_a * exp(eta * g_a); renormalise.

Output: RANDOMISED classifier = uniform mixture of the T classifiers.

Here we implement the DP variant with a logistic-regression base learner.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def train_logistic_weighted(X, y, w, lr=0.5, epochs=300, l2=1e-3):
    d = X.shape[1]; beta = np.zeros(d); n = X.shape[0]
    w = w * n / w.sum()
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        g = X.T @ (w * (p - y)) / n + l2 * beta
        beta -= lr * g
    return beta


def eg_reduction(X, y, A, eps=0.05, eta=0.5, T=15, seed=0):
    """DP-constrained EG-reduction. eps = allowed violation.

    Weight formula (Agarwal 2018 Alg 1) for DP:
      w_i = 1 + sum_a lambda_a * ( 1{A_i = a}/p_a  -  1 )   * (1 if y_i=1 else -1)
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    groups = np.unique(A)
    p_group = {a: float((A == a).mean()) for a in groups}
    # Lagrange multipliers: two per group (upper & lower constraint).
    lam_pos = {a: 0.0 for a in groups}
    lam_neg = {a: 0.0 for a in groups}
    classifiers = []
    for t in range(T):
        # Cost-sensitive weights derived from the DP Lagrangian (Agarwal 2018 Tab 1).
        # If group a is OVER-SELECTED (lam_pos[a] > 0):
        #   penalise predicting 1 there  => weight up NEGATIVES (y=0) in group a.
        # If group a is UNDER-SELECTED (lam_neg[a] > 0):
        #   encourage predicting 1 there => weight up POSITIVES (y=1) in group a.
        w = np.ones(n)
        for a in groups:
            in_a = (A == a).astype(float)
            w += lam_pos[a] * in_a * (1 - y)     # upweight (y=0, A=a) -> classifier says 0 more
            w += lam_neg[a] * in_a * y           # upweight (y=1, A=a) -> classifier says 1 more
        w = np.clip(w, 1e-6, None)
        beta = train_logistic_weighted(X, y, w)
        classifiers.append(beta)
        y_hat = (_sigmoid(X @ beta) > 0.5).astype(float)
        avg = float(y_hat.mean())
        for a in groups:
            g = float(y_hat[A == a].mean()) - avg
            # constraint g - eps <= 0 (upper) and -g - eps <= 0 (lower)
            lam_pos[a] = lam_pos[a] * np.exp(eta * (g - eps))
            lam_neg[a] = lam_neg[a] * np.exp(eta * (-g - eps))
            # Clip to avoid runaway
            lam_pos[a] = min(lam_pos[a], 5.0)
            lam_neg[a] = min(lam_neg[a], 5.0)
        # Warm up: initialise multipliers on round 0
        if t == 0:
            for a in groups:
                lam_pos[a] = 0.1
                lam_neg[a] = 0.1
    return classifiers


def randomised_predict(classifiers, X, seed=0):
    """Uniform-mixture prediction (Agarwal 2018)."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(classifiers), X.shape[0])
    y_hat = np.zeros(X.shape[0], dtype=int)
    for i, k in enumerate(picks):
        y_hat[i] = int(_sigmoid(X[i] @ classifiers[k]) > 0.5)
    return y_hat


def dp_ratio(y_hat, A):
    r = [float(y_hat[A == a].mean()) for a in np.unique(A)]
    return min(r) / max(r) if max(r) > 0 else float("nan")


if __name__ == "__main__":
    print("=== Exponentiated-gradient reduction for DP (Agarwal 2018) ===\n")
    rng = np.random.default_rng(0)
    n_per = 500
    y0 = (rng.random(n_per) < 0.60).astype(float)
    y1 = (rng.random(n_per) < 0.25).astype(float)
    y = np.concatenate([y0, y1])
    A = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)
    x0 = y + rng.normal(0, 0.5, len(y))
    x1 = A + rng.normal(0, 0.4, len(y))
    X = np.stack([x0, x1, np.ones_like(y)], axis=1)

    # Baseline
    beta_baseline = train_logistic_weighted(X, y, np.ones(len(y)))
    y_hat_base = (_sigmoid(X @ beta_baseline) > 0.5).astype(int)
    print(f"  ERM baseline:                accuracy={(y_hat_base == y).mean():.3f}"
          f"   DP ratio={dp_ratio(y_hat_base, A):.3f}")

    for T in (5, 15, 30):
        classifiers = eg_reduction(X, y, A, eps=0.02, eta=5.0, T=T)
        y_hat = randomised_predict(classifiers, X, seed=1)
        print(f"  EG reduction  T={T:>2}   accuracy={(y_hat == y).mean():.3f}"
              f"   DP ratio={dp_ratio(y_hat, A):.3f}")

    print("\n  EG-reduction lets you set a DP tolerance eps and returns a randomised meta-classifier.\n")
    print("--- library cross-check (fairlearn.reductions.ExponentiatedGradient with"
          " DemographicParity/EqualizedOdds constraints) ---")
