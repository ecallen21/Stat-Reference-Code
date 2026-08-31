"""Learning Fair Representations (LFR) (Reference Ch 31 Fairness).

Zemel, Wu, Swersky, Pitassi & Dwork (2013) 'Learning Fair
Representations.'

CORE IDEA: learn a mapping x -> z such that
   (i) z RETAINS TASK-RELEVANT information about y,
   (ii) z IS INDEPENDENT of the protected attribute A.

Zemel's original formulation uses K prototypes M_k and a softmax
assignment P(z=k | x); it minimises

   L  =  A_x * L_reconstruction  +  A_y * L_prediction  +  A_z * L_fairness

For a self-contained demo without the numerical fragility of the
alternating prototype-vs-assignment optimisation, we implement the
CLEANER PROJECTIVE COUSIN (Bolukbasi 2016 / Ravfogel 2020 INLP):

   1. Compute the direction v in feature space that best predicts A
      (unit-normalised regression coefficient).
   2. Project features onto the SUBSPACE ORTHOGONAL TO v:
          z = x - (x . v) v.
   3. Train a classifier on z.

This is a first-order LFR variant: the representation is linearly
guarded against a linear adversary on A. It's the same intuition as
Zemel's fairness loss expressed via linear projection.

The demo compares:
  - raw-feature logistic regression,
  - projection-debiased logistic regression,
on synthetic data with a group-proxy feature.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def train_logistic(X, y, lr=0.5, epochs=600, l2=1e-3):
    d = X.shape[1]
    beta = np.zeros(d)
    n = X.shape[0]
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        g = X.T @ (p - y) / n + l2 * beta
        beta -= lr * g
    return beta


def learn_debiasing_direction(X, A, epochs=400, lr=0.5, l2=1e-3):
    """Fit logistic regression to predict A from X; return unit direction."""
    d = X.shape[1]
    beta = np.zeros(d)
    n = X.shape[0]
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        g = X.T @ (p - A) / n + l2 * beta
        beta -= lr * g
    return beta / (np.linalg.norm(beta) + 1e-12)


def project_out(X, v):
    """Return X with the component along v removed (rank-1 projection)."""
    return X - np.outer(X @ v, v)


def dp_ratio(y_hat, groups):
    r = [float(y_hat[groups == a].mean()) for a in np.unique(groups)]
    return min(r) / max(r) if max(r) > 0 else float("nan")


if __name__ == "__main__":
    print("=== LFR (linear-projection variant a la Bolukbasi 2016 / Ravfogel 2020) ===\n")
    rng = np.random.default_rng(0)
    n_per = 500
    y0 = (rng.random(n_per) < 0.60).astype(float)
    y1 = (rng.random(n_per) < 0.25).astype(float)
    y = np.concatenate([y0, y1])
    A = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)
    # Higher-dim features: x0 signals y; two group-proxy features; three noise
    # features; a bias. Bigger feature space so projections don't kill the signal.
    x0 = y + rng.normal(0, 0.5, len(y))
    x1 = A + rng.normal(0, 0.4, len(y))
    x2 = A + rng.normal(0, 0.6, len(y))
    x3 = rng.normal(0, 1, len(y))
    x4 = rng.normal(0, 1, len(y))
    x5 = rng.normal(0, 1, len(y))
    x6 = np.ones_like(y)
    X = np.stack([x0, x1, x2, x3, x4, x5, x6], axis=1)

    # ---- Baseline: raw features ----
    beta = train_logistic(X, y)
    y_hat_raw = (_sigmoid(X @ beta) > 0.5).astype(int)
    print("  Raw features:")
    print(f"    task accuracy = {(y_hat_raw == y).mean():.3f}"
          f"   DP ratio = {dp_ratio(y_hat_raw, A):.3f}"
          f"   coeffs = {np.round(beta, 3).tolist()}")

    # ---- Fair representation: ITERATED null-space projection (INLP) ----
    # Ravfogel 2020: repeatedly train an A-predictor and null out its direction.
    Z = X.copy()
    for it in range(3):
        v = learn_debiasing_direction(Z, A)
        Z = project_out(Z, v)
    beta_z = train_logistic(Z, y)
    y_hat_z = (_sigmoid(Z @ beta_z) > 0.5).astype(int)
    print(f"\n  Iterated projection-debiased features (3 rounds):")
    print(f"    task accuracy = {(y_hat_z == y).mean():.3f}"
          f"   DP ratio = {dp_ratio(y_hat_z, A):.3f}"
          f"   coeffs = {np.round(beta_z, 3).tolist()}")

    # Linear-adversary accuracy after INLP
    v_adv = learn_debiasing_direction(Z, A)
    A_pred = (_sigmoid(Z @ v_adv) > 0.5).astype(int)
    A_baseline = (_sigmoid(X @ learn_debiasing_direction(X, A)) > 0.5).astype(int)
    print(f"\n  Linear-adversary accuracy predicting A from raw X: {(A_baseline == A).mean():.3f}")
    print(f"  Linear-adversary accuracy predicting A from Z    : {(A_pred == A).mean():.3f}"
          "   <- linear guardedness after INLP.\n")

    print("--- library cross-check (aif360.algorithms.preprocessing.LFR;"
          " concept-erasure / INLP for the linear projection variant) ---")
