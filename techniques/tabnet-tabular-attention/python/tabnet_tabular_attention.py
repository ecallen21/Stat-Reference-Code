"""TabNet tabular attention (Reference Sec 47.112).

Arik & Pfister 2021 'TabNet: Attentive interpretable tabular
learning', AAAI. Sequential decision architecture for tabular data:

    1. At each STEP a learnable ATTENTIVE TRANSFORMER computes a
       SPARSE feature mask (sparsemax activation) selecting which
       features to reason about.
    2. A FEATURE TRANSFORMER processes the masked features.
    3. Outputs across steps aggregate into the final prediction;
       masks aggregate into a per-instance FEATURE-IMPORTANCE map.

Competitive with XGBoost / LightGBM on many tabular benchmarks
plus native interpretability. Illustrated here as a per-instance
sparse-attention linear head learning WHICH FEATURES to route.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression    # per-step linear head


def sparsemax(z):
    """Martins & Astudillo 2016 sparsemax projection onto the simplex."""
    z = np.atleast_2d(z); n, d = z.shape
    z_sorted = -np.sort(-z, axis=1)
    cum = np.cumsum(z_sorted, axis=1)
    rho = np.arange(1, d + 1)
    support = (1 + rho * z_sorted) > cum
    k = support.sum(axis=1)
    tau = (cum[np.arange(n), k - 1] - 1) / k
    p = np.clip(z - tau[:, None], 0, None)
    return p if z.shape[0] > 1 else p[0]


def tabnet_toy(X, y, n_steps=3, rng=None):
    """Very simplified TabNet with linear feature-selector per step."""
    rng = rng or np.random.default_rng(0)
    n, p = X.shape
    masks = []
    accumulated_pred = np.zeros(n)
    residual = y - y.mean()
    for _ in range(n_steps):
        # Attentive transformer: score each feature by |corr(x_j, residual)|
        scores = np.abs(X.T @ residual) / (np.linalg.norm(X, axis=0) + 1e-9)
        m = sparsemax(scores * 5.0)
        masks.append(m)
        # Feature transformer: linear regression on masked features
        Xw = X * m
        beta, *_ = np.linalg.lstsq(Xw, residual, rcond=None)
        step_pred = Xw @ beta
        accumulated_pred += step_pred
        residual = y - y.mean() - accumulated_pred
    return {"masks": np.array(masks), "yhat": accumulated_pred + y.mean()}


if __name__ == "__main__":
    print("=== TabNet-style tabular attention (Arik-Pfister 2021) ===\n")
    rng = np.random.default_rng(0)
    n, p = 800, 10
    X = rng.normal(size=(n, p))
    # Only features {0, 3, 7} matter
    y = 2 * X[:, 0] - X[:, 3] + 0.5 * X[:, 7] + 0.3 * rng.normal(size=n)

    res = tabnet_toy(X, y, n_steps=3, rng=rng)
    for i, m in enumerate(res["masks"]):
        top = np.argsort(-m)[:5]
        print(f"  Step {i+1}: mask top-5 = {[(int(j), round(m[j], 3)) for j in top]}   "
              f"nnz = {int((m > 1e-3).sum())}")
    mse = float(((y - res["yhat"]) ** 2).mean())
    print(f"\n  Final MSE = {mse:.3f}   (baseline var y = {y.var():.3f})")

    # Aggregate feature importance (sum of masks)
    imp = res["masks"].sum(axis=0)
    order = np.argsort(-imp)
    print(f"  Aggregate mask importance top-5 = {[(int(j), round(imp[j], 3)) for j in order[:5]]}")
    print(f"  True active features = {[0, 3, 7]}")

    print("\n--- library cross-check (limited R; pytorch-tabnet Python) ---")
