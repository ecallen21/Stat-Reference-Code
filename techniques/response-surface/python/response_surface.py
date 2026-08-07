"""Response-surface methodology (Reference §16.11).

Iterative process for locating the optimum of a physical or industrial
process:
    1. FIRST-ORDER design (2^k factorial + center) fit a linear model.
    2. Estimate the gradient; move along it (steepest ascent).
    3. Once near the optimum, augment to a SECOND-ORDER design (CCD or
       Box-Behnken) and fit a quadratic surface.
    4. Solve for the stationary point via calculus:
        d y_hat / d x = 0 -> x_s = -0.5 B^-1 b
        where y_hat = b_0 + x^T b + x^T B x (b vector, B symmetric matrix).

Central Composite Design (CCD)
    2^k factorial + 2k axial (star) points at +/- alpha + n_c center reps.
    Rotatable CCD: alpha = (2^k)^(1/4).

Box-Behnken Design
    3-level design where each factor takes only three values;
    no extreme corners.  Fewer runs than CCD for k = 3-4.

Fit a full quadratic:
    y = b_0 + sum_i b_i x_i + sum_i b_ii x_i^2 + sum_{i<j} b_ij x_i x_j
    OLS on the design matrix.

Stationary-point analysis
    Eigenvalues of B: all positive -> minimum; all negative -> maximum;
    mixed signs -> saddle point (ridge).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from itertools import product    # stdlib: cartesian product for factorial designs

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def ccd_design(k: int, alpha: float = None, n_center: int = 3) -> np.ndarray:
    """Central composite design in coded units for k factors."""
    if alpha is None:
        alpha = (2 ** k) ** 0.25
    factorial = np.array(list(product([-1, 1], repeat=k)), dtype=float)
    axial = np.zeros((2 * k, k))
    for i in range(k):
        axial[2 * i, i] = -alpha; axial[2 * i + 1, i] = alpha
    center = np.zeros((n_center, k))
    return np.vstack([factorial, axial, center])


def box_behnken(k: int, n_center: int = 3) -> np.ndarray:
    """Simple Box-Behnken design for k = 3 (canonical case)."""
    if k != 3: raise NotImplementedError("only k=3 implemented in this demo")
    b = []
    for pair in ((0, 1), (0, 2), (1, 2)):
        for sign_a in (-1, 1):
            for sign_b in (-1, 1):
                row = [0, 0, 0]
                row[pair[0]] = sign_a; row[pair[1]] = sign_b
                b.append(row)
    b.extend([[0, 0, 0]] * n_center)
    return np.array(b, dtype=float)


def _quadratic_design(X):
    """Full quadratic design matrix with intercept, linear, quadratic, interaction terms."""
    n, k = X.shape
    cols = [np.ones(n)]
    cols += [X[:, i] for i in range(k)]
    cols += [X[:, i] ** 2 for i in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            cols.append(X[:, i] * X[:, j])
    return np.column_stack(cols)


def rsm_fit(X, y) -> dict:
    """Fit a full quadratic response surface + stationary-point analysis."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, k = X.shape
    D = _quadratic_design(X)
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    # Extract b (linear) and B (quadratic + interactions -> symmetric matrix)
    b = beta[1:1 + k]
    B = np.zeros((k, k))
    for i in range(k):
        B[i, i] = beta[1 + k + i]
    idx = 1 + 2 * k
    for i in range(k):
        for j in range(i + 1, k):
            B[i, j] = B[j, i] = beta[idx] / 2
            idx += 1
    # Stationary point: -0.5 B^-1 b
    x_s = -0.5 * np.linalg.solve(B, b)
    y_s = float(beta[0] + b @ x_s + x_s @ B @ x_s)
    eigvals = np.linalg.eigvalsh(B)
    kind = "maximum" if (eigvals < 0).all() else "minimum" if (eigvals > 0).all() else "saddle"
    return {"coef": beta, "b": b, "B": B,
            "stationary_point": x_s, "y_at_stationary": y_s,
            "surface_type": kind,
            "eigenvalues_B": eigvals,
            "method": "Second-order response-surface fit + stationary-point analysis"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== CCD design for k = 2 ===")
    X = ccd_design(k=2, n_center=3)
    print(f"  {X.shape[0]} runs")
    print(X.round(3))

    print("\n=== True response: y = 5 + 2 x1 + 3 x2 - x1^2 - 2 x2^2 + noise ===")
    def f(X): return 5 + 2 * X[:, 0] + 3 * X[:, 1] - X[:, 0] ** 2 - 2 * X[:, 1] ** 2
    y = f(X) + rng.normal(0, 0.1, len(X))
    r = rsm_fit(X, y)
    print(f"\n  fitted coefs: {r['coef'].round(3)}")
    print(f"  stationary point x* = {r['stationary_point'].round(3)}")
    print(f"  y at x*             = {r['y_at_stationary']:.3f}")
    print(f"  surface type        = {r['surface_type']}")
    print(f"  true maximum at x = (1, 0.75); y = 7.125")

    print("\n=== Box-Behnken design for k = 3 ===")
    Xbb = box_behnken(k=3, n_center=3)
    print(f"  {Xbb.shape[0]} runs (fewer than CCD's { (1 << 3) + 2*3 + 3 })")

    print("\n--- library cross-check (pyDOE / R rsm) ---")
    print("  R: rsm::rsm(y ~ SO(x1, x2), data = df); rsm::steepest(fit)")
