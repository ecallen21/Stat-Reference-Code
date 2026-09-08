"""SINDy - Sparse Identification of Nonlinear Dynamics (Ref Sec 47.125).

Brunton, Proctor & Kutz 2016 'Discovering governing equations
from data by sparse identification of nonlinear dynamical
systems', PNAS 113(15). Given time-series x(t) and derivatives
dx/dt, fit each equation as a sparse combination of candidate
basis functions Theta(x) (polynomials, trig, etc.):

    dX/dt = Theta(X) * Xi     (sparse Xi via STLSQ or lasso)

Sequentially thresholded least squares (STLSQ):
    Xi = argmin ||dX/dt - Theta(X) Xi||^2  s.t.  |Xi_j| >= lambda.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def poly_features(X, order=2):
    """Polynomial basis up to `order` in the dimensions of X."""
    n, d = X.shape
    feats = [np.ones(n)]
    names = ["1"]
    for i in range(d):
        feats.append(X[:, i]); names.append(f"x{i}")
    if order >= 2:
        for i in range(d):
            for j in range(i, d):
                feats.append(X[:, i] * X[:, j])
                names.append(f"x{i}*x{j}")
    if order >= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    feats.append(X[:, i] * X[:, j] * X[:, k])
                    names.append(f"x{i}*x{j}*x{k}")
    return np.column_stack(feats), names


def stlsq(Theta, dXdt, lam=0.1, max_iter=20):
    """Sequentially-thresholded least squares (Brunton 2016)."""
    d = dXdt.shape[1]
    Xi, *_ = np.linalg.lstsq(Theta, dXdt, rcond=None)
    for _ in range(max_iter):
        small = np.abs(Xi) < lam
        Xi[small] = 0
        for k in range(d):
            keep = ~small[:, k]
            if keep.sum() == 0: continue
            Xi[keep, k], *_ = np.linalg.lstsq(Theta[:, keep], dXdt[:, k], rcond=None)
    return Xi


if __name__ == "__main__":
    print("=== SINDy (Brunton-Proctor-Kutz 2016) ===\n")
    rng = np.random.default_rng(0)
    # Simulate Van der Pol via RK4:  dx/dt = y, dy/dt = mu(1 - x^2)y - x
    mu = 1.0
    def rhs(state):
        x, y = state
        return np.array([y, mu * (1 - x ** 2) * y - x])

    dt = 0.01; T = 20.0
    N = int(T / dt)
    X = np.zeros((N, 2)); X[0] = [1.5, 0]
    for k in range(N - 1):
        k1 = rhs(X[k])
        k2 = rhs(X[k] + 0.5 * dt * k1)
        k3 = rhs(X[k] + 0.5 * dt * k2)
        k4 = rhs(X[k] + dt * k3)
        X[k + 1] = X[k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Finite-difference derivatives (central)
    dXdt = np.zeros_like(X)
    dXdt[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
    X = X[1:-1]; dXdt = dXdt[1:-1]

    Theta, names = poly_features(X, order=3)
    Xi = stlsq(Theta, dXdt, lam=0.05)
    print(f"  Van der Pol true equations: dx/dt = y,   dy/dt = {mu}(1 - x^2)y - x")
    for k in range(2):
        eq = " + ".join(f"{Xi[i, k]:+.3f}*{names[i]}" for i in range(len(names)) if abs(Xi[i, k]) > 1e-6)
        print(f"  Recovered dx{k}/dt = {eq}")

    print("\n--- library cross-check (pysindy Python; no established R port) ---")
