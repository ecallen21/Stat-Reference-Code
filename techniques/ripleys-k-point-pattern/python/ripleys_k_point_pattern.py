"""Ripley's K and L functions for spatial point patterns (Reference §23.12).

    K(r) = (1 / lambda) * E[# other events within r of a typical event]

Under complete spatial randomness (CSR / homogeneous Poisson):
    K(r) = pi * r^2       and    L(r) = sqrt(K(r) / pi) - r = 0.

Estimator with border-edge correction (points within r of the boundary
excluded from the sum, denominator adjusted):

    Khat(r) = |A| * sum_i sum_{j != i} 1{d_ij <= r} * 1{b_i > r} / (n * n_R(r))

where b_i is the distance from point i to the window boundary and
n_R(r) = # points with b_i > r.

L(r) - r above the CSR envelope suggests clustering; below suggests
regularity (inhibition).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math (sqrt, log, pi)

import numpy as np    # numerical arrays + linear algebra


def _pairwise_dist(pts):
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def ripley_k(pts, r_grid, window) -> dict:
    """K(r) with border-method edge correction.

    window = (xmin, xmax, ymin, ymax) — rectangular study window.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    xmin, xmax, ymin, ymax = window
    area = (xmax - xmin) * (ymax - ymin)
    lam = n / area
    # boundary distance for each point (min of the four side distances)
    b = np.minimum.reduce([pts[:, 0] - xmin, xmax - pts[:, 0],
                           pts[:, 1] - ymin, ymax - pts[:, 1]])
    D = _pairwise_dist(pts)
    np.fill_diagonal(D, np.inf)
    K = np.zeros_like(r_grid, dtype=float)
    for k, r in enumerate(r_grid):
        eligible = b > r
        n_R = int(eligible.sum())
        if n_R == 0:
            K[k] = np.nan
            continue
        # sum over eligible i of neighbours within r
        counts = (D[eligible] <= r).sum()
        K[k] = area * counts / (n * n_R)
    return {"r": np.asarray(r_grid, dtype=float), "K": K,
            "L_minus_r": np.sqrt(K / math.pi) - np.asarray(r_grid, dtype=float),
            "lambda_hat": lam,
            "method": "Ripley's K with border edge correction"}


def csr_envelope(n, r_grid, window, n_sim: int = 199,
                 alpha: float = 0.05, seed: int = 0) -> dict:
    """Simulation envelope for L(r) - r under CSR."""
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = window
    L_sims = np.zeros((n_sim, len(r_grid)))
    for s in range(n_sim):
        sim_pts = np.column_stack([rng.uniform(xmin, xmax, n),
                                    rng.uniform(ymin, ymax, n)])
        L_sims[s] = ripley_k(sim_pts, r_grid, window)["L_minus_r"]
    lo = np.nanquantile(L_sims, alpha / 2, axis=0)
    hi = np.nanquantile(L_sims, 1 - alpha / 2, axis=0)
    return {"r": np.asarray(r_grid, dtype=float),
            "lo": lo, "hi": hi, "n_sim": n_sim, "alpha": alpha}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    window = (0.0, 1.0, 0.0, 1.0)
    r_grid = np.linspace(0.02, 0.25, 12)

    # ---- CSR pattern ----
    n = 150
    pts_csr = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
    Kc = ripley_k(pts_csr, r_grid, window)
    env = csr_envelope(n, r_grid, window, n_sim=99, seed=1)

    print("=== Ripley's K / L for CSR pattern (n=150) ===")
    print(f"  {'r':>6} {'L-r':>10} {'CSR lo':>10} {'CSR hi':>10}  verdict")
    inside_all = True
    for k in range(len(r_grid)):
        v = "CSR" if env["lo"][k] <= Kc["L_minus_r"][k] <= env["hi"][k] else "out"
        if v == "out":
            inside_all = False
        print(f"  {r_grid[k]:>6.3f} {Kc['L_minus_r'][k]:>10.4f} "
              f"{env['lo'][k]:>10.4f} {env['hi'][k]:>10.4f}  {v}")
    print(f"  CSR pattern inside envelope at all r? {inside_all}")

    # ---- clustered pattern (Poisson cluster / Matern-like) ----
    n_parents = 12
    parents = rng.uniform(0.1, 0.9, (n_parents, 2))
    offspring_per = 12
    off = []
    for pp in parents:
        off.append(pp + rng.normal(0, 0.03, (offspring_per, 2)))
    pts_clust = np.clip(np.vstack(off), 0.0, 1.0)
    Kk = ripley_k(pts_clust, r_grid, window)
    env_k = csr_envelope(len(pts_clust), r_grid, window, n_sim=99, seed=2)
    above = int(np.sum(Kk["L_minus_r"] > env_k["hi"]))
    print(f"\n=== Clustered pattern (n={len(pts_clust)}) ===")
    print(f"  # r values where L-r > CSR upper: {above} / {len(r_grid)}")
    print(f"  L(r=0.05)-r = {Kk['L_minus_r'][np.argmin(np.abs(r_grid-0.05))]:.4f}  "
          f"(CSR upper = {env_k['hi'][np.argmin(np.abs(r_grid-0.05))]:.4f})")

    print("\n--- library cross-check (R spatstat::Kest / Lest / envelope) ---")
