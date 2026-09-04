"""Shannon entropy (Reference Sec 34.1).

Shannon (1948) 'A mathematical theory of communication.'

For a discrete distribution p:  H(p) = -sum_x p(x) log p(x).
For a joint (X, Y):              H(X, Y) = -sum_{x,y} p(x, y) log p(x, y).
Conditional:                     H(Y | X) = H(X, Y) - H(X).
Chain rule:                      H(X, Y) = H(X) + H(Y | X).

Differential entropy for continuous p:  h(p) = -int p(x) log p(x) dx.
Units: bits (log_2), nats (ln), or Hartleys (log_10).

Estimators used here:
  * Discrete: plug-in from empirical PMF.
  * Continuous: histogram-based + Kozachenko-Leonenko k-NN estimator.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def shannon_entropy(p, base=2):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p) / np.log(base)).sum())


def entropy_from_samples(x, base=2):
    _, counts = np.unique(x, return_counts=True)
    return shannon_entropy(counts / counts.sum(), base=base)


def joint_entropy(x, y, base=2):
    pairs = np.stack([x, y], axis=1)
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    return shannon_entropy(counts / counts.sum(), base=base)


def conditional_entropy(y_given_x, x, base=2):
    """H(Y|X) = H(X, Y) - H(X)."""
    return joint_entropy(x, y_given_x, base) - entropy_from_samples(x, base)


def histogram_entropy(x, n_bins=20, base=2):
    hist, edges = np.histogram(x, bins=n_bins, density=False)
    p = hist / hist.sum()
    bin_width = float(edges[1] - edges[0])
    return shannon_entropy(p, base) + np.log(bin_width) / np.log(base)   # continuous correction


def kozachenko_leonenko(x, k=3):
    """k-NN differential entropy estimator (nats)."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma
    x = np.atleast_2d(x).reshape(-1, 1) if x.ndim == 1 else x
    n, d = x.shape
    tree = cKDTree(x)
    dists, _ = tree.query(x, k=k + 1)
    r_k = dists[:, k]
    log_r = np.log(r_k + 1e-12)
    volume = np.pi ** (d / 2) / _gamma_half(d / 2 + 1)
    return float(digamma(n) - digamma(k) + np.log(volume) + d * log_r.mean())


def _gamma_half(x):
    from scipy.special import gamma
    return float(gamma(x))


if __name__ == "__main__":
    print("=== Shannon entropy ===\n")
    # 1. Discrete: coin flips
    p_fair = [0.5, 0.5]
    p_biased = [0.9, 0.1]
    p_deterministic = [1.0, 0.0]
    print(f"  H(fair coin)          = {shannon_entropy(p_fair):.4f} bits  (should be 1.0)")
    print(f"  H(biased 0.9/0.1)     = {shannon_entropy(p_biased):.4f} bits")
    print(f"  H(deterministic)      = {shannon_entropy(p_deterministic):.4f} bits\n")

    # 2. Joint + conditional
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.integers(0, 4, n)                        # 2 bits marginal
    y_indep = rng.integers(0, 2, n)                  # 1 bit marginal
    y_copy = x % 2                                    # perfectly determined by x
    Hx = entropy_from_samples(x)
    Hy_indep = entropy_from_samples(y_indep)
    Hxy_indep = joint_entropy(x, y_indep)
    Hxy_copy = joint_entropy(x, y_copy)
    print(f"  H(X) = {Hx:.3f}   H(Y_indep) = {Hy_indep:.3f}")
    print(f"  H(X, Y_indep) = {Hxy_indep:.3f}   should ~= H(X) + H(Y_indep) = {Hx + Hy_indep:.3f}")
    print(f"  H(X, Y_copy)  = {Hxy_copy:.3f}   should ~= H(X) = {Hx:.3f} (Y is a function of X)")
    print(f"  H(Y_copy | X) = {conditional_entropy(y_copy, x):.3f}   should ~= 0")

    # 3. Continuous: differential entropy of a Gaussian
    xr = rng.normal(0, 1, 5000)
    print(f"\n  Differential entropy of N(0, 1) (nats)")
    print(f"    true value             = {0.5 * np.log(2 * np.pi * np.e):.4f}")
    print(f"    histogram estimate     = {histogram_entropy(xr, n_bins=30, base=np.e):.4f}")
    print(f"    k-NN estimate (k=3)    = {kozachenko_leonenko(xr, k=3):.4f}\n")

    print("--- library cross-check (scipy.stats.entropy; R entropy; NPEET) ---")
