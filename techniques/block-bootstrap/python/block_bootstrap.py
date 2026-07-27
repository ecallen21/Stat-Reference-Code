"""Block bootstrap for dependent (e.g. time-series) data (Reference §10.4).

The IID bootstrap breaks the correlation structure of the data. For time series
or spatially-correlated data we need to resample BLOCKS of consecutive
observations to preserve local dependence.

Variants:
    - Non-overlapping blocks (Carlstein 1986) : partition x into ceil(n/L) blocks
      of length L, sample n_blocks blocks with replacement.
    - Moving blocks (Kunsch 1989)             : blocks {x_i..x_{i+L-1}} for i = 1..n-L+1;
      sample ceil(n/L) blocks with replacement and concatenate to length n.
    - Circular (Politis-Romano 1992)          : same as moving blocks but the series
      is wrapped around (x_n is adjacent to x_1), so all indices are treated
      symmetrically.

Choosing L: rule-of-thumb is L ~ n^(1/3) for weakly-dependent series (Hall 1985;
Politis-White 2004 give a data-driven optimum). Larger L preserves more of the
correlation; smaller L acts more like IID.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable, Sequence    # stdlib: type hints (Callable = function; Sequence = indexable iterable)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def moving_block_bootstrap(x, statistic: Callable, block_length: int,
                            n_boot: int = 2000, conf: float = 0.95, seed: int = 0) -> dict:
    """Moving-block bootstrap (Kunsch 1989) for a 1-D time series."""
    x = np.asarray(x, dtype=float)
    n = x.size
    L = int(block_length)
    if L < 1 or L > n:
        raise ValueError("block_length must satisfy 1 <= L <= n")
    n_blocks = int(math.ceil(n / L))
    n_starts = n - L + 1
    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(x))
    theta_star = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n_starts, size=n_blocks)
        # concat blocks then trim to length n
        pieces = [x[s:s + L] for s in starts]
        xb = np.concatenate(pieces)[:n]
        theta_star[b] = float(statistic(xb))
    alpha = 1 - conf
    lo, hi = np.quantile(theta_star, [alpha / 2, 1 - alpha / 2])
    return {"theta_hat": theta_hat,
            "bootstrap_SE": float(theta_star.std(ddof=1)),
            "CI_percentile": {"lower": float(lo), "upper": float(hi)},
            "block_length": L,
            "n_boot": n_boot, "n": n,
            "method": "moving-block bootstrap (Kunsch 1989)"}


def circular_block_bootstrap(x, statistic: Callable, block_length: int,
                              n_boot: int = 2000, conf: float = 0.95, seed: int = 0) -> dict:
    """Circular block bootstrap (Politis & Romano 1992): wrap the series so
    every starting index has an equal chance of being drawn."""
    x = np.asarray(x, dtype=float)
    n = x.size; L = int(block_length)
    if L < 1 or L > n:
        raise ValueError("block_length must satisfy 1 <= L <= n")
    n_blocks = int(math.ceil(n / L))
    # extend the series circularly for easy slicing
    x_ext = np.concatenate([x, x[:L - 1]])
    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(x))
    theta_star = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)         # any start in 0..n-1
        xb = np.concatenate([x_ext[s:s + L] for s in starts])[:n]
        theta_star[b] = float(statistic(xb))
    alpha = 1 - conf
    lo, hi = np.quantile(theta_star, [alpha / 2, 1 - alpha / 2])
    return {"theta_hat": theta_hat,
            "bootstrap_SE": float(theta_star.std(ddof=1)),
            "CI_percentile": {"lower": float(lo), "upper": float(hi)},
            "block_length": L, "n_boot": n_boot, "n": n,
            "method": "circular block bootstrap (Politis-Romano 1992)"}


def rule_of_thumb_block_length(n: int) -> int:
    """Simple L ~ n^(1/3) heuristic; round to nearest integer >= 1."""
    return max(1, int(round(n ** (1 / 3))))


def library_versions(x, statistic=np.mean):
    """arch.bootstrap.MovingBlockBootstrap if available."""
    try:
        from arch.bootstrap import MovingBlockBootstrap
        bs = MovingBlockBootstrap(block_size=10, x=x, seed=0)
        results = bs.apply(lambda xs: statistic(xs['x']), 2000)
        lo, hi = np.quantile(results, [0.025, 0.975])
        return {"arch MovingBlockBootstrap (L=10, B=2000) mean 95% CI":
                {"lower": float(lo), "upper": float(hi)}}
    except Exception as ex:
        return {"arch (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    rng = np.random.default_rng(19)
    # AR(1) with phi = 0.7 (positive autocorrelation)
    n = 300; phi = 0.7
    x = np.empty(n); x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()

    L = rule_of_thumb_block_length(n)
    print(f"=== Moving-block bootstrap for mean (n={n}, L={L}, B=2000) ===")
    out = moving_block_bootstrap(x, np.mean, block_length=L, n_boot=2000)
    print(f"  theta_hat = {out['theta_hat']:.4f}")
    print(f"  SE_boot   = {out['bootstrap_SE']:.4f}   (naive IID SE would UNDERESTIMATE for AR data)")
    print(f"  CI (percentile): [{out['CI_percentile']['lower']:.4f}, {out['CI_percentile']['upper']:.4f}]")

    print(f"\n=== Circular block bootstrap ===")
    out2 = circular_block_bootstrap(x, np.mean, block_length=L, n_boot=2000)
    print(f"  SE_boot   = {out2['bootstrap_SE']:.4f}")
    print(f"  CI: [{out2['CI_percentile']['lower']:.4f}, {out2['CI_percentile']['upper']:.4f}]")

    print(f"\n=== Naive IID bootstrap (SE would UNDERESTIMATE for autocorrelated data) ===")
    iid_star = np.array([np.mean(x[rng.integers(0, n, size=n)]) for _ in range(2000)])
    print(f"  SE_iid    = {iid_star.std(ddof=1):.4f}   (vs SE_block = {out['bootstrap_SE']:.4f})")

    print("\n--- library (arch, if installed) ---")
    for k, v in library_versions(x, np.mean).items():
        print(f"  {k}: {v}")
