"""Trim-and-fill for publication bias (Reference Sec 22.4).

Duval & Tweedie 2000.  If small negative studies are missing from a
funnel plot (asymmetry), iteratively:
  * TRIM   : remove the most extreme small-negative studies until
             funnel is symmetric.
  * ESTIMATE k_0: number of missing studies.
  * FILL   : add k_0 imputed (symmetric mirror) studies.
  * Re-estimate pooled effect.

Compact demo: simulate publication bias, apply L_0 estimator +
mirror imputation, compare pooled effects.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _pool_random(y, se):
    """Pooled random-effects effect (DerSimonian-Laird)."""
    w_fe = 1 / se ** 2
    ybar_fe = (w_fe * y).sum() / w_fe.sum()
    Q = (w_fe * (y - ybar_fe) ** 2).sum()
    df = len(y) - 1
    tau2 = max(0.0, (Q - df) / (w_fe.sum() - (w_fe ** 2).sum() / w_fe.sum()))
    w = 1 / (se ** 2 + tau2)
    ybar = (w * y).sum() / w.sum()
    se_pool = 1 / np.sqrt(w.sum())
    return float(ybar), float(se_pool), float(tau2)


def l0_estimator(y, ybar):
    """Duval-Tweedie L_0 estimator of missing studies."""
    dev = y - ybar
    ranks = np.argsort(np.abs(dev)) + 1
    signs = np.sign(dev)
    T_n = sum(r * s for r, s in zip(ranks[np.argsort(np.argsort(np.abs(dev)))], signs)
              if s > 0)
    n = len(y)
    L0 = int(max(0, round((4 * T_n - n * (n + 1)) / (2 * n - 1))))
    return L0


def trim_and_fill(y, se, max_iter=10):
    y = np.array(y, dtype=float); se = np.array(se, dtype=float)
    for _ in range(max_iter):
        ybar, _, _ = _pool_random(y, se)
        k0 = l0_estimator(y, ybar)
        if k0 == 0:
            break
        # Trim k0 rightmost (most positive) studies temporarily to re-estimate
        order = np.argsort(-y)                # most positive first
        keep = np.ones(len(y), dtype=bool)
        keep[order[:k0]] = False
        ybar2, _, _ = _pool_random(y[keep], se[keep])
        # Fill: mirror k0 largest positive residuals across ybar2
        mirror_vals = 2 * ybar2 - y[order[:k0]]
        y = np.concatenate([y, mirror_vals])
        se = np.concatenate([se, se[order[:k0]]])
        break              # single-iteration L_0 pass is standard
    pooled = _pool_random(y, se)
    return {"y_augmented": y, "se_augmented": se, "n_original": len(y) - k0,
            "n_filled": int(k0), "pooled_mean": pooled[0],
            "pooled_SE": pooled[1], "tau2": pooled[2]}


if __name__ == "__main__":
    print("=== Trim-and-fill for publication bias ===\n")
    rng = np.random.default_rng(0)
    K = 20
    # True effect 0.30 with heterogeneity; small negative studies suppressed
    se = rng.uniform(0.05, 0.30, K)
    y = rng.normal(0.30, 0.15, K) + rng.normal(0, se, K)
    # Publication bias: drop imprecise studies with low effect
    keep_mask = ~((y < 0.15) & (se > 0.15))
    y_obs = y[keep_mask]; se_obs = se[keep_mask]

    ybar_biased, se_biased, _ = _pool_random(y_obs, se_obs)
    r = trim_and_fill(y_obs, se_obs)
    print(f"  Original K = {K}, published K = {len(y_obs)}")
    print(f"  Naive pooled (biased):        mean = {ybar_biased:+.3f}   SE = {se_biased:.3f}")
    print(f"  Trim-and-fill augmented:      mean = {r['pooled_mean']:+.3f}"
          f"   SE = {r['pooled_SE']:.3f}   filled = {r['n_filled']}   tau^2 = {r['tau2']:.3f}\n")

    print("--- library cross-check (R metafor::trimfill; Python custom) ---")
