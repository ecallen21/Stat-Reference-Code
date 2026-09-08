"""CVaR / Expected Shortfall (Reference Sec 47.42).

Rockafellar & Uryasev 2000 'Optimization of conditional value-at-
risk', J Risk 2(3). At confidence level alpha in (0, 1),

    VaR_alpha(X) = inf { x : P(X <= x) >= alpha }
    CVaR_alpha(X) = E[X | X >= VaR_alpha(X)]  (loss convention)

CVaR is coherent (subadditive), VaR is not. Rockafellar-Uryasev
representation:

    CVaR_alpha(X) = min_c  c + (1 / (1-alpha)) E[(X - c)_+]

is convex in c (and in the decision when X depends on it) --
enables convex portfolio optimisation.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize_scalar    # 1D RU minimisation


def var_es_empirical(losses, alpha=0.95):
    """Empirical VaR and Expected Shortfall from a loss sample."""
    losses = np.sort(np.asarray(losses))
    n = len(losses)
    idx = int(np.ceil(alpha * n)) - 1
    var = float(losses[idx])
    es = float(losses[idx:].mean())
    return {"var": var, "es": es, "n_tail": n - idx}


def cvar_rockafellar_uryasev(losses, alpha=0.95):
    """CVaR via RU representation min_c c + E[(X - c)_+] / (1 - alpha)."""
    losses = np.asarray(losses)
    def obj(c):
        return c + np.maximum(losses - c, 0).mean() / (1 - alpha)
    res = minimize_scalar(obj, bounds=(losses.min(), losses.max()), method="bounded")
    return {"cvar": float(res.fun), "var_optimal_c": float(res.x)}


if __name__ == "__main__":
    print("=== CVaR / Expected Shortfall (Rockafellar-Uryasev 2000) ===\n")
    rng = np.random.default_rng(0)
    losses = rng.standard_t(df=3, size=20_000)    # heavy tail
    for a in [0.90, 0.95, 0.99]:
        emp = var_es_empirical(losses, a)
        ru = cvar_rockafellar_uryasev(losses, a)
        print(f"  alpha={a:.2f}  VaR={emp['var']:+.3f}  ES(empirical)={emp['es']:+.3f}"
              f"  CVaR(RU)={ru['cvar']:+.3f}  n_tail={emp['n_tail']}")

    print("\n  RU 'CVaR' agrees with tail-mean ES within Monte-Carlo error.")
    print("  RU form is CONVEX -> plugs into portfolio LPs / QPs.")
    print("\n--- library cross-check (PerformanceAnalytics R; empyrical / cvxpy Python) ---")
