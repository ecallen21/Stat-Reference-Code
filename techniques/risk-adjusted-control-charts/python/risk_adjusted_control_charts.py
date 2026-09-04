"""Risk-adjusted control charts (Reference Sec 37.10).

Steiner, Cook, Farewell & Treasure (2000); Grigg-Farewell (2004).

Healthcare quality monitoring: outcomes (mortality, readmission) depend
on PATIENT MIX. Risk-adjusted charts standardise by the RISK SCORE from
a predictive model and monitor the residual deviations.

Two common tools:
  * VLAD (Variable Life-Adjusted Display, Lovegrove 1997):
      cumsum of  (expected deaths - observed)  per patient.
      A downward slope = more deaths than expected; upward = better.

  * Risk-adjusted CUSUM (Steiner 2000):
      log-likelihood-ratio-based CUSUM testing an odds-ratio shift OR.
      W_i =  y_i log(OR)  +  log( (1 - p_i + p_i OR) / (1 - p_i + p_i OR) )   inverse for lower control
      L_i^+ = max(0, L_{i-1}^+ + W_i - k^+),   L_i^- = min(0, ...)  and flag on threshold h.

Here we implement VLAD + a compact risk-adjusted CUSUM and demonstrate
alarm behaviour on a synthetic surgeon-outcome series.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def vlad(y, p_hat):
    """Cumulative (expected - observed)."""
    return np.cumsum(p_hat - y)


def risk_adjusted_cusum(y, p_hat, OR=2.0, h=4.5):
    """Log-likelihood-ratio CUSUM for detecting an odds-ratio shift."""
    W = y * np.log(OR) - np.log(1 - p_hat + p_hat * OR)
    L = np.zeros(len(y))
    for i in range(len(y)):
        prev = L[i - 1] if i > 0 else 0.0
        L[i] = max(0.0, prev + W[i])
    signals = np.where(L > h)[0]
    return {"L": L, "first_signal": int(signals[0]) if len(signals) else None}


if __name__ == "__main__":
    print("=== Risk-adjusted control charts (VLAD + CUSUM) ===\n")
    rng = np.random.default_rng(0)
    n = 200
    # Baseline risk uniform in [0.05, 0.30]
    p_hat = rng.uniform(0.05, 0.30, n)
    # First 100 patients OK (odds ratio 1); next 100 have OR = 2 (mortality doubled)
    odds = p_hat / (1 - p_hat)
    ORs = np.concatenate([np.ones(100), 2.0 * np.ones(100)])
    p_obs = (odds * ORs) / (1 + odds * ORs)
    y = (rng.random(n) < p_obs).astype(int)

    v = vlad(y, p_hat)
    print(f"  VLAD cumulative (expected - observed) at n=100: {v[99]:.2f}   at n=200: {v[199]:.2f}")

    cs = risk_adjusted_cusum(y, p_hat, OR=2.0, h=4.5)
    print(f"  Risk-adjusted CUSUM (OR=2, h=4.5): first signal at t = {cs['first_signal']}"
          f"   detection delay (shift @ t=100) = "
          f"{cs['first_signal'] - 100 if cs['first_signal'] else 'never'}\n")

    print("--- library cross-check (R vlad; Python pyspc + custom) ---")
