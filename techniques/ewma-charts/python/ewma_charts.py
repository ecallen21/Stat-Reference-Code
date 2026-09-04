"""EWMA control chart (Reference Sec 37.3).

Roberts (1959). Exponentially-weighted moving-average chart:

  z_i = lambda * x_i + (1 - lambda) * z_{i-1},   z_0 = mu_0.

Variance:
  sigma_z^2 = sigma^2 * (lambda / (2 - lambda)) * (1 - (1 - lambda)^{2i}).

Limits at 3 * sigma_z. lambda ~ 0.1-0.3 for small shifts;
lambda ~ 0.4 for medium; lambda -> 1 recovers Shewhart.

Here we implement EWMA with time-varying variance limits and
detect a 0.75-sigma shift.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ewma_chart(x, mu0, sigma, lam=0.2, L=3.0):
    n = len(x)
    z = np.zeros(n)
    z[0] = lam * x[0] + (1 - lam) * mu0
    for i in range(1, n):
        z[i] = lam * x[i] + (1 - lam) * z[i - 1]
    idx = np.arange(1, n + 1)
    var_i = sigma ** 2 * (lam / (2 - lam)) * (1 - (1 - lam) ** (2 * idx))
    sig_i = np.sqrt(var_i)
    UCL = mu0 + L * sig_i
    LCL = mu0 - L * sig_i
    signals = np.where((z > UCL) | (z < LCL))[0]
    return {"z": z, "UCL": UCL, "LCL": LCL,
             "first_signal": int(signals[0]) if len(signals) else None}


if __name__ == "__main__":
    print("=== EWMA chart (Roberts 1959) ===\n")
    rng = np.random.default_rng(0)
    n = 200
    change = 100
    shift = 0.75
    x = np.concatenate([rng.normal(0, 1, change),
                         rng.normal(shift, 1, n - change)])

    for lam in (0.1, 0.2, 0.4):
        r = ewma_chart(x, mu0=0, sigma=1, lam=lam, L=3.5)
        d = (r["first_signal"] - change) if r["first_signal"] is not None else "never"
        print(f"  lambda = {lam}   first signal at t = {r['first_signal']}"
              f"   detection delay = {d}")

    print("\n  Smaller lambda -> better for smaller shifts (longer memory).\n")
    print("--- library cross-check (R qcc::ewma; Python pyspc) ---")
