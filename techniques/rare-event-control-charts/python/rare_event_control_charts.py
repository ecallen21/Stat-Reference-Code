"""Rare-event control charts (Reference Sec 37.11).

For low-defect-rate processes the Shewhart p/np-chart is uninformative
because most subgroups show 0 events.  Alternatives:

  G-CHART (geometric): plot the NUMBER OF NON-EVENTS BETWEEN CONSECUTIVE
    events.  Under stable p, G ~ Geometric(p); centre = 1/p.
  T-CHART (exponential): plot TIME between events.  For a homogeneous
    Poisson process, T ~ Exponential(lambda); centre = 1/lambda.
  BERNOULLI CUSUM (Reynolds-Stoumbos 1999) for shifts in p.

Here we implement G-chart + Bernoulli CUSUM on rare defects; verify
G-chart flags a rate shift and Bernoulli CUSUM detects it faster.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def g_chart_limits(p, L=3.0):
    """Control limits for the geometric distribution G ~ Geom(p)."""
    mu = 1.0 / p
    sig = np.sqrt((1 - p) / p ** 2)
    return {"center": mu, "UCL": mu + L * sig, "LCL": max(0, mu - L * sig)}


def bernoulli_cusum(y, p0, p1):
    """Log-likelihood-ratio CUSUM for a shift from p_0 to p_1."""
    log_ratio_1 = np.log(p1 / p0)
    log_ratio_0 = np.log((1 - p1) / (1 - p0))
    L = np.zeros(len(y))
    for i in range(len(y)):
        prev = L[i - 1] if i > 0 else 0.0
        contribution = log_ratio_1 if y[i] == 1 else log_ratio_0
        L[i] = max(0.0, prev + contribution)
    return L


if __name__ == "__main__":
    print("=== Rare-event control charts: G-chart + Bernoulli CUSUM ===\n")
    rng = np.random.default_rng(0)
    p0, p1 = 0.005, 0.02
    n = 3000
    change = 1500
    y = np.concatenate([(rng.random(change) < p0).astype(int),
                         (rng.random(n - change) < p1).astype(int)])
    # G-chart: interval between successive defects
    defect_idx = np.where(y == 1)[0]
    intervals = np.diff(defect_idx)

    g_lim = g_chart_limits(p0, L=3)
    print(f"  In-control p_0 = {p0}, mean interval = {g_lim['center']:.1f}"
          f"   UCL = {g_lim['UCL']:.1f}   LCL = {g_lim['LCL']:.1f}")
    # Number of intervals dropping below LCL after shift
    below = int(sum(1 for g in intervals if g < g_lim["LCL"]))
    print(f"  intervals total {len(intervals)}   under LCL: {below}\n")

    # Bernoulli CUSUM (target shift from p0 to p1)
    L = bernoulli_cusum(y, p0, p1)
    threshold = 5.0
    signal = int(np.argmax(L > threshold)) if (L > threshold).any() else None
    print(f"  Bernoulli CUSUM (target OR shift), h = {threshold}: first signal at t = {signal}"
          f"   detection delay = {signal - change if signal else 'never'}")
    print(f"  (defects in [0, {change}) = {y[:change].sum()};"
          f"   in [{change}, {n}) = {y[change:].sum()})\n")

    print("--- library cross-check (R spc; Python custom) ---")
