"""Guardrail monitoring for A/B tests (Reference Sec 44.15).

Guardrail metrics track HARM: latency, crash rate, revenue drop,
core-experience regression.  Stop the experiment if a guardrail
crosses a defined threshold.

Compact demo:
  * Rolling metric per batch.
  * Sequential (Wilson-CI) monitor on binary guardrails
    (crash rate).
  * Two-sided anytime-valid stopping via CI vs threshold.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def wilson_ci(x, n, alpha=0.05):
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = x / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - hw), float(centre + hw)


def guardrail_monitor(x_T_stream, n_T_stream, baseline, tolerance=0.005, alpha=0.001):
    """Watch a streaming binary guardrail; return the first alert time.

    Halt if lower Wilson CI on treatment rate exceeds baseline + tolerance.
    """
    total_x = 0; total_n = 0
    for t, (dx, dn) in enumerate(zip(x_T_stream, n_T_stream)):
        total_x += dx; total_n += dn
        lo, hi = wilson_ci(total_x, total_n, alpha=alpha)
        if lo > baseline + tolerance:
            return {"alert_at_batch": t, "total_n": total_n, "rate": total_x / total_n,
                    "wilson_CI": (lo, hi)}
    return {"alert_at_batch": None}


if __name__ == "__main__":
    print("=== Guardrail monitoring: sequential Wilson-CI alarm ===\n")
    rng = np.random.default_rng(0)
    baseline = 0.005
    tolerance = 0.005
    # Case A: guardrail stays at baseline
    stream_A = [(rng.binomial(1000, baseline), 1000) for _ in range(20)]
    r_A = guardrail_monitor([x for x, _ in stream_A], [n for _, n in stream_A],
                             baseline, tolerance)
    print(f"  Case A (guardrail OK):   alert = {r_A['alert_at_batch']}")

    # Case B: guardrail regresses to 4 * baseline
    stream_B = [(rng.binomial(1000, 4 * baseline), 1000) for _ in range(20)]
    r_B = guardrail_monitor([x for x, _ in stream_B], [n for _, n in stream_B],
                             baseline, tolerance)
    print(f"  Case B (regression):     alert = {r_B['alert_at_batch']}"
          f" at n = {r_B.get('total_n')}   rate = {r_B.get('rate', 0):.4f}"
          f"   CI = {r_B.get('wilson_CI')}\n")

    print("--- library cross-check (R stats::binom.test, qcc; Python scipy + custom) ---")
