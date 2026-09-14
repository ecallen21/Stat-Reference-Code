"""Leave-One-Out Meta-Analysis (Ref Sec 47.273).

Standard sensitivity analysis: refit the meta-analysis leaving
out each study in turn, showing how the pooled estimate and
its CI change. Flags studies that are individually influential
or that drive the overall conclusion.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def dl(y, v):
    """DerSimonian-Laird RE pooled estimate + SE."""
    w_fe = 1.0 / v
    ybar_fe = np.sum(w_fe * y) / np.sum(w_fe)
    Q = np.sum(w_fe * (y - ybar_fe) ** 2); K = len(y)
    tau2 = max(0.0, (Q - (K - 1)) / (np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)))
    w_re = 1.0 / (v + tau2)
    ybar = np.sum(w_re * y) / np.sum(w_re)
    se = 1.0 / np.sqrt(np.sum(w_re))
    return float(ybar), float(se), float(tau2)


def leave_one_out(y, v):
    """Return list of (study_removed, pooled_est, SE, 95% CI)."""
    y = np.asarray(y); v = np.asarray(v)
    out = []
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        m, s, _ = dl(y[mask], v[mask])
        out.append((i + 1, m, s, m - 1.96 * s, m + 1.96 * s))
    return out


if __name__ == "__main__":
    print("=== Leave-One-Out Meta-Analysis ===\n")
    rng = np.random.default_rng(0)

    # 8 studies + 1 outlier
    K = 8
    theta_k = rng.normal(0.4, 0.1, K)
    v_k = 1.0 / rng.integers(50, 300, K)
    y_k = np.array([rng.normal(t, np.sqrt(v)) for t, v in zip(theta_k, v_k)])
    # Add an outlier study
    y_k = np.concatenate([y_k, [1.5]]); v_k = np.concatenate([v_k, [0.02]])
    K = len(y_k)

    m_all, se_all, tau2 = dl(y_k, v_k)
    print(f"  Full meta-analysis (K={K}):  pooled = {m_all:.3f}   SE = {se_all:.3f}   "
          f"tau^2 = {tau2:.4f}")
    print(f"  95% CI: ({m_all - 1.96 * se_all:.3f}, {m_all + 1.96 * se_all:.3f})\n")

    print(f"  Leave-one-out (study removed -> pooled, 95% CI):")
    print(f"    {'removed':>7}  {'y_k':>7}  {'v_k':>7}  {'pooled':>7}  {'CI':>20}   change")
    for i, m, s, lo, hi in leave_one_out(y_k, v_k):
        diff = m - m_all
        marker = "  <-- INFLUENTIAL" if abs(diff) > 2 * se_all else ""
        print(f"    {i:>7}  {y_k[i-1]:>+7.3f}  {v_k[i-1]:>7.4f}  {m:>7.3f}  ({lo:.3f}, {hi:.3f})  {diff:+.3f}{marker}")

    print(f"\n  Studies whose removal shifts the pooled estimate by > 2 SE are FLAGGED.")
    print(f"  Common practice: report full and leave-one-out estimates for transparency.")

    print("\n--- library cross-check (metafor::leave1out R; meta::metainf R; PythonMeta) ---")
