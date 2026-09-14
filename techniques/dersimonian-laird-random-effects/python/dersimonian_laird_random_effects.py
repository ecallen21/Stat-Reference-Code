"""DerSimonian-Laird Random-Effects Meta-Analysis (Ref Sec 47.269).

DerSimonian & Laird 1986 Contr Clin Trials. Classical
random-effects estimator for pooling K studies with estimated
effect y_k and within-study variance v_k. Between-study
variance tau^2 is estimated via method-of-moments from the
Cochran Q statistic:

    Q = sum w_k^FE * (y_k - y_bar^FE)^2      where w_k^FE = 1/v_k
    tau^2 = max(0, (Q - (K-1)) / (sum w_k - sum w_k^2 / sum w_k))
    w_k^RE = 1 / (v_k + tau^2)
    y_bar^RE = sum(w_k^RE y_k) / sum(w_k^RE)

Wider CI than fixed-effect when heterogeneity is present.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def dersimonian_laird(y, v):
    """DL random-effects meta-analysis; returns dict of estimates."""
    y = np.asarray(y, float); v = np.asarray(v, float)
    K = len(y)
    w_fe = 1.0 / v
    ybar_fe = np.sum(w_fe * y) / np.sum(w_fe)
    Q = float(np.sum(w_fe * (y - ybar_fe) ** 2))
    df = K - 1
    tau2 = max(0.0, (Q - df) / (np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)))
    w_re = 1.0 / (v + tau2)
    ybar_re = float(np.sum(w_re * y) / np.sum(w_re))
    se_re = float(1.0 / np.sqrt(np.sum(w_re)))
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    return dict(y_fe=float(ybar_fe), y_re=ybar_re, se_fe=1.0 / np.sqrt(np.sum(w_fe)),
                se_re=se_re, Q=Q, df=df, tau2=float(tau2), I2=float(I2), w_re=w_re)


if __name__ == "__main__":
    print("=== DerSimonian-Laird Random-Effects (DL 1986) ===\n")
    rng = np.random.default_rng(0)

    # Simulate 10 studies with true effect 0.5 + between-study SD tau=0.2
    tau_true = 0.2
    K = 10
    theta_k = rng.normal(0.5, tau_true, K)
    n_k = rng.integers(50, 500, K)
    v_k = 1.0 / n_k                                              # within-study SE^2
    y_k = np.array([rng.normal(t, np.sqrt(v)) for t, v in zip(theta_k, v_k)])

    print(f"  K = {K} studies, true mean = 0.5, true tau = {tau_true}\n")
    print(f"  {'study':>5}  {'y_k':>7}  {'v_k':>7}  {'w_FE':>7}  {'w_RE':>7}")
    fit = dersimonian_laird(y_k, v_k)
    w_fe = 1 / v_k
    for k in range(K):
        print(f"  {k + 1:>5}  {y_k[k]:>7.3f}  {v_k[k]:>7.4f}  {w_fe[k]:>7.1f}  {fit['w_re'][k]:>7.1f}")

    print(f"\n  Fixed-effect pooled:   {fit['y_fe']:.3f}   SE {fit['se_fe']:.3f}")
    print(f"  DL random-effects:     {fit['y_re']:.3f}   SE {fit['se_re']:.3f}")
    print(f"  Cochran Q = {fit['Q']:.2f}   df = {fit['df']}   I^2 = {fit['I2']*100:.1f}%")
    print(f"  tau^2 (between-study var) = {fit['tau2']:.4f}   (true tau^2 = {tau_true**2:.4f})")

    print(f"\n  RE CI is wider than FE because heterogeneity is acknowledged.")
    print(f"  95% CI (FE):  ({fit['y_fe']-1.96*fit['se_fe']:.3f}, {fit['y_fe']+1.96*fit['se_fe']:.3f})")
    print(f"  95% CI (RE):  ({fit['y_re']-1.96*fit['se_re']:.3f}, {fit['y_re']+1.96*fit['se_re']:.3f})")

    print("\n--- library cross-check (metafor::rma R; PythonMeta / pymeta) ---")
