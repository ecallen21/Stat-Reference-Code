"""Sample Size Re-estimation (Reference Sec 47.260).

Wittes & Brittain 1990; Cui, Hung & Wang 1999 Biometrics.
At an interim look, re-estimate the required sample size
based on OBSERVED variance (blinded, preserves alpha) or
effect size (unblinded, needs adjustment).

    Fixed design n_planned = 2 * (z_{alpha/2} + z_beta)^2 * sigma^2 / delta^2
    Blinded SSR: after n_1 patients, plug in sigma_hat_blinded to get new n
    Unblinded SSR (Cui-Hung-Wang): keep alpha via weighted combination
        Z_final = sqrt(w_1) Z_1 + sqrt(w_2) Z_2   (pre-specified weights)
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def required_n(delta, sigma, alpha=0.025, power=0.8):
    """Two-sample z-test sample size per arm (approximate)."""
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha); z_b = norm.ppf(power)
    return int(np.ceil(2 * (z_a + z_b) ** 2 * sigma ** 2 / delta ** 2))


def blinded_variance(y_all):
    """Pooled variance ignoring group labels (blinded estimator)."""
    return float(np.var(y_all, ddof=1))


def chw_combined_z(Z_1, Z_2, w_1, w_2):
    """Cui-Hung-Wang weighted-Z that preserves alpha regardless of new n_2."""
    return np.sqrt(w_1) * Z_1 + np.sqrt(w_2) * Z_2


if __name__ == "__main__":
    print("=== Sample Size Re-estimation (Wittes-Brittain 1990; Cui-Hung-Wang 1999) ===\n")
    rng = np.random.default_rng(0)

    delta_planned = 0.5; sigma_planned = 1.0
    n_plan = required_n(delta_planned, sigma_planned, alpha=0.025, power=0.8)
    print(f"  Planning: delta = {delta_planned}, sigma = {sigma_planned},")
    print(f"    n_per_arm = {n_plan} (80% power at alpha=0.025)\n")

    # But TRUE sigma is larger: 1.4
    sigma_true = 1.4
    n_needed = required_n(delta_planned, sigma_true, alpha=0.025, power=0.8)
    print(f"  True sigma = {sigma_true} => really need n_per_arm = {n_needed}\n")

    # Blinded interim at n_1 = 30 per arm; estimate variance from pooled outcomes
    n_1 = 30
    yA = rng.normal(0, sigma_true, n_1)
    yB = rng.normal(delta_planned, sigma_true, n_1)
    sigma_hat = blinded_variance(np.concatenate([yA, yB])) ** 0.5
    n_reest = required_n(delta_planned, sigma_hat, alpha=0.025, power=0.8)
    print(f"  Blinded SSR at n_1={n_1}: sigma_hat = {sigma_hat:.3f}, revised n = {n_reest}")
    print(f"    (blinded uses pooled variance so alpha is preserved automatically)\n")

    # Unblinded SSR with CHW re-weighting
    # Stage 1 at info fraction 0.4: Z_1 based on n_1
    Z_1 = (yB.mean() - yA.mean()) / (sigma_hat * np.sqrt(2 / n_1))
    # Stage 2 with re-estimated (increased) sample size n_2 > planned
    n_2 = n_reest - n_1
    yA2 = rng.normal(0, sigma_true, n_2)
    yB2 = rng.normal(delta_planned, sigma_true, n_2)
    Z_2 = (yB2.mean() - yA2.mean()) / (sigma_true * np.sqrt(2 / n_2))
    w_1, w_2 = 0.4, 0.6                                            # pre-specified weights (sum to 1)
    Z_chw = chw_combined_z(Z_1, Z_2, w_1, w_2)
    print(f"  Unblinded CHW SSR:  Z_1 = {Z_1:.2f}, Z_2 = {Z_2:.2f}, Z_CHW = {Z_chw:.2f}")
    print(f"    reject H0 (|Z| > 1.96)? {'YES' if abs(Z_chw) > 1.96 else 'NO'}")

    # Type-I preservation: simulate under H0
    print(f"\n  Type-I under H0 (delta_true=0) with CHW re-weighting:")
    n_sim = 20_000
    rejects = 0
    for _ in range(n_sim):
        yA = rng.normal(0, sigma_true, n_1); yB = rng.normal(0, sigma_true, n_1)
        sh = float(np.std(np.concatenate([yA, yB]), ddof=1))
        Z1 = (yB.mean() - yA.mean()) / (sh * np.sqrt(2 / n_1))
        n2 = rng.integers(40, 200)                                # ANY re-estimated n
        yA2 = rng.normal(0, sigma_true, n2); yB2 = rng.normal(0, sigma_true, n2)
        Z2 = (yB2.mean() - yA2.mean()) / (sigma_true * np.sqrt(2 / n2))
        if abs(chw_combined_z(Z1, Z2, 0.4, 0.6)) > 1.96: rejects += 1
    print(f"    Type-I = {rejects / n_sim:.4f}   (nominal 0.05, CHW preserves)")

    print("\n--- library cross-check (rpact R; gsDesign R; PASS commercial) ---")
