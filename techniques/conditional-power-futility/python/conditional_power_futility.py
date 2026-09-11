"""Conditional Power / Futility Stopping (Reference Sec 47.259).

Lan & Wittes 1988 Biometrics. Given INTERIM data Z_1 at
information fraction t_1, the CONDITIONAL POWER (CP) is the
probability of rejecting H0 at the final look given the
current data and an ASSUMED true effect:

    CP(delta) = P(reject at t=1 | Z_1, delta)
              = 1 - Phi( (z_alpha - Z_1 sqrt(t_1) - delta*(1-t_1)) / sqrt(1-t_1) )

Common CHOICES for delta:
    - null:      delta = 0                (worst case)
    - alternative: delta = planned effect  (best case)
    - current:   delta = observed effect  (Bayesian-flavour)

Rule of thumb: STOP FOR FUTILITY if CP(observed) < 0.20.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def conditional_power(Z_1, t_1, delta, alpha=0.025):
    """CP given interim Z, info fraction t_1, assumed drift delta."""
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha)
    num = z_alpha - Z_1 * np.sqrt(t_1) - delta * (1 - t_1)
    return 1 - norm.cdf(num / np.sqrt(1 - t_1))


if __name__ == "__main__":
    print("=== Conditional Power / Futility (Lan & Wittes 1988) ===\n")

    # Trial with planned drift delta_1 = 3 (nominal 80% power at alpha=0.025)
    delta_planned = 3.0
    alpha = 0.025
    t_1 = 0.5                                                     # 50% of information

    print(f"  Planned drift = {delta_planned}, alpha = {alpha}, interim at t = {t_1}\n")
    print(f"  {'Interim Z':>10}  |  CP(delta=0)  CP(observed)  CP(planned)")

    for Z_1 in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]:
        # 'observed' drift infers delta from Z_1 = delta * sqrt(t_1)
        delta_obs = Z_1 / np.sqrt(t_1)
        cp_null = conditional_power(Z_1, t_1, 0, alpha)
        cp_obs  = conditional_power(Z_1, t_1, delta_obs, alpha)
        cp_plan = conditional_power(Z_1, t_1, delta_planned, alpha)
        marker = "  <-- FUTILITY" if cp_obs < 0.20 else ""
        print(f"  {Z_1:>10.2f}  |  {cp_null:>10.3f}  {cp_obs:>11.3f}  {cp_plan:>10.3f}{marker}")

    print(f"\n  Interpretation:")
    print(f"    CP(null)     = pessimistic, assume no effect from now on")
    print(f"    CP(observed) = keep trending as we are (common futility criterion)")
    print(f"    CP(planned)  = optimistic, effect equals the design assumption")
    print(f"  When CP(observed) < 0.20, futility stopping is often triggered.")

    # Non-binding futility inflates real Type I only slightly; simulate:
    print(f"\n  Non-binding futility (may continue after CP<20%): simulate under H0")
    rng = np.random.default_rng(0)
    n_sim = 50_000
    x = rng.standard_normal((n_sim, 2))
    dn = np.array([t_1, 1 - t_1])
    dW = x * np.sqrt(dn)
    Z_int = dW[:, 0] / np.sqrt(t_1)
    Z_fin = dW.sum(axis=1) / 1.0
    # Type-I without futility
    typeI_full = float((Z_fin > 1.96).mean())
    # Type-I with strict futility: stop if CP(obs) < 0.20 at interim
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha)
    cp_obs_arr = 1 - norm.cdf((z_alpha - Z_int * np.sqrt(t_1) - (Z_int / np.sqrt(t_1)) * (1 - t_1)) / np.sqrt(1 - t_1))
    keep = cp_obs_arr >= 0.20
    typeI_futil = float(((Z_fin > 1.96) & keep).mean())
    print(f"    Type-I without futility: {typeI_full:.4f}")
    print(f"    Type-I with futility:    {typeI_futil:.4f}   (lowered - futility is CONSERVATIVE)")

    print("\n--- library cross-check (gsDesign R; rpact; statsmodels: manual) ---")
