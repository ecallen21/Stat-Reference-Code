"""Bridge sampling for marginal likelihood / evidence (Reference Sec 25.7).

Meng & Wong 1996 'Simulating ratios of normalising constants via a
simple identity: a theoretical exploration', Statistica Sinica; Gronau
et al. 2017. Given samples theta_i ~ p(theta) (target) with
unnormalised density p_u = p(theta) * Z (Z unknown), estimate Z by
comparing to a PROPOSAL g(theta) with known normalising constant:

    Z_hat_bridge =
       [ (1/N1) sum_{i=1}^{N1} p_u(theta_prop^i) / (alpha * p_u + beta * g)(theta_prop^i) ]
     / [ (1/N2) sum_{j=1}^{N2} g(theta_target^j) / (alpha * p_u + beta * g)(theta_target^j) ]

with weights alpha, beta chosen adaptively. Meng-Wong optimal bridge:
    b_MW(theta) = 1 / (r * p_u + g)     iteratively solve for r.

More stable than importance sampling (harmonic-mean estimator is
notoriously bad).

Application: model evidence p(y | M) = integral p(y | theta, M) *
p(theta | M) dtheta. Ratio -> Bayes factor.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def bridge_sampling(samples_target, log_p_u, log_g, samples_proposal, log_g_prop):
    """Iterative Meng-Wong bridge estimator.

    samples_target : N1 draws from the target posterior.
    log_p_u        : function log p_u(theta) (unnormalised target log-density).
    log_g          : function log g(theta) (proposal log-density, normalised).
    samples_proposal : N2 draws from g.
    """
    #  l1 : samples from TARGET, w1 = p_u(theta) / g(theta)
    #  l2 : samples from PROPOSAL, w2 = p_u(theta) / g(theta)
    l1 = np.exp(np.array([log_p_u(t) - log_g(t) for t in samples_target]))
    l2 = np.exp(np.array([log_p_u(t) - log_g_prop(t) for t in samples_proposal]))
    N1 = len(l1); N2 = len(l2)
    s1 = N1 / (N1 + N2); s2 = N2 / (N1 + N2)
    #  Iterate Gronau et al. (2017) eq (5)-(6): r = numerator / denominator
    r = 1.0
    for _ in range(200):
        num = np.mean(l2 / (s1 * l2 + s2 * r))
        den = np.mean(1.0 / (s1 * l1 + s2 * r))
        r_new = num / den
        if abs(r_new - r) / max(r, 1e-8) < 1e-10:
            r = r_new; break
        r = r_new
    return np.log(r)


if __name__ == "__main__":
    print("=== Bridge sampling -- normalising constant estimation ===\n")
    #  Target: N(3, 1) treated as if unknown Z: p_u(theta) = 5 * phi(theta; 3, 1)
    #  True Z = 5.
    rng = np.random.default_rng(0)
    N1 = N2 = 2000
    samples_target = rng.normal(3, 1, size=N1)          # exact target draws
    #  Proposal g = N(3.5, 1.5) (offset a bit)
    samples_proposal = rng.normal(3.5, 1.5, size=N2)

    def log_p_u(t): return np.log(5.0) + stats.norm.logpdf(t, 3, 1)
    def log_g(t): return stats.norm.logpdf(t, 3.5, 1.5)
    def log_g_prop(t): return stats.norm.logpdf(t, 3.5, 1.5)

    log_Z_bridge = bridge_sampling(samples_target, log_p_u, log_g, samples_proposal, log_g_prop)
    Z_bridge = float(np.exp(log_Z_bridge))
    print(f"  True Z            = 5.000")
    print(f"  Bridge estimator  = {Z_bridge:.3f}")

    #  Compare to harmonic-mean estimator (known bad)
    hme = 1 / np.mean(1 / np.exp([log_p_u(t) for t in samples_target]))
    print(f"  Harmonic-mean     = {hme:.3f}   (unstable / biased)")

    #  Compare to naive importance sampling on the proposal
    w = np.array([np.exp(log_p_u(t) - log_g_prop(t)) for t in samples_proposal])
    is_est = float(np.mean(w))
    print(f"  Naive IS estimator = {is_est:.3f}")

    print("\n--- library cross-check (bridgesampling R; from-scratch Python) ---")
