"""Variational inference (Reference §14.24, §14.25).

Approximate the posterior p(theta | y) with a simpler family q(theta; phi)
by MAXIMIZING the evidence lower bound:

    ELBO(phi) = E_q[log p(y, theta)] - E_q[log q(theta; phi)]

Maximizing the ELBO minimizes KL(q || p).  Vastly faster than MCMC on big
data; trades off exactness (q rarely equals p).

Mean-field VI
    Restrict q to a product of independent factors:
        q(theta) = prod_j q_j(theta_j)
    Coordinate-ascent updates: cycle through each q_j and set it equal to
    exp(E_{-j}[log p(y, theta)]) / Z_j.  For conjugate models this gives
    closed-form recurrences (analog of Gibbs sampling).

ADVI (Automatic Differentiation VI, Kucukelbir et al. 2017)
    Fix q to a diagonal Gaussian in a transformed unconstrained space,
    use pathwise / reparameterization gradients of the ELBO, optimize with
    Adam.  Implemented in Stan / PyMC / NumPyro.

The demo below fits a mean-field Normal q(mu, log sigma) to the posterior
of a Gaussian model with known variance, and separately runs coordinate-
ascent VI on a Beta-Binomial (which is a conjugate example so VI is exact).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def advi_normal_mean(y, sigma_known: float, mu_prior: float = 0.0,
                     tau_prior: float = 100.0, n_iter: int = 500,
                     lr: float = 0.05, n_mc: int = 32, seed: int = 0) -> dict:
    """Mean-field Gaussian VI on a 1-D Normal-Normal (should recover the exact posterior).

    q(mu) = Normal(m, exp(log_s)^2).  Optimize (m, log_s) by SGD on ELBO.
    """
    y = np.asarray(y, dtype=float); n = len(y); rng = np.random.default_rng(seed)
    m = float(y.mean()); log_s = math.log(sigma_known / math.sqrt(n))
    for it in range(n_iter):
        # Sample epsilon ~ N(0, 1); theta = m + exp(log_s) * eps
        eps = rng.standard_normal(n_mc)
        s = math.exp(log_s); theta = m + s * eps
        # d/dtheta [log p(y | theta) + log p(theta)]; log q handled by entropy term below
        d_ll = -n * (theta - y.mean()) / sigma_known ** 2
        d_prior = -(theta - mu_prior) / tau_prior ** 2
        g_theta = d_ll + d_prior
        # Reparameterization gradients wrt (m, log_s)
        g_m = g_theta.mean()
        g_log_s = (g_theta * eps * s).mean() + 1  # +1 from d/dlog_s [log_s]
        m += lr * g_m
        log_s += lr * g_log_s
    return {"posterior_mean_mu": float(m),
            "posterior_sd_mu": float(math.exp(log_s)),
            "n_iter": int(n_iter),
            "method": "Mean-field Gaussian VI (reparameterization gradient)"}


def cavi_beta_binomial(alpha_prior: float, beta_prior: float,
                       successes: int, trials: int) -> dict:
    """Coordinate-ascent VI on Beta-Binomial (single Beta factor -> closed-form)."""
    a = alpha_prior + successes
    b = beta_prior + trials - successes
    return {"q_alpha": float(a), "q_beta": float(b),
            "posterior_mean_theta": float(a / (a + b)),
            "note": "For Beta-Binomial the mean-field q equals the exact posterior.",
            "method": "CAVI on Beta-Binomial (exact)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Mean-field VI on Normal-Normal (known sigma=1, n=50) ===")
    y = rng.normal(2.5, 1.0, 50)
    r = advi_normal_mean(y, sigma_known=1.0, tau_prior=10.0, n_iter=2000, lr=0.01, n_mc=64, seed=0)
    print(f"  VI posterior mean mu: {r['posterior_mean_mu']:.4f}")
    print(f"  VI posterior sd mu:   {r['posterior_sd_mu']:.4f}")
    # Analytic
    n = len(y); ybar = y.mean()
    exact_prec = 1 / 100 + n / 1.0
    exact_var = 1 / exact_prec
    exact_mean = exact_var * (0 / 100 + n * ybar / 1.0)
    print(f"  Exact posterior mean: {exact_mean:.4f}, sd: {math.sqrt(exact_var):.4f}")

    print("\n=== CAVI on Beta-Binomial with 8/12 successes and Uniform(1,1) prior ===")
    r = cavi_beta_binomial(1.0, 1.0, 8, 12)
    print(f"  q(theta) = Beta({r['q_alpha']}, {r['q_beta']}), posterior mean = {r['posterior_mean_theta']:.4f}")

    print("\n--- library cross-check (pymc, if available) ---")
    try:
        import pymc as pm
        with pm.Model() as m:
            mu = pm.Normal("mu", 0, 10)
            pm.Normal("y", mu=mu, sigma=1.0, observed=y)
            approx = pm.fit(3000, method="advi", progressbar=False, random_seed=0)
            trace = approx.sample(1000)
        vi_mean = float(trace.posterior["mu"].mean())
        vi_sd = float(trace.posterior["mu"].std())
        print(f"  pymc ADVI: mean = {vi_mean:.4f}, sd = {vi_sd:.4f}")
    except Exception as ex:
        print(f"  (pymc not available: {ex})")
