"""Bayesian model comparison (Reference §14.20, §14.21, §14.22).

Given posterior draws from two models fit to the same data, compare their
out-of-sample predictive performance.  All three information criteria below
approximate the expected log pointwise predictive density (elpd).

DIC (Spiegelhalter et al. 2002)
    DIC = -2 log p(y | theta_bar) + 2 p_D
    p_D = 2 (log p(y | theta_bar) - E_post[log p(y | theta)])
    Sensitive to parameterization; deprecated in favor of WAIC/LOO.

WAIC (Watanabe 2010; Gelman-Hwang-Vehtari 2014)
    lpd  = sum_i log E_post[p(y_i | theta)]
    p_WAIC = sum_i Var_post[log p(y_i | theta)]
    elpd_WAIC = lpd - p_WAIC
    WAIC = -2 * elpd_WAIC

PSIS-LOO (Pareto smoothed importance sampling LOO; Vehtari-Gelman-Gabry 2017)
    Approximate leave-one-out CV using importance weights r_i = 1 / p(y_i | theta_s).
    Weights are smoothed by fitting a Pareto distribution to the tails.
    elpd_LOO = sum_i log E_LOO[p(y_i | theta)]
    LOO = -2 * elpd_LOO
    Reports Pareto-k diagnostics; high k means the LOO approximation is unreliable.

Bayes factors (caveats)
    B_12 = p(y | M_1) / p(y | M_2) = ratio of MARGINAL likelihoods.
    Extremely sensitive to prior width; can shift by orders of magnitude
    when priors change.  Not usable with improper priors.  Prefer WAIC / LOO
    for predictive comparison.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def waic(log_lik) -> dict:
    """WAIC from an S x N matrix of per-obs log-likelihoods across S posterior draws."""
    log_lik = np.asarray(log_lik, dtype=float)
    S, N = log_lik.shape
    lpd = np.sum(_logsumexp(log_lik, axis=0) - math.log(S))
    p_waic = np.sum(np.var(log_lik, axis=0, ddof=1))
    elpd = lpd - p_waic
    return {"waic": float(-2 * elpd), "elpd_waic": float(elpd),
            "lpd": float(lpd), "p_waic": float(p_waic),
            "n_obs": int(N), "n_draws": int(S)}


def loo_psis(log_lik) -> dict:
    """PSIS-LOO from an S x N log-lik matrix.

    Simplified Pareto-tail fit; for production use arviz.loo or loo::loo.
    """
    log_lik = np.asarray(log_lik, dtype=float)
    S, N = log_lik.shape
    log_weights = -log_lik  # importance weights for LOO
    # Smooth top 20% of weights with Pareto (Vehtari, Gelman, Gabry 2017)
    ks = np.empty(N); elpd_i = np.empty(N)
    for i in range(N):
        lw = log_weights[:, i] - np.max(log_weights[:, i])
        w = np.exp(lw)
        M = max(3, int(math.ceil(min(0.2 * S, 3 * math.sqrt(S)))))
        top = np.sort(w)[-M:]
        # Simple Hill/method-of-moments-ish shape estimate
        thresh = top[0]
        excess = top - thresh
        if excess.mean() > 0:
            k = 1 - thresh / excess.mean()
            k = max(k, 0.0)
            # Replace top M with Pareto-smoothed quantiles
            ranks = (np.arange(M) + 0.5) / M
            smoothed = thresh + excess.mean() * (1 - ranks) ** (-k) * k / (1 + k) if k > 0 else top
            smoothed = np.clip(smoothed, 0, top.max())
            w_sorted = np.sort(w)
            w_sorted[-M:] = smoothed
            w = w_sorted
        else:
            k = 0.0
        w /= w.sum()
        # elpd_i = log E_LOO[p(y_i | theta)] via importance sampling of log_lik[:, i]
        elpd_i[i] = _logsumexp(np.log(w + 1e-300) + log_lik[:, i])
        ks[i] = k
    elpd = float(np.sum(elpd_i))
    return {"loo": float(-2 * elpd), "elpd_loo": elpd,
            "pareto_k_max": float(ks.max()),
            "pareto_k_high_frac": float(np.mean(ks > 0.7)),
            "n_obs": int(N), "n_draws": int(S)}


def dic(theta_draws, log_lik_at_theta_bar: float, log_lik_draws) -> dict:
    """DIC from posterior draws and pointwise log-lik at posterior mean."""
    log_lik_draws = np.asarray(log_lik_draws, dtype=float)  # S vector
    D_bar = -2 * log_lik_draws.mean()
    D_hat = -2 * log_lik_at_theta_bar
    p_D = D_bar - D_hat
    return {"dic": float(D_bar + p_D), "p_D": float(p_D),
            "n_draws": int(len(log_lik_draws))}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y_true = 1.0 + 0.8 * x + rng.normal(0, 1.0, n)

    # Fit two conjugate Normal regressions (correct vs intercept-only)
    def bayes_lr_draws(X, y, n_draws=1500, seed=1):
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=float); n, p = X.shape
        V_n = np.linalg.pinv(X.T @ X + 1e-4 * np.eye(p))
        m_n = V_n @ (X.T @ y)
        # Posterior over sigma^2 (empirical Bayes-ish)
        resid = y - X @ m_n
        a_n = n / 2 + 1; b_n = 0.5 * resid @ resid + 1
        sig2_draws = 1 / rng.gamma(a_n, 1 / b_n, n_draws)
        beta_draws = np.array([rng.multivariate_normal(m_n, s * V_n) for s in sig2_draws])
        return beta_draws, sig2_draws

    def per_obs_log_lik(X, y, betas, sig2s):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        S = len(sig2s); N = len(y)
        ll = np.empty((S, N))
        for s in range(S):
            mu = X @ betas[s]
            ll[s] = -0.5 * math.log(2 * math.pi * sig2s[s]) - 0.5 * (y - mu) ** 2 / sig2s[s]
        return ll

    # Model 1: intercept + slope
    X1 = np.column_stack([np.ones(n), x])
    betas1, sig2s1 = bayes_lr_draws(X1, y_true)
    ll1 = per_obs_log_lik(X1, y_true, betas1, sig2s1)

    # Model 2: intercept only
    X2 = np.ones((n, 1))
    betas2, sig2s2 = bayes_lr_draws(X2, y_true)
    ll2 = per_obs_log_lik(X2, y_true, betas2, sig2s2)

    print("=== Two Bayesian LR models on same data (M1: intercept+slope, M2: intercept) ===")
    w1 = waic(ll1); w2 = waic(ll2)
    print(f"  Model 1 (correct):   WAIC = {w1['waic']:.2f}  (elpd = {w1['elpd_waic']:.2f}, p_WAIC = {w1['p_waic']:.2f})")
    print(f"  Model 2 (nested):    WAIC = {w2['waic']:.2f}  (elpd = {w2['elpd_waic']:.2f}, p_WAIC = {w2['p_waic']:.2f})")
    print(f"  Delta WAIC (M1 - M2) = {w1['waic'] - w2['waic']:.2f}  (smaller WAIC -> better predictive)")

    print("\n=== PSIS-LOO ===")
    l1 = loo_psis(ll1); l2 = loo_psis(ll2)
    print(f"  Model 1: LOO = {l1['loo']:.2f}  (elpd = {l1['elpd_loo']:.2f})  max Pareto k = {l1['pareto_k_max']:.2f}")
    print(f"  Model 2: LOO = {l2['loo']:.2f}  (elpd = {l2['elpd_loo']:.2f})  max Pareto k = {l2['pareto_k_max']:.2f}")

    print("\n--- library cross-check (arviz.waic / arviz.loo) ---")
    try:
        import arviz as az
        idata1 = az.from_dict(posterior_predictive={"y": ll1.reshape(1, *ll1.shape)},
                              log_likelihood={"y": ll1.reshape(1, *ll1.shape)},
                              observed_data={"y": y_true})
        print(f"  arviz.waic: {az.waic(idata1).elpd_waic:.2f}")
    except Exception as ex:
        print(f"  (arviz not available or errored: {ex})")
