"""Heckman selection model (Reference §5.21).

Selection bias: y_i is observed only for a self-selected subset (labour
force participants, adopters, respondents to a survey).  If the selection
process is correlated with the outcome error, OLS on the OBSERVED subset
is biased.

Type-II tobit / Heckman (1979):
    Selection:   z_i^* = W_i gamma + u_i,      d_i = I(z_i^* > 0)
    Outcome  :   y_i^* = X_i beta  + eps_i,    observed only when d_i = 1
    (u_i, eps_i) ~ N(0, [[1, rho sigma], [rho sigma, sigma^2]])

Two-step estimator (Heckman 1979)
    1. Probit regression of d on W -> gamma_hat.
    2. Compute the INVERSE MILLS RATIO
        lambda_i = phi(W_i gamma_hat) / Phi(W_i gamma_hat)
    3. OLS regress y on X + lambda for the SELECTED subset:
        E[y | X, d = 1] = X beta + rho sigma * lambda_i
    The coefficient on lambda estimates rho * sigma; a t-test on it is a
    test of selection bias.

Second-step SEs need a correction because lambda is estimated (standard
Heckman formula); the demo below reports the naive OLS SE and notes the
correction.

For unbiased efficient estimates use joint MLE (Heckman 1976).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _probit_fit(W, d):
    W = np.asarray(W, dtype=float); d = np.asarray(d, dtype=int)
    def neg_ll(gamma):
        z = W @ gamma
        return -np.sum(d * stats.norm.logcdf(z) + (1 - d) * stats.norm.logcdf(-z))
    res = minimize(neg_ll, np.zeros(W.shape[1]), method="BFGS")
    return res.x


def heckman_two_step(X, y, W, d) -> dict:
    """Heckman two-step selection correction.

    X : outcome design (must include intercept), only used for d==1 rows.
    y : outcome, observed only for d==1 rows.
    W : selection design (must include intercept).
    d : 1 if y observed, 0 if not.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    W = np.asarray(W, dtype=float); d = np.asarray(d, dtype=int)
    # Stage 1: probit
    gamma_hat = _probit_fit(W, d)
    # Inverse Mills ratio for the observed subset
    Wg = W @ gamma_hat
    lambda_i = stats.norm.pdf(Wg) / stats.norm.cdf(Wg)
    # Stage 2: OLS on selected
    obs = (d == 1)
    X_obs = X[obs]; y_obs = y[obs]; lam_obs = lambda_i[obs]
    X_aug = np.column_stack([X_obs, lam_obs])
    beta_aug, *_ = np.linalg.lstsq(X_aug, y_obs, rcond=None)
    resid = y_obs - X_aug @ beta_aug
    sigma2_naive = float(resid @ resid / (len(y_obs) - X_aug.shape[1]))
    se_naive = np.sqrt(sigma2_naive * np.diag(np.linalg.pinv(X_aug.T @ X_aug)))
    return {"selection_gamma": gamma_hat,
            "outcome_beta": beta_aug[:-1],
            "inverse_mills_coef": float(beta_aug[-1]),
            "outcome_se_naive": se_naive[:-1],
            "mills_se_naive": float(se_naive[-1]),
            "mills_t": float(beta_aug[-1] / se_naive[-1]),
            "mills_p": float(2 * stats.norm.sf(abs(beta_aug[-1] / se_naive[-1]))),
            "n_total": int(len(d)), "n_observed": int(obs.sum()),
            "method": "Heckman two-step (naive SEs; use MLE / Heckman correction for exact SEs)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 800
    # Selection eq: participation
    w = rng.normal(size=n); z_x = rng.normal(size=n)
    u = rng.normal(size=n)
    eps = 0.7 * u + math.sqrt(1 - 0.7 ** 2) * rng.normal(size=n)  # rho = 0.7
    # Latent selection: z^* = 0.5 + 1.0 w + u
    z_star = 0.5 + 1.0 * w + u
    d = (z_star > 0).astype(int)
    # Latent outcome: y^* = 2 + 1.5 z_x + eps  (sigma_eps = 1)
    y_star = 2 + 1.5 * z_x + eps
    y = np.where(d == 1, y_star, np.nan)

    print(f"=== N = {n}, observed = {int(d.sum())} ===")

    print("\n=== Naive OLS on selected only (biased) ===")
    obs = d == 1
    X_naive = np.column_stack([np.ones(obs.sum()), z_x[obs]])
    y_naive = y_star[obs]
    beta_naive, *_ = np.linalg.lstsq(X_naive, y_naive, rcond=None)
    print(f"  intercept = {beta_naive[0]:.3f}, slope = {beta_naive[1]:.3f}  (true 2.0, 1.5)")

    print("\n=== Heckman two-step ===")
    W = np.column_stack([np.ones(n), w])
    X = np.column_stack([np.ones(n), z_x])
    r = heckman_two_step(X, np.where(d == 1, y_star, 0), W, d)
    print(f"  selection gamma  = {r['selection_gamma'].round(3)}   (true intercept 0.5, slope 1.0)")
    print(f"  outcome beta     = {r['outcome_beta'].round(3)}      (true 2.0, 1.5)")
    print(f"  inverse-Mills coef = {r['inverse_mills_coef']:.3f}  (t = {r['mills_t']:.2f}, p = {r['mills_p']:.4f})")

    print("\n--- library cross-check (statsmodels: no dedicated Heckman; use R's sampleSelection) ---")
