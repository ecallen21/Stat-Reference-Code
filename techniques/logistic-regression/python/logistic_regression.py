"""Binary logistic regression (Reference §7.1).

Model:
    P(y_i = 1 | x_i) = pi_i = 1 / (1 + exp(-x_i' beta))
    log(pi_i / (1 - pi_i)) = x_i' beta      <- the logit / log-odds link

Fit by maximum likelihood. We implement Fisher scoring / IRLS:

  Working response:  z_i = eta_i + (y_i - pi_i) / (pi_i (1 - pi_i))
  Weights:           w_i = pi_i (1 - pi_i)
  Update:            beta <- (X' W X)^{-1} X' W z

Each step is a weighted least-squares fit; converges quadratically near the MLE.
The information matrix at convergence gives the SE of beta and the OR CIs.

Outputs
-------
- coefficients beta, SE, Wald z, p-values
- odds ratios = exp(beta) with 95% CIs
- log-likelihood, AIC, deviance
- model deviance test (vs null) and likelihood-ratio chi-square
- classification accuracy at threshold 0.5 (just a sanity check; for honest
  performance use cross-validation)
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


def _logistic(eta):
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))


def fit(X, y, max_iter: int = 50, tol: float = 1e-8) -> dict:
    """Fit a binary logistic regression by Fisher scoring / IRLS.

    Parameters
    ----------
    X : design matrix (intercept column included if you want one).
    y : 0/1 response.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)
    ll_prev = -np.inf
    converged = False; it = 0
    for it in range(1, max_iter + 1):
        eta = X @ beta
        pi = _logistic(eta)
        w = pi * (1 - pi)
        # Avoid singular W near 0/1 fitted probabilities
        w = np.clip(w, 1e-12, None)
        z = eta + (y - pi) / w
        # Weighted least squares step:  beta = (X' W X)^{-1} X' W z
        sw = np.sqrt(w)
        Xw = X * sw[:, None]; zw = z * sw
        beta_new, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
        # Log-likelihood
        ll = float(np.sum(y * np.log(np.clip(pi, 1e-15, None))
                          + (1 - y) * np.log(np.clip(1 - pi, 1e-15, None))))
        if abs(ll - ll_prev) < tol:
            converged = True
            break
        ll_prev = ll
        beta = beta_new
    eta = X @ beta; pi = _logistic(eta)
    w = np.clip(pi * (1 - pi), 1e-12, None)
    fisher_info = X.T @ (X * w[:, None])
    var_beta = np.linalg.pinv(fisher_info)
    se = np.sqrt(np.diag(var_beta))
    z_stats = beta / se
    p_vals = 2 * stats.norm.sf(np.abs(z_stats))
    or_ = np.exp(beta)
    zc = stats.norm.ppf(0.975)
    or_lo = np.exp(beta - zc * se); or_hi = np.exp(beta + zc * se)

    # Null model log-lik (intercept-only)
    p_bar = float(np.mean(y))
    ll_null = float(n * (p_bar * math.log(max(p_bar, 1e-15))
                          + (1 - p_bar) * math.log(max(1 - p_bar, 1e-15))))
    ll_full = float(np.sum(y * np.log(np.clip(pi, 1e-15, None))
                            + (1 - y) * np.log(np.clip(1 - pi, 1e-15, None))))
    lr_chi2 = 2 * (ll_full - ll_null)
    lr_df = p - 1                                         # for intercept-only null
    lr_p = float(stats.chi2.sf(lr_chi2, lr_df)) if lr_df > 0 else float("nan")

    # Pseudo R^2 measures
    mcfadden = 1 - ll_full / ll_null if ll_null != 0 else float("nan")
    null_dev = -2 * ll_null; resid_dev = -2 * ll_full
    aic = 2 * p - 2 * ll_full
    accuracy = float(np.mean((pi >= 0.5).astype(int) == y))

    return {"beta": beta.tolist(), "se": se.tolist(),
            "z": z_stats.tolist(), "p_values": p_vals.tolist(),
            "odds_ratio": or_.tolist(),
            "or_ci_lower": or_lo.tolist(), "or_ci_upper": or_hi.tolist(),
            "log_likelihood": ll_full, "ll_null": ll_null,
            "lr_chi_square": lr_chi2, "lr_df": lr_df, "lr_p_value": lr_p,
            "null_deviance": null_dev, "residual_deviance": resid_dev,
            "aic": aic, "mcfadden_r_squared": mcfadden,
            "accuracy_at_0.5": accuracy, "iterations": it,
            "converged": converged}


def library_versions(X_no_const, y):
    import statsmodels.api as sm
    X = sm.add_constant(X_no_const)
    res = sm.Logit(y, X).fit(disp=False)
    return {"statsmodels Logit params": res.params.tolist(),
            "statsmodels Logit bse": res.bse.tolist(),
            "statsmodels Logit pvalues": res.pvalues.tolist(),
            "statsmodels Logit llf": float(res.llf)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    true_beta = np.array([-0.5, 1.0, -0.7])
    eta = true_beta[0] + true_beta[1] * x1 + true_beta[2] * x2
    p = 1 / (1 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    X = np.column_stack([np.ones(n), x1, x2])

    res = fit(X, y)
    names = ["intercept", "x1", "x2"]
    print(f"=== Fit ({res['iterations']} iterations, converged={res['converged']}) ===")
    print(f"  {'name':10s} {'beta':>9s} {'se':>9s} {'z':>7s} {'p':>10s} "
          f"{'OR':>8s} {'OR 95% CI':>22s}")
    for i, nm in enumerate(names):
        print(f"  {nm:10s} {res['beta'][i]:+9.4f} {res['se'][i]:9.4f} "
              f"{res['z'][i]:+7.2f} {res['p_values'][i]:10.4g} "
              f"{res['odds_ratio'][i]:8.4f} "
              f"[{res['or_ci_lower'][i]:.3f}, {res['or_ci_upper'][i]:.3f}]")
    print(f"\n  log-likelihood = {res['log_likelihood']:.4f}")
    print(f"  null deviance  = {res['null_deviance']:.4f}")
    print(f"  resid deviance = {res['residual_deviance']:.4f}")
    print(f"  AIC            = {res['aic']:.4f}")
    print(f"  LR chi^2({res['lr_df']}) = {res['lr_chi_square']:.4f}  p = {res['lr_p_value']:.4g}")
    print(f"  McFadden R^2   = {res['mcfadden_r_squared']:.4f}")
    print(f"  accuracy @ 0.5 = {res['accuracy_at_0.5']:.4f}")
    print("\n--- library (statsmodels.Logit) ---")
    for k, v in library_versions(np.column_stack([x1, x2]), y).items():
        print(f"  {k}: {v}")
