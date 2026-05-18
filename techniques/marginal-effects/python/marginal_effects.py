"""Marginal effects for GLMs (Reference §7.37, §7.47).

A GLM coefficient `beta_j` is on the link scale (log-odds for logistic, log-rate
for Poisson, ...). Often the substantive question is "what's the effect on the
response, in original units, for a unit increase in x_j?" -- a derivative or
discrete change on the response (mu) scale, not the link (eta) scale.

Three flavors implemented
-------------------------
- AME (Average Marginal Effect):
      mean over observations of  d_mu / d_x_j evaluated at each observed x_i.
      The default; reports an "average across the sample" effect.
- MEM (Marginal Effect at the Means):
      d_mu / d_x_j evaluated at  x = mean(x).
      Easier to compute by hand but can be misleading if the mean point isn't
      representative (e.g. mean of a binary variable is nonsense).
- MER (Marginal Effect at Representative values):
      d_mu / d_x_j evaluated at a user-specified scenario (e.g. men, age 60).

For BINARY variables, the marginal effect is the discrete difference:
      mu(x_j = 1) - mu(x_j = 0), with other covariates held at the chosen values.

Derivatives of the link function
  logit   : d_pi / d_x_j = pi * (1 - pi) * beta_j
  probit  : d_pi / d_x_j = phi(eta) * beta_j
  log     : d_mu / d_x_j = mu * beta_j        (Poisson, Gamma w/ log)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from typing import Callable    # stdlib: type hint meaning 'a function'

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def ame_logit(X, beta) -> np.ndarray:
    """Average marginal effects for logit: AME_j = mean_i pi_i (1 - pi_i) beta_j."""
    eta = X @ beta; pi = 1 / (1 + np.exp(-np.clip(eta, -500, 500)))
    return float(np.mean(pi * (1 - pi))) * np.asarray(beta)


def mem_logit(X, beta) -> np.ndarray:
    """Marginal effect at means: pi(x_bar) (1 - pi(x_bar)) * beta."""
    eta = float(np.mean(X, axis=0) @ beta)
    pi = 1 / (1 + np.exp(-eta))
    return pi * (1 - pi) * np.asarray(beta)


def ame_probit(X, beta) -> np.ndarray:
    """AME for probit: mean_i phi(eta_i) * beta_j."""
    eta = X @ beta
    return float(np.mean(stats.norm.pdf(eta))) * np.asarray(beta)


def ame_poisson(X, beta, offset=None) -> np.ndarray:
    """AME for Poisson with log link: mean_i mu_i * beta_j."""
    eta = X @ beta
    if offset is not None:
        eta = eta + np.asarray(offset)
    mu = np.exp(np.clip(eta, -500, 500))
    return float(np.mean(mu)) * np.asarray(beta)


def discrete_ame_logit(X, beta, binary_index: int) -> float:
    """For a binary predictor x_j: average of (pi(x_j=1) - pi(x_j=0)) over obs."""
    X = np.asarray(X, dtype=float)
    X0 = X.copy(); X0[:, binary_index] = 0
    X1 = X.copy(); X1[:, binary_index] = 1
    pi0 = 1 / (1 + np.exp(-(X0 @ beta)))
    pi1 = 1 / (1 + np.exp(-(X1 @ beta)))
    return float(np.mean(pi1 - pi0))


def library_versions(X, y, beta_hat):
    out = {}
    try:
        import statsmodels.api as sm
        Xc = sm.add_constant(X)
        res = sm.Logit(y, Xc).fit(disp=False)
        # AME via statsmodels' get_margeff
        marg = res.get_margeff(at="overall")
        out["statsmodels AME"] = marg.margeff.tolist()
        marg_mean = res.get_margeff(at="mean")
        out["statsmodels MEM"] = marg_mean.margeff.tolist()
    except Exception as exc:
        out["statsmodels"] = f"error: {exc}"
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(10)
    n = 500
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    treat = rng.integers(0, 2, n).astype(float)
    eta = -0.5 + 0.8 * x1 - 0.4 * x2 + 0.6 * treat
    p = 1 / (1 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    X = np.column_stack([np.ones(n), x1, x2, treat])
    beta_true = np.array([-0.5, 0.8, -0.4, 0.6])

    names = ["intercept", "x1", "x2", "treat"]
    print("=== AME (logit, continuous derivative form) ===")
    ame = ame_logit(X, beta_true)
    for j, nm in enumerate(names):
        print(f"  {nm:10s}: AME = {ame[j]:+.4f}  (beta = {beta_true[j]:+.4f})")
    print("\n=== MEM (at sample means) ===")
    mem = mem_logit(X, beta_true)
    for j, nm in enumerate(names):
        print(f"  {nm:10s}: MEM = {mem[j]:+.4f}")
    print("\n=== Discrete AME for the binary 'treat' (E[pi|treat=1] - E[pi|treat=0]) ===")
    print(f"  discrete AME(treat) = {discrete_ame_logit(X, beta_true, 3):+.4f}")
    print("\n--- library (statsmodels get_margeff) ---")
    for k, v in library_versions(np.column_stack([x1, x2, treat]), y, beta_true).items():
        print(f"  {k}: {v}")
