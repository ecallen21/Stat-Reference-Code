"""Pattern-mixture models for MNAR missing data (Reference §16.x extra).

Factor the joint distribution of (Y, R) as
    f(Y, R) = f(Y | R) * f(R)
i.e. condition on the *missingness pattern* R.

Pattern-mixture identifies the mean of the missing (R=1) responses by an
unverifiable SENSITIVITY delta:

    E[Y | R = 1] = E[Y | R = 0] + delta      (delta = departure from MAR)

With covariates x, the "delta adjustment" shifts the imputed values by delta:

    Y_imp = MAR-imputed Y + delta * R

Reported: ATE estimate under delta = 0 (MAR) vs delta in a range (tipping-point).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def pattern_mixture_delta(y, r, x=None, deltas=(-1, -0.5, 0, 0.5, 1),
                          n_impute: int = 20, seed: int = 0) -> dict:
    """MAR imputation of missing y (regression + Gaussian residual);
    then shift imputed values by delta to simulate MNAR departures."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float); r = np.asarray(r, dtype=int)
    n = len(y)
    if x is None:
        x = np.ones((n, 1))
    else:
        x = np.column_stack([np.ones(n), np.asarray(x, dtype=float)])
    p = x.shape[1]
    # OLS on observed
    obs = r == 0
    beta, *_ = np.linalg.lstsq(x[obs], y[obs], rcond=None)
    resid = y[obs] - x[obs] @ beta
    sigma = float(resid.std(ddof=1))

    means_by_delta = {}
    for d in deltas:
        ests = []
        for _ in range(n_impute):
            y_imp = y.copy()
            miss = ~obs
            y_imp[miss] = x[miss] @ beta + rng.normal(scale=sigma, size=miss.sum()) + d
            ests.append(y_imp.mean())
        means_by_delta[float(d)] = {"mean": float(np.mean(ests)),
                                     "sd": float(np.std(ests, ddof=1))}
    return {"means_by_delta": means_by_delta,
            "n": n, "n_miss": int((r == 1).sum()),
            "method": "pattern-mixture with delta sensitivity"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    y = 2 + 1.5 * x + rng.normal(scale=1.0, size=n)
    # true mean of y (population): E[y] = 2

    # MNAR mechanism: missing more likely when y is HIGH
    p_miss = 1 / (1 + np.exp(-(y - 3) * 0.7))
    r = (rng.uniform(size=n) < p_miss).astype(int)
    y_obs = y.copy(); y_obs[r == 1] = np.nan
    print(f"=== Pattern-mixture MNAR sensitivity (n={n}) ===")
    print(f"  fraction missing: {r.mean():.3f}")
    print(f"  true E[Y] = {y.mean():.3f}")
    print(f"  complete-case mean = {y[r == 0].mean():.3f}   (biased low; missing were high)")

    res = pattern_mixture_delta(np.where(np.isnan(y_obs), 0, y_obs), r, x=x,
                                 deltas=(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
                                 n_impute=30)
    print(f"\n  {'delta':>7} {'mean E[Y]':>10} {'MC SE':>7}")
    for d in sorted(res["means_by_delta"].keys()):
        m = res["means_by_delta"][d]
        print(f"  {d:>7.2f} {m['mean']:>10.3f} {m['sd']:>7.3f}")
    print(f"\n  delta = 0 corresponds to MAR; positive delta = missing-are-higher (matches truth here)")

    print("\n--- library cross-check (R jomo::jomo; mice::mice with delta adjust) ---")
