"""Internal-external cross-validation (IECV) for multi-site prediction (Reference Sec 39.25).

Debray et al. (2013), Steyerberg-Harrell (2016).  For a prediction
model developed on data from K sites:

  For k = 1..K:
    develop model on the K-1 non-k sites
    validate on held-out site k -- AUC_k, CITL_k, slope_k
  Report per-site performance + pooled (via meta-analysis of DerSimonian-Laird
  or fixed-effects) discrimination and calibration.

Compared to leave-one-observation-out CV, IECV honestly reflects
between-site heterogeneity and transportability.
"""
from __future__ import annotations    # stdlib

import warnings                                          # suppress noise

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression    # baseline logistic

warnings.filterwarnings("ignore")


def _lr():
    return LogisticRegression(C=1e12, solver="lbfgs", max_iter=500)


def _auc(y, p):
    order = np.argsort(-p)
    y_o = y[order]
    pos = y_o == 1; neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return float(np.cumsum(pos)[neg].sum() / (pos.sum() * neg.sum()))


def _cal(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    m = _lr().fit(lp[:, None], y)
    return {"CITL": float(y.mean() - p.mean()), "slope": float(m.coef_[0][0])}


def iecv(X, y, site):
    """Leave-one-site-out CV; return per-site + pooled discrimination + calibration."""
    sites = np.unique(site)
    rows = []
    for k in sites:
        te = site == k
        tr = ~te
        m = _lr().fit(X[tr], y[tr])
        p_te = m.predict_proba(X[te])[:, 1]
        row = {"site": int(k), "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "AUC": _auc(y[te], p_te)}
        row.update(_cal(y[te], p_te))
        rows.append(row)
    aucs = np.array([r["AUC"] for r in rows])
    ns = np.array([r["n_test"] for r in rows], dtype=float)
    pooled_auc = float((aucs * ns).sum() / ns.sum())
    pooled_slope = float(np.average([r["slope"] for r in rows], weights=ns))
    pooled_citl = float(np.average([r["CITL"] for r in rows], weights=ns))
    return {"per_site": rows, "pooled_AUC": pooled_auc,
            "pooled_slope": pooled_slope, "pooled_CITL": pooled_citl}


if __name__ == "__main__":
    print("=== Internal-external CV (leave-one-site-out) ===\n")
    rng = np.random.default_rng(0)
    K = 5
    n_per_site = 200
    site = np.repeat(np.arange(K), n_per_site)
    p = 4
    # Global coefficient + site-specific perturbation
    beta_global = np.array([0.7, -0.5, 0.4, 0.2])
    intercepts = np.array([-0.6, -0.2, -0.9, -0.3, -1.0])[site]
    site_beta_scale = np.array([1.0, 0.85, 1.1, 0.95, 0.90])[site][:, None]
    X = rng.normal(0, 1, (K * n_per_site, p))
    logit = (X * (beta_global * site_beta_scale)).sum(axis=1) + intercepts
    y = (rng.random(K * n_per_site) < 1 / (1 + np.exp(-logit))).astype(int)

    r = iecv(X, y, site)
    print(f"  {'site':>4s} {'n_test':>7s} {'AUC':>6s} {'CITL':>7s} {'slope':>7s}")
    for row in r["per_site"]:
        print(f"  {row['site']:>4d} {row['n_test']:>7d} {row['AUC']:>6.3f}"
              f" {row['CITL']:>+7.3f} {row['slope']:>7.3f}")
    print(f"\n  Pooled (weighted by n_test):")
    print(f"    AUC   = {r['pooled_AUC']:.3f}")
    print(f"    CITL  = {r['pooled_CITL']:+.3f}")
    print(f"    slope = {r['pooled_slope']:.3f}\n")

    print("--- library cross-check (R metamisc::valmeta, rms::validate; Python custom) ---")
