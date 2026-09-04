"""High-dimensional propensity scores (Reference Sec 43.4).

Schneeweiss et al. 2009.  Automated confounder selection from
thousands of EHR / claims codes (diagnoses, procedures, prescriptions
in multiple time windows).

Pipeline:
  1. FEATURE GENERATION: for each code x time-window, create binary
     indicator.
  2. PREVALENCE FILTER: drop codes with prevalence < 5 % or > 95 %.
  3. RANK BY BIAS: for each code, compute the Bross bias multiplier
     approximating potential confounding, keep the top-K (typical
     100-500).
  4. Estimate propensity score by LOGISTIC REGRESSION of treatment on
     selected codes + a few investigator-selected covariates.
  5. Adjust outcome analysis by matching / weighting / stratification
     on the PS.
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression


def bross_multiplier(p1, p0, RR):
    """Bross 1966 approximate bias multiplier from a confounder.

    p1: prevalence of code among treated; p0: among untreated;
    RR: outcome-confounder association proxy (default 2).
    """
    return abs((p1 * (RR - 1) + 1) / (p0 * (RR - 1) + 1) - 1)


def hdps_select(X, T, K=100, min_prev=0.05, RR_prior=2.0):
    """Return top-K high-dim confounders ranked by Bross multiplier."""
    n, p = X.shape
    prev = X.mean(axis=0)
    keep = np.where((prev >= min_prev) & (prev <= 1 - min_prev))[0]
    scores = []
    for j in keep:
        p1 = X[T == 1, j].mean(); p0 = X[T == 0, j].mean()
        scores.append((j, bross_multiplier(p1, p0, RR_prior)))
    scores.sort(key=lambda kv: -kv[1])
    return [j for j, _ in scores[:K]]


def hdps_propensity(X, T, K=100):
    idx = hdps_select(X, T, K=K)
    m = LogisticRegression(C=1e12, solver="lbfgs", max_iter=500).fit(X[:, idx], T)
    return m.predict_proba(X[:, idx])[:, 1], idx


def att_iptw(y, T, ps):
    """Treatment-on-treated IPTW effect."""
    w = np.where(T == 1, 1.0, ps / (1 - ps))
    return float((w * y * T).sum() / (T.sum()) - (w * y * (1 - T)).sum() / ((1 - T) * w).sum())


if __name__ == "__main__":
    print("=== High-dimensional propensity score (hdPS) ===\n")
    rng = np.random.default_rng(0)
    n, p = 2000, 500
    X = (rng.random((n, p)) < 0.10).astype(int)          # sparse codes
    # 5 real confounders (columns 0-4)
    conf = X[:, :5].sum(axis=1)
    logit_T = -1.5 + 0.6 * conf + rng.normal(0, 0.5, n)
    T = (rng.random(n) < 1 / (1 + np.exp(-logit_T))).astype(int)
    y = 2.0 * T + 0.5 * conf + rng.normal(0, 1, n)       # true ATE = 2

    # Naive vs adjusted for hdPS-selected covariates
    naive = y[T == 1].mean() - y[T == 0].mean()
    ps, idx = hdps_propensity(X, T, K=50)
    att = att_iptw(y, T, ps)
    print(f"  True ATE = 2.0")
    print(f"  Naive (unadjusted) mean diff = {naive:.3f}   (confounded)")
    print(f"  hdPS IPTW estimate           = {att:.3f}")
    print(f"  # of high-dim codes selected  = {len(idx)}")
    print(f"  Of top-10 selected, how many are true confounders (idx < 5)? {sum(1 for j in idx[:10] if j < 5)}\n")

    print("--- library cross-check (R FeatureExtraction + CohortMethod OHDSI; Python sklearn + custom) ---")
