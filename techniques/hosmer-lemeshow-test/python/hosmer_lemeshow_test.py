"""Hosmer-Lemeshow Goodness-of-Fit Test (Reference Sec 47.256).

Hosmer & Lemeshow 1980 'Goodness-of-fit tests for the multiple
logistic regression model', CommStat. Assess calibration of a
binary probability model by binning predictions into g deciles
of risk and comparing observed vs expected events per bin:

    H = sum_{k=1}^g (O_k - E_k)^2 / (E_k (1 - E_k / n_k))

Under H0 (model calibrated), H ~ chi-square with g - 2 df.
Small p-value => poor fit. Standard for logistic-regression
diagnostics; use with caution (sensitive to g, weak power).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def hosmer_lemeshow(y, p, g=10):
    """Hosmer-Lemeshow chi-square + df + p-value."""
    from scipy.stats import chi2
    order = np.argsort(p)
    y = np.asarray(y)[order]; p = np.asarray(p)[order]
    bins = np.array_split(np.arange(len(p)), g)
    H = 0.0
    for b in bins:
        O = int(y[b].sum())
        E = float(p[b].sum())
        n = len(b)
        pi_bar = E / n
        denom = E * (1 - pi_bar)
        if denom <= 0: continue
        H += (O - E) ** 2 / denom
    df = g - 2
    return {"H_statistic": H, "df": df, "p_value": 1 - chi2.cdf(H, df)}


if __name__ == "__main__":
    print("=== Hosmer-Lemeshow Test (Hosmer & Lemeshow 1980) ===\n")
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=3000, n_features=8, n_informative=5,
                                    weights=[0.7, 0.3], random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)

    # Well-calibrated model: plain LR
    clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    p_te = clf.predict_proba(Xte)[:, 1]
    res = hosmer_lemeshow(yte, p_te, g=10)
    print(f"  Plain LR (should fit well):")
    print(f"    H = {res['H_statistic']:.2f}   df = {res['df']}   p = {res['p_value']:.3f}")
    print(f"    -> {'REJECT H0 (poor fit)' if res['p_value'] < 0.05 else 'do NOT reject (fit ok)'}")

    # Deliberately miscalibrate: shift probabilities toward 0
    p_bad = p_te ** 2
    res_bad = hosmer_lemeshow(yte, p_bad, g=10)
    print(f"\n  Squashed probabilities p -> p^2 (should fit poorly):")
    print(f"    H = {res_bad['H_statistic']:.2f}   df = {res_bad['df']}   p = {res_bad['p_value']:.3g}")
    print(f"    -> {'REJECT H0 (poor fit)' if res_bad['p_value'] < 0.05 else 'do NOT reject (fit ok)'}")

    # Per-decile observed vs expected for the good model
    print("\n  Decile calibration table (well-fit model):")
    order = np.argsort(p_te)
    y_s = yte[order]; p_s = p_te[order]
    bins = np.array_split(np.arange(len(p_s)), 10)
    print(f"    {'decile':>6}  {'mean_p':>8}  {'observed':>10}  {'expected':>10}")
    for i, b in enumerate(bins, 1):
        print(f"    {i:>6}  {p_s[b].mean():>8.3f}  {y_s[b].sum():>10}  {p_s[b].sum():>10.2f}")

    print("\n  Well-calibrated models show observed ~ expected within each risk decile.")

    print("\n--- library cross-check (ResourceSelection::hoslem.test R; statsmodels workaround) ---")
