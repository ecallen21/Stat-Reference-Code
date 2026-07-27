"""Cohen's kappa: agreement between two raters (Reference §8.4).

For two raters classifying the same n subjects into K categories, kappa
corrects the raw agreement rate for the amount of agreement expected by chance:

    kappa = (p_o - p_e) / (1 - p_e)

    p_o = sum_i n_ii / n            (observed agreement rate)
    p_e = sum_i (row_i * col_i)/n^2  (expected under independence)

Interpretation (Landis-Koch, 1977):
    < 0.20  slight   0.21-0.40  fair   0.41-0.60  moderate
    0.61-0.80 substantial       0.81-1.00  almost perfect

ASE (Fleiss 1969) and Wald z-test are provided. PABAK (Byrt et al., 1993)
adjusts kappa for the prevalence + bias artifacts that make it drop toward zero
even when raw agreement is high, when one category dominates.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def confusion_matrix(rater1: Sequence, rater2: Sequence, categories=None) -> tuple:
    """Build the K x K confusion matrix from two parallel label sequences.

    Returns
    -------
    (categories, matrix)  -- ``matrix[i][j]`` is the count of subjects that
    rater1 said = categories[i] and rater2 said = categories[j].
    """
    if len(rater1) != len(rater2):
        raise ValueError("rater1 and rater2 must have equal length")
    if categories is None:
        categories = sorted(set(rater1) | set(rater2))
    idx = {c: i for i, c in enumerate(categories)}
    K = len(categories)
    m = np.zeros((K, K), dtype=int)
    for a, b in zip(rater1, rater2):
        m[idx[a], idx[b]] += 1
    return categories, m


def cohens_kappa(confusion) -> dict:
    """Cohen's kappa + Fleiss (1969) ASE + Wald z-test of kappa != 0.

    Parameters
    ----------
    confusion : K x K matrix of paired-rating counts (as returned by
        :func:`confusion_matrix`).
    """
    m = np.asarray(confusion, dtype=float)
    n = m.sum()
    if n == 0:
        return {"kappa": float("nan"), "note": "empty confusion matrix"}
    K = m.shape[0]
    p_o = np.trace(m) / n
    row = m.sum(axis=1) / n
    col = m.sum(axis=0) / n
    p_e = float((row * col).sum())
    if p_e == 1.0:
        return {"kappa": 1.0 if p_o == 1.0 else 0.0,
                "p_observed": p_o, "p_expected": p_e,
                "note": "chance agreement is 1; kappa undefined"}
    kappa = (p_o - p_e) / (1 - p_e)

    # Fleiss (1969) ASE under H0: kappa = 0
    diag = np.diag(m) / n
    A = ((diag / (1 - p_e)) *
         (1 - (row + col) * (1 - kappa))).sum()
    off_diag = 0.0
    for i in range(K):
        for j in range(K):
            if i == j: continue
            off_diag += m[i, j] / n * (col[i] + row[j]) ** 2
    B = ((1 - kappa) ** 2) / (1 - p_e) ** 2 * off_diag
    C = (kappa - p_e * (1 - kappa)) ** 2 / (1 - p_e) ** 2
    var = (A + B - C) / n
    se = math.sqrt(max(var, 0.0))
    z = kappa / se if se > 0 else float("inf")
    p_two = float(2 * stats.norm.sf(abs(z)))
    return {"kappa": kappa, "p_observed": p_o, "p_expected": p_e,
            "ASE": se, "z": z, "p_value": p_two,
            "CI95_lower": kappa - 1.96 * se, "CI95_upper": kappa + 1.96 * se,
            "n": int(n), "K": K,
            "method": "Cohen's kappa (Fleiss 1969 ASE)"}


def pabak(confusion) -> dict:
    """Prevalence-Adjusted, Bias-Adjusted Kappa (Byrt et al., 1993).

    PABAK = (K * p_o - 1) / (K - 1)     for a K x K table (2 x 2 special case:
        PABAK = 2 * p_o - 1).

    Sensitive only to the diagonal sum; ignores marginals.
    """
    m = np.asarray(confusion, dtype=float)
    n = m.sum()
    K = m.shape[0]
    if n == 0 or K < 2:
        return {"PABAK": float("nan")}
    p_o = np.trace(m) / n
    return {"PABAK": (K * p_o - 1) / (K - 1),
            "p_observed": p_o,
            "note": "prevalence-and-bias-adjusted (Byrt et al., 1993)"}


def run_all(rater1, rater2) -> dict:
    cats, cm = confusion_matrix(rater1, rater2)
    return {"categories": cats, "confusion": cm.tolist(),
            "cohens_kappa": cohens_kappa(cm),
            "PABAK": pabak(cm)}


def library_versions(rater1, rater2):
    from sklearn.metrics import cohen_kappa_score
    try:
        from statsmodels.stats.inter_rater import cohens_kappa as sm_kappa
        cats, cm = confusion_matrix(rater1, rater2)
        sm = sm_kappa(np.asarray(cm))
        sm_dict = {"kappa": float(sm.kappa), "ASE": float(sm.std_kappa),
                   "z": float(sm.z_value), "p": float(sm.pvalue_one_sided) * 2}
    except Exception as ex:
        sm_dict = {"unavailable": str(ex)}
    return {"sklearn.metrics.cohen_kappa_score": float(cohen_kappa_score(rater1, rater2)),
            "statsmodels.cohens_kappa": sm_dict}


if __name__ == "__main__":
    import random
    random.seed(3)
    cats = ["low", "medium", "high"]
    n = 200
    # Simulate two raters who agree ~70% of the time
    rater1 = [random.choice(cats) for _ in range(n)]
    rater2 = [r if random.random() < 0.70 else random.choice([c for c in cats if c != r])
              for r in rater1]

    print("=== Confusion + Cohen's kappa ===")
    out = run_all(rater1, rater2)
    print("  categories:", out["categories"])
    print("  confusion:", out["confusion"])
    print("  kappa:", out["cohens_kappa"])
    print("  PABAK:", out["PABAK"])

    print("\n--- library ---")
    for k, v in library_versions(rater1, rater2).items():
        print(f"  {k}: {v}")
