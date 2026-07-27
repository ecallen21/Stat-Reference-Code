"""Weighted kappa for ordinal agreement between two raters (Reference §8.4).

Same setup as Cohen's kappa (K x K confusion matrix), but disagreements are
NOT treated as equally bad -- a "mild" vs "moderate" disagreement is smaller
than "mild" vs "severe". A weight matrix W with W_ii = 0 (perfect agreement)
and W_ij increasing in |i - j| encodes this.

Two standard weight schemes for ordinal categories 1..K:
    linear     : W_ij = |i - j| / (K - 1)
    quadratic  : W_ij = (i - j)^2 / (K - 1)^2

Kappa_w = 1 - (sum W_ij * p_ij) / (sum W_ij * p_i. * p_.j)

Cicchetti-Allison (linear) is common in behavioral research; Fleiss-Cohen
(quadratic) is the default in medicine and IS equal to a specific ICC
(Fleiss & Cohen 1973), which is why it dominates in inter-rater reliability
work.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def build_weight_matrix(K: int, scheme: str = "quadratic") -> np.ndarray:
    """Ordinal weight matrix. ``scheme`` in {"linear", "quadratic"}."""
    idx = np.arange(K)
    D = np.abs(idx[:, None] - idx[None, :]).astype(float)
    if scheme == "linear":
        return D / (K - 1) if K > 1 else np.zeros((K, K))
    if scheme == "quadratic":
        return (D ** 2) / ((K - 1) ** 2) if K > 1 else np.zeros((K, K))
    raise ValueError("scheme must be 'linear' or 'quadratic'")


def confusion_matrix(rater1, rater2, categories):
    if len(rater1) != len(rater2):
        raise ValueError("rater1 and rater2 must have equal length")
    idx = {c: i for i, c in enumerate(categories)}
    K = len(categories)
    m = np.zeros((K, K), dtype=int)
    for a, b in zip(rater1, rater2):
        m[idx[a], idx[b]] += 1
    return m


def _kappa_w_point(p, W):
    """Point estimate of weighted kappa from a probability matrix p (not counts)."""
    row = p.sum(axis=1); col = p.sum(axis=0)
    p_o = float((W * p).sum())
    p_e = float((W * np.outer(row, col)).sum())
    if p_e == 0:
        return 1.0 - p_o, p_o, p_e
    return 1.0 - p_o / p_e, p_o, p_e


def weighted_kappa(confusion, scheme: str = "quadratic",
                    n_boot: int = 2000, seed: int = 0) -> dict:
    """Weighted kappa on a K x K ordinal confusion matrix.

    SE via nonparametric bootstrap of paired ratings. Robust to the numerical
    cancellation that the Fleiss-Cohen-Everitt delta-method ASE suffers when
    disagreements concentrate along the diagonal (common when raters mostly
    agree, which is the typical use case).
    """
    m = np.asarray(confusion, dtype=float)
    K = m.shape[0]
    if K < 2:
        raise ValueError("need at least 2 categories")
    n = int(m.sum())
    p = m / n
    W = build_weight_matrix(K, scheme)
    kappa, p_o, p_e = _kappa_w_point(p, W)

    rng = np.random.default_rng(seed)
    flat = p.flatten()
    picks = rng.choice(K * K, size=(n_boot, n), p=flat)
    boot_kappas = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(picks[b], minlength=K * K).reshape(K, K).astype(float)
        boot_kappas[b], _, _ = _kappa_w_point(counts / n, W)
    se = float(boot_kappas.std(ddof=1))
    ci_lo, ci_hi = np.quantile(boot_kappas, [0.025, 0.975])
    z = kappa / se if se > 0 else float("inf")
    p_two = float(2 * stats.norm.sf(abs(z))) if math.isfinite(z) else 0.0
    return {"kappa_weighted": float(kappa), "scheme": scheme,
            "weighted_p_o": p_o, "weighted_p_e": p_e,
            "SE_bootstrap": se, "z": float(z), "p_value": p_two,
            "CI95_lower_bootstrap": float(ci_lo),
            "CI95_upper_bootstrap": float(ci_hi),
            "n_boot": n_boot, "K": K, "n": n,
            "method": f"weighted kappa ({scheme}, bootstrap SE / percentile CI)"}


def library_versions(rater1, rater2, categories):
    from sklearn.metrics import cohen_kappa_score
    return {
        "sklearn linear":    float(cohen_kappa_score(rater1, rater2, labels=categories, weights="linear")),
        "sklearn quadratic": float(cohen_kappa_score(rater1, rater2, labels=categories, weights="quadratic")),
    }


if __name__ == "__main__":
    import random
    random.seed(4)
    cats = ["mild", "moderate", "severe", "critical"]
    n = 200
    rater1 = [random.choice(cats) for _ in range(n)]
    # rater2 disagrees but usually by only 1 step
    idx = {c: i for i, c in enumerate(cats)}
    def perturb(c):
        j = idx[c]
        # 60% stay, 30% off-by-one, 10% off-by-two
        r = random.random()
        if r < 0.60: return c
        if r < 0.90:
            shift = random.choice([-1, 1]); j2 = max(0, min(len(cats) - 1, j + shift))
        else:
            shift = random.choice([-2, 2]); j2 = max(0, min(len(cats) - 1, j + shift))
        return cats[j2]
    rater2 = [perturb(r) for r in rater1]

    cm = confusion_matrix(rater1, rater2, cats)
    print("=== Confusion (ordered categories) ===")
    print(cm)
    for scheme in ("linear", "quadratic"):
        print(f"\n=== Weighted kappa ({scheme}) ===")
        out = weighted_kappa(cm, scheme)
        for k, v in out.items():
            print(f"  {k:22s}: {v}")
    print("\n--- library ---")
    for k, v in library_versions(rater1, rater2, cats).items():
        print(f"  {k:20s}: {v}")
