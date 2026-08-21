"""ROC curve + AUC + DeLong CI (Reference §21.5).

Binary classifier / diagnostic test evaluation.

ROC curve
    For every possible threshold on a continuous score s(x), plot
        TPR (sensitivity) vs FPR (1 - specificity)

AUC (Area Under the Curve)
    Equal to Pr(s(X_+ ) > s(X_-)) for random pairs from positive and
    negative classes (a.k.a. Mann-Whitney U statistic).
    AUC = 0.5 -> random; AUC = 1 -> perfect.

Youden index J = TPR - FPR      maximizing gives an "optimal" threshold
Partial AUC over a specificity range (e.g. sp >= 0.9) - clinical relevance
when only high-specificity operating points matter.

DeLong et al. (1988) SE via Mann-Whitney U variance:
    SE(AUC) analytical from the joint of scores in each class.

Comparison of two AUCs (dependent samples, same subjects): DeLong z-test.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def roc_curve(y_true, scores) -> dict:
    """ROC curve as arrays of thresholds, FPR, TPR."""
    y = np.asarray(y_true, dtype=int); s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    y_ordered = y[order]; s_ordered = s[order]
    P = int(y.sum()); N = len(y) - P
    tp = np.cumsum(y_ordered); fp = np.cumsum(1 - y_ordered)
    tpr = tp / max(P, 1); fpr = fp / max(N, 1)
    # Include (0,0) and (1,1) endpoints
    tpr = np.concatenate([[0], tpr, [1]])
    fpr = np.concatenate([[0], fpr, [1]])
    return {"thresholds": s_ordered, "fpr": fpr, "tpr": tpr,
            "P": int(P), "N": int(N)}


def auc(y_true, scores) -> dict:
    """AUC via Mann-Whitney U (equivalent to trapezoidal integration under ROC)."""
    y = np.asarray(y_true, dtype=int); s = np.asarray(scores, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    # Mann-Whitney U / (n_pos * n_neg) = AUC
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    U_pos = np.sum(r[:len(pos)]) - len(pos) * (len(pos) + 1) / 2
    a = U_pos / (len(pos) * len(neg))
    # DeLong-style SE (Hanley-McNeil approximation)
    Q1 = a / (2 - a); Q2 = 2 * a ** 2 / (1 + a)
    se = math.sqrt((a * (1 - a) + (len(pos) - 1) * (Q1 - a ** 2) +
                    (len(neg) - 1) * (Q2 - a ** 2)) / (len(pos) * len(neg)))
    return {"AUC": float(a), "SE_HanleyMcNeil": float(se),
            "ci_95": (float(a - 1.96 * se), float(a + 1.96 * se)),
            "n_positive": int(len(pos)), "n_negative": int(len(neg))}


def youden_index(y_true, scores) -> dict:
    """Threshold maximizing J = TPR - FPR."""
    r = roc_curve(y_true, scores)
    J = r["tpr"] - r["fpr"]
    best = int(np.argmax(J))
    return {"best_J": float(J[best]),
            "tpr_at_best": float(r["tpr"][best]),
            "fpr_at_best": float(r["fpr"][best]),
            "threshold_at_best": float(r["thresholds"][max(best - 1, 0)])
                                if best - 1 < len(r["thresholds"]) else float("nan")}


def partial_auc(y_true, scores, min_sp: float = 0.8) -> float:
    """Partial AUC in the specificity >= min_sp region (mapped to [0, 1] scale)."""
    r = roc_curve(y_true, scores)
    fpr, tpr = r["fpr"], r["tpr"]
    max_fpr = 1 - min_sp
    idx = fpr <= max_fpr
    if idx.sum() < 2: return float("nan")
    p_auc = float(np.trapezoid(tpr[idx], fpr[idx]))
    # Normalize: (partial - min possible) / (max possible - min possible)
    max_possible = max_fpr
    min_possible = 0.5 * max_fpr ** 2
    return (p_auc - min_possible) / (max_possible - min_possible)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_pos, n_neg = 100, 200
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(int)
    scores = np.concatenate([rng.normal(1, 1, n_pos), rng.normal(0, 1, n_neg)])

    r = auc(y, scores)
    print("=== AUC ===")
    print(f"  AUC = {r['AUC']:.4f}   SE (Hanley-McNeil) = {r['SE_HanleyMcNeil']:.4f}")
    print(f"  95% CI = ({r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f})")
    print(f"  n_positive = {r['n_positive']}, n_negative = {r['n_negative']}")

    y_ = youden_index(y, scores)
    print(f"\n  Best Youden J = {y_['best_J']:.4f}   TPR = {y_['tpr_at_best']:.3f}   FPR = {y_['fpr_at_best']:.3f}")

    pa = partial_auc(y, scores, min_sp=0.8)
    print(f"\n  Partial AUC (specificity >= 0.8), normalized = {pa:.4f}")

    print("\n--- library cross-check (sklearn roc_auc_score) ---")
    try:
        from sklearn.metrics import roc_auc_score
        print(f"  sklearn AUC = {roc_auc_score(y, scores):.4f}")
    except Exception as ex:
        print(f"  (sklearn unavailable: {ex})")
