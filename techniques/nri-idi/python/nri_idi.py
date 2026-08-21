"""Net Reclassification Improvement (NRI) and Integrated Discrimination
Improvement (IDI) — Pencina et al. 2008 (Reference §20.x extra).

Compare an OLD risk model to a NEW risk model in a binary-outcome cohort.

Category-based NRI: given K risk categories,
    NRI = P(up | event) - P(down | event) + P(down | non-event) - P(up | non-event)
where "up" means the NEW model puts a subject in a HIGHER category than OLD.

Continuous NRI: same idea using raw probabilities (up if p_new > p_old).

IDI: mean improvement in the predicted probability of event for events,
     minus the mean improvement for non-events:
    IDI = (mean_new_e - mean_old_e) - (mean_new_ne - mean_old_ne)
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

import numpy as np    # numerical arrays + linear algebra


def continuous_nri(y, p_old, p_new) -> dict:
    y = np.asarray(y, dtype=int); p_old = np.asarray(p_old, dtype=float); p_new = np.asarray(p_new, dtype=float)
    events = y == 1; ne = y == 0
    up_e = float(((p_new > p_old) & events).sum() / events.sum())
    down_e = float(((p_new < p_old) & events).sum() / events.sum())
    up_ne = float(((p_new > p_old) & ne).sum() / ne.sum())
    down_ne = float(((p_new < p_old) & ne).sum() / ne.sum())
    return {"NRI_events": up_e - down_e,
            "NRI_nonevents": down_ne - up_ne,
            "NRI_total": (up_e - down_e) + (down_ne - up_ne),
            "method": "continuous NRI"}


def category_nri(y, p_old, p_new, cutoffs) -> dict:
    """cutoffs: list of thresholds (strictly increasing), K = len + 1 categories."""
    y = np.asarray(y, dtype=int); p_old = np.asarray(p_old, dtype=float); p_new = np.asarray(p_new, dtype=float)
    edges = np.concatenate([[-np.inf], np.asarray(cutoffs), [np.inf]])
    cat_old = np.digitize(p_old, edges) - 1
    cat_new = np.digitize(p_new, edges) - 1
    events = y == 1; ne = y == 0
    up_e = float(((cat_new > cat_old) & events).sum() / events.sum())
    down_e = float(((cat_new < cat_old) & events).sum() / events.sum())
    up_ne = float(((cat_new > cat_old) & ne).sum() / ne.sum())
    down_ne = float(((cat_new < cat_old) & ne).sum() / ne.sum())
    return {"cutoffs": list(cutoffs),
            "NRI_events": up_e - down_e,
            "NRI_nonevents": down_ne - up_ne,
            "NRI_total": (up_e - down_e) + (down_ne - up_ne),
            "method": "categorical NRI"}


def idi(y, p_old, p_new) -> dict:
    y = np.asarray(y, dtype=int); p_old = np.asarray(p_old, dtype=float); p_new = np.asarray(p_new, dtype=float)
    m_new_e = p_new[y == 1].mean(); m_old_e = p_old[y == 1].mean()
    m_new_ne = p_new[y == 0].mean(); m_old_ne = p_old[y == 0].mean()
    return {"IDI_events": m_new_e - m_old_e,
            "IDI_nonevents": m_new_ne - m_old_ne,
            "IDI_total": (m_new_e - m_old_e) - (m_new_ne - m_old_ne),
            "method": "Integrated Discrimination Improvement"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    # baseline model: predicts by x1 only.  New model adds x2 (a true predictor).
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    eta = -0.5 + 0.6 * x1 + 0.9 * x2
    p_true = 1 / (1 + np.exp(-eta))
    y = (rng.uniform(size=n) < p_true).astype(int)

    from sklearn.linear_model import LogisticRegression
    m_old = LogisticRegression().fit(x1.reshape(-1, 1), y)
    m_new = LogisticRegression().fit(np.column_stack([x1, x2]), y)
    p_old = m_old.predict_proba(x1.reshape(-1, 1))[:, 1]
    p_new = m_new.predict_proba(np.column_stack([x1, x2]))[:, 1]

    print("=== NRI / IDI: old (x1 only) vs new (x1 + x2) risk model, n=2000 ===")
    c = continuous_nri(y, p_old, p_new)
    print(f"\n  continuous NRI: events {c['NRI_events']:+.3f}   "
          f"nonevents {c['NRI_nonevents']:+.3f}   total {c['NRI_total']:+.3f}")

    cat = category_nri(y, p_old, p_new, cutoffs=[0.10, 0.30, 0.60])
    print(f"\n  categorical NRI at cutoffs {cat['cutoffs']}: "
          f"events {cat['NRI_events']:+.3f}   "
          f"nonevents {cat['NRI_nonevents']:+.3f}   total {cat['NRI_total']:+.3f}")

    d = idi(y, p_old, p_new)
    print(f"\n  IDI: events {d['IDI_events']:+.3f}   nonevents {d['IDI_nonevents']:+.3f}   "
          f"total {d['IDI_total']:+.3f}")

    # ROC-AUC comparison
    from sklearn.metrics import roc_auc_score
    print(f"\n  AUC old = {roc_auc_score(y, p_old):.3f}   "
          f"AUC new = {roc_auc_score(y, p_new):.3f}   "
          f"delta_AUC = {roc_auc_score(y, p_new) - roc_auc_score(y, p_old):+.3f}")

    print("\n--- library cross-check (R Hmisc::improveProb; R nricens; Python lifelines / manual) ---")
