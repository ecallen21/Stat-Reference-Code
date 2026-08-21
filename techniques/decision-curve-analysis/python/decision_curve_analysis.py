"""Decision Curve Analysis (Vickers-Elkin 2006; Reference §20.x extra).

For a binary risk model and a threshold probability p_t, define:

    net_benefit(p_t) = TP / n - FP / n * (p_t / (1 - p_t))

for the decision rule "treat if predicted risk >= p_t".  Compare to:
    treat-all:  NB = prevalence - (1 - prevalence) * p_t / (1 - p_t)
    treat-none: NB = 0

A model is clinically useful if its net-benefit curve lies above BOTH the
treat-all and treat-none reference curves across the relevant threshold range.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def net_benefit(y, p, thresholds) -> dict:
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    n = len(y); prev = y.mean()
    NB_model = []
    NB_all = []
    for pt in thresholds:
        yhat = (p >= pt).astype(int)
        TP = int(((yhat == 1) & (y == 1)).sum())
        FP = int(((yhat == 1) & (y == 0)).sum())
        w = pt / (1 - pt) if pt < 1 else np.inf
        NB_model.append(TP / n - FP / n * w)
        NB_all.append(prev - (1 - prev) * w if pt < 1 else -np.inf)
    return {"thresholds": list(thresholds),
            "NB_model": NB_model,
            "NB_treat_all": NB_all,
            "NB_treat_none": [0.0] * len(thresholds),
            "prevalence": float(prev),
            "method": "decision curve analysis (Vickers-Elkin 2006)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    eta = -1.0 + 0.7 * x1 + 0.9 * x2
    p_true = 1 / (1 + np.exp(-eta))
    y = (rng.uniform(size=n) < p_true).astype(int)

    from sklearn.linear_model import LogisticRegression
    m_new = LogisticRegression().fit(np.column_stack([x1, x2]), y)
    m_old = LogisticRegression().fit(x1.reshape(-1, 1), y)
    p_new = m_new.predict_proba(np.column_stack([x1, x2]))[:, 1]
    p_old = m_old.predict_proba(x1.reshape(-1, 1))[:, 1]

    thresholds = np.arange(0.05, 0.65, 0.05)
    dc_new = net_benefit(y, p_new, thresholds)
    dc_old = net_benefit(y, p_old, thresholds)

    print(f"=== Decision curve analysis (n={n}, prevalence = "
          f"{dc_new['prevalence']:.3f}) ===")
    print(f"  {'p_t':>6}  {'NB_new':>8}  {'NB_old':>8}  "
          f"{'NB_all':>8}  best-strategy")
    for k, pt in enumerate(thresholds):
        nb_n = dc_new["NB_model"][k]; nb_o = dc_old["NB_model"][k]
        nb_a = dc_new["NB_treat_all"][k]
        best = max([("new", nb_n), ("old", nb_o), ("all", nb_a), ("none", 0.0)],
                    key=lambda t: t[1])[0]
        print(f"  {pt:>6.2f}  {nb_n:>8.3f}  {nb_o:>8.3f}  {nb_a:>8.3f}  {best}")

    # summary: over what threshold range does 'new' dominate?
    dom = [(t, nn) for t, nn, no, na in zip(thresholds, dc_new["NB_model"],
                                             dc_old["NB_model"],
                                             dc_new["NB_treat_all"])
           if nn > max(no, na, 0.0)]
    if dom:
        print(f"\n  'new' model is best across thresholds "
              f"[{dom[0][0]:.2f}, {dom[-1][0]:.2f}]  ({len(dom)} of {len(thresholds)} grid points)")

    print("\n--- library cross-check (R rmda::decision_curve; dcurves; Python dcurves) ---")
