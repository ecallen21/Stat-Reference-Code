"""Equalized odds (Reference Ch 31 Fairness).

Hardt, Price & Srebro (2016) "Equality of Opportunity in Supervised
Learning."

DEFINITION:  P(Y_hat = 1 | Y = y, A = a)  is equal across all groups a
for BOTH y = 0 (equal false-positive rate) and y = 1 (equal true-positive
rate).

Two headline summaries:

  Equalized-odds DIFFERENCE = max( |TPR_a - TPR_b|,  |FPR_a - FPR_b| )
  Equalized-odds RATIO       = min per-group  / max per-group  (of TPR and FPR)

Fairlearn reports the DIFFERENCE by default; higher-is-more-unfair.

Contrast with:
  - DEMOGRAPHIC PARITY -- ignores Y; matches SELECTION RATE only.
  - EQUAL OPPORTUNITY  -- Hardt's WEAKER criterion, requires only TPR match.
  - CALIBRATION PARITY -- P(Y=1 | Y_hat, A) equal.

Impossibility (Chouldechova 2017, Kleinberg 2016): with different group
BASE RATES, equalized odds + calibration parity cannot both hold.

Here we compute per-group TPR / FPR / equalized-odds difference on a
synthetic classifier with base-rate disparity.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def per_group_rates(y_true, y_hat, groups):
    """Return dict { group: {tpr, fpr, tnr, fnr, base_rate, n} }."""
    out = {}
    for a in np.unique(groups):
        m = groups == a
        y_a, p_a = y_true[m], y_hat[m]
        pos = y_a == 1; neg = y_a == 0
        tpr = float((p_a[pos] == 1).mean()) if pos.any() else float("nan")
        fpr = float((p_a[neg] == 1).mean()) if neg.any() else float("nan")
        out[int(a)] = {"tpr": tpr, "fpr": fpr, "base_rate": float(pos.mean()),
                        "n": int(m.sum())}
    return out


def equalized_odds_diff(y_true, y_hat, groups):
    r = per_group_rates(y_true, y_hat, groups)
    tprs = [v["tpr"] for v in r.values()]
    fprs = [v["fpr"] for v in r.values()]
    return max(max(tprs) - min(tprs), max(fprs) - min(fprs))


def equalized_odds_ratio(y_true, y_hat, groups):
    r = per_group_rates(y_true, y_hat, groups)
    tprs = [v["tpr"] for v in r.values()]
    fprs = [v["fpr"] for v in r.values()]
    return min(min(tprs) / max(tprs), min(fprs) / max(fprs))


if __name__ == "__main__":
    print("=== Equalized odds (Hardt 2016) ===\n")
    rng = np.random.default_rng(0)
    n_per = 800
    # Base-rate disparity: group 0 has P(Y=1)=0.50, group 1 has P(Y=1)=0.20.
    y0 = (rng.random(n_per) < 0.50).astype(int)
    y1 = (rng.random(n_per) < 0.20).astype(int)
    # Classifier: score = y + noise; noise LARGER for group 1.
    s0 = y0 + rng.normal(0, 0.5, n_per)
    s1 = y1 + rng.normal(0, 1.0, n_per)

    scores = np.concatenate([s0, s1])
    y = np.concatenate([y0, y1])
    groups = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)

    print("  Per-group rates at THRESHOLD 0.5:\n")
    y_hat = (scores >= 0.5).astype(int)
    r = per_group_rates(y, y_hat, groups)
    print(f"    {'grp':>3}  {'n':>4}  {'base_rate':>9}  {'TPR':>6}  {'FPR':>6}")
    for a, v in r.items():
        print(f"    {a:>3}  {v['n']:>4}  {v['base_rate']:>9.3f}  {v['tpr']:>6.3f}  {v['fpr']:>6.3f}")
    print(f"\n  equalized-odds difference: {equalized_odds_diff(y, y_hat, groups):.3f}"
          f"   equalized-odds ratio: {equalized_odds_ratio(y, y_hat, groups):.3f}\n")

    print("  Sweep decision threshold:\n")
    print(f"    {'thr':>4}  {'TPR_0':>6}  {'TPR_1':>6}  {'FPR_0':>6}  {'FPR_1':>6}  {'EO_diff':>7}")
    for t in (0.2, 0.5, 0.8, 1.1):
        y_hat = (scores >= t).astype(int)
        r = per_group_rates(y, y_hat, groups)
        print(f"    {t:>4.1f}  {r[0]['tpr']:>6.3f}  {r[1]['tpr']:>6.3f}"
              f"  {r[0]['fpr']:>6.3f}  {r[1]['fpr']:>6.3f}"
              f"  {equalized_odds_diff(y, y_hat, groups):>7.3f}")

    print("\n--- library cross-check (fairlearn.metrics.equalized_odds_difference/_ratio) ---")
