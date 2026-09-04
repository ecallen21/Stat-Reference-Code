"""Multiple metrics + FDR (Reference Sec 44.5).

A single A/B test typically tracks dozens of metrics: primary,
secondary, guardrail.  Family-wise error correction:

  BONFERRONI     : alpha / m  -- FWER control, very conservative.
  BH (Benjamini-Hochberg): FDR control, less conservative,
                            widely used for large metric families.
  HIERARCHICAL   : primary at alpha_1; if primary passes, secondary
                   at alpha_2; etc.  Kohavi-Tang-Xu Ch 17.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def bh_fdr(p, alpha=0.05):
    p = np.asarray(p, dtype=float); m = len(p)
    order = np.argsort(p); p_s = p[order]
    thr = alpha * np.arange(1, m + 1) / m
    passed = p_s <= thr
    if not passed.any():
        return np.zeros(m, dtype=bool)
    k = int(np.max(np.where(passed)[0]))
    rej_s = np.zeros(m, dtype=bool); rej_s[:k + 1] = True
    rej = np.zeros(m, dtype=bool); rej[order] = rej_s
    return rej


def bonferroni(p, alpha=0.05):
    return np.asarray(p) < alpha / len(p)


def hierarchical(p_primary, p_secondary_list, alpha_primary=0.05, alpha_secondary=0.05):
    """If primary significant, test secondaries at alpha_secondary each."""
    if p_primary >= alpha_primary:
        return {"primary_rej": False, "secondary_rejs": [False] * len(p_secondary_list)}
    return {"primary_rej": True,
            "secondary_rejs": [p < alpha_secondary for p in p_secondary_list]}


if __name__ == "__main__":
    print("=== Multiple metrics: FDR vs Bonferroni vs hierarchical ===\n")
    rng = np.random.default_rng(0)
    # 20 metrics; 5 are truly non-null (effect), 15 null
    m = 20; n_true = 5
    p_null = rng.uniform(0, 1, m - n_true)
    p_true = rng.beta(0.5, 5, n_true)              # skewed toward small p
    p = np.concatenate([p_true, p_null])
    truth = np.array([True] * n_true + [False] * (m - n_true))

    rej_uncorr = p < 0.05
    rej_bonf = bonferroni(p, 0.05)
    rej_bh = bh_fdr(p, 0.05)
    def _summary(rej, name):
        tp = int((rej & truth).sum()); fp = int((rej & ~truth).sum())
        print(f"  {name:>14s}   rejected {int(rej.sum())}   TP = {tp}   FP = {fp}")

    _summary(rej_uncorr, "uncorrected")
    _summary(rej_bonf, "Bonferroni")
    _summary(rej_bh, "BH-FDR")

    # Hierarchical: primary metric p=0.03, 4 secondaries
    r = hierarchical(0.03, [0.02, 0.10, 0.04, 0.06])
    print(f"\n  Hierarchical (primary + 4 secondaries): {r}\n")
    print("--- library cross-check (R stats::p.adjust, qvalue; Python statsmodels.stats.multitest) ---")
