"""Hardy-Weinberg equilibrium testing (Reference Sec 40.25).

Under HWE with minor-allele freq p (major q = 1 - p):
    P(AA) = q^2,  P(Aa) = 2pq,  P(aa) = p^2

Two workhorse tests:

  CHI-SQUARE (approx, 1 df):
    chi2 = sum_{genotype} (O - E)^2 / E

  EXACT TEST (Wigginton 2005): enumerate heterozygote counts with
    the same allele total; sum probabilities of configurations at
    least as extreme.  Preferred for rare variants and small n.

Used routinely in GWAS QC: SNPs with HWE p < 1e-6 in controls are
usually excluded (likely genotyping errors).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from math import comb
from scipy import stats    # chi2


def hwe_chi2(n_AA, n_Aa, n_aa):
    n = n_AA + n_Aa + n_aa
    p = (2 * n_AA + n_Aa) / (2 * n)                       # freq(A)
    q = 1 - p
    E = np.array([n * p ** 2, n * 2 * p * q, n * q ** 2])
    O = np.array([n_AA, n_Aa, n_aa])
    chi2 = ((O - E) ** 2 / np.where(E > 0, E, 1)).sum()
    p_val = float(stats.chi2.sf(chi2, df=1))
    return {"chi2": float(chi2), "p_value": p_val, "expected": E.tolist()}


def hwe_exact(n_AA, n_Aa, n_aa):
    """Wigginton 2005 exact test.  Two-sided by summing configurations with
    likelihood <= observed."""
    n = n_AA + n_Aa + n_aa
    n_A = 2 * n_AA + n_Aa
    n_a = 2 * n_aa + n_Aa
    # Precompute probabilities across all possible n_Aa
    probs = {}
    for h in range(n_A % 2, min(n_A, n_a) + 1, 2):
        AA = (n_A - h) // 2
        aa = (n_a - h) // 2
        if AA < 0 or aa < 0:
            continue
        log_p = (np.log(2) * h
                 + np.sum(np.log(range(1, n + 1)))
                 - np.sum(np.log(range(1, AA + 1)))
                 - np.sum(np.log(range(1, h + 1)))
                 - np.sum(np.log(range(1, aa + 1)))
                 + np.sum(np.log(range(1, n_A + 1)))
                 + np.sum(np.log(range(1, n_a + 1)))
                 - np.sum(np.log(range(1, 2 * n + 1))))
        probs[h] = np.exp(log_p)
    total = sum(probs.values())
    for h in probs:
        probs[h] /= total
    obs_prob = probs.get(n_Aa, 0.0)
    p = sum(v for v in probs.values() if v <= obs_prob + 1e-15)
    return {"observed_prob": float(obs_prob), "p_value": float(p),
            "n_configs": len(probs)}


if __name__ == "__main__":
    print("=== Hardy-Weinberg equilibrium testing (chi^2 + exact) ===\n")
    # Case A: near-perfect HWE
    print("  Case A (in HWE, n=100, p=0.3):")
    for AA, Aa, aa in [(49, 42, 9), (48, 44, 8)]:
        c = hwe_chi2(AA, Aa, aa); e = hwe_exact(AA, Aa, aa)
        print(f"    genotypes (AA={AA}, Aa={Aa}, aa={aa})")
        print(f"       chi^2 = {c['chi2']:.3f}   chi^2 p = {c['p_value']:.3f}")
        print(f"       exact p = {e['p_value']:.3f}")

    print("\n  Case B (excess heterozygotes -- possible genotyping error):")
    AA, Aa, aa = 30, 60, 10
    c = hwe_chi2(AA, Aa, aa); e = hwe_exact(AA, Aa, aa)
    print(f"    genotypes (AA={AA}, Aa={Aa}, aa={aa})")
    print(f"       chi^2 = {c['chi2']:.3f}   chi^2 p = {c['p_value']:.4f}")
    print(f"       exact p = {e['p_value']:.4f}")

    print("\n--- library cross-check (R HardyWeinberg::HWExact; Python scikit-allel) ---")
