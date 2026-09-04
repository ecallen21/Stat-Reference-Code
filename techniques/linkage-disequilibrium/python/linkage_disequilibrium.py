"""Linkage disequilibrium (Reference Sec 40.26).

Non-random association of alleles at different loci.  Three common
measures for two bi-allelic SNPs A/a and B/b with haplotype
frequencies p_AB, p_Ab, p_aB, p_ab:

  D    = p_AB - p_A * p_B                    (raw; scale-dependent)
  D'   = D / D_max                           (normalised; |D'|=1 = complete LD)
  r^2  = D^2 / (p_A * p_a * p_B * p_b)       (correlation squared; used
                                              in GWAS LD-pruning and LDSC)

D_max = min(p_A * p_b, p_a * p_B) if D > 0
        min(p_A * p_B, p_a * p_b) if D < 0
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ld_measures(hap_counts):
    """hap_counts : dict with keys 'AB','Ab','aB','ab'."""
    total = sum(hap_counts.values())
    f = {k: v / total for k, v in hap_counts.items()}
    p_A = f["AB"] + f["Ab"]; p_a = 1 - p_A
    p_B = f["AB"] + f["aB"]; p_b = 1 - p_B
    D = f["AB"] - p_A * p_B
    if D > 0:
        D_max = min(p_A * p_b, p_a * p_B)
    else:
        D_max = min(p_A * p_B, p_a * p_b)
    D_prime = D / D_max if D_max > 0 else 0.0
    r2 = D ** 2 / max(p_A * p_a * p_B * p_b, 1e-12)
    return {"p_A": p_A, "p_B": p_B, "D": D, "D_prime": D_prime, "r2": r2}


def ld_from_dosage(G):
    """Pairwise r^2 matrix from a (n_samples, m_snps) genotype-dosage matrix (0/1/2)."""
    G = G - G.mean(axis=0)
    r = np.corrcoef(G, rowvar=False)
    return r ** 2


if __name__ == "__main__":
    print("=== Linkage disequilibrium: D, D', r^2 ===\n")
    # Perfect LD: only two haplotypes
    print("  Case A (perfect LD, only AB and ab):")
    print(f"    {ld_measures({'AB': 60, 'Ab': 0, 'aB': 0, 'ab': 40})}")

    print("\n  Case B (linkage equilibrium):")
    print(f"    {ld_measures({'AB': 24, 'Ab': 36, 'aB': 16, 'ab': 24})}")

    print("\n  Case C (moderate LD):")
    print(f"    {ld_measures({'AB': 40, 'Ab': 10, 'aB': 10, 'ab': 40})}")

    print("\n  LD decay along a synthetic 20-SNP block:")
    rng = np.random.default_rng(0)
    n = 300; m = 20
    # Ancestral haplotypes with recombination between adjacent SNPs
    Hap = rng.integers(0, 2, (n, m))
    for j in range(1, m):
        recomb = rng.random(n) < 0.1
        Hap[recomb, j] = rng.integers(0, 2, size=int(recomb.sum()))
    G = Hap + rng.integers(0, 2, Hap.shape)   # add second chromosome
    R2 = ld_from_dosage(G)
    print(f"    r^2 vs. distance from SNP 0: {[f'{d}: {R2[0, d]:.2f}' for d in range(6)]}")

    print("\n--- library cross-check (R genetics/LDheatmap/snpStats; Python scikit-allel) ---")
