"""Haplotype phasing and estimation (Reference Sec 40.27).

Genotype data is UNPHASED -- for a double-heterozygote at two loci we
cannot tell whether the two mutant alleles sit on the same chromosome
(cis) or opposite chromosomes (trans).  We estimate haplotype
frequencies via EM (Excoffier & Slatkin 1995):

  E-step: for each ambiguous genotype, assign responsibilities to
          consistent haplotype pairs proportional to prior products.
  M-step: haplotype frequencies = weighted counts / (2 * n).

Reference / production tools (SHAPEIT, Beagle) do sequential imputation
on large panels; the EM here demonstrates the principle on 2 SNPs.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def em_haplotype_2snp(genotypes, n_iter=200, tol=1e-8):
    """EM for 2-SNP haplotype frequencies.

    genotypes : list of tuples (g1, g2) with g in {0, 1, 2} minor allele count.
    """
    # Unambiguous / ambiguous pair enumeration
    haps = ["AB", "Ab", "aB", "ab"]
    idx = {h: i for i, h in enumerate(haps)}
    freqs = np.ones(4) / 4                  # uniform start
    # For each individual, list of compatible haplotype pairs
    def compatible(g1, g2):
        # a = minor allele count; enumerate haplotype pairs summing to (g1, g2)
        pairs = []
        for i, hi in enumerate(haps):
            for j, hj in enumerate(haps):
                # allele at locus 1: A=0, a=1; upper case=lower dose 0, lower=1
                s1 = (hi[0].islower()) + (hj[0].islower())
                s2 = (hi[1].islower()) + (hj[1].islower())
                if s1 == g1 and s2 == g2:
                    pairs.append((i, j))
        return pairs

    pair_lists = [compatible(g1, g2) for g1, g2 in genotypes]
    n = len(genotypes)
    for _ in range(n_iter):
        new = np.zeros(4)
        for pairs in pair_lists:
            weights = np.array([freqs[i] * freqs[j] for i, j in pairs])
            weights = weights / weights.sum()
            for w, (i, j) in zip(weights, pairs):
                new[i] += w; new[j] += w
        new = new / new.sum()
        if np.max(np.abs(new - freqs)) < tol:
            freqs = new
            break
        freqs = new
    return dict(zip(haps, freqs.tolist()))


if __name__ == "__main__":
    print("=== Haplotype phasing: EM for 2-SNP haplotype frequencies ===\n")
    rng = np.random.default_rng(0)
    # True haplotype frequencies: majority "AB" and "ab" (perfect LD scenario)
    true_freq = {"AB": 0.4, "Ab": 0.1, "aB": 0.1, "ab": 0.4}
    n = 500
    haps = list(true_freq.keys()); probs = list(true_freq.values())

    def hap_to_genotype(h1, h2):
        return (int(h1[0].islower()) + int(h2[0].islower()),
                int(h1[1].islower()) + int(h2[1].islower()))

    genotypes = []
    for _ in range(n):
        h1 = rng.choice(haps, p=probs); h2 = rng.choice(haps, p=probs)
        genotypes.append(hap_to_genotype(h1, h2))

    est = em_haplotype_2snp(genotypes)
    print(f"  n = {n}")
    print(f"  {'hap':>5s}  {'true':>7s}  {'est':>7s}")
    for h in ("AB", "Ab", "aB", "ab"):
        print(f"  {h:>5s}  {true_freq[h]:>7.3f}  {est[h]:>7.3f}")

    # Recovery of LD (D and r^2) from phased freqs
    p_A = est["AB"] + est["Ab"]; p_B = est["AB"] + est["aB"]
    D = est["AB"] - p_A * p_B
    r2 = D ** 2 / max(p_A * (1 - p_A) * p_B * (1 - p_B), 1e-12)
    print(f"\n  Implied D = {D:+.3f}   r^2 = {r2:.3f}\n")

    print("--- library cross-check (R haplo.stats/gap; Python custom + SHAPEIT/Beagle external) ---")
