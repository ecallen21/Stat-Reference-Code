"""Gene set enrichment analysis (Reference Sec 40.4, 40.18).

GSEA (Subramanian et al. 2005): given a ranked list of genes (by
signed test statistic or log-fold-change), test whether members of
a predefined GENE SET are unusually clustered near the top or bottom
of the list.

Enrichment score (ES) walks along the ranked list, adding +hit_weight
for genes IN the set and subtracting -miss_weight otherwise; the
maximum absolute cumulative deviation is the ES.  Significance via
permutation of gene labels.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def gsea_enrichment_score(rank, gene_set, weight=1.0):
    """Compute the signed Kolmogorov-Smirnov-style enrichment score.

    rank      : list of gene identifiers, most-up first.
    gene_set  : iterable of gene ids belonging to the set.
    weight    : GSEA's p exponent on |ranking statistic|; default 1.
    """
    n = len(rank)
    in_set = np.array([g in gene_set for g in rank], dtype=bool)
    n_hit = in_set.sum(); n_miss = n - n_hit
    if n_hit == 0 or n_miss == 0:
        return {"ES": 0.0, "leading_edge_at": None}
    hit_inc  = 1.0 / n_hit
    miss_inc = 1.0 / n_miss
    running = np.where(in_set, hit_inc, -miss_inc).cumsum()
    idx = int(np.argmax(np.abs(running)))
    return {"ES": float(running[idx]), "leading_edge_at": idx,
            "curve": running}


def gsea_permute(rank, gene_set, B=1000, seed=0):
    """Permutation null via label permutation on the gene list."""
    rng = np.random.default_rng(seed)
    obs = gsea_enrichment_score(rank, gene_set)["ES"]
    perm = []
    rank_arr = np.array(rank)
    for _ in range(B):
        perm.append(gsea_enrichment_score(rng.permutation(rank_arr).tolist(), gene_set)["ES"])
    perm = np.array(perm)
    p = float((np.abs(perm) >= abs(obs)).mean())
    return {"ES": obs, "p_value": p, "null_mean": float(perm.mean()),
            "null_sd": float(perm.std()), "B": B}


if __name__ == "__main__":
    print("=== GSEA: enrichment score + permutation p ===\n")
    rng = np.random.default_rng(0)
    n_genes = 200
    rank = [f"g{i}" for i in range(n_genes)]
    # Gene set of 20 genes CONCENTRATED near the top of the list
    top_biased_set = set(f"g{i}" for i in [1, 3, 5, 8, 12, 15, 18, 22, 27, 34,
                                            42, 51, 63, 71, 88, 92, 101, 118, 130, 150])
    random_set    = set(f"g{i}" for i in rng.choice(n_genes, size=20, replace=False))

    for name, gs in [("Top-biased 20-gene set", top_biased_set),
                     ("Random 20-gene set",    random_set)]:
        r = gsea_permute(rank, gs, B=500)
        print(f"  {name}: ES = {r['ES']:+.3f}   perm p = {r['p_value']:.3f}"
              f"   (null mean = {r['null_mean']:+.3f} +/- {r['null_sd']:.3f})")

    print("\n--- library cross-check (R fgsea/clusterProfiler; Python gseapy) ---")
