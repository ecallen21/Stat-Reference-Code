# Hierarchical (Agglomerative) Clustering (Reference §9.8)

Starts with each observation as its own cluster and repeatedly merges the **two closest clusters** until one remains. The full merge history is the *dendrogram* — cut at any height to get a specific number of clusters.

## Linkage methods

How "distance between clusters" is defined given pairwise point distances:

| Linkage | Cluster distance | Tendency |
|---|---|---|
| **Single** | `min` distance between any pair | Chaining; picks up thin bridges |
| **Complete** | `max` distance | Compact, similar-size clusters |
| **Average** | mean pairwise distance | Balance of single/complete |
| **Ward** | merge minimally increasing within-cluster variance | Compact spherical clusters (requires Euclidean) |

Updates use the Lance–Williams recurrence — the merged cluster's distance to every other cluster is a linear combination of the old distances plus the just-merged distance.

## Diagnostic: cophenetic correlation

Correlation between the original pairwise distances and the *cophenetic* distances (the merge height at which each pair first joins the same cluster). Higher ⇒ the tree faithfully represents the data. Cophenetic ~0.9+ on well-separated data is common.

## Files

- `python/hierarchical_clustering.py` — from-scratch agglomerative with single/complete/average/Ward via Lance–Williams; `cut_tree` to get labels at any k; cophenetic correlation; cross-check against `scipy.cluster.hierarchy.linkage`.
- `r/hierarchical_clustering.R` — thin wrappers around base `stats::hclust` (which is authoritative) + optional `cluster::agnes`.

## Assumptions

- No probability model; hierarchical clustering is a deterministic-agglomeration procedure.
- Ward assumes Euclidean; single/complete/average work with any dissimilarity you can compute.
- Choice of k is not part of the fit — either cut visually from the dendrogram or use validation criteria (see `cluster-validation`).

## Run

```
python techniques/hierarchical-clustering/python/hierarchical_clustering.py
Rscript techniques/hierarchical-clustering/r/hierarchical_clustering.R
```

**Refs:** Ward, J.H. "Hierarchical grouping to optimize an objective function." *JASA* 58(301), 236–244, 1963; Lance, G.N. & Williams, W.T. "A general theory of classificatory sorting strategies." *Comp. J.* 9(4), 373–380, 1967; Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (Ch. 14.3).

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
