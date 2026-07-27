# Cluster Validation and Selection of k (Reference §9.14)

Clustering doesn't tell you `k` — you have to. This file collects the standard **internal** validation criteria (data + labels, no external labels needed):

| Criterion | Formula (schematic) | Direction | Notes |
|---|---|---|---|
| **Silhouette** | per-point `(b − a) / max(a, b)`; average | Higher | Interpretable per-point; hovers `[−1, 1]` |
| **Calinski–Harabasz** | `(B/(k−1)) / (W/(n−k))` | Higher | Between/within SS ratio, df-scaled |
| **Davies–Bouldin** | mean of "worst-neighbor" ratios | Lower | Rewards separated + compact clusters |
| **Elbow (WCSS knee)** | plot inertia vs k; eyeball the knee | — | Informal; use with the others |
| **Gap statistic** (Tibshirani 2001) | `E[log W_uniform] − log W_data`; 1-SE rule | Higher | Compares to uniform-null; principled k choice |

Silhouette details: `a` = mean distance to same-cluster points; `b` = min mean distance to any *other* cluster. Positive = well-matched to own cluster; near 0 = boundary; negative = wrong cluster.

Use several indices — they disagree by construction on some data. Convergence across indices is the signal.

## Files

- `python/cluster_validation.py` — silhouette, Calinski–Harabasz, Davies–Bouldin, WCSS elbow, and gap statistic with the 1-SE rule. All three internal indices match `sklearn.metrics.*_score` to 12 dp.
- `r/cluster_validation.R` — from-scratch + `cluster::silhouette` + `cluster::clusGap`.

## Assumptions

- No external ground truth needed. If you *do* have ground truth (labeled synthetic data or held-out gold), use ARI/NMI/Fowlkes–Mallows instead — not implemented here.
- Silhouette and DB are Euclidean by default; both extend to any dissimilarity.
- Gap statistic uses uniform reference over the data's bounding box — sensitive to axis-aligned data structure. Consider a PCA-rotated bounding box for elongated clusters.

## Run

```
python techniques/cluster-validation/python/cluster_validation.py
Rscript techniques/cluster-validation/r/cluster_validation.R
```

**Refs:** Rousseeuw, P.J. "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis." *J. Comp. App. Math.* 20, 53–65, 1987; Caliński, T. & Harabasz, J. "A dendrite method for cluster analysis." *Comm. Stat.* 3(1), 1–27, 1974; Davies, D.L. & Bouldin, D.W. "A cluster separation measure." *IEEE TPAMI* 1(2), 224–227, 1979; Tibshirani, R., Walther, G. & Hastie, T. "Estimating the number of clusters in a data set via the gap statistic." *JRSS B* 63(2), 411–423, 2001.

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
