# Patient Similarity Network (Reference §30.25)

Build a network where **each node is a patient** and edges reflect
**feature similarity**, then apply community detection to discover
**subtypes**. Widely used in precision medicine (Li 2015, Ching 2018)
to stratify heterogeneous cohorts.

## Pipeline

```
1. Standardise the patient × feature matrix.
2. Compute pairwise similarity (Gaussian / cosine / mixed).
3. k-nearest-neighbour graph (sparsify).
4. Community detection (Louvain, label propagation, spectral).
5. Report subtypes + within-subtype feature profiles + outcome
   heterogeneity.
```

## When to use

- **Cohort subtyping** with rich phenotype features — diabetes, MS,
  cancer, sepsis subtypes.
- **Prognostic stratification** — associate subtypes with outcomes.
- **Multi-omics integration** — Similarity Network Fusion (SNF, Wang
  2014) combines networks from different data types.

## When NOT to use

- **Small n** — networks with too few nodes give unstable communities.
- **Continuous outcome regression** — supervised regression / trees
  work better.
- **Feature relevance unknown** — clustering discovers structure but
  may not align with clinical outcomes.

## Files

- `python/patient_similarity_network.py` — Gaussian similarity + k-NN
  sparsification + label propagation (Raghavan-Albert-Kumara 2007).
  Demo: 60-patient cohort with 3 planted subtypes on 6 features:
  **discovered 5 communities; cluster purity 0.967** vs true subtype.
- `r/patient_similarity_network.R` — `SNFtool`, `igraph`, `bootnet`,
  `qgraph` (R); `snfpy`, `python-louvain`, `networkx`,
  `scikit-network` (Python).

## Assumptions & caveats

- **Feature scaling / weighting** — normalise features; consider
  domain-weighted distances (mixed continuous + categorical via
  Gower).
- **k in k-NN** — small `k` = fragmented graph; large `k` = over-
  connected. Rule-of-thumb `k = ⌊√n⌋`.
- **σ in the Gaussian kernel** — set to the median pairwise distance.
- **Community algorithm sensitivity** — Louvain / Leiden are
  resolution-parametric; different resolutions reveal different
  granularities.
- **Stability** — bootstrap resampling of features / patients + Jaccard
  similarity across resamplings; `bootnet::CS`.
- **Downstream validation** — always test whether discovered subtypes
  associate with an independent clinical outcome.

## Related in this repo

- `community-detection`, `stochastic-block-model`,
  `latent-space-network`, `qap-network-regression`,
  `gaussian-graphical-model`, `node2vec-deepwalk` — network family
  (this batch).
- `hierarchical-clustering`, `dbscan`, `k-means` — non-graph
  clustering alternatives.
- `distributionally-robust-optimization` — worst-subgroup risk when
  subtypes matter for predictions.
- `dimensionality-reduction-pca`, `sparse-pca`, `variational-
  autoencoder` — feature-space transforms before similarity.

## Run

```
python techniques/patient-similarity-network/python/patient_similarity_network.py
Rscript techniques/patient-similarity-network/r/patient_similarity_network.R
```

**Refs:** Wang, B. et al. "Similarity network fusion for aggregating data types on a genomic scale." *Nature Methods*, 2014; Li, L. et al. "Identification of type 2 diabetes subgroups through topological analysis of patient similarity." *Science Translational Medicine*, 2015; Ching, T. et al. "Opportunities and obstacles for deep learning in biology and medicine." *JRSS Interface*, 2018.

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
