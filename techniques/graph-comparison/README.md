# Graph Comparison (Reference §24.12)

Quantify how (dis)similar two graphs are. Three broad families.

## Structural / graph-edit distance

**GED** = minimum cost sequence of edge / node insertions / deletions /
substitutions to convert G1 into G2. Exact GED is NP-hard; approximations
via bipartite matching or A* search (`gmatch4py`).

## Spectral distance

```
d_spec(G1, G2) = ‖ sort(λ(L₁)) − sort(λ(L₂)) ‖₂
```

- Cheap and permutation-invariant.
- Insensitive to some structural changes with the same Laplacian spectrum ("iso-spectral" graphs).
- Use the **normalized Laplacian** spectrum for scale invariance.

## Feature-signature distance

Weighted Euclidean over hand-picked global summaries:

```
d_feat = √( Σ_k w_k · (f_k(G1) − f_k(G2))² )
```

Fast and interpretable; loses local structure information.

## DeltaCon (Koutra et al. 2013)

Affinity matrix from a **fast belief-propagation approximation**:

```
S = (I + ε² D − ε A)⁻¹
```

with `ε ≈ 1 / (1 + max_deg)`. Similarity:

```
sim(G1, G2) = 1 / (1 + ρ(S₁, S₂)),
ρ = √( Σ_ij ( √S₁_ij − √S₂_ij )² )      (Matusita rooted sum of squares)
```

Works on graphs with the same node set — sensitive to local rewiring, not just global spectrum.

## Graph kernels

- **Random-walk kernel**, **shortest-path kernel** — inner products for SVMs.
- **Weisfeiler-Lehman kernel** — colour-refinement based, scalable, popular in graph classification.

## When to use

- **Different snapshots of the same graph** (temporal) — DeltaCon, edit distance.
- **Different graphs, no node correspondence** — kernels, spectral distances, portrait divergence.
- **Anomaly detection** — spike in `d(G_t, G_{t−1})` signals structural change.
- **Model checking** — compare simulated (from an SBM / ERGM) to observed.

## Files

- `python/graph_comparison.py` — from-scratch spectral distance, feature-signature distance, DeltaCon similarity. Demo (n=40 base ER graph vs perturbations flipping 5% / 25% / 50% of dyads): all three metrics monotone — spectral 0.27 → 0.89 → 1.33; feature 31 → 265 → 1128; DeltaCon similarity 0.30 → 0.14 → 0.10; self-comparison returns 0 / 1 exactly.
- `r/graph_comparison.R` — `NetworkDistance` (12+ built-in distances), `graphkernels::CalculateWLKernel`, `igraph::graph.isomorphic.vf2`.

## Assumptions & caveats

- **Node correspondence matters** — DeltaCon needs the SAME node set. Kernels / spectral distances handle non-corresponding graphs.
- **Iso-spectral graphs** — different graphs can share Laplacian spectrum; report multiple metrics.
- **Weight sensitivity** — feature-signature distance depends heavily on the weights `w_k`; standardize each feature or use Mahalanobis distance.
- **DeltaCon's ε** — too large collapses affinities to 0; too small collapses to identity; `1 / (1 + max_deg)` is the default.
- **Scale** — DeltaCon inverse is `O(n³)`; use its `DeltaCon-0` sampling variant for large graphs.

## Run

```
python techniques/graph-comparison/python/graph_comparison.py
Rscript techniques/graph-comparison/r/graph_comparison.R
```

**Refs:** Koutra, D., Vogelstein, J.T. & Faloutsos, C. "DeltaCon: A principled massive-graph similarity function." *SDM*, 2013; Wilson, R.C. & Zhu, P. "A study of graph spectra for comparing graphs and trees." *Pattern Recognition* 41, 2833–2841, 2008; Shervashidze, N. et al. "Weisfeiler-Lehman graph kernels." *JMLR* 12, 2539–2561, 2011.

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
