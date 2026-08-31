# Stochastic Block Model (Reference §30.4)

**Nodes belong to one of K latent blocks; edge probability depends
only on the block pair.** Nowicki & Snijders (2001).

## Model

```
P(A_ij = 1 | z_i, z_j)  =  B[z_i, z_j].
```

- **Assortative** structure: `B` is diagonally dominant → clusters.
- **Disassortative** structure: `B` has small diagonal → bipartite-like.
- **Core-periphery** structure: one block connects densely to all
  others.

## Estimation

- **Variational EM** (Daudin-Picard-Robin 2008): E-step for the
  posterior `q(z)` factorises; M-step updates block probabilities `B`
  and priors `α`.
- **Belief propagation** (Decelle 2011) is faster on sparse graphs.
- **Bayesian nested SBM** (Peixoto 2014) infers `K` automatically.

## When to use

- **Community detection** with a generative model — enables model
  comparison and hypothesis testing.
- **Weighted / covariate-adjusted SBM** — modular extensions cover
  edge counts, edge covariates, dynamic networks.
- **Bipartite / directed / degree-corrected** variants exist.

## When NOT to use

- **Heterogeneous degree within blocks** — plain SBM confounds blocks
  with degree. Use degree-corrected SBM (Karrer-Newman 2011).
- **Very sparse graphs** — likelihood becomes flat; regularise or
  Bayesian nested SBM.
- **Small K unknown** — Bayesian nested SBM or hierarchical model
  selection required.

## Files

- `python/stochastic_block_model.py` — from-scratch variational EM
  with **spectral warm start** (k-means on top-K singular vectors of
  the adjacency) to escape symmetric local optima. Demo: n = 60,
  planted 3-block graph, within-block edge prob 0.6, cross-block 0.05.
  **Recovered cluster accuracy = 1.000**; estimated diagonal `B̂ ≈
  0.61`, off-diagonal `≈ 0.05`; block priors `[0.33, 0.33, 0.33]`.
- `r/stochastic_block_model.R` — `sbm`, `blockmodels`, `latentnet`
  (R); `graspologic`, `graph-tool` (Python).

## Assumptions & caveats

- **Symmetric local minima** — plain random init often collapses to
  the all-equal block; spectral / k-means warm start avoids that.
- **K choice** — ICL / BIC / cross-validation; Bayesian nested SBM
  learns K.
- **Edge independence** given z — violated in real networks with
  degree heterogeneity → degree-corrected SBM.
- **Directed / bipartite / weighted** extensions require different
  likelihoods.
- **Identifiability** — labels are only recovered up to permutation.

## Related in this repo

- `community-detection` — classical modularity-based alternative.
- `ergm-exponential-random-graph` — likelihood-based alternative with
  local statistics.
- `latent-space-network`, `gaussian-graphical-model`, `qap-network-
  regression`, `node2vec-deepwalk`, `patient-similarity-network` —
  network family (this batch).
- `random-graph-models` — the null-model catalogue.

## Run

```
python techniques/stochastic-block-model/python/stochastic_block_model.py
Rscript techniques/stochastic-block-model/r/stochastic_block_model.R
```

**Refs:** Nowicki, K. & Snijders, T.A.B. "Estimation and prediction for stochastic blockstructures." *JASA*, 2001; Daudin, J.-J., Picard, F. & Robin, S. "A mixture model for random graphs." *Statistics and Computing*, 2008; Peixoto, T.P. "Hierarchical block structures and high-resolution model selection in large networks." *Physical Review X*, 2014.

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
