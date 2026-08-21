# Stochastic Block Model — SBM (Reference §24.6)

Generative model for graphs with community / role structure:

```
z_i ~ Categorical(π)                    (block label of node i)
A_ij | z_i, z_j ~ Bernoulli(B[z_i, z_j])   (edge probability by block pair)
```

Special cases:

- **Assortative** — `B_kk > B_kl` (diagonal-heavy): communities.
- **Disassortative** — `B_kl > B_kk` off-diagonal-heavy: bipartite / core-periphery.
- **Degree-corrected SBM** — adds per-node degree parameters θ_i so that hubs don't hijack the block assignment.

## Fitting

Full likelihood involves summing over all label configurations; three practical routes:

1. **Hard EM / classification EM** (implemented here): given labels, MLE of B is empirical density within each block pair; given B, reassign each node to the block maximising its log-likelihood; iterate.
2. **Variational EM** (`blockmodels::BM_bernoulli`) — soft assignments, more principled inference on π and B.
3. **MCMC + MDL** (`graph_tool.minimize_blockmodel_dl`) — Bayesian, selects K automatically via Minimum Description Length; handles nested / hierarchical SBM.

## When to use

- **Generative community detection** with proper block probabilities (not just a partition).
- **Bipartite / directed / disassortative structure** — modularity misses these; SBM detects them.
- **Model selection** — likelihood-based comparison of K.
- **Missing-edge / link-prediction** with a plug-in block posterior.

## Files

- `python/stochastic_block_model.py` — from-scratch simulator + hard-EM fitter with spectral warm start. Demo (n=45, K=3, planted diagonal 0.7–0.9, off-diagonal 0.05): converges in 1 iteration, 100% block-recovery accuracy; estimated diagonal 0.71 / 0.69 / 0.92 vs true 0.80 / 0.70 / 0.90, off-diagonal ≈ 0.05.
- `r/stochastic_block_model.R` — `blockmodels::BM_bernoulli`, `sbm::estimateSimpleSBM`, `greed::greed`.

## Contrast with modularity

- **Modularity** — heuristic score for an *assortative* partition. Cannot detect disassortative structure. See `community-detection`.
- **SBM** — full generative model; detects both assortative and disassortative blocks; enables likelihood-based K selection and Bayesian uncertainty.
- **Degree-corrected SBM** (Karrer-Newman 2011) — the go-to for real assortative networks with heavy-tailed degrees.

## Assumptions & caveats

- **K unknown** in practice — use MDL / ICL / cross-validation; `greed`/`graph_tool` automate this.
- **Hard EM has local optima** — restart from multiple random / spectral seeds.
- **Label switching** — cluster identity is arbitrary; align via majority vote before reporting recovery accuracy.
- **Heavy-tailed degree** distorts vanilla SBM — use the degree-corrected variant.
- **Time complexity** of the reassignment step: `O(n · K · avg_degree)` per pass.

## Run

```
python techniques/stochastic-block-model/python/stochastic_block_model.py
Rscript techniques/stochastic-block-model/r/stochastic_block_model.R
```

**Refs:** Holland, P.W., Laskey, K.B. & Leinhardt, S. "Stochastic blockmodels: first steps." *Social Networks* 5(2), 109–137, 1983; Karrer, B. & Newman, M.E.J. "Stochastic blockmodels and community structure in networks." *Phys. Rev. E* 83, 016107, 2011; Peixoto, T.P. "Bayesian stochastic blockmodeling." In *Advances in Network Clustering*, Wiley, 2019.

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
