# Latent-Space Network Model (Reference §30.5)

Hoff, Raftery & Handcock (2002). Each node has a **latent position**
`z_i ∈ ℝ^d`; edge probability shrinks with distance:

```
logit P(A_ij = 1)  =  α  −  ‖z_i − z_j‖.
```

Positions capture **transitivity, homophily, and clustering** without
an explicit block structure — a smooth alternative to SBM.

## Estimation

- **MLE via gradient descent** on the log-likelihood (this demo).
- **MCMC (Bayesian)** with a Gaussian prior on positions (`latentnet`).
- **Two-stage spectral + refit** (Sussman 2012) for scalability.

## When to use

- **Social / friendship networks** where geography or a hidden 2-D
  space seems to organise ties.
- **Visualisation** — plot `ẑ_i` to see clusters and outliers.
- **Continuous alternative to SBM** when block boundaries are fuzzy.

## When NOT to use

- **Very sparse networks** — likelihood is flat; use SBM or spectral.
- **Directed networks with asymmetric ties** — use bilinear-effects
  latent-space (Hoff 2005).
- **Massive n** — MLE is `O(n²)` per gradient step; scale via
  landmark or variational approximations.

## Files

- `python/latent_space_network.py` — from-scratch gradient descent on
  `(Z, α)` for the undirected distance model + Procrustes alignment
  against the truth. Demo: `n = 40`, two hidden clusters at
  `(±2, 0)`, true `α = 3`: **α̂ = 1.66; Procrustes mean position
  error 0.87** (vs cluster separation 4) — geometry recovered.
- `r/latent_space_network.R` — `latentnet` (R reference);
  `graspologic` (Python).

## Assumptions & caveats

- **α-vs-scale identifiability** — `α − ‖z‖` is unchanged under
  translation, rotation, reflection. Report Procrustes-aligned
  positions.
- **d choice** — 2 for visualisation; 3-5 for prediction.
- **Non-convex objective** — random restarts + spectral warm start
  help.
- **Standard errors** — sandwich or MCMC; the gradient-based MLE only
  gives a point estimate.
- **Extensions**: bilinear-effects (Hoff 2005), latent cluster model
  (Handcock 2007), stochastic latent-space with covariates.

## Related in this repo

- `stochastic-block-model` — discrete-block cousin.
- `ergm-exponential-random-graph` — likelihood-based alternative with
  local statistics.
- `graph-embedding-spectral`, `node2vec-deepwalk` — non-generative
  node embeddings (this batch).
- `gaussian-graphical-model` — precision-matrix networks (this batch).
- `procrustes-analysis` — the alignment method used to compare
  embeddings.

## Run

```
python techniques/latent-space-network/python/latent_space_network.py
Rscript techniques/latent-space-network/r/latent_space_network.R
```

**Refs:** Hoff, P.D., Raftery, A.E. & Handcock, M.S. "Latent space approaches to social network analysis." *JASA*, 2002; Handcock, M.S., Raftery, A.E. & Tantrum, J.M. "Model-based clustering for social networks." *JRSS-A*, 2007.

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
