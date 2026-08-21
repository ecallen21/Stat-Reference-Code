# Exponential Random Graph Model — ERGM (Reference §24.5)

Exponential family over graphs:

```
P(G) = exp(θᵀ s(G)) / Z(θ)
```

`s(G)` is a vector of user-chosen network statistics (edges, triangles, k-stars, GWESP, node covariates, homophily terms). The partition function `Z(θ)` sums over all `2^C(n,2)` graphs and is intractable — full MLE requires **MCMC-MLE** (Snijders 2002; Hunter & Handcock 2006).

## Pseudo-likelihood

Approximate by treating each dyad conditional on the rest as an independent logistic regression:

```
logit P(A_ij = 1 | A_{-ij}) = θᵀ · Δ_ij s(G)
```

where `Δ_ij s(G)` is the **change in the statistic** when the (i, j) edge toggles (e.g. `Δ_ij edges = 1`, `Δ_ij triangles = #{common neighbours of i and j}`).

Fit by IRLS on the resulting logistic regression. Pseudo-likelihood is fast but **biased** in general — use it as a warm start for MCMC-MLE (as `statnet::ergm` does).

## When to use

- **Test whether a network statistic is over- or under-represented** relative to a chosen null (edges baseline + optional homophily / degree covariates).
- **Compare specifications** — nested models with LR / AIC / BIC.
- **Simulate networks** with prescribed statistical properties from the fit.

## Files

- `python/ergm_exponential_random_graph.py` — from-scratch pseudo-likelihood for [edges + triangles] via IRLS + step halving. Demo (ER graph n=30, p=0.15): θ_edges ≈ −2.06 vs true logit(0.15) = −1.73, θ_triangles ≈ +0.39. Two-15-clique graph: θ_edges ≈ −5.04, θ_triangles ≈ +0.99 (strong triangle preference).
- `r/ergm_exponential_random_graph.R` — `ergm::ergm(net ~ edges + triangles + ...)`, `ergm::simulate`, `ergm::gof`.

## Common specifications

| Term | Meaning |
|---|---|
| `edges` | intercept — controls density |
| `triangle` | transitivity — degenerate on its own; use with care |
| `gwesp(α, fixed=TRUE)` | curved-ERGM alternative to raw triangles — much more stable |
| `kstar(2)` | 2-star count — measures degree heterogeneity |
| `nodefactor("attr")` | main effect for a categorical node attribute |
| `nodematch("attr")` | homophily — extra edges within an attribute group |
| `absdiff("attr")` | edge weight from a continuous covariate |

## Assumptions & caveats

- **Model degeneracy** — raw triangle terms often place all mass on the empty or complete graph. Prefer `gwesp`, `gwd`, `gwdsp` "geometrically weighted" alternatives.
- **Pseudo-likelihood bias** — variance is understated, and the point estimate is biased when the network is far from independence; report both PL and MCMC-MLE when possible.
- **MCMC convergence** — check trace plots, `mcmc.diagnostics(fit)`, and simulate from the fit to compare simulated vs observed statistics.
- **Sample size = 1 network** — inference is about a hypothetical population of similar networks, not conditional on rows of a table.
- **Directed vs undirected** — different terms and change statistics apply.

## Run

```
python techniques/ergm-exponential-random-graph/python/ergm_exponential_random_graph.py
Rscript techniques/ergm-exponential-random-graph/r/ergm_exponential_random_graph.R
```

**Refs:** Frank, O. & Strauss, D. "Markov graphs." *JASA* 81(395), 832–842, 1986; Snijders, T.A.B. "Markov chain Monte Carlo estimation of exponential random graph models." *J. Soc. Struct.* 3(2), 2002; Hunter, D.R. et al. "ergm: A package to fit, simulate and diagnose exponential-family models for networks." *J. Stat. Softw.* 24(3), 2008.

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
