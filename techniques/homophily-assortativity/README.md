# Homophily + Assortativity (Reference §30.17)

Newman (2003). Measures whether edges preferentially link **like with
like** (homophily / assortative mixing) or **like with unlike**
(heterophily / disassortative mixing).

## Discrete attribute assortativity

```
e_ij   = fraction of edges between group i and group j
a_i    = Σ_j e_ij
r_attr = (Σ_i e_ii − Σ_i a_i²) / (1 − Σ_i a_i²)   ∈ [−1, 1].
```

`r > 0` → homophily; `r ≈ 0` → random mixing; `r < 0` → heterophily.

## Degree assortativity (Newman 2002)

Correlation between the degrees of the two nodes at either end of a
random edge. `r_deg > 0` → high-degree nodes connect to high-degree
nodes.

## When to use

- **Social networks** — race / gender / language homophily.
- **Biological networks** — degree-assortativity patterns are
  characteristic of many biological systems (usually disassortative).
- **Auditing embeddings / recommendations** for demographic filter
  bubbles.

## When NOT to use

- **Weighted / directed networks** — extensions exist but the plain
  formula is undirected.
- **Very small networks** — noise dominates; bootstrap CIs.

## Files

- `python/homophily_assortativity.py` — from-scratch mixing matrix
  (symmetric-normalised) + attribute assortativity + degree
  assortativity via edge-endpoint correlation. Demo three
  configurations on 3-group graph:
  - **HOMOPHILY** (`p_within=0.5, p_cross=0.05`): `r_attr = 0.760`.
  - **NEUTRAL** (`p_within = p_cross = 0.15`): `r_attr = −0.046`.
  - **HETEROPHILY** (`p_within=0.05, p_cross=0.5`): `r_attr = −0.428`.
- `r/homophily_assortativity.R` — `igraph::assortativity_nominal /
  _degree` (R); `networkx.algorithms.assortativity`,
  `graph-tool.correlations` (Python).

## Assumptions & caveats

- **Symmetric mixing matrix** — undirected networks only; directed
  version tracks out- vs in-degrees separately.
- **Node-attribute types** — categorical (nominal), ordinal, or
  numeric variants exist.
- **Normalisation** — dividing by `(1 − Σ a_i²)` re-scales to `[-1,
  1]`.
- **Confidence intervals** — bootstrap over edge resampling.

## Related in this repo

- `stochastic-block-model`, `ergm-exponential-random-graph` —
  generative models capturing mixing patterns.
- `community-detection`, `graph-descriptives` — related descriptive
  measures.
- `small-world-scale-free` — sibling network summary statistics.

## Run

```
python techniques/homophily-assortativity/python/homophily_assortativity.py
Rscript techniques/homophily-assortativity/r/homophily_assortativity.R
```

**Refs:** Newman, M.E.J. "Mixing patterns in networks." *Phys Rev E*, 2003; Newman, M.E.J. "Assortative mixing in networks." *Phys Rev Lett*, 2002.

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
