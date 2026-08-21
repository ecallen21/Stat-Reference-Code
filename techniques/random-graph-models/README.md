# Random Graph Models (Reference §24.4)

Generative models used as **nulls** for network hypothesis tests and as
**building blocks** for studying how structural features (degree, clustering,
paths) emerge from simple rules.

| Model | Rule | Signature feature |
|---|---|---|
| **Erdős-Rényi** `G(n, p)` | each edge independent Bernoulli(p) | Poisson degree, low clustering, small diameter |
| **Watts-Strogatz** `(n, k, β)` | ring lattice, rewire each edge w.p. `β` | high clustering + short paths (small-world) at intermediate `β` |
| **Barabási-Albert** `(n, m)` | grow by attaching each new node to `m` existing nodes with `P ∝ deg` | power-law degree `P(k) ∝ k⁻³` (scale-free) |

## Additional generators

- **G(n, m)** — fixed edge count.
- **Chung-Lu / expected-degree** — expected degree sequence.
- **Configuration model** — exact degree sequence with random pairing.
- **Stochastic block model** — see `stochastic-block-model` for a generative community model.
- **ERGM** — see `ergm-exponential-random-graph` for a maximum-entropy model with configurable sufficient statistics.

## When to use

- **Null baseline** for observed statistics: is the observed clustering, degree assortativity, or motif count higher than we'd expect under ER / configuration?
- **Simulation study** to check estimator behaviour on graphs of a known type.
- **Teaching** — the three models cleanly illustrate why *some* structures (degree, clustering, paths) can be produced by very simple rules.

## Files

- `python/random_graph_models.py` — from-scratch ER / WS / BA generators. Demo (n=200, target mean degree ~8): ER mean deg 8.3, max 16, clustering 0.05; WS mean 8.0, clustering 0.57 (small-world); BA mean 7.9, max 39, 6.5% of nodes with deg > 20 (scale-free tail). NetworkX cross-check agrees within noise.
- `r/random_graph_models.R` — `igraph::sample_gnp / sample_smallworld / sample_pa / sample_fitness / sample_degseq`.

## Assumptions & caveats

- **ER is unrealistic** for most real-world networks — it misses clustering and heavy tails; useful only as a null.
- **WS "small-world"** appears in a narrow `β` window; `β → 0` is a lattice, `β → 1` is ER.
- **BA power-law exponent is fixed at 3**; fitness models or edge copying give tunable exponents.
- **Preferential attachment ≠ scale-free** — many other mechanisms produce power laws.
- **Configuration model** can produce multi-edges and self-loops; filter or use MCMC edge-swapping if you need simple graphs.

## Run

```
python techniques/random-graph-models/python/random_graph_models.py
Rscript techniques/random-graph-models/r/random_graph_models.R
```

**Refs:** Erdős, P. & Rényi, A. "On random graphs I." *Publ. Math. Debrecen* 6, 290–297, 1959; Watts, D.J. & Strogatz, S.H. "Collective dynamics of small-world networks." *Nature* 393, 440–442, 1998; Barabási, A.-L. & Albert, R. "Emergence of scaling in random networks." *Science* 286, 509–512, 1999.

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
