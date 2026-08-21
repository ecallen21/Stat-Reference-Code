# Centrality Measures (Reference §24.2)

Six ways to score node importance in a network. All defined here on an
undirected, unweighted adjacency matrix `A`; extensions to weighted /
directed graphs are noted below.

| Measure | Definition | What it captures |
|---|---|---|
| **Degree** | `deg(v) / (n − 1)` | popularity, local connectivity |
| **Closeness** | `(n − 1) / Σ_u d(v, u)` | overall proximity |
| **Betweenness** | `Σ σ_{st}(v) / σ_{st}` | brokerage / gatekeeping |
| **Eigenvector** | leading eigenvector of `A` | prestige — being connected to high-prestige neighbours |
| **Katz** | `(I − α A^T)^{-1} β 1` (`α < 1 / ρ(A)`) | eigenvector w/ base attention `β` (avoids all-zero on DAGs) |
| **PageRank** | stationary distribution of the damped random walk | eigenvector for directed / weighted graphs with teleportation |

Betweenness uses **Brandes' algorithm** — `O(n · m)` with BFS accumulation.

## When to use

- **Degree** — the null / cheap baseline; often correlates with everything else.
- **Closeness** — reachability in bounded time.
- **Betweenness** — brokers, bridges, cut-vertex candidates.
- **Eigenvector / Katz** — prestige / influence in citation, endorsement graphs.
- **PageRank** — directed graphs with dangling nodes (web, references).
- **HITS** (hubs/authorities) — bipartite prestige on directed graphs.

## Files

- `python/centrality_measures.py` — from-scratch implementations of all six; NetworkX cross-check. Demo (two 5-cliques joined by an edge 4–5): bridge nodes 4 and 5 have highest betweenness 0.556 as expected; degree, closeness, eigenvector, Katz, PageRank all agree with NetworkX to 4 sig figs.
- `r/centrality_measures.R` — `igraph::degree / closeness / betweenness / eigen_centrality / alpha_centrality / page_rank / authority_score / hub_score`.

## Assumptions & caveats

- **Disconnected graphs** — closeness is only meaningful within a component; some conventions use the harmonic centrality `Σ 1 / d(v, u)` instead.
- **Betweenness scales poorly** — `O(n · m)` becomes expensive above a few thousand nodes; use `k`-approximation (`nx.betweenness_centrality(G, k=100)`).
- **Eigenvector needs a strongly connected graph** with a unique leading eigenvalue; Katz and PageRank handle disconnected / periodic graphs via the base term.
- **α in Katz** must satisfy `α < 1 / ρ(A)` for convergence; the code defaults to `0.9 / ρ(A)`.
- **Directed graphs** — the module here treats `A` as symmetric; use in-degree, out-degree, and reverse-graph versions for directed centralities.

## Run

```
python techniques/centrality-measures/python/centrality_measures.py
Rscript techniques/centrality-measures/r/centrality_measures.R
```

**Refs:** Freeman, L.C. "Centrality in social networks: conceptual clarification." *Social Networks* 1(3), 215–239, 1979; Brandes, U. "A faster algorithm for betweenness centrality." *J. Math. Sociol.* 25(2), 163–177, 2001; Page, L. et al. "The PageRank citation ranking." Stanford InfoLab Tech. Report, 1998.

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
