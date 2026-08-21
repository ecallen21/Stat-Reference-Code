# HITS: Authorities and Hubs (Reference §24.x extra)

Kleinberg's (1999) Hyperlink-Induced Topic Search — a directed-graph
prestige algorithm that assigns **two** scores per node:

```
authority a_i  = Σ_{j → i}  h_j        ("who's cited by good hubs?")
hub       h_i  = Σ_{i → j}  a_j        ("who cites good authorities?")
```

Iterating with L² normalisation converges to the **leading eigenvector** of
`AᵀA` (authority) and `AAᵀ` (hub). Equivalently, `a` and `h` are the leading
singular vectors of `A`.

## Contrast with PageRank

| Aspect | PageRank | HITS |
|---|---|---|
| Number of scores | 1 (stationary distribution) | 2 (a, h) |
| Query-dependent? | usually no (global) | usually yes (topic subgraph) |
| Random-walk interpretation | teleporting random walker | alternating "sample citation, sample cited" walker |
| Handles dead-ends via | damping / teleport | not natively — pre-filter |
| Rewards | being pointed at by anyone | pointing to good authorities *and* being pointed to by good hubs |

Both are commonly reported side-by-side.

## When to use

- **Directed prestige** — citations, hyperlinks, "follows" relationships.
- **Query subgraph** — expand a seed of URLs by 1-hop out-neighbours and 1-hop in-neighbours, then run HITS on the subgraph.
- **Two-role summaries** — reviewers vs authors, sources vs sinks in energy flow / trade networks.

## Files

- `python/hits_authority_hub.py` — power iteration with L² normalisation, NetworkX cross-check. Demo (5 pure hubs → 5 pure authorities, plus one weak cross-link each way): converges in 8 iterations; NetworkX agreement to 1e-12; hubs {0-4} and authorities {5-9} cleanly separated in the two rankings.
- `r/hits_authority_hub.R` — `igraph::authority_score`, `igraph::hub_score`.

## Assumptions & caveats

- **Undirected graphs** — HITS collapses to eigenvector centrality; use eigenvector or Katz instead.
- **Convergence** — the ratio `λ_2 / λ_1` of `AᵀA` sets the rate; ties in the top singular value give unstable rankings.
- **Sign ambiguity of eigenvectors** — the algorithm returns `|scores|` by convention (all positive) via non-negative power iteration.
- **Topic drift** — running HITS on the entire web finds generic hubs (Yahoo!, Google) rather than topic-specific ones; Kleinberg's original paper used query-specific subgraphs.
- **SALSA** (Lempel-Moran 2001) is a random-walk variant of HITS that is less susceptible to "topic drift" and tightly-knit-community bias.

## Run

```
python techniques/hits-authority-hub/python/hits_authority_hub.py
Rscript techniques/hits-authority-hub/r/hits_authority_hub.R
```

**Refs:** Kleinberg, J.M. "Authoritative sources in a hyperlinked environment." *J. ACM* 46(5), 604–632, 1999; Lempel, R. & Moran, S. "SALSA: The stochastic approach for link-structure analysis." *ACM Trans. Inf. Sys.* 19(2), 131–160, 2001.

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
