# Graph Descriptive Statistics (Reference §24.1)

Basic summaries of an undirected, unweighted graph `G = (V, E)`:

| Quantity | Definition |
|---|---|
| **density** | `|E| / C(|V|, 2)` |
| **degree** | `deg(v) = # edges at v` |
| **local clustering** | `C(v) = 2 · triangles(v) / (deg(v) · (deg(v) − 1))` |
| **transitivity** | `3 · triangles / connected-triples` (global clustering coef) |
| **assortativity** | Pearson r of degrees at edge endpoints |
| **avg path length** | mean over all connected pairs of shortest-path distance |
| **diameter** | max shortest-path distance |
| **components** | equivalence classes under connectedness |

BFS is used for unweighted shortest paths; use Dijkstra for weighted graphs.

## When to use

- **First pass** on any network dataset before modelling.
- **Detecting hubs** (max degree, degree distribution tail).
- **Small-world diagnostics** (high clustering + low path length).
- **Assortative mixing** — do hubs connect to hubs or to leaves?

## Files

- `python/graph_descriptives.py` — from-scratch summaries via BFS + matrix powers, with a NetworkX cross-check. Demo (Erdős-Rényi G(40, 0.128)): density 0.1244, clustering 0.1456, transitivity 0.1104, assortativity −0.087. NetworkX matches to 4 sig figs. Ring lattice (n=40, k=4): clustering 0.500, diameter 10, path length 5.38, assortativity NaN (regular graph).
- `r/graph_descriptives.R` — `igraph::edge_density / degree / transitivity / assortativity_degree / mean_distance / diameter / components`.

## Assumptions & caveats

- **Directed vs undirected** — this module assumes undirected; use `A + A.T` isn't right for directed. igraph handles both natively.
- **Weighted edges** — density ignores weights; clustering and paths need weighted definitions.
- **Multigraphs / self-loops** — trace-based triangle count includes self-loops if any (usually zero'd on the diagonal).
- **Regular graphs** — degree variance is zero, so degree assortativity is undefined (returns NaN).
- **Path length only defined within components** — global averages exclude unreachable pairs (report component structure alongside).

## Run

```
python techniques/graph-descriptives/python/graph_descriptives.py
Rscript techniques/graph-descriptives/r/graph_descriptives.R
```

**Refs:** Newman, M. *Networks*, 2nd ed., Oxford UP, 2018; Wasserman, S. & Faust, K. *Social Network Analysis*, Cambridge UP, 1994.

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
