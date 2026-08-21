# k-Core Decomposition and k-Truss (Reference §24.x extra)

Both are **hierarchical cohesive-subgraph** notions that peel a graph from
its periphery inward.

## k-core

The **k-core** of `G` is the maximal subgraph in which every vertex has
degree ≥ `k` **within the subgraph**. The **coreness** (core number) of a
vertex `v` is the largest `k` for which `v` belongs to the `k`-core.

- k-cores are nested: `(k+1)-core ⊆ k-core`.
- The **degeneracy** is `max_v coreness(v)`.

Batagelj-Zaversnik (2003) `O(m + n)` algorithm: repeatedly remove the
minimum-degree vertex; its coreness is `max(previous k, current degree)`.

## k-truss (Cohen 2008)

Edge-based analogue: the **k-truss** is the maximal subgraph in which every
edge participates in at least `k − 2` triangles. Trusses are denser than
cores at the same k: a k-truss is contained in the (k−1)-core.

## When to use

- **Dense-subgraph mining** — social network cores, protein complexes, community seeds.
- **Anomaly detection** — vertices with unusually high or low coreness stand out.
- **Graph visualisation** — k-shell layout (0-1 core outermost, degeneracy innermost) organises hairball plots.
- **Streaming / big-graph** — coreness is a `O(m + n)` global summary; used at scale (billion-edge graphs) as a coarse importance signal.
- **Robustness / percolation** — successive core-peels model cascade failures.

## Files

- `python/k_core_decomposition.py` — Batagelj-Zaversnik coreness + k-core subgraph extraction + naive k-truss. NetworkX cross-check for coreness. Demo (K5 joined by a bridge to K7 + a pendant): K5 vertices → coreness 4, K7 vertices → coreness 6, pendant → 1; 6-core cleanly extracts K7 (7 vertices); NetworkX agreement to 0.
- `r/k_core_decomposition.R` — `igraph::coreness`, `igraph::k_core`, `igraph::max_cores`.

## Related methods

- **k-plex, quasi-cliques** — relax edge density inside; more expensive.
- **Densest subgraph** — max mean degree; Charikar's 2-approximation via LP.
- **k-truss decomposition** with the O(m^{1.5}) algorithm (Cohen 2008); the naive implementation here re-counts triangles each pass — fine for small demos, slow for large graphs.

## Assumptions & caveats

- **Undirected, unweighted** in this module; degree-based extensions to weighted / directed exist (in-core, out-core).
- **Coreness ≠ community label** — a k-core can span multiple communities. Combine with `community-detection` for structure and density.
- **Degeneracy grows with density** — very dense graphs have flat coreness distributions; use only alongside other summaries.
- **Streaming decomposition** requires specialised algorithms (Sariyüce et al.); the offline algorithm above needs the whole graph.
- **Truss vs core** — trusses are stricter (require closed triangles); use trusses when triangle-density is the substantive notion (e.g. dense social groups).

## Run

```
python techniques/k-core-decomposition/python/k_core_decomposition.py
Rscript techniques/k-core-decomposition/r/k_core_decomposition.R
```

**Refs:** Seidman, S.B. "Network structure and minimum degree." *Social Networks* 5(3), 269–287, 1983; Batagelj, V. & Zaversnik, M. "An O(m) algorithm for cores decomposition of networks." arXiv:cs.DS/0310049, 2003; Cohen, J. "Trusses: cohesive subgraphs for social network analysis." NSA Tech Report, 2008.

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
