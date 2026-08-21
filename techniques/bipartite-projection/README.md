# Bipartite Projection & Modularity (Reference §24.10)

A bipartite (two-mode) graph `B = (U, V, E)` has edges only between two
disjoint node sets — e.g. users × products, papers × keywords, patients × diagnoses.

## Biadjacency matrix

`B` is `|U| × |V|`; `B[u, v] = 1` iff `u` is linked to `v`.

## One-mode projections

Collapse to a one-mode graph on either side:

- **Unweighted**: `G_U[i, j] = 1` if `u_i` and `u_j` share **any** V-neighbour.
- **Weighted (BB^T)**: `G_U[i, j] = |N(i) ∩ N(j)|` — count of shared V-neighbours.
- **Newman hyperbolic**: `G_U[i, j] = Σ_{v ∈ N(i) ∩ N(j)} 1 / (deg(v) − 1)` — down-weight generic V-nodes (e.g. best-seller books, ubiquitous keywords).
- **Jaccard / cosine** — normalised alternatives.

## Bipartite modularity (Barber 2007)

For a joint partition assigning U-nodes to labels `c(u)` and V-nodes to `c(v)`:

```
Q_B = (1 / m) · Σ_{u, v} [B_{uv} − k_u · d_v / m] · 1{c(u) = c(v)}
```

Same shape as Newman-Girvan modularity but with the *bipartite* degree-preserving null.

## When to use

- **Two-mode data** — users × items (recommendation), authors × papers (co-authorship), species × sites (ecology), transactions × features (fraud).
- **Comparing entities across a shared context** — projection makes the entity graph explicit.
- **Community detection** — bipartite modularity (via `bipartite::computeModules`) or SBM extensions (`sbm::estimateBipartiteSBM`).
- **Recommendation baseline** — projection + neighbourhood similarity (Adamic-Adar on the projection).

## Files

- `python/bipartite_projection.py` — weighted BB^T and Newman hyperbolic projections + Barber bipartite modularity. Demo (|U|=8, |V|=10, 2 planted communities, within-p 0.9, between-p 0.05): weighted projection cleanly shows two blocks; Newman projection down-weights uniformly to ~1.67; bipartite Q of planted partition = 0.497 vs random 0.013; NetworkX weighted projection matches total-weight 52.
- `r/bipartite_projection.R` — `igraph::bipartite_projection`, `bipartite::computeModules`, `tnet::projecting_tm`.

## Assumptions & caveats

- **Projection loses information** — very different bipartite graphs can yield the same projection; always analyse the two-mode graph directly when you can.
- **Weighted vs unweighted** — the choice matters more than modeling details later. Newman weighting is usually preferable when V-degrees are heavy-tailed.
- **Bipartite null** — using the vanilla modularity's null on a projection double-counts the bipartite structure; use Barber's Q_B.
- **Bipartite SBM** — more principled generative alternative for community detection on two-mode data.
- **Directionality / temporality** — bipartite graphs from transaction / event logs have implicit time; sliding-window projections are standard.

## Run

```
python techniques/bipartite-projection/python/bipartite_projection.py
Rscript techniques/bipartite-projection/r/bipartite_projection.R
```

**Refs:** Newman, M.E.J. "Scientific collaboration networks. II. Shortest paths, weighted networks, and centrality." *Phys. Rev. E* 64, 016132, 2001; Barber, M.J. "Modularity and community detection in bipartite networks." *Phys. Rev. E* 76, 066102, 2007; Latapy, M., Magnien, C. & Del Vecchio, N. "Basic notions for the analysis of large two-mode networks." *Social Networks* 30(1), 31–48, 2008.

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
