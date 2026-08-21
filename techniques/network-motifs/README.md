# Network Motifs (Reference §24.11)

Small connected subgraphs whose count in the observed graph is **significantly
different** from a null preserving basic properties. Milo et al. (2002)
introduced motif analysis to identify recurring "building blocks" of a
network — e.g. feed-forward loops in transcription networks, mutual dyads in
online friendship graphs.

## 3-node undirected motifs

Only two connected patterns:

- **Wedge / open 2-path** — 3 nodes, 2 edges.
- **Triangle** — 3 nodes, 3 edges.

Relationship:

```
# wedges = Σ_v C(deg(v), 2) − 3 · # triangles
```

## 3-node directed motifs

Thirteen distinct patterns (**Holland-Leinhardt triad census** with codes
003, 012, 102, 021D, …, 300). `igraph::triad_census` returns 16 categories
including disconnected triples.

## Statistical test

```
Z = (obs − mean_null) / sd_null
```

The null is usually a **degree-preserving randomised graph** — double-edge
swap MCMC or the configuration model. `|Z| ≳ 2–3` is the usual threshold;
correct for multiple testing when scanning many motifs.

## When to use

- **Systems biology** — feed-forward loops, bi-fans in transcription regulatory networks (Milo et al.).
- **Ecology** — over-represented interaction motifs in food webs.
- **Social networks** — closed triads (transitivity), mutual dyads.
- **Fraud / anomaly detection** — unusual local structures.

## Files

- `python/network_motifs.py` — triangle & wedge counts + double-edge-swap null + Z-scores. Demo (n=40, 100 nulls): Erdős-Rényi Z_triangles = −0.80, Z_wedges = +0.80 (no signal); two-clique-of-cliques Z_triangles = +90.2, Z_wedges = −90.2 (2280 triangles observed vs 1038 expected).
- `r/network_motifs.R` — `igraph::triad_census / motifs / rewire`.

## Assumptions & caveats

- **Choice of null** — vanilla ER null flags every real network as motif-rich; use degree-preserving. For finer control preserve degree + joint-degree, or use SBM baseline.
- **Multiple testing** across many motif types — Bonferroni / Holm.
- **Directed / labelled** motifs need proper enumeration (`FANMOD`, `mfinder`) — the undirected case here handles only two patterns.
- **Motif significance ≠ mechanism** — over-representation is descriptive; causal generative claims need model fits.
- **Computational cost** — 3-node census is `O(n · d²)`; 4-node census is much heavier and often sampled.

## Run

```
python techniques/network-motifs/python/network_motifs.py
Rscript techniques/network-motifs/r/network_motifs.R
```

**Refs:** Milo, R. et al. "Network motifs: simple building blocks of complex networks." *Science* 298(5594), 824–827, 2002; Holland, P.W. & Leinhardt, S. "A method for detecting structure in sociometric data." *Amer. J. Sociol.* 76(3), 492–513, 1970.

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
