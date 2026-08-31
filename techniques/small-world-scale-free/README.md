# Small-World + Scale-Free Networks (Reference §30.12)

Two canonical network generators.

## Watts-Strogatz small-world (1998)

Start from a **k-ring lattice**; rewire each edge with probability
`p` to a random node. High clustering **+** short path lengths (unlike
either extreme).

## Barabási-Albert scale-free (1999)

Grow the network by **preferential attachment**: each new node with
`m` edges attaches to existing nodes with probability `∝` degree.
Degree distribution is a **power law** `P(k) ~ k^{−3}`.

## Signature statistics

- **Clustering coefficient** — average triangle density around nodes.
- **Average shortest-path length** — mean over all node pairs.
- **Degree exponent** — slope of `log P(k > x)` vs `log x`.

## When to use

- **Null / generative models** for comparing empirical networks.
- **Simulation** of contagion / diffusion / cascade processes.
- **Teaching / benchmarking** community-detection algorithms.

## When NOT to use

- **Fit to a real network** — SBM / latent-space / ERGM are richer.
- **Explanation of real-world power laws** — many real degree
  distributions are truncated / log-normal (Broido-Clauset 2019).

## Files

- `python/small_world_scale_free.py` — from-scratch ring, WS,
  Barabási-Albert, Erdős-Rényi generators + clustering + BFS shortest
  path + log-log degree slope. Demo `n=60`:
  - **Erdős-Rényi (p=0.10)**: clustering 0.10, avg path 2.44.
  - **Ring (k=6)**: clustering 0.60, avg path 5.42.
  - **WS p=0.05**: clustering 0.52, avg path 3.77 (**small-world**
    signature: keeps clustering, shortens paths).
  - **BA (m=3)**: clustering 0.15, avg path 2.38, largest degree
    exponent.
- `r/small_world_scale_free.R` — `igraph::sample_smallworld`,
  `igraph::sample_pa`, `poweRlaw` (R); `networkx`, `powerlaw`
  (Python).

## Assumptions & caveats

- **Small n** — power-law fits are noisy for `n < 1000`.
- **Degree-exponent estimation** — MLE (Clauset-Shalizi-Newman 2009)
  with cutoff selection; log-log regression is a heuristic.
- **Watts-Strogatz `p`** — small `p` (0.01-0.1) preserves clustering
  while dramatically shortening paths — the "small-world" regime.
- **Barabási-Albert `m`** — average degree `≈ 2m`; higher `m` yields
  denser networks but the same power-law tail.

## Related in this repo

- `random-graph-models` — the null-model catalogue.
- `community-detection`, `stochastic-block-model`,
  `latent-space-network` — generative models with structure.
- `network-diffusion`, `network-motifs`, `graph-descriptives`,
  `graph-comparison`, `graph-embedding-spectral` — analyses to run on
  generated networks.
- `homophily-assortativity` — attribute-based network signature.

## Run

```
python techniques/small-world-scale-free/python/small_world_scale_free.py
Rscript techniques/small-world-scale-free/r/small_world_scale_free.R
```

**Refs:** Watts, D.J. & Strogatz, S.H. "Collective dynamics of small-world networks." *Nature*, 1998; Barabási, A.-L. & Albert, R. "Emergence of scaling in random networks." *Science*, 1999; Clauset, A., Shalizi, C.R. & Newman, M.E.J. "Power-law distributions in empirical data." *SIAM Review*, 2009.

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
