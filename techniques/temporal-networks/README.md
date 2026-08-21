# Temporal Networks (Reference §24.x extra)

A **temporal network** is a stream of edge events `(u, v, t)` — contacts,
messages, transactions — with explicit time stamps. Static-network analysis
collapses time, losing the causal ordering that determines what can influence
what.

## Time-respecting paths

A path `u = v_0 → v_1 → … → v_k = w` is **time-respecting** iff the edge
times `t_i` are non-decreasing along the path (`<` for strict temporal
paths). Only these paths can carry causal influence forward.

Static path `⇒` a set of edges exists in the aggregate — but not necessarily
in a valid order. So temporal reachability is a subset of aggregate
reachability.

## Key quantities

- **Snapshot** `A_t` — static graph of edges with `t_k ≤ t` (or in window `(t − w, t]`).
- **Foremost / earliest-arrival time** `τ(s → v)` — earliest time a time-respecting path from `s` reaches `v`.
- **Temporal reachability matrix** `R[s, v] = 1` iff a time-respecting path exists.
- **Temporal betweenness / closeness** — path-count and inverse-arrival-time centralities.
- **Burstiness** `B = (σ_τ − μ_τ) / (σ_τ + μ_τ)` — heavy-tailed inter-event times.
- **Motif counts on time-ordered triples** — temporal analogues of triangles.

## Higher-order models

Static graphs collapse `A → B → C` into edges `AB, BC` — losing the fact
that `A → B` must precede `B → C` for causal transfer. **Higher-order networks**
(Scholtes, Rosvall) or **De Bruijn graphs** encode second-order transitions
so that random walks respect empirical path statistics.

## When to use

- **Epidemic spread** on contact data (SocioPatterns, Bluetooth proximity).
- **Communication analytics** — email, messaging, transaction graphs.
- **Financial markets** — order-book event streams.
- **Anomaly / fraud detection** — bursts of unusual temporal motifs.
- **Behavioural science** — time-ordered interactions in a group.

## Files

- `python/temporal_networks.py` — sliding-window snapshot builder, earliest-arrival BFS, temporal reachability matrix. Demo (n=8, 30 random contact events on [0, 100]): aggregated density 0.643; sliding windows (T−25, T−50, T−100] give 8, 14, 18 edges; earliest arrival from node 0 reaches all 8 nodes with times 10.5, 29.9, 35.8, …; temporal reachability equals static reachability (small graph, easy propagation).
- `r/temporal_networks.R` — `networkDynamic`, `tsna::tPath / tReach / tSnaStats`, `timeordered`; Python `teneto`, `pathpy2`.

## Assumptions & caveats

- **Time resolution** — many datasets have coarse timestamps (day-level); ties in `t` need a rule (allow `≤`, sort by original order, or randomly jitter).
- **Window / snapshot choices** — different `w` give different networks; report a range.
- **Directed vs undirected temporal edges** — matters for time-respecting paths; disease contacts are usually undirected, messages are directed.
- **Sparsity of temporal paths** — most `(s, v)` pairs may have `τ = ∞` in bursty data; static-graph reachability overstates influence.
- **Time-respecting-path enumeration is expensive** — use foremost / fastest / shortest / latest variants depending on the substantive question.
- **Higher-order models** are the right choice when the process on the network has memory (repeat callers, funnels).

## Run

```
python techniques/temporal-networks/python/temporal_networks.py
Rscript techniques/temporal-networks/r/temporal_networks.R
```

**Refs:** Holme, P. & Saramäki, J. "Temporal networks." *Physics Reports* 519(3), 97–125, 2012; Kempe, D., Kleinberg, J. & Kumar, A. "Connectivity and inference problems for temporal networks." *STOC*, 2000; Scholtes, I. et al. "Causality-driven slow-down and speed-up of diffusion in non-Markovian temporal networks." *Nat. Commun.* 5, 5024, 2014.

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
