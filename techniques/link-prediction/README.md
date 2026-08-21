# Link Prediction (Reference §24.7)

Score every candidate edge (i, j) and rank them; the top-K non-edges are the
model's guesses for **future** or **hidden** links.

## Neighbourhood scores

| Score | Formula |
|---|---|
| **Common neighbours** | `|N(i) ∩ N(j)|` |
| **Jaccard** | `|N(i) ∩ N(j)| / |N(i) ∪ N(j)|` |
| **Adamic-Adar** | `Σ_{z ∈ N(i) ∩ N(j)} 1 / log(deg(z))` |
| **Resource allocation** | `Σ_{z ∈ N(i) ∩ N(j)} 1 / deg(z)` |
| **Preferential attachment** | `deg(i) · deg(j)` |

Adamic-Adar and resource allocation penalise "generic" high-degree common
neighbours — a shared low-degree contact is stronger evidence than a
shared hub.

## Path-based scores

- **Katz index** `Σ_l β^l · (A^l)_ij` — walks of all lengths, decayed by `β < 1 / ρ(A)`.
- **Random walk with restart** — stationary distribution of a walk that restarts at `i` with probability `α`.
- **SimRank**, **Rooted PageRank** — related structural similarities.

## Model-based scores

- **SBM / degree-corrected SBM** — plug-in edge probability from a fitted generative model (see `stochastic-block-model`).
- **Graph embedding** — inner product / MLP on learned node vectors (see `graph-embedding-spectral`, node2vec).

## Evaluation

1. Hide a random fraction (`test_frac = 0.1 – 0.3`) of edges.
2. Score every candidate pair not present in the *train* graph.
3. Compute ROC-AUC of the score against the "is this a hidden edge" label. Also useful: precision@K, average precision.

## When to use

- **Recommendation** — "friends you may know", collaborator suggestions.
- **Discovery of missing / future edges** in observational networks.
- **Baseline for graph representation learning** — neighbourhood scores are hard to beat on plain communities.

## Files

- `python/link_prediction.py` — from-scratch scoring + AUC via Mann-Whitney U on all 5 methods. Demo (planted partition n=60, K=3, within-p 0.4, between-p 0.02, 20% edges held out): Jaccard 0.798, Adamic-Adar 0.789, resource-allocation 0.789, common-neighbours 0.779, preferential 0.497 (no degree signal in a planted-partition graph).
- `r/link_prediction.R` — `igraph::similarity`, `linkprediction::proxfun`, `pROC::roc`.

## Assumptions & caveats

- **Undirected, unweighted** in this module — extend to directed / weighted with directed neighbourhoods and weighted degrees.
- **Class imbalance** — the negative class dominates; AUC is misleading if the top-K is what matters. Report precision@K too.
- **Snapshots vs streams** — for temporal networks, evaluate future-edge prediction with a time-based split, not a random split.
- **Cold-start** — nodes with degree 0 in the train graph have zero score for all neighbourhood methods; use content or SBM-style back-off.
- **Small-world / cliques** — neighbourhood scores dominate; hub-heavy networks favour Adamic-Adar / RA over raw common-neighbour count.

## Run

```
python techniques/link-prediction/python/link_prediction.py
Rscript techniques/link-prediction/r/link_prediction.R
```

**Refs:** Liben-Nowell, D. & Kleinberg, J. "The link-prediction problem for social networks." *JASIS&T* 58(7), 1019–1031, 2007; Adamic, L.A. & Adar, E. "Friends and neighbors on the web." *Social Networks* 25(3), 211–230, 2003; Zhou, T., Lü, L. & Zhang, Y.-C. "Predicting missing links via local information." *Eur. Phys. J. B* 71, 623–630, 2009.

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
