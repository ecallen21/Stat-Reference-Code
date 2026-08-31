# Node2Vec / DeepWalk (Reference §30.19)

Represent each node by a low-dim vector so that nodes co-occurring in
**random walks** have similar embeddings. Perozzi (2014) DeepWalk uses
uniform walks + Skip-gram; Grover-Leskovec (2016) node2vec biases the
walks with `(p, q)` return / in-out parameters (BFS vs DFS behaviour).

## Algorithm

```
1. Generate M walks of length L from each node (biased or uniform).
2. Treat walks as "sentences"; train Word2Vec Skip-gram with window w.
3. Read off the trained node vectors as embeddings.
```

**Levy-Goldberg 2014 equivalence**: Skip-gram with negative sampling
factorises a **shifted-PMI** matrix, so DeepWalk can be computed as
`PMI = log( n_uv · N / (n_u · n_v) )` followed by low-rank SVD.

## When to use

- **Downstream node classification / link prediction / clustering**
  on any (weighted, directed) graph.
- **Interpretable proximity structure** — community and role
  discovery.
- **Warm-start for GNNs** — pretrained embeddings.

## When NOT to use

- **Node features present** — a GNN (`graph-neural-network`) integrates
  them naturally.
- **Very small graphs** — direct spectral embeddings (already in the
  repo as `graph-embedding-spectral`) are cheaper.
- **Highly dynamic graphs** — retrain expensive; use online methods.

## Files

- `python/node2vec_deepwalk.py` — from-scratch uniform random walks +
  shifted-PMI + SVD embedding (Levy-Goldberg equivalence). Demo on a
  2-cluster graph with a bridge: **cross-cluster embedding distance
  1.92× intra-cluster** — community structure recovered.
- `r/node2vec_deepwalk.R` — `node2vec` R wrapper (R); `node2vec`,
  `gensim`, `pytorch-geometric`, `karateclub` (Python).

## Assumptions & caveats

- **Walk parameters** — length `L = 40-80` and `M = 10-20` walks per
  node are typical defaults; window `w = 5-10`.
- **Bias `(p, q)`**: `q > 1` favours BFS (structural roles); `q < 1`
  favours DFS (community-like).
- **Signed / weighted edges** — walks generalise via weighted
  transitions.
- **Skip-gram-with-negative-sampling** exactly matches shifted-PMI
  factorisation (Levy-Goldberg 2014); the demo uses the closed-form.
- **Interpretation** — embeddings capture 2nd-order proximity, not
  literal shortest paths.

## Related in this repo

- `graph-embedding-spectral` — Laplacian eigen-embedding cousin.
- `graph-neural-network` — the modern deep alternative.
- `stochastic-block-model`, `latent-space-network`,
  `qap-network-regression`, `gaussian-graphical-model`, `patient-
  similarity-network` — network family (this batch).
- `community-detection` — modularity-based classical alternative.
- `word-embeddings` — same Skip-gram machinery on text.

## Run

```
python techniques/node2vec-deepwalk/python/node2vec_deepwalk.py
Rscript techniques/node2vec-deepwalk/r/node2vec_deepwalk.R
```

**Refs:** Perozzi, B., Al-Rfou, R. & Skiena, S. "DeepWalk: online learning of social representations." *KDD*, 2014; Grover, A. & Leskovec, J. "node2vec: scalable feature learning for networks." *KDD*, 2016; Levy, O. & Goldberg, Y. "Neural word embedding as implicit matrix factorization." *NeurIPS*, 2014.

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
