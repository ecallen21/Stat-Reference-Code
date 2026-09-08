# Graph Convolutional Network — GCN (Reference §47.148)

Kipf & Welling (2017). Two-layer graph convolutional network for
semi-supervised node classification:

    H^(1) = ReLU(Â X W^(0))
    Ŷ    = softmax(Â H^(1) W^(1))

with Â = D̃^{-½} (A + I) D̃^{-½}, D̃ = D + I (symmetric
normalisation + self-loops). Cross-entropy trained on the labelled
nodes only; graph structure propagates gradient signal to unlabelled
neighbours.

## Files

- `python/gcn_graph_convolutional.py` — from-scratch two-layer GCN
  with hand-derived backprop. 3-community stochastic block model,
  n = 120 nodes, p_in = 0.15, p_out = 0.01, weak one-hot-plus-noise
  features. Semi-supervised: 5 labels per class (15/120 = 12.5 %).
  - **GCN test accuracy = 93.3 %**
  - Baseline logistic regression on features alone
    (no graph): **37.1 %** — GCN gains 56 pts by exploiting
    homophily.
- `r/gcn_graph_convolutional.R` — illustrative forward pass with
  random weights; recommends `pytorch-geometric` / `dgl` (Python)
  via `reticulate` for real training.

## When to use

- **Node classification** on graphs with homophily (labels of
  neighbours correlate).
- **Semi-supervised learning** with abundant graph structure and
  scarce labels (citation networks, protein interaction, clinical
  patient similarity).
- **Graph representation learning** as an encoder stage.

## When NOT to use

- **Heterophilic graphs** — labels anti-correlate with neighbours
  (H2GCN / GPR-GNN variants handle this).
- **Very deep networks** (> 3 layers) — over-smoothing; embeddings
  collapse to a single vector across the whole component.
- **Dynamic / evolving graphs** at scale — use temporal GNNs.
- **Very large graphs** (> 1M nodes) — full-batch GCN needs the
  whole adjacency in memory; use GraphSAGE / cluster-GCN /
  neighbour sampling.

## Assumptions & caveats

- **Undirected graphs** by construction; directed = symmetrised
  adjacency or use RGCN.
- **Self-loops** essential — otherwise each node's own features
  don't reach its own output.
- **Feature preprocessing** matters: normalise rows or use L2
  normalisation.
- **Hidden dim** typically small (16-64) for citation-network
  benchmarks; L2 penalty prevents over-fit with tiny labelled sets.

## Related in this repo

- `stochastic-block-model` — the generative graph model used
  here as testbed.
- `node2vec-deepwalk` — random-walk-based node embedding
  alternative.
- `latent-space-network`, `patient-similarity-network` — network
  analysis siblings.
- `label-propagation` — non-parametric semi-supervised analogue.

## Run

```
python techniques/gcn-graph-convolutional/python/gcn_graph_convolutional.py
Rscript techniques/gcn-graph-convolutional/r/gcn_graph_convolutional.R
```

**Refs:** Kipf, T. N. & Welling, M. "Semi-supervised classification with graph convolutional networks." *ICLR*, 2017; Hamilton, W. L. *Graph Representation Learning*, Morgan & Claypool, 2020; Wu, Z. et al. "A comprehensive survey on graph neural networks." *IEEE TNNLS* 32(1), 2021.

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
