# Graph Attention Networks — GAT (Reference §47.150)

Veličković et al. (2018). Replaces GCN's fixed symmetric-
normalised adjacency with **learned attention weights** on edges:

    e_ij = LeakyReLU(a^T [W h_i ‖ W h_j])
    α_ij = softmax_j(e_ij)   (over neighbours of i)
    h_i^(new) = σ(Σ_j α_ij W h_j)

Multi-head attention concatenated / averaged for stability. Handles
heterophilic edges better than GCN because attention can down-weight
uninformative neighbours.

## Files

- `python/graph_attention_networks_gat.py` — from-scratch multi-
  head GAT with softmax attention over sampled edges. 3-community
  SBM (120 nodes, 15 labels):
  - 1 head: test acc 0.810
  - **4 heads: test acc 0.819**
  - 8 heads: 0.771 (over-parameterised for tiny data).
  - Baseline LR (no graph): 0.371.
- `r/graph_attention_networks_gat.R` — no native R port;
  recommends `pytorch-geometric.GATConv` / `GATv2Conv`.

## When to use

- **Node classification** where neighbours differ in relevance
  (heterogeneous or heterophilic graphs).
- **Interpretability** — attention weights give per-edge
  contribution (with caveats).
- **Heterogeneous graphs** — natural extension via typed
  attention.

## When NOT to use

- **Homophilic, small graphs** — GCN is simpler and as accurate.
- **Extremely large graphs** — dense attention is O(|E| · d);
  neighbourhood sampling or sparse variants required.
- **Feature-poor nodes** — attention has nothing to weight on.

## Assumptions & caveats

- **Attention as explanation** is not causal (Jain-Wallace 2019).
- **Multi-head averaging** at the last layer avoids concat's
  dimensionality blow-up.
- **GATv2** (Brody 2022) fixes GAT's limited attention
  expressiveness — same code, different score computation.
- **Softmax over neighbours** normalises per-node, unlike a
  global attention.

## Related in this repo

- `gcn-graph-convolutional`, `graphsage-inductive-gnn` — sibling
  GNN architectures.
- `attention-mechanism`, `transformer-encoder`,
  `transformer-decoder` — the general attention framework.
- `stochastic-block-model` — testbed graph model.

## Run

```
python techniques/graph-attention-networks-gat/python/graph_attention_networks_gat.py
Rscript techniques/graph-attention-networks-gat/r/graph_attention_networks_gat.R
```

**Refs:** Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P. & Bengio, Y. "Graph attention networks." *ICLR*, 2018; Brody, S., Alon, U. & Yahav, E. "How attentive are graph attention networks?" *ICLR*, 2022.

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
