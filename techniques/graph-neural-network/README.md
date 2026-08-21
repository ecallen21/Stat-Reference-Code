# Graph Neural Network — GCN (Reference §27.x extra)

Kipf & Welling (2017). Message-passing on a graph: each node's embedding is
updated as a **weighted sum of its neighbours' embeddings**.

## GCN layer

```
Â   = A + I                         add self-loops
D̂_ii = Σ_j Â_ij                    diagonal degree
H^{l+1} = σ( D̂^{-½} Â D̂^{-½} H^l W^l )
```

The `D̂^{-½} Â D̂^{-½}` matrix is the **symmetric-normalised adjacency**;
it's a mean over 1-hop neighbours (with self). Stacking `L` layers gives an
`L`-hop receptive field.

## When to use

- **Semi-supervised node classification** — few labels; graph structure carries most of the signal (Cora, Citeseer, Pubmed benchmarks).
- **Node representation learning** — feed a downstream classifier with GCN embeddings.
- **Molecular property prediction** — atoms are nodes, bonds are edges.
- **Recommendation** — bipartite user-item graph message passing (PinSAGE, LightGCN).
- **Knowledge-graph completion** — R-GCN, CompGCN.

## Family

- **GCN** (Kipf-Welling 2017) — the simplest, above.
- **GraphSAGE** (Hamilton 2017) — sampled aggregation, scales to inductive settings.
- **GAT** (Veličković 2018) — attention weights over neighbours.
- **GIN** (Xu 2019) — MLP-based sum aggregation; maximally expressive under Weisfeiler-Lehman.
- **MPNN** (Gilmer 2017) — the general framework subsuming all of the above.
- **Graph transformers / GraphGPS** — attention over all nodes + local edges.

## Files

- `python/graph_neural_network.py` — from-scratch 2-layer GCN in numpy with manual back-prop for node classification. Demo on a 3-community stochastic block model (n=30 nodes, within-p 0.5, between-p 0.05, 2 labels per class, **noise-only node features**): GCN reaches 87.5% test accuracy on unlabelled nodes because the graph structure carries the class signal; MLP on the features alone gets 50% (chance for 3-way in a small test set). This isolates the value added by message-passing.
- `r/graph_neural_network.R` — `torch` (manual GCN layer), `reticulate` + `torch_geometric.nn.GCNConv / GATConv / SAGEConv`, `dgl.nn.GraphConv`.

## Assumptions & caveats

- **Over-smoothing** — many stacked GCN layers make every node's embedding converge to the graph mean; usually stop at 2–4 layers. Fixes: residual connections in GCN (JKNet, GCNII), PairNorm, edge-dropout.
- **Over-squashing** — long-range information gets compressed through bottleneck edges; recent work addresses this via graph rewiring / positional encodings.
- **Graph-level tasks** need a **readout** (mean / sum / attention pooling over nodes) after the last layer.
- **Sparse implementations** are essential at scale — dense `A_norm @ H` is `O(n²)`; use scipy sparse or PyG's message-passing kernels.
- **Homophily assumption** — GCN averages neighbours, which helps when connected nodes share labels (homophilic) and hurts when they don't (heterophilic). Use heterophily-aware GNNs (MixHop, GPRGNN, H2GCN) if the graph is disassortative.
- **Inductive vs transductive** — vanilla GCN is transductive (needs the whole `A` at training); GraphSAGE / PinSAGE handle new nodes.

## Related in this repo

- `graph-descriptives`, `centrality-measures`, `community-detection`, `graph-embedding-spectral` — classical (non-neural) graph analysis.
- `attention-mechanism` — the primitive behind GAT and graph-transformer models.
- `contrastive-learning` — self-supervised graph pretraining (GraphCL, GCA).

## Run

```
python techniques/graph-neural-network/python/graph_neural_network.py
Rscript techniques/graph-neural-network/r/graph_neural_network.R
```

**Refs:** Kipf, T.N. & Welling, M. "Semi-supervised classification with graph convolutional networks." *ICLR*, 2017; Hamilton, W.L., Ying, Z. & Leskovec, J. "Inductive representation learning on large graphs (GraphSAGE)." *NeurIPS*, 2017; Veličković, P. et al. "Graph Attention Networks." *ICLR*, 2018.

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
