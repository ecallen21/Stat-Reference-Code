# GraphSAGE — Inductive Graph Neural Network (Reference §47.149)

Hamilton, Ying & Leskovec (2017). Unlike GCN — which is
transductive and needs the full adjacency at training — GraphSAGE
learns **aggregator functions** over sampled neighbourhoods, so
embeddings can be produced for nodes never seen at training
(**inductive**):

    h_N(v)^(k) = AGG_k({h_u^(k−1) : u ∈ N(v) sampled})
    h_v^(k)    = σ(W^(k) [h_v^(k−1) ‖ h_N(v)^(k)])

Aggregators include mean, LSTM, and max-pool.

## Files

- `python/graphsage_inductive_gnn.py` — 2-layer mean-aggregator
  GraphSAGE with concat + neighbourhood sampling. 3-community SBM
  (120 nodes, 15 labels): **GraphSAGE test acc 66 %** vs LR
  baseline 37 %.
- `r/graphsage_inductive_gnn.R` — no native R port; recommends
  `pytorch-geometric.SAGEConv` / `dgl.SAGEConv`.

## When to use

- **Inductive node classification / regression** where new nodes
  arrive after training (dynamic graphs, growing user bases).
- **Large graphs** where full-batch GCN is infeasible —
  neighbourhood sampling bounds per-node compute.
- **Heterogeneous features** on nodes with edges of a single type.

## When NOT to use

- **Small transductive** benchmarks — GCN / GAT are simpler and
  usually beat GraphSAGE.
- **Edge-heavy tasks** (link prediction, knowledge graphs) — use
  R-GCN / TransE / RotatE.
- **Very heterophilic** graphs — attention-based (GAT) or H2GCN
  work better.

## Assumptions & caveats

- **Sample size k** trades compute vs bias; k = 5-25 typical.
- **Aggregator choice**: mean is default; max-pool captures max
  activation; LSTM breaks permutation invariance (needs shuffle).
- **Feature normalisation** at each layer helps.
- **Loss** — supervised cross-entropy or unsupervised
  neighbourhood-based (predict edges vs random pairs).

## Related in this repo

- `gcn-graph-convolutional` — transductive predecessor.
- `graph-attention-networks-gat` — attention-weighted aggregator.
- `node2vec-deepwalk` — random-walk embedding alternative.
- `stochastic-block-model` — generative graph model used as
  benchmark.

## Run

```
python techniques/graphsage-inductive-gnn/python/graphsage_inductive_gnn.py
Rscript techniques/graphsage-inductive-gnn/r/graphsage_inductive_gnn.R
```

**Refs:** Hamilton, W. L., Ying, R. & Leskovec, J. "Inductive representation learning on large graphs." *NeurIPS*, 2017; Hamilton, W. L. *Graph Representation Learning*, Morgan & Claypool, 2020.

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
