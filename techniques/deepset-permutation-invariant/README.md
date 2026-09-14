# Deep Sets — Permutation-Invariant Networks (Reference §47.312)

Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov & Smola
(2017). Any permutation-INVARIANT function on a set can be
decomposed as:

```
f({x_1, …, x_n}) = ρ(Σ_i φ(x_i))
```

where φ is an element-wise transform and ρ a downstream MLP.
Foundation for set-input networks (point clouds, tabular
aggregations, molecular fingerprints).

## Files

- `python/deepset_permutation_invariant.py` — Verify
  invariance: apply the Deep-Sets pipeline (φ MLP → sum
  pool → ρ MLP) to a 5-vector set under multiple
  permutations. Output is IDENTICAL. Naive-concatenation
  baseline produces a different output for each permutation.
- `r/deepset_permutation_invariant.R` — `torch` (R) +
  rowwise MLP + pool (R); PyG.aggr, `set-transformer`,
  from-scratch (Python).

## When to use

- **Set-valued inputs** — point clouds, molecular graphs
  (as sets of atoms), tabular aggregations.
- **Multiset / bag-of-features** where element order is
  meaningless.
- **Graph neural networks** — Deep Sets is the invariant
  aggregator behind message passing.

## When NOT to use

- **Ordered / sequential data** — use RNN / Transformer.
- **When element interactions matter beyond pairwise sums**
  — use Set Transformer (Lee et al 2019) with attention.
- **Set of varying elements with type** — combine with typed
  sub-networks per class.

## Assumptions & caveats

- **Universality theorem** — Deep Sets can represent any
  invariant continuous function of a fixed-size set (Zaheer
  2017); for varying-size sets under continuity, additional
  care is needed (Wagstaff 2019).
- **Pooling choice** — sum is theoretically expressive; mean
  / max often used for stability.
- **Latent dimension** — needs to be ≥ max set size for
  full expressivity.
- **Padding** — for varying-size sets, mask padded elements
  before pooling.

## Related in this repo

- `graph-neural-network`, `gcn-graph-convolutional`,
  `graph-attention-networks-gat`, `graphsage-inductive-gnn`
  — Deep Sets as the aggregator inside message passing.
- `attention-mechanism` — the Set Transformer alternative.
- `siamese-networks` — related shared-parameter architecture.

## Run

```
python techniques/deepset-permutation-invariant/python/deepset_permutation_invariant.py
Rscript techniques/deepset-permutation-invariant/r/deepset_permutation_invariant.R
```

**Refs:** Zaheer, M., Kottur, S., Ravanbakhsh, S., Póczos, B., Salakhutdinov, R.R. and Smola, A.J. "Deep sets." In *NeurIPS*, pp. 3391-3401, 2017.

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
