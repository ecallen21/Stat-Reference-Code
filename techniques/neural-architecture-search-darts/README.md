# DARTS — Differentiable Architecture Search (Reference §47.279)

Liu, Simonyan & Yang (2019). NAS via CONTINUOUS RELAXATION of
the discrete choice:

```
y = Σ_o softmax(α_o) · o(x)     o ∈ Ops = {conv3, conv5, id, ...}
```

Train (w, α) jointly with bilevel optimisation:

- **w step**: train weights on training data
- **α step**: train architecture params on validation data

At the end, discretise by picking `o* = argmax_o α_o` at each
edge. Orders of magnitude faster than RL / evolutionary NAS.

## Files

- `python/neural_architecture_search_darts.py` — 1-edge, 4-op
  DARTS-lite. Ops: sqrt, identity, square, abs. True function
  is 0.5 · x² so "square" should win. Per-op fitted MSE:
  square 0.002 (winner), identity/abs 0.09, sqrt 0.25. DARTS
  softmax over 400 iterations shifts to square with p=0.883.
- `r/neural_architecture_search_darts.R` — reticulate +
  PyTorch DARTS / NNI (R); quark0/darts, NNI NAS,
  AutoKeras, from-scratch (Python).

## When to use

- **Cell-based NAS** — DARTS's canonical setting (ResNet /
  DenseNet cell search).
- **Cheap gradient-based search** — orders faster than RL /
  evolutionary NAS.
- **Continuous relaxations of discrete choices** — dropout
  patterns, activation ops, block structures.

## When NOT to use

- **Very small search spaces** — grid or random search is
  simpler.
- **Categorical HPs unrelated to network structure** — DARTS
  needs a differentiable relaxation.
- **Real-time constraints** — DARTS's initial supernet is
  memory-hungry.

## Assumptions & caveats

- **DARTS collapse** — early iterations often over-select
  parameter-free ops (skip / identity); mitigations:
  PC-DARTS, P-DARTS, Fair DARTS.
- **Bilevel optimisation approximation** — the α gradient
  uses a first-order (finite-difference or single-step)
  approximation; second-order is more expensive but tighter.
- **Discretisation loss** — the argmax at the end can diverge
  from the trained supernet; retraining the discretised
  architecture is essential.
- **Validation-set overfitting** — the α update can overfit;
  use a fresh val split at each α step.

## Related in this repo

- `bayesian-optimization`, `tpe-tree-parzen-estimator` —
  discrete NAS alternatives.
- `hyperband-multi-fidelity`, `successive-halving-asha` —
  budget-adaptive search.
- `meta-learning-maml`, `reptile-meta-learning` — related
  fast-adaptation methods.

## Run

```
python techniques/neural-architecture-search-darts/python/neural_architecture_search_darts.py
Rscript techniques/neural-architecture-search-darts/r/neural_architecture_search_darts.R
```

**Refs:** Liu, H., Simonyan, K. and Yang, Y. "DARTS: Differentiable architecture search." In *ICLR*, 2019.

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
