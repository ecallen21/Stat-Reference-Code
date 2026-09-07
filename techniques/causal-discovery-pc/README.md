# Causal Discovery -- PC Algorithm (Reference §15.35)

Spirtes, Glymour & Scheines (2000). The **PC (Peter-Clark)**
algorithm learns the skeleton and colliders of a DAG from
observational data using conditional-independence tests.

## Two-phase procedure

1. **Skeleton phase.** Start from the complete undirected graph; for
   each edge `(i, j)`, test `H₀: Xᵢ ⊥ Xⱼ | S` for growing subsets
   `S ⊆ Ne(i) \ {j}`. If any `S` makes them independent, delete the
   edge and record `sep(i, j) = S`.
2. **Orientation phase.** For every unshielded triple `i − k − j`
   (`i, j` non-adjacent), orient as collider `i → k ← j`
   iff `k ∉ sep(i, j)`. Apply Meek propagation rules for further
   orientation.

Returns a **CPDAG** — the Markov equivalence class of DAGs consistent
with the observed CI structure.

## Files

- `python/causal_discovery_pc.py` — Fisher-z partial-correlation CI
  test + skeleton + collider orientation from scratch. Demo (n=3000,
  DAG: X0→X2←X1, X2→X3, X4 isolated): recovers skeleton (X0-X2,
  X1-X2, X2-X3) exactly, orients the v-structure at X2 correctly,
  leaves X4 isolated.
- `r/causal_discovery_pc.R` — `pcalg::pc`, `pcalg::fci`,
  `bnlearn::pc.stable`, `bnlearn::iamb` (R); `causal-learn`, `cdt`
  (Python).

## When to use

- **Exploratory DAG discovery** — no prior knowledge of the causal
  structure; want a data-driven starting point.
- **Auditing causal assumptions** — check whether the assumed graph
  is consistent with observed conditional independencies.
- **Feature-selection through backdoor structure** — identify the
  adjustment set for a target intervention.

## When NOT to use

- **Small n / high p** — CI tests are underpowered; false-negative
  edges dominate.
- **Non-Gaussian dependencies** — Fisher-z is only valid for jointly
  Gaussian data. Use rank-based (GSquared) or kernel-based CI
  (KCIT) instead.
- **Latent confounders** — PC assumes causal sufficiency; use **FCI**
  (`pcalg::fci`) for latent-robust discovery.
- **Cyclic causal systems** — PC returns a DAG; feedback loops need
  SEM / cyclic-graph methods.

## Assumptions & caveats

- **Causal Markov + Faithfulness** — observed CI structure exactly
  matches the true graph's d-separations; violations lead to spurious
  edges.
- **Order-dependence** — the classical PC output can depend on
  variable ordering when tests conflict; use `pcalg::pc(stable=TRUE)`
  or `bnlearn::pc.stable`.
- **Multiple testing** — the algorithm runs thousands of CI tests;
  consider a stricter `α` or Bonferroni-style adjustment.
- **Not all edges get oriented** — the CPDAG contains undirected
  edges within the same Markov equivalence class.

## Related in this repo

- `gaussian-graphical-model` — undirected graph via glasso;
  precursor / alternative for skeleton.
- `bayesian-hierarchical-models`, `path-analysis` — parameter
  estimation once the DAG is fixed.
- `mendelian-randomization`, `iv-2sls` — identifying strategies for
  effects on the discovered DAG.

## Run

```
python techniques/causal-discovery-pc/python/causal_discovery_pc.py
Rscript techniques/causal-discovery-pc/r/causal_discovery_pc.R
```

**Refs:** Spirtes, P., Glymour, C. & Scheines, R. *Causation, Prediction and Search*, 2nd ed., MIT Press, 2000; Kalisch, M. & Bühlmann, P. "Estimating high-dimensional DAGs with the PC-algorithm." *JMLR*, 8: 613-636, 2007.

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
