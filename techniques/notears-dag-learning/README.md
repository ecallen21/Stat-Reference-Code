# NOTEARS DAG Learning (Reference §47.55)

Zheng, Aragam, Ravikumar & Xing (2018). Replaces the combinatorial
acyclicity constraint of DAG structure learning with a smooth,
differentiable equivalent:

    h(W) = tr( exp(W ∘ W) ) − d = 0     iff   W is a DAG.

Minimises   0.5‖X − X W‖²/n + λ‖W‖₁   subject to h(W) = 0 via
augmented Lagrangian + L-BFGS-B. Turns NP-hard structure search
into a continuous nonconvex program.

## Files

- `python/notears_dag_learning.py` — from-scratch augmented-
  Lagrangian NOTEARS for the linear-Gaussian SEM using
  `scipy.linalg.expm` and `scipy.optimize.L-BFGS-B`. Demo (d=4,
  n=800, true edges 0→1 (1.5), 0→2 (−1.0), 1→3 (0.7), 2→3 (0.9)):
  **exact recovery** — SHD 0, TP 4, FP 0, FN 0. Estimated
  weights (1.26, −0.49, 0.52, 0.64) after L1 shrinkage.
- `r/notears_dag_learning.R` — `notearsC`, `pcalg` (R);
  `causalnex`, `dagma`, from-scratch (Python).

## When to use

- **Continuous / mixed observational data** with plausible
  DAG structure — gene networks, sensor causality, biomarkers.
- **Complement PC / GES output** — same input, different search.
- **Score-based structure learning** at moderate d (up to a
  few hundred with modern DAGMA / TOPO variants).

## When NOT to use

- **Time-series causality** — use Granger / DYNOTEARS.
- **Latent confounders** suspected — use FCI / RFCI; NOTEARS
  assumes causal sufficiency.
- **Very large d** — vanilla NOTEARS O(d³) per iter; use DAGMA
  or sparse-gradient variants.
- **Purely discrete data** — use NOTEARS-MLP or GSGES.

## Assumptions & caveats

- **Faithfulness / causal sufficiency** — standard DAG-learning
  assumption; no hidden confounders.
- **Linear-Gaussian SEM** by default; nonlinear extensions
  (NOTEARS-MLP) exist.
- **Local optima** — augmented-Lagrangian is non-convex; run
  multiple restarts.
- **L1 tuning** — cross-validate or use CV-BIC; too small keeps
  spurious edges.

## Related in this repo

- `causal-discovery-pc` — constraint-based baseline.
- `causal-forest`, `hte-uplift`, `dml-double-ml` — HTE / causal
  effect siblings.
- `bayesian-network`-style tools: `gaussian-graphical-model`.
- `granger-causality` — time-series causality.

## Run

```
python techniques/notears-dag-learning/python/notears_dag_learning.py
Rscript techniques/notears-dag-learning/r/notears_dag_learning.R
```

**Refs:** Zheng, X., Aragam, B., Ravikumar, P. & Xing, E.P. "DAGs with NO TEARS: Continuous optimization for structure learning." *NeurIPS*, 2018; Bello, K., Aragam, B. & Ravikumar, P. "DAGMA: Learning DAGs via M-matrices and a log-determinant acyclicity characterization." *NeurIPS*, 2022.

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
