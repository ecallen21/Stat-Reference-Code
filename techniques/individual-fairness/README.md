# Individual Fairness (Reference Ch 31 Fairness)

**Similar individuals should get similar predictions.** Dwork, Hardt,
Pitassi, Reingold & Zemel (2012) — "Fairness Through Awareness." A
Lipschitz constraint on the classifier under a task-specific similarity
metric.

## Definition

```
d_pred( f(x), f(x') )  ≤  L · d_task( x, x' )     for all pairs (x, x').
```

- `d_task` — a **task metric** on the input space that encodes "these
  two individuals are equally deserving of the same prediction". Must
  ignore protected-attribute proxies.
- `d_pred` — a distance on predictions (usually `|·|`).
- `L` — the Lipschitz constant; smaller `L` means stronger fairness.

## Diagnostic

```
IF_loss  =  𝔼_{i, j} [ max( 0,  d_pred(f_i, f_j) − L · d_task(x_i, x_j) )² ]
```

Zero ⇒ Lipschitz w.r.t. `d_task` at constant `L`.

## Enforcement

Add the IF-loss as a soft penalty during training (Yurochkin 2020 SenSR,
John-Vempala 2020):

```
L_total  =  L_task(y, ŷ)  +  λ · IF_loss( ŷ, X; d_task, L )
```

Or use **projection-onto-Lipschitz-ball** for a hard constraint.

## When to use

- **"Case-by-case" fairness** — bar exam, medical triage, criminal
  sentencing, admissions — where the harm story is per-individual.
- **A trusted task metric exists** — a domain expert can articulate
  "these two applicants deserve the same outcome".
- **Complement to group fairness** — passing DP or EO does not imply
  IF; audit both.

## When NOT to use

- **No plausible `d_task`** — the criterion collapses.
- **Feature space is nominal / categorical** — hard to define a
  meaningful metric.
- **Adversarially-crafted task metric** — a bad `d_task` can *legalise*
  discrimination by including the protected attribute.

## Files

- `python/individual_fairness.py` — from-scratch: IF-loss diagnostic;
  pair-wise Lipschitz penalty added to logistic-regression training.
  Task metric = Euclidean distance on the two *informative* features
  only, ignoring a synthetic spurious feature. Result: **ERM IF-loss
  0.0048** → **IF-trained 0.0028** (−42 %) with **accuracy preserved
  at 0.92**; spurious coefficient shrinks −0.088 → −0.052.
- `r/individual_fairness.R` — `fairness` / `fairmodels` (R);
  Python `aif360.algorithms.inprocessing.PrejudiceRemover` /
  `sen-fair-consistency` / `fairtorch`.

## Assumptions & caveats

- **`d_task` is the whole ballgame** — the criterion is only as good as
  the task metric.
- **Sampling pairs** — the demo samples `n_pairs` random pairs; larger
  scales need locality-sensitive hashing or nearest-neighbour lookups.
- **Interaction with group fairness** — IF and DP can be complementary
  but sometimes conflict; audit both, don't assume one implies the other.
- **Adversary awareness** — a task metric that itself encodes protected
  attributes launders discrimination through fairness.
- **Robust variants** (Yurochkin 2020 SenSR) tune `d_task` from data +
  a "sensitive subspace" instead of hand-coding it.

## Related in this repo

- `demographic-parity`, `equalized-odds`, `equal-opportunity`,
  `calibration-parity` — group-level fairness alternatives.
- `counterfactual-fairness` — a causal-graph flavour.
- `jacobian-regularization`, `spectral-normalization` — smoothness
  penalties that give Lipschitz control for a global `d_task = ‖·‖_2`.
- `fair-representations-lfr` — pre-processing that helps individual
  fairness when the metric downweights removed directions.

## Run

```
python techniques/individual-fairness/python/individual_fairness.py
Rscript techniques/individual-fairness/r/individual_fairness.R
```

**Refs:** Dwork, C., Hardt, M., Pitassi, T., Reingold, O. & Zemel, R. "Fairness through awareness." *ITCS*, 2012; Yurochkin, M., Bower, A. & Sun, Y. "Training individually fair ML models with sensitive subspace robustness (SenSR)." *ICLR*, 2020; John, P.G., Vempala, S. & Vaidya, R. "Verifying individual fairness in machine learning models." *UAI*, 2020.

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
