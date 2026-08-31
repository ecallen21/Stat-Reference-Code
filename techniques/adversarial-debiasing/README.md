# Adversarial Debiasing (Reference Ch 31 Fairness)

Two-player game between a **predictor** and an **adversary** that tries
to recover the protected attribute from the predictor's output. Zhang,
Lemoine & Mitchell (2018) — the in-training fairness method that
underpins AIF360's default recipe.

## Objective

```
Predictor f_θ(x)          → ŷ
Adversary g_φ(ŷ)          → Â

L_pred  =  L_task(ŷ, y)  −  α · L_adv(Â, A)      (fool the adversary)
L_adv   =  L_bce(Â, A)                            (detect A from ŷ)
```

Alternating SGD; at optimum, `ŷ` carries **no information about A** —
which under mild conditions implies demographic parity (Zhang 2018 Prop 1).

## Two variants in the paper

- **Plain**: gradient descent on `L_pred`.
- **Gradient projection**: project the adversary's gradient onto the
  space orthogonal to the predictor's, so the fairness term only
  contributes to directions the task loss doesn't care about.

## When to use

- **Deep predictors** with rich features — adversarial pressure prevents
  the network from proxying the protected attribute internally.
- **You control training** — needs joint optimisation, not post-hoc.
- **Demographic parity** target — adversarial debiasing on `ŷ` alone
  approximates DP; conditioning on `y` targets equalized odds.

## When NOT to use

- **Very small groups** — the adversary is data-hungry; a weak
  adversary is worse than none.
- **Post-hoc audit only** — use a post-processor instead.
- **Tabular models with few features** — reweighing is usually enough
  and cheaper.

## Files

- `python/adversarial_debiasing.py` — from-scratch: logistic-regression
  predictor + logistic-regression adversary that reads only the
  predictor's sigmoid output. Alternating SGD on synthetic two-group
  data with a proxy feature: **ERM DP ratio 0.36 → α = 3 DP ratio
  0.97** at a 6-pt accuracy cost.
- `r/adversarial_debiasing.R` — `reticulate` + `aif360` /
  `fairtorch`; native R `torch` port for a manual two-player loop.

## Assumptions & caveats

- **α is delicate** — small α does nothing; huge α destroys accuracy.
  Grid over {0.5, 1, 3, 10} and pick from a Pareto front.
- **Adversary capacity** — a weak adversary hides bias in a strong
  predictor; use an adversary at least as expressive as the predictor.
- **Instability** — GAN-like training. Use gradient projection (Zhang
  eq 5) and/or a scheduler on α to stabilise.
- **Target choice**:
  - Adversary reads `ŷ` alone → approximates **DP**.
  - Adversary reads `(ŷ, y)` → approximates **equalized odds**.
- **Not certified** — no formal fairness guarantee; report post-hoc
  metrics after training.

## Related in this repo

- `reweighing-preprocessing` — pre-training alternative.
- `exponentiated-gradient-reduction` — in-training with fairness
  constraints and a formal Pareto guarantee.
- `equalized-odds-postprocessing` — post-hoc alternative.
- `demographic-parity`, `equalized-odds`, `equal-opportunity` — the
  criteria this method targets.

## Run

```
python techniques/adversarial-debiasing/python/adversarial_debiasing.py
Rscript techniques/adversarial-debiasing/r/adversarial_debiasing.R
```

**Refs:** Zhang, B.H., Lemoine, B. & Mitchell, M. "Mitigating unwanted biases with adversarial learning." *AIES*, 2018; Edwards, H. & Storkey, A. "Censoring representations with an adversary." *ICLR*, 2016.

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
