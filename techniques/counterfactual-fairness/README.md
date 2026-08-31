# Counterfactual Fairness (Reference Ch 31 Fairness)

**Would this prediction change if the protected attribute were
different, holding the individual's exogenous latents constant?**
Kusner, Loftus, Russell & Silva (2017) — the causal-inference lens on
fairness.

## Definition

Given a structural causal model (SCM) with exogenous variables `U` and
observed `(A, X, Y)`, a predictor `Ŷ` is **counterfactually fair** iff

```
P( Ŷ_{A ← a}(U) = y | X = x, A = a )
  =
P( Ŷ_{A ← a'}(U) = y | X = x, A = a )
```

for every valid counterfactual `a' ≠ a`. In words: swapping `A` in the
individual's SCM should not change the predicted distribution.

## Kusner's three-level recipe

- **Level 1** — use only ancestors of `Y` that are non-descendants of `A`.
- **Level 2** — infer `U | (X, A)` and predict from `U` only. Fair by
  construction because `Ŷ = g(U)` never depends on `A`.
- **Level 3** — additive-noise SCM; use `X − f_A(A)` residuals as
  features.

## When to use

- **A causal narrative is available** — protected attributes with a
  well-understood causal role (race, sex, age).
- **Legal / medical / hiring** contexts where "would this person be
  treated the same in a counterfactual world" is the operative fairness
  intuition.
- **Downstream individual audit** — can compute per-example
  counterfactual gaps.

## When NOT to use

- **No causal model available** — falls back to correlational fairness
  metrics.
- **A cannot be counterfactually manipulated** (e.g. race, ethnicity)
  — the philosophical validity of the counterfactual is debated
  (Kohler-Hausmann 2019).
- **Model misspecification** — a wrong SCM gives a wrong "counterfactual
  fair" prediction that can be worse than a purely statistical one.

## Files

- `python/counterfactual_fairness.py` — from-scratch: SCM `X = α A + U`
  fitted by OLS (recovered α = 1.96 vs true 2.0); logistic-regression
  predictor from `(X, A)` (naive) vs from `U` alone (Kusner Level 2).
  **Counterfactual flip gap**: naive 0.141 → CF-fair 0.000 (by
  construction); accuracy trade-off 0.953 → 0.916.
- `r/counterfactual_fairness.R` — `dagitty` / `bnlearn` (native R);
  `doWhy` / `pyro` (Python) for the SCM step.

## Assumptions & caveats

- **SCM must be given** — model misspecification propagates to unfair
  "counterfactually fair" predictions.
- **Latent U inference is hard** — additive Gaussian noise is a strong
  simplifying assumption; nonparametric SCMs need MCMC / VI.
- **Descendant features** cannot enter the predictor at all — throws
  out potentially useful signal.
- **Path-specific fairness** (Nabi-Shpitser 2018) generalises the
  criterion when SOME causal paths through A are considered acceptable
  ("business necessity").
- **Aggregate fairness metrics can still fail** even if every
  individual's counterfactual gap is 0 — audit both.

## Related in this repo

- `causal-inference`, `dag-inference`, `do-calculus` (if present) —
  the SCM machinery.
- `fair-representations-lfr` — non-causal fair-encoding alternatives.
- `equalized-odds`, `demographic-parity`, `calibration-parity` —
  correlational fairness criteria.
- `individual-fairness` — a distinct "similar individuals" notion.
- `propensity-score-methods`, `tmle-doubly-robust` — closely related
  causal-inference toolkits.

## Run

```
python techniques/counterfactual-fairness/python/counterfactual_fairness.py
Rscript techniques/counterfactual-fairness/r/counterfactual_fairness.R
```

**Refs:** Kusner, M.J., Loftus, J.R., Russell, C. & Silva, R. "Counterfactual fairness." *NeurIPS*, 2017; Nabi, R. & Shpitser, I. "Fair inference on outcomes." *AAAI*, 2018; Kohler-Hausmann, I. "Eddie Murphy and the dangers of counterfactual causal thinking about detecting racial discrimination." *Northwestern University Law Review*, 2019.

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
