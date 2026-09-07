# Causal Forest -- Generalized Random Forests (Reference §15.37)

Athey, Tibshirani & Wager (2019). A random-forest generalisation that
estimates the **conditional average treatment effect (CATE)**
`τ(x) = E[Y(1) − Y(0) | X = x]` with valid pointwise confidence
intervals.

## Key ideas

- **Honest splitting.** Split-selection and leaf-estimation samples
  are disjoint — prevents overfitting bias and enables valid CIs.
- **Heterogeneity-targeted splits.** Split rule maximises
  `(τ_left − τ_right)² · n_left · n_right / n` rather than outcome
  variance.
- **Local moment equations.** For a query `x`, weights `α_i(x)`
  from tree co-occurrence solve
  `Σ α_i(x)·(Tᵢ − ê(x))·(Yᵢ − m̂(x) − (Tᵢ − ê(x))·τ(x)) = 0`.
- **R-learner / doubly-robust form.** First-stage nuisance
  models `m(x) = E[Y|X]` and `e(x) = P(T|X)` are partialled out,
  then `τ(x)` from residual-on-residual regression.

## Files

- `python/causal_forest.py` — compact R-forest from scratch:
  cross-fitted residuals + heterogeneity-driven tree ensemble +
  weighted moment estimator. Demo (n=800, τ(x)=1+2·x0):
  x0 = −1.5→τ̂=−0.7, 0.0→τ̂=+1.1, +1.5→τ̂=+2.0 (attenuated at
  extremes as expected with a shallow demo forest). Real GRF uses
  RF-based nuisance, honest splitting, and analytic variance.
- `r/causal_forest.R` — `grf::causal_forest`, `grf::instrumental_forest`,
  `policytree` (R); `econml.grf.CausalForest`, `econml.dml.CausalForestDML`
  (Python).

## When to use

- **Heterogeneous treatment effects (HTE)** — you want τ(x), not a
  single ATE.
- **Personalised medicine / policy targeting** — who benefits most /
  least.
- **Nonlinear moderators** — the forest captures interactions without
  a specified functional form.

## When NOT to use

- **You only need the ATE** — use IPW / AIPW / DML; the forest adds
  variance without benefit.
- **Very small n** — CATE estimation needs a lot of data; τ(x)
  variance grows quickly with p.
- **Strong extrapolation** — τ(x) at the covariate boundary is
  effectively an average of neighbours; report only within the
  support.

## Assumptions & caveats

- **Unconfoundedness** and **overlap** — same as AIPW / DML.
- **Honesty** — subsampling to separate splitting and estimation is
  what makes CIs valid; do not skip it.
- **Tuning matters** — `min_node_size`, `sample_fraction`, and
  `honesty` fraction change bias-variance.
- **Multiple testing** — reporting τ̂(x) for every x is a whole
  function; use `grf::test_calibration` or best-linear projections
  for interpretable summaries.

## Related in this repo

- `iptw`, `aipw-doubly-robust`, `dml-double-ml` — ATE counterparts.
- `hte-uplift`, `meta-learning-maml` — HTE alternatives.
- `random-forest`, `gradient-boosting` — the non-causal siblings.

## Run

```
python techniques/causal-forest/python/causal_forest.py
Rscript techniques/causal-forest/r/causal_forest.R
```

**Refs:** Athey, S., Tibshirani, J. & Wager, S. "Generalized random forests." *Annals of Statistics*, 47(2): 1148-1178, 2019; Wager, S. & Athey, S. "Estimation and inference of heterogeneous treatment effects using random forests." *JASA*, 113(523): 1228-1242, 2018.

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
