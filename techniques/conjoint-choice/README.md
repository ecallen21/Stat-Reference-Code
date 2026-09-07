# Conjoint / Discrete Choice Experiment (Reference §36.7)

McFadden (1974), Train (2009). Respondents choose one alternative
from a menu of J options characterised by attributes. **Multinomial
logit (MNL)** yields part-worth utilities `β` from
random-utility maximisation:

```
U_ij  = X_ij^T β + ε_ij,          ε ~ Type-I extreme value
Pr(y_i = j) = exp(X_ij β) / Σ_k exp(X_ik β)
```

Estimated by MLE. Extensions: **mixed logit** for respondent-level
heterogeneity, **nested logit** for correlated alternatives.

## When to use

- **Marketing** — product design, willingness-to-pay,
  segmentation.
- **Transportation** — mode choice, route choice.
- **Health** — patient preference elicitation, benefit-risk
  trade-offs.

## When NOT to use

- **Non-choice outcomes** — ratings / rankings use different
  models.
- **IIA assumption grossly violated** — nested logit or mixed
  logit needed.

## Files

- `python/conjoint_choice.py` — MNL log-likelihood + BFGS
  optimisation. Demo (n=500, 3 alternatives, 3 attributes with
  true β=(−1.5, 1.0, 0.5)): estimated **β̂ = (−1.58, 0.97, 0.40)**;
  **McFadden pseudo-R² = 0.44**.
- `r/conjoint_choice.R` — `mlogit`, `apollo`, `ChoiceModelR`,
  `gmnl` (R); `xlogit`, `pylogit`, `biogeme`,
  `statsmodels.MNLogit` (Python).

## Assumptions & caveats

- **IIA** (Independence of Irrelevant Alternatives) — MNL red bus /
  blue bus problem; use nested or mixed logit.
- **Utility linearity in attributes** — extend with interactions
  or splines.
- **Alternative-specific constants** — always include unless the
  design is generic.
- **Heterogeneity** — mixed logit with random coefficients captures
  individual-level preferences.

## Related in this repo

- `mnl-multinomial-logit` (if present), `latent-class-analysis`
  (segmentation of preferences), `mixture-regression` — cousins.

## Run

```
python techniques/conjoint-choice/python/conjoint_choice.py
Rscript techniques/conjoint-choice/r/conjoint_choice.R
```

**Refs:** McFadden, D. "Conditional logit analysis of qualitative choice behavior." In *Frontiers in Econometrics*, 1974; Train, K.E. *Discrete Choice Methods with Simulation*, 2nd ed., Cambridge University Press, 2009.

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
