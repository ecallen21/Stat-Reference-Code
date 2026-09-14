# Weibull Distribution (Reference §47.389)

Weibull (1939, 1951). Two-parameter continuous distribution on
`[0, ∞)` with PDF

```
f(x; k, λ) = (k/λ) (x/λ)^{k−1} exp(−(x/λ)^k)
```

Shape `k` controls the hazard function:

| `k`  | Hazard          | Interpretation           |
|:----:|:---------------:|:-------------------------|
| < 1  | decreasing      | infant mortality         |
| = 1  | constant        | exponential (memoryless) |
| > 1  | increasing      | wear-out / ageing        |

Foundational for reliability engineering, survival analysis,
and extreme-value block-maxima limits.

## Files

- `python/weibull_distribution.py` — inverse-CDF sampling,
  Weibull PDF, MLE on shape via Newton on the profile
  log-likelihood, closed-form scale given shape.
  Recovers `(k, λ)` from n=1 000 to 2 decimals for
  `(0.7, 2.0), (1.5, 3.0), (3.0, 4.0)`.
- `r/weibull_distribution.R` — `stats::dweibull` /
  `rweibull` / `pweibull`, `MASS::fitdistr`,
  `fitdistrplus::fitdist` (R); `scipy.stats.weibull_min`,
  from-scratch (Python).

## Where else it appears in this repo

- `parametric-survival` — Weibull is a standard baseline
  parametric survival model.
- `accelerated-failure-time`, `buckley-james-aft` — AFT
  with Weibull baseline.
- `deep-survival-network` — DL survival with Weibull /
  log-Normal / log-Logistic outputs.
- `extreme-value-theory` — Weibull is one of the three
  limiting max-stable distributions in the Fisher-Tippett-
  Gnedenko theorem.
- `piecewise-exponential-model` — closely related hazard
  model (constant hazard within pieces).

## Assumptions & caveats

- **Shape identifiability** — needs `n ≥ 30` for stable MLE;
  small samples benefit from Bayesian priors.
- **Location parameter** — 3-parameter Weibull adds a
  location shift; harder to fit.
- **Type-I / Type-II censoring** — MLE straightforward with
  contribution `S(t)` for censored observations.
- **Right-tail** — Weibull tail is heavier than Gaussian for
  `k < 3.5` and lighter for `k > 3.5`.

## Run

```
python techniques/weibull-distribution/python/weibull_distribution.py
Rscript techniques/weibull-distribution/r/weibull_distribution.R
```

**Refs:** Weibull, W. "A statistical theory of the strength of materials." *Proc. Royal Swedish Academy of Engineering Sciences*, 151: 1-45, 1939; Weibull, W. "A statistical distribution function of wide applicability." *J. Appl. Mech.*, 18: 293-297, 1951.

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
