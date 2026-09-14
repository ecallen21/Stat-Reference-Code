# Exponential-Family Framework (Reference §47.397)

Fisher (1934), Koopman (1936), Pitman (1936). Unified
parametric family:

```
f(x; θ) = h(x) · exp(η(θ)ᵀ T(x) − A(η))
```

- `T(x)` — sufficient statistic
- `η` — natural parameter
- `A(η)` — log-partition (cumulant generating function).
  `A'(η) = E[T(X)]`, `A''(η) = Var[T(X)] = 1 / I(η)`.
- `h(x)` — base measure

Members: Normal (σ known), Bernoulli, Binomial, Poisson,
Gamma, Beta, Dirichlet, Categorical, Exponential,
Chi-square, Inverse-Gaussian, Tweedie EDM.

**Why it matters**: MLE is method-of-moments matching
`T̄(x) = A'(η̂)`; conjugate Bayesian priors have closed
form; Fisher information is `A''(η)`; GLMs plug the natural
parameter to a linear predictor.

## Files

- `python/exponential_family_framework.py` — verifies the
  mean-parameter identity `E[X] = A'(η)` for Bernoulli
  (η = logit, A' = sigmoid), Poisson (η = log λ, A' = exp),
  and Normal-σ-known (η = μ, A' = η). Confirms Cramér-Rao
  bound saturation for Bernoulli on n=10 000.
- `r/exponential_family_framework.R` — `stats::glm` with any
  `family` (R); `statsmodels.genmod.families`, from-scratch
  (Python).

## Where else it appears in this repo

- `bayesian-glms`, `bayesian-hierarchical-models`,
  `conjugate-priors` — Bayesian workflows leverage
  conjugacy in this family.
- `poisson-regression`, `gamma-regression`,
  `tweedie-glm-regression`, `beta-regression`,
  `dirichlet-regression`, `inverse-gaussian-glm`,
  `negative-binomial-regression`, `multinomial-logistic`,
  `probit-regression`, `zero-inflated-regression`,
  `hurdle-model` — GLM members.
- `fisher-information`, `information-geometry`,
  `information-criteria`, `maximum-entropy` — theoretical
  neighbours.
- `glm-diagnostics`, `overdispersion-tests`, `gee` — GLM
  inference cousins.
- `firth-logistic` — bias-reduction inside the family.

## Assumptions & caveats

- **Minimal / curved families** — canonical exponential
  families have `dim(η) = dim(T)`; curved families
  (e.g. Weibull with both shape and scale free) have fewer
  natural parameters than sufficient statistics.
- **Dispersion** — many GLM textbooks split `A(η) → A(η)/φ`
  with a dispersion parameter `φ`; Poisson has `φ = 1`,
  Normal has `φ = σ²`.
- **Sufficient statistic** — matches the LIKELIHOOD via
  the Neyman factorisation criterion; the summary all one
  needs from the data.
- **Fisher-Neyman theorem** — sufficient statistic `T(x)`
  is minimal when it partitions the likelihood coarsely.
- **Robustness** — MLE in exp families is efficient but
  not robust; a single outlier can shift `T̄`.

## Run

```
python techniques/exponential-family-framework/python/exponential_family_framework.py
Rscript techniques/exponential-family-framework/r/exponential_family_framework.R
```

**Refs:** Fisher, R.A. "Two new properties of mathematical likelihood." *Proc. Roy. Soc. A*, 144: 285-307, 1934; Koopman, B.O. "On distributions admitting a sufficient statistic." *Trans. Amer. Math. Soc.*, 39: 399-409, 1936; Brown, L.D. *Fundamentals of Statistical Exponential Families*, IMS Lecture Notes 9, 1986.

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
