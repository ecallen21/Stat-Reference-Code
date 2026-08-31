# Empirical Likelihood (Reference §33.5)

Owen (1988) — a **nonparametric analogue of the parametric likelihood
ratio test**. Assign probabilities `p_i` to the sample points and
maximise `∏ p_i` subject to `Σ p_i = 1` and a moment constraint
`Σ p_i g(Y_i, θ) = 0` that identifies the target parameter.

## Formula (for the mean)

```
Maximise    ∏ p_i
s.t.        p_i ≥ 0,   Σ p_i = 1,   Σ p_i (Y_i − θ) = 0.
```

Owen's theorem:

```
−2 log R(θ)  =  2 Σ log(1 + λ (Y_i − θ))    →    χ²(1)    under H_0
```

with `λ` solving `Σ (Y_i − θ) / (1 + λ (Y_i − θ)) = 0`. A `(1 − α)` CI is

```
{ θ : −2 log R(θ) ≤ χ²_{1, 1-α} }.
```

## Advantages over the t interval

- **Data-driven shape**: skew and kurtosis reflect in CI asymmetry.
- **No sd or normality assumption**.
- **Range-preserving**: never leaves `[y_min, y_max]`.
- Generalises to regression, quantiles, survival, U-statistics
  (Qin-Lawless 1994, Kolaczyk 1994).

## When to use

- **Small n + skewed data** where the t interval mis-covers.
- **Multiple estimating equations** — GMM-EL provides asymptotically
  efficient estimators.
- **Constrained inference** — impose known moments to sharpen CI.

## When NOT to use

- **Large n + light tails** — the t interval is easier and comparable.
- **Very small n (< 15)** — chi-square approximation lags; use Bartlett
  correction (DiCiccio-Hall-Romano 1991).

## Files

- `python/empirical_likelihood.py` — from-scratch EL for `E[Y]` with
  Newton-Raphson on `λ` and bisection root-finding for the CI. Demo
  on `n=60` log-normal data: EL 95 % CI [1.14, 1.66] (width 0.52) vs
  t 95 % CI [1.10, 1.63] (width 0.53). Coverage over 200 trials:
  EL 0.875, t 0.865 — both under-cover for skewed data with modest n,
  as expected.
- `r/empirical_likelihood.R` — `emplik`, `survELtest`, `ELYP` (R);
  `statsmodels.emplike`, `empirical-likelihood` (Python).

## Assumptions & caveats

- **Chi-square approximation** — improved by Bartlett correction for
  small n.
- **Numerical caveat** — `1 + λ(y − θ)` must stay positive; Owen's
  constrained Newton with backtracking is standard.
- **Range restriction** — the CI cannot extend beyond `[y_min, y_max]`,
  which is safer than the t CI's possibility of covering impossible
  values.
- **Empirical vs profile EL** — profile EL profiles out nuisance
  parameters; more efficient when the model has structure.
- **Higher-order terms**: EL is second-order correct (Hall 1990) and
  self-Bartlett-adjusting.

## Related in this repo

- `jackknife`, `jackknife-plus`, `bca-bootstrap`, `block-bootstrap` —
  other nonparametric CI methods.
- `bayesian-linear-regression`, `bayesian-glms` — parametric Bayesian
  alternatives.
- `sandwich-robust-se` — robust asymptotic CI for regression.
- `semiparametric-efficiency` — EL attains the semiparametric
  efficiency bound.

## Run

```
python techniques/empirical-likelihood/python/empirical_likelihood.py
Rscript techniques/empirical-likelihood/r/empirical_likelihood.R
```

**Refs:** Owen, A.B. "Empirical likelihood ratio confidence intervals for a single functional." *Biometrika*, 1988; Owen, A.B. *Empirical Likelihood*, CRC, 2001; Qin, J. & Lawless, J. "Empirical likelihood and general estimating equations." *Annals of Statistics*, 1994.

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
