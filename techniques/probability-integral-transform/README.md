# Probability Integral Transform (Reference §47.399)

Two identities that undergird copulas, calibration
diagnostics, and virtually every random-variable simulator:

1. **PIT**: if `X` has continuous CDF `F`, then `U = F(X)`
   is `Uniform(0, 1)`.
2. **Inverse-CDF sampling**: if `U ~ Uniform(0, 1)` and `F`
   is any CDF, then `X = F⁻¹(U)` has CDF `F`.

## Files

- `python/probability_integral_transform.py` — Three
  demonstrations. (1) PIT of `Exp(1)` produces samples with
  mean 0.507 and sd 0.288 (target Uniform(0,1)). (2) Inverse-
  CDF sampling for `Weibull(1.5, 2)` gives sample mean 1.79
  (theory `2·Γ(1 + 1/1.5) = 1.805`). (3) Gaussian-copula
  construction: MVN → Φ → Uniform → target marginals, giving
  empirical `Corr = 0.57` under `ρ_Gaussian = 0.6`.
- `r/probability_integral_transform.R` — base R
  `pnorm/qnorm/pexp/qexp/pweibull/qweibull` (R);
  `scipy.stats.<dist>.cdf/.ppf`, `scipy.special.ndtr`,
  from-scratch (Python).

## Where else it appears in this repo

- `copulas` — Sklar's theorem is a direct PIT construction;
  every copula method starts here.
- `latin-hypercube-sampling`, `quasi-monte-carlo-sobol` —
  low-discrepancy uniform points fed through `F⁻¹` for
  variance-reduced Monte Carlo.
- `rejection-sampling`, `importance-sampling` —
  inverse-CDF as the first line of defence.
- `calibration-plots`, `platt-scaling`,
  `histogram-binning-calibration`,
  `expected-calibration-error` — miscalibrated forecast CDFs
  fail the PIT-uniformity test.
- `proper-scoring-rules-crps` — CRPS decomposition uses PIT.
- `evidential-deep-learning`, `bayesian-neural-network` — PIT
  histograms are standard uncertainty-quantification checks.

## Assumptions & caveats

- **Continuity of F** — PIT is Uniform only for continuous
  targets; discrete targets give randomised-PIT.
- **Invertibility** — `F⁻¹` closed-form for Exp / Weibull /
  Cauchy / logistic; numerical (root-solve) for Normal
  (though `scipy.special.ndtri` is cheap).
- **PIT calibration test** — Kolmogorov-Smirnov,
  Anderson-Darling, or a simple histogram-uniformity chi²
  test all work.
- **Copula ρ vs Kendall's τ** — Gaussian copula's `ρ`
  parameter is not equal to the Pearson correlation of the
  transformed marginals; watch the calibration.

## Run

```
python techniques/probability-integral-transform/python/probability_integral_transform.py
Rscript techniques/probability-integral-transform/r/probability_integral_transform.R
```

**Refs:** Fisher, R.A. "On the mathematical foundations of theoretical statistics." *Philos. Trans. Roy. Soc. A*, 222: 309-368, 1922; Rosenblatt, M. "Remarks on a multivariate transformation." *Ann. Math. Statist.*, 23: 470-472, 1952.

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
