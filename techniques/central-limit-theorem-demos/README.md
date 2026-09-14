# Central Limit Theorem — Demonstrations (Reference §47.400)

Laplace (1810); Lyapunov (1901); Lindeberg (1922); Feller
(1935). Classical CLT: if `X_1, …, X_n` iid with finite mean
`μ` and variance `σ²`,

```
√n (X̄ − μ) / σ  →  N(0, 1)  as n → ∞
```

Berry-Esseen tightens convergence to `O(1/√n)` when the third
absolute moment is finite. Generalised CLT (Gnedenko-
Kolmogorov 1954): sums of iid heavy-tail rvs converge to
α-STABLE distributions, NOT Gaussian, when variance is
infinite.

## Files

- `python/central_limit_theorem_demos.py` — Four demos:
  1. Exponential(1) sample-mean skewness shrinks from 2.06
     at n=1 to 0.13 at n=300 (Berry-Esseen O(1/√n) rate);
     `E[Z]` and `sd(Z)` hit `(0, 1)` from n=5 onward.
  2. Berry-Esseen bound illustration.
  3. Cauchy sample-mean does NOT converge; range stays huge
     (−1158 to +6507 for n=100).
  4. Multivariate CLT: variance of the log-normal sample
     mean matches `σ² / n` to 4 decimals for n=300.
- `r/central_limit_theorem_demos.R` — base R sampling +
  `moments::skewness`; `numpy.random`, `scipy.stats.skew`,
  from-scratch (Python).

## Where else it appears in this repo

- `multivariate-normal-distribution` — the CLT LIMIT.
- `delta-method` — first-order CLT for smooth transforms.
- `bootstrap`, `nonparametric-bootstrap`, `parametric-bootstrap`,
  `wild-bootstrap`, `bca-bootstrap`, `block-bootstrap` —
  bootstrap CLT / Edgeworth expansions.
- `stable-distributions`, `cauchy-distribution` —
  generalised CLT counterpart.
- `fisher-information`, `wald-lrt-score` — asymptotic-normal
  inference relies on CLT.
- `bayesian-linear-regression`, `bayesian-hierarchical-models`
  — Bernstein-von Mises (Bayesian CLT).
- `sequential-analysis`, `wald-sprt` — CLT underlies the
  test-statistic asymptotics.

## Assumptions & caveats

- **Finite second moment** — mandatory for classical CLT;
  Cauchy fails.
- **Berry-Esseen** — non-uniform bound depends on third
  moment; sharper than sup-norm classical CLT.
- **Lindeberg-Feller** — extends to non-identically-
  distributed summands under a mild uniform-variance
  condition.
- **Convergence rate** — `O(n^(−1/2))` in KS distance; can
  be slower in the tails (Cramér moderate deviations).
- **Multivariate CLT** — same rate; requires finite
  covariance matrix.
- **Generalised CLT** — when second moment is infinite,
  scaled sums converge to α-stable, not Gaussian.

## Run

```
python techniques/central-limit-theorem-demos/python/central_limit_theorem_demos.py
Rscript techniques/central-limit-theorem-demos/r/central_limit_theorem_demos.R
```

**Refs:** Laplace, P.S. *Théorie analytique des probabilités*, 1810; Lyapunov, A.M. "Nouvelle forme du théorème sur la limite de probabilité." *Mém. Acad. Sci. St.-Petersburg*, 12: 1-24, 1901; Feller, W. "Über den zentralen Grenzwertsatz der Wahrscheinlichkeitsrechnung." *Math. Z.*, 40: 521-559, 1935; Gnedenko, B.V. and Kolmogorov, A.N. *Limit Distributions for Sums of Independent Random Variables*, Addison-Wesley, 1954.

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
