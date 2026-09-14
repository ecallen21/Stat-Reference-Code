# Alpha-Stable Distributions (Reference §47.394)

Lévy (1925); characteristic-function family CLOSED under
linear combinations:

```
φ(t) = exp(i μ t − |c t|^α · (1 − i β sign(t) ω(t, α)))
```

Parameters: `α ∈ (0, 2]` tail (`2` = Gaussian, `1` = Cauchy),
`β ∈ [−1, 1]` skew, `c > 0` scale, `μ` location. For `α < 2`
the variance is infinite. Chambers-Mallows-Stuck (1976) is
the standard sampler for any `α`.

Special cases:

- `α = 2, β = 0` → Gaussian
- `α = 1, β = 0` → Cauchy
- `α = 0.5, β = 1, μ = 0` → Lévy (one-sided)

## Files

- `python/stable_distributions.py` — Chambers-Mallows-Stuck
  sampler, n=20 000 draws. 97.5-percentile grows from 2.7
  (α=2, Gaussian) to 4.5 (α=1.5), 12.7 (α=1, Cauchy), and
  236 (α=0.5, Lévy) — the tail exploding as α decreases.
- `r/stable_distributions.R` — `stabledist::rstable`,
  `libstableR` (R); `scipy.stats.levy_stable`, from-scratch
  (Python).

## Where else it appears in this repo

- `cauchy-distribution` — `α = 1` special case.
- `extreme-value-theory` — related max-stable framework.
- `garch`, `stochastic-volatility`, `cvar-expected-shortfall`
  — heavy-tail financial modelling (many practitioners fit
  `α ≈ 1.5` to log-returns).
- `robust-regression`, `tukey-biweight-m-estimator`,
  `huber-m-estimator` — robust estimators must survive
  α-stable tails.
- `hawkes-process`, `poisson-point-process-inhomog` —
  event-time models can carry α-stable inter-arrivals.

## Assumptions & caveats

- **No closed-form PDF/CDF** for general α — computation via
  Zolotarev integral or Nolan's inversion.
- **MLE / MoM** — numerical; McCulloch's quantile method is a
  fast rough estimator, MLE (Nolan 1997) is more precise.
- **Parameterisation** — Nolan's `S0` (continuous in α) vs
  `S1` (traditional); different software packages use
  different conventions — always check.
- **Symmetric stable at α = 2** — variance IS defined; Cauchy
  at α = 1 is the tipping point for finite mean.
- **CLT generalisation** — sums of iid heavy-tail rvs
  converge to α-stable (Generalized Central Limit Theorem),
  not to Gaussian, when variance is infinite.

## Run

```
python techniques/stable-distributions/python/stable_distributions.py
Rscript techniques/stable-distributions/r/stable_distributions.R
```

**Refs:** Lévy, P. *Calcul des probabilités*, Gauthier-Villars, 1925; Chambers, J.M., Mallows, C.L. and Stuck, B.W. "A method for simulating stable random variables." *J. Amer. Statist. Assoc.*, 71: 340-344, 1976; Nolan, J.P. *Stable Distributions — Models for Heavy Tailed Data*, Springer, 2020.

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
