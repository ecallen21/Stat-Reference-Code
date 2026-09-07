# Efron-Stein Inequality (Reference §46.12)

Efron & Stein (1981). Distribution-free bound on the variance of
any function `Z = f(X₁, …, X_n)` of independent random variables:

    Var(Z) ≤ ½ · Σᵢ E[(Z − Z_i')²]

where `Z_i' = f(X₁, …, X_{i−1}, X_i', X_{i+1}, …, X_n)` — resample
the `i`-th coordinate. Equivalent leave-one-out form:

    Var(Z) ≤ Σᵢ E[Var(Z | X_{−i})]

## Corollaries

- **Rademacher complexity concentration** — plugs directly into the
  bounded-difference argument to give `√(log(1/δ) / n)` fluctuations.
- **Bootstrap variance efficiency** — the bootstrap distribution
  estimates `Var(Z)` at rate `O(1/n)`.
- **Stability-based generalisation** — algorithm stability bounds
  are consequences via Efron-Stein.

## Files

- `python/efron_stein_inequality.py` — Monte-Carlo estimator of the
  ES bound and of the true variance for three functions. Demo (n=40,
  X ∼ N(0,1)): sample mean Var 0.024 vs bound 0.025 (nearly tight —
  the mean saturates ES); sample variance Var 0.047 vs bound 0.050;
  sample median Var 0.037 vs bound 0.057 (loose but valid).
- `r/efron_stein_inequality.R` — no CRAN package; describes the
  from-scratch approach and related bootstrap variance tooling.

## When to use

- **Proving concentration** for a data-driven statistic without
  knowing its distribution.
- **Justifying bootstrap variance** as an unbiased upper bound.
- **Deriving PAC bounds** for learning algorithms with bounded
  stability.

## When NOT to use

- **You want a matching lower bound** — ES is only an upper bound;
  use CLT + delta method for asymptotics.
- **Non-independent inputs** — the classical form needs iid; use
  martingale-difference / entropy-method extensions.

## Assumptions & caveats

- **Independence** of `X₁, …, X_n`.
- **Finite second moment** of `Z`.
- **Symmetric ES form** — pair `Z` with resampled `Z_i'`. The
  jackknife (leave-one-out) form is a different but equivalent
  bound.
- **Tight for sums** — `Var(ΣXᵢ) = Σ Var(Xᵢ)`; ES gives exactly
  this. For nonlinear `f` it is loose (see the median in the demo).

## Related in this repo

- `rademacher-complexity`, `vc-dimension` — capacity measures used
  jointly with ES to bound generalisation error.
- `jackknife`, `jackknife-plus`, `nonparametric-bootstrap` — the
  practical variance estimators ES underpins.
- `u-statistics` — ES also bounds `Var(U_n)`.

## Run

```
python techniques/efron-stein-inequality/python/efron_stein_inequality.py
Rscript techniques/efron-stein-inequality/r/efron_stein_inequality.R
```

**Refs:** Efron, B. & Stein, C. "The jackknife estimate of variance." *Annals of Statistics*, 9(3): 586-596, 1981; Boucheron, S., Lugosi, G. & Massart, P. *Concentration Inequalities: A Nonasymptotic Theory of Independence*, OUP, 2013.

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
