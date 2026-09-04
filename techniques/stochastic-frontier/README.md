# Stochastic Frontier Analysis (Reference §35.22)

Aigner, Lovell & Schmidt (1977). Estimate **firm-level efficiency**
via a production frontier with a **composed error**:

```
y_i = x_iᵀ β + v_i − u_i
v_i ~ N(0, σ_v²)       (symmetric noise)
u_i ~ N⁺(0, σ_u²)      (half-normal INEFFICIENCY, u_i ≥ 0)
```

Composite error `e_i = v_i − u_i` has a **normal-half-normal**
density.

## Firm-specific efficiency (Jondrow 1982)

```
𝔼[u_i | e_i] = σ* · ( φ(μ*_i / σ*) / (1 − Φ(−μ*_i / σ*)) − μ*_i / σ* )
σ* = σ_u σ_v / σ
μ*_i = − e_i σ_u² / σ²
```

Technical-efficiency score: **`TE_i = exp(−u_i)` ∈ (0, 1]**; higher =
more efficient.

## When to use

- **Firm / farm / hospital productivity** analysis.
- **Cost / profit frontiers** (u_i now added, not subtracted).
- **Explaining efficiency** with covariates via the Battese-Coelli
  1995 extension.

## When NOT to use

- **Individual-level noisy production** with poor measurement.
- **Panel data** — use time-varying-inefficiency variants
  (Kumbhakar-Lovell 2000).
- **Very heavy-tailed noise** — the normal-half-normal assumption is
  restrictive.

## Files

- `python/stochastic_frontier.py` — from-scratch MLE via
  `scipy.optimize`. Demo `n = 300`, true `β = (2.0, 0.5, 0.3)`,
  `σ_v = 0.3`, `σ_u ≈ 0.7`.
  Result: **β_hat = (1.97, 0.52, 0.34), σ_v = 0.275, σ_u = 0.660,
  λ = 2.40**; TE scores range 0.16 – 0.90 with mean 0.62.
- `r/stochastic_frontier.R` — `frontier`, `Benchmarking`, `sfaR` (R);
  `pysfa` (Python).

## Assumptions & caveats

- **Distributional choice** — half-normal is standard; exponential,
  truncated-normal, gamma also common.
- **Skew of composite error** — should be negative (`skew(e) < 0`);
  positive skew suggests wrong sign or misspecification.
- **Firm-level efficiency** noisy for small n.
- **Panel extensions**: Kumbhakar 2000 time-invariant vs time-varying;
  true-FE (Greene 2005) with unit dummies.
- **DEA alternative** — non-parametric; sensitive to outliers.

## Related in this repo

- `fixed-effects-panel` — related panel machinery.
- `heckman-selection` — sibling MLE-based econometric method.
- `gmm-general`, `nonlinear-least-squares` — sibling estimators.
- `stochastic-optimization` (if present) — related name but different
  concept.

## Run

```
python techniques/stochastic-frontier/python/stochastic_frontier.py
Rscript techniques/stochastic-frontier/r/stochastic_frontier.R
```

**Refs:** Aigner, D., Lovell, C.A.K. & Schmidt, P. "Formulation and estimation of stochastic frontier production function models." *Journal of Econometrics*, 1977; Kumbhakar, S.C. & Lovell, C.A.K. *Stochastic Frontier Analysis*, Cambridge University Press, 2000.

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
