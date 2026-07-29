# Nonlinear Mixed-Effects Models (Reference §12.12)

Standard NLME:

```
y_{ij}  =  f(θ_i, t_{ij})  +  ε_{ij}
θ_i     =  θ_pop  +  b_i             b_i ~ N(0, D)
```

`f` is a **nonlinear** function — pharmacokinetic decay, logistic growth, Michaelis–Menten, four-parameter dose-response, etc. Each subject `i` has their own parameter vector `θ_i` drawn from a population distribution.

## Two-stage estimation (this file)

- **Stage 1**: fit per-subject nonlinear least-squares → `θ̂_i`.
- **Stage 2**: compute the population mean and covariance of the `θ̂_i`.

Transparent, single-purpose, works when each subject has enough observations to fit `f` alone. **Less efficient** than a joint one-stage NLME MLE that borrows strength across subjects (for that, use `nlme::nlme` in R).

## When two-stage is fine vs. when to switch

| Two-stage OK | Use joint NLME |
|---|---|
| Many observations per subject (≥ 2× # parameters) | Sparse per-subject data |
| Similar sampling schedules across subjects | Very unbalanced designs |
| Quick sanity check / exploratory | Publication / regulatory submission |

## Common `f` shipped in the file

- **`logistic_growth`**: `A / (1 + exp(−r · (t − m)))` — asymptote / rate / midpoint.
- **`exp_decay`**: `y₀ · exp(−k · t)` — initial value / decay rate.

## Files

- `python/nonlinear_mixed_effects.py` — two-stage NLME driver + logistic-growth + exponential-decay helpers. Recovers population parameters `[99.5, 0.61, 4.67]` vs. true `[100, 0.6, 5.0]` on the demo.
- `r/nonlinear_mixed_effects.R` — thin wrapper around `nlme::nlme()` (joint one-stage MLE, the authoritative approach).

## Assumptions

- **`f` correctly specifies the mean function** — misspecification is fatal.
- Random effects are normal; residuals independent and normal.
- Convergence sensitive to starting values — always try several.

## Run

```
python techniques/nonlinear-mixed-effects/python/nonlinear_mixed_effects.py
Rscript techniques/nonlinear-mixed-effects/r/nonlinear_mixed_effects.R
```

**Refs:** Lindstrom, M.J. & Bates, D.M. "Nonlinear mixed effects models for repeated measures data." *Biometrics* 46(3), 673–687, 1990; Davidian, M. & Giltinan, D.M. *Nonlinear Models for Repeated Measurement Data*, Chapman & Hall, 1995; Pinheiro, J.C. & Bates, D.M. *Mixed-Effects Models in S and S-PLUS*, Springer, 2000.

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
