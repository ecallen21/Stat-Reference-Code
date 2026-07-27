# Nelson-Aalen Cumulative Hazard Estimator (Reference §11.3; also covers §11.65)

Non-parametric estimator of the **cumulative hazard** `H(t)` from right-censored data:

```
Ĥ(t)  =  Σ_{t_j ≤ t}  d_j / n_j
```

Related to Kaplan-Meier via `Ŝ(t) ≈ exp(−Ĥ(t))` — the two agree closely on real data and coincide in the continuous-time limit.

## Variance (Aalen 1978)

```
Var(Ĥ(t))  =  Σ_{t_j ≤ t}  d_j / n_j²
```

Log-transformed 95% CI (works well because H is positive):
```
CI: Ĥ(t) · exp( ± z · √Var(Ĥ) / Ĥ(t) )
```

## Hazard rate estimation (§11.65)

The instantaneous hazard `h(t)` is the derivative of `H(t)`. Since `Ĥ` is a step function, we kernel-smooth its jumps to estimate the hazard rate:

```
ĥ(t)  =  (1/b) · Σ_j  K((t − t_j) / b) · ΔĤ_j
        ΔĤ_j = 1 / n_j at event time t_j
```

Kernels supplied: Epanechnikov (default), Gaussian, uniform.

## When to prefer NA over KM

- Need cumulative hazard explicitly (e.g. for a Cox baseline hazard, or for parametric-model residuals).
- Small samples with censoring: NA is slightly less biased than KM in the tail.
- Otherwise both give the same story.

## Files

- `python/nelson_aalen.py` — NA + Aalen variance + log-CI + kernel-smoothed hazard rate.
- `r/nelson_aalen.R` — from-scratch + `survival::survfit(type = "fh")`.

## Assumptions

Same as Kaplan-Meier: independent right-censoring, no informative dropout.

## Run

```
python techniques/nelson-aalen/python/nelson_aalen.py
Rscript techniques/nelson-aalen/r/nelson_aalen.R
```

**Refs:** Nelson, W. "Theory and applications of hazard plotting for censored failure data." *Technometrics* 14(4), 945–966, 1972; Aalen, O. "Nonparametric inference for a family of counting processes." *Ann. Stat.* 6(4), 701–726, 1978; Klein, J.P. & Moeschberger, M.L. *Survival Analysis*, 2nd ed., Springer, 2003 (Ch. 4).

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
