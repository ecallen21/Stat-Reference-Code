# Parametric AFT Survival Models (Reference §11.10–§11.15; also covers §11.44, §11.58)

Fully-parametric survival regression via the **Accelerated Failure Time** form:

```
log T  =  X·β  +  σ·W
```

`W` is a family-specific noise term:

| Family (Ref §) | `W` distribution | Notes |
|---|---|---|
| Exponential (§11.10) | Gumbel; `σ = 1` fixed | Constant hazard |
| Weibull (§11.11) | Gumbel; `σ` free | Monotone hazard (shape = 1/σ) |
| Log-normal (§11.13) | `N(0, 1)` | Non-monotone hazard (initially rises then falls) |
| Log-logistic (§11.12) | Logistic | Non-monotone, similar shape to log-normal |
| Generalized gamma (§11.14) | Extra shape parameter | Nests all above — use for shape-comparison LR tests |

Fit by MLE on the observed-data log-likelihood with censoring:
```
ll_i  =  log f(t_i | X_i, β, σ)   if event
         log S(t_i | X_i, β, σ)   if right-censored
```

## AFT interpretation

`exp(β_k)` is a **time-ratio** — if `X_k` increases by 1, the survival time is *multiplied* by `exp(β_k)`. Positive β_k = longer survival. Contrast with Cox where `exp(β)` is a hazard ratio.

## Piecewise-exponential (§11.44)

Partition time into K intervals; fit a constant hazard per interval. Equivalent to a Poisson GLM on person-time. Flexible baseline hazard without committing to a parametric family.

## Weibull reliability plot (§11.58)

Weibull is the classical reliability model. `log(−log Ŝ(t))` vs. `log t` should be a **straight line** with slope = shape. Non-linearity flags Weibull as a poor fit.

## Files

- `python/parametric_survival.py` — MLE for exponential / Weibull / log-normal / log-logistic AFT (unified BFGS driver); piecewise-exponential; Weibull reliability-plot helper. Recovers true Weibull (shape, scale) closely on synthetic data.
- `r/parametric_survival.R` — thin wrapper around `survival::survreg` (all four families) with a note pointing to `flexsurv` for generalized gamma.

## Assumptions

- The chosen family really describes the hazard shape. Compare families by AIC (or LR test if nested).
- **AFT vs Cox**: AFT is a fully-parametric alternative to Cox; picks up a specific parametric hazard shape but doesn't require PH.

## Run

```
python techniques/parametric-survival/python/parametric_survival.py
Rscript techniques/parametric-survival/r/parametric_survival.R
```

**Refs:** Kalbfleisch, J.D. & Prentice, R.L. *The Statistical Analysis of Failure Time Data*, 2nd ed., Wiley, 2002; Collett, D. *Modelling Survival Data in Medical Research*, 3rd ed., Chapman & Hall/CRC, 2015; Wei, L.J. "The accelerated failure time model: a useful alternative to the Cox regression model in survival analysis." *Stat. Med.* 11(14–15), 1871–1879, 1992.

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
