# Parametric Bootstrap (Reference §10.2)

Same idea as the nonparametric bootstrap, but instead of resampling the observed data, **fit a parametric model** and simulate new samples from the fitted distribution.

## Nonparametric vs. parametric

| | Nonparametric | Parametric |
|---|---|---|
| Resampling from | The observed data (with replacement) | The **fitted distribution** |
| Assumes | Only that observations are IID | The parametric family is (nearly) correct |
| Efficiency | Lower (uses only what's in the sample) | Higher if the model is right |
| Risk | Robust to model misspecification | Misleadingly tight CIs if the model is wrong |
| Best used when | You don't want to commit to a distribution | You do have a well-justified parametric family |

## Common uses

- CI for a parameter of a specific distribution (e.g. gamma shape, Weibull scale).
- Small-sample inference where the sampling distribution has a known form.
- Posterior-predictive-check-style diagnostics under a fitted GLM.

## Files

- `python/parametric_bootstrap.py` — generic driver `parametric_bootstrap(x, fit_fn, sample_fn, statistic, ...)` with normal / gamma / exponential families pre-wired.
- `r/parametric_bootstrap.R` — from-scratch driver + `MASS::fitdistr` for MLE + `boot::boot(sim="parametric")` in library form.

## Assumptions

- The parametric family assumed by `sample_fn` really does describe the data-generating process. **Always** check fit (QQ plot / KS test / graphical PPC) before trusting parametric-bootstrap CIs.
- Independent observations. For dependent data use [`block-bootstrap`](../block-bootstrap).

## Run

```
python techniques/parametric-bootstrap/python/parametric_bootstrap.py
Rscript techniques/parametric-bootstrap/r/parametric_bootstrap.R
```

**Refs:** Efron, B. & Tibshirani, R.J. *An Introduction to the Bootstrap*, Chapman & Hall, 1993 (Ch. 6); Davison, A.C. & Hinkley, D.V. *Bootstrap Methods and Their Application*, Cambridge, 1997 (Ch. 2).

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
