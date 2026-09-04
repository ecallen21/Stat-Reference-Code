# Generalized Method of Moments (Reference §35.5)

Hansen (1982). Estimate `θ` by matching **sample moment conditions**
`ĝ(θ)` to zero in a weighted quadratic norm:

```
θ̂  =  argmin_θ  ĝ(θ)ᵀ W ĝ(θ),   ĝ(θ) = (1/n) Σ_i g(z_i; θ).
```

## Just-identified vs over-identified

- `dim(g) = dim(θ)` — one solution; `W` doesn't matter.
- `dim(g) > dim(θ)` — over-identified; optimal weight `W* = S⁻¹` where
  `S = Var(g(z; θ*))`.

## Two-step GMM

```
1.  Solve with W = I  →  θ̂₁.
2.  Ŝ = (1/n) Σ g(z_i; θ̂₁) g(z_i; θ̂₁)ᵀ.
3.  Re-solve with W = Ŝ⁻¹  →  θ̂₂  (efficient).
```

## Hansen J overidentifying test

```
J  =  n · ĝ(θ̂₂)ᵀ W ĝ(θ̂₂)   ~   χ²(q − p)   under H₀ (moment conditions hold).
```

## When to use

- **Any moment-based estimation** — mean, IV, dynamic panels, Euler
  equations, MSM (method of simulated moments).
- **Overidentified models** to test model specification via J.
- **Semiparametric estimation** where full likelihood is intractable.

## When NOT to use

- **Full likelihood is available and tractable** — MLE is usually
  more efficient.
- **Weak moments** — estimator can be very biased; weak-instrument
  robust tests exist (Kleibergen-Moreira 2005).

## Files

- `python/gmm_general.py` — from-scratch two-step GMM using
  `scipy.optimize`. Demo: over-identified estimation of `μ = 𝔼[X]`
  under two moments `𝔼[X − μ] = 0` and `𝔼[X² − (μ² + 1)] = 0` (uses
  Var(X) = 1). Truth 2.5; step-1 = 2.47, step-2 = 2.46; **Hansen
  J = 0.18, p = 0.67** — correctly specified.
- `r/gmm_general.R` — `gmm` (R reference); `statsmodels.sandbox
  .regression.gmm.GMM`, `linearmodels` (Python).

## Assumptions & caveats

- **Moment-condition validity** — J-test is essential.
- **Weak identification** — GMM can be badly biased if moments are
  weak. Kleibergen-Moreira robust tests.
- **Efficient weight matrix** — first step consistent, second step
  efficient; iterated / CUE (continuously-updating estimator) go
  further.
- **Bootstrap SEs** — recommended when the number of moments is
  moderate relative to n (Windmeijer correction for AB).

## Related in this repo

- `arellano-bond-gmm` — dynamic-panel special case.
- `iv-2sls` — GMM with linear moments = 2SLS.
- `empirical-likelihood`, `semiparametric-efficiency` — sibling
  moment-based inference.
- `information-criteria` — Hansen J complements AIC / BIC.

## Run

```
python techniques/gmm-general/python/gmm_general.py
Rscript techniques/gmm-general/r/gmm_general.R
```

**Refs:** Hansen, L.P. "Large sample properties of generalized method of moments estimators." *Econometrica*, 1982; Hall, A.R. *Generalized Method of Moments*, Oxford University Press, 2005.

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
