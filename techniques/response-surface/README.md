# Response-Surface Methodology (Reference §16.11)

Iterative process for locating the optimum of a physical or industrial process.

## The RSM loop

1. **First-order** experimental design (2ᵏ factorial + centers), fit `y = β₀ + Σβᵢxᵢ`.
2. **Steepest-ascent path** — move along the gradient in unit steps until improvement stalls.
3. Near the optimum, augment to a **second-order** design (CCD or Box-Behnken) and fit a quadratic surface.
4. **Stationary-point analysis**: solve `∂ŷ/∂x = 0`, i.e. `x_s = −½ B⁻¹ b` for `ŷ = β₀ + xᵀb + xᵀBx`.

## Standard second-order designs

- **Central Composite Design (CCD)**: 2ᵏ factorial + 2k axial (star) points at ±α + `n_c` center replicates. Rotatable CCD: `α = (2ᵏ)^(1/4)`.
- **Box-Behnken**: 3-level design where each factor takes only three values; no extreme corners. Fewer runs than CCD for k = 3 – 4.

## Stationary-point classification

Eigenvalues of `B`:
- **All positive** → minimum.
- **All negative** → maximum.
- **Mixed signs** → **saddle point / ridge** — you've reached a rising ridge, not a peak; report the ridge path via canonical analysis.

## Files

- `python/response_surface.py` — CCD and (small) Box-Behnken generators + full quadratic fit + stationary-point analysis (linear vector, symmetric quadratic matrix, eigenvalue-sign classification). Demo (k = 2 CCD, true `y = 5 + 2x₁ + 3x₂ − x₁² − 2x₂²`): recovers coefficients to 3 decimals; stationary point `x_s = (1.04, 0.78)` close to truth `(1, 0.75)`; surface correctly classified as maximum.
- `r/response_surface.R` — `rsm::rsm(y ~ SO(x1, x2))` + `rsm::steepest` (Lenth's canonical package).

## When to use

- **Industrial process optimization** — yield, purity, throughput.
- **Formulation development** — pharma, food, materials.
- **Any physical experiment** where each trial is expensive and you need a small design to locate the optimum.

## Contrast with Bayesian optimization

|                | RSM (classical)                    | Bayesian Optimization                    |
|----------------|-----------------------------------|-------------------------------------------|
| Model          | polynomial (linear then quadratic) | Gaussian process (nonparametric)          |
| Design         | pre-specified CCD / BBD           | sequential + acquisition-driven          |
| Small budget   | very efficient                    | good with informative kernel              |
| High-D         | breaks down at k > 5              | scales to k ≈ 10 – 20                     |
| Interpretable? | yes (coefficients)                 | less so (kernel form)                     |

## Assumptions & caveats

- **Second-order polynomial** is the working assumption — check lack-of-fit with additional runs.
- **Coded units** `x ∈ [−1, 1]` — translates from natural units to statistical design.
- **Randomize run order** — protects against nuisance time trends.
- **Replicate center points** — quantifies pure error.

## Run

```
python techniques/response-surface/python/response_surface.py
Rscript techniques/response-surface/r/response_surface.R
```

**Refs:** Box, G.E.P. & Wilson, K.B. "On the experimental attainment of optimum conditions." *J. R. Stat. Soc. B* 13(1), 1–45, 1951; Myers, R.H., Montgomery, D.C. & Anderson-Cook, C.M. *Response Surface Methodology*, 4th ed., Wiley, 2016; Box, G.E.P. & Behnken, D.W. "Some new three level designs for the study of quantitative variables." *Technometrics* 2(4), 455–475, 1960.

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
