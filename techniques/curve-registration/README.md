# Curve Registration (Reference §31.11)

Curves `x_i(t)` often differ in **timing (phase)** as well as
**magnitude (amplitude)**. Un-aligned curves inflate cross-curve
variance and confuse FPCA / clustering / regression.

## Two approaches

- **Landmark registration** — identify a small number of features
  (peak times, zero-crossings), warp `t` so those landmarks coincide.
- **Continuous registration** — estimate a smooth warping `h_i(t)`
  and template `μ(t)` jointly (SRSF, elastic distances).

## When to use

- **Any collection of curves with variable timing** — growth spurts,
  glucose peaks post-meal, gait cycles.
- **Before FPCA / clustering** to reveal amplitude structure.
- **Comparing shapes when phase is confounding**.

## When NOT to use

- **Curves with meaningful timing differences** you want to preserve
  — registration destroys those.
- **Sparse curves** — landmarks are unreliable; use SRSF.
- **No clear landmarks** — continuous methods only.

## Files

- `python/curve_registration.py` — from-scratch peak-based landmark
  registration with a piecewise-linear warp. Demo: 25 bell-shaped
  curves with peak times drawn uniformly from `[0.35, 0.65]`:
  **cross-curve mean variance 0.0475 → 0.0031 (93.5 % reduction);
  all aligned peaks lock exactly to the target 0.533**.
- `r/curve_registration.R` — `fda::landmarkreg`, `fda::register.fd`,
  `fdasrvf` (R); `fdasrsf`, `scikit-fda` (Python).

## Assumptions & caveats

- **Landmark identifiability** — peaks / zero-crossings must exist and
  be reliable across curves.
- **Warp identifiability** — the warping function must be monotone
  (`h_i'(t) > 0`); PL-linear satisfies this trivially.
- **Amplitude vs phase decomposition** — SRSF (Srivastava-Klassen)
  separates the two rigorously.
- **Post-registration FPCA** — do FPCA on aligned curves; on the
  warping functions to describe phase variability.
- **Interpretation** — always report `phase` and `amplitude` FDAs
  separately if you register.

## Related in this repo

- `functional-pca`, `functional-regression`, `functional-anova`,
  `functional-clustering`, `functional-depth` — FDA family (this
  batch); registration is a preprocessing step.
- `dynamic-time-warping` (if present) — the discrete-signal cousin.
- `local-regression-loess` — smoothing before landmark detection.

## Run

```
python techniques/curve-registration/python/curve_registration.py
Rscript techniques/curve-registration/r/curve_registration.R
```

**Refs:** Ramsay, J.O. & Silverman, B.W. *Functional Data Analysis*, Springer, 2005 (Ch. 7); Srivastava, A. & Klassen, E. *Functional and Shape Data Analysis*, Springer, 2016 (SRVF framework).

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
