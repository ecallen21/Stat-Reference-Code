# Arellano-Bond Difference GMM (Reference §35.3)

Arellano & Bond (1991). Consistent estimator of a **dynamic panel
model** with unit effects:

```
y_it = ρ y_{i, t-1} + x_itᵀ β + α_i + ε_it.
```

## Why FE / OLS fail

- **Nickell bias** — within-FE demeaning correlates `y_{i,t-1}` with
  the transformed error; the FE estimator is `−O(1/T)` biased.
- **OLS** without unit effects picks up `α_i`.

## Arellano-Bond difference GMM

1. **First-difference** the equation to eliminate `α_i`:
   `Δy_it = ρ Δy_{i,t-1} + Δx_itᵀ β + Δε_it`.
2. **Instrument** `Δy_{i,t-1}` with lagged levels
   `y_{i,t-2}, y_{i,t-3}, …` — valid moment conditions
   `𝔼[y_{i,t-2} · Δε_it] = 0`.
3. Optimal GMM combines all valid moments.

## When to use

- **Panel data with dynamic (lagged-dependent) regressors**.
- **Endogenous predictors** — extended AB with additional external
  instruments.
- **Micro panels** with large N, small T.

## When NOT to use

- **Small N** — GMM is asymptotic; small-sample bias can be severe.
- **Very persistent series** (ρ near 1) — lagged levels are weak
  instruments; use `system GMM` (Blundell-Bond 1998).
- **Long T** — Nickell bias vanishes; use FE-OLS.

## Files

- `python/arellano_bond_gmm.py` — from-scratch AB estimator with a
  stacked block-diagonal instrument set. Compares AB to within-FE
  OLS on a synthetic AR(1) panel (n_units = 500, T = 15, true ρ =
  0.6). Both estimators show finite-sample bias (well-documented AB
  small-sample issue: Álvarez-Arellano 2003, Bun-Windmeijer 2010).
- `r/arellano_bond_gmm.R` — `plm::pgmm` (R reference);
  `linearmodels.panel.PanelGMM`, `pydynpd` (Python).

## Assumptions & caveats

- **Small-sample bias** — AB in finite samples can still be biased;
  Windmeijer-corrected SEs and system GMM are standard fixes.
- **Instrument proliferation** — with large T, the number of moment
  conditions grows quickly; collapse or lag-limit instruments
  (Roodman 2009).
- **Weak instruments** for persistent processes — use system GMM
  (Blundell-Bond 1998) which adds level equations with differenced
  lagged instruments.
- **Overidentification test** — Hansen J for validity of the moment
  conditions.
- **Serial correlation** — AR(1) / AR(2) tests on the differenced
  residuals (Arellano-Bond m2).

## Related in this repo

- `fixed-effects-panel`, `hausman-test` — the FE / RE workhorses.
- `gmm-general` — the underlying framework.
- `iv-2sls`, `panel-cointegration` — sibling econometric tools.
- `state-space-kalman` — alternative for dynamic panels via state-space.

## Run

```
python techniques/arellano-bond-gmm/python/arellano_bond_gmm.py
Rscript techniques/arellano-bond-gmm/r/arellano_bond_gmm.R
```

**Refs:** Arellano, M. & Bond, S. "Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations." *Review of Economic Studies*, 1991; Blundell, R. & Bond, S. "Initial conditions and moment restrictions in dynamic panel data models." *Journal of Econometrics*, 1998; Roodman, D. "How to do xtabond2." *Stata Journal*, 2009.

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
