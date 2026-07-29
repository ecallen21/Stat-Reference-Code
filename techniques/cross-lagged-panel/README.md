# Cross-Lagged Panel Model + RI-CLPM (Reference §12.10; also covers §12.18)

Two variables `X` and `Y` measured at 2+ time points on the same subjects. Does `X` predict later `Y`, does `Y` predict later `X`, or both?

## Classic 2-wave CLPM

Two regressions:

```
X_t  =  a_x  +  φ_x · X_{t-1}  +  β_yx · Y_{t-1}  +  e_x
Y_t  =  a_y  +  φ_y · Y_{t-1}  +  β_xy · X_{t-1}  +  e_y
```

| Coefficient | Meaning |
|---|---|
| `φ_x, φ_y` | autoregressive stability of each variable |
| `β_yx` | Y_{t-1} predicting X_t — "does Y lead X?" |
| `β_xy` | X_{t-1} predicting Y_t — "does X lead Y?" |

**Interpretation**: `β_xy` significant with `β_yx` not ⇒ **X Granger-leads Y in the panel sense**. (This is prediction, not causation — see caveats below.)

## §12.18 Random-Intercept CLPM (RI-CLPM; Hamaker, Kuiper & Grasman 2015)

The classic CLPM conflates two things:

- **Between-person differences** (subject A is always high on X, subject B is always low)
- **Within-person change** (subject A's X went up between t-1 and t)

Adding a **random intercept** per subject per variable separates them. The cross-lagged effects then refer to **within-person** dynamics only — the more defensible "does X actually push Y up within a person?" question. Needs at least 3 waves; usually fit via SEM (`lavaan` in R, `semopy` in Python).

This file ships a **simplified RI-CLPM** via person-centering (subtract each subject's mean across time before running the CLPM). Approximate but shows the concept.

## Files

- `python/cross_lagged_panel.py` — classic 2-wave CLPM via two OLS + person-centered RI-CLPM approximation for ≥ 3 waves. On synthetic data with true X → Y and no Y → X, CLPM correctly rejects Y→X (p = 0.77) and confirms X→Y (p < 0.001).
- `r/cross_lagged_panel.R` — classic CLPM via `lm` + pointer to `lavaan` for full RI-CLPM SEM.

## Assumptions & caveats

- **Time-ordered** measurements — direction is meaningful.
- **Same lag length** for both cross-lagged paths.
- **Not causal** — CLPM tests panel-Granger prediction, not intervention effects.
- **Measurement error** biases lagged coefficients (usually toward zero for autoregressive paths, unpredictably for cross-lagged). RI-CLPM helps somewhat by separating trait from state.

## Run

```
python techniques/cross-lagged-panel/python/cross_lagged_panel.py
Rscript techniques/cross-lagged-panel/r/cross_lagged_panel.R
```

**Refs:** Kessler, R.C. & Greenberg, D.F. *Linear Panel Analysis*, Academic Press, 1981; Hamaker, E.L., Kuiper, R.M. & Grasman, R.P.P.P. "A critique of the cross-lagged panel model." *Psych. Methods* 20(1), 102–116, 2015; Mulder, J.D. & Hamaker, E.L. "Three extensions of the random intercept cross-lagged panel model." *Struct. Equ. Model.* 28(4), 638–648, 2021.

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
