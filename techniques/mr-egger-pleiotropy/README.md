# MR-Egger Regression (Reference §47.64)

Bowden, Davey Smith & Burgess (2015). Extends inverse-variance-
weighted Mendelian randomization by fitting a weighted linear
regression WITH INTERCEPT:

    β̂_YG_j = α₀ + β_MR · β̂_XG_j + ε_j,     w_j = 1 / se(β̂_YG_j)².

- `α₀ = 0` under InSIDE (Instrument Strength Independent of Direct
  Effect) — no directional pleiotropy.
- If `α₀ ≠ 0`, IVW is biased but MR-Egger's slope `β_MR` remains
  consistent.

Weaker assumption than IVW but wider CIs; report both.

## Files

- `python/mr_egger_pleiotropy.py` — from-scratch weighted-OLS
  IVW + MR-Egger with intercept SE. Demo (K=40 SNPs, β_true=0.30):
  - Scenario 1 (no pleiotropy): IVW β̂ = 0.34, MR-Egger β̂ = 0.26,
    α̂ = +0.009 (n.s.)
  - Scenario 2 (directional pleiotropy α₀=0.02): IVW β̂ = 0.50
    (biased upward), MR-Egger β̂ = 0.38 (consistent), α̂ = +0.012
    (detects the pleiotropy).
- `r/mr_egger_pleiotropy.R` — `MendelianRandomization::mr_egger`,
  `TwoSampleMR::mr` (R); `pymr`, from-scratch (Python).

## When to use

- **Two-sample MR** with many SNP instruments.
- **After IVW** — always include as a sensitivity analysis for
  horizontal pleiotropy.
- **Alongside MR-PRESSO, weighted-median, MR-Lasso** — triangulate.
- **Genome-wide instrument sets** — many SNPs improve the InSIDE
  assumption's plausibility.

## When NOT to use

- **Few (< 3) instruments** — no power for the intercept.
- **All instruments in one gene / pathway** — likely correlated
  pleiotropy; InSIDE unlikely.
- **Balanced pleiotropy across instruments** — IVW already
  unbiased; MR-Egger loses precision unnecessarily.

## Assumptions & caveats

- **InSIDE** — the crucial assumption; hard to verify empirically.
- **NOME** (no measurement error) — bias if SE(β_XG) is not
  negligible; use `I²_GX` / SIMEX correction.
- **Weighted regression** — heteroskedasticity handled via SE_YG.
- **Interpretation** — the intercept α₀ estimates AVERAGE
  directional pleiotropy; individual SNP effects can still be
  pleiotropic.

## Related in this repo

- `mendelian-randomization` — the IVW baseline this extends.
- `iv-2sls`, `weak-instruments-anderson-rubin`,
  `arellano-bond-gmm` — IV econometrics.
- `meta-analysis`, `meta-regression`, `network-meta-analysis` —
  weighted-effect aggregation cousins.
- `sensitivity-e-value`, `negative-outcome-controls`,
  `hdps-high-dim-propensity` — sensitivity toolkit.

## Run

```
python techniques/mr-egger-pleiotropy/python/mr_egger_pleiotropy.py
Rscript techniques/mr-egger-pleiotropy/r/mr_egger_pleiotropy.R
```

**Refs:** Bowden, J., Davey Smith, G. & Burgess, S. "Mendelian randomization with invalid instruments: effect estimation and bias detection through Egger regression." *Int J Epidemiol* 44(2): 512-525, 2015; Hemani, G., Bowden, J. & Davey Smith, G. "Evaluating the potential role of pleiotropy in Mendelian randomization studies." *Hum Mol Genet* 27(R2): R195-R208, 2018.

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
