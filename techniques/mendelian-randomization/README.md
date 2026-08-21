# Mendelian Randomization (Reference §15.x extra)

Instrumental-variable estimation of a **causal effect of an exposure X on
outcome Y** using genetic variants (SNPs) as instruments. Under the
Mendelian randomization triangle:

- **Relevance**: SNP → X (strong first stage).
- **Exchangeability**: SNP is independent of confounders (justified by random meiosis).
- **Exclusion**: SNP affects Y only through X (no direct pleiotropy).

Per-SNP Wald ratio: `r_k = β_Y_k / β_X_k`. Combining across K SNPs:

## Estimators

| Estimator | Formula | Robust to |
|---|---|---|
| **IVW** (inverse-variance-weighted) | `Σ w_k r_k / Σ w_k`, `w_k = β_X_k² / SE_Y_k²` | balanced (mean-zero) pleiotropy |
| **MR-Egger** | weighted regression `β_Y = α + β · β_X`; intercept α tests directional pleiotropy | directional pleiotropy (via non-zero α) |
| **Weighted median** | 50th percentile of `r_k` weighted by IVW weights | up to 50% invalid instruments |
| **Weighted mode (MBE)** | mode of `r_k` (kernel-smoothed) | plurality-valid instruments |
| **MR-PRESSO** | outlier detection + refit | single-SNP outliers |

## When to use

- **Causal effect estimation** in observational epidemiology / social science when a strong SNP → exposure relationship exists (BMI, LDL, blood pressure, alcohol) and randomised trials are infeasible.
- **Triangulation** — MR + a matched RCT + a sibling-comparison study, each with different biases, converging on the causal answer.
- **Two-sample MR** with GWAS summary stats (β and SE from separate cohorts for X and Y).

## Files

- `python/mendelian_randomization.py` — from-scratch IVW, MR-Egger, weighted median. Demo (K=40 SNPs, true causal 0.5): valid IVs → IVW 0.54, MR-Egger slope 0.47 with intercept ≈ 0; 20% pleiotropic IVs (positive-shifted) → IVW inflates to 0.63 (biased), MR-Egger slope 0.47 (recovers truth, intercept +0.024 flags pleiotropy), weighted median 0.57 (robust).
- `r/mendelian_randomization.R` — `MendelianRandomization::mr_ivw / mr_egger / mr_median / mr_mbe`, `TwoSampleMR::mr / mr_pleiotropy_test / mr_heterogeneity`.

## Assumptions & caveats

- **Weak-instrument bias** — F-statistic per SNP should be > 10; otherwise MR-Egger has particularly poor precision.
- **Winner's curse** — using discovery-cohort β_X biases MR toward the null; use independent GWAS or split samples.
- **Population stratification** — SNPs correlate with ancestry which correlates with outcome; adjust for principal components.
- **Canalisation / developmental compensation** — long-term genetic exposure ≠ short-term intervention on the same biomarker; MR estimates a *lifetime* effect.
- **Pleiotropy is common** — always report MR-Egger, weighted median, and MR-PRESSO alongside IVW; a large intercept flags trouble.
- **InSIDE assumption** for MR-Egger (instrument-strength independent of direct effects) is often violated; check sensitivity.

## Related in this repo

- `iv-2sls` — classical instrumental-variables regression with observational instruments.
- `tmle-doubly-robust`, `inverse-probability-weighting`, `propensity-score-matching` — causal alternatives when random-genetic-instruments aren't available.
- `sensitivity-e-value` — quantify unmeasured-confounding robustness of a non-IV estimate.

## Run

```
python techniques/mendelian-randomization/python/mendelian_randomization.py
Rscript techniques/mendelian-randomization/r/mendelian_randomization.R
```

**Refs:** Davey Smith, G. & Ebrahim, S. "'Mendelian randomization': can genetic epidemiology contribute to understanding environmental determinants of disease?" *Int. J. Epidemiol.* 32(1), 1–22, 2003; Bowden, J. et al. "Mendelian randomization with invalid instruments: effect estimation and bias detection through Egger regression." *Int. J. Epidemiol.* 44(2), 512–525, 2015; Hartwig, F.P., Davey Smith, G. & Bowden, J. "Robust inference in summary data Mendelian randomization via the zero modal pleiotropy assumption." *Int. J. Epidemiol.* 46(6), 1985–1998, 2017.

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
