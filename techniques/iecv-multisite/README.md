# Internal-External Cross-Validation (IECV) (Reference §39.25)

Debray et al. (2013), Steyerberg & Harrell (2016). When development
data span multiple sites, **leave-one-site-out** validation is the
right measure of transportability — better than random-observation
cross-validation, which hides between-site heterogeneity.

## Procedure

```
For each site k in 1..K:
    Develop model on K − 1 other sites
    Validate on held-out site k → AUC_k, CITL_k, slope_k
Report per-site metrics + pooled summary
       (fixed-effect or DerSimonian-Laird meta-analysis)
```

## When to use

- **Multi-site development cohorts** — always report IECV in
  addition to internal-validation bootstrap.
- **IPD meta-analytic prediction** — the standard workflow (Debray
  framework).
- **Sanity check** before deploying at a new site — a large
  between-site slope spread predicts poor transportability.

## When NOT to use

- **Single-site data** — no groups to leave out; use bootstrap
  optimism.
- **Very small `K`** (≤ 3 sites) — per-site estimates are unstable;
  report cautiously.

## Files

- `python/iecv_multisite.py` — leave-one-site-out logistic
  regression with per-site AUC, CITL, slope + weighted pooling.
  Demo (K=5 sites, n_i=200, site-specific intercepts and
  coefficient scaling): per-site **AUC 0.64-0.73, slope 0.62-1.45**;
  pooled AUC 0.69, slope 1.00, CITL 0.00.
- `r/iecv_multisite.R` — `metamisc::valmeta`, `rms::validate`,
  `pmsampsize` (R); `sklearn.model_selection.LeaveOneGroupOut` +
  custom (Python).

## Assumptions & caveats

- **Site membership defined a priori** — do not cluster observations
  post-hoc.
- **Meta-analytic pooling** — random-effects (DerSimonian-Laird) is
  more honest when between-site heterogeneity is present; report τ²
  or `I²` alongside pooled estimates.
- **Weight choice** — event count or sample size for
  discrimination; sample size or `n_events` for calibration.
- **Small held-out site** produces noisy per-site AUC; consider
  bootstrap confidence intervals per site.
- **Development sites should be exchangeable** — non-random site
  inclusion biases the IECV estimate of transportability.

## Related in this repo

- `bootstrap-optimism-correction` — internal-validation cousin.
- `external-validation` — single held-out cohort test.
- `discrimination-calibration`, `calibration-plots` — the metrics
  reported per site.

## Run

```
python techniques/iecv-multisite/python/iecv_multisite.py
Rscript techniques/iecv-multisite/r/iecv_multisite.R
```

**Refs:** Debray, T.P.A., Moons, K.G.M., Ahmed, I., Koffijberg, H., & Riley, R.D. "A framework for developing, implementing, and evaluating clinical prediction models in an individual participant data meta-analysis." *Statistics in Medicine*, 2013; Steyerberg, E.W. & Harrell, F.E. "Prediction models need appropriate internal, internal-external, and external validation." *Journal of Clinical Epidemiology*, 2016.

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
