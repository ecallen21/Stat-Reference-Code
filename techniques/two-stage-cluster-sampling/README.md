# Two-Stage Cluster Sampling (Reference §3.x extra)

Sample structure:

- **Stage 1** — from a population of `M` primary sampling units (PSUs, e.g. schools, blocks, hospitals), draw `m` PSUs.
- **Stage 2** — within each sampled PSU `i` (size `N_i`), draw `n_i` secondary sampling units (SSUs, e.g. students, households, patients).

Total sample size = `Σᵢ n_i`.

## Mean estimator (SRS at both stages, approximately equal cluster sizes)

```
ȳ = (1 / m) · Σ_i ȳ_i
```

## Variance decomposition (Cochran 1977)

```
Var(ȳ) = (1 − m/M) · S_b² / m
       + (1/m) · mean_i (1 − n_i/N_i) · S_{w,i}² / n_i
```

- `S_b²` — variance of true PSU means (between-cluster).
- `S_{w,i}²` — within-cluster variance in cluster `i`.
- Finite-population corrections `(1 − m/M)` and `(1 − n_i/N_i)` shrink the estimate when sampling fractions are large.

## Design effect

```
DEFF ≈ 1 + (n̄ − 1) · ICC
```

where `ICC = (MS_B − MS_W) / (MS_B + (n̄ − 1) · MS_W)`. `DEFF > 1` means the cluster design is **less efficient** than SRS of the same total size; the **effective sample size** is `n_eff = n_total / DEFF`.

## When to use

- **Cost-driven surveys** — enumeration is cheap within a cluster (school, block) but reaching a new cluster is expensive.
- **Areas without a full population list** — you can list clusters (e.g. villages) but not every household.
- **Rare-outcome studies** where clustered sampling naturally captures cases (households of a diseased person).
- **Multi-stage government surveys** (NHANES, DHS, PISA).

## Files

- `python/two_stage_cluster_sampling.py` — mean estimator + two-stage variance decomposition + ICC + DEFF. Demo (M=200 PSUs, N=50 each; m=15 sampled clusters, n_i=10 SSUs each; true cluster-mean SD 8, within SD 4): two-stage SE = 1.67 vs naive SRS SE = 0.62; ICC = 0.69 (strong clustering); DEFF = 7.21 → n_eff = 20 instead of 150.
- `r/two_stage_cluster_sampling.R` — `survey::svydesign(ids = ~ PSU + SSU, fpc = ...)`, `survey::svymean / svyglm`, `ICC::ICCbare`.

## Assumptions & caveats

- **Equal cluster sizes** assumed in the closed-form Cochran variance; for very unequal sizes use `survey::svydesign` (ratio-of-two-means estimator).
- **PPS sampling at stage 1** — probability-proportional-to-size selection gives more efficient estimates; use `survey::svydesign(probs = ...)`.
- **Second-stage sampling frame** must be constructed inside each sampled cluster; that field-work step matters as much as the sampling design.
- **ICC drives efficiency** — small `ICC (< 0.05)` gives modest `DEFF`; large `ICC (> 0.5)` can wipe out most of the "n on paper".
- **Analytic weights** (`survey::svyglm`) generalise regression / GLMs to complex designs.
- **Model-based alternative** — mixed model with random intercept per PSU (`linear-mixed-models`), reconciling design-based and model-based inference on the same data.

## Related in this repo

- `linear-mixed-models` — model-based analysis of clustered data (random-intercept ANOVA is exactly the ICC decomposition here, done via ML).
- `gee` — semi-parametric alternative for correlated outcomes.
- `sandwich-robust-se` — CR-type cluster-robust standard errors in regression settings.

## Run

```
python techniques/two-stage-cluster-sampling/python/two_stage_cluster_sampling.py
Rscript techniques/two-stage-cluster-sampling/r/two_stage_cluster_sampling.R
```

**Refs:** Cochran, W.G. *Sampling Techniques*, 3rd ed., Wiley, 1977; Lohr, S.L. *Sampling: Design and Analysis*, 3rd ed., CRC Press, 2021; Lumley, T. *Complex Surveys: A Guide to Analysis Using R*, Wiley, 2010.

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
