# Getis-Ord Gi and Gi* Hot-Spot Statistics (Reference §23.x extra)

Local indicators of spatial association that flag **concentrations of high (or low) values** by summing neighbouring values:

```
Gi(d)  = Σ_{j ≠ i} w_ij(d) x_j / Σ_{j ≠ i} x_j
Gi*(d) = Σ_j w_ij(d) x_j / Σ_j x_j            (includes focal cell i)
```

The `Gi*` form is usually preferred — it includes the focal cell and has a cleaner asymptotic normal distribution under the CSR null:

```
E[Gi*] = W_i / n
Var[Gi*] = W_i (n − W_i) s² / ( n² (n − 1) x̄² )
```

where `W_i = Σ_j w_ij`, `s² = (1/n) Σ (x_j − x̄)²`. Standardised as `z = (Gi* − E) / √Var`, positive `z` → **hot spot**, negative → **cold spot**. `|z| > 1.96` is the usual pointwise 5% cutoff.

## Contrast with LISA (local Moran's I)

- **LISA** decomposes global Moran's I into per-location contributions; captures *similarity* of `x_i` to its neighbours (HH, LL, HL, LH — see `local-moran-lisa`).
- **Gi*** does not depend on `x_i − x̄`; it captures *magnitude* — is the neighbourhood high or low, regardless of whether the focal cell stands out. A cell surrounded by high values but itself average is a Gi* hot spot but not a LISA HH.

## When to use

- **Crime / epidemiology hot spots** — pinpoint contiguous high-rate regions.
- **Retail catchments** — identify where sales density is significantly high.
- **Ecological hot spots of species density**, **temperature anomalies**.
- **Regulatory / policy targeting** — Gi* pinpoints *which* cells rather than *whether* there's global clustering.

## Files

- `python/getis_ord_g_statistic.py` — from-scratch Gi* with asymptotic z-scores and permutation-based two-sided p-values. Demo (10×10 grid with a planted 3×3 hot cluster of +3σ values, queen-like distance band): all 9 planted cells recovered (TP = 9), 3 near-neighbour false positives; top-5 z-scores all in the planted region; permutation flags 13 significant cells at p < 0.05.
- `r/getis_ord_g_statistic.R` — `spdep::localG`, `spdep::localG_perm`, `rgeoda::local_gstar`.

## Assumptions & caveats

- **Distance-band matters** — too small → sparse W_i, unstable estimates; too large → oversmoothed, misses local structure. Try a range and report sensitivity.
- **Skewed `x`** — the asymptotic Gaussian null relies on approximate normality of `x`; permutation p-values are more reliable for count / rate data.
- **Multiple testing** — with n cells there are n z-tests; correct via FDR (`p.adjust(..., 'fdr')`) or use conditional permutation (`spdep::localG_perm`).
- **Edge effects** — cells near the boundary have fewer neighbours, so their `Var[Gi*]` is inflated and z-scores understated.
- **Rate data with denominators** — use empirical-Bayes smoothing first (see `empirical-bayes`) or compute Gi* on standardised morbidity ratios (SMRs).

## Run

```
python techniques/getis-ord-g-statistic/python/getis_ord_g_statistic.py
Rscript techniques/getis-ord-g-statistic/r/getis_ord_g_statistic.R
```

**Refs:** Getis, A. & Ord, J.K. "The analysis of spatial association by use of distance statistics." *Geogr. Anal.* 24(3), 189–206, 1992; Ord, J.K. & Getis, A. "Local spatial autocorrelation statistics: distributional issues and an application." *Geogr. Anal.* 27(4), 286–306, 1995.

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
