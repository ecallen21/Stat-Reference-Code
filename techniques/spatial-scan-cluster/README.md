# Kulldorff Spatial Scan Statistic (Reference §23.13)

Cluster-detection method that scans **circular candidate windows** of growing
radius over a set of locations and, for each, computes a likelihood-ratio
against a constant-risk null.

## Poisson form

```
LLR(Z) = c_Z · log(c_Z / μ_Z) + (C − c_Z) · log((C − c_Z) / (C − μ_Z))
         · 1{c_Z / μ_Z > (C − c_Z) / (C − μ_Z)}
```

- `c_Z` = observed cases in candidate window `Z`.
- `μ_Z = C · pop_Z / pop_total` = expected cases under uniform risk.
- The indicator restricts attention to **excess** clusters (rate inside > rate outside).

The window with the largest LLR is the **most likely cluster**. A Monte-Carlo
p-value is obtained by permuting cases across locations (fixed populations,
multinomial with probabilities `pop / total_pop`).

## Bernoulli form

Same shape but with cases and controls as two categorical outcomes; use when
you have case-control point data instead of counts + population.

## When to use

- **Disease surveillance** — cancer clusters, outbreak detection.
- **Crime hot-spots** — Poisson counts by census tract.
- **Ecology** — species aggregation over sampling units.
- **Any count / rate mapping** where the question is *where* the excess is.

## Files

- `python/spatial_scan_cluster.py` — from-scratch Poisson scan with MC p-value. Demo on 10×10 grid with a planted 3×3 hotspot at (4.5, 4.5) at 4× baseline rate: detects centre at (4, 5), radius 2.0, LLR 168, p = 0.010; captures 10 of the 16 true hotspot cells.
- `r/spatial_scan_cluster.R` — `SpatialEpi::kulldorff`, `smerc::scan.test`, or the SaTScan program.

## Assumptions & caveats

- **Circular window** — biased against non-circular clusters; elliptic scan (`SaTScan`) or graph-based scans (`smerc::flex.test`) relax this.
- **Multiple testing** absorbed in the MC p-value — the likelihood ratio implicitly ranks all windows.
- **Population confounding** — expected counts should already adjust for known confounders (age, sex) via indirect standardization.
- **Fixed maximum window** (e.g. 50% of cases) is a user choice; too large → dilution, too small → miss diffuse clusters.

## Related methods

- **Besag-Newell**: distance to the *k*-th case as a local test.
- **Tango's MEET**: max excess events test with proximity weights.
- **Getis-Ord G / G\***: local statistics from the same family as LISA (see `local-moran-lisa`).

## Run

```
python techniques/spatial-scan-cluster/python/spatial_scan_cluster.py
Rscript techniques/spatial-scan-cluster/r/spatial_scan_cluster.R
```

**Refs:** Kulldorff, M. "A spatial scan statistic." *Comm. Stat. Theory Meth.* 26(6), 1481–1496, 1997; Kulldorff, M. & Nagarwalla, N. "Spatial disease clusters: detection and inference." *Stat. Med.* 14(8), 799–810, 1995.

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
