# Geographically Weighted Regression (Reference §23.11)

**Local** weighted linear regression, run at each observation location (Brunsdon-Fotheringham-Charlton 1996):

```
β̂_i = (Xᵀ W_i X)⁻¹ Xᵀ W_i y
```

`W_i` is a diagonal matrix of kernel weights depending on distance from location `i`. Kernels: **Gaussian** or **bisquare**. Bandwidth chosen by cross-validation (leave-one-out) or AICc.

Output: `n × p` coefficient surface (one `β̂` vector per location).

## When to use

- **Test for spatial non-stationarity** — does the regression relationship vary across space?
- **Local pricing / valuation** — hedonic house-price models where marginal effects differ by neighbourhood.
- **Ecology** — species-environment relationships that vary with region.

## Files

- `python/geographically_weighted_regression.py` — from-scratch GWR with Gaussian kernel + LOO CV bandwidth selection. Demo (n = 200, true `β_x = 1 + x_coord/5`): CV picks bandwidth 1.01; recovers spatially-varying β_x nearly perfectly across quartiles (1.66, 2.14, 2.62 vs true 1.62, 2.13, 2.63).
- `r/geographically_weighted_regression.R` — `spgwr::gwr` or `GWmodel::gwr.basic` (more kernel and bandwidth options).

## Extensions

- **Adaptive bandwidth** — vary `h_i` so each local fit uses `k` neighbours (avoids sparse-data cells).
- **Multi-scale GWR (MGWR)** — each covariate has its own bandwidth; captures effects that vary at different spatial scales.
- **Robust GWR** — iterative down-weighting of outliers.
- **GTWR** — geographically **and temporally** weighted regression.

## Assumptions & caveats

- **Computationally expensive** — O(n) fits, each O(np²).
- **Multicollinearity is local** — a stable global fit can be unstable in some neighbourhoods.
- **Inference is tricky** — pointwise SEs aren't independent; global tests of coefficient variation exist.
- **Standardize predictors** for interpretable coefficient surfaces.

## Contrast with SAR / CAR

- **GWR** — varying-coefficient regression across space.
- **SAR / CAR** — homogeneous coefficients + spatially structured errors or spatially lagged outcome.
- Choose based on the question: "does the relationship differ across space?" (GWR) vs "is there residual spatial dependence?" (SAR/CAR).

## Run

```
python techniques/geographically-weighted-regression/python/geographically_weighted_regression.py
Rscript techniques/geographically-weighted-regression/r/geographically_weighted_regression.R
```

**Refs:** Brunsdon, C., Fotheringham, A.S. & Charlton, M. "Geographically weighted regression: a method for exploring spatial nonstationarity." *Geogr. Anal.* 28(4), 281–298, 1996; Fotheringham, A.S., Brunsdon, C. & Charlton, M. *Geographically Weighted Regression: the Analysis of Spatially Varying Relationships*, Wiley, 2002.

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
