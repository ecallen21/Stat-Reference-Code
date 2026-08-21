# Inverse Distance Weighting (Reference §23.8)

Shepard 1968. Weighted average of neighbouring values, weight inversely proportional to distance raised to power `p`:

```
Ẑ(x_0) = Σ_i (Z_i / d_i^p) / Σ_i (1 / d_i^p)
```

- `p = 2` — most common.
- `p → 0` — approaches unweighted mean.
- `p → ∞` — approaches nearest-neighbour.

Restrict to `k` nearest samples for local smoothing.

## Files

- `python/inverse_distance_weighting.py` — from-scratch IDW with power + optional k-nearest. Demo (n = 80, `sin(x)+cos(y)` field): RMSE decreases with `p` (0.81 → 0.57 → 0.50); k = 5 nearest gives 0.45 (best).
- `r/inverse_distance_weighting.R` — `gstat::krige(model = NULL, idp = 2)` or `gstat::idw`.

## IDW vs kriging

|                  | IDW                             | Kriging                              |
|------------------|---------------------------------|--------------------------------------|
| Speed            | fast                            | O(n³) per prediction                 |
| Uncertainty      | none                            | principled (kriging variance)        |
| Requires         | choice of p and k               | fitted variogram                     |
| Interpolant      | exact at samples                | exact (with zero-nugget) or smooth   |
| Extrapolation    | reverts to nearest-neighbour    | reverts to mean                      |

## When to use

- **Quick-and-simple** spatial interpolation for visualization.
- **When variogram doesn't converge** or you don't have enough samples to fit one.
- **Baseline** to compare against kriging.

## Assumptions & caveats

- **No uncertainty** — pair with cross-validation for prediction-error estimates.
- **Bull's-eye effect** — near samples the surface bulges toward the sample value; can look unnatural.
- **Choice of p** — often 2 by convention; cross-validate over p ∈ {0.5, 1, 2, 4}.

## Run

```
python techniques/inverse-distance-weighting/python/inverse_distance_weighting.py
Rscript techniques/inverse-distance-weighting/r/inverse_distance_weighting.R
```

**Refs:** Shepard, D. "A two-dimensional interpolation function for irregularly-spaced data." *Proc. ACM National Conf.*, 1968; Li, J. & Heap, A.D. "A review of spatial interpolation methods for environmental scientists." *Geosci. Aust. Rec.* 2008/23.

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
