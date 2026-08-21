# 2D Kernel Intensity Estimation (Reference §23.14)

Smoothed estimate of the **intensity** `λ(s)` (points per unit area) of a
spatial point pattern:

```
λ̂(s) = (1 / edge(s)) · Σ_i K_h(s − x_i)
```

- `K_h` is a bivariate Gaussian (or Epanechnikov, biweight) kernel with bandwidth `h`.
- `edge(s) = ∫_W K_h(s − u) du` — Diggle's border correction; without it, the intensity is pulled toward zero near the window edge.
- Integrating `λ̂` over the window gives (approximately) `n`.

**Density KDE vs intensity:** a density sums to 1; an intensity sums to `n` and has units of counts / area. Use `spatstat::density.ppp` (intensity) vs `MASS::kde2d` or `scipy.stats.gaussian_kde` (density).

## Bandwidth selection

- **Scott's rule** (used in the demo): `h = n^(-1/6) · mean(σ_x, σ_y)` — quick and reasonable.
- **Diggle** (`bw.diggle`): minimises mean-square error under Cox-process assumption.
- **Likelihood cross-validation** (`bw.CvL`).
- **Plug-in** (`ks::Hns`, `ks::Hpi`) — full 2×2 bandwidth matrix (allows anisotropy).

## When to use

- **Hotspot mapping** — crime, disease incidence, wildlife sightings.
- **Continuous-surface companion** to Ripley's K / scan statistics.
- **Denominator of a smoothed rate** — `λ̂_case(s) / λ̂_control(s)` gives a relative-risk surface (see Kelsall-Diggle).

## Files

- `python/kernel_intensity_2d.py` — Gaussian kernel with Diggle edge correction, Scott bandwidth. Demo (60-point cluster at (3, 7) + 40 uniform background on 10×10 window): integrated intensity 102.2 ≈ n = 100; peak at (3.25, 7.25) vs true (3, 7); peak λ̂ = 7.96 vs background 0.65.
- `r/kernel_intensity_2d.R` — `spatstat.explore::density.ppp`, `MASS::kde2d`, `ks::kde`.

## Assumptions & caveats

- **Bandwidth dominates the picture** — small h → noisy; large h → over-smooth.
- **Isotropic kernel** in this demo; use `ks::kde(H = Hns(pts))` for a full 2×2 bandwidth matrix when the pattern is anisotropic.
- **Edge correction matters** — without it, boundary regions look artificially quiet.
- **Not a probability density** — this is an intensity in counts / area; do not compare to a KDE density surface directly.

## Related methods

- **Adaptive bandwidth** — `h_i` grows in sparse regions (`spatstat::adaptive.density`).
- **Kernel intensity for marked point patterns** — smoothed mark values (`spatstat::Smooth.ppp`).
- **Relative-risk surface** — case-control ratio of intensities (Kelsall-Diggle).

## Run

```
python techniques/kernel-intensity-2d/python/kernel_intensity_2d.py
Rscript techniques/kernel-intensity-2d/r/kernel_intensity_2d.R
```

**Refs:** Diggle, P.J. "A kernel method for smoothing point process data." *Applied Statistics* 34(2), 138–147, 1985; Wand, M.P. & Jones, M.C. *Kernel Smoothing*, Chapman & Hall/CRC, 1994; Baddeley, A., Rubak, E. & Turner, R. *Spatial Point Patterns: Methodology and Applications with R*, 2015.

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
