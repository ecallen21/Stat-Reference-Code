# Circular Statistics (Reference §38.3)

Mardia & Jupp (2000). Angles and directions live on a circle — 359°
and 1° are neighbours, so arithmetic on degrees / radians is wrong.
Build every statistic from the vector-mean representation
`(cos θ, sin θ)`.

## Core statistics

- **Mean direction** `θ̄ = atan2( Σ sin θ_i, Σ cos θ_i )`.
- **Mean resultant length** `R̄ = √((Σ cos)² + (Σ sin)²) / n`,
  range `[0, 1]`.
- **Circular variance** `V = 1 − R̄`.
- **Concentration** `κ` (von Mises) — invert Bessel-ratio
  `I_1(κ) / I_0(κ) = R̄`.

## Rayleigh test

Test `H_0`: uniform on circle vs `H_1`: unimodal (von Mises-like).
Statistic `Z = n · R̄²`; large `Z` rejects uniformity.

## Von Mises distribution

```
f(θ ; μ, κ) = exp(κ · cos(θ − μ)) / (2π · I_0(κ))
```

Circular analogue of the normal.

## When to use

- **Wind direction, animal migration bearings, protein backbone
  angles**.
- **Time-of-day / time-of-year** effects when the clock is intrinsic
  to the problem.
- **Compass headings** in navigation.

## When NOT to use

- **Bounded linear data** (0 to 1 proportions, ages) — the wrap does
  not apply.
- **Bimodal / axial data** — use axial statistics (double the
  angle) or Kent/Watson distributions.

## Files

- `python/circular_statistics.py` — mean direction, resultant,
  Rayleigh test, von Mises κ MLE via Bessel-ratio inversion. Demo:
  60 draws from vM(μ=10°, κ=4). Recovers **μ̂ = 10.7°**,
  **κ̂ = 3.95**, Rayleigh `Z = 44.55, p ≈ 10⁻¹⁹`; contrasted with
  naive degree-mean (129° — nonsense).
- `r/circular_statistics.R` — `circular`, `movMF`, `Directional`
  (R); `pycircstat`, `scipy.stats.circmean/circvar` (Python).

## Assumptions & caveats

- **Wrap-invariance** — never take a plain mean of raw degrees; wrap
  systematically to `[0, 2π)` first, but even then compute the
  vector mean, not the scalar mean.
- **Bimodality** — Rayleigh has no power against bimodal
  alternatives; use Watson `U²` or Kuiper `V` for bimodal
  uniformity alternatives.
- **Kappa MLE bias** — small-sample bias in `κ̂` for `n < 15`;
  use bias-corrected `κ̂` (Best-Fisher 1981).
- **Zero point definition** — decide whether 0 is north or east
  before comparing conclusions across studies.

## Related in this repo

- `von-mises-fisher-clustering` (if present) — clustering on the
  sphere.
- `mixture-models` — for multimodal circular data.
- `shape-analysis` — related manifold-valued setup.

## Run

```
python techniques/circular-statistics/python/circular_statistics.py
Rscript techniques/circular-statistics/r/circular_statistics.R
```

**Refs:** Mardia, K.V. & Jupp, P.E. *Directional Statistics*, Wiley, 2000; Fisher, N.I. *Statistical Analysis of Circular Data*, Cambridge University Press, 1993.

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
