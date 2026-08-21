# Ripley's K and L Functions (Reference §23.12)

Second-order summary of a **spatial point pattern**:

```
K(r) = (1 / λ) · E[# other events within r of a typical event]
```

Under **complete spatial randomness** (homogeneous Poisson): `K(r) = π r²` and

```
L(r) = √(K(r) / π)  →  L(r) − r = 0 under CSR.
```

- `L(r) − r > 0` → clustering at scale `r`.
- `L(r) − r < 0` → regularity / inhibition at scale `r`.

## Edge correction

Border method (used here): sum only over points whose boundary distance
`b_i > r`, adjust denominator by `n_R(r) = #{i : b_i > r}`. Ripley's
isotropic correction (`spatstat::Kest`) reweights each pair by the fraction
of the circle of radius `d_ij` around `i` that falls inside the window.

## Monte-Carlo envelope

Simulate `n_sim` CSR patterns with the same `n` and window; compute `L(r) − r`
for each. Point-wise `α / 2` and `1 − α / 2` quantiles give the CSR envelope.
Note: pointwise envelopes are not simultaneous — use **global envelopes**
(`spatstat.explore::envelope(..., global=TRUE)`) for a formal test.

## When to use

- **Ecology / forestry** — tree spacing, nest locations.
- **Epidemiology** — disease-case clustering (with a population correction).
- **Astronomy** — galaxy clustering.
- **Crime analysis** — hotspot detection at multiple scales.

## Files

- `python/ripleys_k_point_pattern.py` — from-scratch K̂ + L̂ with border correction + CSR envelope. Demo (unit square): CSR pattern (n = 150) stays inside 95% envelope at every r; clustered pattern (12 parents × 12 offspring, σ = 0.03) exceeds upper envelope at all 12 r values; L(0.05) − 0.05 = 0.067 vs CSR upper 0.005.
- `r/ripleys_k_point_pattern.R` — `spatstat.explore::Kest / Lest / envelope / pcf`.

## Related second-order summaries

- **Pair correlation function** `g(r) = K'(r) / (2 π r)` — density (not cumulative) form; easier to interpret at a single scale.
- **F, G, J functions** — empty-space, nearest-neighbour, and combined.
- **Cross-K** `K_{ij}(r)` — bivariate patterns (e.g. two species).

## Assumptions & caveats

- **Stationarity** — the intensity `λ` is assumed constant. Inhomogeneous versions (`Kinhom`) use an estimated intensity surface.
- **Window matters** — extending the window changes edge correction and can change the verdict.
- **Pointwise envelope ≠ global test** — many `r` values inflate type-I error.
- **Second-order only** — K captures pairwise structure but not higher-order geometry.

## Run

```
python techniques/ripleys-k-point-pattern/python/ripleys_k_point_pattern.py
Rscript techniques/ripleys-k-point-pattern/r/ripleys_k_point_pattern.R
```

**Refs:** Ripley, B.D. "The second-order analysis of stationary point processes." *J. Appl. Probab.* 13(2), 255–266, 1976; Baddeley, A., Rubak, E. & Turner, R. *Spatial Point Patterns: Methodology and Applications with R*, Chapman & Hall/CRC, 2015.

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
